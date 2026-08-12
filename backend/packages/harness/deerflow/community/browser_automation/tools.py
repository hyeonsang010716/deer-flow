"""Agentic browser 도구. navigate → observe → act 를 상태를 유지하며 반복한다.

읽기 전용인 ``web_fetch`` / ``web_capture`` 도구와 달리, 이 도구들은 thread별 live browser
session(:mod:`.session` 참고)을 유지한다. 덕분에 agent가 JavaScript 비중이 크거나 인증이 필요한
page에서 클릭, 입력, 폼 제출, 다단계 흐름을 수행할 수 있다. 모든 액션은 새 page snapshot을
반환하고, 그 안의 interactive element는 안정적인 ``[ref]`` 인덱스로 지정된다. 모델은 selector를
추측하지 않고 방금 관찰한 대상에 대해 동작한다.

모든 URL은 공용 :func:`validate_public_http_url` 헬퍼로 SSRF 검사를 거친다
(의도적인 내부 대상에 한해서만 opt-out한다).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.community.url_safety import resolve_host_addresses as _resolve_host_addresses
from deerflow.community.url_safety import validate_public_http_url
from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.constants import BROWSER_FRAMES_DIRNAME
from deerflow.tools.types import Runtime

from .session import BrowserSession, BrowserSessionManager, PageSnapshot, get_browser_session_manager

logger = logging.getLogger(__name__)

_OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
# step마다 자동 캡처하는 screenshot은 산출물이 아니라 진행 상황 피드백이다
# (browser panel과 인라인 썸네일에 표시된다). 숨김 하위 디렉터리에 두어야
# workspace-changes 검토가 이를 파일 변경으로 나열하지 않는다.
# 디렉터리 이름은 공용 상수로 두어 scanner의 무시 목록과 어긋나지 않게 한다.
_BROWSER_FRAMES_DIRNAME = BROWSER_FRAMES_DIRNAME
_FRAMES_VIRTUAL_PREFIX = f"{_OUTPUTS_VIRTUAL_PREFIX}/{_BROWSER_FRAMES_DIRNAME}"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _get_tool_config(tool_name: str) -> dict:
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return {}
    return config.model_extra or {}


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _thread_id(runtime: Runtime) -> str | None:
    return runtime.context.get("thread_id") if runtime.context else None


def _as_str(value: object) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


class _SessionLease:
    """프로세스 로컬 browser session을 고정 상태로 유지하는 context manager."""

    def __init__(self, manager: BrowserSessionManager, thread_id: str | None, session: BrowserSession) -> None:
        self._manager = manager
        self._thread_id = thread_id
        self.session = session

    def __enter__(self) -> BrowserSession:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._manager.release_session(self._thread_id, self.session)


def _resolve_session(runtime: Runtime, tool_name: str) -> _SessionLease:
    # launch 설정(headless/viewport/timeout/cdp_url)은 어떤 도구가 session을 먼저 만들든
    # 항상 ``browser_navigate`` 라는 단일 기준 출처에서 읽는다. ``get_session``은 thread별로
    # 캐시하고 이후 호출자의 이 파라미터들은 무시하므로, 호출한 도구를 기준으로 launch 설정을
    # 잡으면 "먼저 실행된 도구가 이긴다"가 된다. ``browser_navigate``에만 지정한
    # ``headless: false``가 다른 도구(또는 live WS)가 session을 먼저 초기화하면 조용히 무시됐다.
    # ``tool_name``은 launch와 무관한 자기 설정을 읽는 호출자를 위해 남겨 둔다
    # (예: ``browser_get_text``의 max_chars).
    del tool_name
    cfg = _get_tool_config("browser_navigate")
    headless = _as_bool(cfg.get("headless"), True)
    timeout_ms = _as_int(cfg.get("timeout_ms"), 30000)
    width = _as_int(cfg.get("viewport_width"), 1280)
    height = _as_int(cfg.get("viewport_height"), 720)
    cdp_url = _as_str(cfg.get("cdp_url"))
    manager = get_browser_session_manager()
    thread_id = _thread_id(runtime)
    session = manager.get_session(
        thread_id,
        headless=headless,
        timeout_ms=timeout_ms,
        viewport={"width": width, "height": height},
        cdp_url=cdp_url,
        allow_unguarded_cdp=_as_bool(cfg.get("allow_unguarded_cdp"), False),
        url_guard=validate_browser_url,
        pin=True,
    )
    return _SessionLease(manager, thread_id, session)


def validate_browser_url(url: str, *, tool_name: str = "browser_navigate") -> str | None:
    """도구 설정 정책에 따라 browser 이동 URL을 SSRF 검사한다.

    URL을 거부해야 하면 ``"Error: ..."`` 문자열을, 이동해도 되면 ``None``을 반환한다.
    agent 도구와 Gateway live stream이 공유하므로, browser를 조종할 수 있는 모든 경로가
    같은 허용/거부 정책(``browser_navigate`` 도구 설정의 ``allow_private_addresses``)을 따른다.
    """
    cfg = _get_tool_config(tool_name)
    allow_private = _as_bool(cfg.get("allow_private_addresses"), False)
    return validate_public_http_url(
        url,
        allow_private_addresses=allow_private,
        action="browse",
        resolver=_resolve_host_addresses,
    )


def _validate_url(tool_name: str, url: str) -> str | None:
    return validate_browser_url(url, tool_name=tool_name)


def _snapshot_message(snapshot: PageSnapshot, prefix: str = "") -> str:
    body = snapshot.render()
    return f"{prefix}\n\n{body}" if prefix else body


def _tool_message(content: str, tool_call_id: str) -> Command:
    return Command(update={"messages": [ToolMessage(content, tool_call_id=tool_call_id)]})


def _step_screenshot_name(action: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    safe = _SAFE_FILENAME_RE.sub("_", action).strip("._-") or "step"
    return f"browser-{safe}-{stamp}.png"


async def _capture_step_screenshot(runtime: Runtime, session: BrowserSession, action: str) -> str | None:
    """액션별 screenshot을 best-effort로 찍어 숨김 진행 피드백으로 저장한다.

    artifact virtual path를 반환한다(숨김 ``.browser-frames`` 디렉터리 아래이므로
    workspace-changes 검토에 잡히지 않는다). outputs를 쓸 수 없거나 캡처가 실패하면
    ``None``을 반환한다. 캡처 실패가 액션을 깨뜨려서는 안 된다.
    """
    outputs_path = _thread_outputs_path(runtime)
    if isinstance(outputs_path, str):
        return None
    try:
        content = await session.screenshot_bytes(full_page=False)
        name = _step_screenshot_name(action)
        frames_dir = outputs_path / _BROWSER_FRAMES_DIRNAME
        final_name = await asyncio.to_thread(_write_screenshot, frames_dir, name, content)
        with contextlib.suppress(Exception):
            session.schedule_live_frames()
        return f"{_FRAMES_VIRTUAL_PREFIX}/{final_name}"
    except Exception as e:
        logger.warning(f"browser step screenshot failed: {e}")
        return None


def _snapshot_command(
    runtime: Runtime,
    session: BrowserSession,
    snapshot: PageSnapshot,
    tool_call_id: str,
    prefix: str,
    screenshot_path: str | None,
) -> Command:
    """텍스트 snapshot과 인라인 screenshot을 함께 담은 ToolMessage를 만든다.

    screenshot은 thread ``artifacts`` 항목으로도(artifacts 사이드 패널에서 열리도록),
    ``ToolMessage.additional_kwargs.browser_view``로도(채팅이 browser step마다 인라인
    썸네일을 렌더링하도록) 실려 간다.
    """
    text = _snapshot_message(snapshot, prefix)
    additional_kwargs: dict = {}
    update: dict = {}
    if screenshot_path:
        additional_kwargs["browser_view"] = {"screenshot": screenshot_path, "url": snapshot.url, "title": snapshot.title}
        update["artifacts"] = [screenshot_path]
    update["messages"] = [ToolMessage(text, tool_call_id=tool_call_id, additional_kwargs=additional_kwargs)]
    return Command(update=update)


async def navigate_and_capture(*, thread_id: str | None, url: str, outputs_path: Path) -> dict:
    """thread별 browser session을 *url*로 이동시키고 screenshot을 캡처한다.

    Gateway browser router가 사용하며, 사용자가 UI 주소창에서 live session을 조종할 수 있게 한다.
    :func:`browser_navigate_tool`과 같은 thread별 session, SSRF 정책, screenshot 파이프라인을 공유한다.

    Returns:
        ``{"screenshot": virtual_path|None, "url": str, "title": str}``.

    Raises:
        ValueError: URL이 SSRF 검증을 통과하지 못한 경우.
    """
    url_error = _validate_url("browser_navigate", url)
    if url_error:
        raise ValueError(url_error)
    cfg = _get_tool_config("browser_navigate")
    manager = get_browser_session_manager()
    with manager.acquire_session(
        thread_id,
        headless=_as_bool(cfg.get("headless"), True),
        timeout_ms=_as_int(cfg.get("timeout_ms"), 30000),
        viewport={"width": _as_int(cfg.get("viewport_width"), 1280), "height": _as_int(cfg.get("viewport_height"), 720)},
        cdp_url=_as_str(cfg.get("cdp_url")),
        allow_unguarded_cdp=_as_bool(cfg.get("allow_unguarded_cdp"), False),
        url_guard=validate_browser_url,
    ) as session:
        snapshot = await session.navigate(url)
        screenshot_path: str | None = None
        try:
            content = await session.screenshot_bytes(full_page=False)
            name = _step_screenshot_name("navigate")
            frames_dir = outputs_path / _BROWSER_FRAMES_DIRNAME
            final_name = await asyncio.to_thread(_write_screenshot, frames_dir, name, content)
            screenshot_path = f"{_FRAMES_VIRTUAL_PREFIX}/{final_name}"
        except Exception as e:
            logger.warning(f"browser gateway navigate screenshot failed: {e}")
    return {"screenshot": screenshot_path, "url": snapshot.url, "title": snapshot.title}


@tool("browser_navigate", parse_docstring=True)
async def browser_navigate_tool(runtime: Runtime, url: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """live browser session에서 URL을 열고 그 page의 interactive element 목록을 반환한다.

    browsing 흐름을 시작할 때 사용하라. 읽기 전용인 web_fetch와 달리 상태를 유지하는
    browser를 띄우므로, 이후 page에서 클릭과 입력을 할 수 있다. 결과는 interactive element를
    ``[ref] role: name`` 형식으로 나열한다 — 이 ``[ref]`` 번호를 browser_click과
    browser_type에 그대로 넘겨라. session은 browser_close 전까지 이 대화의 tool call들
    사이에서 유지된다. navigate/click/type 단계마다 사용자가 볼 수 있는 screenshot이 자동
    캡처되므로, 진행 상황을 보여주려고 browser_screenshot을 따로 호출할 필요는 없다.
    URL에는 scheme을 반드시 포함하라. 예: https://example.com.

    Args:
        url: 열려는 http(s) URL.
    """
    try:
        url_error = _validate_url("browser_navigate", url)
        if url_error:
            return _tool_message(url_error, tool_call_id)
        with _resolve_session(runtime, "browser_navigate") as session:
            snapshot = await session.navigate(url)
            screenshot = await _capture_step_screenshot(runtime, session, "navigate")
            return _snapshot_command(runtime, session, snapshot, tool_call_id, f"Navigated to {url}.", screenshot)
    except Exception as e:
        logger.error(f"browser_navigate failed: {e}")
        return _tool_message(f"Error: browser navigation failed: {e}", tool_call_id)


@tool("browser_snapshot")
async def browser_snapshot_tool(runtime: Runtime, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """아무 액션도 하지 않고 현재 page의 interactive element를 다시 읽는다. page가 스스로 바뀌었거나(예: async 콘텐츠 로드) 현재 상태가 확실하지 않을 때, [ref] element 목록을 갱신하려면 사용하라."""
    try:
        with _resolve_session(runtime, "browser_snapshot") as session:
            snapshot = await session.snapshot()
            screenshot = await _capture_step_screenshot(runtime, session, "snapshot")
            return _snapshot_command(runtime, session, snapshot, tool_call_id, "", screenshot)
    except Exception as e:
        logger.error(f"browser_snapshot failed: {e}")
        return _tool_message(f"Error: browser snapshot failed: {e}", tool_call_id)


@tool("browser_click", parse_docstring=True)
async def browser_click_tool(runtime: Runtime, ref: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """최신 snapshot의 ``[ref]`` 번호로 interactive element를 클릭한다.

    ref는 browser_navigate, browser_snapshot, browser_click, browser_type이 반환한 번호가
    붙은 element 목록에서 가져온다. 클릭 후 갱신된 page snapshot을 반환하며, 그 안의
    ``[ref]`` 번호는 새로 매겨진 값이다.

    Args:
        ref: 클릭할 element의 reference 번호.
    """
    try:
        with _resolve_session(runtime, "browser_click") as session:
            snapshot = await session.click(ref)
            screenshot = await _capture_step_screenshot(runtime, session, "click")
            return _snapshot_command(runtime, session, snapshot, tool_call_id, f"Clicked element [{ref}].", screenshot)
    except Exception as e:
        logger.error(f"browser_click failed: {e}")
        return _tool_message(f"Error: could not click element [{ref}]: {e}", tool_call_id)


@tool("browser_type", parse_docstring=True)
async def browser_type_tool(
    runtime: Runtime,
    ref: int,
    text: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    submit: bool = False,
) -> Command:
    """``[ref]`` 번호로 지정한 input/textarea element에 텍스트를 입력한다.

    ref로 식별된 필드를 채운다. 입력 후 Enter를 누르려면(예: 검색 실행이나 폼 제출)
    submit=true로 설정하라. 갱신된 page snapshot을 반환한다.

    Args:
        ref: 입력할 input element의 reference 번호.
        text: 입력할 텍스트.
        submit: true이면 입력 후 Enter를 눌러 제출한다.
    """
    try:
        with _resolve_session(runtime, "browser_type") as session:
            snapshot = await session.type_text(ref, text, submit=submit)
            action = f"Typed into element [{ref}] and submitted." if submit else f"Typed into element [{ref}]."
            screenshot = await _capture_step_screenshot(runtime, session, "type")
            return _snapshot_command(runtime, session, snapshot, tool_call_id, action, screenshot)
    except Exception as e:
        logger.error(f"browser_type failed: {e}")
        return _tool_message(f"Error: could not type into element [{ref}]: {e}", tool_call_id)


@tool("browser_get_text")
async def browser_get_text_tool(runtime: Runtime, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """현재 page에서 보이는 텍스트 내용을 읽는다. 이동하거나 상호작용한 뒤 읽을 수 있는 텍스트를 추출할 때, 예를 들어 결과를 인용하거나 내용을 요약할 때 사용하라. page가 크면 출력은 잘린다."""
    try:
        with _resolve_session(runtime, "browser_get_text") as session:
            cfg = _get_tool_config("browser_get_text")
            max_chars = _as_int(cfg.get("max_chars"), 8000)
            text = await session.get_text(max_chars=max_chars)
        return _tool_message(text or "(page has no visible text)", tool_call_id)
    except Exception as e:
        logger.error(f"browser_get_text failed: {e}")
        return _tool_message(f"Error: could not read page text: {e}", tool_call_id)


@tool("browser_back")
async def browser_back_tool(runtime: Runtime, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """browser session의 history에서 이전 page로 돌아간다."""
    try:
        with _resolve_session(runtime, "browser_back") as session:
            snapshot = await session.back()
            screenshot = await _capture_step_screenshot(runtime, session, "back")
            return _snapshot_command(runtime, session, snapshot, tool_call_id, "Went back.", screenshot)
    except Exception as e:
        logger.error(f"browser_back failed: {e}")
        return _tool_message(f"Error: could not go back: {e}", tool_call_id)


def _safe_screenshot_name(filename: str | None) -> str:
    if filename:
        stem = Path(filename).stem or "browser-capture"
    else:
        stem = f"browser-capture-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    safe = _SAFE_FILENAME_RE.sub("_", stem).strip("._-") or "browser-capture"
    return f"{safe[:100]}.png"


def _thread_outputs_path(runtime: Runtime) -> Path | str:
    if runtime.state is None:
        return "Error: Thread runtime state is not available"
    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        return "Error: Thread outputs path is not available"
    return Path(outputs_path)


def _write_screenshot(outputs_path: Path, name: str, content: bytes) -> str:
    outputs_path.mkdir(parents=True, exist_ok=True)
    (outputs_path / name).write_bytes(content)
    return name


@tool("browser_screenshot", parse_docstring=True)
async def browser_screenshot_tool(
    runtime: Runtime,
    tool_call_id: Annotated[str, InjectedToolCallId],
    filename: str | None = None,
    full_page: bool = False,
) -> Command:
    """현재 browser page의 screenshot을 찍어 artifact로 저장한다.

    클릭이나 입력을 마친 뒤 현재 interactive session 상태의 시각적 증거가 필요할 때 사용하라.
    web_capture는 상태 없이 page를 새로 로드해 렌더링하므로 이 정보를 제공할 수 없다.

    Args:
        filename: 선택적 출력 파일명(확장자는 .png로 강제된다).
        full_page: viewport만이 아니라 스크롤 가능한 page 전체를 캡처한다.
    """
    try:
        outputs_path = _thread_outputs_path(runtime)
        if isinstance(outputs_path, str):
            return _tool_message(outputs_path, tool_call_id)
        with _resolve_session(runtime, "browser_screenshot") as session:
            content = await session.screenshot_bytes(full_page=full_page)
            name = _safe_screenshot_name(filename)
            final_name = await asyncio.to_thread(_write_screenshot, outputs_path, name, content)
        virtual_path = f"{_OUTPUTS_VIRTUAL_PREFIX}/{final_name}"
        return Command(
            update={
                "artifacts": [virtual_path],
                "messages": [ToolMessage(f"Saved browser screenshot: {virtual_path}", tool_call_id=tool_call_id)],
            }
        )
    except Exception as e:
        logger.error(f"browser_screenshot failed: {e}")
        return _tool_message(f"Error: could not capture screenshot: {e}", tool_call_id)


@tool("browser_close")
async def browser_close_tool(runtime: Runtime, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """현재 browser session을 닫고 리소스를 해제한다. browsing 흐름을 마쳤을 때 호출하라. 이후 browser_navigate를 호출하면 새 session이 시작된다."""
    try:
        manager = get_browser_session_manager()
        closed = await manager.close_session(_thread_id(runtime))
        msg = "Browser session closed." if closed else "No active browser session to close."
        return _tool_message(msg, tool_call_id)
    except Exception as e:
        logger.error(f"browser_close failed: {e}")
        return _tool_message(f"Error: could not close browser session: {e}", tool_call_id)
