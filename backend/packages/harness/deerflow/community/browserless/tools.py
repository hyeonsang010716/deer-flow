import asyncio
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.community.url_safety import resolve_host_addresses as _resolve_host_addresses
from deerflow.community.url_safety import validate_public_http_url
from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.tools.types import Runtime
from deerflow.utils.readability import ReadabilityExtractor

from .browserless_client import BrowserlessClient, BrowserlessFetchResult, BrowserlessScreenshotResult

logger = logging.getLogger(__name__)

# readability_extractor는 CPU 바운드 파싱을 수행하므로 항상 asyncio.to_thread로 호출한다
_readability_extractor = ReadabilityExtractor()
_OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
_OUTPUT_FORMAT_TO_EXTENSION = {
    "png": "png",
    "jpeg": "jpeg",
    "webp": "webp",
}
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
# 충돌 접미사 탐색 횟수를 제한해, 포화된 outputs 디렉터리에서 무한 반복하지 않게 한다.
_MAX_FILENAME_COLLISION_PROBES = 1000


def _get_tool_config(tool_name: str) -> dict | None:
    """도구 설정의 extra 필드를 안전하게 읽는다. 설정이 없으면 None을 반환한다."""
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return None
    extras = config.model_extra
    return extras if extras is not None else {}


def _coerce_timeout(value: object, default: float) -> float:
    """설정의 timeout을 초 단위로 변환한다. 잘못된 입력이면 ``default``로 되돌린다.

    ``crawl4ai._coerce_timeout`` / ``jina_ai._coerce_timeout``과 같은 방식이다. bool과
    숫자가 아닌 문자열은 기본값으로 떨어뜨린다. 그래야 ``timeout_s: off``(YAML의 ``False``)가
    ``0.0``이 되어 정상 서버에 대한 모든 요청을 timeout시키지 않고, 오타 값이 도구 생성 도중
    예외를 던지지도 않는다.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            logger.warning("Browserless: invalid timeout %r in config; using %ss", value, default)
    return default


def _resolve_timeout(cfg: dict, default: float) -> float:
    """timeout을 읽되 이 provider의 키와 형제 provider들의 키를 모두 받아들인다.

    ``browserless``는 ``timeout_s``를 문서화하지만 ``crawl4ai``와 ``jina_ai``는 ``timeout``을
    읽는다. 인식되지 않는 키는 extra 필드 도구 설정이 조용히 버리므로, 다른 provider의 예시를
    가져다 쓴 사람은 아무 진단 없이 기본값을 받았다. 둘 다 받아들이되 함께 있으면 문서화된
    ``timeout_s``를 우선한다.
    """
    if "timeout_s" in cfg:
        return _coerce_timeout(cfg["timeout_s"], default)
    if "timeout" in cfg:
        return _coerce_timeout(cfg["timeout"], default)
    return default


def _get_browserless_client(tool_name: str = "web_fetch") -> BrowserlessClient:
    cfg = _get_tool_config(tool_name)
    base_url = "http://localhost:3032"
    token = os.getenv("BROWSERLESS_TOKEN", "")
    timeout_s = 30.0
    if cfg is not None:
        base_url = cfg.get("base_url", base_url)
        token = cfg.get("token", token)
        timeout_s = _resolve_timeout(cfg, timeout_s)
    return BrowserlessClient(base_url=base_url, token=token, timeout_s=timeout_s)


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


def _as_optional_quality(value: object, output_format: str) -> int | None:
    if output_format not in {"jpeg", "webp"}:
        return None
    quality = _as_int(value, -1)
    return quality if 0 <= quality <= 100 else None


def _normalize_output_format(value: object) -> str:
    output_format = str(value or "png").strip().lower()
    return output_format if output_format in _OUTPUT_FORMAT_TO_EXTENSION else "png"


def _validate_capture_url(url: str, allow_private_addresses: bool = False) -> str | None:
    """capture URL의 scheme과 (opt-out하지 않은 한) SSRF 안전성을 검증한다.

    loopback, 사설, link-local(169.254.169.254 cloud metadata endpoint 포함), 예약,
    멀티캐스트, unspecified 주소로 resolve되는 요청을 차단한다. 내부 Browserless 대상을
    의도적으로 지정하는 운영자는 ``allow_private_addresses``로 opt-out할 수 있다.
    """
    return validate_public_http_url(
        url,
        allow_private_addresses=allow_private_addresses,
        action="capture",
        resolver=_resolve_host_addresses,
    )


def _default_capture_stem(url: str) -> str:
    parsed = urlparse(url)
    parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    raw = "-".join(parts) or "web-capture"
    return raw[:80]


def _safe_capture_filename(filename: str | None, url: str, output_format: str) -> str:
    extension = _OUTPUT_FORMAT_TO_EXTENSION[output_format]
    if filename:
        raw_name = Path(filename).name
        stem = Path(raw_name).stem or "web-capture"
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        stem = f"{_default_capture_stem(url)}-{timestamp}"

    safe_stem = _SAFE_FILENAME_RE.sub("_", stem).strip("._-") or "web-capture"
    return f"{safe_stem[:100]}.{extension}"


def _thread_outputs_path(runtime: Runtime) -> Path | str:
    if runtime.state is None:
        return "Error: Thread runtime state is not available"
    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        return "Error: Thread outputs path is not available"
    return Path(outputs_path)


def _tool_message(content: str, tool_call_id: str) -> Command:
    return Command(update={"messages": [ToolMessage(content, tool_call_id=tool_call_id)]})


def _dedupe_output_name(outputs_path: Path, output_name: str) -> str:
    """``outputs_path`` 아래에서 충돌하지 않는 파일명을 반환한다.

    비어 있으면 원래 이름을 그대로 쓰고, 아니면 확장자 앞에 ``-1``, ``-2``, ... 를 붙인다.
    그래야 명시적 파일명이 이전 capture를 조용히 덮어쓰지 않는다. 제한된 탐색 범위 안에서
    디렉터리가 포화되면 timestamp 접미사로 대체한다.
    """
    candidate = outputs_path / output_name
    if not candidate.exists():
        return output_name

    stem = Path(output_name).stem
    suffix = Path(output_name).suffix
    for index in range(1, _MAX_FILENAME_COLLISION_PROBES + 1):
        probe = f"{stem}-{index}{suffix}"
        if not (outputs_path / probe).exists():
            return probe

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return f"{stem}-{timestamp}{suffix}"


def _write_capture_output(outputs_path: Path, output_name: str, content: bytes) -> str:
    """``content``를 ``outputs_path``에 쓰고 실제로 사용한 파일명을 반환한다."""
    outputs_path.mkdir(parents=True, exist_ok=True)
    final_name = _dedupe_output_name(outputs_path, output_name)
    (outputs_path / final_name).write_bytes(content)
    return final_name


def _target_status_warning(result: BrowserlessScreenshotResult | BrowserlessFetchResult) -> str:
    """가져오거나 캡처한 page 자체가 에러였다면 사람이 읽을 수 있는 경고를 반환한다.

    Browserless는 대상 page가 4xx/5xx를 응답했거나 에러/anti-bot page였어도 렌더 요청 자체에는
    HTTP 200을 준다. 따라서 원본 내용만으로는 정상 성공 응답이라고 믿을 수 없다. 대상의 실제
    status는 X-Response-Code 헤더로 드러난다.
    """
    code = result.target_status_code.strip()
    if not code or code.startswith(("2", "3")):
        return ""
    status = result.target_status.strip()
    detail = f"{code} {status}".strip()
    return f" (warning: target page responded {detail})"


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """Browserless(headless Chrome)를 사용해 주어진 URL의 web page 내용을 가져온다.
    사용자가 직접 제공했거나 web_search와 web_fetch 도구의 결과로 반환된 URL만 정확히 그대로 가져와라.
    이 도구는 비공개 Google Docs나 로그인 장벽 뒤의 page처럼 인증이 필요한 콘텐츠에는 접근할 수 없다.
    www.가 없는 URL에 www.를 임의로 붙이지 마라.
    URL에는 schema를 반드시 포함해야 한다. https://example.com은 유효하지만 example.com은 유효하지 않다.

    Args:
        url: 내용을 가져올 URL.
    """
    try:
        cfg = _get_tool_config("web_fetch") or {}
        allow_private_addresses = _as_bool(cfg.get("allow_private_addresses"), False)
        url_error = validate_public_http_url(
            url,
            allow_private_addresses=allow_private_addresses,
            resolver=_resolve_host_addresses,
        )
        if url_error:
            return url_error

        wait_for_event = ""
        wait_for_timeout_ms = 0
        wait_for_selector = ""
        wait_for_selector_timeout_ms = 5000
        reject_resource_types: list[str] | None = None
        reject_request_pattern: list[str] | None = None

        wait_for_event = cfg.get("wait_for_event", wait_for_event)
        raw_wait = cfg.get("wait_for_timeout_ms", wait_for_timeout_ms)
        wait_for_timeout_ms = int(raw_wait) if not isinstance(raw_wait, int) else raw_wait
        wait_for_selector = cfg.get("wait_for_selector", wait_for_selector)

        client = _get_browserless_client("web_fetch")
        result = await client.fetch_html_with_status(
            url=url,
            wait_for_event=wait_for_event,
            wait_for_timeout_ms=wait_for_timeout_ms,
            wait_for_selector=wait_for_selector,
            wait_for_selector_timeout_ms=wait_for_selector_timeout_ms,
            reject_resource_types=reject_resource_types,
            reject_request_pattern=reject_request_pattern,
        )

        if isinstance(result, str):
            return result

        article = await asyncio.to_thread(_readability_extractor.extract_article, result.html)
        return f"{article.to_markdown()[:4096]}{_target_status_warning(result)}"

    except Exception as e:
        logger.error(f"Error in web_fetch_tool: {e}")
        return f"Error: {str(e)}"


@tool("web_capture", parse_docstring=True)
async def web_capture_tool(
    runtime: Runtime,
    url: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    filename: str | None = None,
    full_page: bool | None = None,
    output_format: str | None = None,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
) -> Command:
    """렌더링된 webpage의 screenshot을 캡처해 artifact로 제시한다.

    공개된 webpage의 시각적 캡처가 필요할 때, 특히 JavaScript 비중이 큰 page, UI 상태, dashboard, 보고서에 넣을 시각적 증거가 필요할 때 사용하라.
    사용자가 제공했거나 다른 도구로 찾아낸 URL만 정확히 그대로 캡처하라. 사용자가 DeerFlow 외부에서 Browserless를 명시적으로 설정하지 않은 한, 로그인 뒤의 비공개 page에는 사용하지 마라.
    URL에는 schema를 반드시 포함해야 한다. https://example.com은 유효하지만 example.com은 유효하지 않다.

    Args:
        url: 캡처할 http(s) URL.
        filename: 선택적 출력 파일명. 디렉터리는 무시되며 확장자는 output_format으로 결정된다.
        full_page: 전체 page 캡처 여부의 선택적 override.
        output_format: 선택적 이미지 포맷: png, jpeg, webp 중 하나.
        viewport_width: 선택적 viewport 너비(픽셀).
        viewport_height: 선택적 viewport 높이(픽셀).
    """
    try:
        cfg = _get_tool_config("web_capture") or {}
        allow_private_addresses = _as_bool(cfg.get("allow_private_addresses"), False)

        url_error = _validate_capture_url(url, allow_private_addresses=allow_private_addresses)
        if url_error:
            return _tool_message(url_error, tool_call_id)

        outputs_path = _thread_outputs_path(runtime)
        if isinstance(outputs_path, str):
            return _tool_message(outputs_path, tool_call_id)

        final_format = _normalize_output_format(output_format or cfg.get("output_format", "png"))
        final_full_page = full_page if full_page is not None else _as_bool(cfg.get("full_page"), True)
        final_width = viewport_width if viewport_width is not None else _as_int(cfg.get("viewport_width"), 1280)
        final_height = viewport_height if viewport_height is not None else _as_int(cfg.get("viewport_height"), 720)
        quality = _as_optional_quality(cfg.get("quality"), final_format)
        wait_for_selector = str(cfg.get("wait_for_selector") or "")
        wait_for_selector_timeout_ms = _as_int(cfg.get("wait_for_selector_timeout_ms"), 5000)
        wait_for_timeout_ms = _as_int(cfg.get("wait_for_timeout_ms"), 0)
        best_attempt = _as_bool(cfg.get("best_attempt"), False)

        output_name = _safe_capture_filename(filename, url, final_format)

        client = _get_browserless_client("web_capture")
        result = await client.capture_screenshot(
            url=url,
            full_page=final_full_page,
            output_format=final_format,
            quality=quality,
            viewport={"width": final_width, "height": final_height},
            wait_for_selector=wait_for_selector,
            wait_for_selector_timeout_ms=wait_for_selector_timeout_ms,
            wait_for_timeout_ms=wait_for_timeout_ms,
            best_attempt=best_attempt,
        )
        if isinstance(result, str):
            return _tool_message(result, tool_call_id)

        final_name = await asyncio.to_thread(_write_capture_output, outputs_path, output_name, result.content)
        virtual_path = f"{_OUTPUTS_VIRTUAL_PREFIX}/{final_name}"
        message = f"Captured screenshot: {virtual_path}{_target_status_warning(result)}"
        return Command(
            update={
                "artifacts": [virtual_path],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        logger.error(f"Error in web_capture_tool: {e}")
        return _tool_message(f"Error: {str(e)}", tool_call_id)
