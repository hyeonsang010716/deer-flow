"""도구 출력에 결과 단위 예산을 강제하는 middleware.

크기가 큰 도구 결과는 디스크에 저장하고, 파일 참조를 담은 간결한 typed synopsis로
대체한다. 디스크 저장이 불가능하면 head+tail 잘라내기로 fallback해서 큰 도구 반환값
하나가 모델 context를 날려버리지 않게 한다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.tool_output_synopsis import render_tool_output_preview
from deerflow.config.tool_output_config import ToolOutputConfig
from deerflow.sandbox.sandbox_provider import get_sandbox_provider

if TYPE_CHECKING:
    from deerflow.sandbox.sandbox import Sandbox

logger = logging.getLogger(__name__)

# sandbox 내부의 virtual outputs 루트. host mount 방식 sandbox는 이를 host의 thread
# outputs 디렉터리로 매핑한다. mount하지 않는(원격) sandbox에서는 같은 경로를 sandbox
# 파일시스템에 직접 쓴다. 그래야 모델의 ``read_file`` 도구가 다시 읽을 수 있다(issue #3416).
_VIRTUAL_OUTPUTS_BASE = "/mnt/user-data/outputs"


def _default_config() -> ToolOutputConfig:
    return ToolOutputConfig()


# ---------------------------------------------------------------------------
# 텍스트 헬퍼
# ---------------------------------------------------------------------------


def _message_text(content: Any) -> str | None:
    """ToolMessage content 필드에서 평문 표현을 뽑아낸다.

    문자열이 아니거나 multimodal인 콘텐츠(이미지, 구조화 블록 등)는 ``None``을 반환해
    호출자가 예산 적용을 건너뛰게 한다.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return None
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
            else:
                return None
        return "\n".join(pieces) if pieces else None
    return None


def _snap_to_line_boundary(text: str, pos: int) -> int:
    """*pos* 또는 그보다 앞의 가장 가까운 개행+1 중 더 가까운 쪽을 반환한다.

    preview와 잘라내기가 가능하면 완결된 줄에서 끝나도록 하기 위한 것이다.
    ``text[:pos]``의 뒷절반에 개행이 없으면 원래 *pos*를 그대로 반환한다.

    *끝* offset에만 유효하다. 뒤로 옮기면 여기서 끝나는 slice가 짧아지기 때문이다.
    시작 offset에는 :func:`_snap_start_to_line_boundary`를 쓴다.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    half = pos // 2
    nl = text.rfind("\n", half, pos)
    if nl >= 0:
        return nl + 1
    return pos


def _snap_start_to_line_boundary(text: str, pos: int) -> int:
    """*pos* 또는 그 뒤의 가장 가까운 개행+1 중 더 가까운 쪽을 반환한다.

    :func:`_snap_to_line_boundary`의 시작 offset 버전이다. 시작을 뒤로 당기면 거기서
    시작하는 slice가 *길어지므로*, 예산이 적용된 preview의 꼬리는 앞으로 당겨야 한다.
    ``text[pos:]``의 앞절반에 개행이 없으면 원래 *pos*를 그대로 반환한다.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    half = pos + (len(text) - pos) // 2
    nl = text.find("\n", pos, half)
    if nl >= 0:
        return nl + 1
    return pos


# ---------------------------------------------------------------------------
# 디스크 저장
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, str] = {
    "bash": "log",
    "bash_tool": "log",
    "web_fetch": "log",
}


def _sanitize_tool_name(name: str) -> str:
    """도구 이름에서 경로 구분자와 traversal 요소를 제거한다."""
    base = os.path.basename(name)
    safe = base.replace("..", "").replace("/", "_").replace("\\", "_")
    return safe or "unknown"


def _build_externalized_filename(*, tool_name: str, tool_call_id: str) -> str:
    """외부화된 도구 출력의 디스크 파일명을 만든다.

    host 디스크 경로와 sandbox 외부화 경로가 공유해서 동일한 명명 규칙을 갖게 한다.
    """
    safe_name = _sanitize_tool_name(tool_name)
    ext = _EXT_MAP.get(tool_name, "txt")
    short_id = uuid.uuid4().hex[:12]
    return f"{safe_name}-{short_id}.{ext}"


def _externalize(
    content: str,
    *,
    tool_name: str,
    tool_call_id: str,
    outputs_path: str,
    storage_subdir: str,
) -> str | None:
    """*content*를 디스크에 쓰고 virtual path를 반환한다. 실패하면 ``None``."""
    if os.path.isabs(storage_subdir) or ".." in storage_subdir:
        return None
    storage_dir = os.path.join(outputs_path, storage_subdir)
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except OSError:
        return None

    filename = _build_externalized_filename(tool_name=tool_name, tool_call_id=tool_call_id)
    filepath = os.path.join(storage_dir, filename)

    if not os.path.abspath(filepath).startswith(os.path.abspath(storage_dir)):
        return None

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError:
        return None

    return f"{_VIRTUAL_OUTPUTS_BASE}/{storage_subdir}/{filename}"


def _externalize_to_sandbox(
    content: str,
    *,
    tool_name: str,
    tool_call_id: str,
    storage_subdir: str,
    sandbox: Sandbox,
) -> str | None:
    """*content*를 sandbox 파일시스템에 쓰고 virtual path를 반환한다.

    sandbox가 thread-data mount를 쓰지 않을 때(예: 원격 AIO sandbox) 사용한다. host 쪽
    :func:`_externalize`의 virtual path는 sandbox 안에 존재하지 않아 모델의 ``read_file``
    도구가 다시 읽을 수 없기 때문이다(issue #3416). 성공하면 동일한 virtual-path 계약을
    반환하고, ``None``이면 호출자가 inline 잘라내기로 fallback하라는 신호다.
    """
    if os.path.isabs(storage_subdir) or ".." in storage_subdir:
        return None
    filename = _build_externalized_filename(tool_name=tool_name, tool_call_id=tool_call_id)
    virtual_dir = f"{_VIRTUAL_OUTPUTS_BASE}/{storage_subdir}"
    virtual_path = f"{virtual_dir}/{filename}"
    try:
        # AIO sandbox의 write_file은 상위 디렉터리를 만들지 않으므로 쓰기 전에 명시적으로
        # 생성한다. execute_command는 예외를 던지지 않고 stdout을 그대로 반환하므로
        # (실패 시 "Error: ..." 문자열 포함) 여기서 예외 전파에 의존할 수 없다.
        sandbox.execute_command(f"mkdir -p {shlex.quote(virtual_dir)}")
        sandbox.write_file(virtual_path, content)
        # 파일이 실제로 생겼는지 검증한다. execute_command가 조용히 디렉터리 생성에
        # 실패했을 수 있고 write_file backend마다 동작이 다르다. 읽을 수 없는 read_file
        # 경로를 모델에게 넘기지 않는다.
        check = sandbox.execute_command(f"test -s {shlex.quote(virtual_path)} && echo OK || echo MISSING")
        if not isinstance(check, str) or check.strip() != "OK":
            logger.warning(
                "Sandbox externalize validation failed: path=%s, check=%r",
                virtual_path,
                check,
            )
            return None
    except Exception:
        logger.exception(
            "Failed to externalize %s output to sandbox (call_id=%s)",
            tool_name,
            tool_call_id,
        )
        return None
    return virtual_path


# ---------------------------------------------------------------------------
# Preview / fallback 빌더
# ---------------------------------------------------------------------------


def _build_preview(
    content: str,
    *,
    tool_name: str,
    virtual_path: str,
    head_chars: int,
    tail_chars: int,
) -> str:
    """외부화된 출력에 대해 파일 참조를 포함한 typed synopsis preview를 만든다."""
    return render_tool_output_preview(
        content,
        tool_name=tool_name,
        virtual_path=virtual_path,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )


def _build_fallback(
    content: str,
    *,
    tool_name: str,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> str:
    """디스크 저장이 불가능할 때 head+tail 잘라내기 결과를 만든다.

    반환 문자열은 *max_chars*를 넘지 않음이 보장된다.
    """
    total = len(content)
    if max_chars <= 0 or total <= max_chars:
        return content

    marker_template = "\n\n[... {n} chars omitted from {tn} output. Persistent storage unavailable. Consider narrowing the query or using more specific parameters.]\n\n"
    marker_overhead = len(marker_template.format(n=total, tn=tool_name))

    if marker_overhead >= max_chars:
        return content[:max_chars]

    budget = max_chars - marker_overhead
    effective_head = min(head_chars, budget)
    effective_tail = min(tail_chars, max(0, budget - effective_head))

    head_end = _snap_to_line_boundary(content, min(effective_head, total))
    tail_start = _snap_start_to_line_boundary(content, max(head_end, total - effective_tail))

    head = content[:head_end]
    tail = content[tail_start:] if tail_start < total else ""
    omitted = total - len(head) - len(tail)

    marker = marker_template.format(n=omitted, tn=tool_name)

    parts = [head, marker]
    if tail:
        parts.append(tail)
    return "".join(parts)


# ---------------------------------------------------------------------------
# 핵심 예산 로직
# ---------------------------------------------------------------------------


def _resolve_outputs_path(request: ToolCallRequest) -> str | None:
    """thread outputs 경로를 best-effort로 추출한다."""
    runtime = getattr(request, "runtime", None)
    if runtime is None:
        return None
    state = getattr(runtime, "state", None)
    if state is None:
        return None
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        return None
    outputs_path = thread_data.get("outputs_path")
    return outputs_path if isinstance(outputs_path, str) else None


def _resolve_sandbox(request: ToolCallRequest) -> Sandbox | None:
    """현재 도구 호출에 해당하는 활성 sandbox를 찾는다. 없으면 ``None``.

    ``SandboxMiddleware``(및 sandbox 도구들)가 ``runtime.state["sandbox"]``에 써 둔
    sandbox_id를 읽는다. 여기서 ``provider.acquire``는 의도적으로 호출하지 않는다.
    sandbox 획득은 블로킹 원격 I/O를 유발할 수 있는데 이 resolver는 모든 도구 호출마다
    돌기 때문이다. sandbox를 쓰지 않는 도구(``web_search``, MCP 등)는 여기서 ``None``을
    받는데 문제없다. 호출자가 inline 잘라내기로 fallback한다.
    """
    runtime = getattr(request, "runtime", None)
    state = getattr(runtime, "state", None)
    if not isinstance(state, dict):
        return None
    sandbox_state = state.get("sandbox")
    if not isinstance(sandbox_state, dict):
        return None
    sandbox_id = sandbox_state.get("sandbox_id")
    if not sandbox_id:
        return None
    try:
        return get_sandbox_provider().get(sandbox_id)
    except Exception:
        logger.exception("Failed to look up sandbox %s for tool-output externalization", sandbox_id)
        return None


def _budget_content(
    content: str,
    *,
    tool_name: str,
    tool_call_id: str,
    outputs_path: str | None,
    config: ToolOutputConfig,
    sandbox: Sandbox | None = None,
) -> str | None:
    """*content*에 예산을 적용한다. 변경이 필요 없으면 ``None``을 반환한다."""
    threshold = config.tool_overrides.get(tool_name, config.externalize_min_chars)
    if threshold <= 0 and config.fallback_max_chars <= 0:
        return None
    if len(content) <= threshold and len(content) <= config.fallback_max_chars:
        return None

    if threshold > 0 and len(content) > threshold:
        virtual_path: str | None = None
        # 이 호출에 대해 실제로 sandbox가 잡혔을 때만 sandbox provider를 건드리고,
        # 그 외에는 사용 가능한 것에 따라 저장 대상을 정한다. 이렇게 하면 기존 host-disk
        # 경로가 provider와 무관하게 유지되어, sandbox를 설정하지 않은 호출자나
        # config.yaml이 없는 CI 환경도 예전처럼 host에 외부화한다.
        if sandbox is not None:
            provider = None
            try:
                provider = get_sandbox_provider()
            except Exception:
                logger.exception("Failed to get sandbox provider for tool-output externalization; falling back to inline truncation")
            if provider is not None and getattr(provider, "uses_thread_data_mounts", False):
                # host mount 방식 sandbox: host outputs 경로가 같은 virtual path로
                # sandbox에 bind-mount되므로 host 쪽에 쓰는 것과 동등하다. 추가 sandbox
                # 왕복을 피하기 위해 원래 동작을 유지한다.
                if outputs_path:
                    virtual_path = _externalize(
                        content,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        outputs_path=outputs_path,
                        storage_subdir=config.storage_subdir,
                    )
            else:
                virtual_path = _externalize_to_sandbox(
                    content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    storage_subdir=config.storage_subdir,
                    sandbox=sandbox,
                )
        elif outputs_path:
            # 이 호출에 sandbox가 없다(legacy 또는 sandbox를 쓰지 않는 도구).
            # provider 없이 host outputs 경로에 바로 쓴다.
            virtual_path = _externalize(
                content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                outputs_path=outputs_path,
                storage_subdir=config.storage_subdir,
            )
        if virtual_path is not None:
            logger.info(
                "Externalized %s output (%d chars) to %s",
                tool_name,
                len(content),
                virtual_path,
            )
            return _build_preview(
                content,
                tool_name=tool_name,
                virtual_path=virtual_path,
                head_chars=config.preview_head_chars,
                tail_chars=config.preview_tail_chars,
            )

    if config.fallback_max_chars > 0 and len(content) > config.fallback_max_chars:
        logger.warning(
            "Fallback-truncating %s output: %d chars → %d max",
            tool_name,
            len(content),
            config.fallback_max_chars,
        )
        return _build_fallback(
            content,
            tool_name=tool_name,
            max_chars=config.fallback_max_chars,
            head_chars=config.fallback_head_chars,
            tail_chars=config.fallback_tail_chars,
        )

    return None


# ---------------------------------------------------------------------------
# 결과 패처
# ---------------------------------------------------------------------------


def _patch_tool_message(
    msg: ToolMessage,
    config: ToolOutputConfig,
    outputs_path: str | None,
    sandbox: Sandbox | None = None,
) -> ToolMessage:
    """ToolMessage 하나에 예산을 적용한다. 변경이 없으면 원본을 그대로 반환한다."""
    tool_name = msg.name or "unknown"
    if tool_name in config.exempt_tools:
        return msg

    text = _message_text(msg.content)
    if text is None:
        return msg

    replacement = _budget_content(
        text,
        tool_name=tool_name,
        tool_call_id=msg.tool_call_id or "",
        outputs_path=outputs_path,
        config=config,
        sandbox=sandbox,
    )
    if replacement is None:
        return msg

    update: dict[str, Any] = {"content": replacement}
    if getattr(msg, "response_metadata", None):
        update["response_metadata"] = dict(msg.response_metadata)
    if getattr(msg, "additional_kwargs", None):
        update["additional_kwargs"] = dict(msg.additional_kwargs)
    return msg.model_copy(update=update)


def _effective_trigger(tool_name: str, config: ToolOutputConfig) -> int:
    """*tool_name*에 대해 예산 적용을 유발할 수 있는 최소 콘텐츠 길이.

    :func:`_budget_content`의 발동 조건(도구별 externalize 임계값 또는 전역 fallback)을
    그대로 반영해 사전 스캔이 false negative를 내지 않게 한다. 어떤 경우에도 발동하지
    않으면 ``-1``을 반환한다.
    """
    candidates: list[int] = []
    externalize = config.tool_overrides.get(tool_name, config.externalize_min_chars)
    if externalize > 0:
        candidates.append(externalize)
    if config.fallback_max_chars > 0:
        candidates.append(config.fallback_max_chars)
    return min(candidates) if candidates else -1


def _tool_message_over_budget(msg: ToolMessage, config: ToolOutputConfig) -> bool:
    """도구별 기준을 반영한 저비용 검사. 이 ToolMessage가 예외 대상이 아니면서 발동 기준을 넘는가."""
    if (msg.name or "") in config.exempt_tools:
        return False
    trigger = _effective_trigger(msg.name or "", config)
    if trigger < 0:
        return False
    text = _message_text(msg.content)
    return text is not None and len(text) > trigger


def _needs_budget(result: ToolMessage | Command, config: ToolOutputConfig) -> bool:
    """*result*에 예산 적용이 필요할 수 있는지 빠르게 확인한다(작은 출력은 thread offload를 피한다)."""
    if isinstance(result, ToolMessage):
        return _tool_message_over_budget(result, config)
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        for msg in update.get("messages", []):
            if isinstance(msg, ToolMessage) and _tool_message_over_budget(msg, config):
                return True
    return False


def _patch_result(
    result: ToolMessage | Command,
    config: ToolOutputConfig,
    outputs_path: str | None,
    sandbox: Sandbox | None = None,
) -> ToolMessage | Command:
    """도구 호출 결과(ToolMessage 또는 Command)에 예산을 적용한다."""
    if isinstance(result, ToolMessage):
        return _patch_tool_message(result, config, outputs_path, sandbox)

    update = getattr(result, "update", None)
    if not isinstance(update, dict):
        return result

    messages = update.get("messages")
    if not isinstance(messages, list):
        return result

    new_messages: list[Any] = []
    changed = False
    for msg in messages:
        if isinstance(msg, ToolMessage):
            patched = _patch_tool_message(msg, config, outputs_path, sandbox)
            if patched is not msg:
                changed = True
            new_messages.append(patched)
        else:
            new_messages.append(msg)

    if not changed:
        return result

    return dc_replace(result, update={**update, "messages": new_messages})


def _patch_model_messages(messages: list[Any], config: ToolOutputConfig) -> list[Any] | None:
    """모델 요청에 담긴 과거 ToolMessage에 예산을 적용한다. 변경이 없으면 ``None``을 반환한다.

    예산을 넘는 과거 ToolMessage가 없으면 저비용 사전 스캔이 새 리스트를 할당하기 전에
    빠져나온다. 모든 결과가 이미 도구 호출 시점에 예산 처리된 일반적인 경우가 이에 해당해,
    긴 히스토리를 모델 호출마다 다시 만들지 않는다.

    과거 메시지에는 ``sandbox`` 인자를 넘기지 않는다. 히스토리의 큰 도구 메시지는 이미
    도구 호출 시점에 예산 처리(필요하면 외부화)되었으므로, 히스토리 경로에 남은 일은
    sandbox가 필요 없는 inline fallback 잘라내기뿐이다.
    """
    if not any(isinstance(msg, ToolMessage) and _tool_message_over_budget(msg, config) for msg in messages):
        return None

    updated: list[Any] = []
    changed = False
    for msg in messages:
        if isinstance(msg, ToolMessage):
            patched = _patch_tool_message(msg, config, outputs_path=None)
            if patched is not msg:
                changed = True
            updated.append(patched)
        else:
            updated.append(msg)
    return updated if changed else None


# ---------------------------------------------------------------------------
# Middleware 클래스
# ---------------------------------------------------------------------------


class ToolOutputBudgetMiddleware(AgentMiddleware[AgentState]):
    """외부화 또는 잘라내기로 도구 출력에 결과 단위 예산을 강제한다."""

    def __init__(self, config: ToolOutputConfig | None = None) -> None:
        super().__init__()
        self._config = config if config is not None else _default_config()

    @classmethod
    def from_app_config(cls, app_config: Any) -> ToolOutputBudgetMiddleware:
        tool_output = getattr(app_config, "tool_output", None)
        if isinstance(tool_output, ToolOutputConfig):
            return cls(config=tool_output)
        return cls()

    # -- 도구 호출 hook ---------------------------------------------------

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if not self._config.enabled:
            return result
        if not _needs_budget(result, self._config):
            return result
        outputs_path = _resolve_outputs_path(request)
        sandbox = _resolve_sandbox(request)
        return _patch_result(result, self._config, outputs_path, sandbox)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if not self._config.enabled:
            return result
        if not _needs_budget(result, self._config):
            return result
        outputs_path = _resolve_outputs_path(request)
        # _resolve_sandbox는 runtime.state와 provider의 in-memory sandbox registry만
        # 건드리므로 event loop에서 호출해도 안전하다. 실제 sandbox I/O(mkdir/write/test)는
        # 아래에서 worker thread로 offload되는 _patch_result 안에서 일어난다.
        sandbox = _resolve_sandbox(request)
        return await asyncio.to_thread(_patch_result, result, self._config, outputs_path, sandbox)

    # -- 모델 호출 hook(과거 메시지 잘라내기) ------------------

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        if self._config.enabled:
            messages = getattr(request, "messages", None)
            if isinstance(messages, list):
                patched = _patch_model_messages(messages, self._config)
                if patched is not None:
                    request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        if self._config.enabled:
            messages = getattr(request, "messages", None)
            if isinstance(messages, list):
                patched = _patch_model_messages(messages, self._config)
                if patched is not None:
                    request = request.override(messages=patched)
        return await handler(request)
