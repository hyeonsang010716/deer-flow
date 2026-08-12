"""provider가 길이 제한으로 끊은 model 응답을 run stop reason으로 드러낸다.

배경은 issue bytedance/deer-flow#4271을 참고한다.

일부 provider는 출력 예산이 소진되어 생성을 멈추고, assistant content는 그대로 반환하면서
``finish_reason='length'``로 그 사실을 알린다. DeerFlow는 감사 목적으로 그 content를 보존해야
하지만, provider가 명시적으로 잘림을 알렸는데도 run을 제한 없이 깔끔하게 끝난 것으로 조용히
취급해서는 안 된다.

이 middleware는 그 경계를 좁게 유지한다:
- 마지막 AIMessage가 provider의 길이 신호로 잘렸고 눈에 보이는 content가 남아 있을 때만
  run 수준 stop reason을 표시한다.
- assistant content를 다시 쓰거나 XML 비슷한 텍스트를 tool call로 재파싱하지 않는다.
- tool call 의도, 잘못된 tool call metadata, 또는 보이는 content가 없는 응답은 무시한다.
  따라서 보이는 content를 가진 종단 assistant 응답만 capped로 표시될 수 있다.

"""

from __future__ import annotations

import logging
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.model_length_termination_detectors import (
    ModelLengthTermination,
    ModelLengthTerminationDetector,
    default_detectors,
)

MODEL_LENGTH_CAPPED_STOP_REASON = "model_length_capped"
logger = logging.getLogger(__name__)


def _has_tool_call_intent_or_error(message: AIMessage) -> bool:
    if message.tool_calls or getattr(message, "invalid_tool_calls", None):
        return True
    additional_kwargs = message.additional_kwargs or {}
    return bool(additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"))


def _has_visible_content(message: AIMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                return True
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    return False


class ModelLengthFinishReasonMiddleware(AgentMiddleware[AgentState]):
    """content가 있는 종단 assistant 응답에 대해 provider의 길이 제한을 기록한다.

    마지막 AIMessage가 여전히 tool call 의도를 갖고 있으면 이 middleware는 손대지 않고 평소의
    tool 처리 경로가 판단하도록 둔다.
    """

    def __init__(self, detectors: list[ModelLengthTerminationDetector] | None = None) -> None:
        super().__init__()
        self._detectors: list[ModelLengthTerminationDetector] = list(detectors) if detectors else default_detectors()

    def _detect(self, message: AIMessage) -> ModelLengthTermination | None:
        for detector in self._detectors:
            try:
                hit = detector.detect(message)
            except Exception:  # noqa: BLE001 - provider detector가 run을 깨뜨려서는 안 된다
                logger.exception("ModelLengthTerminationDetector %r raised; treating as no-match", getattr(detector, "name", type(detector).__name__))
                continue
            if hit is not None:
                return hit
        return None

    def _apply(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        last = messages[-1]
        if _has_tool_call_intent_or_error(last):
            return None
        if not _has_visible_content(last):
            return None

        termination = self._detect(last)
        if termination is None:
            return None

        ctx = getattr(runtime, "context", None)
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        run_id = ctx.get("run_id") if isinstance(ctx, dict) else None
        stamped_stop_reason = False
        if isinstance(ctx, dict):
            # 숨겨진 continuation turn을 거쳐 넘어온 앞선 cap reason은 그대로 보존한다.
            if "stop_reason" not in ctx:
                ctx["stop_reason"] = MODEL_LENGTH_CAPPED_STOP_REASON
                stamped_stop_reason = True
        logger.info(
            "Provider model length cap detected",
            extra={
                "thread_id": thread_id,
                "run_id": run_id,
                "message_id": getattr(last, "id", None),
                "detector": termination.detector,
                "reason_field": termination.reason_field,
                "reason_value": termination.reason_value,
                "stamped_stop_reason": stamped_stop_reason,
            },
        )
        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)
