"""provider가 safety로 종료한 AIMessage를 복구해, 실행되지도 빈 채로 영속화되지도 않게 한다.

배경 — issue bytedance/deer-flow#3028(잘린 tool call)과 #4393(빈 응답이 thread를 오염)을 참고.

일부 provider(OpenAI ``finish_reason='content_filter'``, Anthropic
``stop_reason='refusal'``, Gemini ``finish_reason='SAFETY'`` 등)는 생성 도중 중단하면서도
일부만 만들어진 ``tool_calls``를 반환할 수 있다. LangChain의 tool router는 ``tool_calls``
필드가 비어 있지 않은 AIMessage를 전부 "이것들을 실행하라"로 취급하므로, 문장 중간에서 끊긴
markdown ``write_file`` 같은 반쯤 잘린 인자도 완성된 것처럼 dispatch된다. 그러면 agent는 잘린
파일을 보고 고치려다 다시 필터링당하며 루프에 빠진다.

이 middleware는 ``after_model``에 위치해 그 동작을 막는다. 설정된
``SafetyTerminationDetector``가 발동하면 다음 중 하나를 수행한다.

* AIMessage에 tool call이 있으면 이를 제거한다(structured와 raw provider payload 모두).
  잘린 tool call 사례(#3028).
* 메시지가 그 외에는 비어 있으면(tool call 없음, 보이는 content 없음) 사용자에게 보이는 설명을
  채워 넣는다. 빈 응답 사례(#4393). 그대로 두면 빈 assistant 메시지가 영속화되고, 이후 모든
  요청에서 엄격한 OpenAI 호환 provider가 이를 거부해("message ... with role 'assistant' must
  not be empty") 새 대화를 시작할 때까지 thread 전체가 막힌다.

보이는 텍스트는 있고 tool call이 없는 safety 종료 메시지는 부분 답변이 사용자에게 전달되도록
그대로 둔다. 어느 경우든 설명을 덧붙이고 관측용 필드를 ``additional_kwargs.safety_termination``에
저장해, 로그·trace·SSE 소비자가 무슨 일이 있었는지 볼 수 있게 한다.

hook 선택: ``wrap_model_call``이 아니라 ``after_model``이다. 응답이 예외가 아니라 *정상*
반환이고, tool call 제거 메커니즘은 같지만 trigger가 다른 ``LoopDetectionMiddleware``와 같은
after-model chain에 참여하고 싶기 때문이다.

배치: middleware 목록에서 ``LoopDetectionMiddleware`` *뒤에* 등록한다. LangChain factory는
``after_model`` edge를 목록 역순으로 연결하므로
(``langchain/agents/factory.py:add_edge("model", middleware_w_after_model[-1])`` 이후
``range(len-1, 0, -1)``를 순회), *마지막*에 등록된 middleware가 모델 출력을 *가장 먼저*
관찰한다. Safety를 Loop 뒤에 등록하면 Safety가 raw 응답을 먼저 보고 발동 시 tool call을 지우며,
Loop는 정리된 메시지를 기준으로 집계한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.safety_termination_detectors import (
    SafetyTermination,
    SafetyTerminationDetector,
    default_detectors,
)
from deerflow.agents.middlewares.tool_call_metadata import clone_ai_message_with_tool_calls
from deerflow.runtime.events.catalog import MIDDLEWARE_SAFETY_TERMINATION_TAG
from deerflow.utils.custom_events import aemit_custom_event, emit_custom_event
from deerflow.utils.messages import message_content_to_text

if TYPE_CHECKING:
    from deerflow.config.safety_finish_reason_config import SafetyFinishReasonConfig

logger = logging.getLogger(__name__)


_USER_FACING_MESSAGE = (
    "The model provider stopped this response with a safety-related signal "
    "({reason_field}={reason_value!r}, detector={detector!r}). Any tool "
    "calls produced in this turn were suppressed because their arguments "
    "may be truncated and unsafe to execute. Please rephrase the request "
    "or ask for a narrower output."
)

# safety 종료가 tool call도 content도 만들지 않은 경우에 쓴다. 이때 메시지는 빈 채로
# 영속화되지 않게 하려고만 다시 쓰는 것이므로(#4393 참고), tool call이 억제됐다고 주장하면
# 안 된다.
_USER_FACING_EMPTY_MESSAGE = "The model provider stopped this response with a safety-related signal ({reason_field}={reason_value!r}, detector={detector!r}) and returned no content. Please rephrase your request or start a new conversation."


@dataclass(frozen=True)
class _SafetyIntervention:
    update: dict
    termination: SafetyTermination
    suppressed_names: list[str]
    message: AIMessage
    tool_calls: list[dict]


class SafetyFinishReasonMiddleware(AgentMiddleware[AgentState]):
    """SafetyTerminationDetector가 표시한 AIMessage를 복구한다. tool call을 제거하거나,
    메시지가 그 외에 비어 있으면 설명을 채워 넣는다."""

    def __init__(self, detectors: list[SafetyTerminationDetector] | None = None) -> None:
        super().__init__()
        # 생성 이후 호출자가 변경해도 우리 쪽에 새어 들어오지 않도록 복사한다.
        self._detectors: list[SafetyTerminationDetector] = list(detectors) if detectors else default_detectors()

    @classmethod
    def from_config(cls, config: SafetyFinishReasonConfig) -> SafetyFinishReasonMiddleware:
        """검증된 Pydantic config로부터 생성한다. detector 목록이 주어지면 reflection으로
        로드해 사용한다.

        명시적인 빈 목록은 의도적으로 거부한다. middleware는 chain에 남긴 채 탐지만 조용히
        꺼버리는, 양쪽 모두 최악인 상태가 되기 때문이다. 대신 ``enabled: false``를 쓴다.
        """
        if config.detectors is None:
            return cls()

        if not config.detectors:
            raise ValueError("safety_finish_reason.detectors must be omitted (use built-ins) or contain at least one entry; use enabled=false to disable the middleware entirely.")

        from deerflow.reflection import resolve_variable

        detectors: list[SafetyTerminationDetector] = []
        for entry in config.detectors:
            detector_cls = resolve_variable(entry.use)
            kwargs = dict(entry.config) if entry.config else {}
            detector = detector_cls(**kwargs)
            if not isinstance(detector, SafetyTerminationDetector):
                raise TypeError(f"{entry.use} did not produce a SafetyTerminationDetector (got {type(detector).__name__}); ensure it has a `name` attribute and a `detect(message)` method")
            detectors.append(detector)
        return cls(detectors=detectors)

    # ----- 탐지 ------------------------------------------------------------

    def _detect(self, message: AIMessage) -> SafetyTermination | None:
        for detector in self._detectors:
            try:
                hit = detector.detect(message)
            except Exception:  # noqa: BLE001 - never let a buggy detector break the agent run
                logger.exception("SafetyTerminationDetector %r raised; treating as no-match", getattr(detector, "name", type(detector).__name__))
                continue
            if hit is not None:
                return hit
        return None

    # ----- 메시지 재작성 ----------------------------------------------------

    @staticmethod
    def _append_user_message(content: object, text: str) -> str | list:
        """AIMessage content에 평문 설명을 덧붙인다.

        ``LoopDetectionMiddleware._append_text``와 동일하게 동작해, list content 응답
        (Anthropic thinking 블록, vLLM reasoning 분할)이 문자열로 강제 변환돼 TypeError가
        나지 않고 구조를 유지하게 한다.
        """
        if content is None or content == "":
            return text
        if isinstance(content, list):
            return [*content, {"type": "text", "text": f"\n\n{text}"}]
        if isinstance(content, str):
            return content + f"\n\n{text}"
        return str(content) + f"\n\n{text}"

    def _build_suppressed_message(
        self,
        message: AIMessage,
        termination: SafetyTermination,
    ) -> AIMessage:
        tool_calls = message.tool_calls or []
        suppressed_names = [tc.get("name") or "unknown" for tc in tool_calls]
        template = _USER_FACING_MESSAGE if tool_calls else _USER_FACING_EMPTY_MESSAGE
        explanation = template.format(
            reason_field=termination.reason_field,
            reason_value=termination.reason_value,
            detector=termination.detector,
        )
        new_content = self._append_user_message(message.content, explanation)

        # clone_ai_message_with_tool_calls가 structured tool_calls, raw
        # additional_kwargs.tool_calls, function_call을 한 번에 처리한다. finish_reason은 기존
        # 값이 "tool_calls"일 때만 다시 쓰는데 여기는 해당되지 않으므로, content_filter /
        # refusal / SAFETY는 그대로 남아 하위 SSE와 converter가 실제 provider 사유를 계속 본다.
        cleared = clone_ai_message_with_tool_calls(message, [], content=new_content)

        # clone_ai_message_with_tool_calls가 반환한 dict를 실수로 변경하지 않도록
        # additional_kwargs를 다시 복사한다(이미 shallow copy를 만들었지만 하위 model_copy가
        # 여전히 그것을 참조한다). 그다음 관측용 레코드를 기록한다.
        kwargs = dict(getattr(cleared, "additional_kwargs", None) or {})
        kwargs["safety_termination"] = {
            "detector": termination.detector,
            "reason_field": termination.reason_field,
            "reason_value": termination.reason_value,
            "suppressed_tool_call_count": len(suppressed_names),
            "suppressed_tool_call_names": suppressed_names,
            "extras": dict(termination.extras) if termination.extras else {},
        }
        return cleared.model_copy(update={"additional_kwargs": kwargs})

    # ----- 관측 ------------------------------------------------------------

    @staticmethod
    def _build_event_payload(
        termination: SafetyTermination,
        suppressed_names: list[str],
        runtime: Runtime,
    ) -> dict:
        thread_id = None
        if runtime is not None and getattr(runtime, "context", None):
            thread_id = runtime.context.get("thread_id") if isinstance(runtime.context, dict) else None
        return {
            "type": "safety_termination",
            "detector": termination.detector,
            "reason_field": termination.reason_field,
            "reason_value": termination.reason_value,
            "suppressed_tool_call_count": len(suppressed_names),
            "suppressed_tool_call_names": suppressed_names,
            "thread_id": thread_id,
        }

    def _emit_event(
        self,
        termination: SafetyTermination,
        suppressed_names: list[str],
        runtime: Runtime,
    ) -> None:
        """tool turn이 억제됐음을 SSE 소비자(예: 웹 UI)에 알려, 이미 사용자에게 스트리밍된
        "tool starting..." placeholder를 정리할 수 있게 한다. 실패는 debug로 로깅하고
        무시한다. best-effort 신호이기 때문이다."""
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("get_stream_writer unavailable; skipping safety_termination event", exc_info=True)
            return

        try:
            emit_custom_event(self._build_event_payload(termination, suppressed_names, runtime), writer=writer)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("Failed to emit safety_termination stream event", exc_info=True)

    async def _aemit_event(
        self,
        termination: SafetyTermination,
        suppressed_names: list[str],
        runtime: Runtime,
    ) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("get_stream_writer unavailable; skipping async safety_termination event", exc_info=True)
            return

        try:
            await aemit_custom_event(self._build_event_payload(termination, suppressed_names, runtime), writer=writer)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("Failed to emit async safety_termination stream event", exc_info=True)

    def _record_audit_event(
        self,
        termination: SafetyTermination,
        message,
        tool_calls: list[dict],
        runtime: Runtime,
    ) -> None:
        """run 이후 감사를 위해 ``middleware:safety_termination`` 레코드를 RunEventStore에
        기록한다.

        ``_emit_event``의 custom stream event는 실시간 SSE client가 소비하고 run이 끝나면
        사라진다. 이 event는 영속화되므로 운영자가 메시지 본문을 join하지 않고 SQL 한 번으로
        "오늘 어떤 run이 safety로 억제됐나?"에 답할 수 있다. worker는 run 범위의
        ``RunJournal``을 ``runtime.context["__run_journal"]``로 노출한다. unit test / subagent /
        event store 없는 경로에서는 없으므로 조용히 건너뛴다.

        tool **인자**는 의도적으로 기록하지 **않는다**. 그 내용이야말로 provider가 필터링한
        대상이며, 영속화하면 safety filter의 목적이 무너진다. 감사와 디버깅에는 이름 / 개수 /
        id면 충분하다(issue #3028 리뷰).
        """
        journal = None
        if runtime is not None and getattr(runtime, "context", None):
            context = runtime.context
            if isinstance(context, dict):
                journal = context.get("__run_journal")
        if journal is None:
            return

        suppressed_names = [tc.get("name") or "unknown" for tc in tool_calls]
        suppressed_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]

        changes = {
            "detector": termination.detector,
            "reason_field": termination.reason_field,
            "reason_value": termination.reason_value,
            "suppressed_tool_call_count": len(tool_calls),
            "suppressed_tool_call_names": suppressed_names,
            "suppressed_tool_call_ids": suppressed_ids,
            "message_id": getattr(message, "id", None),
            "extras": dict(termination.extras) if termination.extras else {},
        }

        try:
            journal.record_middleware(
                tag=MIDDLEWARE_SAFETY_TERMINATION_TAG,
                name=type(self).__name__,
                hook="after_model",
                action="suppress_tool_calls",
                changes=changes,
            )
        except Exception:  # noqa: BLE001
            # 감사 event 영속화가 agent 실행을 깨뜨려서는 안 된다.
            logger.warning("Failed to record middleware:safety_termination event", exc_info=True)

    # ----- 주 적용 ---------------------------------------------------------

    def _prepare_intervention(self, state: AgentState, runtime: Runtime) -> _SafetyIntervention | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # 다시 쓸 가치가 있는 provider safety 실패 유형은 둘이다. tool call 없이 보이는 텍스트를
        # 만든 safety 종료는 부분 답변이 자연스럽게 사용자에게 전달되도록 건드리지 않는다.
        #   1. tool_calls가 있는 경우: 잘렸거나 안전하지 않을 수 있으므로(#3028) 억제한다.
        #   2. content가 비어 있고 tool_calls도 없는 경우: 빈 assistant 메시지는 엄격한 OpenAI
        #      호환 provider(Moonshot/Kimi 등)가 *다음* 요청에서 거부하며("message ... with
        #      role 'assistant' must not be empty", #4393), 새 대화를 시작할 때까지 thread
        #      전체를 오염시킨다. 영속화되는 메시지가 비지 않도록 설명을 채워 넣는다.
        tool_calls = list(last.tool_calls or [])
        # ``or ""``는 "보이는 content 없음"의 모든 형태를 빈 값으로 정규화한다. None, "", [],
        # 공백 전부 해당된다. None은 ``model_copy(update={"content": None})``(검증을 건너뛰는
        # 재작성 경로)로 도달할 수 있다. 이 가드가 없으면 message_content_to_text가 "None"
        # 문자열로 만들어 backfill을 건너뛰고, 이 수정이 보호하려던 thread를 다시 오염시킨다.
        content_is_blank = not message_content_to_text(last.content or "").strip()
        if not tool_calls and not content_is_blank:
            return None

        termination = self._detect(last)
        if termination is None:
            return None

        backfilled_empty = content_is_blank and not tool_calls

        # worker가 이 capped 완료를 loop_capped / token_capped와 함께 드러낼 수 있도록
        # stop_reason을 기록한다(#4176).
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            ctx["stop_reason"] = "safety_capped"
        patched = self._build_suppressed_message(last, termination)

        thread_id = None
        if runtime is not None and getattr(runtime, "context", None):
            thread_id = runtime.context.get("thread_id") if isinstance(runtime.context, dict) else None

        logger.warning(
            "Provider safety termination detected — suppressed %d tool call(s), backfilled_empty_content=%s",
            len(tool_calls),
            backfilled_empty,
            extra={
                "thread_id": thread_id,
                "detector": termination.detector,
                "reason_field": termination.reason_field,
                "reason_value": termination.reason_value,
                "suppressed_tool_call_names": [tc.get("name") for tc in tool_calls],
                "backfilled_empty_content": backfilled_empty,
            },
        )

        tool_calls = list(tool_calls)
        return _SafetyIntervention(
            update={"messages": [patched]},
            termination=termination,
            suppressed_names=[tc.get("name") or "unknown" for tc in tool_calls],
            message=last,
            tool_calls=tool_calls,
        )

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        intervention = self._prepare_intervention(state, runtime)
        if intervention is None:
            return None

        self._emit_event(intervention.termination, intervention.suppressed_names, runtime)
        self._record_audit_event(intervention.termination, intervention.message, intervention.tool_calls, runtime)
        return intervention.update

    # ----- hook ------------------------------------------------------------

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        intervention = self._prepare_intervention(state, runtime)
        if intervention is None:
            return None

        await self._aemit_event(intervention.termination, intervention.suppressed_names, runtime)
        self._record_audit_event(intervention.termination, intervention.message, intervention.tool_calls, runtime)
        return intervention.update
