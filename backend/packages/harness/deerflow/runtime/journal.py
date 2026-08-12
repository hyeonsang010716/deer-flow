"""LangChain callback을 통한 run event 수집.

RunJournal은 LangChain의 callback 메커니즘과 교체 가능한 RunEventStore 사이에 놓인다. callback
데이터를 RunEvent 레코드로 표준화하고 token usage 누적을 처리한다.

핵심 설계 결정:
- on_llm_new_token은 구현하지 않는다 — on_llm_end로 완성된 메시지만 다룬다
- on_chat_model_start가 사용자에게 보이는 첫 prompt를 llm.human.input으로 기록하고 run.input용
  첫 human message를 추출한다. 모든 node에서 발생하는 on_chain_start보다 신뢰할 수 있고,
  여기서는 메시지가 완전히 구조화돼 있기 때문이다.
- parent_run_id=None인 on_chain_start는 root 호출을 표시하는 run.start trace를 발생시킨다.
- on_llm_end는 checkpoint와 정렬된 AIMessage.model_dump() 형식으로 llm.ai.response를 낸다
- token usage는 메모리에 누적했다가 run 완료 시 RunRow에 기록한다
- 호출자 식별은 tag 주입으로 한다(lead_agent / subagent:{name} / middleware:{name})
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage, messages_from_dict
from langgraph.types import Command

from deerflow.agents.human_input import read_human_input_response
from deerflow.runtime.events.catalog import (
    LLM_AI_RESPONSE_EVENT,
    LLM_ERROR_EVENT,
    LLM_HUMAN_INPUT_EVENT,
    LLM_TOOL_RESULT_EVENT,
    MEMORY_CONTEXT_EVENT,
    MIDDLEWARE_EVENT_PATTERN,
    RUN_END_EVENT,
    RUN_ERROR_EVENT,
    RUN_START_EVENT,
)
from deerflow.utils.messages import message_to_text, restore_original_human_message

if TYPE_CHECKING:
    from deerflow.runtime.events.store.base import RunEventStore

logger = logging.getLogger(__name__)

_LEGACY_SUMMARY_MESSAGE_NAME = "summary"
_RECONCILED_TOOL_MESSAGE_NAMES = frozenset({"ask_clarification"})
_PERSISTED_HIDDEN_HUMAN_INPUT_RESPONSE_SOURCES = frozenset({"ask_clarification"})


def _should_persist_human_input_message(message: BaseMessage) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    if message.name == _LEGACY_SUMMARY_MESSAGE_NAME:
        return False
    if message.additional_kwargs.get("hide_from_ui") is not True:
        return True
    response = read_human_input_response(message.additional_kwargs)
    return response is not None and response["source"] in _PERSISTED_HIDDEN_HUMAN_INPUT_RESPONSE_SOURCES


def _coerce_seed_message(message: Any) -> Any:
    """``message``를 ``BaseMessage``로 반환하고, 필요하면 dict 형태를 역직렬화한다.

    ``_checkpoint_messages``(threads.py)는 snapshot이 담고 있는 값을 그대로 반환하고, 형제격인
    branch 매칭 헬퍼들은 메시지가 ``BaseMessage``이든 ``model_dump()`` 형태의 dict이든 모두
    처리한다(checkpoint backend/mode마다 serde가 다르기 때문). seed 경로도 둘 다 처리해야 한다 —
    그러지 않으면 dict 기반 checkpoint는 아무것도 seed하지 못하고, 히스토리가 있는데도 branch가
    조용히 ``skipped_empty``를 보고한다. 파싱할 수 없는 dict는 그대로 통과해
    ``isinstance(BaseMessage)`` 가드에서 걸러진다.
    """
    if isinstance(message, BaseMessage):
        return message
    if isinstance(message, Mapping):
        msg_type = message.get("type")
        if isinstance(msg_type, str) and msg_type:
            try:
                return messages_from_dict([{"type": msg_type, "data": dict(message)}])[0]
            except Exception:
                logger.warning("branch seed: could not deserialize checkpoint message dict (type=%s)", msg_type)
    return message


def _build_history_seed_events(
    messages: Sequence[Any],
    *,
    thread_id: str,
    run_id_prefix: str,
    seed_metadata: Mapping[str, Any],
) -> list[dict]:
    """checkpoint 메시지를 run-event row로 직렬화한다.

    row는 checkpoint turn 단위로 하나의 합성 run(``{run_id_prefix}-{n}``)으로 묶이며, 저장된 human
    message마다 새 turn이 시작된다 — 실제 run과 같은 경계다. run은 human 입력으로 시작하기
    때문이다(자기 run으로 재개되는, allowlist에 있는 숨김 ``ask_clarification`` 응답 포함).
    ``run_id``는 feed 소비자에게 단순한 출처 tag가 아니라 *turn* 정체성이다. 마지막으로 상속받은
    답변을 regenerate하면 그 row의 ``run_id``가 대체된 원본으로 해석되고
    (``_find_target_run_id``), ``GET /messages/page``는 그 id를 가진 **모든** row를 버린다. 그래서
    seed 전체에 id 하나를 공유하면 branch의 첫 regenerate에서 상속받은 히스토리 전체가
    삭제됐다(#4458). turn마다 id를 따로 주면 삭제 범위가 실제로 regenerate된 turn으로 한정된다.

    RunJournal의 message-event 계약을 그대로 따르므로, seed된 row는 주어진 seed metadata를 빼면
    journal이 기록한 row와 구분되지 않는다. 동일한 event type, ``category="message"``,
    ``content=message.model_dump()``, human 입력 저장 규칙
    (``_should_persist_human_input_message``), 원본 사용자 텍스트 복원, 그리고 ``hide_from_ui``
    AI/tool row에 대한 동일한 처리 — RunJournal은 이들을 저장하고(``on_llm_end`` /
    ``_persist_tool_result_message``는 걸러 내지 않는다) frontend가 클라이언트 쪽에서 숨기므로,
    seed도 버리지 않고 함께 기록한다.

    checkpoint 메시지에는 run 범위가 없어서 생기는 의도적인 차이 하나: AI row는 RunJournal의
    run 범위 부가 정보(``usage`` / ``latency_ms`` / ``llm_call_index``)를 생략하고, ``caller``는
    (여기서는 복원할 수 없는) 메시지의 원래 호출자 대신 ``lead_agent``로 찍는다. 현재로서는 둘 다
    관측되지 않는다 — 그 metadata 키를 인덱싱하는 소비자가 없고, 메시지별 ``caller``는 어떤
    귀속에도 쓰이지 않는다(``by_caller`` usage 패널은 run 범위이며 message feed에서 나오지 않는다).
    """
    events: list[dict] = []
    created_at = datetime.now(UTC).isoformat()
    # 첫 human turn보다 앞선 메시지(실제로는 없다)는 turn 0에 남는다.
    turn_index = 0
    for raw_message in messages:
        message = _coerce_seed_message(raw_message)
        if not isinstance(message, BaseMessage):
            continue
        if isinstance(message, HumanMessage):
            if not _should_persist_human_input_message(message):
                continue
            turn_index += 1
            event_type = "llm.human.input"
            content = restore_original_human_message(message).model_dump()
            metadata: dict[str, Any] = {"caller": "lead_agent", **seed_metadata}
        elif isinstance(message, AIMessage):
            event_type = "llm.ai.response"
            content = message.model_dump()
            metadata = {"caller": "lead_agent", **seed_metadata}
        elif isinstance(message, ToolMessage):
            event_type = "llm.tool.result"
            content = message.model_dump()
            metadata = dict(seed_metadata)
        else:
            # system / remove / summary 산출물은 thread feed에 들어가지 않는다.
            continue
        events.append(
            {
                "thread_id": thread_id,
                "run_id": f"{run_id_prefix}-{turn_index}",
                "event_type": event_type,
                "category": "message",
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
            }
        )
    return events


def build_branch_history_seed_events(
    messages: Sequence[Any],
    *,
    thread_id: str,
    run_id_prefix: str,
    parent_thread_id: str,
) -> list[dict]:
    """상속받은 branch 히스토리를 branch의 비어 있는 event feed로 직렬화한다."""
    return _build_history_seed_events(
        messages,
        thread_id=thread_id,
        run_id_prefix=run_id_prefix,
        seed_metadata={
            "branch_seed": True,
            "branch_parent_thread_id": parent_thread_id,
        },
    )


def build_checkpoint_history_seed_events(
    messages: Sequence[Any],
    *,
    thread_id: str,
    run_id_prefix: str,
) -> list[dict]:
    """thread의 비어 있는 event feed를 위해 레거시 checkpoint 히스토리를 직렬화한다.

    branch seed의 메시지 정규화와 turn별 합성 run 그룹핑을 재사용하되, migration 전용 metadata를
    찍어서 이 row들이 다른 thread에서 상속받은 히스토리로 오인되지 않게 한다.
    """
    return _build_history_seed_events(
        messages,
        thread_id=thread_id,
        run_id_prefix=run_id_prefix,
        seed_metadata={"checkpoint_history_seed": True},
    )


class RunJournal(BaseCallbackHandler):
    """event를 RunEventStore에 기록하는 LangChain callback handler."""

    # subagent는 다른 thread의 persistent event loop에서 실행될 수 있다. 이 handler는 loop 지역
    # task와 부모 run용으로 만든 store/pool을 소유하므로, 격리된 loop의 context 복사기가 이것을
    # 상속해서는 안 된다. LangGraph 자체 stream callback은 상속 가능하게 남아 자식 token frame이
    # 계속 흐르게 한다.
    deerflow_loop_bound = True

    # 모든 callback은 메모리 상의 run state만 갱신하거나 async IO를 예약한다. callback을 run의
    # event-loop thread에 두면 병렬 tool call에서 오는 변경이 직렬화되고, 취소된 executor
    # callback이 terminal delivery 기록 및 flush와 경쟁하는 것을 막는다.
    run_inline = True

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        event_store: RunEventStore,
        *,
        track_token_usage: bool = True,
        flush_threshold: int = 20,
        progress_reporter: Callable[[dict], Awaitable[None]] | None = None,
        progress_flush_interval: float = 5.0,
    ):
        super().__init__()
        self.run_id = run_id
        self.thread_id = thread_id
        self._store = event_store
        self._track_tokens = track_token_usage
        self._flush_threshold = flush_threshold
        self._progress_reporter = progress_reporter
        self._progress_flush_interval = progress_flush_interval

        # 쓰기 buffer
        self._buffer: list[dict] = []
        self._pending_flush_tasks: set[asyncio.Task[None]] = set()
        self._pending_progress_task: asyncio.Task[None] | None = None
        self._pending_progress_delayed = False
        self._progress_dirty = False
        self._last_progress_flush = 0.0

        # token 누적기
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        self._llm_call_count = 0

        # 호출자별 token 누적기
        self._lead_agent_tokens = 0
        self._subagent_tokens = 0
        self._middleware_tokens = 0

        # 모델별 token 누적기
        self._tokens_by_model: dict[str, dict[str, int]] = {}

        # 중복 제거: LangChain은 같은 run_id에 대해 on_llm_end를 여러 번 발생시킬 수 있다
        self._counted_llm_run_ids: set[str] = set()
        self._counted_external_source_ids: set[str] = set()
        self._counted_message_llm_run_ids: set[str] = set()
        self._memory_context_recorded = False

        # 편의 필드
        self._last_ai_msg: str | None = None
        self._first_human_msg: str | None = None
        self._msg_count = 0
        self._had_llm_error_fallback = False
        self._llm_error_fallback_message: str | None = None

        # latency 추적
        self._llm_start_times: dict[str, float] = {}  # langchain run_id -> 시작 시각

        # LLM 요청/응답 추적
        self._llm_call_index = 0
        self._seen_llm_starts: set[str] = set()  # on_chat_model_start가 발생한 langchain run_id들
        self._current_run_tool_call_names: dict[str, str] = {}
        self._persisted_tool_message_identities: set[str] = set()

        # terminal run.delivery event를 위한 artifact 생성 추적(#4272 slice 1).
        # (path, tool_name)으로 중복 제거하고 삽입 순서를 유지한다.
        self._produced_artifacts: list[tuple[str, str | None]] = []
        self._produced_artifact_keys: set[tuple[str, str | None]] = set()

    # -- lifecycle callback 처리 --

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """메시지의 혼합 content 형태에서 표시 가능한 텍스트를 추출한다."""
        return message_to_text(message, text_attribute_fallback=True)

    def _record_message_summary(self, message: BaseMessage, *, caller: str | None = None) -> None:
        """저장될 run row를 위해 run 수준의 편의 필드를 갱신한다."""
        self._msg_count += 1

        # ``last_ai_message``는 lead agent가 사용자에게 보여 준 답변이어야 한다.
        # middleware/subagent의 모델 호출이나 tool call만 있는 빈 AI 메시지가 마지막으로 쓸모
        # 있는 assistant 텍스트를 덮어써서는 안 된다.
        is_ai_message = isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
        if is_ai_message and (caller is None or caller == "lead_agent"):
            text = self._message_text(message).strip()
            if text:
                self._last_ai_msg = text[:2000]

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        caller = self._identify_caller(tags)
        if parent_run_id is None:
            # root graph 호출 — run 시작에 대한 trace event를 하나만 낸다.
            chain_name = (serialized or {}).get("name", "unknown")
            self._put(
                event_type=RUN_START_EVENT.event_type,
                category=RUN_START_EVENT.category,
                content={"chain": chain_name},
                metadata={"caller": caller, **(metadata or {})},
            )

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # 중첩된 chain 종료는 내부 graph node에서도 발생한다. 사용자에게 보이는 run
        # lifecycle을 나타내는 것은 root chain뿐이다.
        if parent_run_id is not None:
            return
        self._reconcile_final_tool_messages(outputs)
        self._put(
            event_type=RUN_END_EVENT.event_type,
            category=RUN_END_EVENT.category,
            content=outputs,
            metadata={"status": "success"},
        )
        self._flush_sync()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._put(
            event_type=RUN_ERROR_EVENT.event_type,
            category=RUN_ERROR_EVENT.category,
            content=str(error),
            metadata={"error_type": type(error).__name__},
        )
        self._flush_sync()

    # -- LLM callback 처리 --

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """사용자에게 보이는 첫 prompt를 llm.human.input으로 기록한다.

        첫 human message를 추출하기에도 여기가 정석이다. 메시지가 완전히 구조화돼 있고, 실제 LLM
        호출에서만 발생하며, content가 checkpoint trimming으로 압축되는 일이 없다.
        """
        rid = str(run_id)
        self._llm_start_times[rid] = time.monotonic()
        self._llm_call_index += 1
        self._seen_llm_starts.add(rid)

        logger.debug(
            "on_chat_model_start %s: tags=%s num_batches=%d message_counts=%s",
            run_id,
            tags,
            len(messages),
            [len(batch) for batch in messages],
        )

        # 이번 run에서 lead agent로 보내진 첫 사용자 메시지를 기록한다.
        caller = self._identify_caller(tags)
        if caller == "lead_agent" and not self._first_human_msg and messages:
            for batch in reversed(messages):
                for m in reversed(batch):
                    if _should_persist_human_input_message(m):
                        persisted_message = restore_original_human_message(m)
                        self.set_first_human_message(self._message_text(persisted_message))
                        self._put(
                            event_type=LLM_HUMAN_INPUT_EVENT.event_type,
                            category=LLM_HUMAN_INPUT_EVENT.category,
                            content=persisted_message.model_dump(),
                            metadata={"caller": caller},
                        )
                        self._record_message_summary(persisted_message, caller=caller)
                        break
                if self._first_human_msg:
                    break

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        # fallback 경로다. on_chat_model_start를 우선 쓰며, 여기서는 latency만 추적한다.
        self._llm_start_times[str(run_id)] = time.monotonic()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        messages: list[AnyMessage] = []
        logger.debug("on_llm_end %s: tags=%s", run_id, tags)
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "message"):
                    messages.append(gen.message)
                else:
                    logger.warning(f"on_llm_end {run_id}: generation has no message attribute: {gen}")

        for message in messages:
            caller = self._identify_caller(tags)
            self._remember_current_run_tool_calls(message, caller=caller)

            # latency 계산
            rid = str(run_id)
            start = self._llm_start_times.pop(rid, None)
            latency_ms = int((time.monotonic() - start) * 1000) if start else None

            # 메시지에서 얻은 token usage
            usage = getattr(message, "usage_metadata", None)
            usage_dict = dict(usage) if usage else {}
            additional_kwargs = getattr(message, "additional_kwargs", None) or {}
            if isinstance(additional_kwargs, dict) and additional_kwargs.get("deerflow_error_fallback"):
                self._had_llm_error_fallback = True
                detail = additional_kwargs.get("error_detail")
                reason = additional_kwargs.get("error_reason")
                fallback_text = self._message_text(message).strip()
                if isinstance(detail, str) and detail.strip():
                    self._llm_error_fallback_message = detail.strip()
                elif isinstance(reason, str) and reason.strip():
                    self._llm_error_fallback_message = reason.strip()
                elif fallback_text:
                    self._llm_error_fallback_message = fallback_text[:2000]

            # 호출 index 결정
            call_index = self._llm_call_index
            if rid not in self._seen_llm_starts:
                # fallback: on_chat_model_start가 호출되지 않은 경우
                self._llm_call_index += 1
                call_index = self._llm_call_index
                self._seen_llm_starts.add(rid)

            # 메시지 event: checkpoint와 정렬된 llm.ai.response payload.
            self._put(
                event_type=LLM_AI_RESPONSE_EVENT.event_type,
                category=LLM_AI_RESPONSE_EVENT.category,
                content=message.model_dump(),
                metadata={
                    "caller": caller,
                    "usage": usage_dict,
                    "latency_ms": latency_ms,
                    "llm_call_index": call_index,
                },
            )
            if rid not in self._counted_message_llm_run_ids:
                self._record_message_summary(message, caller=caller)

            # token 누적(같은 응답에 대해 callback이 여러 번 발생해도 이중 집계되지 않도록
            # langchain run_id로 중복 제거)
            if self._track_tokens:
                input_tk = usage_dict.get("input_tokens", 0) or 0
                output_tk = usage_dict.get("output_tokens", 0) or 0
                total_tk = usage_dict.get("total_tokens", 0) or 0
                if total_tk == 0:
                    total_tk = input_tk + output_tk
                if total_tk > 0 and rid not in self._counted_llm_run_ids:
                    self._counted_llm_run_ids.add(rid)
                    self._total_input_tokens += input_tk
                    self._total_output_tokens += output_tk
                    self._total_tokens += total_tk
                    self._llm_call_count += 1

                    if caller.startswith("subagent:"):
                        self._subagent_tokens += total_tk
                    elif caller.startswith("middleware:"):
                        self._middleware_tokens += total_tk
                    else:
                        self._lead_agent_tokens += total_tk

                    # 모델별 bucket
                    response_metadata = getattr(message, "response_metadata", None) or {}
                    per_call_model: str | None = None
                    if isinstance(response_metadata, Mapping):
                        per_call_model = response_metadata.get("model_name") or response_metadata.get("model")
                    self._record_model_usage(per_call_model, input_tk, output_tk, total_tk, self._extract_cache_read(usage_dict))

                    self._schedule_progress_flush()

        if messages:
            self._counted_message_llm_run_ids.add(str(run_id))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._llm_start_times.pop(str(run_id), None)
        self._put(
            event_type=LLM_ERROR_EVENT.event_type,
            category=LLM_ERROR_EVENT.category,
            content=str(error),
        )

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        """tool 시작 event를 처리하고, 나중에 대조할 수 있게 tool call ID를 캐시한다"""
        tool_call_id = str(run_id)
        logger.debug("Tool start for node %s, tool_call_id=%s, tags=%s", run_id, tool_call_id, tags)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        """tool 종료 event를 처리해 메시지를 추가하고 node 데이터를 정리한다"""
        try:
            if isinstance(output, ToolMessage):
                msg = cast(ToolMessage, output)
                self._persist_tool_result_message(msg)
            elif isinstance(output, Command):
                cmd = cast(Command, output)
                messages = cmd.update.get("messages", [])
                # 비어 있지 않은 ``artifacts`` 갱신은 성공 경로에서만 나온다(예: 검증이 실패하면
                # present_files는 state를 건드리지 않고 오류 ToolMessage를 반환한다). 따라서 그
                # 존재 자체가 artifact 생성 신호다(#4272 slice 1).
                artifacts = cmd.update.get("artifacts")
                artifact_tool_names: set[str] = set()
                for message in messages:
                    if isinstance(message, BaseMessage):
                        self._persist_tool_result_message(message)
                        if artifacts and isinstance(message, ToolMessage):
                            tool_call_id = getattr(message, "tool_call_id", None)
                            if isinstance(tool_call_id, str):
                                tool_name = self._current_run_tool_call_names.get(tool_call_id)
                                if tool_name:
                                    artifact_tool_names.add(tool_name)
                    else:
                        logger.warning(f"on_tool_end {run_id}: command update message is not BaseMessage: {type(message)}")
                if artifacts:
                    artifact_tool_name = next(iter(artifact_tool_names)) if len(artifact_tool_names) == 1 else None
                    self._record_produced_artifacts(artifacts, artifact_tool_name)
            else:
                logger.warning(f"on_tool_end {run_id}: output is not ToolMessage: {type(output)}")
        finally:
            logger.debug("Tool end for node %s", run_id)

    # -- 내부 메서드 --

    @staticmethod
    def _message_identity(message: BaseMessage) -> str | None:
        tool_call_id = getattr(message, "tool_call_id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            return f"tool:{tool_call_id}"
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id:
            return f"message:{message_id}"
        return None

    @staticmethod
    def _tool_call_value(tool_call: Any, key: str) -> Any:
        if isinstance(tool_call, Mapping):
            return tool_call.get(key)
        return getattr(tool_call, key, None)

    def _remember_current_run_tool_calls(self, message: AnyMessage, *, caller: str) -> None:
        if caller != "lead_agent":
            return
        is_ai_message = isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
        if not is_ai_message:
            return
        tool_calls = getattr(message, "tool_calls", None) or []
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            tool_call_id = self._tool_call_value(tool_call, "id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            name = self._tool_call_value(tool_call, "name")
            self._current_run_tool_call_names[tool_call_id] = str(name or "")

    def _persist_tool_result_message(self, message: BaseMessage) -> None:
        self._put(
            event_type=LLM_TOOL_RESULT_EVENT.event_type,
            category=LLM_TOOL_RESULT_EVENT.category,
            content=message.model_dump(),
        )
        identity = self._message_identity(message)
        if identity:
            self._persisted_tool_message_identities.add(identity)
        self._record_message_summary(message)

    def _final_output_messages(self, outputs: Any) -> list[Any]:
        if isinstance(outputs, Mapping):
            messages = outputs.get("messages", [])
            return messages if isinstance(messages, list) else []
        return []

    def _should_reconcile_tool_message(self, message: ToolMessage) -> bool:
        if message.additional_kwargs.get("hide_from_ui") is True:
            return False
        tool_call_id = getattr(message, "tool_call_id", None)
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return False
        tool_call_name = self._current_run_tool_call_names.get(tool_call_id)
        if tool_call_name is None:
            return False
        message_name = getattr(message, "name", None)
        if message_name not in _RECONCILED_TOOL_MESSAGE_NAMES and tool_call_name not in _RECONCILED_TOOL_MESSAGE_NAMES:
            return False
        identity = self._message_identity(message)
        return identity is not None and identity not in self._persisted_tool_message_identities

    def _reconcile_final_tool_messages(self, outputs: Any) -> None:
        for message in self._final_output_messages(outputs):
            if not isinstance(message, ToolMessage):
                continue
            if self._should_reconcile_tool_message(message):
                self._persist_tool_result_message(message)

    def _put(self, *, event_type: str, category: str, content: str | dict = "", metadata: dict | None = None) -> None:
        self._buffer.append(
            {
                "thread_id": self.thread_id,
                "run_id": self.run_id,
                "event_type": event_type,
                "category": category,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        if len(self._buffer) >= self._flush_threshold:
            self._flush_sync()

    def _flush_sync(self) -> None:
        """buffer를 RunEventStore로 best-effort flush한다.

        BaseCallbackHandler 메서드는 동기다. event loop가 돌고 있으면 async ``put_batch``를
        예약하고, 아니면 event를 buffer에 남겨 두었다가 worker의 ``finally`` 블록에 있는 async
        ``flush()`` 호출로 flush한다.
        """
        if not self._buffer:
            return
        # 이미 flush가 진행 중이면 건너뛴다 — 여러 fire-and-forget task가 같은 SQLite 파일에
        # 동시에 쓰는 것을 피한다.
        if self._pending_flush_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # event loop가 없으면 나중의 async flush를 위해 event를 buffer에 남긴다.
            return
        batch = self._buffer.copy()
        self._buffer.clear()
        task = loop.create_task(self._flush_async(batch))
        self._pending_flush_tasks.add(task)
        task.add_done_callback(self._on_flush_done)

    async def _flush_async(self, batch: list[dict]) -> None:
        try:
            await self._store.put_batch(batch)
        except Exception:
            logger.warning(
                "Failed to flush %d events for run %s — returning to buffer",
                len(batch),
                self.run_id,
                exc_info=True,
            )
            # 실패한 event는 다음 flush에서 재시도하도록 buffer로 되돌린다
            self._buffer = batch + self._buffer

    def _on_flush_done(self, task: asyncio.Task) -> None:
        self._pending_flush_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.warning("Journal flush task failed: %s", exc)

    def _identify_caller(self, tags: list[str] | None) -> str:
        _tags = tags or []
        for tag in _tags:
            if isinstance(tag, str) and (tag.startswith("subagent:") or tag.startswith("middleware:") or tag == "lead_agent"):
                return tag
        # 기본값은 lead_agent다. 메인 agent graph는 callback tag를 주입하지 않는 반면 subagent와
        # middleware는 자기 자신을 명시적으로 tag하기 때문이다.
        return "lead_agent"

    def _record_model_usage(
        self,
        model_name: str | None,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cache_read_tokens: int = 0,
    ) -> None:
        """LLM 호출 하나의 token usage를 모델별 누적기에 더한다.

        ``model_name``이 없거나 비어 있으면 공용 ``"unknown"`` bucket으로 합쳐서, provider가
        ``response_metadata.model_name``을 주지 않아도 분해 집계를 쓸 수 있게 한다.

        ``cache_read_tokens``(prompt-cache 적중,
        ``usage_metadata.input_token_details.cache_read``에서 옴)는 sparse bucket 키로 저장한다 —
        0이 아닐 때만 쓴다 — 그래서 cache를 보고하지 않는 provider의 bucket은 기존 형태를
        유지한다.
        """
        if total_tokens <= 0:
            return
        bucket = self._tokens_by_model.setdefault(
            model_name or "unknown",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        bucket["input_tokens"] += int(input_tokens or 0)
        bucket["output_tokens"] += int(output_tokens or 0)
        bucket["total_tokens"] += int(total_tokens)
        if cache_read_tokens > 0:
            bucket["cache_read_tokens"] = bucket.get("cache_read_tokens", 0) + int(cache_read_tokens)

    @staticmethod
    def _extract_cache_read(usage_dict: dict) -> int:
        """LangChain의 정규화된 usage에서 prompt-cache 적중 input token 수를 얻는다."""
        details = usage_dict.get("input_token_details") or {}
        if not isinstance(details, Mapping):
            return 0
        try:
            return max(int(details.get("cache_read") or 0), 0)
        except (TypeError, ValueError):
            return 0

    # -- 공개 메서드(worker가 호출) --

    def record_external_llm_usage_records(
        self,
        records: list[dict[str, int | str | None]],
    ) -> None:
        """외부 소스(예: subagent)의 token usage를 기록한다.

        각 record는 다음을 담아야 한다:
            source_run_id: 이중 집계를 막는 고유 식별자
            caller: 호출자 tag (예: "subagent:general-purpose")
            model_name: 호출별 실제 모델 이름(str 또는 None. 없으면 ``"unknown"`` bucket으로
                되돌아간다)
            input_tokens: input token 수
            output_tokens: output token 수
            total_tokens: 전체 token 수(0이거나 없으면 input+output으로 계산한다)
            cache_read_tokens: 선택 항목인 prompt-cache 적중 input token 수
        """
        if not self._track_tokens:
            return
        for record in records:
            source_id = str(record.get("source_run_id", ""))
            if not source_id:
                continue
            if source_id in self._counted_external_source_ids:
                continue

            total_tk = record.get("total_tokens", 0) or 0
            if total_tk <= 0:
                input_tk = record.get("input_tokens", 0) or 0
                output_tk = record.get("output_tokens", 0) or 0
                total_tk = input_tk + output_tk
            if total_tk <= 0:
                continue

            input_tk = record.get("input_tokens", 0) or 0
            output_tk = record.get("output_tokens", 0) or 0

            self._counted_external_source_ids.add(source_id)
            self._total_input_tokens += input_tk
            self._total_output_tokens += output_tk
            self._total_tokens += total_tk

            caller = str(record.get("caller", ""))
            if caller.startswith("subagent:"):
                self._subagent_tokens += total_tk
            elif caller.startswith("middleware:"):
                self._middleware_tokens += total_tk
            else:
                self._lead_agent_tokens += total_tk

            cache_read_tk = record.get("cache_read_tokens", 0) or 0
            self._record_model_usage(record.get("model_name"), input_tk, output_tk, total_tk, int(cache_read_tk))

            self._schedule_progress_flush()

    def set_first_human_message(self, content: str) -> None:
        """편의 필드용으로 첫 human message를 기록한다."""
        self._first_human_msg = content[:2000] if content else None

    def record_middleware(self, tag: str, *, name: str, hook: str, action: str, changes: dict) -> None:
        """middleware의 state 변경 event를 기록한다.

        middleware 구현이 의미 있는 state 변경(예: title 생성, 요약, HITL 승인)을 수행했을 때
        호출한다. 관찰만 하는 middleware는 호출하면 안 된다.

        Args:
            tag: middleware의 짧은 식별자(예: "title", "summarize", "guardrail").
                 event_type="middleware:{tag}"를 구성하는 데 쓰이며, 저장되는 event-type 컬럼
                 폭에 의해 길이가 제한된다.
            name: middleware 클래스의 전체 이름.
            hook: 이 동작을 유발한 lifecycle hook(예: "after_model").
            action: 수행한 구체적인 동작(예: "generate_title").
            changes: 이루어진 state 변경을 설명하는 dict.
        """
        self._put(
            event_type=MIDDLEWARE_EVENT_PATTERN.event_type(tag),
            category=MIDDLEWARE_EVENT_PATTERN.category,
            content={"name": name, "hook": hook, "action": action, "changes": changes},
        )

    def record_memory_context(self, *, content_sha256: str) -> None:
        """이번 run에 적용된 숨김 memory 블록을 기록한다.

        블록 전체는 이미 checkpoint state에 있고 사용자 데이터를 담을 수 있으므로, event에는
        정확한 SHA-256 정체성만 저장한다. 운영자는 기존 run-events 디버그 API로 이를 조회해,
        내용을 복사하지 않고도 run별로 실제 사용된 memory를 비교한다.
        """
        if self._memory_context_recorded:
            return
        self._put(
            event_type=MEMORY_CONTEXT_EVENT.event_type,
            category=MEMORY_CONTEXT_EVENT.category,
            content={"content_sha256": content_sha256},
        )
        self._memory_context_recorded = True

    def _record_produced_artifacts(self, artifacts: Any, tool_name: str | None) -> None:
        """생성된 artifact 경로를 (path, tool_name) 기준으로 중복 제거하며 누적한다."""
        if not isinstance(artifacts, list):
            return
        for path in artifacts:
            if not isinstance(path, str) or not path:
                continue
            key = (path, tool_name)
            if key not in self._produced_artifact_keys:
                self._produced_artifact_keys.add(key)
                self._produced_artifacts.append(key)

    def get_delivery_content(self) -> dict[str, Any]:
        """이번 run에 대해 누적된 terminal delivery 사실을 반환한다.

        이것은 판정이 아니라 사실 기록이다. artifact를 만들지 않은 run은 ``presented: 0``을 낸다.
        """
        by_tool: dict[str, list[str]] = {}
        paths: list[str] = []
        for path, tool_name in self._produced_artifacts:
            paths.append(path)
            if tool_name:
                by_tool.setdefault(tool_name, []).append(path)
        return {"presented": len(paths), "paths": paths, "by_tool": by_tool}

    def record_delivery(self) -> None:
        """이번 run의 terminal ``run.delivery`` event를 buffer에 넣는다(#4272 slice 1).

        journal을 직접 쓰는 사용자를 위해 남겨 둔다. worker는 event store의 idempotent singleton
        쓰기를 사용하므로, crash 복구가 이를 안전하게 backfill할 수 있다.
        """
        self._put(
            event_type="run.delivery",
            category="outputs",
            content=self.get_delivery_content(),
        )

    async def flush(self) -> None:
        """남은 buffer를 강제로 flush한다. worker의 finally 블록에서 호출된다."""
        if self._pending_flush_tasks:
            await asyncio.gather(*tuple(self._pending_flush_tasks), return_exceptions=True)
        while self._pending_progress_task is not None and not self._pending_progress_task.done():
            if self._pending_progress_delayed:
                self._pending_progress_task.cancel()
                await asyncio.gather(self._pending_progress_task, return_exceptions=True)
                self._progress_dirty = False
                self._pending_progress_delayed = False
                break
            await asyncio.gather(self._pending_progress_task, return_exceptions=True)

        while self._buffer:
            batch = self._buffer[: self._flush_threshold]
            del self._buffer[: self._flush_threshold]
            try:
                await self._store.put_batch(batch)
            except Exception:
                self._buffer = batch + self._buffer
                raise

    def _schedule_progress_flush(self) -> None:
        """진행 중인 run을 관측할 수 있도록 best-effort로 throttle된 progress snapshot을 예약한다."""
        if self._progress_reporter is None:
            return
        now = time.monotonic()
        elapsed = now - self._last_progress_flush
        if elapsed < self._progress_flush_interval:
            self._progress_dirty = True
            self._schedule_delayed_progress_flush(self._progress_flush_interval - elapsed)
            return
        if self._pending_progress_task is not None and not self._pending_progress_task.done():
            self._progress_dirty = True
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._progress_dirty = False
        self._pending_progress_task = loop.create_task(self._flush_progress_async(snapshot=self.get_completion_data()))

    def _schedule_delayed_progress_flush(self, delay: float) -> None:
        if self._pending_progress_task is not None and not self._pending_progress_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.0, delay)
        self._pending_progress_delayed = delay > 0
        self._pending_progress_task = loop.create_task(self._flush_progress_async(delay=delay))

    async def _flush_progress_async(self, *, snapshot: dict | None = None, delay: float = 0.0) -> None:
        if self._progress_reporter is None:
            return
        if delay > 0:
            self._pending_progress_delayed = True
            await asyncio.sleep(delay)
            self._pending_progress_delayed = False
        dirty_before_write = self._progress_dirty
        self._progress_dirty = False
        snapshot_to_write = snapshot or self.get_completion_data()
        try:
            await self._progress_reporter(snapshot_to_write)
            self._last_progress_flush = time.monotonic()
        except Exception:
            logger.warning("Failed to persist progress snapshot for run %s", self.run_id, exc_info=True)
        if dirty_before_write or self._progress_dirty:
            self._progress_dirty = False
            self._pending_progress_task = None
            self._schedule_delayed_progress_flush(self._progress_flush_interval)

    def get_completion_data(self) -> dict:
        """run 완료 시 사용할, 누적된 token 및 메시지 데이터를 반환한다."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_tokens,
            "llm_call_count": self._llm_call_count,
            "lead_agent_tokens": self._lead_agent_tokens,
            "subagent_tokens": self._subagent_tokens,
            "middleware_tokens": self._middleware_tokens,
            "token_usage_by_model": {model: dict(usage) for model, usage in self._tokens_by_model.items()},
            "message_count": self._msg_count,
            "last_ai_message": self._last_ai_msg,
            "first_human_message": self._first_human_msg,
        }

    @property
    def had_llm_error_fallback(self) -> bool:
        return self._had_llm_error_fallback

    @property
    def llm_error_fallback_message(self) -> str | None:
        return self._llm_error_fallback_message
