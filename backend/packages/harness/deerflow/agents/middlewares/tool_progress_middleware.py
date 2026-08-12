"""상태 기계로 task 수준의 도구 호출 진행 상황을 추적하는 middleware.

RFC #3177 구현. 구조화된 도구 결과 신호가 (thread, tool)별 상태 기계를 구동해 정체와
반복을 감지하고, 이른 시점에 힌트를 주입하며(WARNED), 도구가 더 이상 가치를 내지 못하면
완전히 차단한다(BLOCKED).

구조:
  ToolProgressMiddleware (바깥)
    └── handler → ToolErrorHandlingMiddleware (안쪽) → 실제 도구
                                                              ↓
  ToolProgressMiddleware가 정규화된 결과에서 deerflow_tool_meta를 읽는다

(thread_id, tool_name)별 상태 전이:
  ACTIVE → WARNED (문제가 stagnation_threshold개 쌓일 때)
  문제 없는 호출이 하나라도 들어오면 consecutive_problems=0으로 리셋하고 ACTIVE로 돌아간다.

  WARNED에서 BLOCKED로 승격 가능한지는 recoverable_by_model에 달려 있다.
  - recoverable_by_model=True (no_results, not_found, permission, Jaccard 중복 success):
      WARNED가 종착점이다. 모델이 힌트를 받았고 전략을 바꿀 것으로 기대되며, 차단하면
      다른 파라미터로 하는 정당한 재시도까지 막힌다.
  - recoverable_by_model=False, action≠stop (transient, rate_limited):
      문제가 warn_escalation_count개 더 쌓이면 WARNED → BLOCKED. 같은 도구를 재시도해
      해결할 수 없으므로 차단이 API 호출을 아낀다.
  - recoverable_by_model=False, action=stop (auth, config, internal):
      첫 발생에서 즉시 BLOCKED. 재시도가 도움이 되지 않는다.

LoopDetectionMiddleware(middleware 위치 23)와의 역할 분담:
  ToolProgressMiddleware(위치 10)는 결과 품질 guard다. 도구 실행 후 발동해 돌아온 내용을
  검사하고, 새 정보를 못 내놓는 *특정 도구*를 차단한다.

  LoopDetectionMiddleware는 호출 패턴 guard다. 모델 응답 후(도구 실행 전) 발동해 AIMessage의
  tool_calls 서명을 검사하고, 결과와 무관하게 같은 호출을 반복하면 *턴 전체*를 멈춘다.

  둘은 경쟁이 아니라 보완 관계다.
  - ToolProgressMiddleware는 세밀하다(도구 단위 BLOCK, 나머지 도구는 정상).
  - LoopDetectionMiddleware는 거칠다(모든 tool_calls 제거, 턴 종료).
  - 둘 다 같은 모델 호출에서 HumanMessage 힌트를 주입해도 충돌하지 않는다.
    모델은 두 힌트를 모두 보고 판단할 수 있다.
  - LoopDetectionMiddleware가 hard-stop하면(tool_calls 제거) wrap_tool_call이 발생하지 않아
    ToolProgressMiddleware는 아예 발동하지 않는다. 이중 정지는 없다.
  - ToolProgressMiddleware가 도구를 BLOCK해도(error ToolMessage 반환) 모델은 여전히
    LoopDetectionMiddleware가 추적하는 도구 호출을 하므로, 둘은 각자의 독립 상태로 계속 동작한다.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deerflow.agents.middlewares.tool_result_meta import TOOL_META_KEY, ToolResultMeta

if TYPE_CHECKING:
    from deerflow.config.tool_progress_config import ToolProgressConfig

logger = logging.getLogger(__name__)

_MAX_PENDING_PER_RUN = 3
# 아주 큰 도구 결과에 O(n) 정규식 작업을 하지 않도록 Jaccard word-set 계산 크기를 제한한다.
_MAX_CONTENT_FOR_WORDSET = 8192


# ---------------------------------------------------------------------------
# 상태 자료 구조


@dataclass(slots=True)
class ToolPhaseState:
    """(thread_id, tool_name)별 추적 상태."""

    phase: Literal["active", "warned", "blocked"] = "active"
    consecutive_problems: int = 0
    block_reason: str | None = None
    # 불변 tuple이다. recent_word_sets를 생략한 dataclasses.replace() 호출(문제 경로)이
    # 이전 상태와 새 상태 사이에서 가변 리스트를 공유해 .append()로 조용히 서로를
    # 오염시키는 일을 막는다.
    recent_word_sets: tuple[frozenset[str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 콘텐츠 헬퍼


def word_set(content: str) -> frozenset[str]:
    """Jaccard 유사도를 위해 길이 3 이상의 소문자 단어를 추출한다.

    큰 도구 결과(예: 웹 페이지)에서 메모리와 CPU 비용을 제한하려고 콘텐츠를
    _MAX_CONTENT_FOR_WORDSET 문자로 자른다. 그 뒤 내용은 집합에서 빠지지만, 중복 감지는
    보장이 아니라 휴리스틱이므로 문제없다.
    """
    return frozenset(re.findall(r"\b\w{3,}\b", content[:_MAX_CONTENT_FOR_WORDSET].lower()))


def is_near_duplicate(
    current: frozenset[str],
    recent: Sequence[frozenset[str]],
    threshold: float,
    min_words: int,
) -> bool:
    """current가 최근 word set 3개 중 하나와 유사하면 True를 반환한다."""
    if len(current) < min_words:
        return False
    for prev in recent[-3:]:
        if len(prev) < min_words:
            continue
        union = len(current | prev)
        if union == 0:
            continue
        if len(current & prev) / union >= threshold:
            return True
    return False


def _message_content_str(msg: ToolMessage) -> str:
    return msg.content if isinstance(msg.content, str) else ""


def _parse_tool_meta(meta_dict: object) -> ToolResultMeta | None:
    """raw dict에서 ToolResultMeta를 안전하게 역직렬화한다. 스키마가 맞지 않으면 None."""
    if not isinstance(meta_dict, dict):
        return None
    try:
        return ToolResultMeta(**meta_dict)
    except TypeError:
        logger.warning("Unexpected tool meta schema, skipping progress tracking: %s", meta_dict)
        return None


# ---------------------------------------------------------------------------
# 힌트 / 차단 사유 포맷팅


def _format_hint(meta: ToolResultMeta) -> str:
    action_map = {
        "rewrite_query": "Try rephrasing your search query with different keywords or approach.",
        "try_alternative": "Consider using a different tool or strategy.",
        "summarize": "Consider summarizing your current findings and moving forward.",
        "stop": "Do not retry this operation — it is not recoverable.",
        # 거의 중복인 success 결과: recommended_next_action은 기본이 "continue"지만,
        # 같은 콘텐츠를 다시 가져오지 않도록 모델은 전략을 바꿔야 한다.
        "continue": "Try rephrasing your query or using a different search term.",
    }
    base = {
        "no_results": "[PROGRESS HINT] Your search returned no results.",
        "not_found": "[PROGRESS HINT] The resource was not found repeatedly.",
        "rate_limited": "[PROGRESS HINT] The tool is being rate-limited.",
        "transient": "[PROGRESS HINT] The tool encountered repeated transient failures.",
        "partial_success": "[PROGRESS HINT] The tool has returned incomplete results multiple times.",
        # Jaccard 기준 거의 중복인 success: 도구가 같은 콘텐츠를 반복해서 돌려주고 있다.
        "success": "[PROGRESS HINT] The tool is returning duplicate results.",
    }.get(
        meta.error_type or meta.status,
        "[PROGRESS HINT] The tool is not producing new information.",
    )
    suffix = action_map.get(meta.recommended_next_action, "")
    return f"{base} {suffix}".strip()


def _block_reason(meta: ToolResultMeta) -> str:
    return {
        "no_results": "Repeated no-results — rewrite your query or try a different tool.",
        "not_found": "Repeated not-found — rewrite your query or try a different resource.",
        "rate_limited": "Repeated rate-limiting — summarize current findings and proceed.",
        "transient": "Repeated transient failures — try a different approach.",
        "auth": "Authentication failure — this tool cannot be used.",
        "config": "Tool is not configured — this tool cannot be used.",
        "internal": "Repeated internal errors — this tool is unavailable.",
    }.get(
        meta.error_type or "",
        "Tool has not produced new information after multiple attempts — summarize and move on.",
    )


# ---------------------------------------------------------------------------
# Middleware


class ToolProgressMiddleware(AgentMiddleware[AgentState]):
    """상태 기계 기반 도구 정체 guard (RFC #3177)."""

    def __init__(
        self,
        *,
        stagnation_threshold: int = 3,
        warn_escalation_count: int = 2,
        inject_assessment: bool = True,
        jaccard_threshold: float = 0.8,
        min_words: int = 10,
        exempt_tools: set[str] | None = None,
        max_tracked_threads: int = 100,
    ) -> None:
        self._stagnation_threshold = stagnation_threshold
        self._warn_escalation = warn_escalation_count
        self._inject_assessment = inject_assessment
        self._jaccard_threshold = jaccard_threshold
        self._min_words = min_words
        self._exempt_tools: set[str] = exempt_tools if exempt_tools is not None else {"ask_clarification", "write_todos", "present_files", "task"}
        self._max_tracked_threads = max_tracked_threads

        # asyncio.Lock이 아닌 threading.Lock을 쓴다. 임계 구역은 I/O 없는 짧은 in-memory
        # dict 연산이라 event loop 지연 위험이 무시할 수준이다. asyncio.Lock은 subagent
        # executor thread pool이 쓰는 동기 wrap_tool_call 경로를 보호하지 못해 락이 둘
        # 필요해진다. 기존 LoopDetectionMiddleware와 같은 방식이며 자세한 내용은 모듈
        # docstring을 참고한다.
        self._lock = threading.Lock()
        # LRU 방식 저장소: thread_id → {tool_name → ToolPhaseState}
        self._phase_states: OrderedDict[str, dict[str, ToolPhaseState]] = OrderedDict()
        # 대기 중 힌트 큐: (thread_id, run_id) → [힌트 텍스트]
        self._pending: dict[tuple[str, str], list[str]] = defaultdict(list)

    @classmethod
    def from_config(cls, config: ToolProgressConfig) -> ToolProgressMiddleware:
        return cls(
            stagnation_threshold=config.stagnation_threshold,
            warn_escalation_count=config.warn_escalation_count,
            inject_assessment=config.inject_assessment,
            jaccard_threshold=config.jaccard_similarity_threshold,
            min_words=config.min_word_count_for_similarity,
            exempt_tools=set(config.exempt_tools),
            max_tracked_threads=config.max_tracked_threads,
        )

    # ------------------------------------------------------------------
    # Runtime 헬퍼

    @staticmethod
    def _thread_id(runtime: Runtime) -> str:
        tid = runtime.context.get("thread_id") if runtime.context else None
        return str(tid) if tid else "default"

    @staticmethod
    def _run_id(runtime: Runtime) -> str:
        rid = runtime.context.get("run_id") if runtime.context else None
        return str(rid) if rid else "default"

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        return self._thread_id(runtime), self._run_id(runtime)

    # ------------------------------------------------------------------
    # 상태 저장소(호출자가 lock을 잡고 있다)

    def _get_state(self, thread_id: str, tool_name: str) -> ToolPhaseState:
        if thread_id not in self._phase_states:
            self._phase_states[thread_id] = {}
            while len(self._phase_states) > self._max_tracked_threads:
                evicted_thread, _ = self._phase_states.popitem(last=False)
                # 무한정 커지지 않도록 제거된 thread의 대기 힌트도 함께 제거한다.
                for key in [k for k in self._pending if k[0] == evicted_thread]:
                    del self._pending[key]
        self._phase_states.move_to_end(thread_id)
        return self._phase_states[thread_id].get(tool_name, ToolPhaseState())

    def _set_state(self, thread_id: str, tool_name: str, state: ToolPhaseState) -> None:
        self._phase_states[thread_id][tool_name] = state

    def _get_block_reason(self, runtime: Runtime, tool_name: str) -> str | None:
        thread_id = self._thread_id(runtime)
        with self._lock:
            thread_tools = self._phase_states.get(thread_id)
            if thread_tools is None:
                return None
            # 읽기 전용 검사다. 여기서 move_to_end를 호출하면 안 된다. 읽기 경로에서
            # 최신성을 갱신하면 차단된 thread가 LRU에 영구히 남아 정상 thread가 그 자리를
            # 차지하지 못한다. 최신성은 _get_state 쓰기에서만 갱신한다.
            tool_state = thread_tools.get(tool_name)
            return tool_state.block_reason if tool_state is not None and tool_state.phase == "blocked" else None

    def _make_blocked_message(self, request: ToolCallRequest, tool_name: str, block_reason: str) -> ToolMessage:
        return ToolMessage(
            content=f"[TOOL_BLOCKED] {block_reason}",
            tool_call_id=str(request.tool_call.get("id", "")),
            name=tool_name,
            status="error",
            additional_kwargs={
                TOOL_META_KEY: {
                    "status": "error",
                    "error_type": "blocked_by_progress_guard",
                    "recoverable_by_model": True,
                    "recommended_next_action": "summarize",
                    "source": "progress_middleware",
                }
            },
        )

    def _update_state_from_result(
        self,
        result: ToolMessage | Command,
        tool_name: str,
        runtime: Runtime,
    ) -> ToolMessage | Command:
        """도구 결과로 상태 기계를 갱신하고, 필요하면 힌트를 큐에 넣는다."""
        if not isinstance(result, ToolMessage):
            return result
        meta = _parse_tool_meta((result.additional_kwargs or {}).get(TOOL_META_KEY))
        if meta is None:
            if tool_name not in self._exempt_tools:
                logger.warning(
                    "tool_progress: deerflow_tool_meta missing for non-exempt tool %s — verify ToolProgressMiddleware is outer of ToolErrorHandlingMiddleware",
                    tool_name,
                )
            return result
        content = _message_content_str(result)
        thread_id = self._thread_id(runtime)
        with self._lock:
            state = self._get_state(thread_id, tool_name)
            new_state, hint = self._assess_and_transition(state, meta, content)
            self._set_state(thread_id, tool_name, new_state)
        if new_state.phase != state.phase:
            if new_state.phase == "blocked":
                logger.warning(
                    "tool_progress: %s/%s -> BLOCKED: %s",
                    thread_id,
                    tool_name,
                    new_state.block_reason,
                )
            elif new_state.phase == "warned":
                logger.info(
                    "tool_progress: %s/%s -> WARNED (consecutive_problems=%d)",
                    thread_id,
                    tool_name,
                    new_state.consecutive_problems,
                )
            elif new_state.phase == "active":
                logger.info(
                    "tool_progress: %s/%s -> ACTIVE (reset after good result)",
                    thread_id,
                    tool_name,
                )
        if hint and self._inject_assessment:
            self._queue_assessment(runtime, hint)
        return result

    # ------------------------------------------------------------------
    # 상태 기계

    def _assess_and_transition(
        self,
        state: ToolPhaseState,
        meta: ToolResultMeta,
        content: str,
    ) -> tuple[ToolPhaseState, str | None]:
        """(new_state, 힌트 텍스트 또는 None)을 반환한다.

        바깥의 wrap_tool_call gate가 handler 호출 전에 이미 blocked인 상태를 가로채므로,
        이 함수는 보통 active/warned 상태에서만 도달한다. blocked 상태가 들어오면(예:
        동시 전이) 그대로 반환한다. 카운터를 부풀리지도, phase를 되돌리지도 않는다.
        """
        # 가드: blocked는 종착 상태이고 여기서 바뀌면 안 된다.
        # (정상 흐름에서는 wrap_tool_call이 handler 호출 전에 차단된 도구를 가로채므로
        # 이 분기에 도달하지 않는다. 동시성 race의 의미를 명확히 하고, 복구 가능한 오류
        # 결과가 phase를 조용히 warned로 되돌리는 일을 막기 위해 남겨 둔다.)
        if state.phase == "blocked":
            return state, None

        # 분기 전에 이 호출을 문제로 계산해 모든 종료 경로가 consecutive_problems를 일관된
        # 상태로 남기게 한다(도구가 실패했는데 0인 경우가 없다).
        new_count = state.consecutive_problems + 1

        # 복구 불가능한 stop 신호(auth, config, internal)는 즉시 차단한다.
        if not meta.recoverable_by_model and meta.recommended_next_action == "stop":
            return replace(
                state,
                phase="blocked",
                consecutive_problems=new_count,
                block_reason=_block_reason(meta),
            ), None

        # word_set은 success 결과에만 계산한다. error/partial_success는 정의상 문제라
        # Jaccard 검사에 도달하지 않으므로 O(n) 정규식이 낭비된다.
        ws = word_set(content) if meta.status == "success" else frozenset()
        is_problem = meta.status in ("error", "partial_success") or (meta.status == "success" and is_near_duplicate(ws, state.recent_word_sets, self._jaccard_threshold, self._min_words))

        if not is_problem:
            # 정상 결과: 연속 카운트를 리셋하고 active로 돌아간다.
            new_recent = (*state.recent_word_sets, ws)[-3:]
            return replace(state, consecutive_problems=0, phase="active", recent_word_sets=new_recent), None

        hint: str | None = None

        if new_count >= self._stagnation_threshold + self._warn_escalation:
            if meta.recoverable_by_model:
                # 모델이 전략을 바꿔 해결할 수 있으므로 warned를 유지하고 힌트를 다시 넣는다.
                # BLOCKED로 두면 다른 파라미터로 하는 정당한 재시도까지 막힌다.
                hint = _format_hint(meta)
                new_state = replace(state, consecutive_problems=new_count, phase="warned")
            else:
                # 재시도로는 해결할 수 없으므로 도구를 차단한다.
                reason = _block_reason(meta)
                new_state = replace(state, consecutive_problems=new_count, phase="blocked", block_reason=reason)
        elif new_count >= self._stagnation_threshold:
            hint = _format_hint(meta)
            new_state = replace(state, consecutive_problems=new_count, phase="warned")
        else:
            new_state = replace(state, consecutive_problems=new_count)

        return new_state, hint

    # ------------------------------------------------------------------
    # 대기 큐 헬퍼

    def _queue_assessment(self, runtime: Runtime, text: str) -> None:
        key = self._pending_key(runtime)
        thread_id = key[0]
        with self._lock:
            # 방금 LRU로 _phase_states에서 제거된 thread에 대해 유령 _pending 항목이
            # 생기지 않게 막는다. 그런 항목은 _phase_states만 도는 eviction 루프가 절대
            # 정리하지 못해 조용히 쌓인다.
            if thread_id not in self._phase_states:
                return
            queue = self._pending[key]
            if len(queue) < _MAX_PENDING_PER_RUN:
                queue.append(text)

    def _drain_pending(self, runtime: Runtime) -> list[str]:
        key = self._pending_key(runtime)
        with self._lock:
            return self._pending.pop(key, [])

    def _clear_stale_pending(self, runtime: Runtime) -> None:
        thread_id, current_run = self._pending_key(runtime)
        with self._lock:
            for key in list(self._pending):
                if key[0] == thread_id and key[1] != current_run:
                    del self._pending[key]

    def _reset_run_states(self, runtime: Runtime) -> None:
        """새 agent run 시작 시 해당 thread의 run 단위 도구 상태를 모두 리셋한다.

        이전 run의 상태가 다음 run으로 새지 않도록 모든 도구의 consecutive_problems 카운터와
        recent_word_sets Jaccard 창을 무조건 비운다.

        - BLOCKED/WARNED 도구는 ACTIVE로 되돌린다(근본 원인이 남아 있으면 즉시 다시 차단되고,
          모델은 이전 run의 힌트를 기억하지 못한다).
        - 이전 run에서 consecutive_problems가 0이 아니거나 recent_word_sets가 비어 있지 않은
          ACTIVE 도구도 비운다. 그러지 않으면 새 run의 첫 호출 문제 하나가 모델이 더는 보지
          못하는 오래된 context 때문에 잘못 WARNED로 넘어갈 수 있다.

        **LoopDetectionMiddleware와의 run 간 범위 차이**: 이 run 단위 리셋은 의도한 정책이지
        누락이 아니다. ``rate_limited``, ``transient`` 같은 오류는 시간에 종속적이라 사용자 턴
        사이에 원인이 해소될 수 있고, 오래된 카운터를 이어가면 지금은 성공할 호출에 대해
        BLOCKED 오탐이 난다. LoopDetectionMiddleware는 반대 입장이다. ``_history``를 run 간에
        유지한다(``before_agent``에서 다른 run의 *대기* 경고만 지운다). 호출 패턴 루프는 시간과
        무관하기 때문이다. 결과와 무관하게 같은 tool_calls를 반복하는 모델은 run이 언제
        시작됐든 똑같이 반복한다. 두 middleware는 서로 다른 실패 양상(결과 품질 대 호출 패턴)을
        막으므로 run 간 범위 정책도 의도적으로 다르다.
        """
        thread_id = self._thread_id(runtime)
        with self._lock:
            thread_tools = self._phase_states.get(thread_id)
            if thread_tools is None:
                return
            for tool_name, tool_state in list(thread_tools.items()):
                thread_tools[tool_name] = replace(
                    tool_state,
                    phase="active",
                    consecutive_problems=0,
                    block_reason=None,
                    recent_word_sets=(),
                )

    # ------------------------------------------------------------------
    # wrap_tool_call

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = str(request.tool_call.get("name", ""))
        if not tool_name or tool_name in self._exempt_tools:
            return handler(request)
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return handler(request)
        block_reason = self._get_block_reason(runtime, tool_name)
        if block_reason:
            logger.info(
                "tool_progress: %s/%s call intercepted (blocked): %s",
                self._thread_id(runtime),
                tool_name,
                block_reason,
            )
            return self._make_blocked_message(request, tool_name, block_reason)
        return self._update_state_from_result(handler(request), tool_name, runtime)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = str(request.tool_call.get("name", ""))
        if not tool_name or tool_name in self._exempt_tools:
            return await handler(request)
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return await handler(request)
        block_reason = self._get_block_reason(runtime, tool_name)
        if block_reason:
            logger.info(
                "tool_progress: %s/%s call intercepted (blocked): %s",
                self._thread_id(runtime),
                tool_name,
                block_reason,
            )
            return self._make_blocked_message(request, tool_name, block_reason)
        return self._update_state_from_result(await handler(request), tool_name, runtime)

    # ------------------------------------------------------------------
    # wrap_model_call: 대기 힌트를 꺼내 모델이 메시지를 보기 전에 주입한다

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        hints = self._drain_pending(request.runtime)
        if not hints:
            return request
        deduped = list(dict.fromkeys(hints))
        logger.debug(
            "tool_progress: injecting %d hint(s) for %s/%s",
            len(deduped),
            *self._pending_key(request.runtime),
        )
        new_messages = [
            *request.messages,
            HumanMessage(content="\n\n".join(deduped), name="progress_hint"),
        ]
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._augment_request(request))

    # ------------------------------------------------------------------
    # before_agent: 이전 run에서 남은 대기 힌트를 정리한다

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_stale_pending(runtime)
        self._reset_run_states(runtime)
        return None

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_stale_pending(runtime)
        self._reset_run_states(runtime)
        return None
