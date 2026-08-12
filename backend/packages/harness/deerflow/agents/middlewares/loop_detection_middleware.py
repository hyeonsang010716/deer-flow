"""반복되는 tool call loop를 감지하고 끊는 middleware.

P0 안전장치다. 에이전트가 같은 인자로 같은 도구를 계속 호출하다 recursion limit에 걸려
실행이 죽는 것을 막는다.

감지 전략:
  1. 모델 응답마다 tool call들(name + args)을 해싱한다.
  2. 최근 해시를 sliding window로 추적한다.
  3. 같은 해시가 warn_threshold 이상 나오면 현재 thread/run에 "같은 일을 반복하고 있으니
     마무리하라"는 경고를 큐에 넣는다. 이 경고는 **다음 모델 호출 시점에**
     (``wrap_model_call``에서) 메시지 목록 끝에 ``HumanMessage``로 주입된다. 직전
     AIMessage(tool_calls)에 대한 모든 ToolMessage 응답 *뒤*다.
  4. hard_limit 이상 나오면 응답에서 tool_calls를 전부 제거해 에이전트가 최종 텍스트
     답변을 내도록 강제한다.

경고를 ``after_model``이 아니라 ``wrap_model_call``에서 주입하는 이유:

  ``after_model``은 모델이 ``tool_calls``를 담을 수 있는 ``AIMessage``를 낸 직후에
  실행된다. tools 노드는 아직 돌지 않았으므로 대응하는 ``ToolMessage``가 히스토리에
  없다. 여기서 메시지를 추가하면 assistant의 tool_calls와 그 응답 *사이*에 끼어든다.
  OpenAI/Moonshot의 검증기는 assistant의 tool_calls 바로 뒤에 tool message가 오기를
  요구하므로 다음 요청을 ``"tool_call_ids did not have response messages"``로 거부한다.
  Anthropic도 중간에 낀 ``SystemMessage``를 허용하지 않는다. 경고를
  ``wrap_model_call``까지 미루면 이전 ToolMessage가 이미 요청 메시지 목록에 모두 들어
  있고 경고는 맨 끝에 붙는다. 쌍이 깨지지 않고 ``AIMessage`` 의미도 건드리지 않는다.

큐에 쌓인 경고는 의도적으로 일회성이다. 다음 모델 요청이 경고를 소비하기 전에 실행이
끝나면 ``after_agent``가 같은 thread의 다음 호출로 넘기지 않고 버린다. 설정된 안전
한계에 도달하면 hard-stop 경로가 여전히 종료를 강제한다.

Stop-reason 노출(#3875 Phase 2):
  token-budget guard와 마찬가지로 loop hard stop은 예외를 던지지 않는다. ``tool_calls``를
  제거해서 에이전트 loop가 최종 답변과 함께 자연스럽게 끝나게 한다. 호출자(subagent
  executor)가 loop 때문에 잘린 완료와 깨끗한 완료를 구분할 수 있도록, hard stop을 유발한
  실행을 ``_stop_reason``에 기록하고 :meth:`consume_stop_reason`으로 노출한다. executor는
  token-budget guard의 reason과 함께 이를 수집하므로 loop로 잘린 실행은
  ``completed + loop_capped``로 드러나고, lead와 ledger는 결과 텍스트를 파싱하지 않고도
  잘렸음을 알 수 있다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares._bounded_dict import BoundedDict

if TYPE_CHECKING:
    from deerflow.config.loop_detection_config import LoopDetectionConfig

logger = logging.getLogger(__name__)

# 기본값. 생성자에서 덮어쓸 수 있다
_DEFAULT_WARN_THRESHOLD = 3  # 동일 호출 3회 후 경고 주입
_DEFAULT_HARD_LIMIT = 5  # 동일 호출 5회 후 강제 중단
_DEFAULT_WINDOW_SIZE = 20  # 최근 N개의 tool call 추적
_DEFAULT_MAX_TRACKED_THREADS = 100  # LRU 축출 한계
_DEFAULT_TOOL_FREQ_WARN = 30  # 같은 도구 종류 30회 호출 후 경고
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50  # 같은 도구 종류 50회 호출 후 강제 중단
_MAX_PENDING_WARNINGS_PER_RUN = 4


def _normalize_tool_call_args(raw_args: object) -> tuple[dict, str | None]:
    """tool call 인자를 dict와 선택적 fallback 키로 정규화한다.

    일부 provider는 ``args``를 dict가 아니라 JSON 문자열로 직렬화한다. loop 감지가 죽지
    않도록 방어적으로 파싱하되, dict가 아닌 payload에 대해서는 안정적인 fallback 키를
    유지한다.
    """
    if isinstance(raw_args, dict):
        return raw_args, None

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}, raw_args

        if isinstance(parsed, dict):
            return parsed, None
        return {}, json.dumps(parsed, sort_keys=True, default=str)

    if raw_args is None:
        return {}, None

    return {}, json.dumps(raw_args, sort_keys=True, default=str)


def _stable_tool_key(name: str, args: dict, fallback_key: str | None) -> str:
    """노이즈에 과적합하지 않으면서 핵심 인자로 안정적인 키를 만든다."""
    if name == "read_file" and fallback_key is None:
        path = args.get("path") or ""
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        bucket_size = 200
        try:
            start_line = int(start_line) if start_line is not None else 1
        except (TypeError, ValueError):
            start_line = 1
        try:
            end_line = int(end_line) if end_line is not None else start_line
        except (TypeError, ValueError):
            end_line = start_line

        start_line, end_line = sorted((start_line, end_line))
        bucket_start = max(start_line, 1)
        bucket_end = max(end_line, 1)
        bucket_start = (bucket_start - 1) // bucket_size
        bucket_end = (bucket_end - 1) // bucket_size
        return f"{path}:{bucket_start}-{bucket_end}"

    # write_file / str_replace는 내용에 민감하다. 반복 작업 중 같은 경로가 서로 다른
    # payload로 갱신될 수 있다. 핵심 필드(path)만 쓰면 서로 다른 호출이 하나로 뭉뚱그려지므로
    # 오탐을 줄이려고 전체 인자를 해싱한다.
    if name in {"write_file", "str_replace"}:
        if fallback_key is not None:
            return fallback_key
        return json.dumps(args, sort_keys=True, default=str)

    salient_fields = ("path", "url", "query", "command", "pattern", "glob", "cmd")
    stable_args = {field: args[field] for field in salient_fields if args.get(field) is not None}
    if stable_args:
        return json.dumps(stable_args, sort_keys=True, default=str)

    if fallback_key is not None:
        return fallback_key

    return json.dumps(args, sort_keys=True, default=str)


def _hash_tool_calls(tool_calls: list[dict]) -> str:
    """tool call 집합(name + 안정 키)의 결정적 해시를 만든다.

    순서에 무관해야 한다. 같은 tool call 다중집합이면 입력 순서와 상관없이 항상 같은
    해시가 나와야 한다.
    """
    # 각 tool call을 안정적인 (name, key) 구조로 정규화한다.
    normalized: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args, fallback_key = _normalize_tool_call_args(tc.get("args", {}))
        key = _stable_tool_key(name, args, fallback_key)

        normalized.append(f"{name}:{key}")

    # 같은 호출 다중집합의 순열이 동일한 순서가 되도록 정렬한다.
    normalized.sort()
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


_WARNING_MSG = "[LOOP DETECTED] You are repeating the same tool calls. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."

_TOOL_FREQ_WARNING_MSG = (
    "[LOOP DETECTED] You have called {tool_name} {count} times without producing a final answer. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."
)

_HARD_STOP_MSG = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."

_TOOL_FREQ_HARD_STOP_MSG = "[FORCED STOP] Tool {tool_name} called {count} times — exceeded the per-tool safety limit. Producing final answer with results collected so far."


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    """반복되는 tool call loop를 감지하고 끊는다.

    임계값 파라미터는 상위의 :class:`LoopDetectionConfig`가 검증한다. Pydantic 검증을
    거치도록 :meth:`from_config`로 생성한다.

    Args:
        warn_threshold: 경고 메시지를 주입하기까지 허용하는 동일 tool call 집합 횟수.
            기본값 3.
        hard_limit: tool_calls를 전부 제거하기까지 허용하는 동일 tool call 집합 횟수.
            기본값 5.
        window_size: 호출 추적용 sliding window 크기. 기본값 20.
        max_tracked_threads: LRU 축출 전까지 추적할 최대 thread 수. 기본값 100.
        tool_freq_warn: ``_tool_freq_window`` 구간 안에서 빈도 경고를 주입하기까지 허용하는
            동일 도구 종류 호출 수. 해시 기반 감지가 놓치는 파일 간 read loop를 잡는다.
            기본값 30(window 50 기준).
        tool_freq_hard_limit: ``_tool_freq_window`` 구간 안에서 강제 중단하기까지 허용하는
            동일 도구 종류 호출 수. 기본값 50(window 50 기준).
        tool_freq_overrides: 도구 이름을 키로 하는 도구별 빈도 임계값 override. 값은 해당
            도구의 ``tool_freq_warn`` / ``tool_freq_hard_limit``를 대체하는
            ``(warn, hard_limit)`` 튜플이다. 여기 없는 도구는 전역 임계값을 쓴다. 다른 모든
            도구의 보호를 약화시키지 않으면서 의도적으로 자주 쓰는 도구(예: 배치 파이프라인의
            ``bash``)의 한계만 올릴 때 유용하다. 기본값 ``None``(override 없음).
    """

    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
        tool_freq_warn: int = _DEFAULT_TOOL_FREQ_WARN,
        tool_freq_hard_limit: int = _DEFAULT_TOOL_FREQ_HARD_LIMIT,
        tool_freq_overrides: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self.tool_freq_warn = tool_freq_warn
        self.tool_freq_hard_limit = tool_freq_hard_limit
        self._tool_freq_overrides: dict[str, tuple[int, int]] = tool_freq_overrides or {}
        # Layer 2의 window 빈도 카운트는 deque 길이를 넘을 수 없다. 따라서 deque는 비교
        # 대상인 가장 큰 hard limit 이상이어야 한다. 아니면 hard-stop 분기가 죽은 코드가
        # 된다. Layer 1의 ``window_size``를 재사용하면 안 된다(무관하며 기본값이 빈도
        # 임계값보다 작다. 예: 20 < hard 50). 빈도 window는 실제로 쓰이는 가장 큰 hard
        # limit(전역 + 모든 도구별 override)에 맞춘다. 그래야 짧고 굵은 burst가 실제로
        # 도달할 수 있으면서 띄엄띄엄한 호출은 window 밖으로 빠져 감쇠한다. warn 임계값은
        # 의도적으로 제외한다. 정상 설정은 warn <= hard이고(hard에 맞추면 함께 커버된다),
        # warn > hard인 잘못된 설정은 어차피 hard-stop이 먼저 걸린다. 도달 불가능한 warn은
        # 무해하므로 window를 부풀려서는 안 된다.
        self._tool_freq_window = max(
            self.window_size,
            self.tool_freq_hard_limit,
            *(hard for _, hard in self._tool_freq_overrides.values()),
        )
        self._lock = threading.Lock()
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        self._warned: dict[str, set[str]] = defaultdict(set)
        # 도구 종류별 window 빈도: thread별 최근 도구 이름을 ``window_size``로 잘라
        # 카운트가 단조 증가하지 않고 감쇠하게 한다(기존의 단조 증가 정수 ``_tool_freq``를
        # 대체한다).
        self._tool_name_history: defaultdict[str, deque[str]] = defaultdict(deque)
        # deque를 그대로 반영하는 thread별 Counter. tool call마다 window 전체를 훑지 않고
        # freq_count를 O(1)로 얻는다. 도구별 override 하나가 크면(예: bash: {hard_limit: 1000})
        # window가 전역적으로 부풀어 모든 도구가 호출마다 1000번씩 훑게 된다. Counter는
        # append에서 증가하고 popleft에서 감소한다.
        self._tool_name_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
        # Layer 2에서 이미 경고한 도구 이름의 thread별 집합. 빈도 경고를 이후 모든 호출마다가
        # 아니라 한 번만 큐에 넣기 위해서다. window 카운트가 warn 임계값 아래로 감쇠하면
        # 해당 이름을 지운다. 해시 계층의 ``_warned`` 정리와 같은 방식이다.
        self._tool_freq_warned: dict[str, set[str]] = defaultdict(set)
        # 다음 모델 호출에 주입할 경고의 thread/run별 큐. ``after_model``(감지)이 채우고
        # ``wrap_model_call``(주입)이 비운다. 모듈 docstring 참고.
        self._pending_warnings: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._pending_warning_touch_order: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._max_pending_warning_keys = max(1, self.max_tracked_threads * 2)
        # hard-stop이 발동할 때 설정되는 stop reason(#3875 Phase 2). ``TokenBudgetMiddleware``와
        # 동일하게 run_id를 키로 쓰고 크기를 제한한다. lead agent의 middleware 인스턴스는 여러
        # 실행에 걸쳐 오래 살아남으므로, 상한이 없으면 loop에 걸린 lead 실행마다 항목이 쌓인다.
        # subagent executor가 실행 종료 후 소비할 수 있도록
        # ``after_agent``/``_clear_current_run_pending_warnings``에서 의도적으로 지우지 않는다.
        # ``reset()``은 여전히 지운다.
        self._stop_reason: BoundedDict[str, str] = BoundedDict(1000)

    @classmethod
    def from_config(cls, config: LoopDetectionConfig) -> LoopDetectionMiddleware:
        """Pydantic 검증을 거친 config를 그대로 신뢰해 인스턴스를 만든다."""
        return cls(
            warn_threshold=config.warn_threshold,
            hard_limit=config.hard_limit,
            window_size=config.window_size,
            max_tracked_threads=config.max_tracked_threads,
            tool_freq_warn=config.tool_freq_warn,
            tool_freq_hard_limit=config.tool_freq_hard_limit,
            tool_freq_overrides={name: (o.warn, o.hard_limit) for name, o in config.tool_freq_overrides.items()},
        )

    def _get_thread_id(self, runtime: Runtime) -> str:
        """thread별 추적을 위해 runtime context에서 thread_id를 꺼낸다."""
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id:
            return str(thread_id)
        return "default"

    def _get_run_id(self, runtime: Runtime) -> str:
        """실행별 경고 범위를 위해 runtime context에서 run_id를 꺼낸다.

        truthy 여부가 아니라 키의 존재 여부로 판단한다. ``SubagentExecutor``는 truthy 검사
        없이 ``context["run_id"] = self.run_id``를 무조건 설정한다. 그래서 embedded/TUI에서
        디스패치된 subagent는(``AGENTS.md``의 embedded ``DeerFlowClient`` 설명대로 ``run_id``가
        할당되지 않는다) 정당하게 ``run_id=None``을 담은 context로 실행된다. 키가 *있고* 값이
        None인 것이다. executor는 나중에 원시 속성으로 ``consume_stop_reason(self.run_id)``를
        호출해 stop reason을 읽으므로, 키가 있으면 (``None``을 포함해) 그 값을 그대로
        반환해야 한다. 키가 없는 경우와 구분되지 않는 공용 fallback으로 뭉뚱그리면 안 된다.
        예전의 truthy 검사(``if run_id:``)는 "있지만 None/falsy"와 "없음"을 모두 리터럴
        ``"default"``로 뭉갰다. 그래서 실제 ``run_id=None``의 hard-stop이 여기서는
        ``"default"``로 기록되고 executor는 ``None``으로 조회해 ``loop_capped`` stop reason이
        조용히 사라졌다. ``TokenBudgetMiddleware._get_run_id``와 같은 방식이다.
        """
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict) and "run_id" in ctx:
            return ctx["run_id"]
        # embedded client 실행 간 충돌을 막기 위해 runtime 객체 ID로 대체한다
        return str(id(runtime))

    def consume_stop_reason(self, run_id: str | None) -> str | None:
        """이 실행에서 hard-stop이 설정한 stop reason을 꺼내 반환한다.

        실행 중 반복 tool-call loop가 hard stop을 유발했으면 ``"loop_capped"``를 반환한다.
        hard stop은 예외를 던지지 않고 ``tool_calls``만 제거하므로 실행 자체는 강제된 최종
        답변과 함께 완료된다. subagent executor가 실행 종료 후 이를 호출해서, loop로 잘린
        완료가 깨끗한 ``completed``처럼 보이지 않고 ``stop_reason=loop_capped``를 달고 lead에
        전달되게 한다. ``TokenBudgetMiddleware.consume_stop_reason``과 같은 방식이며, pop
        방식이라 인스턴스를 재사용해도 dict가 쌓이지 않는다.
        """
        with self._lock:
            return self._stop_reason.pop(run_id, None)

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        """현재 thread/run의 pending-warning 키를 반환한다."""
        return self._get_thread_id(runtime), self._get_run_id(runtime)

    def _evict_if_needed(self) -> None:
        """한계를 넘으면 가장 오래 쓰이지 않은 thread를 축출한다.

        self._lock을 잡은 상태에서 호출해야 한다.
        """
        while len(self._history) > self.max_tracked_threads:
            evicted_id, _ = self._history.popitem(last=False)
            self._warned.pop(evicted_id, None)
            self._tool_name_history.pop(evicted_id, None)
            self._tool_name_counter.pop(evicted_id, None)
            self._tool_freq_warned.pop(evicted_id, None)
            for key in list(self._pending_warnings):
                if key[0] == evicted_id:
                    self._drop_pending_warning_key_locked(key)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    def _drop_pending_warning_key_locked(self, key: tuple[str, str]) -> None:
        """thread/run 키 하나에 대한 pending-warning 상태를 전부 버린다.

        self._lock을 잡은 상태에서 호출해야 한다.
        """
        self._pending_warnings.pop(key, None)
        self._pending_warning_touch_order.pop(key, None)

    def _touch_pending_warning_key_locked(self, key: tuple[str, str]) -> None:
        """pending-warning 키를 최근 사용으로 표시한다.

        self._lock을 잡은 상태에서 호출해야 한다.
        """
        self._pending_warning_touch_order[key] = None
        self._pending_warning_touch_order.move_to_end(key)

    def _prune_pending_warning_state_locked(self, protected_key: tuple[str, str]) -> None:
        """비정상 종료나 동시 실행에 걸쳐 pending-warning 상태 크기를 제한한다.

        self._lock을 잡은 상태에서 호출해야 한다.
        """
        overflow = len(self._pending_warning_touch_order) - self._max_pending_warning_keys
        if overflow <= 0:
            return

        candidates = [key for key in self._pending_warning_touch_order if key != protected_key]
        for key in candidates[:overflow]:
            self._drop_pending_warning_key_locked(key)

    def _queue_pending_warning(self, runtime: Runtime, warning: str) -> None:
        """현재 thread/run에 일회성 경고 하나를 상한을 지키며 큐에 넣는다."""
        pending_key = self._pending_key(runtime)
        with self._lock:
            warnings = self._pending_warnings[pending_key]
            if warning not in warnings:
                warnings.append(warning)
            if len(warnings) > _MAX_PENDING_WARNINGS_PER_RUN:
                del warnings[: len(warnings) - _MAX_PENDING_WARNINGS_PER_RUN]
            self._touch_pending_warning_key_locked(pending_key)
            self._prune_pending_warning_state_locked(protected_key=pending_key)

    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """tool call을 추적하고 loop 여부를 확인한다.

        감지 계층은 둘이다.
          1. **해시 기반**(기존): 동일한 tool call 집합을 잡는다.
          2. **빈도 기반**(신규): 같은 *도구 종류*가 인자만 바꿔가며 여러 번 호출되는 경우를
             잡는다(예: 서로 다른 40개 파일에 대한 ``read_file``).

        Returns:
            (경고 메시지 또는 None, hard stop 여부)
        """
        messages = state.get("messages", [])
        if not messages:
            return None, False

        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        thread_id = self._get_thread_id(runtime)
        call_hash = _hash_tool_calls(tool_calls)

        with self._lock:
            # 항목을 갱신하거나 생성한다(LRU를 위해 끝으로 이동)
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                self._history[thread_id] = []
                self._evict_if_needed()

            history = self._history[thread_id]
            history.append(call_hash)
            if len(history) > self.window_size:
                history[:] = history[-self.window_size :]

            warned_hashes = self._warned.get(thread_id)
            if warned_hashes is not None:
                warned_hashes.intersection_update(history)
                if not warned_hashes:
                    self._warned.pop(thread_id, None)

            count = history.count(call_hash)
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            # --- Layer 1: 해시 기반(동일 호출 집합) ---
            if count >= self.hard_limit:
                logger.error(
                    "Loop hard limit reached — forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                return _HARD_STOP_MSG, True

            if count >= self.warn_threshold:
                warned = self._warned[thread_id]
                if call_hash not in warned:
                    warned.add(call_hash)
                    logger.warning(
                        "Repetitive tool calls detected — injecting warning",
                        extra={
                            "thread_id": thread_id,
                            "call_hash": call_hash,
                            "count": count,
                            "tools": tool_names,
                        },
                    )
                    return _WARNING_MSG, False

            # --- Layer 2: 도구 종류별 빈도(window 기반) ---
            tool_name_history = self._tool_name_history[thread_id]
            name_counter = self._tool_name_counter[thread_id]
            for tc in tool_calls:
                name = tc.get("name", "")
                if not name:
                    continue
                # window 기반 집계: 이름을 추가하고 빈도 window(가장 큰 임계값 이상)로
                # 자른다. 그래야 짧고 굵은 burst에서는 카운트가 warn/hard 한계에 도달하고
                # 띄엄띄엄한 호출은 감쇠한다. 도구별 override가 window를 전역적으로
                # 부풀려도 미러링된 Counter 덕분에 freq_count는 O(1)이다.
                tool_name_history.append(name)
                name_counter[name] += 1
                while len(tool_name_history) > self._tool_freq_window:
                    old = tool_name_history.popleft()
                    c = name_counter[old] - 1
                    if c <= 0:
                        del name_counter[old]
                    else:
                        name_counter[old] = c
                freq_count = name_counter.get(name, 0)

                if name in self._tool_freq_overrides:
                    eff_warn, eff_hard = self._tool_freq_overrides[name]
                else:
                    eff_warn, eff_hard = self.tool_freq_warn, self.tool_freq_hard_limit

                if freq_count >= eff_hard:
                    logger.error(
                        "Tool frequency hard limit reached — forcing stop",
                        extra={
                            "thread_id": thread_id,
                            "tool_name": name,
                            "count": freq_count,
                        },
                    )
                    return _TOOL_FREQ_HARD_STOP_MSG.format(tool_name=name, count=freq_count), True

                if freq_count >= eff_warn:
                    freq_warned = self._tool_freq_warned[thread_id]
                    if name not in freq_warned:
                        freq_warned.add(name)
                        logger.warning(
                            "Tool frequency warning — too many calls to same tool type",
                            extra={
                                "thread_id": thread_id,
                                "tool_name": name,
                                "count": freq_count,
                            },
                        )
                        return _TOOL_FREQ_WARNING_MSG.format(tool_name=name, count=freq_count), False
                else:
                    # window 카운트가 warn 임계값 아래로 감쇠했다. 이 도구가 나중에 다시
                    # 몰리면 경고할 수 있게 한다.
                    self._tool_freq_warned[thread_id].discard(name)

        return None, False

    @staticmethod
    def _append_text(content: str | list | None, text: str) -> str | list:
        """AIMessage content에 *text*를 덧붙인다. str, list, None을 모두 처리한다.

        content가 content block 리스트인 경우(예: Anthropic thinking 모드)에는 리스트에
        문자열을 이어붙이면 ``TypeError``가 나므로 ``{"type": "text", ...}`` 블록을
        새로 추가한다.
        """
        if content is None:
            return text
        if isinstance(content, list):
            return [*content, {"type": "text", "text": f"\n\n{text}"}]
        if isinstance(content, str):
            return content + f"\n\n{text}"
        # fallback: 예상 밖의 타입은 TypeError를 피하려고 str로 강제 변환한다
        return str(content) + f"\n\n{text}"

    @staticmethod
    def _build_hard_stop_update(last_msg, content: str | list) -> dict:
        """강제 중단 메시지가 평범한 assistant 텍스트로 직렬화되도록 tool-call 메타데이터를 지운다."""
        update = {
            "tool_calls": [],
            "content": content,
        }

        additional_kwargs = dict(getattr(last_msg, "additional_kwargs", {}) or {})
        for key in ("tool_calls", "function_call"):
            additional_kwargs.pop(key, None)
        update["additional_kwargs"] = additional_kwargs

        response_metadata = deepcopy(getattr(last_msg, "response_metadata", {}) or {})
        if response_metadata.get("finish_reason") == "tool_calls":
            response_metadata["finish_reason"] = "stop"
        update["response_metadata"] = response_metadata

        return update

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        warning, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            # 실행 종료 후 executor가 ``stop_reason=loop_capped``를 드러낼 수 있도록 stop
            # reason을 기록한다(#3875 Phase 2). hard stop은 예외를 던지지 않고 tool_calls만
            # 제거해 강제된 최종 답변으로 실행을 끝내므로, 이게 없으면 호출자는 깨끗한
            # ``completed``로 본다. ``consume_stop_reason`` 참고. lead agent의 middleware
            # 인스턴스는 동시에 도는 여러 Gateway thread가 공유하므로,
            # ``TokenBudgetMiddleware``와 동일하게 bounded dict 쓰기를 lock 아래서 한다.
            run_id = self._get_run_id(runtime)
            with self._lock:
                self._stop_reason[run_id] = "loop_capped"
            # lead worker가 이 middleware 인스턴스 참조 없이도 읽을 수 있도록
            # runtime.context에도 쓴다(#4176).
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict):
                ctx["stop_reason"] = "loop_capped"
            # 텍스트 출력을 강제하려고 마지막 AIMessage에서 tool_calls를 제거한다.
            # tool_calls가 제거되면 그 AIMessage는 더 이상 대응하는 ToolMessage 응답을
            # 요구하지 않으므로, 여기서 제자리 수정해도 OpenAI/Moonshot의 쌍 검증기에
            # 안전하다.
            messages = state.get("messages", [])
            last_msg = messages[-1]
            content = self._append_text(last_msg.content, warning or _HARD_STOP_MSG)
            stripped_msg = last_msg.model_copy(update=self._build_hard_stop_update(last_msg, content))
            return {"messages": [stripped_msg]}

        if warning:
            # 주입은 다음 모델 호출로 미룬다. 여기서 AIMessage(tool_calls=...)를 고치면
            # 안 된다(프레임워크의 말을 모델 입에 넣는 셈이라 MemoryMiddleware 같은 후단
            # 소비자를 오염시킨다). 별도의 non-tool 메시지를 끼워 넣어도 안 된다(tools
            # 노드가 아직 ToolMessage 응답을 만들지 않아 OpenAI/Moonshot의 tool-call 쌍이
            # 깨진다). 경고는 아래 ``wrap_model_call``에서 전달한다.
            self._queue_pending_warning(runtime, warning)
            return None

        return None

    def _clear_other_run_pending_warnings(self, runtime: Runtime) -> None:
        """이 thread의 이전 실행에 남은 오래된 pending warning을 버린다."""
        thread_id, current_run_id = self._pending_key(runtime)
        with self._lock:
            for key in list(self._pending_warnings):
                if key[0] == thread_id and key[1] != current_run_id:
                    self._drop_pending_warning_key_locked(key)

    def _clear_current_run_pending_warnings(self, runtime: Runtime) -> None:
        """현재 thread/run이 소유한 pending warning을 버린다."""
        pending_key = self._pending_key(runtime)
        with self._lock:
            self._drop_pending_warning_key_locked(pending_key)

    @staticmethod
    def _format_warning_message(warnings: list[str]) -> str:
        """대기 중인 경고들을 하나의 prompt 메시지로 합친다."""
        deduped = list(dict.fromkeys(warnings))
        return "\n\n".join(deduped)

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_other_run_pending_warnings(runtime)
        return None

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_other_run_pending_warnings(runtime)
        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_current_run_pending_warnings(runtime)
        return None

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_current_run_pending_warnings(runtime)
        return None

    def _drain_pending_warnings(self, runtime: Runtime) -> list[str]:
        """*runtime*의 thread/run에 큐잉된 경고를 모두 꺼내 반환한다."""
        pending_key = self._pending_key(runtime)
        with self._lock:
            warnings = self._pending_warnings.pop(pending_key, [])
            self._pending_warning_touch_order.pop(pending_key, None)
        return warnings

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        """큐에 있는 loop 경고를 나가는 메시지 목록에 덧붙인다.

        경고는 직전 AIMessage(tool_calls)에 대한 ToolMessage 응답을 포함해 기존의 모든
        메시지 *뒤*에 놓인다. 그래야 OpenAI/Moonshot의
        ``assistant tool_calls -> tool_messages`` 쌍이 유지되고, HumanMessage를 쓰므로
        Anthropic의 중간 SystemMessage 제약도 피하며, 기존 AIMessage를 건드리지 않는다.
        """
        warnings = self._drain_pending_warnings(request.runtime)
        if not warnings:
            return request
        new_messages = [
            *request.messages,
            HumanMessage(content=self._format_warning_message(warnings), name="loop_warning"),
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

    def reset(self, thread_id: str | None = None) -> None:
        """추적 상태를 비운다. thread_id를 주면 해당 thread만 비운다."""
        with self._lock:
            if thread_id:
                self._history.pop(thread_id, None)
                self._warned.pop(thread_id, None)
                self._tool_name_history.pop(thread_id, None)
                self._tool_name_counter.pop(thread_id, None)
                self._tool_freq_warned.pop(thread_id, None)
                for key in list(self._pending_warnings):
                    if key[0] == thread_id:
                        self._drop_pending_warning_key_locked(key)
            else:
                self._history.clear()
                self._warned.clear()
                self._tool_name_history.clear()
                self._tool_name_counter.clear()
                self._tool_freq_warned.clear()
                self._pending_warnings.clear()
                self._pending_warning_touch_order.clear()
                self._stop_reason.clear()
