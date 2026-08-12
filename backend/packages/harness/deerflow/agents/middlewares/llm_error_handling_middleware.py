"""LLM 오류 처리 middleware — retry/backoff와 사용자에게 보여줄 fallback을 담당한다."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from deerflow.config.app_config import AppConfig
from deerflow.utils.custom_events import aemit_custom_event, emit_custom_event

logger = logging.getLogger(__name__)

_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BUSY_PATTERNS = (
    "server busy",
    "temporarily unavailable",
    "try again later",
    "please retry",
    "please try again",
    "overloaded",
    "high demand",
    "rate limit",
    "负载较高",
    "服务繁忙",
    "稍后重试",
    "请稍后重试",
)
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit",
    "payment",
    "余额不足",
    "超出限额",
    "额度不足",
    "欠费",
)
_AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "permission",
    "forbidden",
    "access denied",
    "无权",
    "未授权",
)

# provider의 burst-rate(``limit_burst_rate``) 신호. quota 제한이 아니라 *증가율* 제한이다.
# 요청 RPM이 너무 가파르게 오르면 provider가 throttle한다(예: 08:30 아침 피크에 몇 초 만에 0 -> 최대).
# 에러 메시지와 에러 ``code``/``type`` 양쪽에 대해 매칭한다.
_BURST_PATTERNS = (
    "limit_burst_rate",
    "rate increased too quickly",
    "burst rate",
    "请求速率增长过快",
    "突发速率",
)

# 예외별 retry 예산 override.
#
# 일부 transient 에러는 원리상 재시도 가능하지만 기본 예산으로 재시도하기엔 비용이 크다.
# 특히 StreamChunkTimeoutError는 upstream provider가 이미 `stream_chunk_timeout`초
# (보통 120-240초) 멈춘 뒤에 발생한다. 3회 시도를 다 돌면 사용자에게 실패를 알리기까지
# 6-12분의 침묵이 쌓인다. 그래서 retry는 정확히 한 번만 남기고(진짜 일시적인 TCP 끊김을
# 잡는 값싼 재연결) 바로 실패시킨다. 같은 버퍼된 payload는 같은 이유로 upstream에서
# 다시 실패할 가능성이 압도적으로 높기 때문이다.
#
# 키는 예외 클래스 *이름*이다(클래스가 아니다). langchain-openai 같은 선택적 의존성에
# import 시점 결합을 만들지 않기 위해서다. 값은 추가 retry 횟수가 아니라 절대 최대 시도
# 횟수이므로, 2는 "첫 시도 1 + retry 1"을 뜻한다(CR에서 요청한 "retry 한 번 유지" 동작).
_RETRY_BUDGET_OVERRIDES: dict[str, int] = {
    "StreamChunkTimeoutError": 2,
}

# reason별 retry 예산 override. 위의 예외별 override와 함께 적용되며 가장 빡빡한 값이
# 이긴다(어느 쪽도 다른 쪽을 느슨하게 만들지 않는다). 사용자가 설정한
# ``retry_max_attempts``가 여전히 전체 상한이다.
#
# burst-rate(``limit_burst_rate``) 429에 의도적으로 좁은 예산을 준다. burst 상황에서
# 재시도하면 throttle 대상인 요청 증가율 자체를 더 밀어올리므로, retry는 최대 한 번만
# 하고(backoff는 더 길게) 부하를 덜어낸다. 키는 ``_classify_error``의 reason이다.
_REASON_RETRY_BUDGETS: dict[str, int] = {
    "burst_rate": 2,
}

# 모델이 응답 도중 멈춰서 upstream의 stream-chunk watchdog가 발동했음을 뜻하는 예외
# 클래스 이름들. 흔한 원인이 긴 tool-call 직렬화가 upstream stream을 막는 것이라,
# 일반적인 "일시적으로 사용 불가" 문구보다 더 구체적인 안내가 필요하다. 사용자에게 줄 수
# 있는 가장 실행 가능한 조언은 "기다렸다 재시도"가 아니라 "출력을 짧게 하거나 나눠 달라"다.
# 일반적인 연결 끊김(httpx RemoteProtocolError / ReadError)은 의도적으로 제외한다.
# 정상 payload에서도 일시적 네트워크 문제로 흔히 발생하며, 그때 "작업을 나누라"는 안내는
# 오히려 오해를 준다.
_STREAM_DROP_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "StreamChunkTimeoutError",
    }
)


# 프로세스 전역 LLM 호출 동시성 상한. limiter 하나를 모든
# ``LLMErrorHandlingMiddleware`` 인스턴스와 모든 호출 경로가 공유한다: lead agent(메인
# event loop), subagent(subagents/executor.py의 격리된 상주 loop), ``asyncio.run`` 테스트,
# sync graph 경로. provider의 burst-rate(``limit_burst_rate``) 제한은 요청률의 *기울기*에
# 걸리므로, 상한은 프로세스 전체의 in-flight 호출 합계를 묶어야 한다. loop별 상한
# (asyncio.Semaphore가 주는 것)은 subagent fan-out이 두 번째 loop에서 도는 순간 무력해진다.
#
# 아래 설계가 지키는 정합성 불변식:
#   * 무손실 waiter 인계: waiter에게 넘긴 permit은 dequeue 시점에 그 waiter 앞으로
#     *예약*된다(``granted=True``). waiter가 깨어나기 전에 취소되면 예약된 permit은 다음
#     waiter에게 다시 넘기거나 반납한다. 따라서 dequeue 후 재획득 전 구간에서 취소가
#     일어나도 용량을 놀린 채 다음 waiter를 방치하는 일이 없다.
#   * 시작 시점 고정 상한: 상한은 첫 middleware 생성 시점(``_apply_configured_cap``)에
#     한 번만 결정되고 이후 고정된다. 이후 ``__init__``은 더 새로운 ``AppConfig``
#     스냅샷을 들고 있든 더 오래된 것을 들고 있든 상한을 건드리지 않는다. 이로써
#     의사 generation 경로가 통째로 사라진다. 런타임에 상한이 바뀌지 않으므로 상한을
#     낮출 때 대기 중인 waiter에게 초과 permit이 넘어가 ``in_flight``가 옛 상한에 고정되는
#     일도, 오래된 config가 새 config보다 늦게 생성되어 더 높은 상한을 복원하는 생성 순서
#     race도 없다. 시도 단위 호출자는 acquire/release만 한다. 상한 변경은 gateway
#     재시작이 필요하다(``LlmCallConfig.max_concurrent_calls`` 참고).


class _AsyncWaiter:
    """permit 인계를 기다리며 대기 중인 async 호출자.

    ``granted``는 이 waiter 앞으로 permit이 예약되는 바로 그 순간 (limiter lock 아래서)
    ``True``가 된다. ``release``가 반납된 permit을 넘겨줄 때, 또는 취소되는 다른 waiter가
    자기 예약 permit을 넘겨줄 때다. 예약은 dequeue와 원자적이라
    ``granted is True  <=>  not in _async_waiters`` 불변식이 항상 성립한다. 일단 granted가
    되면 permit은 이미 ``_in_flight``에 계상되어 있으므로 waiter는 깨어나 반환만 하면 된다.
    따라서 취소된 waiter는 ``granted``만 보고 인계 의무가 있는지(granted) 단순 등록 해제인지
    (아직 granted 아님) 판단할 수 있다.
    """

    __slots__ = ("loop", "event", "granted")

    def __init__(self, loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
        self.loop = loop
        self.event = event
        self.granted = False


class _ProcessWideLimiter:
    """event loop와 sync/async wrapper를 가로질러 공유되는 in-flight 호출 limiter.

    ``asyncio.Semaphore``는 처음 사용한 event loop에 묶이고 다른 loop에서 획득하면 예외를
    던진다. 그래서 서로 다른 loop에서 도는 lead agent와 subagent 호출을 함께 제한할 수 없고
    sync graph 경로도 못 다룬다. 이 limiter는 loop에 묶이지 않는 ``threading`` 프리미티브로
    만들어져 모든 호출 경로가 하나의 in-flight 카운터와 하나의 상한을 공유한다.

    상한은 **불변**이다. 첫 middleware ``__init__``에서 ``_apply_configured_cap``이 한 번
    설정한 뒤 절대 바뀌지 않는다. 런타임에 상한이 변하지 않으므로 상한을 낮췄을 때
    ``in_flight``가 빠질 때까지 대기 중인 waiter를 계속 들여보내는 race도, 오래된 스냅샷이
    나중에 생성되어 더 높은 상한을 복원하는 race도 없다. 시도 단위 호출자
    (``acquire_sync``/``acquire_async``/``release``)는 상한을 건드리지 않는다. permit은
    ``finally``에서 반납하고, permit이 예약된 뒤 취소된 async waiter는 그 예약을 다음
    waiter에게 넘긴다. 따라서 용량이 새지 않고 취소가 뒤의 waiter를 방치하지 않는다.
    """

    def __init__(self, limit: int) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._in_flight = 0
        self._limit = max(0, limit)
        # 용량을 기다리는 async 호출자의 FIFO. 각 waiter는 자기 호출자의 loop에 있으므로
        # release/인계는 call_soon_threadsafe로 loop를 건너 깨운다. 그래야 wakeup이
        # 올바른 loop에서 실행된다.
        self._async_waiters: deque[_AsyncWaiter] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def acquire_sync(self) -> None:
        """permit이 생길 때까지 호출 thread를 막았다가 하나를 가져간다."""
        with self._cond:
            while not self._try_acquire_locked():
                self._cond.wait()

    def release(self) -> None:
        """permit 하나를 반납한다. 대기 중인 waiter가 있으면 그쪽으로 넘긴다.

        async waiter가 대기 중이면 반납되는 permit이 그쪽으로 *이전*되고(소유권만 이동하고
        ``_in_flight``는 그대로) event를 set해서 permit을 이미 가진 상태로 깨어나게 한다.
        그렇지 않으면 permit은 free pool로 돌아가고(``_in_flight -= 1``) sync waiter 하나에게
        알려 다음 ``_try_acquire_locked`` 재확인에서 가져가게 한다.
        """
        with self._cond:
            if self._async_waiters:
                waiter = self._async_waiters.popleft()
                waiter.granted = True
                if not self._wake_locked(waiter):
                    # 소유 loop가 닫혔다. 이전된 permit이 붕 뜨므로 다음 waiter에게
                    # 넘기거나 반납한다.
                    self._handoff_granted_permit_locked()
                return
            if self._in_flight > 0:
                self._in_flight -= 1
            self._cond.notify()

    async def acquire_async(self) -> None:
        """event loop를 막지 않고 permit을 획득한다.

        여유 용량이 있으면 즉시 하나를 가져간다. 없으면 ``asyncio.Event``에 대기한다.
        ``release``나 상한 상향이 permit을 우리에게 이전하고(``granted=True``) event를 set한다.
        취소될 때 이미 permit이 예약되어 있었다면 다음 waiter에게 넘기거나 반납해서 예약이
        사라지지 않게 한다. 아직 대기열에만 있었다면(granted 이전) 등록만 해제한다. 예약된
        permit이 없으니 반납할 것도 없다.
        """
        loop = asyncio.get_running_loop()
        while True:
            waiter = _AsyncWaiter(loop=loop, event=asyncio.Event())
            with self._cond:
                if self._try_acquire_locked():
                    return
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "LLM call parking on process-wide limiter (in_flight=%d, limit=%d, queued=%d)",
                        self._in_flight,
                        self._limit,
                        len(self._async_waiters) + 1,
                    )
                self._async_waiters.append(waiter)
            try:
                await waiter.event.wait()
            except asyncio.CancelledError:
                with self._cond:
                    if waiter.granted:
                        # permit이 예약됐지만 깨어나기 전에 취소된다. 예약이 붕 뜨지
                        # 않도록 다음 waiter에게 넘기거나 반납한다.
                        self._handoff_granted_permit_locked()
                    else:
                        # 아직 대기열에 있고 granted된 적 없다(granted는 lock 아래
                        # dequeue될 때만 설정된다). 등록만 해제한다.
                        self._async_waiters.remove(waiter)
                raise
            return  # 깨어남 => granted => permit 보유(이미 _in_flight에 계상됨)

    def _try_acquire_locked(self) -> bool:
        if self._in_flight < self._limit:
            self._in_flight += 1
            return True
        return False

    def _handoff_granted_permit_locked(self) -> None:
        """이미 예약된 permit을 다음 대기 waiter에게 이전하거나 반납한다.

        permit을 예약받은 waiter가 깨어나기 전에 취소됐거나 예약 대상의 loop가 죽었을 때
        쓴다. permit은 이미 ``_in_flight``에 계상되어 있으므로, 이전하면 계상이 유지되고
        (소유권만 다음 waiter로 이동) 반납하면 pool로 돌아간다. 어느 쪽이든 ``_in_flight``는
        정확하게 유지되고 예약은 사라지지 않는다.
        """
        while self._async_waiters:
            waiter = self._async_waiters.popleft()
            waiter.granted = True
            if self._wake_locked(waiter):
                return  # 소유권 이전 완료. _in_flight는 그대로
            # 죽은 loop다. 다음 waiter를 시도한다
        # 받아갈 async waiter가 없다. permit을 반납하고 sync waiter를 깨운다.
        if self._in_flight > 0:
            self._in_flight -= 1
        self._cond.notify()

    def _wake_locked(self, waiter: _AsyncWaiter) -> bool:
        """waiter의 loop에 ``event.set``을 예약한다. loop가 죽었으면 False를 반환한다."""
        try:
            waiter.loop.call_soon_threadsafe(waiter.event.set)
            return True
        except RuntimeError:
            return False  # 소유 loop가 닫혀 wakeup이 도달할 수 없다


_LIMITER_LOCK = threading.Lock()
_PROCESS_LIMITER: _ProcessWideLimiter | None = None

# 프로세스 전역 상한이 결정됐는지 여부. 상한은 시작 시점에만 정해진다. 첫
# ``LLMErrorHandlingMiddleware`` ``__init__``이 이를 결정하고(양수면 limiter 생성, 아니면
# ``None``으로 두어 비활성) 이후의 모든 ``__init__``은 no-op이다. 그 인스턴스의
# ``AppConfig`` 스냅샷이 첫 번째보다 새롭든 오래됐든 마찬가지다. 여기가 상한의 유일한
# 소유자이며, 시도 단위 호출자는 acquire/release만 한다.
_CAP_RESOLVED: bool = False


def _get_process_limiter() -> _ProcessWideLimiter | None:
    """프로세스 전역 LLM 호출 limiter를 반환한다. 상한이 비활성이거나 첫 middleware 생성
    전이면 ``None``이다.

    시도 단위 호출자는 acquire/release 용도로만 쓴다. 상한은 절대 바뀌지 않는다.
    "상한 비활성"의 유일한 판정은 ``limiter is None``이다. 인스턴스의 설정값으로 호출마다
    단락하면 ``max_concurrent_calls=0``인 (reload된) 나중 인스턴스가 프로세스 도중 상한을
    조용히 없앨 수 있는데, 그게 바로 시작 시점 고정 설계가 제거하려는 hot-reload 혼란이다.
    """
    return _PROCESS_LIMITER


def _apply_configured_cap(limit: int) -> None:
    """첫 middleware ``__init__``에서 프로세스 전역 상한을 결정한다.

    시작 시점 전용이다. 맨 처음 호출이 이기고 상한을 고정한다. ``limit``이 양수면 그 값으로
    limiter를 만들고, ``limit <= 0``이면 상한을 비활성으로 확정한다(limiter는 ``None``으로
    남고 호출자는 ``limiter is None``에서 단락한다). 이후 호출은 ``AppConfig`` 스냅샷이
    새롭든 오래됐든, 상한을 올리든 내리든 전부 무시되므로 런타임에 상한이 바뀔 수 없다.
    변경하려면 gateway를 재시작해야 한다.
    """
    global _PROCESS_LIMITER, _CAP_RESOLVED
    if _CAP_RESOLVED:
        return  # 첫 생성에서 이미 상한이 고정됐다. 이 인스턴스는 no-op이다
    with _LIMITER_LOCK:
        if _CAP_RESOLVED:
            return
        _CAP_RESOLVED = True
        if limit > 0:
            _PROCESS_LIMITER = _ProcessWideLimiter(limit)


class LLMErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """일시적인 LLM 오류를 재시도하고 사용자에게 자연스러운 assistant 메시지를 보여준다."""

    retry_max_attempts: int = 3
    retry_base_delay_ms: int = 1000
    retry_cap_delay_ms: int = 8000
    # burst-rate(limit_burst_rate) 429에만 쓰는 더 긴 backoff 기준값. 단 한 번의 burst
    # retry가 throttle 구간이 지나간 뒤에 떨어지도록 한다.
    burst_retry_base_delay_ms: int = 5000
    # 동시 in-flight LLM 호출의 프로세스 전역 상한. 0(기본값)이면 상한을 끄므로 기존 배포는
    # 동작이 바뀌지 않는다. 양수로 두면 전체 동시성을 묶어 provider의
    # burst-rate(limit_burst_rate) 급증을 완만하게 만든다. _get_process_limiter 참고.
    max_concurrent_llm_calls: int = 0

    def __init__(self, *, app_config: AppConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.circuit_failure_threshold = app_config.circuit_breaker.failure_threshold
        self.circuit_recovery_timeout_sec = app_config.circuit_breaker.recovery_timeout_sec

        # retry / backoff / 동시성 설정값은 모두 config.yaml의 ``llm_call`` 섹션에서 온다.
        # 위의 클래스 기본값을 덮어쓰므로 운영자가 코드 수정 없이 조정할 수 있다.
        llm_call = app_config.llm_call
        self.retry_max_attempts = llm_call.retry_max_attempts
        self.retry_base_delay_ms = llm_call.retry_base_delay_ms
        self.retry_cap_delay_ms = llm_call.retry_cap_delay_ms
        self.burst_retry_base_delay_ms = llm_call.burst_retry_base_delay_ms
        self.max_concurrent_llm_calls = llm_call.max_concurrent_calls

        # 프로세스 전역 상한을 결정한다(시작 시점 전용: 프로세스의 첫 ``__init__``이 이겨
        # 상한을 고정하고, 이후 인스턴스는 config가 새롭든 오래됐든 no-op이다). 시도 단위
        # 호출자는 acquire/release만 하므로 런타임에 상한이 바뀔 수 없고, 현재 상한을 넘겨
        # waiter를 들여보내는 하향 조정 race나 config 신선도 race도 없다.
        _apply_configured_cap(self.max_concurrent_llm_calls)

        # Circuit Breaker 상태
        self._circuit_lock = threading.Lock()
        self._circuit_failure_count = 0
        self._circuit_open_until = 0.0
        self._circuit_state = "closed"
        self._circuit_probe_in_flight = False

    def _max_attempts_for(self, exc: BaseException, reason: str = "transient") -> int:
        """이 예외에 적용할 실효 최대 시도 횟수를 반환한다.

        사용자가 설정한 ``retry_max_attempts``가 상한이고, 예외별
        (``_RETRY_BUDGET_OVERRIDES``, 클래스 이름 기준)과 reason별
        (``_REASON_RETRY_BUDGETS``, ``_classify_error``의 reason 기준) override는 이를
        *좁히기만* 한다. 가장 빡빡한 값이 이기므로, 운영자가 전역 상한을 올려도 burst-rate
        429는 전용 예산보다 많이 시도하지 않는다.
        """
        candidates = [self.retry_max_attempts]
        class_override = _RETRY_BUDGET_OVERRIDES.get(type(exc).__name__)
        if class_override is not None:
            candidates.append(class_override)
        reason_override = _REASON_RETRY_BUDGETS.get(reason)
        if reason_override is not None:
            candidates.append(reason_override)
        return min(candidates)

    def _check_circuit(self) -> bool:
        """circuit이 OPEN이면(즉시 실패) True, 아니면 False를 반환한다."""
        with self._circuit_lock:
            now = time.time()

            if self._circuit_state == "open":
                if now < self._circuit_open_until:
                    return True
                self._circuit_state = "half_open"
                self._circuit_probe_in_flight = False

            if self._circuit_state == "half_open":
                if self._circuit_probe_in_flight:
                    return True
                self._circuit_probe_in_flight = True
                return False

            return False

    def _record_success(self) -> None:
        with self._circuit_lock:
            if self._circuit_state != "closed" or self._circuit_failure_count > 0:
                logger.info("Circuit breaker reset (Closed). LLM service recovered.")
            self._circuit_failure_count = 0
            self._circuit_open_until = 0.0
            self._circuit_state = "closed"
            self._circuit_probe_in_flight = False

    def _record_failure(self) -> None:
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                self._circuit_state = "open"
                self._circuit_probe_in_flight = False
                logger.error(
                    "Circuit breaker probe failed (Open). Will probe again after %ds.",
                    self.circuit_recovery_timeout_sec,
                )
                return

            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self.circuit_failure_threshold:
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                if self._circuit_state != "open":
                    self._circuit_state = "open"
                    self._circuit_probe_in_flight = False
                    logger.error(
                        "Circuit breaker tripped (Open). Threshold reached (%d). Will probe after %ds.",
                        self.circuit_failure_threshold,
                        self.circuit_recovery_timeout_sec,
                    )

    def _release_half_open_probe(self) -> None:
        """실패로 기록하지 않고 진행 중인 half-open probe를 해제한다.

        성공/실패로 분류되지 않은 무언가가 probe를 소비했을 때(GraphBubbleUp 제어 흐름
        시그널이나 재시도 불가 에러) 쓴다. 그래야 circuit이 영원히 즉시 실패하지 않고 다음
        probe를 받아들인다.
        """
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_probe_in_flight = False

    def _classify_error(self, exc: BaseException) -> tuple[bool, str]:
        detail = _extract_error_detail(exc)
        lowered = detail.lower()
        error_code = _extract_error_code(exc)
        status_code = _extract_status_code(exc)

        if _matches_any(lowered, _QUOTA_PATTERNS) or _matches_any(str(error_code).lower(), _QUOTA_PATTERNS):
            return False, "quota"
        if _matches_any(lowered, _AUTH_PATTERNS):
            return False, "auth"
        # burst-rate(limit_burst_rate) 429는 재시도 가능하지만 별도 정책이 필요하다.
        # 좁은 retry 예산과 더 긴 backoff 기준값을 쓴다(_REASON_RETRY_BUDGETS /
        # _build_retry_delay_ms 참고). 일반적인 429->transient 매핑보다 먼저 판정해서
        # 평범한 transient 에러와 뭉뚱그려지지 않게 한다.
        if _matches_any(lowered, _BURST_PATTERNS) or _matches_any(str(error_code).lower(), _BURST_PATTERNS):
            return True, "burst_rate"

        exc_name = exc.__class__.__name__
        if exc_name in {
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ReadError",  # httpx.ReadError: stream 도중 연결 끊김
            "RemoteProtocolError",  # httpx: 서버가 예기치 않게 연결을 닫음
            "StreamChunkTimeoutError",  # langchain-openai: chunk 간격이 stream_chunk_timeout 초과
        }:
            return True, "transient"
        # upstream이 빈 ``generations`` 리스트와 함께 ``200 OK``를 반환할 때가 있다
        # (Volces "coding" / ark.cn-beijing.volces.com에서 관측). 그러면
        # ``langchain_core.language_models.chat_models.ainvoke``가
        # ``llm_result.generations[0][0].message``에서
        # ``IndexError: list index out of range``로 죽는다. 이건 클라이언트 버그가 아니라
        # 일시적인 upstream payload 결함이므로, 실행 전체를 실패시키지 않고 다른 transient
        # provider 실패와 같은 retry/backoff 경로로 보낸다.
        if isinstance(exc, IndexError):
            return True, "transient"
        if status_code in _RETRIABLE_STATUS_CODES:
            return True, "transient"
        if _matches_any(lowered, _BUSY_PATTERNS):
            return True, "busy"

        return False, "generic"

    def _bounded_model_call_sync(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """프로세스 전역 동시성 상한 아래에서 sync 모델 시도 한 번을 실행한다.

        limiter는 retry loop 전체가 아니라 *한 번의* 시도만 감싼다. 그래서 backoff sleep
        동안 슬롯이 다른 호출자에게 풀린다. ``limiter is None``(시작 시점에 상한 비활성)이면
        그대로 통과시키고, ``None``이 아니면 항상 limiter를 거친다. 상한은 첫 ``__init__``에서
        고정되므로 ``max_concurrent_llm_calls``가 0인 나중 인스턴스가 이를 조용히 없앨 수 없다.
        permit은 ``finally``에서 반환/예외 어느 경로로 빠져나가도 반납되므로 handler가 예외를
        던져도 슬롯이 새지 않는다.
        """
        limiter = _get_process_limiter()
        if limiter is None:
            return handler(request)
        limiter.acquire_sync()
        try:
            return handler(request)
        finally:
            limiter.release()

    async def _bounded_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """프로세스 전역 동시성 상한 아래에서 async 모델 시도 한 번을 실행한다.

        limiter는 retry loop 전체가 아니라 *한 번의* 시도만 감싼다. 그래서 backoff sleep
        동안 슬롯이 다른 호출자에게 풀린다. 대기 중인 요청이 아니라 in-flight 요청을 묶는
        것이다. ``limiter is None``(시작 시점에 상한 비활성)이면 그대로 통과시키고,
        ``None``이 아니면 항상 limiter를 거친다(상한은 첫 ``__init__``에서 고정). permit은
        ``finally``에서 반환/예외/취소 어느 경로로 빠져나가도 반납되고, 대기 중 취소된
        경우는 ``acquire_async``가 따로 정리하므로 용량이 새지 않는다.
        """
        limiter = _get_process_limiter()
        if limiter is None:
            return await handler(request)
        await limiter.acquire_async()
        try:
            return await handler(request)
        finally:
            limiter.release()

    def _build_retry_delay_ms(self, prev_delay_ms: int | None, exc: BaseException, reason: str = "transient") -> int:
        """decorrelated jitter로 다음 retry 지연(ms)을 계산한다.

        provider가 명시한 ``Retry-After``는 jitter 없이 그대로 따른다. 서버가 언제 다시
        오라고 정확히 알려준 것이고, burst-rate 429에서는 계산된 지연보다 이쪽이 훨씬 낫다.
        그 외에는 AWS 스타일 "decorrelated jitter"를 적용한다:
        ``delay = random(base, min(cap, max(base, seed * 3)))``. 여기서 ``seed``는 직전
        지연이며, 첫 retry(``prev_delay_ms is None``)에서는 reason별 기준값이다. 추첨 *전에*
        구간을 cap으로 자르므로(추첨 후가 아니라) 분포가 cap에 쌓이지 않고 cap까지 균등하게
        유지된다. ``reason="burst_rate"``면 일반 기준값보다 긴
        ``burst_retry_base_delay_ms``를 써서 단 한 번의 burst retry가 throttle 구간이 지난
        뒤에 떨어지게 한다.

        첫 retry의 seed를 항상 일반 기준값이 아니라 *reason별* 기준값으로 잡는 이유는,
        처음이자 마지막인 burst retry가 퇴화하지 않게 하기 위해서다. 일반 기준값(1000ms)을
        쓰면 burst 구간이 ``randint(5000, max(5000, 1000*3)) = randint(5000, 5000)``으로
        무너져 동시에 실패한 burst들이 전부 같은 5초 지점에 다시 정렬된다. 5000ms를 seed로
        쓰면 기본값 기준 ``randint(5000, min(8000, 15000)) = randint(5000, 8000)``이 되어
        함께 실패한 fleet이 구간 전체로 흩어진다.

        결정적 지수 backoff(``base * 2^(attempt-1)``)는 동시에 재시도하는 모두를 같은 backoff
        지점에 다시 정렬시킨다. fleet 전체가 한꺼번에 실패하면(예: 아침 피크의 provider
        burst-rate 제한) 그 동기화된 retry 폭풍이 지금 피하려던 바로 그 제한을 다시 발동시킨다.
        decorrelated jitter는 retry를 무작위 구간에 흩어 같은 박자로 재차 몰리지 않게 한다.
        """
        retry_after = _extract_retry_after_ms(exc)
        if retry_after is not None:
            return retry_after
        base = self.burst_retry_base_delay_ms if reason == "burst_rate" else self.retry_base_delay_ms
        cap = self.retry_cap_delay_ms
        seed = base if prev_delay_ms is None else prev_delay_ms
        # 추첨 *전에* 구간을 cap으로 자른다. 그래야 jitter가 cap에 몰리지 않고
        # [base, min(cap, seed*3)]에 균등하게 퍼진다. 기본값에서는 seed*3(=15000)이
        # cap(=8000)보다 훨씬 크므로, randint(base, seed*3) 후 min(delay, cap)을 하면
        # 추첨의 약 70%가 정확히 cap에 몰려 jitter가 흩으려던 fleet이 다시 뭉친다.
        high = min(cap, max(base, seed * 3))
        if high < base:
            return cap  # base가 cap을 넘는 잘못된 설정이다. cap이 이긴다
        return random.randint(base, high)

    def _build_retry_message(
        self,
        attempt: int,
        wait_ms: int,
        reason: str,
        *,
        max_attempts: int,
    ) -> str:
        seconds = max(1, round(wait_ms / 1000))
        reason_text = {
            "busy": "provider is busy",
            "burst_rate": "provider is throttling request burst rate",
        }.get(reason, "provider request failed temporarily")
        # ``max_attempts``는 설정된 상한이 아니라 이 호출의 *실효* 예산이다
        # (``_max_attempts_for``에서 온다). burst-rate 호출은 2회로 제한되므로
        # ``retry_max_attempts``가 기본값 3이어도 메시지는 ``1/3``이 아니라 ``1/2``여야 한다.
        # 아니면 UI가 일어나지 않을 retry를 약속하게 된다.
        return f"LLM request retry {attempt}/{max_attempts}: {reason_text}. Retrying in {seconds}s."

    def _build_circuit_breaker_message(self) -> str:
        return "The configured LLM provider is currently unavailable due to continuous failures. Circuit breaker is engaged to protect the system. Please wait a moment before trying again."

    def _build_error_fallback_message(
        self,
        content: str,
        *,
        error_type: str,
        reason: str,
        detail: str,
    ) -> AIMessage:
        return AIMessage(
            content=content,
            additional_kwargs={
                "deerflow_error_fallback": True,
                "error_type": error_type,
                "error_reason": reason,
                "error_detail": detail,
            },
        )

    def _build_user_message(self, exc: BaseException, reason: str) -> str:
        detail = _extract_error_detail(exc)
        if reason == "quota":
            return "The configured LLM provider rejected the request because the account is out of quota, billing is unavailable, or usage is restricted. Please fix the provider account and try again."
        if reason == "auth":
            return "The configured LLM provider rejected the request because authentication or access is invalid. Please check the provider credentials and try again."
        if reason == "burst_rate":
            return "The configured LLM provider is temporarily throttling requests because the request rate increased too quickly (burst-rate limit). Please wait a moment and try again."
        if reason in {"busy", "transient"}:
            # stream 끊김 실패(chunk 간격 timeout, 상대가 닫은 연결, raw read error)는
            # 거의 항상 지나치게 큰 tool-call payload 하나가 원인이다. 모델이 JSON 인자를
            # 직렬화하는 데 너무 오래 걸려 upstream provider가 버퍼링했고 stream 간격이
            # `stream_chunk_timeout`을 넘긴 것이다. 이 원인을 따로 알려주면 사용자가 같은
            # prompt를 무작정 재시도하는 대신 요청을 나누거나 줄일 수 있다.
            if type(exc).__name__ in _STREAM_DROP_EXCEPTIONS:
                return (
                    "The model's streaming response was interrupted before it could "
                    "finish. This usually happens when a single response or tool call "
                    "is very large — please ask the assistant to split the work into "
                    "smaller steps, or shorten the requested output, and try again."
                )
            return "The configured LLM provider is temporarily unavailable after multiple retries. Please wait a moment and continue the conversation."
        return f"LLM request failed: {detail}"

    def _build_user_fallback_message(self, exc: BaseException, reason: str) -> AIMessage:
        return self._build_error_fallback_message(
            self._build_user_message(exc, reason),
            error_type=type(exc).__name__,
            reason=reason,
            detail=_extract_error_detail(exc),
        )

    def _build_retry_event(
        self,
        attempt: int,
        wait_ms: int,
        reason: str,
        *,
        max_attempts: int,
    ) -> dict[str, Any]:
        return {
            "type": "llm_retry",
            "attempt": attempt,
            # 설정된 상한이 아니라 이 호출의 실효 예산이다(burst-rate == 2). frontend가
            # 이 값과 아래 ``message``를 함께 렌더링하므로 둘 다 실제로 도는 loop를
            # 설명해야 한다.
            "max_attempts": max_attempts,
            "wait_ms": wait_ms,
            "reason": reason,
            "message": self._build_retry_message(attempt, wait_ms, reason, max_attempts=max_attempts),
        }

    def _emit_retry_event(
        self,
        attempt: int,
        wait_ms: int,
        reason: str,
        *,
        max_attempts: int,
    ) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            emit_custom_event(
                self._build_retry_event(attempt, wait_ms, reason, max_attempts=max_attempts),
                writer=writer,
            )
        except GraphBubbleUp:
            raise
        except Exception:
            logger.debug("Failed to emit llm_retry event", exc_info=True)

    async def _aemit_retry_event(
        self,
        attempt: int,
        wait_ms: int,
        reason: str,
        *,
        max_attempts: int,
    ) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            await aemit_custom_event(
                self._build_retry_event(attempt, wait_ms, reason, max_attempts=max_attempts),
                writer=writer,
            )
        except GraphBubbleUp:
            raise
        except Exception:
            logger.debug("Failed to emit async llm_retry event", exc_info=True)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        if self._check_circuit():
            return self._build_error_fallback_message(
                self._build_circuit_breaker_message(),
                error_type="CircuitBreakerOpen",
                reason="circuit_open",
                detail="LLM circuit breaker is open",
            )

        attempt = 1
        prev_delay_ms: int | None = None
        while True:
            try:
                response = self._bounded_model_call_sync(request, handler)
                self._record_success()
                return response
            except GraphBubbleUp:
                # LangGraph 제어 흐름 시그널(interrupt/pause/resume)은 그대로 전파한다.
                self._release_half_open_probe()
                raise
            except Exception as exc:
                retriable, reason = self._classify_error(exc)
                max_attempts = self._max_attempts_for(exc, reason)
                if retriable and attempt < max_attempts:
                    wait_ms = self._build_retry_delay_ms(prev_delay_ms, exc, reason)
                    prev_delay_ms = wait_ms
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    self._emit_retry_event(attempt, wait_ms, reason, max_attempts=max_attempts)
                    time.sleep(wait_ms / 1000)
                    attempt += 1
                    continue
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                if retriable and reason != "burst_rate":
                    self._record_failure()
                else:
                    # 재시도 불가이거나 burst_rate("provider 다운"이 아니라 일시적인
                    # 증가율 throttle)인 경우다. 실패로 기록하지 않고 half-open probe만
                    # 해제해서 circuit이 열려 복구 구간 동안 모든 호출을 즉시 실패시키는
                    # 일을 막는다. #4290이 막으려던 자초한 장애가 바로 그것이다.
                    self._release_half_open_probe()
                return self._build_user_fallback_message(exc, reason)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        if self._check_circuit():
            return self._build_error_fallback_message(
                self._build_circuit_breaker_message(),
                error_type="CircuitBreakerOpen",
                reason="circuit_open",
                detail="LLM circuit breaker is open",
            )

        attempt = 1
        prev_delay_ms: int | None = None
        while True:
            try:
                response = await self._bounded_model_call(request, handler)
                self._record_success()
                return response
            except GraphBubbleUp:
                # LangGraph 제어 흐름 시그널(interrupt/pause/resume)은 그대로 전파한다.
                self._release_half_open_probe()
                raise
            except Exception as exc:
                retriable, reason = self._classify_error(exc)
                max_attempts = self._max_attempts_for(exc, reason)
                if retriable and attempt < max_attempts:
                    wait_ms = self._build_retry_delay_ms(prev_delay_ms, exc, reason)
                    prev_delay_ms = wait_ms
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    await self._aemit_retry_event(attempt, wait_ms, reason, max_attempts=max_attempts)
                    await asyncio.sleep(wait_ms / 1000)
                    attempt += 1
                    continue
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                if retriable and reason != "burst_rate":
                    self._record_failure()
                else:
                    # 재시도 불가이거나 burst_rate("provider 다운"이 아니라 일시적인
                    # 증가율 throttle)인 경우다. 실패로 기록하지 않고 half-open probe만
                    # 해제해서 circuit이 열려 복구 구간 동안 모든 호출을 즉시 실패시키는
                    # 일을 막는다. #4290이 막으려던 자초한 장애가 바로 그것이다.
                    self._release_half_open_probe()
                return self._build_user_fallback_message(exc, reason)


def _matches_any(detail: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in detail for pattern in patterns)


def _extract_error_code(exc: BaseException) -> Any:
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return value

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value
    return None


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _extract_retry_after_ms(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = None
    header_name = ""
    for key in ("retry-after-ms", "Retry-After-Ms", "retry-after", "Retry-After"):
        header_name = key
        if hasattr(headers, "get"):
            raw = headers.get(key)
        if raw:
            break
    if not raw:
        return None

    try:
        multiplier = 1 if "ms" in header_name.lower() else 1000
        return max(0, int(float(raw) * multiplier))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            delta = target.timestamp() - time.time()
            return max(0, int(delta * 1000))
        except (TypeError, ValueError, OverflowError):
            return None


def _extract_error_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return exc.__class__.__name__
