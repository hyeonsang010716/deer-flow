"""debounce를 적용한 memory 갱신 queue.

대화 context를 모아 설정된 debounce 시간 후에 처리하며, 같은
``(thread_id, user_id, agent_name)`` 키의 context 여러 개는 하나의 갱신으로 합친다.

queue는 프로세스 로컬 in-memory 리스트와 debounce :class:`~threading.Timer`로만
구성된다. 프로세스 종료 시점에 남아 있던 항목은 유실되며, graceful shutdown에서는
best-effort인 :meth:`MemoryUpdateQueue.flush_sync` 드레인이 이를 완화한다.
memory 갱신은 best-effort다. 실패하거나 유실된 갱신은 다음 대화 턴에 다시 들어온다
(middleware가 매 주기 전체 대화를 넘기고, updater의 watermark는 실패 시 전진하지
않는다). 그래서 persistence 계층 없이도 in-memory queue만으로 현실적인 graceful
배포 상황을 감당한다.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..config import DeerMemConfig

if TYPE_CHECKING:
    from .updater import MemoryUpdater

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    """backpressure 상황에서 signal 없는 갱신이 거부될 때 발생한다.

    signal이 감지된 갱신은 중요한 memory를 버리지 않도록 항상 받아들이며,
    ``queue_max_depth``에 도달하면 signal 없는 갱신만 거부한다. 호출자는 이 예외를
    잡아 동작을 낮출 수 있다(예: 비상 경로에서 동기 쓰기로 fallback).
    """


def queue_key(
    thread_id: str,
    user_id: str | None,
    agent_name: str | None,
) -> tuple[str, str | None, str | None]:
    """memory 갱신 대상의 debounce 식별자를 반환한다."""
    return (thread_id, user_id, agent_name)


@dataclass
class ConversationContext:
    """memory 갱신을 위해 처리할 대화의 context."""

    thread_id: str
    messages: list[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    signals: frozenset[str] = field(default_factory=frozenset)
    # 비상(summarization) flush는 updater의 인덱스 watermark를 우회한다. 이때 실리는
    # 부분 집합은 "삭제 전에 추출"하는 일회성 스냅샷이라, 그 길이를 그대로 쓰면 대화
    # watermark가 뒤로 밀린다. 또한 이런 context는 같은 키의 대기 중 일반 갱신을
    # 대체하지 않고 공존한다. flush가 일반 갱신의 미추출 꼬리를 버리지 않게 하기
    # 위해서다. ``_enqueue_locked``의 match key와 backpressure 처리를 참고한다.
    bypass_watermark: bool = False


class MemoryUpdateQueue:
    """debounce를 적용한 memory 갱신 queue.

    대화 context를 모아 설정된 debounce 시간 후에 처리한다. debounce 구간 안에 들어온
    여러 대화는 하나로 묶어 처리한다.
    """

    def __init__(self, config: DeerMemConfig, updater: MemoryUpdater):
        """주입받은 config와 updater로 memory 갱신 queue를 초기화한다."""
        self._config = config
        self._updater = updater
        self._items: list[ConversationContext] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._processing = False
        # 현재 ``_process_queue``를 실행 중인 스레드(유휴 상태면 None). ``flush_sync``는
        # 실행 중인 worker를 join한다. worker가 이미 queue에서 꺼내 처리 중인 context가
        # 있는데도 "완료"라고 잘못 보고하고 종료 시 유실되는 일을 막기 위해서다.
        # ``flush_sync``의 (1)단계를 참고한다.
        self._processing_thread: threading.Thread | None = None
        self._reprocess_pending = False

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
        signals: frozenset[str] | None = None,
    ) -> None:
        """대화를 갱신 queue에 추가한다.

        Args:
            thread_id: thread ID.
            messages: 대화 메시지.
            agent_name: 지정하면 agent별로 memory를 저장하고, None이면 전역 memory를 쓴다.
            user_id: enqueue 시점에 확보한 user ID. ContextVar는 raw 스레드를 넘지
                못하므로 threading.Timer 경계를 넘어 살아남도록 ConversationContext에
                저장한다.
            trace_id: enqueue 시점에 확보한 요청 trace id. 이후 Timer 스레드가 memory
                LLM tracing metadata에 붙일 수 있게 한다.
            signals: 대화에서 감지한 signal 종류(correction / reinforcement /
                preference / ...)로 추출 힌트에 쓴다. signal이 하나라도 있으면
                backpressure 상황에서도 받아들인다.
        """
        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                trace_id=trace_id,
                signals=frozenset(signals) if signals else frozenset(),
                bypass_watermark=False,
            )
            self._reset_timer()

        logger.info("Memory update queued for thread %s, queue size: %d", thread_id, len(self._items))

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
        signals: frozenset[str] | None = None,
    ) -> None:
        """대화를 추가하고 백그라운드에서 즉시 처리를 시작한다."""
        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                trace_id=trace_id,
                signals=frozenset(signals) if signals else frozenset(),
                bypass_watermark=True,
            )
            self._schedule_timer(0)

        logger.info("Memory update queued for immediate processing on thread %s, queue size: %d", thread_id, len(self._items))

    def _enqueue_locked(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None,
        user_id: str | None,
        trace_id: str | None,
        signals: frozenset[str],
        bypass_watermark: bool = False,
    ) -> ConversationContext:
        key = queue_key(thread_id, user_id, agent_name)
        # 비상(bypass) 갱신과 일반 갱신은 공존한다. match key에 ``bypass_watermark``가
        # 들어가므로 summarization flush(bypass=True)가 같은 (thread, user, agent)의
        # 대기 중 일반 갱신을 대체하지 않는다. 대체하면 일반 갱신의 미추출 꼬리가
        # 사라지는데, 사용자가 대화를 멈추면 다음 턴에 다시 들어오지 않을 수 있다.
        # 두 갱신은 각각 독립적으로 처리한다.
        existing = next(
            (c for c in self._items if queue_key(c.thread_id, c.user_id, c.agent_name) == key and c.bypass_watermark == bypass_watermark),
            None,
        )
        # backpressure. 깊이가 상한에 닿으면 signal 없는 새 일반 항목을 거부한다.
        # 같은 키의 갱신은 병합되므로 깊이가 늘지 않고, signal이 있는 항목과
        # 비상(bypass) flush는 항상 받아들인다. signal은 중요한 memory를 담고, 비상
        # 경로는 summarization으로 곧 삭제될 메시지를 담는다. 둘 다 다음 턴에 다시
        # 들어오지 않으므로 부하 상황에서 버리면 지연이 아니라 데이터 유실이 된다.
        max_depth = self._config.queue_max_depth
        if max_depth > 0 and not bypass_watermark and not signals and existing is None and len(self._items) >= max_depth:
            raise QueueFull(f"memory update queue is full (depth {len(self._items)} >= {max_depth}); non-signal update for thread {thread_id} rejected")

        # signal은 합집합으로 병합한다. 이 키의 어느 갱신에서든 본 signal은 남는다.
        merged_signals = signals | (existing.signals if existing is not None else frozenset())
        context = ConversationContext(
            thread_id=thread_id,
            messages=messages,
            agent_name=agent_name,
            user_id=user_id,
            trace_id=trace_id,
            signals=merged_signals,
            bypass_watermark=bypass_watermark,
        )
        if existing is not None:
            self._items = [c for c in self._items if not (queue_key(c.thread_id, c.user_id, c.agent_name) == key and c.bypass_watermark == bypass_watermark)]
        self._items.append(context)
        return context

    def _reset_timer(self) -> None:
        """debounce timer를 재설정한다."""
        config = self._config
        self._schedule_timer(config.debounce_seconds)

        logger.debug("Memory update timer set for %ss", config.debounce_seconds)

    def _schedule_timer(self, delay_seconds: float) -> None:
        """지정한 지연 후에 queue 처리를 예약한다."""
        # 기존 timer가 있으면 취소한다
        if self._timer is not None:
            self._timer.cancel()

        self._timer = threading.Timer(
            delay_seconds,
            self._process_queue,
        )
        self._timer.daemon = True
        self._timer.start()

    def _process_queue(self, *, skip_inter_item_delay: bool = False) -> None:
        """queue에 쌓인 모든 대화 context를 처리한다.

        Args:
            skip_inter_item_delay: 설정하면 항목 사이의 rate-limit용 ``time.sleep``을
                건너뛴다. 제한된 타임아웃과 경쟁하는 종료 드레인 경로
                (:meth:`flush_sync`)에서 항목 사이 대기로 예산을 낭비하지 않기 위한
                옵션이다.
        """
        with self._lock:
            if self._processing:
                # 다른 worker가 이미 queue를 비우는 중이다. 바쁜 동안 0초 Timer를
                # 반복 예약하는 tight spin 대신 재실행 한 번만 예약해 둔다. 실행 중인
                # worker가 finally에서 이 플래그를 확인해 남은 작업이 있으면 한 번
                # 다시 예약한다.
                self._reprocess_pending = True
                return

            if not self._items:
                return

            self._processing = True
            self._processing_thread = threading.current_thread()
            contexts_to_process = self._items
            self._items = []
            self._timer = None

        logger.info("Processing %d queued memory updates", len(contexts_to_process))

        succeeded = 0
        failed = 0
        try:
            for context in contexts_to_process:
                try:
                    logger.info("Updating memory for thread %s (trace_id=%s)", context.thread_id, context.trace_id)
                    success = self._updater.update_memory(
                        messages=context.messages,
                        thread_id=context.thread_id,
                        agent_name=context.agent_name,
                        signals=context.signals,
                        user_id=context.user_id,
                        trace_id=context.trace_id,
                        bypass_watermark=context.bypass_watermark,
                    )
                    if success:
                        succeeded += 1
                        logger.info("Memory updated successfully for thread %s (trace_id=%s)", context.thread_id, context.trace_id)
                    else:
                        failed += 1
                        logger.warning("Memory update skipped/failed for thread %s (trace_id=%s)", context.thread_id, context.trace_id)
                except Exception as e:
                    failed += 1
                    logger.error("Error updating memory for thread %s (trace_id=%s): %s", context.thread_id, context.trace_id, e)

                # rate limit을 피하려고 갱신 사이에 짧게 쉰다. 제한된 타임아웃과
                # 경쟁하는 종료 드레인 경로에서는 그 예산을 항목 사이 대기가 아니라
                # LLM 호출에 써야 하므로 건너뛴다.
                if not skip_inter_item_delay and len(contexts_to_process) > 1:
                    time.sleep(0.5)
        finally:
            # 요약 카운트는 "비웠다"(queue가 빔)와 "저장했다"(모든 추출이 영속화됨)를
            # 구분해 준다. 위에서 항목별 ``update_memory`` 실패를 삼키므로, 이 로그가
            # 없으면 memory 누락을 디버깅하는 운영자에게 정상 경로의 "Processing N"
            # 줄만 보인다.
            if succeeded or failed:
                logger.info("Memory update batch done: %d succeeded, %d failed", succeeded, failed)
            with self._lock:
                self._processing = False
                self._processing_thread = None
                # 재예약은 lock 안에서 한다. ``_schedule_timer``는 ``self._timer``를
                # 읽고 취소하고 다시 대입하는 비원자적 동작이고, 동시에 실행되는
                # ``add``의 ``_reset_timer``도 lock 안에서 같은 필드를 건드린다. lock을
                # 쥐면 ``add``에 대해 재예약이 원자적이 된다. ``_schedule_timer``는
                # ``Timer.start()``만 호출하고 동기적으로 lock을 잡지 않으므로 교착은
                # 발생하지 않는다.
                if self._reprocess_pending:
                    self._reprocess_pending = False
                    if self._items:
                        # 처리 도중 새 작업이 들어왔으므로 즉시 다시 실행한다.
                        self._schedule_timer(0)

    def flush(self, *, skip_inter_item_delay: bool = False) -> None:
        """queue를 즉시 처리하도록 강제한다.

        테스트나 graceful shutdown에 쓴다.

        Args:
            skip_inter_item_delay: :meth:`_process_queue`로 그대로 전달되며 항목 사이
                rate-limit 대기를 건너뛴다. 종료 드레인 경로(:meth:`flush_sync`)용이다.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        self._process_queue(skip_inter_item_delay=skip_inter_item_delay)

    def flush_sync(self, timeout: float) -> bool:
        """``timeout``초로 제한된 best-effort 동기 flush를 수행한다.

        프로세스 종료 시 죽는 daemon timer만 예약하는 :meth:`flush_nowait`와 달리,
        이 메서드는 daemon 스레드에서 :meth:`flush`를 실행하고 최대 ``timeout``초까지
        완료를 기다린다. graceful shutdown용이다. 이것이 없으면 queue가 순수
        in-memory이고 debounce Timer가 daemon 스레드이므로, 마지막 timer 발화 이후
        쌓인 갱신이 재시작 / rolling 배포 / SIGTERM에서 유실된다.

        드레인은 단순한 ``flush()``가 놓치는 두 가지 race를 처리한다.

        - **실행 중인 worker.** debounce Timer가 이미 발화했다면 ``_process_queue``
          worker가 queue에서 꺼낸 context를 들고 LLM 호출 중이다(``_processing=True``,
          queue는 빈 상태). ``flush``만 부르면 ``_processing=True``를 보고 아무것도 하지
          않은 채 성공을 보고하는데, 그 worker는 여전히 실행 중이고 종료 시 죽을
          가능성이 크다. 그래서 남은 예산 범위에서 실행 중인 worker를 먼저 join한다.
        - **실패한 flush.** ``flush``는 예외를 던질 수 있는 동기 LLM 호출을 한다.
          성공은 정상 경로에서만 기록하므로 반환값이 docstring의 "완료"와 일치한다.

        Note: (1)단계와 (3)단계는 같은 ``deadline`` 예산을 나눠 쓴다. 실행 중인 worker가
        느리면 예산 대부분을 소모해 (3)단계가 아무 일도 못 할 수 있다. 따라서
        ``timeout``은 느린 worker와 남은 queue를 모두 감당할 수 있어야 한다. best-effort라
        예산 안에 못 비운 꼬리는 버려지며, 이는 flush를 안 한 것과 같은 실패 방향이고
        범위는 꼬리에 한정된다.

        드레인이 ``timeout`` 안에 실제로 끝났을 때만 ``True``를 반환한다(queue가 비고,
        실행 중인 worker가 없고, flush가 예외를 던지지 않은 경우).
        """
        deadline = time.monotonic() + timeout

        # (1) 실행 중인 _process_queue를 먼저 제한된 시간 안에서 기다린다. 그러지 않으면
        # flush()가 _processing=True를 보고 아무것도 하지 않은 채 성공을 보고하는데,
        # 그 worker는 종료 시 죽는 daemon 스레드에서 LLM 호출 중이라 이미 꺼내 둔
        # context가 유실된다.
        with self._lock:
            in_flight = self._processing_thread
        if in_flight is not None:
            in_flight.join(timeout=max(0.0, deadline - time.monotonic()))

        # (2) 진짜 유휴 상태. 대기 항목도 없고 실행 중인 worker도 없다.
        if self.pending_count == 0 and not self.is_processing:
            return True

        # (3) 타임아웃이 실제 hard stop이 되도록 daemon 스레드에서 queue를 비운다.
        # flush()는 중단할 수 없는 동기 LLM 호출을 하므로 Thread.join이 아니라
        # Event.wait로 기다린다.
        success = False
        done = threading.Event()

        def _run() -> None:
            nonlocal success
            try:
                self.flush(skip_inter_item_delay=True)
                success = True
            except Exception:
                logger.exception("Memory queue flush failed during shutdown drain")
            finally:
                done.set()

        worker = threading.Thread(target=_run, name="memory-shutdown-flush", daemon=True)
        worker.start()
        finished = done.wait(timeout=max(0.0, deadline - time.monotonic()))
        if not finished:
            return False
        # flush()가 반환됐다. 그 사이 다른 worker가 끼어들지 않았을 때만 성공으로 본다.
        return bool(success) and not self.is_processing

    def flush_nowait(self) -> None:
        """백그라운드 스레드에서 queue 처리를 즉시 시작한다."""
        with self._lock:
            # daemon 스레드이므로 _process_queue가 끝나기 전에 프로세스가 종료되면
            # 대기 중인 메시지가 유실될 수 있다. best-effort memory 갱신에서는 허용한다.
            self._schedule_timer(0)

    def clear(self) -> None:
        """처리하지 않고 queue를 비운다.

        테스트에 쓴다.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._items = []
            self._processing = False
            self._processing_thread = None
            self._reprocess_pending = False

    @property
    def pending_count(self) -> int:
        """대기 중인 갱신 개수를 반환한다."""
        with self._lock:
            return len(self._items)

    @property
    def is_processing(self) -> bool:
        """queue가 현재 처리 중인지 확인한다."""
        with self._lock:
            return self._processing
