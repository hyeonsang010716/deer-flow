"""community sandbox provider들이 공유하는 warm pool lifecycle 헬퍼."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT = 600
DEFAULT_REPLICAS = 3
IDLE_CHECK_INTERVAL = 60


class WarmPoolLifecycleMixin[WarmEntryT]:
    """provider의 warm pool 만료와 replica lifecycle 처리를 담는 Mixin."""

    DEFAULT_IDLE_TIMEOUT = DEFAULT_IDLE_TIMEOUT
    DEFAULT_REPLICAS = DEFAULT_REPLICAS
    IDLE_CHECK_INTERVAL = IDLE_CHECK_INTERVAL
    _idle_checker_thread_name = "warm-pool-idle-checker"

    _lock: threading.Lock
    _warm_pool: dict[str, tuple[WarmEntryT, float]]
    _config: dict[str, Any]
    _idle_checker_stop: threading.Event
    _idle_checker_thread: threading.Thread | None

    def _active_count_locked(self) -> int:
        """``_lock`` 을 쥔 상태에서 active 항목 수를 반환한다."""
        raise NotImplementedError

    def _destroy_warm_entry(self, sandbox_id: str, entry: WarmEntryT, *, reason: str) -> None:
        """pool에서 제거된 warm pool 항목을 파괴한다."""
        raise NotImplementedError

    def _replica_count(self) -> tuple[int, int]:
        """설정된 replicas 값과 현재 active + warm 항목 수를 반환한다."""
        replicas = int(self._config.get("replicas", DEFAULT_REPLICAS))
        with self._lock:
            total = self._active_count_locked() + len(self._warm_pool)
        return replicas, total

    def _log_replicas_soft_cap(self, replicas: int, sandbox_id: str, evicted: str | None) -> None:
        """warm pool replica soft cap 적용 결과를 로그로 남긴다."""
        if evicted is not None:
            logger.info("Evicted warm-pool sandbox %s to stay within replicas=%s", evicted, replicas)
            return

        logger.warning(
            "All %s replica slots are in active use; creating sandbox %s beyond the soft limit",
            replicas,
            sandbox_id,
        )

    def _evict_oldest_warm(self) -> str | None:
        """timestamp 기준으로 가장 오래된 warm 항목을 제거하고 파괴한다."""
        with self._lock:
            if not self._warm_pool:
                return None
            sandbox_id, (entry, _) = min(self._warm_pool.items(), key=lambda item: item[1][1])
            self._warm_pool.pop(sandbox_id)

        self._destroy_warm_entry(sandbox_id, entry, reason="replica_enforcement")
        return sandbox_id

    def _reap_expired_warm(self, idle_timeout: float | None = None) -> None:
        """``idle_timeout`` 초보다 오래된 warm 항목을 제거하고 파괴한다."""
        timeout = float(self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT) if idle_timeout is None else idle_timeout)
        if timeout <= 0:
            return

        now = time.time()
        expired: list[tuple[str, WarmEntryT]] = []
        with self._lock:
            for sandbox_id, (entry, timestamp) in self._warm_pool.items():
                if now - timestamp > timeout:
                    expired.append((sandbox_id, entry))
            for sandbox_id, _ in expired:
                self._warm_pool.pop(sandbox_id, None)

        for sandbox_id, entry in expired:
            self._destroy_warm_entry(sandbox_id, entry, reason="idle_timeout")

    def _start_idle_checker(self) -> None:
        """유휴 warm 항목을 주기적으로 정리하는 daemon thread를 시작한다."""
        if self._idle_checker_thread is not None and self._idle_checker_thread.is_alive():
            return

        self._idle_checker_stop.clear()
        self._idle_checker_thread = threading.Thread(
            target=self._idle_checker_loop,
            name=self._idle_checker_thread_name,
            daemon=True,
        )
        self._idle_checker_thread.start()
        logger.info("Started warm-pool idle checker thread (timeout: %ss)", self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))

    def _stop_idle_checker(self) -> None:
        """idle checker thread를 멈추고, 실행 중이면 종료를 기다린다."""
        self._idle_checker_stop.set()
        thread = self._idle_checker_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _idle_checker_loop(self) -> None:
        """stop 이벤트가 설정될 때까지 주기적으로 idle 정리를 수행한다."""
        idle_timeout = float(self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        while not self._idle_checker_stop.wait(self.IDLE_CHECK_INTERVAL):
            try:
                self._cleanup_idle_resources(idle_timeout)
            except Exception:
                logger.exception("Error in warm-pool idle checker loop")

    def _cleanup_idle_resources(self, idle_timeout: float) -> None:
        """``idle_timeout`` 초보다 오래 유휴 상태인 리소스를 정리한다."""
        self._reap_expired_warm(idle_timeout)


__all__ = [
    "DEFAULT_IDLE_TIMEOUT",
    "DEFAULT_REPLICAS",
    "IDLE_CHECK_INTERVAL",
    "WarmPoolLifecycleMixin",
]
