"""단일 instance 배포용 in-process ownership store.

gateway 프로세스 하나가 컨테이너 backend를 독점할 때만 올바르다. 여기 있는 상태는 다른
프로세스에 전혀 보이지 않으므로, peer는 모든 컨테이너를 주인 없는 것으로 보고 흡수한다.
:attr:`supports_cross_process`가 ``False``인 이유이며 provider도 startup에서 경고한다.
multi-worker / multi-instance gateway는 redis store를 써야 한다. `stream_bridge`의
memory backend와 같은 규칙이다.

TTL과 두 lease 상태는 stub이 아니라 실제로 구현한다. 그래야 하나의 store 계약 테스트가
두 backend를 모두 검증하고, lapsed lease가 어떤 store를 설정하든 동일하게 동작한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .base import RenewOutcome, SandboxOwnershipStore


@dataclass(frozen=True, slots=True)
class _Lease:
    owner_id: str
    expires_at: float
    destroying: bool


class MemoryOwnershipStore(SandboxOwnershipStore):
    """이 프로세스 안에서만 유지되는 ownership lease."""

    supports_cross_process = False

    def __init__(self, *, owner_id: str, ttl_seconds: float, time_source=time.monotonic) -> None:
        self._owner_id = owner_id
        self._ttl = float(ttl_seconds)
        self._now = time_source
        # sandbox_id -> _Lease. acquire 경로, idle checker 스레드, renewal 스레드가 모두
        # 건드리므로 _lock으로 보호한다.
        self._leases: dict[str, _Lease] = {}
        self._lock = threading.Lock()

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def _live_lease_locked(self, sandbox_id: str) -> _Lease | None:
        lease = self._leases.get(sandbox_id)
        if lease is None:
            return None
        if self._now() >= lease.expires_at:
            del self._leases[sandbox_id]
            return None
        return lease

    def _write_locked(self, sandbox_id: str, *, destroying: bool) -> None:
        self._leases[sandbox_id] = _Lease(owner_id=self._owner_id, expires_at=self._now() + self._ttl, destroying=destroying)

    def take(self, sandbox_id: str) -> bool:
        with self._lock:
            lease = self._live_lease_locked(sandbox_id)
            # 진행 중인 teardown만 거절한다. 살아 있는 peer의 일반 lease는 인수하며,
            # 그것이 take()의 존재 이유다.
            if lease is not None and lease.destroying:
                return False
            self._write_locked(sandbox_id, destroying=False)
            return True

    def claim(self, sandbox_id: str, *, for_destroy: bool = False) -> bool:
        with self._lock:
            lease = self._live_lease_locked(sandbox_id)
            if lease is not None and lease.owner_id != self._owner_id:
                return False
            if not for_destroy and lease is not None and lease.destroying:
                # 우리 자신의 teardown은 절대 되감지 않는다. stop이 이미 진행 중이라
                # 취소할 수 없으므로, `own:`으로 낮추면 `take()`가 곧 죽을 컨테이너를
                # 넘겨주게 된다.
                return False
            self._write_locked(sandbox_id, destroying=for_destroy)
            return True

    def renew(self, sandbox_id: str) -> RenewOutcome:
        with self._lock:
            lease = self._live_lease_locked(sandbox_id)
            if lease is None:
                return RenewOutcome.LAPSED
            if lease.owner_id != self._owner_id or lease.destroying:
                return RenewOutcome.LOST
            self._write_locked(sandbox_id, destroying=False)
            return RenewOutcome.RENEWED

    def release(self, sandbox_id: str) -> None:
        with self._lock:
            lease = self._live_lease_locked(sandbox_id)
            if lease is not None and lease.owner_id == self._owner_id:
                del self._leases[sandbox_id]

    def owner(self, sandbox_id: str) -> str | None:
        with self._lock:
            lease = self._live_lease_locked(sandbox_id)
            return None if lease is None else lease.owner_id

    def close(self) -> None:
        with self._lock:
            self._leases.clear()
