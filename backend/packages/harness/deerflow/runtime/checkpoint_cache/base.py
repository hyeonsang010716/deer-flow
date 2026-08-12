"""checkpoint delta-history 항목을 위한 cache backend 계약.

항목은 ``DeltaChannelHistory`` 모양의 dict(``{"writes": [...], "seed"?}``)이며, 불변인
(database, thread, namespace, checkpoint_id, channel) 튜플을 키로 쓴다. checkpoint lineage는
append-only이고 checkpoint의 history는 자신의 pending write를 포함하지 않으므로, 한 번 쓰인
항목은 절대 바뀌지 않는다. 즉 정확성을 위해 invalidation이 필요 없고, 공유 backend는 별도 조율
없이도 프로세스 간에 일관성을 유지한다.

유일한 삭제 API는 thread 범위(``adelete_thread``/``delete_thread``)이며, 정확성이 아니라 순수하게
데이터 수명주기를 위해 존재한다. 원본 checkpoint가 지워지면(thread 삭제, tenant 해지, GDPR류
삭제) 해당 thread의 캐시된 history payload도 LRU eviction이나 TTL 만료를 기다리지 않고 함께
사라져야 한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

CACHE_FORMAT_VERSION = 1


def make_history_key(
    key_prefix: str,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    channel: str,
) -> str:
    """충돌하지 않는 cache key를 만든다.

    ``thread_id``는 운영 디버깅을 위해 읽을 수 있게 두고, 나머지 구성 요소는 NUL 구분자와 함께
    해시한다. 이렇게 하면 ':'를 포함한 namespace가 모호한 키를 만들 수 없다.
    """
    digest = hashlib.sha256(f"{checkpoint_ns}\x00{checkpoint_id}\x00{channel}".encode()).hexdigest()[:24]
    return f"{key_prefix}:{thread_id}:{digest}"


def thread_key_stem(key_prefix: str, thread_id: str) -> str:
    """한 thread의 모든 history key에 일치하는 prefix(make_history_key 참고)."""
    return f"{key_prefix}:{thread_id}:"


@dataclass
class CheckpointCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": self.entries,
        }


class CheckpointHistoryCache(Protocol):
    """async backend 계약. 삭제는 thread 범위의 수명주기 정리 용도뿐이다."""

    async def aget_many(self, keys: list[str]) -> dict[str, dict[str, Any]]: ...
    async def aset_many(self, entries: dict[str, dict[str, Any]]) -> None: ...
    async def adelete_thread(self, key_prefix: str, thread_id: str) -> None: ...
    def stats(self) -> CheckpointCacheStats: ...
    async def aclose(self) -> None: ...


class SyncCheckpointHistoryCache(Protocol):
    """sync backend 계약(embedded/TUI 경로). memory backend 전용이다."""

    def get_many(self, keys: list[str]) -> dict[str, dict[str, Any]]: ...
    def set_many(self, entries: dict[str, dict[str, Any]]) -> None: ...
    def delete_thread(self, key_prefix: str, thread_id: str) -> None: ...
    def stats(self) -> CheckpointCacheStats: ...
