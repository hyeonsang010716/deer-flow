"""프로세스 로컬 LRU backend. hit 경로에서는 직렬화를 전혀 하지 않는다."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from deerflow.runtime.checkpoint_cache.base import CheckpointCacheStats, thread_key_stem


def _copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """읽기/쓰기 시 복사한다. writes는 새 list로 만들고, seed는 공유한다(제자리 변경 없음)."""
    copied: dict[str, Any] = {"writes": list(entry["writes"])}
    if "seed" in entry:
        copied["seed"] = entry["seed"]
    return copied


class MemoryCheckpointHistoryCache:
    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        self._max_entries = max_entries
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0

    def get_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for key in keys:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                continue
            self._data.move_to_end(key)
            self._hits += 1
            found[key] = _copy_entry(entry)
        return found

    def set_many(self, entries: dict[str, dict[str, Any]]) -> None:
        if not self.enabled:
            return
        for key, entry in entries.items():
            self._data[key] = _copy_entry(entry)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    async def aget_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        return self.get_many(keys)

    async def aset_many(self, entries: dict[str, dict[str, Any]]) -> None:
        self.set_many(entries)

    def delete_thread(self, key_prefix: str, thread_id: str) -> None:
        """한 thread의 모든 항목을 제거한다(무효화가 아니라 lifecycle 정리다)."""
        stem = thread_key_stem(key_prefix, thread_id)
        for key in [k for k in self._data if k.startswith(stem)]:
            del self._data[key]

    async def adelete_thread(self, key_prefix: str, thread_id: str) -> None:
        self.delete_thread(key_prefix, thread_id)

    def stats(self) -> CheckpointCacheStats:
        return CheckpointCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            entries=len(self._data),
        )

    async def aclose(self) -> None:
        self._data.clear()
