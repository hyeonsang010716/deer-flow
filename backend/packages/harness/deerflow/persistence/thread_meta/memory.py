"""LangGraph BaseStore를 기반으로 하는 in-memory ThreadMetaStore.

database.backend=memory일 때 쓴다. LangGraph Store의 ``("threads",)`` namespace에 위임하며,
이는 Gateway router가 thread 레코드에 쓰는 것과 같은 namespace다.
"""

from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore

from deerflow.persistence.thread_meta.base import THREAD_PINNED_METADATA_KEY, ThreadMetaStore
from deerflow.runtime.user_context import AUTO, _AutoSentinel, resolve_user_id
from deerflow.utils.time import coerce_iso, now_iso

THREADS_NS: tuple[str, ...] = ("threads",)
SEARCH_PAGE_SIZE = 500


class MemoryThreadMetaStore(ThreadMetaStore):
    def __init__(self, store: BaseStore) -> None:
        self._store = store

    async def _get_owned_record(
        self,
        thread_id: str,
        user_id: str | None | _AutoSentinel,
        method_name: str,
    ) -> dict | None:
        """레코드를 가져와 소유권을 확인한다. 변경 가능한 복사본 또는 None을 반환한다."""
        resolved = resolve_user_id(user_id, method_name=method_name)
        item = await self._store.aget(THREADS_NS, thread_id)
        if item is None:
            return None
        record = dict(item.value)
        if resolved is not None and record.get("user_id") != resolved:
            return None
        return record

    async def create(
        self,
        thread_id: str,
        *,
        assistant_id: str | None = None,
        user_id: str | None | _AutoSentinel = AUTO,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        resolved_user_id = resolve_user_id(user_id, method_name="MemoryThreadMetaStore.create")
        now = now_iso()
        record: dict[str, Any] = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "user_id": resolved_user_id,
            "display_name": display_name,
            "status": "idle",
            "metadata": metadata or {},
            "values": {},
            "created_at": now,
            "updated_at": now,
        }
        await self._store.aput(THREADS_NS, thread_id, record)
        return record

    async def get(self, thread_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> dict | None:
        return await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.get")

    async def search(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> list[dict[str, Any]]:
        """일치하는 항목을 모두 구체화한 뒤 Python에서 정렬해 thread를 검색한다.

        memory backend는 SQL의 pinned 우선 정렬을 그대로 재현하기 위해, 자르기 전에 일치하는
        모든 행을 청크 단위로 읽어 들인다. 확장 가능한 페이지네이션 I/O가 필요하면 SQL store를
        쓴다.
        """
        resolved_user_id = resolve_user_id(user_id, method_name="MemoryThreadMetaStore.search")
        filter_dict: dict[str, Any] = {}
        if metadata:
            filter_dict.update(metadata)
        if status:
            filter_dict["status"] = status
        if resolved_user_id is not None:
            filter_dict["user_id"] = resolved_user_id

        items = []
        search_offset = 0
        while True:
            page = await self._store.asearch(
                THREADS_NS,
                filter=filter_dict or None,
                limit=SEARCH_PAGE_SIZE,
                offset=search_offset,
            )
            if not page:
                break
            items.extend(page)
            if len(page) < SEARCH_PAGE_SIZE:
                break
            search_offset += len(page)

        records = [self._item_to_dict(item) for item in items]
        records.sort(key=self._sort_key, reverse=True)
        return records[offset : offset + limit]

    async def check_access(self, thread_id: str, user_id: str, *, require_existing: bool = False) -> bool:
        item = await self._store.aget(THREADS_NS, thread_id)
        if item is None:
            return not require_existing
        record_user_id = item.value.get("user_id")
        if record_user_id is None:
            return True
        return record_user_id == user_id

    async def update_display_name(self, thread_id: str, display_name: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        record = await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.update_display_name")
        if record is None:
            return
        record["display_name"] = display_name
        record["updated_at"] = now_iso()
        await self._store.aput(THREADS_NS, thread_id, record)

    async def update_status(self, thread_id: str, status: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        record = await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.update_status")
        if record is None:
            return
        record["status"] = status
        record["updated_at"] = now_iso()
        await self._store.aput(THREADS_NS, thread_id, record)

    async def update_metadata(self, thread_id: str, metadata: dict, *, touch: bool = True, user_id: str | None | _AutoSentinel = AUTO) -> None:
        record = await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.update_metadata")
        if record is None:
            return
        merged = dict(record.get("metadata") or {})
        merged.update(metadata)
        record["metadata"] = merged
        if touch:
            record["updated_at"] = now_iso()
        await self._store.aput(THREADS_NS, thread_id, record)

    async def update_owner(self, thread_id: str, owner_user_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        record = await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.update_owner")
        if record is None:
            return
        record["user_id"] = owner_user_id
        record["updated_at"] = now_iso()
        await self._store.aput(THREADS_NS, thread_id, record)

    async def delete(self, thread_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        record = await self._get_owned_record(thread_id, user_id, "MemoryThreadMetaStore.delete")
        if record is None:
            return
        await self._store.adelete(THREADS_NS, thread_id)

    @staticmethod
    def _item_to_dict(item) -> dict[str, Any]:
        """Store SearchItem을 호출자가 기대하는 dict 형식으로 변환한다."""
        val = item.value
        return {
            "thread_id": item.key,
            "assistant_id": val.get("assistant_id"),
            "user_id": val.get("user_id"),
            "display_name": val.get("display_name"),
            "status": val.get("status", "idle"),
            "metadata": val.get("metadata", {}),
            # ``coerce_iso``는 ``str(time.time())``을 호출하던 예전 Gateway 버전이 남긴
            # unix 초 값을 보정한다.
            "created_at": coerce_iso(val.get("created_at", "")),
            "updated_at": coerce_iso(val.get("updated_at", "")),
        }

    @staticmethod
    def _sort_key(record: dict[str, Any]) -> tuple[bool, str, str]:
        metadata = record.get("metadata")
        pinned = isinstance(metadata, dict) and metadata.get(THREAD_PINNED_METADATA_KEY) is True
        return (pinned, str(record.get("updated_at") or ""), str(record.get("thread_id") or ""))
