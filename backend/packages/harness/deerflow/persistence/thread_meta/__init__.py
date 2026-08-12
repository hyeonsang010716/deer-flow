"""thread metadata 영속화 — ORM, 추상 store, 구체 구현."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deerflow.persistence.thread_meta.base import THREAD_PINNED_METADATA_KEY, InvalidMetadataFilterError, ThreadMetaStore
from deerflow.persistence.thread_meta.memory import MemoryThreadMetaStore
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.persistence.thread_meta.sql import ThreadMetaRepository

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = [
    "InvalidMetadataFilterError",
    "MemoryThreadMetaStore",
    "THREAD_PINNED_METADATA_KEY",
    "ThreadMetaRepository",
    "ThreadMetaRow",
    "ThreadMetaStore",
    "make_thread_store",
]


def make_thread_store(
    session_factory: async_sessionmaker[AsyncSession] | None,
    store: BaseStore | None = None,
) -> ThreadMetaStore:
    """사용 가능한 backend에 맞는 ThreadMetaStore를 만든다.

    session factory가 있으면 SQL 기반 repository를, 없으면 in-memory LangGraph Store
    구현으로 되돌린다.
    """
    if session_factory is not None:
        return ThreadMetaRepository(session_factory)
    if store is None:
        raise ValueError("make_thread_store requires either a session_factory (SQL) or a store (memory)")
    return MemoryThreadMetaStore(store)
