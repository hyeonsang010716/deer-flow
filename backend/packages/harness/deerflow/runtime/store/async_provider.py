"""async Store factory. backend는 runtime persistence 설정을 그대로 따른다.

deprecated된 ``checkpointer`` 섹션이 있으면 그것이 우선한다. 없으면 Store는
*config.yaml*의 통합 ``database`` 섹션을 따른다:

- ``memory``   → :class:`langgraph.store.memory.InMemoryStore`
- ``sqlite``   → :class:`langgraph.store.sqlite.aio.AsyncSqliteStore`
- ``postgres`` → :class:`langgraph.store.postgres.aio.AsyncPostgresStore`

사용 예(예: FastAPI lifespan)::

    from deerflow.runtime.store import make_store

    async with make_store() as store:
        app.state.store = store
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from langgraph.store.base import BaseStore

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.persistence.postgres_schema import dsn_with_search_path, ensure_postgres_schema_async
from deerflow.runtime.store.provider import (
    POSTGRES_CONN_REQUIRED,
    POSTGRES_STORE_INSTALL,
    SQLITE_STORE_INSTALL,
    _resolve_store_config,
    ensure_sqlite_parent_dir,
    resolve_sqlite_conn_str,
)

logger = logging.getLogger(__name__)


async def _ensure_postgres_schema(conn_string: str, schema: str) -> None:
    """LangGraph가 store 테이블을 만들기 전에 설정된 schema를 생성한다."""
    await ensure_postgres_schema_async(conn_string, schema, install_hint=POSTGRES_STORE_INSTALL)


# ---------------------------------------------------------------------------
# 내부 backend factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_store(config) -> AsyncIterator[BaseStore]:
    """Store를 만들고 정리하는 async context manager.

    ``config`` 인자는 :class:`deerflow.config.checkpointer_config.CheckpointerConfig`
    인스턴스이며, checkpointer factory가 쓰는 것과 같은 객체다.
    """
    if config.type == "memory":
        from langgraph.store.memory import InMemoryStore

        logger.info("Store: using InMemoryStore (in-process, not persistent)")
        yield InMemoryStore()
        return

    if config.type == "sqlite":
        try:
            from langgraph.store.sqlite.aio import AsyncSqliteStore
        except ImportError as exc:
            raise ImportError(SQLITE_STORE_INSTALL) from exc

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        await asyncio.to_thread(ensure_sqlite_parent_dir, conn_str)

        async with AsyncSqliteStore.from_conn_string(conn_str) as store:
            await store.setup()
            logger.info("Store: using AsyncSqliteStore (%s)", conn_str)
            yield store
        return

    if config.type == "postgres":
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(POSTGRES_STORE_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        await _ensure_postgres_schema(config.connection_string, config.postgres_schema)
        conn_string = dsn_with_search_path(config.connection_string, config.postgres_schema)
        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()
            logger.info("Store: using AsyncPostgresStore")
            yield store
        return

    raise ValueError(f"Unknown store backend type: {config.type!r}")


# ---------------------------------------------------------------------------
# 공개 async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def make_store(app_config: AppConfig | None = None) -> AsyncIterator[BaseStore]:
    """legacy 또는 통합 persistence 설정으로 선택한 Store를 내보낸다.

    legacy ``checkpointer`` 섹션이 설정되어 있으면 그것이 우선한다. 없으면 통합
    ``database`` 섹션이 backend를 고르며, 이는
    :func:`deerflow.runtime.checkpointer.async_provider.make_checkpointer`와 동일하다::

        async with make_store(app_config) as store:
            app.state.store = store

    :class:`~langgraph.store.memory.InMemoryStore`는 해석된 backend가 명시적으로
    ``memory``일 때만 반환된다.
    """
    if app_config is None:
        app_config = get_app_config()

    config = _resolve_store_config(app_config)
    async with _async_store(config) as store:
        yield store
