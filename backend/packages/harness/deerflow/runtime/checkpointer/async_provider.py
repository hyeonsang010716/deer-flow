"""async checkpointer 팩토리.

리소스를 제대로 정리해야 하는 장시간 실행 async 서버를 위해 **async context manager**를
제공한다.

지원 backend: memory, sqlite, postgres.

사용법(예: FastAPI lifespan)::

    from deerflow.runtime.checkpointer.async_provider import make_checkpointer

    async with make_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer  # 설정이 없으면 InMemorySaver

동기 방식은 :mod:`deerflow.runtime.checkpointer.provider`를 참고한다.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from langgraph.types import Checkpointer

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.persistence.postgres_schema import create_schema_sql, dsn_with_search_path, normalize_libpq_dsn
from deerflow.runtime.checkpointer.provider import (
    POSTGRES_CONN_REQUIRED,
    POSTGRES_INSTALL,
    SQLITE_INSTALL,
)
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)


def _prepare_sqlite_checkpointer_path(raw: str) -> str:
    conn_str = resolve_sqlite_conn_str(raw)
    ensure_sqlite_parent_dir(conn_str)
    return conn_str


def _prepare_database_sqlite_checkpointer_path(db_config) -> str:
    conn_str = db_config.checkpointer_sqlite_path
    ensure_sqlite_parent_dir(conn_str)
    return conn_str


def _build_postgres_pool(conn_string: str, schema: str = ""):
    """TCP keepalive와 연결 확인이 설정된 AsyncConnectionPool을 만든다."""
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
        "keepalives": 1,
        "keepalives_idle": 60,
        "keepalives_interval": 10,
        "keepalives_count": 6,
    }
    # search_path는 kwargs["options"]가 아니라 DSN에 주입한다(연결 문자열에 이미 있는 libpq
    # option과 병합). psycopg는 kwargs["options"]를 conninfo *위에* 덮어써서 DSN이 제공한
    # statement_timeout 같은 option을 조용히 날려버리기 때문이다. 여기서 SQLAlchemy의
    # ``+driver`` suffix도 제거해 libpq가 DSN을 파싱할 수 있게 한다. 동기/DSN 경로와 동일하다.
    dsn = dsn_with_search_path(normalize_libpq_dsn(conn_string), schema)

    return AsyncConnectionPool(
        dsn,
        kwargs=kwargs,
        check=AsyncConnectionPool.check_connection,
    )


async def _ensure_postgres_schema_with_pool(pool, schema: str) -> None:
    """LangGraph가 테이블을 만들기 전에 설정된 schema를 생성한다."""
    statement = create_schema_sql(schema)
    if statement is None:
        return
    async with pool.connection() as conn:
        await conn.execute(statement)


def _ensure_postgres_imports():
    """(AsyncPostgresSaver, AsyncConnectionPool)을 import해 반환하고, 실패하면 ImportError를 낸다."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise ImportError(POSTGRES_INSTALL) from exc

    try:
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:
        raise ImportError(POSTGRES_INSTALL) from exc

    return AsyncPostgresSaver, AsyncConnectionPool


# ---------------------------------------------------------------------------
# async factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer(config) -> AsyncIterator[Checkpointer]:
    """checkpointer를 만들고 정리하는 async context manager."""
    if config.type == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if config.type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = await asyncio.to_thread(_prepare_sqlite_checkpointer_path, config.connection_string or "store.db")
        async with AsyncSqliteSaver.from_conn_string(conn_str) as saver:
            await saver.setup()
            yield saver
        return

    if config.type == "postgres":
        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        AsyncPostgresSaver, _ = _ensure_postgres_imports()
        pool = _build_postgres_pool(config.connection_string, config.postgres_schema)
        async with pool:
            await _ensure_postgres_schema_with_pool(pool, config.postgres_schema)
            saver = AsyncPostgresSaver(conn=pool)
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ---------------------------------------------------------------------------
# 공개 async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer_from_database(db_config) -> AsyncIterator[Checkpointer]:
    """통합 DatabaseConfig로부터 checkpointer를 만드는 async context manager."""
    if db_config.backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if db_config.backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = await asyncio.to_thread(_prepare_database_sqlite_checkpointer_path, db_config)
        async with AsyncSqliteSaver.from_conn_string(conn_str) as saver:
            await saver.setup()
            yield saver
        return

    if db_config.backend == "postgres":
        if not db_config.postgres_url:
            raise ValueError("database.postgres_url is required for the postgres backend")

        AsyncPostgresSaver, _ = _ensure_postgres_imports()
        pool = _build_postgres_pool(db_config.postgres_url, db_config.postgres_schema)
        async with pool:
            await _ensure_postgres_schema_with_pool(pool, db_config.postgres_schema)
            saver = AsyncPostgresSaver(conn=pool)
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unknown database backend: {db_config.backend!r}")


@contextlib.asynccontextmanager
async def _select_inner_checkpointer(app_config: AppConfig) -> AsyncIterator[Checkpointer]:
    """*app_config*가 선택한 raw checkpointer를 yield한다(delta-cache로 감싸지 않는다).

    우선순위:
    1. legacy ``checkpointer:`` 설정 섹션(하위 호환)
    2. 통합 ``database:`` 설정 섹션
    3. 기본값 InMemorySaver
    """
    # legacy: 독립 checkpointer 설정이 우선한다
    if app_config.checkpointer is not None:
        async with _async_checkpointer(app_config.checkpointer) as saver:
            yield saver
            return

    # 통합 database 설정
    db_config = getattr(app_config, "database", None)
    if db_config is not None and db_config.backend != "memory":
        async with _async_checkpointer_from_database(db_config) as saver:
            yield saver
            return

    # 기본값: in-memory
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()


@contextlib.asynccontextmanager
async def make_checkpointer(app_config: AppConfig | None = None) -> AsyncIterator[Checkpointer]:
    """호출자의 생명주기 동안 사용할 checkpointer를 yield하는 async context manager.

    리소스는 진입 시 열리고 종료 시 닫히며, 전역 상태는 없다::

        async with make_checkpointer(app_config) as checkpointer:
            app.state.checkpointer = checkpointer

    *config.yaml*에 checkpointer 설정이 없으면 ``InMemorySaver``를 yield한다.

    backend 선택 우선순위:
    1. legacy ``checkpointer:`` 설정 섹션(하위 호환)
    2. 통합 ``database:`` 설정 섹션
    3. 기본값 InMemorySaver

    실효 checkpoint channel mode가 ``delta``이면(프로세스에 고정된 mode가 우선하고, 없으면
    ``database.checkpoint_channel_mode``를 쓴다) raw saver를 :class:`CachedHistorySaver`로
    감싼다. 그 history cache의 수명은 이 context manager와 같다.
    """
    from deerflow.runtime.checkpoint_mode import frozen_checkpoint_channel_mode

    if app_config is None:
        app_config = get_app_config()

    async with _select_inner_checkpointer(app_config) as saver:
        db_config = getattr(app_config, "database", None)
        mode = frozen_checkpoint_channel_mode() or (db_config.checkpoint_channel_mode if db_config is not None else "full")
        if mode == "delta":
            from deerflow.runtime.checkpoint_cache.provider import (
                checkpoint_cache_key_prefix,
                make_checkpoint_cache,
            )
            from deerflow.runtime.checkpointer.cached_saver import CachedHistorySaver

            async with make_checkpoint_cache(app_config, serde=saver.serde) as cache:
                yield CachedHistorySaver(saver, cache, key_prefix=checkpoint_cache_key_prefix(app_config))
        else:
            yield saver
