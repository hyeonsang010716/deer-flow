"""DeerFlow 애플리케이션 테이블용 Alembic 환경.

DeerFlow 소유 테이블(runs, threads_meta, feedback, users, run_events,
channel_connections, channel_credentials, channel_oauth_states,
channel_conversations)만 관리한다.

LangGraph의 checkpointer 테이블(``checkpoints``, ``checkpoint_blobs``,
``checkpoint_writes``, ``checkpoint_migrations``)은 LangGraph가 직접 관리한다. 자체 schema
lifecycle을 가지므로 Alembic이 건드려서는 안 된다. 아래 ``include_object`` 필터가 이들을 명시적으로
제외하므로, 이후 ``alembic revision --autogenerate``가 소유하지 않은 테이블에 대해
``drop_table``을 만들어내지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from deerflow.persistence.base import Base
from deerflow.persistence.migrations._env_filters import (
    LANGGRAPH_OWNED_TABLES,
    include_object,
)

# ``env.LANGGRAPH_OWNED_TABLES`` / ``env.include_object``로 접근하는 소비자를 위해 모듈
# 네임스페이스에 다시 내보낸다.
__all__ = ["LANGGRAPH_OWNED_TABLES", "include_object"]

# metadata가 채워지도록 모든 model을 import한다.
try:
    import deerflow.persistence.models as models  # ORM model을 Base.metadata에 등록한다

    _ = models
except ImportError:
    # model을 쓸 수 없다. migration은 기존 metadata만으로 동작한다.
    logging.getLogger(__name__).warning("Could not import deerflow.persistence.models; Alembic may not detect all tables")

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # SQLite ALTER TABLE 지원에 필요하다
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    # custom Postgres schema가 설정되어 있으면 alembic이 만든 engine의 search_path를 거기에
    # 고정한다. 이 engine은 순수 URL로 만들어져서 app engine이 ``connect_args``로 설정하는
    # asyncpg ``server_settings``를 상속받지 않는다. 이 처리가 없으면 ORM 테이블은 custom
    # schema에 생기는데 ``alembic_version``과 모든 migration DDL은 기본(``public``) schema로
    # 들어간다. bootstrap 전에 ``init_engine``이 이미 schema를 만들어 둔다
    # (``CREATE SCHEMA IF NOT EXISTS``).
    pg_schema = config.get_main_option("deerflow_pg_schema")
    connect_args: dict = {}
    # 표준 ``postgresql`` scheme과 libpq의 축약형 ``postgres`` scheme을 모두 받아들인다
    # (SQLAlchemy ``+driver`` 접미사 유무 무관). 그래야 ``postgres://`` DSN도 search_path가
    # 고정되고, ``alembic_version``과 migration DDL이 조용히 기본 schema로 들어가지 않는다.
    if pg_schema and url and url.split("+", 1)[0].split(":", 1)[0] in {"postgresql", "postgres"}:
        from deerflow.persistence.postgres_schema import build_asyncpg_connect_args

        connect_args = build_asyncpg_connect_args(pg_schema)

    connectable = create_async_engine(url, connect_args=connect_args)

    # SQLite의 프로세스 간 bootstrap 안전장치. alembic이 여는 모든 connection에 넉넉한
    # ``busy_timeout``이 필요하다. 그래야 다른 프로세스가 파일 write lock을 쥐고 있을 때
    # (예: bootstrap 도중) 우리 write가 ``database is locked``를 내지 않고 대기한다.
    # ``deerflow.persistence.engine``의 운영 engine은 자기 connection에 이를 설정하지만,
    # alembic은 여기서 자체 engine을 만들기 때문에 같은 hook을 걸어주지 않으면 아무것도
    # 상속받지 못한다.
    if connectable.url.drivername.startswith("sqlite"):
        from sqlalchemy import event

        @event.listens_for(connectable.sync_engine, "connect")
        def _alembic_sqlite_busy_timeout(dbapi_conn, _record):  # noqa: ARG001
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout=30000;")
            finally:
                cursor.close()

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
