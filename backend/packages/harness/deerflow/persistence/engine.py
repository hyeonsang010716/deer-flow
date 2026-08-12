"""async SQLAlchemy engine 라이프사이클 관리.

Gateway 시작 시 초기화하고, repository에 session factory를 제공하며, 종료 시 정리한다.

database.backend="memory"일 때 init_engine은 아무것도 하지 않고
get_session_factory()는 None을 반환한다. repository는 None을 확인하고 in-memory 구현으로
fallback해야 한다.
"""

from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

# 오래 idle 상태인 socket이 pool_pre_ping을 멈춰 세우기 전에 pool의 Postgres connection을
# 재활용한다. command timeout은 별개로 멈춘 ORM 쿼리의 상한을 잡는다.
POSTGRES_POOL_RECYCLE_SECONDS = 300
POSTGRES_COMMAND_TIMEOUT_SECONDS = 30


def _json_serializer(obj: object) -> str:
    """중국어 문자를 지원하기 위해 ensure_ascii=False를 쓰는 JSON serializer."""
    return json.dumps(obj, ensure_ascii=False)


def _postgres_engine_kwargs(
    *,
    echo: bool,
    pool_size: int,
    pool_recycle: int = POSTGRES_POOL_RECYCLE_SECONDS,
    command_timeout: float | None = POSTGRES_COMMAND_TIMEOUT_SECONDS,
    connect_args: dict[str, object] | None = None,
) -> dict[str, object]:
    """PostgreSQL용 공통 SQLAlchemy engine 옵션을 만든다."""
    merged_connect_args = dict(connect_args or {})
    if command_timeout is not None:
        merged_connect_args["command_timeout"] = command_timeout
    return {
        "echo": echo,
        "pool_size": pool_size,
        "pool_pre_ping": True,
        "pool_recycle": pool_recycle,
        "connect_args": merged_connect_args,
        "json_serializer": _json_serializer,
    }


logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def _auto_create_postgres_db(url: str) -> None:
    """``postgres`` 관리용 DB에 접속해 CREATE DATABASE를 실행한다.

    대상 database 이름은 *url*에서 추출한다. 접속은 같은 서버의 기본 ``postgres``
    database에 ``AUTOCOMMIT`` isolation으로 맺는다(CREATE DATABASE는 transaction 안에서
    실행할 수 없다).
    """
    from sqlalchemy import text
    from sqlalchemy.engine.url import make_url

    parsed = make_url(url)
    db_name = parsed.database
    if not db_name:
        raise ValueError("Cannot auto-create database: no database name in URL")

    # CREATE DATABASE를 실행하기 위해 기본 'postgres' database에 접속한다
    maint_url = parsed.set(database="postgres")
    maint_engine = create_async_engine(maint_url, isolation_level="AUTOCOMMIT")
    try:
        async with maint_engine.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        logger.info("Auto-created PostgreSQL database: %s", db_name)
    finally:
        await maint_engine.dispose()


async def init_engine(
    backend: str,
    *,
    url: str = "",
    echo: bool = False,
    pool_size: int = 5,
    pool_recycle: int = POSTGRES_POOL_RECYCLE_SECONDS,
    command_timeout: float | None = POSTGRES_COMMAND_TIMEOUT_SECONDS,
    sqlite_dir: str = "",
    postgres_schema: str = "",
) -> None:
    """async engine과 session factory를 만든 뒤 테이블을 자동 생성한다.

    Args:
        backend: "memory", "sqlite", "postgres" 중 하나.
        url: SQLAlchemy async URL (sqlite/postgres용).
        echo: SQL을 로그로 출력할지 여부.
        pool_size: Postgres connection pool 크기.
        pool_recycle: Postgres connection을 재활용하기까지의 초.
        command_timeout: app ORM Postgres 명령의 timeout(초). None이면 비활성화.
        sqlite_dir: SQLite용으로 생성할 디렉터리(존재를 보장한다).
        postgres_schema: 대상 PostgreSQL schema. 설정하면 engine이 asyncpg
            ``server_settings``로 connection ``search_path``를 고정하고, 테이블
            자동 생성 전에 schema를 (없으면) 생성한다. postgres가 아니면 무시한다.
    """
    global _engine, _session_factory

    if backend == "memory":
        logger.info("Persistence backend=memory -- ORM engine not initialized")
        return

    if backend == "postgres":
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise ImportError(
                "database.backend is set to 'postgres' but asyncpg is not installed.\n"
                "Install it with:\n"
                "    cd backend && uv sync --all-packages --extra postgres\n"
                "On the next `make dev` the postgres extra is auto-detected from\n"
                "config.yaml (database.backend: postgres) and reinstalled, so it\n"
                "will not be wiped again. Set UV_EXTRAS=postgres in .env to opt in\n"
                "explicitly. Or switch to backend: sqlite in config.yaml for\n"
                "single-node deployment."
            ) from None

    if backend == "sqlite":
        import os

        from sqlalchemy import event

        # 디렉터리 생성을 offload한다. ``init_engine``은 FastAPI lifespan event loop에서
        # 돌고, 동기 ``os.makedirs``(stat + mkdir syscall)는 시작 중에 그 loop를 막는다.
        # checkpointer의 ``ensure_sqlite_parent_dir``에 적용한 #1912 수정과 동일하다.
        await asyncio.to_thread(os.makedirs, sqlite_dir or ".", exist_ok=True)
        _engine = create_async_engine(url, echo=echo, json_serializer=_json_serializer)

        # 새 connection마다 WAL을 켠다. SQLite PRAGMA 설정은 connection 단위이므로
        # 시작 시 PRAGMA를 한 번 실행하는 대신 listener를 건다. WAL은 읽기와 쓰기를
        # 서로 막지 않고 동시에 처리하게 해주며, 운영 SQLite 배포의 표준 권장 사항이다
        # (AUTH_TEST_PLAN.md의 TC-UPG-06). 짝이 되는 ``synchronous=NORMAL``은
        # 안전하면서 빠른 조합이다 — commit마다가 아니라 WAL checkpoint 경계에서만
        # fsync한다.
        # 여기서 ``busy_timeout``도 30초로 늘린다. Python sqlite3 driver의 기본값 5초는
        # 일시적인 row 경합에는 충분하지만 프로세스 간 부트스트랩에는 너무 빡빡하다:
        # 두 번째~N번째 Gateway 프로세스가 첫 번째 프로세스의 새 schema용
        # ``ALTER TABLE``/``CREATE TABLE``이 끝나기를 기다려야 할 수 있다. 같은 값으로
        # 늘린 timeout을 ``migrations/env.py::run_migrations_online``의 alembic 생성
        # engine에도 똑같이 적용해 connection 동작을 일치시킨다.
        @event.listens_for(_engine.sync_engine, "connect")
        def _enable_sqlite_wal(dbapi_conn, _record):  # noqa: ARG001 — SQLAlchemy contract
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.execute("PRAGMA busy_timeout=30000;")
            finally:
                cursor.close()
    elif backend == "postgres":
        from deerflow.persistence.postgres_schema import build_asyncpg_connect_args

        pg_connect_args = build_asyncpg_connect_args(postgres_schema)
        _engine = create_async_engine(
            url,
            **_postgres_engine_kwargs(
                echo=echo,
                pool_size=pool_size,
                pool_recycle=pool_recycle,
                command_timeout=command_timeout,
                connect_args=pg_connect_args,
            ),
        )
    else:
        raise ValueError(f"Unknown persistence backend: {backend!r}")

    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    # schema 부트스트랩(하이브리드):
    #   - 빈 DB      -> create_all + alembic stamp head
    #   - 레거시 DB   -> create_all (baseline 테이블만, backfill) + alembic stamp baseline + upgrade head
    #   - 이미 관리 중 -> alembic upgrade head
    # 동시성: Postgres는 advisory lock(진짜 프로세스 간); SQLite는 in-process
    # asyncio.Lock에 30초 PRAGMA busy_timeout을 더한다(env.py의 alembic 자체
    # connection에도 설정) -- 다중 프로세스 SQLite 부트스트랩은 SQLite의 파일 단위 write
    # lock에 기대는 best-effort다.
    # 전체 state machine은 deerflow.persistence.bootstrap 참고.
    from deerflow.persistence.bootstrap import bootstrap_schema

    async def _ensure_postgres_schema() -> None:
        # CREATE SCHEMA는 DDL이라 search_path의 영향을 받지 않으므로, connection의
        # search_path가 이미 (아직 없는) 대상 schema를 가리켜도 안전하다. 이어지는
        # ``create_all``/alembic DDL이 없는 schema 때문에 실패하지 않고 대상 schema에
        # 들어가도록 반드시 ``bootstrap_schema``보다 먼저 실행해야 한다.
        if backend == "postgres" and postgres_schema:
            from sqlalchemy.schema import CreateSchema

            async with _engine.begin() as conn:
                await conn.execute(CreateSchema(postgres_schema, if_not_exists=True))

    try:
        await _ensure_postgres_schema()
        await bootstrap_schema(_engine, backend=backend, postgres_schema=postgres_schema)
    except Exception as exc:
        if backend == "postgres" and "does not exist" in str(exc):
            # database가 아직 없다 -- 자동 생성을 시도한 뒤 재시도한다.
            await _auto_create_postgres_db(url)
            # 이제 존재하는 database를 대상으로 engine을 다시 만든다. 재시도한
            # 부트스트랩이 기본 schema가 아닌 대상 schema에 들어가도록, 다시 만든
            # engine은 반드시 같은 connect_args를 유지해야 한다.
            await _engine.dispose()
            _engine = create_async_engine(
                url,
                **_postgres_engine_kwargs(
                    echo=echo,
                    pool_size=pool_size,
                    pool_recycle=pool_recycle,
                    command_timeout=command_timeout,
                    connect_args=pg_connect_args,
                ),
            )
            _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
            await _ensure_postgres_schema()
            await bootstrap_schema(_engine, backend=backend, postgres_schema=postgres_schema)
        else:
            raise

    logger.info("Persistence engine initialized: backend=%s", backend)


async def init_engine_from_config(config) -> None:
    """편의 함수: DatabaseConfig 객체로 engine을 초기화한다."""
    if config.backend == "memory":
        await init_engine("memory")
        return
    await init_engine(
        backend=config.backend,
        url=config.app_sqlalchemy_url,
        echo=config.echo_sql,
        pool_size=config.pool_size,
        pool_recycle=config.pool_recycle,
        command_timeout=config.command_timeout,
        sqlite_dir=config.sqlite_dir if config.backend == "sqlite" else "",
        postgres_schema=config.postgres_schema if config.backend == "postgres" else "",
    )


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """async session factory를 반환한다. backend=memory면 None."""
    return _session_factory


def get_engine() -> AsyncEngine | None:
    """async engine을 반환한다. 초기화되지 않았으면 None."""
    return _engine


async def close_engine() -> None:
    """engine을 정리하고 모든 connection을 반환한다."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Persistence engine closed")
    _engine = None
    _session_factory = None
