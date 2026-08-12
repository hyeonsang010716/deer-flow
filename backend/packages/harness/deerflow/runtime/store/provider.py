"""동기 Store factory.

CLI 도구와 임베디드 :class:`~deerflow.client.DeerFlowClient`를 위한 **동기 싱글턴**과
**동기 context manager**를 제공한다.

deprecated된 ``checkpointer`` 섹션이 있으면 그쪽이 우선한다. 없으면 Store는 통합
``database`` 섹션을 따른다. 지원 backend: memory, sqlite, postgres.

사용법::

    from deerflow.runtime.store.provider import get_store, store_context

    # 싱글턴 — 호출 간에 재사용되고 프로세스 종료 시 닫힌다
    store = get_store()

    # 일회성 — 새 연결을 열고 블록을 벗어나면 닫는다
    with store_context() as store:
        store.put(("ns",), "key", {"value": 1})
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator

from langgraph.store.base import BaseStore

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.config.checkpointer_config import CheckpointerConfig, ensure_config_loaded, get_checkpointer_config
from deerflow.persistence.postgres_schema import dsn_with_search_path, ensure_postgres_schema
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 오류 메시지 상수
# ---------------------------------------------------------------------------

SQLITE_STORE_INSTALL = "langgraph-checkpoint-sqlite is required for the SQLite store. Install it with: uv add langgraph-checkpoint-sqlite"
POSTGRES_STORE_INSTALL = (
    "langgraph-checkpoint-postgres is required for the PostgreSQL store. Install the package extra with: pip install 'deerflow-harness[postgres]' (or use: uv sync --all-packages --extra postgres when developing locally)"
)
POSTGRES_CONN_REQUIRED = "checkpointer.connection_string is required for the postgres backend"


def _ensure_postgres_schema(conn_string: str, schema: str) -> None:
    """LangGraph가 store 테이블을 만들기 전에 설정된 schema를 생성한다."""
    ensure_postgres_schema(conn_string, schema, install_hint=POSTGRES_STORE_INSTALL)


def _resolve_store_config(app_config: AppConfig) -> CheckpointerConfig:
    """레거시 또는 통합 애플리케이션 config에서 Store backend를 결정한다.

    레거시 ``checkpointer`` 섹션이 있으면 그쪽이 권위를 유지해서 Store와 Checkpointer가 계속
    같은 backend를 쓴다. 없으면 문서대로 통합 ``database`` 섹션이 Store를 결정한다. 통합
    ``postgres_schema``도 전달해서 Store 테이블이 checkpointer 및 app 테이블과 같은 schema에
    생성되게 한다.
    """
    if app_config.checkpointer is not None:
        return app_config.checkpointer

    database = app_config.database
    if database is None or database.backend == "memory":
        return CheckpointerConfig(type="memory")
    if database.backend == "sqlite":
        return CheckpointerConfig(type="sqlite", connection_string=database.checkpointer_sqlite_path)
    if database.backend == "postgres":
        if not database.postgres_url:
            raise ValueError("database.postgres_url is required for the postgres backend")
        return CheckpointerConfig(type="postgres", connection_string=database.postgres_url, postgres_schema=database.postgres_schema)
    raise ValueError(f"Unknown database backend: {database.backend!r}")


def _get_store_config() -> CheckpointerConfig:
    """provider 싱글턴 lock을 잡지 않은 채로 Store config를 로드한다."""
    ensure_config_loaded()

    # 레거시 config 싱글턴을 직접 초기화하는 호출자를 계속 지원한다.
    legacy_config = get_checkpointer_config()
    if legacy_config is not None:
        return legacy_config
    try:
        app_config = get_app_config()
    except FileNotFoundError:
        return CheckpointerConfig(type="memory")
    return _resolve_store_config(app_config)


# ---------------------------------------------------------------------------
# 동기 factory
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _sync_store_cm(config) -> Iterator[BaseStore]:
    """동기 Store를 만들고 정리하는 context manager.

    ``config`` 인자는 :class:`~deerflow.config.checkpointer_config.CheckpointerConfig`
    인스턴스로, checkpointer factory가 쓰는 것과 같은 객체다.
    """
    if config.type == "memory":
        from langgraph.store.memory import InMemoryStore

        logger.info("Store: using InMemoryStore (in-process, not persistent)")
        yield InMemoryStore()
        return

    if config.type == "sqlite":
        try:
            from langgraph.store.sqlite import SqliteStore
        except ImportError as exc:
            raise ImportError(SQLITE_STORE_INSTALL) from exc

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        ensure_sqlite_parent_dir(conn_str)

        with SqliteStore.from_conn_string(conn_str) as store:
            store.setup()
            logger.info("Store: using SqliteStore (%s)", conn_str)
            yield store
        return

    if config.type == "postgres":
        try:
            from langgraph.store.postgres import PostgresStore  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(POSTGRES_STORE_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        _ensure_postgres_schema(config.connection_string, config.postgres_schema)
        conn_string = dsn_with_search_path(config.connection_string, config.postgres_schema)
        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()
            logger.info("Store: using PostgresStore")
            yield store
        return

    raise ValueError(f"Unknown store backend type: {config.type!r}")


# ---------------------------------------------------------------------------
# 동기 싱글턴
# ---------------------------------------------------------------------------

_store: BaseStore | None = None
_store_ctx = None  # 연결을 살려두기 위해 열어둔 context manager
_store_lock = threading.Lock()


def get_store() -> BaseStore:
    """전역 동기 Store 싱글턴을 반환하며, 첫 호출 때 생성한다.

    레거시 ``checkpointer`` 섹션이 설정되어 있으면 우선한다. 없으면 통합 ``database`` 섹션이
    backend를 고른다.

    Raises:
        ImportError: 설정된 backend에 필요한 패키지가 설치되어 있지 않은 경우.
        ValueError: 선택된 backend에 필요한 연결 값이 없는 경우.
    """
    global _store, _store_ctx

    if _store is not None:
        return _store

    # config 로딩은 두 persistence 싱글턴을 모두 리셋할 수 있다. lock 순서 역전을 피하려고
    # 전체 config는 이 provider lock 밖에서 해석한다.
    config = _get_store_config()

    with _store_lock:
        if _store is not None:
            return _store

        store_ctx = _sync_store_cm(config)
        store = store_ctx.__enter__()
        _store_ctx = store_ctx
        _store = store
    return _store


def reset_store() -> None:
    """동기 싱글턴을 리셋해서 다음 호출 때 다시 만들게 한다.

    열려 있는 backend 연결을 닫고 캐시된 인스턴스를 비운다. 테스트나 설정 변경 후에 유용하다.
    """
    global _store, _store_ctx
    with _store_lock:
        if _store_ctx is not None:
            try:
                _store_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("Error during store cleanup", exc_info=True)
            _store_ctx = None
        _store = None


# ---------------------------------------------------------------------------
# 동기 context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def store_context() -> Iterator[BaseStore]:
    """Store를 yield 하고 블록을 벗어나면 정리하는 동기 context manager.

    :func:`get_store`와 달리 인스턴스를 캐시하지 **않는다**. ``with`` 블록마다 자체 연결을
    만들고 없앤다. 정리 시점을 확정적으로 두고 싶은 CLI script나 테스트에서 쓴다::

        with store_context() as store:
            store.put(("threads",), thread_id, {...})

    레거시 ``checkpointer`` 섹션이 설정되어 있으면 우선한다. 없으면 통합 ``database`` 섹션이
    backend를 고른다.
    """
    config = _resolve_store_config(get_app_config())
    with _sync_store_cm(config) as store:
        yield store
