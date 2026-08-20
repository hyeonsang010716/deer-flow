"""동기 checkpointer factory.

LangGraph graph 컴파일과 CLI 도구를 위해 **동기 singleton**과 **동기 context
manager**를 제공한다.

지원 backend: memory, sqlite, postgres.

사용법::

    from deerflow.runtime.checkpointer.provider import get_checkpointer, checkpointer_context

    # singleton — 호출 간 재사용되고 프로세스 종료 시 닫힌다
    cp = get_checkpointer()

    # 일회성 — 새 connection을 만들고 블록을 벗어날 때 닫는다
    with checkpointer_context() as cp:
        graph.invoke(input, config={"configurable": {"thread_id": "1"}})
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator

from langgraph.types import Checkpointer

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.config.checkpointer_config import CheckpointerConfig, ensure_config_loaded, get_checkpointer_config
from deerflow.persistence.postgres_schema import dsn_with_search_path, ensure_postgres_schema
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 에러 메시지 상수 — aio.provider도 import한다
# ---------------------------------------------------------------------------

SQLITE_INSTALL = "langgraph-checkpoint-sqlite is required for the SQLite checkpointer. Install it with: uv add langgraph-checkpoint-sqlite"
POSTGRES_INSTALL = (
    "langgraph-checkpoint-postgres is required for the PostgreSQL checkpointer. Install the package extra with: pip install 'deerflow-harness[postgres]' (or use: uv sync --all-packages --extra postgres when developing locally)"
)
POSTGRES_CONN_REQUIRED = "checkpointer.connection_string is required for the postgres backend"


def _ensure_postgres_schema(conn_string: str, schema: str) -> None:
    """LangGraph가 테이블을 만들기 전에 설정된 schema를 생성한다."""
    ensure_postgres_schema(conn_string, schema, install_hint=POSTGRES_INSTALL)


# ---------------------------------------------------------------------------
# config 해석
# ---------------------------------------------------------------------------


def _resolve_checkpointer_config(app_config: AppConfig) -> CheckpointerConfig:
    """레거시 또는 통합 애플리케이션 config에서 checkpointer backend를 해석한다.

    레거시 ``checkpointer`` 섹션이 있으면 그것이 우선이라 Checkpointer와 Store가 같은
    backend를 계속 쓴다. 없으면 통합 ``database`` 섹션이 checkpointer를 결정하며, 이는
    async :func:`~deerflow.runtime.checkpointer.async_provider.make_checkpointer`
    factory 및 동기 Store provider의 ``_resolve_store_config``와 동일한 동작이다.
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


def _get_checkpointer_config() -> CheckpointerConfig:
    """provider singleton lock을 잡지 않은 채 checkpointer config를 로드한다."""
    ensure_config_loaded()

    # 레거시 config singleton을 직접 초기화하는 호출자를 그대로 지원한다.
    legacy_config = get_checkpointer_config()
    if legacy_config is not None:
        return legacy_config
    try:
        app_config = get_app_config()
    except FileNotFoundError:
        return CheckpointerConfig(type="memory")
    return _resolve_checkpointer_config(app_config)


# ---------------------------------------------------------------------------
# 동기 factory
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _sync_checkpointer_cm(config: CheckpointerConfig) -> Iterator[Checkpointer]:
    """동기 checkpointer를 만들고 정리하는 context manager.

    설정된 ``Checkpointer`` 인스턴스를 반환한다. 하위 connection이나 pool의 리소스
    정리는 이 모듈의 상위 헬퍼(singleton factory나 context manager)가 처리하며, 이
    함수는 별도의 정리 콜백을 반환하지 않는다.
    """
    if config.type == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        logger.info("Checkpointer: using InMemorySaver (in-process, not persistent)")
        yield InMemorySaver()
        return

    if config.type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        ensure_sqlite_parent_dir(conn_str)
        with SqliteSaver.from_conn_string(conn_str) as saver:
            saver.setup()
            logger.info("Checkpointer: using SqliteSaver (%s)", conn_str)
            yield saver
        return

    if config.type == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise ImportError(POSTGRES_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        _ensure_postgres_schema(config.connection_string, config.postgres_schema)
        conn_string = dsn_with_search_path(config.connection_string, config.postgres_schema)
        with PostgresSaver.from_conn_string(conn_string) as saver:
            saver.setup()
            logger.info("Checkpointer: using PostgresSaver")
            yield saver
        return

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ---------------------------------------------------------------------------
# 동기 singleton
# ---------------------------------------------------------------------------

_checkpointer: Checkpointer | None = None
_checkpointer_ctx = None  # connection을 살려두는 열린 context manager
_checkpointer_lock = threading.Lock()


def get_checkpointer() -> Checkpointer:
    """전역 동기 checkpointer singleton을 반환하며, 첫 호출 시 생성한다.

    레거시 ``checkpointer`` 섹션이 설정돼 있으면 그것이 우선이고, 아니면 통합
    ``database`` 섹션이 backend를 고른다. 둘 다 영속 backend를 고르지 않으면
    ``InMemorySaver``를 반환한다.

    Raises:
        ImportError: 설정된 backend에 필요한 패키지가 설치돼 있지 않은 경우.
        ValueError: 해당 backend에 필요한 ``connection_string``이 없는 경우.
    """
    global _checkpointer, _checkpointer_ctx

    if _checkpointer is not None:
        return _checkpointer

    # config 로딩은 두 persistence singleton을 모두 리셋할 수 있다. provider 간 lock 순서
    # 역전을 피하기 위해 전체 config는 이 provider lock 바깥에서 해석한다.
    config = _get_checkpointer_config()

    with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer

        checkpointer_ctx = _sync_checkpointer_cm(config)
        checkpointer = checkpointer_ctx.__enter__()
        _checkpointer_ctx = checkpointer_ctx
        _checkpointer = checkpointer

    return _checkpointer


def reset_checkpointer() -> None:
    """동기 singleton을 리셋해 다음 호출 때 다시 만들게 한다.

    열려 있는 backend connection을 닫고 캐시된 인스턴스를 비운다. 테스트나 설정 변경
    직후에 유용하다.
    """
    global _checkpointer, _checkpointer_ctx
    with _checkpointer_lock:
        if _checkpointer_ctx is not None:
            try:
                _checkpointer_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("Error during checkpointer cleanup", exc_info=True)
            _checkpointer_ctx = None
        _checkpointer = None


# ---------------------------------------------------------------------------
# 동기 context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def checkpointer_context() -> Iterator[Checkpointer]:
    """checkpointer를 yield하고 종료 시 정리하는 동기 context manager.

    :func:`get_checkpointer`와 달리 인스턴스를 캐시하지 **않는다** — 각 ``with`` 블록이
    자기 connection을 만들고 파괴한다. 확정적인 정리가 필요한 CLI 스크립트나 테스트에서
    쓴다::

        with checkpointer_context() as cp:
            graph.invoke(input, config={"configurable": {"thread_id": "1"}})

    레거시 ``checkpointer`` 섹션이 설정돼 있으면 그것이 우선이고, 아니면 통합
    ``database`` 섹션이 backend를 고른다. 둘 다 영속 backend를 고르지 않으면
    ``InMemorySaver``를 yield한다.
    """

    config = _resolve_checkpointer_config(get_app_config())
    with _sync_checkpointer_cm(config) as saver:
        yield saver
