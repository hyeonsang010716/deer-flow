"""custom agent 정의의 persistence. 추상 store와 file/db backend를 제공한다.

공개 진입점은 :func:`get_agent_store`이며, :mod:`deerflow.config.agents_config`의 자유 함수들이
여기로 dispatch한다. ``file``(기본값)은 현재의 디스크 레이아웃을 유지하고, ``db``는 SQL
persistence 레이어를 통해 노드 간에 정의를 공유한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deerflow.persistence.agents.base import (
    AgentDeleteOutcome,
    AgentExistsError,
    AgentStore,
    parse_agent_config,
)
from deerflow.persistence.agents.model import AgentRow

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

__all__ = [
    "AgentDeleteOutcome",
    "AgentExistsError",
    "AgentRow",
    "AgentStore",
    "get_agent_store",
    "make_agent_store",
    "parse_agent_config",
]

_file_store_singleton: AgentStore | None = None


def make_agent_store(config: AppConfig) -> AgentStore:
    """``config.agent_storage.backend``가 선택한 store를 만들거나 재사용한다.

    ``db``는 ``database.backend``가 ``sqlite`` 또는 ``postgres``여야 한다. ``memory``
    데이터베이스는 durable URL이 없어서 여기서 거부한다(gateway도 시작 시 즉시 실패하지만,
    이 guard는 graph 프로세스 경로를 커버한다).
    """
    if config.agent_storage.backend == "db":
        db_backend = config.database.backend
        if db_backend not in ("sqlite", "postgres"):
            raise ValueError(
                f"agent_storage.backend='db' requires database.backend to be 'sqlite' or 'postgres', "
                f"but database.backend is '{db_backend}'. A 'memory' database is per-process and cannot "
                "share agent definitions across nodes; set database.backend accordingly or use "
                "agent_storage.backend='file'."
            )
        from deerflow.persistence.agents.sql import SqlAgentStore

        return SqlAgentStore(config.database.app_sync_sqlalchemy_url)

    return _file_store()


def get_agent_store() -> AgentStore:
    """현재 프로세스 설정에 맞는 store를 반환한다.

    app config를 해석할 수 없으면 file backend를 기본으로 쓴다. ``agents_config``의 자유
    함수들은 전체 ``config.yaml``을 로드하지 않는 가벼운 환경(CLI, 테스트, 도구)에서도 계속
    동작해야 하기 때문이다. 명시적인 ``agent_storage.backend: db``만 file 기본값에서 벗어난다.

    프로세스 간 불변식(``db`` backend의 존재 이유): run별 agent 빌드는 gateway와 다른
    프로세스인 **graph subprocess**에서 실행된다. 노드 간 보장이 성립하는 것은 오직 그곳에서도
    ``get_app_config()``가 ``config.yaml``을 해석해 ``backend: db``를 반환하기 때문이다.
    그래야 graph 프로세스의 읽기 경로가 gateway가 쓴 것과 같은 공유 테이블을 본다. 아래
    ``except``는 진짜로 *해석 가능한 config가 없는* 경우의 fallback(CLI/테스트)이지,
    잘못 설정된 graph 프로세스를 가리는 장치가 **아니다**. 그곳에서 ``config.yaml``에 접근할 수
    있다면(같은 작업 트리이므로 접근 가능하다) ``db``가 그대로 적용되며, 노드 로컬 ``file``로
    조용히 낮아지지 않는다. ``test_get_agent_store_resolves_db_backend_from_on_disk_config``가
    이를 고정한다.
    """
    from deerflow.config.app_config import get_app_config

    try:
        config = get_app_config()
    except Exception:  # noqa: BLE001 — 해석 가능한 config 없음 → file 기본값
        return _file_store()
    return make_agent_store(config)


def _file_store() -> AgentStore:
    global _file_store_singleton
    if _file_store_singleton is None:
        from deerflow.persistence.agents.file import FileAgentStore

        _file_store_singleton = FileAgentStore()
    return _file_store_singleton
