"""SQL 기반 agent store(동기).

``agent_storage.backend: db`` 경로를 담당한다. 의도적으로 동기이며 자체의 작은 engine을
사용한다(store가 동기인 이유는 :mod:`deerflow.persistence.agents.base` 참고). 이 engine은 async
persistence 계층이 관리하는 것과 같은 데이터베이스를 가리킨다 — ``agents`` 테이블은 그 계층의
Alembic bootstrap(migration ``0006``)이 만들고, 이 store는 row를 읽고 쓸 뿐이다.

sqlite(stdlib)와 postgres(psycopg)의 동기 driver 모두 앱에 이미 포함돼 있으므로 의존성이
추가되지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import uuid
from collections.abc import Hashable
from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, delete, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from deerflow.config.agents_config import AgentConfig
from deerflow.config.paths import get_paths
from deerflow.persistence.agents.base import (
    AgentDeleteOutcome,
    AgentExistsError,
    AgentStore,
    parse_agent_config,
)
from deerflow.persistence.agents.model import AgentRow
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)

# 동기 engine을 URL 기준으로 캐시한다. store는 여러 곳(gateway route, graph factory)에서
# 필요할 때마다 생성되므로, 호출마다 연결을 여는 대신 프로세스당 하나의 engine/pool을 재사용해야
# 한다. lock은 같은 URL을 처음 건드리는 두 thread가 중복 engine을 만들고 거기에 connect
# listener를 등록하는 것을 막는다.
_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def _build_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        # async engine의 연결별 PRAGMA를 그대로 맞춘다(persistence/engine.py).
        # journal_mode=WAL은 DB 파일에 영구 반영되지만(async bootstrap이 설정한다), synchronous와
        # busy_timeout은 연결별 설정이다. 이 코드가 없으면 이 동기 연결들은 async engine의
        # NORMAL + 30s가 아니라 synchronous=FULL과 pysqlite 기본값 5s busy_timeout으로 동작한다.
        # 값을 맞춰 두 engine이 공유 DB에 대해 동일하게 동작하고, 동시 writer가 lock 경합에서
        # 일찍 실패하지 않고 최대 30초까지 기다리게 한다.
        @event.listens_for(engine, "connect")
        def _enable_sqlite_pragmas(dbapi_conn, _record):  # noqa: ARG001 — SQLAlchemy 계약
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.execute("PRAGMA busy_timeout=30000;")
            finally:
                cursor.close()

    return engine


def _get_sessionmaker(url: str) -> sessionmaker[Session]:
    engine = _engines.get(url)
    if engine is None:
        with _engines_lock:
            engine = _engines.get(url)
            if engine is None:
                engine = _build_engine(url)
                _engines[url] = engine
    return sessionmaker(engine, expire_on_commit=False)


def _config_document(config: dict) -> dict:
    """저장 문서에서 natural key를 제거한다(``name``은 별도 컬럼이다)."""
    return {k: v for k, v in config.items() if k != "name"}


class SqlAgentStore(AgentStore):
    def __init__(self, url: str) -> None:
        self._Session = _get_sessionmaker(url)

    def _row(self, session: Session, name: str, user_id: str) -> AgentRow | None:
        stmt = select(AgentRow).where(AgentRow.user_id == user_id, AgentRow.name == name.lower())
        return session.execute(stmt).scalar_one_or_none()

    def get(self, name: str, *, user_id: str | None = None) -> AgentConfig:
        effective_user = user_id or get_effective_user_id()
        with self._Session() as session:
            row = self._row(session, name, effective_user)
        if row is None:
            raise FileNotFoundError(f"Agent config not found: {name} (user {effective_user})")
        return parse_agent_config(row.config or {}, row.name)

    def exists(self, name: str, *, user_id: str | None = None) -> bool:
        effective_user = user_id or get_effective_user_id()
        with self._Session() as session:
            return self._row(session, name, effective_user) is not None

    def get_soul(self, name: str, *, user_id: str | None = None) -> str | None:
        effective_user = user_id or get_effective_user_id()
        with self._Session() as session:
            row = self._row(session, name, effective_user)
        if row is None:
            return None
        return row.soul or None

    def list(self, *, user_id: str | None = None) -> list[AgentConfig]:
        effective_user = user_id or get_effective_user_id()
        stmt = select(AgentRow).where(AgentRow.user_id == effective_user).order_by(AgentRow.name.asc())
        with self._Session() as session:
            rows = list(session.execute(stmt).scalars())
        return [parse_agent_config(r.config or {}, r.name) for r in rows]

    def list_all(self) -> list[tuple[str, AgentConfig]]:
        stmt = select(AgentRow).order_by(AgentRow.user_id.asc(), AgentRow.name.asc())
        with self._Session() as session:
            rows = list(session.execute(stmt).scalars())
        return [(r.user_id, parse_agent_config(r.config or {}, r.name)) for r in rows]

    def create(self, name: str, config: dict, soul: str, *, user_id: str | None = None) -> None:
        effective_user = user_id or get_effective_user_id()
        now = datetime.now(UTC)
        row = AgentRow(
            id=uuid.uuid4().hex,
            user_id=effective_user,
            name=name.lower(),
            config=_config_document(config),
            soul=soul or "",
            created_at=now,
            updated_at=now,
        )
        try:
            with self._Session() as session:
                session.add(row)
                session.commit()
        except IntegrityError as e:
            # UNIQUE(user_id, name)이 확인 후 쓰기 경합을 깔끔한 conflict로 바꿔 준다.
            raise AgentExistsError(f"Agent '{name}' already exists for user '{effective_user}'") from e

    def update(self, name: str, config: dict | None, soul: str | None, *, user_id: str | None = None) -> None:
        effective_user = user_id or get_effective_user_id()
        with self._Session() as session:
            row = self._row(session, name, effective_user)
            if row is not None:
                self._apply_update(row, config, soul)
                session.commit()
                return
            # upsert: setup_agent과 모든 최초 쓰기가 여기로 온다. 동시에 실행된 두 최초
            # update(예: setup_agent handshake 두 개)가 모두 row가 None인 것을 보고 둘 다
            # insert할 수 있으며, UNIQUE(user_id, name)이 진 쪽을 거부한다. 날것의
            # IntegrityError가 500으로 드러나게 두지 말고 이긴 쪽 row를 다시 가져와 update를
            # 적용한다 — create()의 conflict 처리와 대칭되는 진짜 upsert다.
            row = AgentRow(
                id=uuid.uuid4().hex,
                user_id=effective_user,
                name=name.lower(),
                config=_config_document(config or {}),
                soul=soul or "",
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = self._row(session, name, effective_user)
                if existing is None:
                    raise
                self._apply_update(existing, config, soul)
                session.commit()

    @staticmethod
    def _apply_update(row: AgentRow, config: dict | None, soul: str | None) -> None:
        if config is not None:
            row.config = _config_document(config)
        if soul is not None:
            row.soul = soul

    def delete(self, name: str, *, user_id: str | None = None) -> AgentDeleteOutcome:
        effective_user = user_id or get_effective_user_id()
        with self._Session() as session:
            result = session.execute(delete(AgentRow).where(AgentRow.user_id == effective_user, AgentRow.name == name.lower()))
            session.commit()
            row_deleted = result.rowcount > 0
        agent_dir = get_paths().user_agent_dir(effective_user, name)
        if row_deleted:
            # agent가 row로 존재했으므로, 같은 위치의 디스크 memory(deermem file backend)를
            # 함께 제거해 고아로 남지 않게 한다. config + soul + memory를 묶어 지우는 file
            # backend의 rmtree와 동일하다.
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            return "deleted"
        # agent row가 없다. 여기 남아 있는 디스크 디렉터리는 memory/facts 데이터만 담고 있으므로
        # (db 모드에서 config는 디스크가 아니라 row에 있다) 사용자의 memory를 지우는 대신
        # 보존한다(#4279) — rmtree하지 않는다.
        if agent_dir.exists():
            return "not-custom-agent"
        return "missing"

    def signature(self) -> Hashable:
        # GitHub registry는 캐시된 agent binding이 아직 최신인지 판단하는 데 이 토큰을 쓴다.
        # 두 번의 쓰기가 같은 데이터베이스 timestamp를 가질 수 있으므로 timestamp만으로는
        # 부족하다. digest 계산은 registry의 캐시 신선도 확인 때만 작은 agents 테이블을 읽는다.
        # agent 수나 webhook 전달량이 늘어 이 스캔이 부담이 되면 다시 검토한다.
        with self._Session() as session:
            rows = session.execute(
                select(
                    AgentRow.user_id,
                    AgentRow.name,
                    AgentRow.config,
                    AgentRow.soul,
                ).order_by(AgentRow.user_id, AgentRow.name)
            ).all()

        payload = [
            {
                "user_id": user_id,
                "name": name,
                "config": config or {},
                "soul": soul or "",
            }
            for user_id, name, config, soul in rows
        ]
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
