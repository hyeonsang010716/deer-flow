"""DeerFlow 애플리케이션 테이블용 하이브리드 schema bootstrap.

Gateway 시작 시 무조건 실행하던 ``Base.metadata.create_all``을 대체한다.
두 가지 아이디어를 결합한다:

1. ``create_all``은 빈 DB용 빠른 경로로 남는다 -- 누구도 baseline 사본을 모델과
   손으로 맞춰둘 필요 없이 SQLite와 Postgres dialect 양쪽(JSON vs JSONB, server
   default, index/FK 이름, 타입 affinity)에 ``Base.metadata``를 충실히 렌더링한다.
2. **baseline 이후의 모든 변경은 Alembic이 소유한다.** 새 ORM 컬럼 / 테이블 /
   index는 반드시 ``migrations/versions/`` 아래의 revision으로 나와야 한다.

세 갈래 판정 (``_decide_state`` 참고)
---------------------------------------------

| DB 상태                                | 동작                                    |
|---------------------------------------|-----------------------------------------|
| empty (DeerFlow 테이블 없음)           | ``create_all`` + ``alembic stamp head`` |
| legacy (DeerFlow 테이블 있고 alembic 없음) | ``create_all`` (baseline 테이블만 backfill) + ``stamp 0001_baseline`` + ``upgrade head`` |
| versioned (``alembic_version`` row 존재) | ``alembic upgrade head``                |

legacy 분기는 DeerFlow 소유 테이블이 이미 하나 이상 있는 alembic 이전 데이터베이스를
다룬다. ``create_all``을 먼저 돌리는 이유는, ``0001_baseline``에 stamp하면 이후 upgrade
에서 alembic이 baseline 자신의 ``create_table`` DDL을 건너뛰기 때문이다 -- 즉 사용자의
DB가 처음 프로비저닝된 뒤 ``Base.metadata``에 추가된 baseline 테이블(예: 여러 릴리스를
건너뛰어 업그레이드하는 사용자의 PR #1930발 ``channel_*`` 테이블)은 그대로 두면 영영
생성되지 않고, 그 테이블을 건드리는 첫 요청이 ``no such table``로 500이 난다. 이
backfill은 **``_BASELINE_TABLE_NAMES``로 제한**되어 있어 미래 revision이 도입하는
테이블까지 만들지는 않는다 -- 그랬다면 해당 revision의 ``op.create_table``이
``relation already exists``로 실패한다. guard 테스트가 이 제한 집합을
``0001_baseline.upgrade()``의 실제 출력과 대조해 고정한다.

컬럼 수준의 형태(``token_usage_by_model``의 #3658 이전 / 이후 / 수동 ALTER 케이스)는
``migrations/_helpers.py``의 멱등 헬퍼를 통해 각 ``versions/*.py`` revision이 스스로
답한다(``safe_add_column``은 컬럼이 이미 있으면 아무 일도 하지 않고, 형태가 어긋나면
``logger.warning``을 남긴다). 따라서 앞으로의 schema 추가는 새 revision 파일을 쓰는
것으로 끝나며 -- 새 revision이 새 baseline 테이블을 만드는 경우가 아니라면
**이 모듈은 수정할 필요가 없다**. 그 경우에는 ``_BASELINE_TABLE_NAMES``를 맞춰
갱신해야 한다(안 그러면 guard 테스트가 터진다).

동시성 안전성
------------------

계층적으로 보장하되 backend마다 보장 수준이 다르다. Postgres는 진짜 프로세스 간
직렬화를 제공한다. SQLite는 단일 프로세스에서는 안전하고 프로세스 간에는 best-effort다.
다중 인스턴스 배포는 Postgres를 써야 한다.

* **Postgres -- 진짜 프로세스 간 직렬화.** ``pg_advisory_lock``이 reflect 후 실행하는
  전체 시퀀스를 프로세스 경계를 넘어 유지되는 배타 lock 아래에서 돌린다. 동시에 뜬
  Gateway 인스턴스들은 깔끔하게 줄을 서고, 두 번째 인스턴스는 head를 관측해 no-op이 된다.

* **SQLite -- 단일 프로세스 직렬화, 프로세스 간은 best-effort.**
  SQLite는 배포상 단일 노드이므로 현실적인 동시성은 하나의 Gateway 프로세스 안의 여러
  async task(테스트, lifespan 재진입)다. engine별 ``asyncio.Lock``이 이를 직렬화한다.
  드문 프로세스 간 케이스(예: 같은 DB 파일에 붙은 두 개의 ``make dev`` worker)는
  SQLite 자체의 파일 수준 쓰기 lock과 30초 ``PRAGMA busy_timeout``에 의존한다 --
  후자는 production engine(``persistence/engine.py``)과 alembic이 띄우는
  engine(``migrations/env.py``) **양쪽**에 설정되어 있어, 어느 writer든 즉시 실패하는
  대신 파일 lock을 최대 30초 기다린다. 이는 best-effort이지 진짜 mutex가 아니다.
  병적으로 겹치면 30초 뒤에도 ``database is locked``를 볼 수 있다. 마지막 방어선인
  멱등 revision이 어차피 정확성을 보장한다.

* **멱등 revision -- 재시도 fallback.** 컬럼 revision은 ``migrations/_helpers.py``의
  헬퍼를 쓰므로 baseline 이후의 반복 변경, 수동 ALTER, SQLite lock 경합 후 재시도가
  작업을 중복 수행하지 않는다.

이미 head인 DB에 대한 ``alembic upgrade head``는 alembic 자체 의미상 no-op이므로,
두 번째 이후의 주체는 head를 관측하고 그냥 종료한다.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


# 이 파일 기준으로 alembic 환경이 있는 위치.
_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# 캐시된 migration head. 디스크의 script 트리에서 프로세스당 한 번 계산한다.
_HEAD_REVISION: str | None = None

# baseline(legacy DB의 stamp 대상). 여기에 고정해두어, baseline revision 이름이 stamp
# 호출을 갱신하지 않은 채 바뀌면 bootstrap 계층이 크게 실패하도록 한다.
# ``tests/test_persistence_bootstrap.py``가 이 문자열이 script 트리에 실재하는
# revision id인지 검증한다.
_BASELINE_REVISION = "0001_baseline"

# Postgres용 고정 advisory lock 키. 다른 애플리케이션의 advisory lock과 절대 충돌하지
# 않도록 32비트 난수 두 개를 한 번 골라 쓴다. 일회성 migration을 함께 준비하지 않고는
# 바꾸지 말 것(키를 바꾸면 사실상 이전 lock을 놓아버리는 셈이다).
_PG_LOCK_KEY = 0x0DEE_12F1_0BEE_3682


# ``0001_baseline.upgrade()``가 만드는 테이블들. legacy 분기는 ``create_all`` backfill을
# 이 집합으로 제한해서, baseline 이후 추가된 모델의 ``op.create_table`` revision을
# 미리 가로채지 않게 한다 -- ``create_all``이 먼저 그 테이블을 만들었다면 해당
# revision이 ``relation already exists``로 실패한다. (컬럼 revision은
# ``migrations/_helpers.py``의 멱등 헬퍼로 이미 안전하다. 아직 대응하는
# ``safe_create_table``은 없으므로, 테이블 수준 안전성은 모든 미래 revision에
# 떠넘기는 대신 이 계층에서 유지한다.)
#
# ``test_baseline_table_names_constant_matches_0001``이 이 집합을 0001이 실제로 만드는
# 것과 대조해 고정한다 -- 이 상수를 갱신하지 않고 0001을 수정하면(혹은 그 반대) 그
# 테스트가 터진다.
_BASELINE_TABLE_NAMES: frozenset[str] = frozenset(
    {
        "channel_connections",
        "channel_conversations",
        "channel_credentials",
        "channel_oauth_states",
        "feedback",
        "run_events",
        "runs",
        "threads_meta",
        "users",
    }
)

# ``test_baseline_index_names_constant_matches_0001``이 이 집합을 0001이 실제로 만드는
# 것과 대조해 고정한다 -- 이 상수를 갱신하지 않고 0001을 수정하면(혹은 그 반대) 그
# 테스트가 터진다.
_BASELINE_INDEX_NAMES: frozenset[str] = frozenset(
    {
        # channel_connections
        "idx_channel_connections_event_lookup",
        "ix_channel_connections_owner_user_id",
        "ix_channel_connections_provider",
        "uq_channel_connection_active_identity",
        # channel_conversations
        "ix_channel_conversations_connection_id",
        "ix_channel_conversations_owner_user_id",
        "ix_channel_conversations_provider",
        "ix_channel_conversations_thread_id",
        # channel_oauth_states
        "ix_channel_oauth_states_owner_user_id",
        "ix_channel_oauth_states_provider",
        # feedback
        "ix_feedback_run_id",
        "ix_feedback_thread_id",
        "ix_feedback_user_id",
        # run_events
        "ix_events_run",
        "ix_events_thread_cat_seq",
        "ix_run_events_user_id",
        # runs
        "ix_runs_thread_id",
        "ix_runs_thread_status",
        "ix_runs_user_id",
        # threads_meta
        "ix_threads_meta_assistant_id",
        "ix_threads_meta_user_id",
        # users
        "idx_users_oauth_identity",
        "ix_users_email",
    }
)


# engine별 SQLite bootstrap lock. 모듈 전역이 아니라 engine별인 이유는 각 engine
# 인스턴스가 그 engine을 쓰는 event loop에 묶인 lock과 짝을 이뤄야 하기 때문이다 --
# ``asyncio.Lock``은 처음 본 loop에 바인딩되고 pytest는 async 테스트마다 자체 loop를
# 주므로 필요하다. production은 프로세스당 engine 하나를 쓰므로 실제로 이 dict는
# 항목 하나로 수렴한다.
#
# ``id(engine)``이 아니라 ``WeakKeyDictionary``로 engine 객체 자체를 키로 쓴다.
# CPython은 GC 후 주소를 재활용하므로, 죽은 engine의 낡은 ``id`` → ``Lock`` 항목이
# 우연히 같은 주소에 자리 잡은 새 engine에 반환될 수 있다. 그 lock은 여전히 죽은
# engine의 event loop에 묶여 있어 ``async with``에서
# ``RuntimeError: ... bound to a different event loop``가 난다. engine 자체를 해싱하면
# engine이 수거될 때 항목도 자동으로 사라지므로, 이 dict는 살아 있는 engine 수를
# 넘어 커지지 않는다.
_SQLITE_LOCKS: weakref.WeakKeyDictionary[AsyncEngine, asyncio.Lock] = weakref.WeakKeyDictionary()


def _get_sqlite_local_lock(engine: AsyncEngine) -> asyncio.Lock:
    lock = _SQLITE_LOCKS.get(engine)
    if lock is None:
        lock = asyncio.Lock()
        _SQLITE_LOCKS[engine] = lock
    return lock


def _escape_url_for_alembic(url: str) -> str:
    """리터럴 ``%``를 두 번 써서 ``ConfigParser`` interpolation이 URL을 건드리지 않게 한다.

    ``alembic.config.Config.set_main_option``은 ``ConfigParser.set``으로 넘기고, 이는
    값에 ``%(name)s`` 스타일 interpolation을 수행한다. 그대로 두면 ``p%40ss``처럼
    URL 인코딩된 비밀번호(``@``가 ``%40``으로 이스케이프됨)가
    ``InterpolationSyntaxError``를 일으킨다. 모든 리터럴 ``%``를 두 번 쓰면
    ConfigParser가 다시 하나로 되돌린다. round-trip 규칙이 한 곳에 있도록
    ``scripts/_autogen_revision.py``와 공유한다.
    """
    return url.replace("%", "%%")


def _alembic_safe_url(engine: AsyncEngine) -> str:
    """*engine*의 URL을 alembic ``set_main_option``이 받아들이는 형태로 렌더링한다.

    두 가지 함정을 처리한다:

    1. ``str(engine.url)``(및 인자 없는 ``URL.render_as_string()``)은 비밀번호를
       ``***``로 가린다 -- 그러면 실제 engine은 멀쩡히 접속하는데도 alembic의
       stamp/upgrade는 쓰레기 자격 증명으로 자기 connection을 열어 runtime에서
       실패한다. 해결: ``render_as_string(hide_password=False)``.
    2. ``%``에 대한 ConfigParser interpolation -- 규칙을 autogen 스크립트와 공유하도록
       ``_escape_url_for_alembic``에 위임한다.
    """
    rendered = engine.url.render_as_string(hide_password=False)
    return _escape_url_for_alembic(rendered)


def _get_alembic_config(engine: AsyncEngine, *, postgres_schema: str = "") -> AlembicConfig:
    """migrations 디렉터리를 가리키는 in-process alembic config를 만든다.

    production runtime이 작업 디렉터리 상대 경로 조회에 의존하지 않도록 디스크에서
    ``alembic.ini``를 읽지 않는다. ``script_location``은 디스크상의 패키지 경로에
    고정한다.

    *postgres_schema*가 설정되면 ``deerflow_pg_schema`` main option으로 전달해서,
    ``env.py``가 alembic이 띄우는 engine의 ``search_path``를 app engine이 쓰는 것과
    같은 schema로 고정할 수 있게 한다. 이것이 없으면 alembic 자체 engine은 -- 맨
    URL로 만들어지므로 -- app 테이블이 커스텀 schema에 생기는 동안
    ``alembic_version``과 모든 migration DDL을 기본(``public``) schema에 만든다.
    """
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", _alembic_safe_url(engine))
    if postgres_schema:
        cfg.set_main_option("deerflow_pg_schema", postgres_schema)
    return cfg


def _get_head_revision() -> str:
    """``versions/``의 head revision id를 반환한다. 프로세스당 캐시된다."""
    global _HEAD_REVISION
    if _HEAD_REVISION is None:
        cfg = AlembicConfig()
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        script = ScriptDirectory.from_config(cfg)
        head = script.get_current_head()
        if head is None:
            raise RuntimeError("alembic has no head revision -- versions/ directory is empty")
        _HEAD_REVISION = head
    return _HEAD_REVISION


def _reflect_state(sync_conn: Any) -> dict[str, bool]:
    """*sync_conn*(``run_sync`` 안의 sync connection)을 조사해 다음을 반환한다:

    - ``has_alembic_version``: bool
    - ``has_deerflow_tables``: ``Base.metadata``가 아는 테이블이 DB에 하나라도 있으면
      True. ``reflected ∩ metadata``로 계산하므로 bootstrap 계층은 특정 테이블이나
      컬럼 이름을 하드코딩하지 않는다 -- 새 ORM 모델을 추가해도 바뀌는 것은
      ``Base.metadata``뿐이고 이 모듈은 그대로다.
    """
    from deerflow.persistence.base import Base

    # 모든 ORM 모델이 import되었는지 확인한다. 그러지 않으면 아직 import되지 않은
    # 서브모듈이 등록하는 테이블을 ``Base.metadata.tables``가 놓칠 수 있다.
    try:
        import deerflow.persistence.models  # noqa: F401
    except ImportError:
        logger.debug("deerflow.persistence.models not found; metadata may be incomplete")

    insp = sa_inspect(sync_conn)
    reflected = set(insp.get_table_names())
    metadata_tables = set(Base.metadata.tables)
    return {
        "has_alembic_version": "alembic_version" in reflected,
        "has_deerflow_tables": bool(reflected & metadata_tables),
    }


def _decide_state(state: dict[str, bool]) -> str:
    """reflect한 DB 상태를 세 갈래 라벨 중 하나로 매핑한다.

    legacy 분기는 alembic 이전의 모든 DB를 동일하게 다룬다 -- 이후 revision이 추가한
    컬럼이 있는지 없는지는 각 revision이 ``migrations/_helpers.py``의 멱등 헬퍼로
    스스로 답할 문제다.
    """
    if state["has_alembic_version"]:
        return "versioned"
    if not state["has_deerflow_tables"]:
        # 완전히 새 DB이거나, 우리가 소유하지 않은 테이블만 있는 DB다(예: 새로 배포한
        # 환경의 LangGraph checkpointer 테이블). empty 분기는 alembic이 소유하는
        # 테이블을 프로비저닝한 뒤 head를 stamp한다.
        return "empty"
    return "legacy"


def _run_create_all_sync(sync_conn: Any) -> None:
    """*sync_conn*에 DeerFlow 소유 테이블을 모두 생성한다."""
    # 모든 모델 클래스가 Base.metadata에 등록되도록 여기서 import한다.
    from deerflow.persistence.base import Base

    try:
        import deerflow.persistence.models  # noqa: F401
    except ImportError:
        logger.debug("deerflow.persistence.models not found; bootstrap will create empty schema")

    Base.metadata.create_all(sync_conn)


def _run_baseline_create_all_sync(sync_conn: Any) -> None:
    """*sync_conn*에 baseline 테이블만 생성한다(checkfirst로 멱등).

    legacy 분기가 사용자 DB에 빠진 baseline 시절 테이블을 backfill할 때 쓴다. 테이블
    목록을 ``_BASELINE_TABLE_NAMES``로 제한하는 것이 안전성의 핵심이다. 제한 없는
    ``create_all``은 이후 revision이 도입한 테이블까지 만들고, 그러면 alembic이
    upgrade를 돌릴 때 해당 revision의 ``op.create_table`` 호출과 충돌한다.
    """
    from deerflow.persistence.base import Base

    try:
        import deerflow.persistence.models  # noqa: F401
    except ImportError:
        logger.debug("deerflow.persistence.models not found; baseline backfill may be incomplete")

    baseline_tables = [Base.metadata.tables[name] for name in _BASELINE_TABLE_NAMES if name in Base.metadata.tables]
    Base.metadata.create_all(sync_conn, tables=baseline_tables, checkfirst=True)

    # ``checkfirst=True``인 ``create_all``은 테이블이 이미 있으면 그 테이블과 하위
    # ``Index`` 객체를 전부 건너뛴다. 따라서 테이블이 처음 프로비저닝된 뒤 ORM 모델에
    # 추가된 index는 영영 생성되지 않고, legacy 분기는 upgrade 전에
    # ``0001_baseline``을 stamp하므로 baseline 시절 index에 대한 alembic 자체의
    # ``batch_op.create_index``도 건너뛴다. 모든 baseline 테이블의 baseline 시절
    # ``Index``를 각각 ``checkfirst=True``로 명시적으로 생성하면, 부모 테이블이 방금
    # 만들어졌든 이미 있었든 상관없이 각 index의 존재가 보장된다.
    #
    # **범위**: ``_BASELINE_INDEX_NAMES``에 있는 index만 생성한다.
    # ``table.indexes``는 *현재* ORM 모델의 전체 index 집합이라, 이후 revision이 추가한
    # baseline 이후 index(예: 0004의 ``uq_runs_thread_active``)도 포함한다. 그것들을
    # 미리 만들면 소유 revision의 데이터 전제조건(dedup 단계, 컬럼 migration)과
    # 충돌해 legacy DB에서 ``IntegrityError``가 난다.
    #
    # baseline 테이블에 index를 추가하는 baseline 이후 revision은 기존
    # ``sa.inspect(bind).get_indexes(...)`` + ``if name not in existing`` guard 패턴을
    # 쓰거나(0004_run_ownership.py:99-103 참고), ``safe_add_column``에 대응하는 미래의
    # ``safe_create_index`` 헬퍼를 써야 한다.
    for table in baseline_tables:
        for idx in table.indexes:
            if idx.name not in _BASELINE_INDEX_NAMES:
                continue
            try:
                idx.create(sync_conn, checkfirst=True)
            except Exception:
                logger.warning(
                    "bootstrap: failed to create baseline index %r on %r -- the DB may contain rows that violate the index constraint. Address the duplicate data, then re-run bootstrap.",
                    idx.name,
                    table.name,
                )


def _stamp(cfg: AlembicConfig, revision: str) -> None:
    """동기 alembic stamp. 호출자는 ``asyncio.to_thread``로 감싸야 한다."""
    alembic_command.stamp(cfg, revision)


def _upgrade(cfg: AlembicConfig, revision: str) -> None:
    """동기 alembic upgrade. 호출자는 ``asyncio.to_thread``로 감싸야 한다."""
    alembic_command.upgrade(cfg, revision)


# ---------------------------------------------------------------------------
# 프로세스 간 locking
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _postgres_lock(engine: AsyncEngine):
    """블록 본문 동안 Postgres session 수준 advisory lock을 유지한다.

    transaction 수준이 아니라 session 수준이므로, alembic이 ``stamp`` / ``upgrade``
    중에 여는 암시적 transaction보다 lock이 오래 산다. lock은 빠져나갈 때 명시적으로
    해제하며, 안전망으로 뒷단 session이 끊길 때(프로세스 크래시, kill -9)도 해제된다.

    idle-in-transaction 보호
    ------------------------------

    ``engine.connect()``는 첫 ``execute``에서 transaction을 자동으로 시작하고, 이
    connection은 ``asyncio.to_thread(_upgrade, ...)``가 *다른* pooled connection에서
    alembic을 돌리는 동안 유휴 상태로 남는다. 관리형 Postgres(RDS, Cloud SQL,
    Supabase)는 기본적으로 ``idle_in_transaction_session_timeout``이 1-10분으로
    설정되어 있다. alembic이 그보다 오래 걸리면 호스트가 이 idle-in-transaction
    session을 죽이고, advisory lock은 session 범위이므로 lock이 **조용히 해제된다.**
    그러면 두 번째 Gateway가 lock을 획득해 첫 번째와 동시에 DDL을 돌리게 되어 lock의
    목적 자체가 무의미해진다.

    방어책: ``SET LOCAL idle_in_transaction_session_timeout = 0``으로 **이 transaction
    에 한해서만** 강제 종료를 끈다(전역/role 수준 영향 없음). 자체 호스팅 Postgres는
    보통 이 timeout이 꺼져 있어 no-op이고, 관리형 PG에서는 DDL이 도는 동안 lock을
    살려두는 역할을 한다. 경합이 심한 클러스터에서 lock 획득이 느린 경우도 보호받도록
    반드시 ``pg_advisory_lock`` *이전*에 실행해야 한다.
    """
    async with engine.connect() as conn:
        await conn.execute(text("SET LOCAL idle_in_transaction_session_timeout = 0"))
        await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _PG_LOCK_KEY})
        try:
            logger.info("bootstrap: acquired postgres advisory lock key=0x%x", _PG_LOCK_KEY)
            yield
        finally:
            try:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _PG_LOCK_KEY})
            except Exception:  # noqa: BLE001
                logger.warning("bootstrap: pg_advisory_unlock raised; session close will release", exc_info=True)


@asynccontextmanager
async def _sqlite_lock(engine: AsyncEngine):
    """SQLite bootstrap을 한 프로세스 안에서 직렬화한다. 프로세스 간에는 SQLite 자체
    파일 lock과 ``PRAGMA busy_timeout``으로 best-effort 처리한다.

    sentinel connection에 ``BEGIN IMMEDIATE``를 쓰지 않는 이유는? SQLite는 파일당
    writer가 하나다. 한 connection에서 쓰기 lock을 잡고 있으면, alembic 자신의
    connection(``stamp`` / ``upgrade`` 안에서 열린다)이 우리와 deadlock에 빠진다.

    프로세스 간 OS 파일 lock을 쓰지 않는 이유는? 동작은 하지만, DeerFlow에서 이미
    권장하지 않는 배포 형태(다중 프로세스 SQLite)를 위해 플랫폼별 ``fcntl`` /
    ``msvcrt`` 호출에 강한 의존을 추가하게 된다. 30초 ``busy_timeout``과 멱등
    revision이 현실적인 경우를 덮으며, 진짜 다중 인스턴스 배포는 Postgres를 써야 한다.

    참고: 30초 ``busy_timeout``은 ``persistence/engine.py``(production)와
    ``migrations/env.py``(alembic이 띄우는 쪽)의 engine 이벤트 hook이 설정한다. 이
    함수는 전파되지도 않을 probe connection에 PRAGMA를 거는 대신, 그 PRAGMA들이 이미
    자리 잡고 있다는 사실에 의존한다.
    """
    async with _get_sqlite_local_lock(engine):
        logger.info("bootstrap: acquired sqlite in-process lock")
        yield


def _bootstrap_lock(engine: AsyncEngine, *, backend: str):
    if backend == "postgres":
        return _postgres_lock(engine)
    if backend == "sqlite":
        return _sqlite_lock(engine)
    raise ValueError(f"bootstrap: unsupported backend {backend!r}")


# ---------------------------------------------------------------------------
# 최상위 진입점
# ---------------------------------------------------------------------------


async def bootstrap_schema(engine: AsyncEngine, *, backend: str, postgres_schema: str = "") -> None:
    """DB schema를 head까지 올린다.

    Postgres 호출은 advisory lock으로 프로세스 간 직렬화된다. SQLite 호출은 한 프로세스
    안에서 직렬화되고, 프로세스 간에는 SQLite의 파일 lock과 ``busy_timeout``으로
    best-effort 처리된다.

    분기 dispatch는 모듈 상단에 문서화되어 있다. ``alembic.command.stamp``와
    ``alembic.command.upgrade``는 동기이고 event loop를 막으므로 둘 다
    ``asyncio.to_thread``로 감싼다.

    *postgres_schema*가 설정되면 alembic config로 전달해서, alembic이 띄우는 engine이
    ``search_path``를 그 schema로 고정하게 한다. 대상 schema는 이미 존재해야 한다
    (``init_engine``이 이 함수를 호출하기 전에 ``CREATE SCHEMA``를 실행한다).
    postgres가 아닌 backend에서는 무시된다.
    """
    head = _get_head_revision()
    cfg = _get_alembic_config(engine, postgres_schema=postgres_schema if backend == "postgres" else "")

    async with _bootstrap_lock(engine, backend=backend):
        async with engine.connect() as conn:
            state = await conn.run_sync(_reflect_state)
        decision = _decide_state(state)

        if decision == "empty":
            logger.info("bootstrap: branch=empty -> create_all + stamp head (%s)", head)
            async with engine.begin() as conn:
                await conn.run_sync(_run_create_all_sync)
            await asyncio.to_thread(_stamp, cfg, head)

        elif decision == "legacy":
            logger.info(
                "bootstrap: branch=legacy -> create_all (backfill missing baseline tables) + stamp %s + upgrade head (%s)",
                _BASELINE_REVISION,
                head,
            )
            # ``_run_baseline_create_all_sync``는 ``_BASELINE_TABLE_NAMES``로
            # 제한된다 -- 평범한 ``Base.metadata.create_all``은 이후 revision이
            # 도입한 테이블까지 만들어, 다음 upgrade에서 그 revision의
            # ``op.create_table``과 충돌한다. 제한을 두면 빠진 baseline 테이블만
            # backfill되고, baseline 이후의 ``create_table`` revision은 자기 테이블이
            # 정말로 아직 없는 DB를 상대로 실행된다. create_all 이후의 컬럼 추가
            # revision은 baseline 시절 테이블이 이제 그 컬럼을 갖고 있으므로
            # ``safe_add_column``을 통해 여전히 no-op이 된다.
            async with engine.begin() as conn:
                await conn.run_sync(_run_baseline_create_all_sync)
            await asyncio.to_thread(_stamp, cfg, _BASELINE_REVISION)
            await asyncio.to_thread(_upgrade, cfg, "head")

        elif decision == "versioned":
            logger.info("bootstrap: branch=versioned -> upgrade head (%s)", head)
            await asyncio.to_thread(_upgrade, cfg, "head")

        else:  # pragma: no cover -- 방어적 처리
            raise RuntimeError(f"bootstrap: unhandled decision {decision!r}")

    logger.info("bootstrap: complete (backend=%s)", backend)
