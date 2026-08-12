"""통합 database backend 설정.

LangGraph checkpointer와 DeerFlow 애플리케이션 persistence layer(runs, threads
metadata, users 등)를 **둘 다** 제어한다. 사용자는 backend 하나만 설정하고, 물리적
분리 세부사항은 시스템이 처리한다.

SQLite 모드: checkpointer와 app이 .db 파일 하나({sqlite_dir}/deerflow.db)를 공유하며
모든 connection에서 WAL journal 모드를 켠다. WAL은 동시 reader와 단일 writer가 서로
막지 않게 해주므로 두 워크로드가 파일 하나를 함께 써도 안전하다. lock을 두고 경쟁하는
writer는 즉시 실패하지 않고 sqlite3 기본 5초 busy timeout만큼 기다린다.

Postgres 모드: 둘 다 같은 database URL을 쓰지만 라이프사이클이 다른 독립 connection
pool을 유지한다.

Memory 모드: checkpointer는 MemorySaver를, app은 in-memory store를 쓴다. database는
초기화하지 않는다.

민감한 값(postgres_url)은 config.yaml에서 $VAR 문법으로 .env의 환경 변수를 참조해야
한다:

    database:
      backend: postgres
      postgres_url: $DATABASE_URL

$VAR 해석은 이 config가 만들어지기 전에 AppConfig.resolve_env_variables()가 처리하므로
DatabaseConfig 자체는 환경 변수 처리를 할 필요가 없다.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deerflow.config.postgres_schema import POSTGRES_SCHEMA_PATTERN, validate_postgres_schema

logger = logging.getLogger(__name__)


def resolve_checkpoint_graph_cache_max(database_config: Any, field_name: str, default: int) -> int:
    """database config 비슷한 객체에서 graph-cache 상한을 읽는다.

    stub config(테스트의 SimpleNamespace/MagicMock, 섹션 누락)도 허용한다: 1 이상의
    순수 int가 아니면 전부 ``default``로 fallback한다.
    """
    section = getattr(database_config, "checkpoint_graph_cache", None)
    value = getattr(section, field_name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


CheckpointChannelMode = Literal["full", "delta"]

DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY = 10


class CheckpointDeltaConfig(BaseModel):
    """``checkpoint_channel_mode: delta``용 튜닝 값.

    ``full`` 모드에서는 무시한다. 모드 자체와 마찬가지로 이 값들은 재시작이 필요하고,
    하나의 checkpoint database를 공유하는 모든 프로세스에서 일치해야 한다: snapshot
    주기는 checkpoint에 저장되지 않고 컴파일된 graph의 channel table에 박히므로, 값이
    다른 프로세스는 같은 thread에 다른 주기를 적용하게 된다.
    """

    snapshot_frequency: int = Field(
        default=DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY,
        ge=1,
        description=(
            "DeltaChannel snapshot cadence: a full messages snapshot is stored "
            "every N per-step writes (higher = smaller checkpoints, slower "
            "materialization). Restart is required, and all processes sharing "
            "one checkpoint database must use the same value."
        ),
    )


class CheckpointGraphCacheConfig(BaseModel):
    """프로세스 로컬 컴파일 checkpoint graph cache의 크기 상한.

    모드나 snapshot 주기와 달리 재시작이 필요하지 않다: 상한을 키우거나 줄여도 graph
    의미는 그대로이고 cache가 언제 eviction하는지만 달라지므로, hot-reload된 값은 다음
    eviction 검사에서 바로 적용된다. cache 키는 (assistant, mode, cadence, app_config)
    이고 상한에 도달하면 통째로 비운다.
    """

    accessor_graph_max: int = Field(
        default=64,
        ge=1,
        description=("Max compiled thread-state accessor graphs cached by the gateway (keyed per assistant, channel mode, and snapshot cadence)."),
    )


class CheckpointCacheConfig(BaseModel):
    """delta history cache 정책. 성능 전용이라 고정되지도 않고, 하나의 checkpoint
    database를 공유하는 프로세스 간에 일치할 필요도 없다.

    ``checkpoint_channel_mode``가 ``delta``일 때만 적용된다. ``max_entries``는 프로세스
    로컬 memory backend의 상한이고 ``0``이면 cache를 완전히 끈다. redis backend는
    ``ttl_seconds``와 서버 자체의 maxmemory 정책으로 제한된다.
    """

    type: Literal["memory", "redis"] = Field(
        default="memory",
        description=("Checkpoint history cache backend. 'memory' = process-local LRU; 'redis' = shared cache for multi-worker deployments (async/Gateway path only; the sync embedded path rejects it)."),
    )
    max_entries: int = Field(
        default=128,
        ge=0,
        description="LRU capacity of the memory backend. 0 disables the cache.",
    )
    redis_url: str | None = Field(
        default=None,
        description=("Redis URL for type=redis. If omitted, DEER_FLOW_CHECKPOINT_CACHE_REDIS_URL, REDIS_URL, or redis://localhost:6379/0 is used."),
    )
    ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description=(
            "Redis entry TTL; a leak safety net, not a correctness mechanism (entries are immutable). "
            "Thread deletion purges that thread's entries immediately; if the purge fails (redis outage), "
            "residual copies of the thread's history persist until this TTL expires. "
            "0 explicitly disables expiry — orphaned keys then rely on the redis maxmemory policy alone."
        ),
    )
    key_prefix: str = Field(
        default="",
        description="Optional override for the redis key prefix; defaults to a hash of the database identity.",
    )


class DatabaseConfig(BaseModel):
    backend: Literal["memory", "sqlite", "postgres"] = Field(
        default="memory",
        description=("Storage backend for both checkpointer and application data. 'memory' for development (no persistence across restarts), 'sqlite' for single-node deployment, 'postgres' for production multi-node deployment."),
    )
    checkpoint_channel_mode: CheckpointChannelMode = Field(
        default="full",
        description=(
            "Checkpoint representation for accumulating channels. "
            "'full' preserves full-value message checkpoints; 'delta' uses "
            "LangGraph DeltaChannel for messages. Restart is required, and all "
            "processes sharing one checkpoint database must use the same value."
        ),
    )
    checkpoint_delta: CheckpointDeltaConfig = Field(
        default_factory=CheckpointDeltaConfig,
        description="Delta-mode checkpoint tuning. Only applies when checkpoint_channel_mode is 'delta'.",
    )
    checkpoint_graph_cache: CheckpointGraphCacheConfig = Field(
        default_factory=CheckpointGraphCacheConfig,
        description="Size caps for the compiled checkpoint graph caches. Hot-reloadable; not restart-required.",
    )
    checkpoint_cache: CheckpointCacheConfig = Field(
        default_factory=CheckpointCacheConfig,
        description="Delta-mode checkpoint history cache. Performance-only; safe to differ across workers.",
    )
    sqlite_dir: str = Field(
        default=".deer-flow/data",
        description=("Directory for the SQLite database file. Both checkpointer and application data share {sqlite_dir}/deerflow.db."),
    )
    postgres_url: str = Field(
        default="",
        description=(
            "PostgreSQL connection URL, shared by checkpointer and app. "
            "Use $DATABASE_URL in config.yaml to reference .env. "
            "Example: postgresql://user:pass@host:5432/deerflow "
            "(the +asyncpg driver suffix is added automatically where needed)."
        ),
    )
    echo_sql: bool = Field(
        default=False,
        description="Echo all SQL statements to log (debug only).",
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size for the app ORM engine (postgres only).",
    )
    pool_recycle: int = Field(
        default=300,
        gt=0,
        description="Seconds before app ORM PostgreSQL connections are recycled.",
    )
    command_timeout: float | None = Field(
        default=30,
        gt=0,
        description="Timeout in seconds for app ORM PostgreSQL commands. Set to null to disable the command timeout.",
    )
    postgres_schema: str = Field(
        default="",
        description=(
            "PostgreSQL schema for both app ORM tables and LangGraph "
            "checkpointer/store tables (postgres only). Empty string keeps "
            "the server default search_path (usually 'public'). When set, "
            "the schema is created automatically at startup and applied via "
            "connection-level search_path. Only plain identifiers are "
            f"allowed: {POSTGRES_SCHEMA_PATTERN}."
        ),
    )

    @field_validator("postgres_schema")
    @classmethod
    def _validate_postgres_schema(cls, value: str) -> str:
        return validate_postgres_schema(value)

    # -- 레거시 키 마이그레이션 (사용자 설정 항목 아님) --

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_snapshot_frequency(cls, data: Any) -> Any:
        """이름을 바꾸기 전의 최상위 ``checkpoint_delta_snapshot_frequency`` 키를
        ``checkpoint_delta.snapshot_frequency``로 옮긴다.

        ``DatabaseConfig``는 알 수 없는 키를 무시하므로(pydantic ``extra="ignore"``),
        이 shim이 없으면 옛 flat 키로 작성된 config.yaml이 운영자가 고른 값 대신 새
        기본 주기로 조용히 fallback한다. 명시적으로 설정된 중첩 키가 항상 이긴다.
        """
        if not isinstance(data, dict) or "checkpoint_delta_snapshot_frequency" not in data:
            return data
        data = dict(data)
        legacy_value = data.pop("checkpoint_delta_snapshot_frequency")
        nested = data.get("checkpoint_delta")
        if isinstance(nested, dict):
            if "snapshot_frequency" in nested:
                logger.warning(
                    "Both database.checkpoint_delta_snapshot_frequency (deprecated) and database.checkpoint_delta.snapshot_frequency are set; the nested key wins.",
                )
                return data
            data["checkpoint_delta"] = {**nested, "snapshot_frequency": legacy_value}
        elif nested is None:
            data["checkpoint_delta"] = {"snapshot_frequency": legacy_value}
        else:
            # 코드로 만든 CheckpointDeltaConfig 인스턴스: 명시적 객체가 레거시 스칼라를
            # 이긴다.
            logger.warning(
                "Ignoring deprecated database.checkpoint_delta_snapshot_frequency because database.checkpoint_delta is already set.",
            )
            return data
        logger.warning(
            "database.checkpoint_delta_snapshot_frequency is deprecated; use database.checkpoint_delta.snapshot_frequency instead. Carried the legacy value (%r) forward.",
            legacy_value,
        )
        return data

    # -- 파생 헬퍼 (사용자 설정 항목 아님) --

    @property
    def _resolved_sqlite_dir(self) -> str:
        """sqlite_dir을 (CWD 기준) 절대 경로로 해석한다."""
        from pathlib import Path

        return str(Path(self.sqlite_dir).resolve())

    @property
    def sqlite_path(self) -> str:
        """checkpointer와 app이 공유하는 통합 SQLite 파일 경로."""
        return os.path.join(self._resolved_sqlite_dir, "deerflow.db")

    # 하위 호환 alias
    @property
    def checkpointer_sqlite_path(self) -> str:
        """LangGraph checkpointer용 SQLite 파일 경로(sqlite_path의 alias)."""
        return self.sqlite_path

    @property
    def app_sqlite_path(self) -> str:
        """애플리케이션 ORM 데이터용 SQLite 파일 경로(sqlite_path의 alias)."""
        return self.sqlite_path

    @property
    def app_sqlalchemy_url(self) -> str:
        """애플리케이션 ORM engine용 SQLAlchemy async URL."""
        if self.backend == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        if self.backend == "postgres":
            url = self.postgres_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                # libpq의 축약 alias: psycopg checkpointer는 받아들이지만 SQLAlchemy dialect는 아니다.
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            return url
        raise ValueError(f"No SQLAlchemy URL for backend={self.backend!r}")

    @property
    def app_sync_sqlalchemy_url(self) -> str:
        """애플리케이션 ORM 데이터용 SQLAlchemy *동기* URL.

        ``agent_storage.backend: db`` store가 쓴다. 이 store의 소비자(LangGraph graph
        factory, setup/update 도구)는 동기이고 event loop 위나 gateway와 다른 프로세스에서
        돌 수 있는데, 그런 곳에서는 async engine을 구동할 수 없다.
        :meth:`app_sqlalchemy_url`과 같은 database 파일/서버를 가리키며 driver만 다르다
        (stdlib sqlite3와 psycopg 둘 다 앱에 함께 배포되므로 의존성이 늘지 않는다).
        """
        if self.backend == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        if self.backend == "postgres":
            url = self.postgres_url
            if url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
            elif url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg://", 1)
            return url
        raise ValueError(f"No SQLAlchemy URL for backend={self.backend!r}")
