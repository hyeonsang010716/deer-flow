"""LangGraph checkpointer 설정."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from deerflow.config.postgres_schema import POSTGRES_SCHEMA_PATTERN, validate_postgres_schema

CheckpointerType = Literal["memory", "sqlite", "postgres"]


class CheckpointerConfig(BaseModel):
    """LangGraph state 영속화 checkpointer 설정."""

    type: CheckpointerType = Field(
        description="Checkpointer backend type. "
        "'memory' is in-process only (lost on restart). "
        "'sqlite' persists to a local file (requires langgraph-checkpoint-sqlite). "
        "'postgres' persists to PostgreSQL (install with deerflow-harness[postgres])."
    )
    connection_string: str | None = Field(
        default=None,
        description="Connection string for sqlite (file path) or postgres (DSN). "
        "Optional for sqlite and defaults to 'store.db' when omitted. "
        "Required for postgres. "
        "For sqlite, use a file path like '.deer-flow/checkpoints.db' or ':memory:' for in-memory. "
        "For postgres, use a DSN like 'postgresql://user:pass@localhost:5432/db'.",
    )
    postgres_schema: str = Field(
        default="",
        description=(f"PostgreSQL schema for legacy checkpointer/store tables (postgres only). Empty string keeps the server default search_path (usually 'public'). Only plain identifiers are allowed: {POSTGRES_SCHEMA_PATTERN}."),
    )

    @field_validator("postgres_schema")
    @classmethod
    def _validate_postgres_schema(cls, value: str) -> str:
        return validate_postgres_schema(value)


# 전역 설정 인스턴스. None이면 checkpointer가 설정되지 않았다는 뜻이다.
_checkpointer_config: CheckpointerConfig | None = None


def get_checkpointer_config() -> CheckpointerConfig | None:
    """현재 checkpointer 설정을 반환한다. 설정되지 않았으면 None."""
    return _checkpointer_config


def set_checkpointer_config(config: CheckpointerConfig | None) -> None:
    """checkpointer 설정을 지정한다."""
    global _checkpointer_config
    _checkpointer_config = config


def ensure_config_loaded() -> None:
    """checkpointer 설정이 아직 초기화되지 않았으면 app config를 지연 로드한다."""
    from deerflow.config.app_config import _app_config, get_app_config

    config = get_checkpointer_config()
    if config is not None or _app_config is not None:
        return

    try:
        get_app_config()
    except FileNotFoundError:
        pass


def load_checkpointer_config_from_dict(config_dict: dict | None) -> None:
    """dict에서 checkpointer 설정을 읽어 들인다."""
    global _checkpointer_config
    if config_dict is None:
        _checkpointer_config = None
        return
    _checkpointer_config = CheckpointerConfig(**config_dict)
