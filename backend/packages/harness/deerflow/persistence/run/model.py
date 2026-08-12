"""run metadata의 ORM model."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    assistant_id: Mapped[str | None] = mapped_column(String(128))
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # "pending" | "running" | "success" | "error" | "timeout" | "interrupted"
    operation_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="run", server_default=text("'run'"))

    model_name: Mapped[str | None] = mapped_column(String(128))
    multitask_strategy: Mapped[str] = mapped_column(String(20), default="reject")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    kwargs_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    stop_reason: Mapped[str | None] = mapped_column(String(50))

    # 편의 필드(RunEventStore를 조회하지 않고 목록 페이지를 그리기 위한 것)
    message_count: Mapped[int] = mapped_column(default=0)
    first_human_message: Mapped[str | None] = mapped_column(Text)
    last_ai_message: Mapped[str | None] = mapped_column(Text)

    # token 사용량(RunJournal이 메모리에 누적했다가 run 완료 시점에 기록한다)
    total_input_tokens: Mapped[int] = mapped_column(default=0)
    total_output_tokens: Mapped[int] = mapped_column(default=0)
    total_tokens: Mapped[int] = mapped_column(default=0)
    llm_call_count: Mapped[int] = mapped_column(default=0)
    lead_agent_tokens: Mapped[int] = mapped_column(default=0)
    subagent_tokens: Mapped[int] = mapped_column(default=0)
    middleware_tokens: Mapped[int] = mapped_column(default=0)
    token_usage_by_model: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))

    # follow-up 연결
    follow_up_to_run_id: Mapped[str | None] = mapped_column(String(64))

    # 멀티 worker run 소유권
    owner_worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 소유자가 아닌 worker가 여기에 취소를 기록하고, 소유자는 lease를 갱신하면서 이를 소비한다.
    # 먼저 기록된 action이 이긴다.
    cancel_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_runs_thread_status", "thread_id", "status"),
        Index("ix_runs_lease", "lease_expires_at"),
        # 프로세스 간 원자성 보장: thread당 pending/running run은 최대 하나다. migration만이
        # 아니라 ORM ``__table_args__``에도 있어야 한다. 빈 DB bootstrap 경로는
        # ``create_all`` + ``stamp head``를 수행하며, 이 index를 정의하는 migration을 실행하지
        # 않기 때문이다.
        Index(
            "uq_runs_thread_active",
            "thread_id",
            unique=True,
            sqlite_where=text("status IN ('pending', 'running')"),
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
    )
