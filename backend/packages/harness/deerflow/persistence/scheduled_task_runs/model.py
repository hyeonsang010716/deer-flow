from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class ScheduledTaskRunRow(Base):
    __tablename__ = "scheduled_task_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trigger: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    __table_args__ = (
        # task당 active(queued/running) run은 최대 하나다. ``dispatch_task``의 skip 정책을
        # 원자적으로 판정하는 주체다. 비원자적인 ``has_active_runs`` 확인 후 생성은 fast path일
        # 뿐이고, 동시 dispatch 두 개(더블클릭 / client 재시도 / poller와 경합하는 수동 trigger)가
        # 모두 통과할 수 있으므로 DB가 두 번째 active insert를 거부해야 한다. ``runs`` 테이블의
        # ``uq_runs_thread_active``(PR #4003)와 형제 관계지만, 그쪽은 ``thread_id``를 키로 쓰기
        # 때문에 기본 ``fresh_thread_per_run`` 상황(dispatch마다 새 thread)을 커버하지 못한다.
        # 그래서 scheduled task run row에는 자체 가드가 필요하다.
        #
        # 조건은 ``overlap_policy``가 아니라 status만 본다. MVP에서 정책이 "skip"으로 고정되어
        # 있으므로, 구현되지도 않은 non-skip 정책을 위해 ``overlap_policy``를 run row에
        # 비정규화하지 않고도 status만으로 현재 불변식을 강제할 수 있다. non-skip 정책이 추가되면
        # 조건부로 바꿔야 한다(예: ``... AND overlap_policy = 'skip'``).
        #
        # 이 정의는 migration뿐 아니라 ORM ``__table_args__``에도 있어야 한다. 빈 DB bootstrap
        # 경로는 ``create_all`` + ``stamp head``를 실행하고, 이 index를 정의하는 migration을
        # 실행하지 않기 때문이다.
        Index(
            "uq_scheduled_task_run_active",
            "task_id",
            unique=True,
            sqlite_where=text("status IN ('queued', 'running')"),
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
    )
