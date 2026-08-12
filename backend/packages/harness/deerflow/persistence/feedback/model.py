"""run에 대한 사용자 feedback의 ORM model."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class FeedbackRow(Base):
    __tablename__ = "feedback"

    __table_args__ = (UniqueConstraint("thread_id", "run_id", "user_id", name="uq_feedback_thread_run_user"),)

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    message_id: Mapped[str | None] = mapped_column(String(64))
    # message_id는 선택적인 RunEventStore 이벤트 식별자다.
    # feedback이 특정 메시지를 가리킬 수도, run 전체를 가리킬 수도 있게 한다.

    rating: Mapped[int] = mapped_column(nullable=False)
    # +1(thumbs-up) 또는 -1(thumbs-down)

    comment: Mapped[str | None] = mapped_column(Text)
    # 사용자가 남기는 선택적 텍스트 feedback

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
