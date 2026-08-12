"""users 테이블의 ORM model.

harness persistence 패키지에 두어 ``threads_meta``, ``runs``, ``run_events``, ``feedback``과
함께 ``Base.metadata.create_all()``에 잡히게 한다. 공유 engine을 쓰면 다음이 보장된다.

- SQLite/Postgres 데이터베이스 하나, connection pool 하나
- schema 초기화 경로 하나
- auth와 persistence 읽기 전반에서 일관된 async session
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class UserRow(Base):
    __tablename__ = "users"

    # backend 간 이식성을 위해 UUID는 36자 문자열로 저장한다.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # "admin" | "user" — 새 role이 추가될 때 ALTER TABLE로 고생하지 않도록 평범한 문자열로
    # 둔다.
    system_role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # OAuth 연결(선택). partial unique index가 (provider, oauth_id) 쌍당 계정 하나를 강제하며,
    # NULL/NULL 행은 제약 없이 남겨 일반 비밀번호 계정이 함께 존재할 수 있게 한다.
    oauth_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    oauth_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # auth lifecycle 플래그
    needs_setup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    token_version: Mapped[int] = mapped_column(nullable=False, default=0)

    __table_args__ = (
        Index(
            "idx_users_oauth_identity",
            "oauth_provider",
            "oauth_id",
            unique=True,
            sqlite_where=text("oauth_provider IS NOT NULL AND oauth_id IS NOT NULL"),
        ),
    )
