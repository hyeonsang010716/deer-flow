"""pod 간 inbound webhook 중복 제거 테이블(issue #4120).

Revision ID: 0009_webhook_dedupe
Revises: 0008_thread_operation_kind
Create Date: 2026-07-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_webhook_dedupe"
down_revision: str | Sequence[str] | None = "0008_thread_operation_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("webhook_deliveries"):
        # 멱등성 보장: 전체 metadata create_all로 이미 테이블이 만들어진 DB(예: 새 DB나 legacy
        # 테스트 seed)에서 여기서 다시 만들면 안 된다.
        return
    op.create_table(
        "webhook_deliveries",
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("chat_id", sa.String(length=512), nullable=False),
        sa.Column("message_id", sa.String(length=1024), nullable=False),
        sa.Column(
            "first_seen",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # 복합 PK는 ChannelManager._inbound_dedupe_key와 정확히 같다. 문자열을 이어 붙인 대리
        # 키는 의도적으로 피한다. 구성 요소에 Postgres TEXT 컬럼에 담을 수 없는 문자(예: NUL)가
        # 들어갈 수 있고, 이어 붙인 키는 길이 초과 위험도 있다.
        sa.PrimaryKeyConstraint(
            "channel",
            "workspace_id",
            "chat_id",
            "message_id",
            name="pk_webhook_deliveries",
        ),
    )
    op.create_index("ix_webhook_deliveries_first_seen", "webhook_deliveries", ["first_seen"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("webhook_deliveries"):
        op.drop_index("ix_webhook_deliveries_first_seen", table_name="webhook_deliveries")
        op.drop_table("webhook_deliveries")
