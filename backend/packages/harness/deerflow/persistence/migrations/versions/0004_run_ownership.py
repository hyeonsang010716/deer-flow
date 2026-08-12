"""run ownership 컬럼과 active-thread unique index를 추가한다.

Revision ID: 0004_run_ownership
Revises: 0003_scheduled_tasks
Create Date: 2026-07-07
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "0004_run_ownership"
down_revision: str | Sequence[str] | None = "0003_scheduled_tasks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _dedupe_active_runs_per_thread() -> None:
    """partial unique index를 만들 수 있도록, 대체된 active row를 취소한다.

    ``uq_runs_thread_active``는 ``thread_id``당 pending/running row를 최대 하나로 강제한다.
    같은 thread에 이미 active row가 둘 이상인 DB는 ``CREATE UNIQUE INDEX``에 실패해 alembic
    upgrade가 중단되고 gateway 시작이 막힌다(실제로 발생 가능하다. Postgres 배포는 예전 sqlite
    전용 gate 때문에 reconciliation을 건너뛰었고, 이 PR 이전에 ``GATEWAY_WORKERS>1``로 실행한
    경우 중복이 생길 수 있다).

    ``thread_id``별로 가장 최근 active row만 남기고(``created_at`` DESC, 결정적 tiebreaker로
    ``run_id`` DESC) 나머지는 ``error``로 표시한다. 취소된 row에는 설명이 담긴 ``error``
    문자열을 넣어 운영자가 왜 run이 종료됐는지 알 수 있게 한다.
    """
    bind = op.get_bind()
    cancel_message = "cancelled during migration 0004_run_ownership: superseded by a newer active run for the same thread (partial unique index uq_runs_thread_active)"
    find_dupe_rows = sa.text(
        """
        SELECT run_id, thread_id
        FROM runs AS r1
        WHERE r1.status IN ('pending', 'running')
          AND EXISTS (
            SELECT 1 FROM runs AS r2
            WHERE r2.thread_id = r1.thread_id
              AND r2.status IN ('pending', 'running')
              AND r2.run_id <> r1.run_id
              AND (
                r2.created_at > r1.created_at
                OR (r2.created_at = r1.created_at AND r2.run_id > r1.run_id)
              )
          )
        """
    )
    rows = list(bind.execute(find_dupe_rows).fetchall())
    if not rows:
        return
    for run_id, thread_id in rows:
        logger.warning(
            "migration 0004_run_ownership: cancelling duplicate active run %s on thread %s",
            run_id,
            thread_id,
        )
    bind.execute(
        sa.text(
            """
            UPDATE runs
            SET status = 'error',
                error = :error_message
            WHERE status IN ('pending', 'running')
              AND EXISTS (
                SELECT 1 FROM runs AS r2
                WHERE r2.thread_id = runs.thread_id
                  AND r2.status IN ('pending', 'running')
                  AND r2.run_id <> runs.run_id
                  AND (
                    r2.created_at > runs.created_at
                    OR (r2.created_at = runs.created_at AND r2.run_id > runs.run_id)
                  )
              )
            """
        ),
        {"error_message": cancel_message},
    )


def upgrade() -> None:
    from deerflow.persistence.migrations._helpers import safe_add_column

    safe_add_column("runs", sa.Column("owner_worker_id", sa.String(length=128), nullable=True))
    safe_add_column("runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))

    # 멱등한 index 생성. legacy bootstrap 경로는 upgrade head 전에 create_all을 실행해
    # ORM __table_args__로부터 index를 만들므로, index가 이미 있어도 migration이 실패하면 안 된다.
    insp = sa.inspect(op.get_bind())
    existing = {ix["name"] for ix in insp.get_indexes("runs")}
    if "ix_runs_lease" not in existing:
        with op.batch_alter_table("runs", schema=None) as batch_op:
            batch_op.create_index("ix_runs_lease", ["lease_expires_at"], unique=False)
    if "uq_runs_thread_active" not in existing:
        # 이미 불변식을 위반한 DB에서도 partial UNIQUE index를 만들 수 있도록 중복 active row를
        # 먼저 취소한다. 정상 DB에서는 no-op이다(일반 경로에서는 create_all이 이미 index를
        # 만들었으므로, 이 분기는 index 이전 시절의 legacy DB에서만 실행된다).
        _dedupe_active_runs_per_thread()
        with op.batch_alter_table("runs", schema=None) as batch_op:
            batch_op.create_index(
                "uq_runs_thread_active",
                ["thread_id"],
                unique=True,
                sqlite_where=sa.text("status IN ('pending', 'running')"),
                postgresql_where=sa.text("status IN ('pending', 'running')"),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = {ix["name"] for ix in insp.get_indexes("runs")}
    if "uq_runs_thread_active" in existing:
        with op.batch_alter_table("runs", schema=None) as batch_op:
            batch_op.drop_index("uq_runs_thread_active")
    if "ix_runs_lease" in existing:
        with op.batch_alter_table("runs", schema=None) as batch_op:
            batch_op.drop_index("ix_runs_lease")

    from deerflow.persistence.migrations._helpers import safe_drop_column

    safe_drop_column("runs", "lease_expires_at")
    safe_drop_column("runs", "owner_worker_id")
