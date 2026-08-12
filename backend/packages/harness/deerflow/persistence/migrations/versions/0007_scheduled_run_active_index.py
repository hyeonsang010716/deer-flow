"""scheduled task run의 active 유일성.

Revision ID: 0007_scheduled_run_active_index
Revises: 0006_agents
Create Date: 2026-07-11
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "0007_scheduled_run_active_index"
down_revision: str | Sequence[str] | None = "0006_agents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _dedupe_active_scheduled_runs_per_task() -> None:
    """partial unique index를 만들 수 있도록 중복된 active row를 대체 처리한다.

    ``uq_scheduled_task_run_active``는 ``task_id``당 queued/running row가 최대 하나임을
    강제한다. 같은 task에 active row가 둘 이상인 DB(이 PR이 닫는 바로 그 TOCTOU: 동시에
    실행된 두 ``dispatch_task``가 모두 ``has_active_runs``를 통과해 "queued" row를 둘 다
    삽입한 경우)는 ``CREATE UNIQUE INDEX``에서 실패해 alembic upgrade를 중단시키고
    gateway 시작을 막는다.

    ``task_id``별로 가장 최신 active row(``created_at`` DESC, 동점이면 결정적 기준으로
    ``id`` DESC)만 남기고 나머지는 설명을 담은 ``error``와 ``finished_at``과 함께
    ``interrupted``로 표시한다. 프로세스가 사라진 run에 대해
    ``ScheduledTaskRunRepository.mark_stale_active_runs``가 쓰는 orphan 처리와 동일하다.
    """
    bind = op.get_bind()
    superseded_message = "interrupted during migration 0007_scheduled_run_active_index: superseded by a newer active run for the same scheduled task (partial unique index uq_scheduled_task_run_active)"
    find_dupe_rows = sa.text(
        """
        SELECT id, task_id
        FROM scheduled_task_runs AS r1
        WHERE r1.status IN ('queued', 'running')
          AND EXISTS (
            SELECT 1 FROM scheduled_task_runs AS r2
            WHERE r2.task_id = r1.task_id
              AND r2.status IN ('queued', 'running')
              AND r2.id <> r1.id
              AND (
                r2.created_at > r1.created_at
                OR (r2.created_at = r1.created_at AND r2.id > r1.id)
              )
          )
        """
    )
    rows = list(bind.execute(find_dupe_rows).fetchall())
    if not rows:
        return
    for run_id, task_id in rows:
        logger.warning(
            "migration 0007_scheduled_run_active_index: superseding duplicate active scheduled run %s on task %s",
            run_id,
            task_id,
        )
    update_stmt = sa.text(
        """
        UPDATE scheduled_task_runs
        SET status = 'interrupted',
            error = :error_message,
            finished_at = :finished_at
        WHERE status IN ('queued', 'running')
          AND EXISTS (
            SELECT 1 FROM scheduled_task_runs AS r2
            WHERE r2.task_id = scheduled_task_runs.task_id
              AND r2.status IN ('queued', 'running')
              AND r2.id <> scheduled_task_runs.id
              AND (
                r2.created_at > scheduled_task_runs.created_at
                OR (r2.created_at = scheduled_task_runs.created_at AND r2.id > scheduled_task_runs.id)
              )
          )
        """
    ).bindparams(
        sa.bindparam("error_message"),
        # 타입을 지정해서 SQLAlchemy가 raw datetime을 DBAPI에 그대로 넘기지 않고 dialect의
        # DateTime bind processor(SQLite 문자열 포맷 / Postgres timestamptz)를 적용하게 한다
        # (Python 3.12에서 sqlite3의 기본 datetime adapter가 제거됐다).
        sa.bindparam("finished_at", type_=sa.DateTime(timezone=True)),
    )
    bind.execute(
        update_stmt,
        {"error_message": superseded_message, "finished_at": datetime.now(UTC)},
    )


def upgrade() -> None:
    # 멱등적인 index 생성: legacy/빈 DB bootstrap 경로는 upgrade head 전에 create_all
    # (ORM __table_args__로부터 index를 만든다)을 실행하므로, index가 이미 있어도 이
    # migration이 실패해서는 안 된다.
    insp = sa.inspect(op.get_bind())
    existing = {ix["name"] for ix in insp.get_indexes("scheduled_task_runs")}
    if "uq_scheduled_task_run_active" not in existing:
        # 이미 불변식을 위반한 DB에서도 partial UNIQUE index를 만들 수 있도록 중복 active
        # row를 먼저 대체 처리한다. 깨끗한 DB에서는 아무 일도 하지 않는다(일반적인 경로에서는
        # create_all이 이미 index를 만들었으므로, 이 분기는 index 도입 이전의 legacy DB에서만
        # 실행된다).
        _dedupe_active_scheduled_runs_per_task()
        with op.batch_alter_table("scheduled_task_runs", schema=None) as batch_op:
            batch_op.create_index(
                "uq_scheduled_task_run_active",
                ["task_id"],
                unique=True,
                sqlite_where=sa.text("status IN ('queued', 'running')"),
                postgresql_where=sa.text("status IN ('queued', 'running')"),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = {ix["name"] for ix in insp.get_indexes("scheduled_task_runs")}
    if "uq_scheduled_task_run_active" in existing:
        with op.batch_alter_table("scheduled_task_runs", schema=None) as batch_op:
            batch_op.drop_index("uq_scheduled_task_run_active")
