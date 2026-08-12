from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.scheduled_tasks.model import ScheduledTaskRow
from deerflow.utils.time import coerce_iso

TERMINAL_TASK_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class ScheduledTaskRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def _row_to_dict(row: ScheduledTaskRow) -> dict[str, Any]:
        data = row.to_dict()
        for key in (
            "created_at",
            "updated_at",
            "next_run_at",
            "last_run_at",
            "lease_expires_at",
        ):
            if data.get(key) is not None:
                data[key] = coerce_iso(data[key])
        return data

    async def create(
        self,
        *,
        task_id: str,
        user_id: str,
        thread_id: str | None,
        context_mode: str,
        assistant_id: str | None,
        title: str,
        prompt: str,
        schedule_type: str,
        schedule_spec: dict[str, Any],
        timezone: str,
        next_run_at: datetime | None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = ScheduledTaskRow(
            id=task_id,
            user_id=user_id,
            thread_id=thread_id,
            context_mode=context_mode,
            assistant_id=assistant_id,
            title=title,
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_spec=schedule_spec,
            timezone=timezone,
            next_run_at=next_run_at,
            created_at=now,
            updated_at=now,
        )
        async with self._sf() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)

    async def get(self, task_id: str, *, user_id: str) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(ScheduledTaskRow, task_id)
            if row is None or row.user_id != user_id:
                return None
            return self._row_to_dict(row)

    async def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        stmt = select(ScheduledTaskRow).where(ScheduledTaskRow.user_id == user_id).order_by(ScheduledTaskRow.created_at.desc(), ScheduledTaskRow.id.desc())
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(row) for row in result.scalars()]

    async def update(
        self,
        task_id: str,
        *,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(ScheduledTaskRow, task_id)
            if row is None or row.user_id != user_id:
                return None
            for key, value in updates.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            row.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)

    async def delete(self, task_id: str, *, user_id: str) -> bool:
        async with self._sf() as session:
            row = await session.get(ScheduledTaskRow, task_id)
            if row is None or row.user_id != user_id:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def claim_due_tasks(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_seconds: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        stmt = (
            select(ScheduledTaskRow)
            .where(
                ScheduledTaskRow.next_run_at.is_not(None),
                ScheduledTaskRow.next_run_at <= now,
                or_(
                    and_(
                        ScheduledTaskRow.status == "enabled",
                        or_(
                            ScheduledTaskRow.lease_expires_at.is_(None),
                            ScheduledTaskRow.lease_expires_at < now,
                        ),
                    ),
                    # lease가 만료된 채 "running"에 멈춘 task는 claim한 프로세스가 claim과
                    # dispatch 사이에서 죽었다는 뜻이다. 다시 claim할 수 있어야 하며, 아니면
                    # 그 task는 영영 죽은 상태로 남는다.
                    and_(
                        ScheduledTaskRow.status == "running",
                        ScheduledTaskRow.lease_expires_at.is_not(None),
                        ScheduledTaskRow.lease_expires_at < now,
                    ),
                ),
            )
            .order_by(ScheduledTaskRow.next_run_at.asc(), ScheduledTaskRow.id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            rows = list(result.scalars())
            for row in rows:
                row.lease_owner = lease_owner
                row.lease_expires_at = lease_expires_at
                row.status = "running"
                row.updated_at = datetime.now(UTC)
            await session.commit()
            return [self._row_to_dict(row) for row in rows]

    async def update_after_launch(
        self,
        task_id: str,
        *,
        status: str,
        next_run_at: datetime | None,
        last_run_at: datetime | None,
        last_run_id: str | None,
        last_thread_id: str | None,
        last_error: str | None,
        increment_run_count: bool,
        protect_terminal: bool = False,
    ) -> None:
        async with self._sf() as session:
            row = await session.get(ScheduledTaskRow, task_id)
            if row is None:
                return
            if protect_terminal and row.status in TERMINAL_TASK_STATUSES:
                # 빠르게 실패한 run은 이 launch 경로 쓰기가 커밋되기 전에
                # handle_run_completion(`once` task를 종결시킨다)에 도달할 수 있다. hook이
                # 남긴 status/error는 유지하고 launch 관련 기록만 남긴다.
                pass
            else:
                row.status = status
                row.last_error = last_error
            row.next_run_at = next_run_at
            row.last_run_at = last_run_at
            row.last_run_id = last_run_id
            row.last_thread_id = last_thread_id
            if increment_run_count:
                row.run_count += 1
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = datetime.now(UTC)
            await session.commit()

    async def list_by_user_and_thread(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(ScheduledTaskRow)
            .where(
                ScheduledTaskRow.user_id == user_id,
                ScheduledTaskRow.thread_id == thread_id,
            )
            .order_by(ScheduledTaskRow.created_at.desc(), ScheduledTaskRow.id.desc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(row) for row in result.scalars()]

    async def cancel_stuck_once_tasks(self, *, error: str) -> int:
        """프로세스 크래시로 ``running``에 고아로 남은 ``once`` task를 정리한다.

        launch된 ``once`` task는 in-process 완료 hook이 terminal 상태로 옮길 때까지 ``running``
        상태로 남는다. lease는 launch 시점에 지워지므로 claim 쿼리의 만료 lease 회수 분기가
        이 행을 절대 보지 못한다. 크래시 후에는 hook이 사라져 task가 영영 멈춰 있게 된다.
        아직 lease를 들고 있는 task는 건드리지 않는다 — claim만 되고 launch되지 않은 상태이며,
        만료 lease 회수가 안전하게 복구한다.
        """
        stmt = select(ScheduledTaskRow).where(
            ScheduledTaskRow.schedule_type == "once",
            ScheduledTaskRow.status == "running",
            ScheduledTaskRow.lease_expires_at.is_(None),
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            rows = list(result.scalars())
            now = datetime.now(UTC)
            for row in rows:
                row.status = "cancelled"
                row.last_error = error
                row.updated_at = now
            await session.commit()
            return len(rows)
