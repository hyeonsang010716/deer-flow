from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.scheduled_task_runs.model import ScheduledTaskRunRow
from deerflow.utils.time import coerce_iso

TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"success", "failed", "skipped", "interrupted"})
ACTIVE_RUN_STATUSES: tuple[str, ...] = ("queued", "running")


class ActiveScheduledRunConflict(Exception):
    """동시에 실행된 다른 dispatch가 이미 해당 task의 유일한 active-run 슬롯을 차지했다.

    active(queued/running) run row를 삽입하면 partial unique index
    ``uq_scheduled_task_run_active``(``task_id``당 active run 최대 1개)를 위반할 때
    :meth:`ScheduledTaskRunRepository.create`가 발생시킨다. ``ScheduledTaskService.dispatch_task``의
    비원자적 ``has_active_runs`` 검사에 대응하는 원자적 장치다. 두 dispatch가 그 검사를 모두
    통과할 수 있지만 active row를 삽입할 수 있는 것은 하나뿐이고, 진 쪽이 여기에 도달한다.

    repository 경계에서 SQLAlchemy ``IntegrityError``를 도메인 예외로 변환하면 service
    레이어가 ``sqlalchemy.exc``에 결합되지 않는다(runs 테이블의
    ``deerflow.runtime.ConflictError``와 같은 방식).
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"scheduled task {task_id!r} already has an active run")


class ScheduledTaskRunRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def _row_to_dict(row: ScheduledTaskRunRow) -> dict[str, Any]:
        data = row.to_dict()
        for key in ("scheduled_for", "started_at", "finished_at", "created_at"):
            if data.get(key) is not None:
                data[key] = coerce_iso(data[key])
        return data

    async def create(
        self,
        *,
        run_record_id: str,
        task_id: str,
        thread_id: str,
        scheduled_for: datetime,
        trigger: str,
        status: str,
    ) -> dict[str, Any]:
        row = ScheduledTaskRunRow(
            id=run_record_id,
            task_id=task_id,
            thread_id=thread_id,
            scheduled_for=scheduled_for,
            trigger=trigger,
            status=status,
            created_at=datetime.now(UTC),
        )
        async with self._sf() as session:
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                # partial unique index ``uq_scheduled_task_run_active``에 걸릴 수 있는 것은
                # active 상태 삽입뿐이다. 종료 상태 row(예: "skipped" tombstone)는 그 조건
                # 밖이라 충돌할 수 없으므로, 거기서 나온 IntegrityError는 진짜 오류이며
                # 변환하지 않고 그대로 다시 던진다.
                if status in ACTIVE_RUN_STATUSES:
                    raise ActiveScheduledRunConflict(task_id) from None
                raise
            await session.refresh(row)
            return self._row_to_dict(row)

    async def list_by_task(self, task_id: str, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        stmt = (
            select(ScheduledTaskRunRow)
            .where(ScheduledTaskRunRow.task_id == task_id)
            .order_by(
                ScheduledTaskRunRow.created_at.desc(),
                ScheduledTaskRunRow.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(row) for row in result.scalars()]

    async def count_active_runs(self) -> int:
        """queued/running row의 전역 개수. task 간 동시성을 제한하는 데 쓴다."""
        stmt = select(func.count()).select_from(ScheduledTaskRunRow).where(ScheduledTaskRunRow.status.in_(ACTIVE_RUN_STATUSES))
        async with self._sf() as session:
            result = await session.execute(stmt)
            return int(result.scalar() or 0)

    async def update_status(
        self,
        run_record_id: str,
        *,
        status: str,
        run_id: str | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        protect_terminal: bool = False,
    ) -> None:
        async with self._sf() as session:
            row = await session.get(ScheduledTaskRunRow, run_record_id)
            if row is None:
                return
            if protect_terminal and row.status in TERMINAL_RUN_STATUSES:
                # launch 경로의 "running" 쓰기가 completion hook과의 경쟁에서 졌다. 종료
                # 상태/에러는 유지하고, completion 쓰기가 알 수 없었던 기록만 채워 넣는다.
                if row.run_id is None and run_id is not None:
                    row.run_id = run_id
                if row.started_at is None and started_at is not None:
                    row.started_at = started_at
                await session.commit()
                return
            row.status = status
            row.run_id = run_id
            row.error = error
            if started_at is not None:
                row.started_at = started_at
            if finished_at is not None:
                row.finished_at = finished_at
            await session.commit()

    async def has_active_runs(self, task_id: str) -> bool:
        stmt = (
            select(ScheduledTaskRunRow.id)
            .where(
                ScheduledTaskRunRow.task_id == task_id,
                ScheduledTaskRunRow.status.in_(ACTIVE_RUN_STATUSES),
            )
            .limit(1)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return result.scalars().first() is not None

    async def mark_stale_active_runs(self, *, error: str) -> int:
        """프로세스 크래시로 고아가 된 run을 즉시 정리하는 기록 작업.

        agent run은 in-process로 실행되므로, scheduler 시작 시 발견되는 ``queued``/``running``
        row는 프로세스가 사라진 run에 속한다. MVP의 단일 scheduler 인스턴스 가정에서만 유효하다.
        """
        stmt = select(ScheduledTaskRunRow).where(ScheduledTaskRunRow.status.in_(ACTIVE_RUN_STATUSES))
        now = datetime.now(UTC)
        async with self._sf() as session:
            result = await session.execute(stmt)
            rows = list(result.scalars())
            for row in rows:
                row.status = "interrupted"
                row.error = error
                row.finished_at = now
            await session.commit()
            return len(rows)
