"""SQLAlchemy 기반 thread metadata repository."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from deerflow.persistence.json_compat import json_match
from deerflow.persistence.thread_meta.base import THREAD_PINNED_METADATA_KEY, InvalidMetadataFilterError, ThreadMetaStore
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.runtime.user_context import AUTO, _AutoSentinel, resolve_user_id
from deerflow.utils.time import coerce_iso

logger = logging.getLogger(__name__)


class ThreadMetaRepository(ThreadMetaStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def _row_to_dict(row: ThreadMetaRow) -> dict[str, Any]:
        d = row.to_dict()
        d["metadata"] = d.pop("metadata_json", None) or {}
        for key in ("created_at", "updated_at"):
            val = d.get(key)
            if isinstance(val, datetime):
                # SQLite는 ``DateTime(timezone=True)``에도 불구하고 tzinfo를 버린다.
                # ``coerce_iso``가 naive 값을 UTC로 정규화해서 wire 포맷이 항상 tz를 갖게 한다.
                d[key] = coerce_iso(val)
        return d

    async def create(
        self,
        thread_id: str,
        *,
        assistant_id: str | None = None,
        user_id: str | None | _AutoSentinel = AUTO,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        # AUTO면 contextvar에서 user_id를 자동으로 해석한다. 명시적 None은 orphan row를
        # 만든다(migration script가 쓴다).
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.create")
        now = datetime.now(UTC)
        row = ThreadMetaRow(
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=resolved_user_id,
            display_name=display_name,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        async with self._sf() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)

    async def get(
        self,
        thread_id: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> dict | None:
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.get")
        async with self._sf() as session:
            row = await session.get(ThreadMetaRow, thread_id)
            if row is None:
                return None
            # 명시적으로 우회(user_id=None)하지 않는 한 owner 필터를 적용한다.
            if resolved_user_id is not None and row.user_id != resolved_user_id:
                return None
            return self._row_to_dict(row)

    async def check_access(self, thread_id: str, user_id: str, *, require_existing: bool = False) -> bool:
        """``user_id``가 ``thread_id``에 접근할 수 있는지 확인한다.

        같은 row에 대해, 호출자가 하려는 일에 따라 두 가지 다른 의미의 모드가 있다:

        - ``require_existing=False``(기본, 허용적):
          row가 없거나(추적되지 않는 레거시 thread), ``row.user_id``가 None이거나(공유 /
          인증 이전 데이터), ``row.user_id == user_id``면 True를 반환한다. 추적되지 않는
          thread를 접근 가능으로 봐야 하위 호환이 유지되는 **읽기 계열** decorator에 쓴다.

        - ``require_existing=True``(엄격):
          row가 존재하고 (``row.user_id == user_id`` 또는 ``row.user_id is None``)일 때만
          True를 반환한다. **파괴적/변경** decorator(DELETE, PATCH, state 갱신)에 쓴다.
          그래야 *이미 삭제된* thread를 아무 호출자나 다시 대상으로 삼을 수 없다. row가
          사라지면 다른 모든 사용자가 그것을 "소유"한 것처럼 보이던 delete 멱등성의
          cross-user 구멍을 막는다.
        """
        async with self._sf() as session:
            row = await session.get(ThreadMetaRow, thread_id)
            if row is None:
                return not require_existing
            if row.user_id is None:
                return True
            return row.user_id == user_id

    async def search(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> list[dict[str, Any]]:
        """metadata와 status 필터를 선택적으로 적용해 thread를 검색한다.

        owner 필터는 기본으로 적용되므로 호출자는 user context 안에 있어야 한다.
        우회하려면 ``user_id=None``을 넘긴다(migration/CLI).
        """
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.search")
        pinned_order = case(
            (json_match(ThreadMetaRow.metadata_json, THREAD_PINNED_METADATA_KEY, True), 1),
            else_=0,
        )
        stmt = select(ThreadMetaRow).order_by(
            pinned_order.desc(),
            ThreadMetaRow.updated_at.desc(),
            ThreadMetaRow.thread_id.desc(),
        )
        if resolved_user_id is not None:
            stmt = stmt.where(ThreadMetaRow.user_id == resolved_user_id)
        if status:
            stmt = stmt.where(ThreadMetaRow.status == status)

        if metadata:
            applied = 0
            for key, value in metadata.items():
                try:
                    stmt = stmt.where(json_match(ThreadMetaRow.metadata_json, key, value))
                    applied += 1
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping metadata filter key %s: %s", ascii(key), exc)
            if applied == 0:
                # Gateway가 노출하는 400 detail을 클라이언트가 읽기 쉽도록 list repr이나
                # 중첩 따옴표 없이 쉼표로 구분한 평문으로 만든다. 결정성을 위해 정렬한다.
                rejected_keys = ", ".join(sorted(str(k) for k in metadata))
                raise InvalidMetadataFilterError(f"All metadata filter keys were rejected as unsafe: {rejected_keys}")

        stmt = stmt.limit(limit).offset(offset)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(r) for r in result.scalars()]

    async def _check_ownership(self, session: AsyncSession, thread_id: str, resolved_user_id: str | None) -> bool:
        """row가 존재하고 소유자가 맞으면(또는 필터를 우회했으면) True를 반환한다."""
        if resolved_user_id is None:
            return True  # 명시적 우회
        row = await session.get(ThreadMetaRow, thread_id)
        return row is not None and row.user_id == resolved_user_id

    async def update_display_name(
        self,
        thread_id: str,
        display_name: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> None:
        """thread의 display_name(제목)을 갱신한다."""
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.update_display_name")
        async with self._sf() as session:
            if not await self._check_ownership(session, thread_id, resolved_user_id):
                return
            await session.execute(update(ThreadMetaRow).where(ThreadMetaRow.thread_id == thread_id).values(display_name=display_name, updated_at=datetime.now(UTC)))
            await session.commit()

    async def update_status(
        self,
        thread_id: str,
        status: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> None:
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.update_status")
        async with self._sf() as session:
            if not await self._check_ownership(session, thread_id, resolved_user_id):
                return
            await session.execute(update(ThreadMetaRow).where(ThreadMetaRow.thread_id == thread_id).values(status=status, updated_at=datetime.now(UTC)))
            await session.commit()

    async def update_metadata(
        self,
        thread_id: str,
        metadata: dict,
        *,
        touch: bool = True,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> None:
        """``metadata``를 ``metadata_json``에 병합한다.

        read-modify-write 병합 전에 row를 잠그므로 동시 호출자가 서로의 키를 덮어쓸 수 없다.
        SQLite는 읽기 전에 write transaction을 잡고, row 단위 잠금이 있는 DB는
        ``SELECT ... FOR UPDATE``를 쓴다. row가 없거나 user_id 검사에 실패하면 아무것도 하지
        않는다.

        ``touch``는 ``updated_at``을 갱신한다(기본). pin/unpin처럼 metadata만 바뀌는 경우
        최신순 정렬을 유지하려면 ``touch=False``를 넘긴다.
        """
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.update_metadata")
        async with self._sf() as session:
            if session.get_bind().dialect.name == "sqlite":
                # SQLite의 deferred transaction은 UPDATE 시점까지 writer를 예약하지 않는데,
                # read-modify-write 병합에는 너무 늦다. BEGIN IMMEDIATE는 읽기 전에 writer를
                # 직렬화하며, 같은 파일을 쓰는 다른 프로세스의 writer도 포함한다.
                await session.execute(text("BEGIN IMMEDIATE"))
                row = await session.get(ThreadMetaRow, thread_id)
            else:
                result = await session.execute(select(ThreadMetaRow).where(ThreadMetaRow.thread_id == thread_id).with_for_update())
                row = result.scalar_one_or_none()
            if row is None:
                return
            if resolved_user_id is not None and row.user_id != resolved_user_id:
                return
            merged = dict(row.metadata_json or {})
            merged.update(metadata)
            row.metadata_json = merged
            if touch:
                row.updated_at = datetime.now(UTC)
            else:
                # ``updated_at``에는 ``onupdate`` hook이 있어서, 컬럼에 명시적 SET 값이 없으면
                # 어떤 row UPDATE에서든 발동한다. 현재 값을 dirty로 표시해 SQLAlchemy가 SET에
                # 포함시키고 hook을 건너뛰게 해서 최신순 정렬을 보존한다.
                flag_modified(row, "updated_at")
            await session.commit()

    async def update_owner(
        self,
        thread_id: str,
        owner_user_id: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> None:
        """thread metadata row의 소유자를 ``owner_user_id``로 옮긴다."""
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.update_owner")
        async with self._sf() as session:
            if not await self._check_ownership(session, thread_id, resolved_user_id):
                return
            await session.execute(update(ThreadMetaRow).where(ThreadMetaRow.thread_id == thread_id).values(user_id=owner_user_id, updated_at=datetime.now(UTC)))
            await session.commit()

    async def delete(
        self,
        thread_id: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> None:
        resolved_user_id = resolve_user_id(user_id, method_name="ThreadMetaRepository.delete")
        async with self._sf() as session:
            row = await session.get(ThreadMetaRow, thread_id)
            if row is None:
                return
            if resolved_user_id is not None and row.user_id != resolved_user_id:
                return
            await session.delete(row)
            await session.commit()
