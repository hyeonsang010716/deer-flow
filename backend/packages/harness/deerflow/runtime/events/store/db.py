"""SQLAlchemy 기반 RunEventStore 구현.

이벤트를 ``run_events`` 테이블에 저장한다. 데이터베이스가 비대해지지 않도록 trace 내용은
``max_trace_content`` 바이트에서 잘린다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.models.run_event import RunEventRow
from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.user_context import AUTO, _AutoSentinel, get_current_user, resolve_user_id
from deerflow.utils.time import coerce_iso

logger = logging.getLogger(__name__)


class DbRunEventStore(RunEventStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, max_trace_content: int = 10240):
        self._sf = session_factory
        self._max_trace_content = max_trace_content
        # thread별 asyncio lock이 같은 thread에 동시에 쓰는 in-process writer들의 seq 할당을
        # 직렬화한다. 프로세스 간 race는 DB 수준의 FOR UPDATE / advisory lock이 막고, 이
        # lock은 두 coroutine이 max(seq) 읽기와 INSERT 사이에 끼어들어 seq가 충돌하는 흔한
        # 단일 프로세스 상황을 막는다.
        self._write_locks: dict[str, asyncio.Lock] = {}

    def _get_write_lock(self, thread_id: str) -> asyncio.Lock:
        """thread별 seq 할당 lock을 반환한다. 없으면 생성한다."""
        lock = self._write_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._write_locks[thread_id] = lock
        return lock

    @staticmethod
    def _row_to_dict(row: RunEventRow) -> dict:
        d = row.to_dict()
        d["metadata"] = d.pop("event_metadata", {})
        val = d.get("created_at")
        if isinstance(val, datetime):
            # SQLite는 ``DateTime(timezone=True)``에도 불구하고 읽을 때 tzinfo를 버리므로,
            # ``coerce_iso``가 naive datetime을 UTC로 정규화한다.
            d["created_at"] = coerce_iso(val)
        d.pop("id", None)
        # 쓸 때 JSON으로 직렬화했던 구조적 content를 복원한다.
        raw = d.get("content", "")
        metadata = d.get("metadata", {})
        if isinstance(raw, str) and (metadata.get("content_is_json") or metadata.get("content_is_dict")):
            try:
                d["content"] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # JSON처럼 보였지만 파싱에 실패했다. raw 문자열을 그대로 둔다.
                logger.debug("Failed to deserialize content as JSON for event seq=%s", d.get("seq"))
        return d

    def _truncate_trace(self, category: str, content: Any, metadata: dict | None) -> tuple[Any, dict]:
        if category == "trace":
            text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
            encoded = text.encode("utf-8")
            if len(encoded) > self._max_trace_content:
                # 바이트 단위로 자른 뒤 다시 디코딩한다(멀티바이트 문자가 잘릴 수 있어 errors="ignore" 사용).
                content = encoded[: self._max_trace_content].decode("utf-8", errors="ignore")
                metadata = {**(metadata or {}), "content_truncated": True, "original_byte_length": len(encoded)}
        return content, metadata or {}

    @staticmethod
    def _content_to_db(content: Any, metadata: dict | None) -> tuple[str, dict]:
        metadata = metadata or {}
        if isinstance(content, str):
            return content, metadata

        db_content = json.dumps(content, default=str, ensure_ascii=False)
        metadata = {**metadata, "content_is_json": True}
        if isinstance(content, dict):
            metadata["content_is_dict"] = True
        return db_content, metadata

    @staticmethod
    def _user_id_from_context() -> str | None:
        """write 경로를 위해 contextvar에서 user_id를 느슨하게 읽는다.

        contextvar가 설정돼 있지 않으면 ``None``(필터 없음 / 기록 없음)을 반환한다.
        background worker write에서는 이것이 정상적인 상황이다. HTTP 요청 write는 auth
        middleware가 contextvar를 설정하므로 user_id가 자동으로 기록된다.

        경계에서 ``user.id``를 ``str``로 강제 변환한다. auth 레이어는 ``User.id``를
        ``UUID``로 타이핑하지만 ``run_events.user_id``는 ``VARCHAR(64)``이고, aiosqlite는
        raw UUID 객체를 VARCHAR 컬럼에 바인딩하지 못한다("type 'UUID' is not supported").
        그러면 INSERT가 조용히 롤백되고 worker가 멈춘다.
        """
        user = get_current_user()
        return str(user.id) if user is not None else None

    @staticmethod
    async def _max_seq_for_thread(session: AsyncSession, thread_id: str) -> int | None:
        """thread별 writer를 직렬화하면서 현재 max seq를 반환한다.

        집계 결과는 잠글 수 있는 row가 아니므로 PostgreSQL은 ``SELECT max(...) FOR UPDATE``를
        거부한다. 릴리스에 안전한 우회책으로, 집계를 읽기 전에 thread_id를 키로 한
        transaction 수준 advisory lock을 잡는다. 다른 dialect는 기존 row 잠금 구문을 쓴다.
        """
        stmt = select(func.max(RunEventRow.seq)).where(RunEventRow.thread_id == thread_id)
        bind = session.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""

        if dialect_name == "postgresql":
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(CAST(:thread_id AS text))::bigint)"),
                {"thread_id": thread_id},
            )
            return await session.scalar(stmt)

        return await session.scalar(stmt.with_for_update())

    async def put(self, *, thread_id, run_id, event_type, category, content="", metadata=None, created_at=None):  # noqa: D401
        """이벤트 하나를 기록한다. 저빈도 경로 전용이다.

        단조 증가하는 *seq*를 할당하기 위해 FOR UPDATE lock을 건 전용 transaction을 연다.
        처리량이 큰 write는 batch 전체에 대해 lock을 한 번만 잡는 :meth:`put_batch`를 쓴다.
        현재 유일한 caller는 최초 ``human_message`` 이벤트를 기록하는 ``worker.run_agent``다
        (run당 한 번).
        """
        content, metadata = self._truncate_trace(category, content, metadata)
        db_content, metadata = self._content_to_db(content, metadata)
        user_id = self._user_id_from_context()
        async with self._get_write_lock(thread_id):
            async with self._sf() as session:
                async with session.begin():
                    max_seq = await self._max_seq_for_thread(session, thread_id)
                    seq = (max_seq or 0) + 1
                    row = RunEventRow(
                        thread_id=thread_id,
                        run_id=run_id,
                        user_id=user_id,
                        event_type=event_type,
                        category=category,
                        content=db_content,
                        event_metadata=metadata,
                        seq=seq,
                        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                    )
                    session.add(row)
                return self._row_to_dict(row)

    async def put_batch(self, events):
        if not events:
            return []
        thread_ids = {e["thread_id"] for e in events}
        if len(thread_ids) > 1:
            raise ValueError(f"put_batch requires all events to belong to the same thread; got {thread_ids!r}")
        user_id = self._user_id_from_context()
        # 모든 이벤트는 같은 thread에 속한다(위에서 검증했다).
        thread_id = events[0]["thread_id"]
        async with self._get_write_lock(thread_id):
            async with self._sf() as session:
                async with session.begin():
                    max_seq = await self._max_seq_for_thread(session, thread_id)
                    seq = max_seq or 0
                    rows = []
                    for e in events:
                        seq += 1
                        content = e.get("content", "")
                        category = e.get("category", "trace")
                        metadata = e.get("metadata")
                        content, metadata = self._truncate_trace(category, content, metadata)
                        db_content, metadata = self._content_to_db(content, metadata)
                        row = RunEventRow(
                            thread_id=e["thread_id"],
                            run_id=e["run_id"],
                            user_id=e.get("user_id", user_id),
                            event_type=e["event_type"],
                            category=category,
                            content=db_content,
                            event_metadata=metadata,
                            seq=seq,
                            created_at=datetime.fromisoformat(e["created_at"]) if e.get("created_at") else datetime.now(UTC),
                        )
                        session.add(row)
                        rows.append(row)
                return [self._row_to_dict(r) for r in rows]

    async def put_if_absent(
        self,
        *,
        thread_id,
        run_id,
        event_type,
        category,
        content="",
        metadata=None,
        created_at=None,
    ):
        """run 범위의 singleton 이벤트를 idempotent하게 삽입한다.

        ``_max_seq_for_thread``가 모든 일반 writer와 같은 PostgreSQL advisory lock을 잡고
        (SQLite는 in-process lock이 담당하므로), 존재 여부 검사가 다른 ``put_if_absent``나
        journal write와 race하지 않는다. terminal delivery receipt가 worker와 recovery 양쪽
        경로에서 이 메서드를 쓰며, 일반 이벤트 타입은 여전히 append-only다.
        """
        content, metadata = self._truncate_trace(category, content, metadata)
        db_content, metadata = self._content_to_db(content, metadata)
        user_id = self._user_id_from_context()
        async with self._get_write_lock(thread_id):
            async with self._sf() as session:
                async with session.begin():
                    max_seq = await self._max_seq_for_thread(session, thread_id)
                    stmt = (
                        select(RunEventRow)
                        .where(
                            RunEventRow.thread_id == thread_id,
                            RunEventRow.run_id == run_id,
                            RunEventRow.event_type == event_type,
                        )
                        .order_by(RunEventRow.seq.asc())
                        .limit(1)
                    )
                    existing = await session.scalar(stmt)
                    if existing is not None:
                        return self._row_to_dict(existing), False
                    row = RunEventRow(
                        thread_id=thread_id,
                        run_id=run_id,
                        user_id=user_id,
                        event_type=event_type,
                        category=category,
                        content=db_content,
                        event_metadata=metadata,
                        seq=(max_seq or 0) + 1,
                        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                    )
                    session.add(row)
                return self._row_to_dict(row), True

    async def list_messages(
        self,
        thread_id,
        *,
        limit=50,
        before_seq=None,
        after_seq=None,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_messages")
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if before_seq is not None:
            stmt = stmt.where(RunEventRow.seq < before_seq)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)

        if after_seq is not None:
            # 정방향 pagination: cursor 이후 첫 `limit`개 레코드
            stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                return [self._row_to_dict(r) for r in result.scalars()]
        else:
            # before_seq 또는 기본값(최신): 마지막 `limit`개를 가져와 오름차순으로 반환한다.
            stmt = stmt.order_by(RunEventRow.seq.desc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                rows = list(result.scalars())
                return [self._row_to_dict(r) for r in reversed(rows)]

    async def list_events(
        self,
        thread_id,
        run_id,
        *,
        event_types=None,
        task_id=None,
        limit=500,
        after_seq=None,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_events")
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id)
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if event_types:
            stmt = stmt.where(RunEventRow.event_type.in_(event_types))
        if task_id is not None:
            # 단일 subagent task에 대한 cursor pagination이 정확하도록 metadata["task_id"]
            # 필터를 (LIMIT 이전에) SQL에서 적용한다(#3779). 쿼리는 이미
            # (thread_id, run_id)로 좁혀져 있어 JSON probe는 이 run의 작은 후보 집합에만
            # 수행된다. ``.as_string()``은 json_extract(SQLite) / ->>(Postgres)로 렌더된다.
            stmt = stmt.where(RunEventRow.event_metadata["task_id"].as_string() == task_id)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)
        stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(r) for r in result.scalars()]

    async def list_messages_by_run(
        self,
        thread_id,
        run_id,
        *,
        limit=50,
        before_seq=None,
        after_seq=None,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_messages_by_run")
        stmt = select(RunEventRow).where(
            RunEventRow.thread_id == thread_id,
            RunEventRow.run_id == run_id,
            RunEventRow.category == "message",
        )
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if before_seq is not None:
            stmt = stmt.where(RunEventRow.seq < before_seq)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)

        if after_seq is not None:
            stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                return [self._row_to_dict(r) for r in result.scalars()]
        else:
            stmt = stmt.order_by(RunEventRow.seq.desc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                rows = list(result.scalars())
                return [self._row_to_dict(r) for r in reversed(rows)]

    async def get_last_visible_ai_seq_by_run(
        self,
        thread_id,
        run_ids,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        if not run_ids:
            return {}
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.get_last_visible_ai_seq_by_run")
        caller = RunEventRow.event_metadata["caller"].as_string()
        # RunJournal은 AI message row를 표준적으로 ``llm.ai.response``로 저장한다.
        # ``ai_message``는 하위 호환을 위해 남아 있다.
        stmt = (
            select(RunEventRow.run_id, func.max(RunEventRow.seq))
            .where(
                RunEventRow.thread_id == thread_id,
                RunEventRow.run_id.in_(run_ids),
                RunEventRow.category == "message",
                RunEventRow.event_type.in_(("llm.ai.response", "ai_message")),
                ~func.coalesce(caller, "").like("middleware:%"),
            )
            .group_by(RunEventRow.run_id)
        )
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return {run_id: seq for run_id, seq in result if isinstance(seq, int)}

    async def count_messages(
        self,
        thread_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.count_messages")
        stmt = select(func.count()).select_from(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        async with self._sf() as session:
            return await session.scalar(stmt) or 0

    async def delete_by_thread(
        self,
        thread_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.delete_by_thread")
        async with self._sf() as session:
            count_conditions = [RunEventRow.thread_id == thread_id]
            if resolved_user_id is not None:
                count_conditions.append(RunEventRow.user_id == resolved_user_id)
            count_stmt = select(func.count()).select_from(RunEventRow).where(*count_conditions)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(*count_conditions))
                await session.commit()
            # (오래 사는 singleton) store의 생애 동안 ``_write_locks``가 무한히 커지지
            # 않도록 thread별 seq 할당 lock을 제거한다. 진행 중인 writer가 없을 때만
            # 제거하며, 이후 write가 lock을 lazy하게 다시 만들고 seq는 이제 삭제된 thread
            # 기준으로 올바르게 다시 시작한다.
            lock = self._write_locks.get(thread_id)
            if lock is not None and not lock.locked():
                self._write_locks.pop(thread_id, None)
            return count

    async def delete_by_run(
        self,
        thread_id,
        run_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.delete_by_run")
        async with self._sf() as session:
            count_conditions = [RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id]
            if resolved_user_id is not None:
                count_conditions.append(RunEventRow.user_id == resolved_user_id)
            count_stmt = select(func.count()).select_from(RunEventRow).where(*count_conditions)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(*count_conditions))
                await session.commit()
            return count
