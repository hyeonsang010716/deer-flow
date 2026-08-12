"""JSONL 파일 기반 RunEventStore 구현.

각 run의 event는 파일 하나에 저장된다:
``.deer-flow/threads/{thread_id}/runs/{run_id}.jsonl``

모든 category(message, trace, lifecycle)가 같은 파일에 들어간다. 이 backend는 가벼운
단일 노드 배포에 적합하다.

**단일 프로세스 보장**: 메모리상의 seq counter는 프로세스 로컬이다. 같은 디렉터리를 공유하는
멀티 프로세스 배포에서는 seq 값이 중복되거나 단조 증가하지 않는다. 멀티 프로세스나 고동시성
배포에는 ``DbRunEventStore``를 쓴다.

파일 I/O는 ``asyncio.to_thread``로 thread pool에 offload하므로 event loop가 절대 막히지
않는다. thread별 ``asyncio.Lock``이 한 프로세스 안의 write를 직렬화해 JSONL 줄이 섞이는 것을
막는다.

알려진 trade-off: 여러 run의 message가 통합된 seq 순서를 필요로 하므로 ``list_messages()``는
해당 thread의 모든 run 파일을 스캔해야 한다. ``list_events()``는 파일 하나만 읽는 fast path다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.user_context import AUTO, _AutoSentinel
from deerflow.utils.thread_id import validate_thread_id

logger = logging.getLogger(__name__)

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


class JsonlRunEventStore(RunEventStore):
    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else Path(".deer-flow")
        self._seq_counters: dict[str, int] = {}  # thread_id -> 현재 최대 seq
        # thread별 asyncio.Lock — 한 프로세스 안의 동시 write를 직렬화한다.
        self._write_locks: dict[str, asyncio.Lock] = {}

    def _get_write_lock(self, thread_id: str) -> asyncio.Lock:
        return self._write_locks.setdefault(thread_id, asyncio.Lock())

    @staticmethod
    def _validate_id(value: str, label: str) -> str:
        """ID를 파일시스템 경로에 써도 안전한지 검증한다."""
        if not value or not _SAFE_ID_PATTERN.match(value):
            raise ValueError(f"Invalid {label}: must be alphanumeric/dash/underscore, got {value!r}")
        return value

    def _thread_dir(self, thread_id: str) -> Path:
        validate_thread_id(thread_id)
        return self._base_dir / "threads" / thread_id / "runs"

    def _run_file(self, thread_id: str, run_id: str) -> Path:
        self._validate_id(run_id, "run_id")
        return self._thread_dir(thread_id) / f"{run_id}.jsonl"

    def _next_seq(self, thread_id: str) -> int:
        self._seq_counters[thread_id] = self._seq_counters.get(thread_id, 0) + 1
        return self._seq_counters[thread_id]

    def _compute_max_seq(self, thread_id: str) -> int:
        """thread의 모든 run 파일을 스캔해 현재 최대 seq를 반환한다(blocking I/O)."""
        max_seq = 0
        thread_dir = self._thread_dir(thread_id)
        if thread_dir.exists():
            for f in thread_dir.glob("*.jsonl"):
                for line in f.read_text(encoding="utf-8").strip().splitlines():
                    try:
                        record = json.loads(line)
                        max_seq = max(max_seq, record.get("seq", 0))
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSONL line in %s", f)
        return max_seq

    async def _ensure_seq_loaded(self, thread_id: str) -> None:
        """기존 파일에서 최대 seq를 읽어 메모리 counter에 적재한다(non-blocking)."""
        if thread_id in self._seq_counters:
            return
        max_seq = await asyncio.to_thread(self._compute_max_seq, thread_id)
        self._seq_counters[thread_id] = max_seq

    def _write_record(self, record: dict) -> None:
        path = self._run_file(record["thread_id"], record["run_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def _read_thread_events(self, thread_id: str) -> list[dict]:
        """thread의 모든 event를 읽어 seq 순으로 정렬해 반환한다(blocking I/O)."""
        events = []
        thread_dir = self._thread_dir(thread_id)
        if not thread_dir.exists():
            return events
        for f in sorted(thread_dir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").strip().splitlines():
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSONL line in %s", f)
        events.sort(key=lambda e: e.get("seq", 0))
        return events

    def _read_run_events(self, thread_id: str, run_id: str) -> list[dict]:
        """특정 run 파일의 event를 읽는다(blocking I/O)."""
        path = self._run_file(thread_id, run_id)
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL line in %s", path)
        events.sort(key=lambda e: e.get("seq", 0))
        return events

    def _delete_thread_files(self, thread_id: str) -> None:
        thread_dir = self._thread_dir(thread_id)
        if thread_dir.exists():
            for f in thread_dir.glob("*.jsonl"):
                f.unlink()

    def _delete_run_file(self, thread_id: str, run_id: str) -> None:
        path = self._run_file(thread_id, run_id)
        if path.exists():
            path.unlink()

    async def put(self, *, thread_id, run_id, event_type, category, content="", metadata=None, created_at=None):
        async with self._get_write_lock(thread_id):
            await self._ensure_seq_loaded(thread_id)
            seq = self._next_seq(thread_id)
            record = {
                "thread_id": thread_id,
                "run_id": run_id,
                "event_type": event_type,
                "category": category,
                "content": content,
                "metadata": metadata or {},
                "seq": seq,
                "created_at": created_at or datetime.now(UTC).isoformat(),
            }
            await asyncio.to_thread(self._write_record, record)
            return record

    async def put_batch(self, events):
        """event batch를 thread 단위로 원자적으로 저장한다.

        batch의 모든 seq 번호는 thread별 write lock 하나 아래에서 예약되고, 모든 record는 파일
        write 한 번으로 append된다. 그래서 batch 도중 실패해도 디스크에 일부 record만 남아
        retry가 이를 중복시키는 일이 없다. 호출자(예: worker.py의 flush-retry 경로)는 실패 시
        batch 전체를 안전하게 다시 버퍼링할 수 있다.
        """
        if not events:
            return []

        # thread_id로 묶는다. 각 thread는 자체 write lock과 seq counter를 갖는다.
        by_thread: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            by_thread.setdefault(ev["thread_id"], []).append(ev)

        results: list[dict[str, Any]] = []
        for thread_id, batch in by_thread.items():
            records = await self._write_batch_async(thread_id, batch)
            results.extend(records)
        return results

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
        async with self._get_write_lock(thread_id):
            existing = await asyncio.to_thread(self._read_run_events, thread_id, run_id)
            for event in existing:
                if event.get("event_type") == event_type:
                    return event, False
            await self._ensure_seq_loaded(thread_id)
            record = {
                "thread_id": thread_id,
                "run_id": run_id,
                "event_type": event_type,
                "category": category,
                "content": content,
                "metadata": metadata or {},
                "seq": self._next_seq(thread_id),
                "created_at": created_at or datetime.now(UTC).isoformat(),
            }
            await asyncio.to_thread(self._write_record, record)
            return record, True

    async def _write_batch_async(self, thread_id: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        async with self._get_write_lock(thread_id):
            await self._ensure_seq_loaded(thread_id)
            records: list[dict[str, Any]] = []
            for ev in batch:
                seq = self._next_seq(thread_id)
                record = {
                    "thread_id": thread_id,
                    "run_id": ev["run_id"],
                    "event_type": ev["event_type"],
                    "category": ev["category"],
                    "content": ev.get("content", ""),
                    "metadata": ev.get("metadata") or {},
                    "seq": seq,
                    "created_at": ev.get("created_at") or datetime.now(UTC).isoformat(),
                }
                records.append(record)
            path = self._run_file(thread_id, batch[0]["run_id"])
            # thread당 append/write 한 번. 여기서 예외가 나면 저장된 record가 하나도 없으므로,
            # 호출자가 다시 버퍼링해도 중복이 생기지 않는다.
            await asyncio.to_thread(self._append_records, path, records)
            return records

    def _append_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in records)
        with open(path, "a", encoding="utf-8") as f:
            f.write(lines)

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None, user_id: str | None | _AutoSentinel = AUTO):
        all_events = await asyncio.to_thread(self._read_thread_events, thread_id)
        messages = [e for e in all_events if e.get("category") == "message"]

        if before_seq is not None:
            messages = [e for e in messages if e["seq"] < before_seq]
            return messages[-limit:]
        elif after_seq is not None:
            messages = [e for e in messages if e["seq"] > after_seq]
            return messages[:limit]
        else:
            return messages[-limit:]

    async def list_events(self, thread_id, run_id, *, event_types=None, task_id=None, limit=500, after_seq=None):
        events = await asyncio.to_thread(self._read_run_events, thread_id, run_id)
        if event_types is not None:
            events = [e for e in events if e.get("event_type") in event_types]
        if task_id is not None:
            events = [e for e in events if (e.get("metadata") or {}).get("task_id") == task_id]
        if after_seq is not None:
            events = [e for e in events if e.get("seq", 0) > after_seq]
        return events[:limit]

    async def list_messages_by_run(self, thread_id, run_id, *, limit=50, before_seq=None, after_seq=None):
        events = await asyncio.to_thread(self._read_run_events, thread_id, run_id)
        filtered = [e for e in events if e.get("category") == "message"]
        if before_seq is not None:
            filtered = [e for e in filtered if e.get("seq", 0) < before_seq]
        if after_seq is not None:
            filtered = [e for e in filtered if e.get("seq", 0) > after_seq]
        if after_seq is not None:
            return filtered[:limit]
        else:
            return filtered[-limit:] if len(filtered) > limit else filtered

    async def get_last_visible_ai_seq_by_run(self, thread_id, run_ids, *, user_id: str | None | _AutoSentinel = AUTO):
        def _scan() -> dict[str, int]:
            result: dict[str, int] = {}
            for run_id in run_ids:
                for event in reversed(self._read_run_events(thread_id, run_id)):
                    caller = str((event.get("metadata") or {}).get("caller", ""))
                    if event.get("category") == "message" and event.get("event_type") in {"llm.ai.response", "ai_message"} and not caller.startswith("middleware:"):
                        result[run_id] = event["seq"]
                        break
            return result

        return await asyncio.to_thread(_scan)

    async def count_messages(self, thread_id):
        all_events = await asyncio.to_thread(self._read_thread_events, thread_id)
        return sum(1 for e in all_events if e.get("category") == "message")

    async def delete_by_thread(self, thread_id):
        async with self._get_write_lock(thread_id):
            all_events = await asyncio.to_thread(self._read_thread_events, thread_id)
            count = len(all_events)
            await asyncio.to_thread(self._delete_thread_files, thread_id)
            self._seq_counters.pop(thread_id, None)
            # lock을 잡고 있는 범위 안에서 pop한다. 대기 중인 coroutine이 아직 옛 lock을 쥔
            # 상태에서 새 호출자가 새 lock을 얻는 구간을 최소화하기 위해서다.
            # 참고: 삭제 전에 이미 이 lock의 참조를 획득한 coroutine은 우리가 해제한 뒤에도
            # 그대로 진행한다. 이는 감수하는 좁은 race다.
            self._write_locks.pop(thread_id, None)
            return count

    async def delete_by_run(self, thread_id, run_id):
        async with self._get_write_lock(thread_id):
            events = await asyncio.to_thread(self._read_run_events, thread_id, run_id)
            count = len(events)
            await asyncio.to_thread(self._delete_run_file, thread_id, run_id)
            return count
