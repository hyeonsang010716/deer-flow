"""메모리 기반 RunEventStore. run_events.backend=memory(기본값)일 때와 테스트에서 쓴다.

단일 프로세스 async 사용에 대해 thread-safe하다(모든 변경이 같은 event loop 안에서
일어나므로 threading lock이 필요 없다).
"""

from __future__ import annotations

import bisect
from datetime import UTC, datetime

from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.user_context import AUTO, _AutoSentinel


class MemoryRunEventStore(RunEventStore):
    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}  # thread_id -> seq 정렬된 event 목록
        # ``_events``의 message 전용 projection(복사 없이 같은 dict 객체를 공유한다). seq 순서를
        # 유지해서, 요청마다 모든 event를 다시 스캔하는 대신 bisect로 O(log m + page)에
        # message pagination이 끝나게 한다.
        self._messages: dict[str, list[dict]] = {}  # thread_id -> seq 정렬된 message 목록
        # 위 두 목록의 run 단위 projection(복사 없이 같은 dict 객체). 역시 seq 순서를 유지한다.
        # 덕분에 run 단위 읽기 비용이 O(thread의 event 수)가 아니라 O(run의 event 수)가 된다.
        # 이것이 없으면 ``list_events``와 ``list_messages_by_run``은 run 하나에 event가 몇 개
        # 없어도 요청마다 thread 전체 event log를 다시 스캔한다. thread 전역 ``_messages``
        # projection의 run 단위 대응물이다.
        self._events_by_run: dict[str, dict[str, list[dict]]] = {}  # thread_id -> run_id -> seq 정렬된 event
        self._messages_by_run: dict[str, dict[str, list[dict]]] = {}  # thread_id -> run_id -> seq 정렬된 message
        self._seq_counters: dict[str, int] = {}  # thread_id -> 마지막으로 할당한 seq

    def _next_seq(self, thread_id: str) -> int:
        current = self._seq_counters.get(thread_id, 0)
        next_val = current + 1
        self._seq_counters[thread_id] = next_val
        return next_val

    def _put_one(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        category: str,
        content: str | dict = "",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> dict:
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
        self._events.setdefault(thread_id, []).append(record)
        self._events_by_run.setdefault(thread_id, {}).setdefault(run_id, []).append(record)
        if category == "message":
            self._messages.setdefault(thread_id, []).append(record)
            self._messages_by_run.setdefault(thread_id, {}).setdefault(run_id, []).append(record)
        return record

    async def put(
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
        return self._put_one(
            thread_id=thread_id,
            run_id=run_id,
            event_type=event_type,
            category=category,
            content=content,
            metadata=metadata,
            created_at=created_at,
        )

    async def put_batch(self, events):
        results = []
        for ev in events:
            record = self._put_one(**ev)
            results.append(record)
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
        # 조회와 append 사이에 await이 없으므로, 이 backend가 명시한 단일 event loop
        # 동시성 모델에서는 원자적이다.
        for event in self._events_by_run.get(thread_id, {}).get(run_id, []):
            if event["event_type"] == event_type:
                return event, False
        return (
            self._put_one(
                thread_id=thread_id,
                run_id=run_id,
                event_type=event_type,
                category=category,
                content=content,
                metadata=metadata,
                created_at=created_at,
            ),
            True,
        )

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None, user_id: str | None | _AutoSentinel = AUTO):
        # ``messages``는 message만 담고 seq로 정렬되어 있으므로, seq 구간은 전체 스캔이 아니라
        # bisect로 찾는 연속 slice다(O(log m)).
        messages = self._messages.get(thread_id, [])

        if before_seq is not None:
            # seq < before_seq인 record 중 마지막 `limit`개.
            hi = bisect.bisect_left(messages, before_seq, key=lambda e: e["seq"])
            return messages[max(0, hi - limit) : hi]
        elif after_seq is not None:
            # seq > after_seq인 record 중 처음 `limit`개.
            lo = bisect.bisect_right(messages, after_seq, key=lambda e: e["seq"])
            return messages[lo : lo + limit]
        else:
            # 최신 `limit`개를 오름차순으로 반환한다.
            return messages[-limit:]

    async def list_events(self, thread_id, run_id, *, event_types=None, task_id=None, limit=500, after_seq=None):
        # ``_events_by_run``은 이미 이 run 범위로 좁혀져 있고 seq 순서라서, thread 전체를
        # 스캔하지 않고 이 run의 event만 건드린다.
        run_events = self._events_by_run.get(thread_id, {}).get(run_id, [])
        if event_types is not None:
            run_events = [e for e in run_events if e["event_type"] in event_types]
        if task_id is not None:
            run_events = [e for e in run_events if (e.get("metadata") or {}).get("task_id") == task_id]
        if after_seq is not None:
            run_events = [e for e in run_events if e.get("seq", 0) > after_seq]
        return run_events[:limit]

    async def list_messages_by_run(self, thread_id, run_id, *, limit=50, before_seq=None, after_seq=None):
        # run 단위, message 전용, seq 정렬. thread 전체 event log를 다시 스캔하는 대신 이 run의
        # message에 대해서만 bisect로 연속 slice를 찾는다(O(log m_run)).
        messages = self._messages_by_run.get(thread_id, {}).get(run_id, [])
        lo = 0 if after_seq is None else bisect.bisect_right(messages, after_seq, key=lambda e: e["seq"])
        hi = len(messages) if before_seq is None else bisect.bisect_left(messages, before_seq, key=lambda e: e["seq"])
        window = messages[lo:hi]
        # ``after_seq`` cursor는 앞으로 페이징한다(처음 ``limit``개). 그 외에는 마지막
        # ``limit``개를 반환한다(최신 페이지, 또는 ``before_seq`` 직전에서 끝나는 페이지).
        # 기존 filter 기반 동작과 동일하다.
        if after_seq is not None:
            return window[:limit]
        return window[-limit:]

    async def get_last_visible_ai_seq_by_run(self, thread_id, run_ids, *, user_id: str | None | _AutoSentinel = AUTO):
        result: dict[str, int] = {}
        messages_by_run = self._messages_by_run.get(thread_id, {})
        for run_id in run_ids:
            for event in reversed(messages_by_run.get(run_id, [])):
                caller = str((event.get("metadata") or {}).get("caller", ""))
                if event.get("category") == "message" and event.get("event_type") in {"llm.ai.response", "ai_message"} and not caller.startswith("middleware:"):
                    result[run_id] = event["seq"]
                    break
        return result

    async def count_messages(self, thread_id):
        return len(self._messages.get(thread_id, []))

    async def delete_by_thread(self, thread_id):
        events = self._events.pop(thread_id, [])
        self._messages.pop(thread_id, None)
        self._events_by_run.pop(thread_id, None)
        self._messages_by_run.pop(thread_id, None)
        self._seq_counters.pop(thread_id, None)
        return len(events)

    async def delete_by_run(self, thread_id, run_id):
        all_events = self._events.get(thread_id, [])
        if not all_events:
            return 0
        remaining = [e for e in all_events if e["run_id"] != run_id]
        removed = len(all_events) - len(remaining)
        self._events[thread_id] = remaining
        # message projection을 같은 상태로 맞춘다(살아남은 동일 dict 객체를 그대로 쓴다).
        self._messages[thread_id] = [e for e in remaining if e["category"] == "message"]
        # 삭제된 run을 run 단위 projection에서 제거한다.
        self._events_by_run.get(thread_id, {}).pop(run_id, None)
        self._messages_by_run.get(thread_id, {}).pop(run_id, None)
        return removed
