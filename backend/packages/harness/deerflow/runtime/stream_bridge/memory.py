"""프로세스 내 event log를 백엔드로 쓰는 in-memory stream bridge."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent, StreamGap, StreamItem

logger = logging.getLogger(__name__)
_MEMORY_STREAM_ID_RE = re.compile(r"\d+-(\d+)")


@dataclass
class _RunStream:
    events: list[StreamEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    ended: bool = False
    start_offset: int = 0


class MemoryStreamBridge(StreamBridge):
    """run별 in-memory event log 구현.

    event는 run마다 제한된 구간만 보관하므로, 늦게 붙은 subscriber나 재접속한 client가
    ``Last-Event-ID``부터 버퍼된 event를 replay할 수 있다.
    """

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        self._maxsize = queue_maxsize
        self._streams: dict[str, _RunStream] = {}
        self._counters: dict[str, int] = {}

    # -- 헬퍼 -------------------------------------------------------------------

    def _get_or_create_stream(self, run_id: str) -> _RunStream:
        if run_id not in self._streams:
            self._streams[run_id] = _RunStream()
            self._counters[run_id] = 0
        return self._streams[run_id]

    def _next_id(self, run_id: str) -> str:
        self._counters[run_id] = self._counters.get(run_id, 0) + 1
        ts = int(time.time() * 1000)
        seq = self._counters[run_id] - 1
        return f"{ts}-{seq}"

    @staticmethod
    def _parse_event_seq(event_id: str) -> int | None:
        """``{ts}-{seq}`` 형식의 event id에서 run별 sequence 번호를 추출한다.

        :meth:`_next_id`가 부여하는 ``seq``는 publish된 event마다 1씩 증가하므로, run 안에서
        event의 절대 offset과 같다. 형식이 맞지 않는 id에는 ``None``을 반환한다.
        """
        match = _MEMORY_STREAM_ID_RE.fullmatch(event_id)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _make_gap(stream: _RunStream, requested_event_id: str | None) -> StreamGap:
        return StreamGap(
            requested_event_id=requested_event_id,
            earliest_available_event_id=stream.events[0].id,
            latest_available_event_id=stream.events[-1].id,
        )

    def _resolve_start_offset(self, stream: _RunStream, last_event_id: str | None) -> int | StreamGap:
        if last_event_id is None:
            return stream.start_offset

        # event id에는 run 안에서 단조 증가하며 event의 절대 offset과 같은 ``seq``가 들어 있다.
        # 따라서 보관 버퍼를 훑는 대신 산술 계산으로 O(1)에 event를 찾는다. 보관된 id는 계산된
        # index에서 검증한다. id가 보관 watermark 아래로 내려가면 timestamp를 대조할 대상이 없으므로,
        # 숫자 형태의 외부 id라도 보수적으로 gap 경로를 탄다. 완전한 replay라고 조용히 주장하는 것보다
        # durable state를 다시 읽는 편이 안전하다. watermark 이상인 알 수 없는 id는 기존의
        # "가장 오래된 것부터 replay" 동작을 유지한다.
        seq = self._parse_event_seq(last_event_id)
        if seq is not None:
            if stream.events and seq < stream.start_offset:
                return self._make_gap(stream, last_event_id)
            local_index = seq - stream.start_offset
            if 0 <= local_index < len(stream.events) and stream.events[local_index].id == last_event_id:
                return stream.start_offset + local_index + 1

        if stream.events:
            logger.warning(
                "last_event_id=%s not found in retained buffer; replaying from earliest retained event",
                last_event_id,
            )
        return stream.start_offset

    async def stream_exists(self, run_id: str) -> bool:
        """프로세스 내 event log에 *run_id*의 데이터가 아직 남아 있는지 반환한다."""
        return run_id in self._streams

    # -- StreamBridge API -------------------------------------------------------

    async def publish(self, run_id: str, event: str, data: Any) -> None:
        stream = self._get_or_create_stream(run_id)
        entry = StreamEvent(id=self._next_id(run_id), event=event, data=data)
        async with stream.condition:
            stream.events.append(entry)
            if len(stream.events) > self._maxsize:
                overflow = len(stream.events) - self._maxsize
                del stream.events[:overflow]
                stream.start_offset += overflow
            stream.condition.notify_all()

    async def publish_end(self, run_id: str) -> None:
        stream = self._get_or_create_stream(run_id)
        async with stream.condition:
            stream.ended = True
            stream.condition.notify_all()

    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamItem]:
        stream = self._get_or_create_stream(run_id)
        async with stream.condition:
            start = self._resolve_start_offset(stream, last_event_id)
            if isinstance(start, StreamGap):
                gap = start
                next_offset = stream.start_offset
            else:
                gap = None
                next_offset = start

        if gap is not None:
            yield gap
            return

        cursor_event_id = last_event_id

        while True:
            async with stream.condition:
                if next_offset < stream.start_offset:
                    logger.warning(
                        "subscriber for run %s fell behind retained buffer at offset %s",
                        run_id,
                        next_offset,
                    )
                    entry: StreamItem = self._make_gap(stream, cursor_event_id)
                    should_stop = True
                else:
                    should_stop = False

                    local_index = next_offset - stream.start_offset
                    if 0 <= local_index < len(stream.events):
                        entry = stream.events[local_index]
                        next_offset += 1
                        cursor_event_id = entry.id
                    elif stream.ended:
                        entry = END_SENTINEL
                    else:
                        try:
                            await asyncio.wait_for(stream.condition.wait(), timeout=heartbeat_interval)
                        except TimeoutError:
                            entry = HEARTBEAT_SENTINEL
                        else:
                            continue

            if entry is END_SENTINEL:
                yield END_SENTINEL
                return
            yield entry
            if should_stop:
                return

    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        self._streams.pop(run_id, None)
        self._counters.pop(run_id, None)

    async def close(self) -> None:
        self._streams.clear()
        self._counters.clear()
