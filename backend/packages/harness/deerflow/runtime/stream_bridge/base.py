"""추상 stream bridge 프로토콜.

StreamBridge는 agent worker(producer)와 SSE endpoint(consumer)를 분리하며,
LangGraph Platform의 Queue + StreamManager 구조와 정렬된다.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    """단일 stream event.

    Attributes:
        id: 단조 증가하는 event ID(SSE ``id:`` 필드로 쓰이며 ``Last-Event-ID``
            재접속을 지원한다).
        event: SSE event 이름. 예: ``"metadata"``, ``"updates"``, ``"events"``,
            ``"error"``, ``"end"``.
        data: JSON 직렬화 가능한 payload.
    """

    id: str
    event: str
    data: Any


@dataclass(frozen=True)
class StreamGap:
    """subscriber cursor를 더 이상 완전히 replay할 수 없다.

    ``requested_event_id``는 재접속 cursor이거나, 뒤처진 live subscriber에게 마지막으로
    전달된 event다. 보존 범위를 함께 알려주므로 호출자는 부분 replay를 완전한 replay로
    착각하지 않고, durable state를 다시 로드해 현재 tail에서 재개할 수 있다.
    """

    requested_event_id: str | None
    earliest_available_event_id: str
    latest_available_event_id: str


HEARTBEAT_SENTINEL = StreamEvent(id="", event="__heartbeat__", data=None)
END_SENTINEL = StreamEvent(id="", event="__end__", data=None)
type StreamItem = StreamEvent | StreamGap


class StreamBridge(abc.ABC):
    """stream bridge의 추상 베이스."""

    supports_cross_process: bool = False

    @abc.abstractmethod
    async def publish(self, run_id: str, event: str, data: Any) -> None:
        """*run_id*에 대한 event 하나를 큐에 넣는다(producer 쪽)."""

    @abc.abstractmethod
    async def publish_end(self, run_id: str) -> None:
        """*run_id*에 대해 더 이상 event가 생성되지 않음을 알린다."""

    @abc.abstractmethod
    def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamItem]:
        """*run_id*의 event를 내보내는 async iterator(consumer 쪽).

        *heartbeat_interval*초 안에 event가 오지 않으면 :data:`HEARTBEAT_SENTINEL`을
        내보낸다. producer가 :meth:`publish_end`를 호출하면 :data:`END_SENTINEL`을
        내보낸다. subscriber가 보존된 히스토리보다 뒤처지면 :class:`StreamGap`을 내보내고
        멈춘다.
        """

    @abc.abstractmethod
    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        """*run_id*와 연결된 리소스를 해제한다.

        *delay* > 0이면 구현체는 해제 전에 대기해서, 늦게 붙은 subscriber가 남은 event를
        모두 소비할 기회를 준다.
        """

    async def close(self) -> None:
        """backend 리소스를 해제한다. 기본 구현은 아무 일도 하지 않는다."""
