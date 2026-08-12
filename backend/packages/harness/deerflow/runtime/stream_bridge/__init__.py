"""Stream bridge — agent worker와 SSE endpoint를 분리한다.

``StreamBridge``는 agent를 실행하는 백그라운드 task(producer)와 클라이언트로 Server-Sent
Events를 밀어 주는 HTTP endpoint(consumer) 사이에 놓인다. 이 패키지는 추상 protocol
(:class:`StreamBridge`)과 :mod:`asyncio.Queue` 기반의 기본 in-memory 구현을 제공한다.
"""

from .async_provider import make_stream_bridge
from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent, StreamGap, StreamItem
from .memory import MemoryStreamBridge

# NOTE: ``RedisStreamBridge``는 의도적으로 여기서 import하지 않는다. ``redis``는 선택적
# extra이고, 이 패키지는 프로세스 시작 시 어디서나 ``deerflow.runtime``을 통해 전이적으로
# 딸려 온다. ``.redis``를 즉시 import하면 모든 프로세스(메모리 전용/단일 프로세스 환경 포함)가
# ``redis.asyncio``를 import하게 되고, 모든 설치가 redis 패키지에 묶인다. ``stream_bridge.type
# == "redis"``일 때만 ``make_stream_bridge`` 안에서 지연 import한다. 클래스가 필요하면
# ``deerflow.runtime.stream_bridge.redis``에서 직접 import한다.

__all__ = [
    "END_SENTINEL",
    "HEARTBEAT_SENTINEL",
    "MemoryStreamBridge",
    "StreamBridge",
    "StreamEvent",
    "StreamGap",
    "StreamItem",
    "make_stream_bridge",
]
