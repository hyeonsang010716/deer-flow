"""Redis Streams 기반 stream bridge."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

try:
    from redis.asyncio import Redis
    from redis.exceptions import RedisError, ResponseError
except ImportError:  # pragma: no cover - optional extra가 없을 때만 도달한다
    # ``redis``는 optional extra다(persistence/engine.py의 ``postgres``/asyncpg 경로와 동일).
    # 이 모듈은 ``stream_bridge.type == "redis"``일 때만 ``make_stream_bridge``에서 lazy하게
    # import되므로, 패키지 없이 Redis bridge를 요청한 경우에만 이 안내가 노출된다.
    raise ImportError(
        "stream_bridge.type is set to 'redis' but the redis package is not installed.\n"
        "Install it with:\n"
        "    cd backend && uv sync --all-packages --extra redis\n"
        "On the next `make dev` the redis extra is auto-detected from config.yaml\n"
        "(stream_bridge.type: redis) and reinstalled, so it will not be wiped again.\n"
        "Or switch to stream_bridge.type: memory in config.yaml for single-process deployment."
    ) from None

from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent, StreamGap, StreamItem

logger = logging.getLogger(__name__)

_KIND_EVENT = "event"
_KIND_END = "end"
_REDIS_STREAM_ID_RE = re.compile(r"\d+(-\d+)?")

# ``XREAD``의 batch 크기. round-trip마다 여러 entry를 읽으면 큰 ``Last-Event-ID`` replay가
# 훨씬 적은 호출로 줄어든다. consume loop가 end marker에서 batch 중간에 반환하므로 live tailing은
# 여전히 이벤트가 도착하는 즉시 하나씩 내보낸다.
_XREAD_COUNT = 64

# ``subscribe`` 중 에러가 호출자에게 전파되기 전까지 허용하는 연속 일시적 Redis 에러
# (``ConnectionError``, ``TimeoutError`` 등)의 최대 횟수. 짧은 장애는 ``heartbeat_interval``로
# 상한이 걸린 지수 backoff로 retry한다.
_MAX_SUBSCRIBE_RETRIES = 3


class RedisStreamBridge(StreamBridge):
    """Redis Streams로 구현한 run 단위 stream bridge.

    각 run은 하나의 Redis Stream에 저장되고 subscriber는 ``XREAD``로 직접 읽는다. 덕분에 SSE
    bridge를 여러 gateway worker 프로세스에서 사용할 수 있으면서도 ``Last-Event-ID`` replay
    의미가 유지된다.
    """

    supports_cross_process = True

    def __init__(
        self,
        *,
        redis_url: str,
        queue_maxsize: int = 256,
        key_prefix: str = "deerflow:stream_bridge",
        max_connections: int | None = None,
        stream_ttl_seconds: int | None = 86400,
        client: Redis | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._maxsize = max(1, queue_maxsize)
        self._key_prefix = key_prefix.rstrip(":")
        if stream_ttl_seconds is not None and stream_ttl_seconds > 0:
            self._stream_ttl_seconds = stream_ttl_seconds
        else:
            self._stream_ttl_seconds = None
        # 활성 SSE subscriber는 각각 pool 연결 하나를 ``XREAD ... BLOCK``으로 최대
        # ``heartbeat_interval``까지 붙잡는다. ``max_connections``가 그 pool의 상한이며,
        # ``None``이면 redis-py의 사실상 무제한 기본값을 쓴다.
        self._redis = client if client is not None else Redis.from_url(redis_url, decode_responses=True, max_connections=max_connections)
        self._owns_client = client is None

    def _stream_key(self, run_id: str) -> str:
        return f"{self._key_prefix}:{run_id}"

    async def _xadd_retained(self, key: str, fields: dict[str, str], *, maxlen: int) -> None:
        if self._stream_ttl_seconds is None:
            await self._redis.xadd(
                key,
                fields,
                maxlen=maxlen,
                approximate=False,
            )
            return

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.xadd(
                key,
                fields,
                maxlen=maxlen,
                approximate=False,
            )
            pipe.expire(key, self._stream_ttl_seconds)
            await pipe.execute()

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _normalise_fields(cls, fields: Mapping[Any, Any]) -> dict[str, str]:
        return {cls._decode(key): cls._decode(value) for key, value in fields.items()}

    @staticmethod
    def _encode_data(data: Any) -> str:
        return json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_data(raw: str | None) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Redis stream bridge received non-JSON event data")
            return raw

    def _entry_from_redis(self, event_id: str, fields: Mapping[Any, Any]) -> StreamEvent:
        payload = self._normalise_fields(fields)
        kind = payload.get("kind", _KIND_EVENT)
        if kind == _KIND_END:
            return END_SENTINEL
        return StreamEvent(
            id=event_id,
            event=payload.get("event", "message"),
            data=self._decode_data(payload.get("data")),
        )

    @classmethod
    def _is_end_entry(cls, fields: Mapping[Any, Any]) -> bool:
        return cls._normalise_fields(fields).get("kind") == _KIND_END

    @staticmethod
    def _parse_stream_id(event_id: str) -> tuple[int, int] | None:
        if _REDIS_STREAM_ID_RE.fullmatch(event_id) is None:
            return None
        milliseconds, separator, sequence = event_id.partition("-")
        return int(milliseconds), int(sequence) if separator else 0

    @classmethod
    def _stream_id_lt(cls, left: str, right: str) -> bool:
        left_parts = cls._parse_stream_id(left)
        right_parts = cls._parse_stream_id(right)
        return left_parts is not None and right_parts is not None and left_parts < right_parts

    @classmethod
    def _response_tail_id(cls, response: list[Any]) -> str | None:
        for _stream_name, entries in reversed(response):
            if entries:
                return cls._decode(entries[-1][0])
        return None

    async def publish(self, run_id: str, event: str, data: Any) -> None:
        key = self._stream_key(run_id)
        await self._xadd_retained(
            key,
            {
                "kind": _KIND_EVENT,
                "event": event,
                "data": self._encode_data(data),
            },
            maxlen=self._maxsize,
        )

    async def publish_end(self, run_id: str) -> None:
        # 설정된 개수의 data 이벤트에 내부 end marker 하나를 더해 보관한다.
        key = self._stream_key(run_id)
        await self._xadd_retained(
            key,
            {"kind": _KIND_END},
            maxlen=self._maxsize + 1,
        )

    async def stream_exists(self, run_id: str) -> bool:
        """Redis에 *run_id*의 stream 데이터가 아직 남아 있는지 반환한다."""
        return bool(await self._redis.exists(self._stream_key(run_id)))

    async def _resolve_start_stream_id(self, key: str, last_event_id: str | None) -> str:
        if last_event_id is None:
            return "0-0"
        if _REDIS_STREAM_ID_RE.fullmatch(last_event_id):
            return last_event_id
        entries = await self._redis.xrevrange(key, count=1)
        if not entries:
            return "0-0"
        event_id, fields = entries[0]
        payload = self._normalise_fields(fields)
        if payload.get("kind") == _KIND_END:
            return "0-0"
        return self._decode(event_id)

    async def _read_retained_snapshot(
        self,
        key: str,
        stream_id: str,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """보관 경계와 ``stream_id`` 이후의 entry를 원자적으로 읽는다.

        blocking ``XREAD``는 Redis 트랜잭션에 참여할 수 없다. 그래서 live subscriber는 정확성을
        위해 이 non-blocking 원자적 snapshot을 쓰고, 별도의 blocking read는 wake-up 신호로만
        쓴다. 보관 경계 확인이 read와 경합하지 않도록 poll마다 3-command pipeline(및 idle 중
        두 번째 round trip)을 의도적으로 추가한다.
        """
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.xrange(key, count=1)
            pipe.xrevrange(key, count=1)
            pipe.xread({key: stream_id}, count=_XREAD_COUNT)
            earliest, latest, response = await pipe.execute()
        return earliest, latest, response

    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamItem]:
        key = self._stream_key(run_id)
        stream_id = await self._resolve_start_stream_id(key, last_event_id)
        gap_detection_enabled = last_event_id is not None and self._parse_stream_id(last_event_id) is not None
        pending_initial_response: list[Any] | None = None
        block_ms = max(1, int(heartbeat_interval * 1000)) if heartbeat_interval > 0 else 1
        consecutive_errors = 0

        while True:
            snapshot_stream_id = stream_id
            pending_initial_tail_id = None
            if pending_initial_response is not None:
                pending_initial_tail_id = self._response_tail_id(pending_initial_response)
                if pending_initial_tail_id is None:
                    pending_initial_response = None
                else:
                    # 첫 blocking XREAD는 비어 있음이 확인된 stream을 대상으로 시작했다. 그
                    # 응답은 잠정적인 live baseline일 뿐 아직 클라이언트에 보이는 데이터가
                    # 아니다. 하나라도 내보내기 전에 보관 watermark와 대조해 검증한다.
                    snapshot_stream_id = pending_initial_tail_id

            try:
                earliest_entries, latest_entries, response = await self._read_retained_snapshot(key, snapshot_stream_id)
            except ResponseError:
                # Last-Event-ID는 클라이언트가 지정하며 XREAD 전에 검증된다. 그럼에도 Redis가
                # id를 거부하면 0-0으로 되돌리지 않고 실패시킨다. 되돌리면 재연결 때 보관
                # buffer 전체가 replay되기 때문이다.
                logger.warning(
                    "Redis rejected stream id %r for stream bridge subscription",
                    snapshot_stream_id,
                    exc_info=True,
                )
                raise
            except RedisError:
                consecutive_errors += 1
                if consecutive_errors > _MAX_SUBSCRIBE_RETRIES:
                    raise
                delay = min(2**consecutive_errors, heartbeat_interval)
                logger.warning(
                    "Transient Redis error in stream bridge subscriber (retry %d/%d); backing off %.1fs",
                    consecutive_errors,
                    _MAX_SUBSCRIBE_RETRIES,
                    delay,
                    exc_info=True,
                )
                await asyncio.sleep(delay)
                continue
            else:
                # 비어 있지 않은 snapshot은 진전이다. 빈 snapshot이면 blocking XREAD 자체가
                # 성공할 때까지 앞선 wake-up 실패 횟수를 유지한다. 그렇지 않으면 non-blocking
                # 트랜잭션이 시도 사이에 성공하기 때문에, 영구적으로 실패하는 blocking read가
                # 무한히 retry될 수 있다.
                if response:
                    consecutive_errors = 0

            if earliest_entries and (gap_detection_enabled or pending_initial_tail_id is not None):
                earliest_id = self._decode(earliest_entries[0][0])
                if self._stream_id_lt(snapshot_stream_id, earliest_id):
                    latest_id = self._decode(latest_entries[0][0])
                    logger.warning(
                        "subscriber for Redis stream %s fell behind retained history at %s",
                        key,
                        snapshot_stream_id,
                    )
                    yield StreamGap(
                        requested_event_id=None if pending_initial_tail_id is not None else stream_id,
                        earliest_available_event_id=earliest_id,
                        latest_available_event_id=latest_id,
                    )
                    return

            responses_to_process = []
            if pending_initial_response is not None:
                responses_to_process.append(pending_initial_response)
                pending_initial_response = None
            if response:
                responses_to_process.append(response)

            if not responses_to_process:
                if latest_entries and self._decode(latest_entries[0][0]) == stream_id and self._is_end_entry(latest_entries[0][1]):
                    yield END_SENTINEL
                    return

                try:
                    wake_response = await self._redis.xread(
                        {key: stream_id},
                        count=_XREAD_COUNT,
                        block=block_ms,
                    )
                except ResponseError:
                    logger.warning(
                        "Redis rejected stream id %r for stream bridge subscription",
                        stream_id,
                        exc_info=True,
                    )
                    raise
                except RedisError:
                    consecutive_errors += 1
                    if consecutive_errors > _MAX_SUBSCRIBE_RETRIES:
                        raise
                    delay = min(2**consecutive_errors, heartbeat_interval)
                    logger.warning(
                        "Transient Redis error in stream bridge subscriber (retry %d/%d); backing off %.1fs",
                        consecutive_errors,
                        _MAX_SUBSCRIBE_RETRIES,
                        delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    consecutive_errors = 0

                if not wake_response:
                    yield HEARTBEAT_SENTINEL
                elif last_event_id is None and not gap_detection_enabled and not earliest_entries:
                    # 비어 있음이 확인된 stream에서 온 첫 wake-up은 버리지 않는다. 다음 원자적
                    # 경계 확인까지 붙들고 있으면, cursor 없는 subscriber도 Memory와 동일한
                    # 뒤처짐 신호를 받으면서 잘못된 cursor의 live tailing 동작은 그대로 남는다.
                    pending_initial_response = wake_response
                continue

            for retained_response in responses_to_process:
                for _stream_name, entries in retained_response:
                    for event_id, fields in entries:
                        event_id = self._decode(event_id)
                        stream_id = event_id
                        gap_detection_enabled = True
                        entry = self._entry_from_redis(event_id, fields)
                        if entry is END_SENTINEL:
                            yield END_SENTINEL
                            return
                        yield entry

    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        await self._redis.delete(self._stream_key(run_id))

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._redis, "aclose", None) or getattr(self._redis, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
