"""공유 Redis backend. 항목이 불변이므로 multi-worker 공유 cache에도 invalidation이 필요 없고,
TTL은 누수 방지용 안전망일 뿐이다.

thread 범위 정리(``adelete_thread``)는 데이터 수명주기를 위해 존재한다. thread의 checkpoint가
삭제되면 캐시된 history payload도 TTL 만료를 기다리지 않고 즉시 제거한다.

redis import는 지연 방식이다(선택 의존성인 ``redis`` extra 없이도 module을 import할 수 있다).
runtime/stream_bridge/redis.py와 같은 방식이다.
"""

from __future__ import annotations

import logging
from typing import Any

from deerflow.runtime.checkpoint_cache.base import CheckpointCacheStats, thread_key_stem

logger = logging.getLogger(__name__)

REDIS_INSTALL = "redis is required for the redis checkpoint cache backend. Install it with: uv sync --extra redis"

_TAG_SEPARATOR = b"\x00"


def _create_client(redis_url: str, *, max_connections: int | None) -> Any:
    try:
        import redis.asyncio as redis_async
    except ImportError as exc:
        raise ImportError(REDIS_INSTALL) from exc
    kwargs: dict[str, Any] = {"decode_responses": False}
    if max_connections is not None:
        kwargs["max_connections"] = max_connections
    return redis_async.from_url(redis_url, **kwargs)


def _redis_error() -> type[Exception]:
    """RedisError를 지연 import한다. 위의 지연 client 생성과 같은 방식이다."""
    try:
        from redis.exceptions import RedisError
    except ImportError as exc:
        raise ImportError(REDIS_INSTALL) from exc
    return RedisError


class RedisCheckpointHistoryCache:
    def __init__(
        self,
        redis_url: str,
        *,
        serde: Any,
        ttl_seconds: int,
        max_connections: int | None = None,
    ) -> None:
        self._client = _create_client(redis_url, max_connections=max_connections)
        self._serde = serde
        # ttl_seconds=0은 만료를 명시적으로 끄는 설정이다(SETEX를 쓰지 않는다). 기본값이 아니며,
        # 이 경우 누수되거나 고아가 된 키는 redis maxmemory에만 의존하게 된다.
        self._ttl = ttl_seconds if ttl_seconds > 0 else None
        self._hits = 0
        self._misses = 0

    async def aget_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        if not keys:
            return {}
        try:
            raws = await self._client.mget(keys)
        except _redis_error() as exc:
            # 성능에만 영향을 주는 우회다. redis 장애는 hit를 잃게 할 뿐 가용성을 해치지 않는다.
            logger.warning("checkpoint history cache mget failed; treating as all-miss: %s", exc)
            self._misses += len(keys)
            return {}
        found: dict[str, dict[str, Any]] = {}
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                self._misses += 1
                continue
            self._hits += 1
            tag, payload = raw.split(_TAG_SEPARATOR, 1)
            found[key] = self._serde.loads_typed((tag.decode(), payload))
        return found

    async def aset_many(self, entries: dict[str, dict[str, Any]]) -> None:
        if not entries:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for key, entry in entries.items():
                tag, data = self._serde.dumps_typed(entry)
                pipe.set(key, tag.encode() + _TAG_SEPARATOR + data, ex=self._ttl)
            await pipe.execute()
        except _redis_error() as exc:
            # 쓰기는 선택 사항이다. 실패해도 다음 읽기에서 history를 다시 계산할 뿐이다.
            logger.warning("checkpoint history cache write failed; skipping: %s", exc)

    async def adelete_thread(self, key_prefix: str, thread_id: str) -> None:
        """한 thread의 모든 항목을 SCAN+UNLINK한다. 실패하면 TTL로 제한된 잔여 보관으로
        낮아진다. 원본 삭제는 이미 끝났으므로 이 함수는 절대 예외를 던지지 않는다."""
        stem = thread_key_stem(key_prefix, thread_id)
        try:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor=cursor, match=stem + "*", count=500)
                if keys:
                    await self._client.unlink(*keys)
                if cursor == 0:
                    break
        except _redis_error() as exc:
            logger.warning("checkpoint history cache thread purge failed; residual entries expire via TTL: %s", exc)

    def stats(self) -> CheckpointCacheStats:
        return CheckpointCacheStats(hits=self._hits, misses=self._misses)

    async def aclose(self) -> None:
        await self._client.aclose()
