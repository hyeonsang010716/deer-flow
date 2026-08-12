"""cache factory. make_stream_bridge와 같은 방식이다: config -> env fallback -> memory."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from deerflow.config.app_config import AppConfig
from deerflow.runtime.checkpoint_cache.base import CACHE_FORMAT_VERSION, CheckpointHistoryCache
from deerflow.runtime.checkpoint_cache.memory import MemoryCheckpointHistoryCache

logger = logging.getLogger(__name__)

_ENV_REDIS_URL = "DEER_FLOW_CHECKPOINT_CACHE_REDIS_URL"


def _resolve_redis_url(config: Any) -> str:
    return config.redis_url or os.getenv(_ENV_REDIS_URL) or os.getenv("REDIS_URL") or "redis://localhost:6379/0"


def _stable_postgres_identity(postgres_url: str) -> str:
    """자격 증명을 뺀 데이터베이스 identity: host/port/database.

    raw URL을 해싱하면 자격 증명을 교체할 때마다 cache namespace가 바뀐다(cold cache에 더해
    TTL까지 고아 키가 남는다). 데이터베이스는, 따라서 캐시된 모든 checkpoint history도 그대로인데
    말이다. 파싱할 수 없는 URL은 raw 문자열로 대체한다(배포 단위로는 여전히 안정적이다).
    """
    if not postgres_url:
        return ""
    try:
        from sqlalchemy.engine.url import make_url

        parsed = make_url(postgres_url)
    except Exception:  # noqa: BLE001 - identity 계산 때문에 config 로드가 실패해서는 안 된다
        return postgres_url
    return f"{parsed.host or 'localhost'}:{parsed.port or 5432}/{parsed.database or ''}"


def checkpoint_cache_db_hash(db_config: Any) -> str:
    """배포 identity 해시. 하나의 Redis를 공유하는 두 배포가 충돌하지 않게 한다."""
    backend = getattr(db_config, "backend", "memory")
    if backend == "postgres":
        identity = f"postgres:{_stable_postgres_identity(getattr(db_config, 'postgres_url', ''))}:{getattr(db_config, 'postgres_schema', '')}"
    elif backend == "sqlite":
        identity = f"sqlite:{getattr(db_config, 'checkpointer_sqlite_path', '')}"
    else:
        identity = "memory"
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def checkpoint_cache_key_prefix(app_config: AppConfig) -> str:
    cache_config = app_config.database.checkpoint_cache
    if cache_config.key_prefix:
        return cache_config.key_prefix
    return f"ckpt-hist:v{CACHE_FORMAT_VERSION}:{checkpoint_cache_db_hash(app_config.database)}"


@contextlib.asynccontextmanager
async def make_checkpoint_cache(
    app_config: AppConfig | None = None,
    *,
    serde: Any,
) -> AsyncIterator[CheckpointHistoryCache]:
    """호출자의 수명 동안 쓸 history cache를 yield한다.

    ``max_entries == 0``이면 비활성화된 memory backend를 통해 두 타입 모두 cache가 균일하게
    꺼지므로, wrapper에서 None 검사를 할 필요가 없다.
    """
    config = app_config.database.checkpoint_cache if app_config is not None else None

    if config is None or config.type == "memory" or config.max_entries == 0:
        max_entries = config.max_entries if config is not None else 128
        cache = MemoryCheckpointHistoryCache(max_entries=max_entries)
        logger.info("Checkpoint history cache initialised: memory (max_entries=%d)", max_entries)
        try:
            yield cache
        finally:
            await cache.aclose()
        return

    if config.type == "redis":
        from deerflow.runtime.checkpoint_cache.redis import RedisCheckpointHistoryCache

        cache = RedisCheckpointHistoryCache(
            _resolve_redis_url(config),
            serde=serde,
            ttl_seconds=config.ttl_seconds,
        )
        logger.info("Checkpoint history cache initialised: redis (ttl_seconds=%d)", config.ttl_seconds)
        try:
            yield cache
        finally:
            await cache.aclose()
        return

    raise ValueError(f"Unknown checkpoint cache type: {config.type!r}")
