"""async stream bridge 팩토리.

:func:`deerflow.runtime.checkpointer.async_provider.make_checkpointer`와 형태를 맞춘
**async context manager**를 제공한다.

사용법(예: FastAPI lifespan)::

    from deerflow.agents.stream_bridge import make_stream_bridge

    async with make_stream_bridge() as bridge:
        app.state.stream_bridge = bridge
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator

from deerflow.config.app_config import AppConfig
from deerflow.config.stream_bridge_config import StreamBridgeConfig, get_stream_bridge_config

from .base import StreamBridge

logger = logging.getLogger(__name__)

_ENV_REDIS_URL = "DEER_FLOW_STREAM_BRIDGE_REDIS_URL"


def _resolve_config(app_config: AppConfig | None) -> StreamBridgeConfig | None:
    if app_config is None:
        config = get_stream_bridge_config()
    else:
        config = app_config.stream_bridge

    if config is None:
        redis_url = os.getenv(_ENV_REDIS_URL)
        if redis_url:
            return StreamBridgeConfig(type="redis", redis_url=redis_url)
    return config


def _resolve_redis_url(config: StreamBridgeConfig) -> str:
    return config.redis_url or os.getenv(_ENV_REDIS_URL) or os.getenv("REDIS_URL") or "redis://localhost:6379/0"


@contextlib.asynccontextmanager
async def make_stream_bridge(app_config: AppConfig | None = None) -> AsyncIterator[StreamBridge]:
    """:class:`StreamBridge`를 yield하는 async context manager.

    설정이 주어지지 않고 전역에도 아무것도 설정되어 있지 않으면 :class:`MemoryStreamBridge`로
    대체한다.
    """
    config = _resolve_config(app_config)

    if config is None or config.type == "memory":
        from deerflow.runtime.stream_bridge.memory import MemoryStreamBridge

        maxsize = config.queue_maxsize if config is not None else 256
        bridge = MemoryStreamBridge(queue_maxsize=maxsize)
        logger.info("Stream bridge initialised: memory (queue_maxsize=%d)", maxsize)
        try:
            yield bridge
        finally:
            await bridge.close()
        return

    if config.type == "redis":
        from deerflow.runtime.stream_bridge.redis import RedisStreamBridge

        redis_url = _resolve_redis_url(config)
        bridge = RedisStreamBridge(
            redis_url=redis_url,
            queue_maxsize=config.queue_maxsize,
            max_connections=config.max_connections,
            stream_ttl_seconds=config.stream_ttl_seconds,
        )
        logger.info(
            "Stream bridge initialised: redis (queue_maxsize=%d, max_connections=%s, stream_ttl_seconds=%d)",
            config.queue_maxsize,
            config.max_connections,
            config.stream_ttl_seconds,
        )
        try:
            yield bridge
        finally:
            await bridge.close()
        return

    raise ValueError(f"Unknown stream bridge type: {config.type!r}")
