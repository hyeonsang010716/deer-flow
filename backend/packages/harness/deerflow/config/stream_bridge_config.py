"""stream bridge 설정."""

from typing import Literal

from pydantic import BaseModel, Field

StreamBridgeType = Literal["memory", "redis"]


class StreamBridgeConfig(BaseModel):
    """agent worker와 SSE endpoint를 잇는 stream bridge 설정."""

    type: StreamBridgeType = Field(
        default="memory",
        description="Stream bridge backend type. 'memory' uses an in-process event log (single-process only). 'redis' uses Redis Streams for multi-worker Docker deployments.",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for the redis stream bridge type. If omitted, DEER_FLOW_STREAM_BRIDGE_REDIS_URL, REDIS_URL, or redis://localhost:6379/0 is used.",
    )
    queue_maxsize: int = Field(
        default=256,
        description="Maximum number of events retained per run (memory bridge queue size / redis stream MAXLEN).",
    )
    max_connections: int | None = Field(
        default=None,
        description=(
            "Max Redis connections in the pool for the redis stream bridge. Each live SSE "
            "client holds one connection blocked in XREAD ... BLOCK for up to heartbeat_interval "
            "(15s), so hundreds of concurrent clients open hundreds of connections. Leave unset "
            "for redis-py's default (effectively unbounded), or set a ceiling sized for peak "
            "concurrent SSE clients. Only applies to the redis bridge."
        ),
    )
    stream_ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description=(
            "Rolling Redis stream key TTL in seconds. The redis bridge refreshes this TTL after "
            "each publish and publish_end so retained SSE replay buffers are eventually reclaimed "
            "even if cleanup never runs. Set to 0 to disable. Only applies to the redis bridge."
        ),
    )
    recovered_stream_cleanup_delay_seconds: float = Field(
        default=60.0,
        ge=0,
        description=("Seconds to wait after publishing an END marker for a recovered orphaned run before deleting the stream key. Gives reconnecting SSE clients time to drain the end signal. Only applies to the redis bridge."),
    )


# 전역 설정 인스턴스. None이면 stream bridge가 설정되지 않았다는 뜻이고,
# 기본값의 memory bridge로 fallback한다.
_stream_bridge_config: StreamBridgeConfig | None = None


def get_stream_bridge_config() -> StreamBridgeConfig | None:
    """현재 stream bridge 설정을 반환한다. 설정되지 않았으면 None이다."""
    return _stream_bridge_config


def set_stream_bridge_config(config: StreamBridgeConfig | None) -> None:
    """stream bridge 설정을 지정한다."""
    global _stream_bridge_config
    _stream_bridge_config = config


def load_stream_bridge_config_from_dict(config_dict: dict | None) -> None:
    """dict에서 stream bridge 설정을 읽는다."""
    global _stream_bridge_config
    if config_dict is None:
        _stream_bridge_config = None
        return
    _stream_bridge_config = StreamBridgeConfig(**config_dict)
