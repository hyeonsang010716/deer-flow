"""설정된 sandbox ownership store를 해석한다.

``stream_bridge``의 ``make_stream_bridge``와 같은 구조다. ``config.type``으로 분기하고,
memory 전용 설치가 ``redis``를 import하지 않도록 분기마다 lazy import하며, 컨테이너
배포에서 config.yaml을 고치지 않고 backend를 바꿀 수 있게 env var 우회로를 둔다.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid

from deerflow.config.sandbox_config import SandboxOwnershipConfig
from deerflow.config.stream_bridge_config import StreamBridgeConfig

from .base import SandboxOwnershipStore

logger = logging.getLogger(__name__)

_ENV_OWNERSHIP_REDIS_URL = "DEER_FLOW_SANDBOX_OWNERSHIP_REDIS_URL"
_ENV_STREAM_BRIDGE_REDIS_URL = "DEER_FLOW_STREAM_BRIDGE_REDIS_URL"


def generate_owner_id() -> str:
    """이 provider instance의 고유 id를 ``hostname:hex`` 형태로 반환한다.

    host 단위가 아니라 instance 단위다. 한 host의 gateway worker 두 개가 서로의 lease를
    구분할 수 있어야 한다.
    """
    return f"{socket.gethostname()}:{uuid.uuid4().hex}"


def resolve_ownership_config(config: SandboxOwnershipConfig | None, *, stream_bridge: StreamBridgeConfig | None = None) -> SandboxOwnershipConfig:
    """생략된 ownership 섹션을 채운다.

    stream bridge를 이미 Redis로 향하게 한 배포는 정의상 multi-instance이므로, 조용히
    memory로 떨어지지 않고 redis ownership store를 받는다. memory는 peer를 볼 수 없어
    #4206을 그대로 열어 둔다.

    stream bridge 자신의 redis 트리거 두 가지를 같은 순서로 따른다
    (``stream_bridge/async_provider.py::_resolve_config``). config.yaml 섹션이 먼저,
    그다음이 env var다. env var만 읽으면 bridge를 Redis로 보내는 config.yaml 방식을
    놓치는데, 그게 바로 이 추론이 필요한 multi-instance 배포다.
    """
    if config is not None:
        return config

    if stream_bridge is not None and stream_bridge.type == "redis":
        redis_url = stream_bridge.redis_url or os.getenv(_ENV_OWNERSHIP_REDIS_URL) or os.getenv(_ENV_STREAM_BRIDGE_REDIS_URL)
        logger.info("Sandbox ownership: redis inferred from stream_bridge.type (multi-instance deployment)")
        return SandboxOwnershipConfig(type="redis", redis_url=redis_url)

    redis_url = os.getenv(_ENV_OWNERSHIP_REDIS_URL) or os.getenv(_ENV_STREAM_BRIDGE_REDIS_URL)
    if redis_url:
        logger.info("Sandbox ownership: redis inferred from environment (multi-instance deployment)")
        return SandboxOwnershipConfig(type="redis", redis_url=redis_url)
    return SandboxOwnershipConfig()


def resolve_ownership_redis_url(
    config: SandboxOwnershipConfig,
) -> str:
    """ownership 계열 store들이 공유하는 Redis endpoint를 해석한다."""
    return config.redis_url or os.getenv(_ENV_OWNERSHIP_REDIS_URL) or os.getenv(_ENV_STREAM_BRIDGE_REDIS_URL) or os.getenv("REDIS_URL") or "redis://localhost:6379/0"


def compute_lease_ttl(config: SandboxOwnershipConfig) -> float:
    """lease TTL(초).

    renewal interval에서 유도하며 ``sandbox.idle_timeout``에서는 절대 유도하지 않는다.
    liveness를 idle reaper에 묶는 바람에 idle checker가 아예 시작되지 않는
    ``idle_timeout: 0``에서 ownership이 lapse됐었다.
    """
    return config.renewal_interval_seconds * config.ttl_multiplier


def make_sandbox_ownership_store(config: SandboxOwnershipConfig | None, *, owner_id: str | None = None) -> SandboxOwnershipStore:
    """*config*에 맞는 ownership store를 만든다.

    반환된 store의 소유권은 호출자에게 있으며 ``close()``를 호출해야 한다.
    """
    # 이미 해석된 config는 신뢰하고, 생략된 섹션만 채운다. provider가 (이 factory가 할 수
    # 없는 stream_bridge 추론까지 포함해) 한 번 해석해 넘기므로 여기서 다시 해석해 봐야
    # no-op이다.
    resolved = config if config is not None else resolve_ownership_config(None)
    effective_owner_id = owner_id or generate_owner_id()
    ttl = compute_lease_ttl(resolved)

    if resolved.type == "memory":
        from .memory import MemoryOwnershipStore

        logger.info("Sandbox ownership store: memory (single-instance; ttl=%.1fs)", ttl)
        return MemoryOwnershipStore(owner_id=effective_owner_id, ttl_seconds=ttl)

    if resolved.type == "redis":
        from .redis import RedisOwnershipStore

        redis_url = resolve_ownership_redis_url(resolved)
        logger.info("Sandbox ownership store: redis (ttl=%.1fs, renewal=%.1fs)", ttl, resolved.renewal_interval_seconds)
        return RedisOwnershipStore(
            owner_id=effective_owner_id,
            redis_url=redis_url,
            ttl_seconds=ttl,
            key_prefix=resolved.key_prefix,
        )

    raise ValueError(f"Unknown sandbox ownership type: {resolved.type!r}")
