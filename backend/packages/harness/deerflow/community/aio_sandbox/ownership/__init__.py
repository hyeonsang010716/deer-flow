"""공유 sandbox 컨테이너를 위한 cross-instance ownership lease (#4206)."""

# NOTE: ``RedisOwnershipStore``는 의도적으로 여기서 import하지 않는다. ``redis``는
# optional extra이고, 이 패키지는 provider 생성 시점에 ``aio_sandbox_provider``가
# import한다. ``.redis``를 즉시 import하면 ownership이 memory 전용인 설치에서도
# 모든 AIO sandbox가 redis 패키지에 묶인다. 대신 ``sandbox.ownership.type == "redis"``일
# 때만 ``make_sandbox_ownership_store`` 안에서 lazy import한다.

from .base import OwnershipBackendError, RenewOutcome, SandboxOwnershipStore
from .factory import compute_lease_ttl, generate_owner_id, make_sandbox_ownership_store, resolve_ownership_config
from .memory import MemoryOwnershipStore

__all__ = [
    "MemoryOwnershipStore",
    "OwnershipBackendError",
    "RenewOutcome",
    "SandboxOwnershipStore",
    "compute_lease_ttl",
    "generate_owner_id",
    "make_sandbox_ownership_store",
    "resolve_ownership_config",
]
