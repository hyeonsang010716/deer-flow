"""multi-instance gateway용 Redis 기반 ownership store (#4206).

ownership은 sandbox마다 키 하나이며, 값이 소유자와 lease 상태를 함께 인코딩한다.
``own:<owner_id>``(이 컨테이너를 책임짐) 또는 ``del:<owner_id>``(내리는 중)이고, TTL은
소유 instance가 갱신한다.

lock 없이도 destroy 구간을 안전하게 만드는 것이 이 상태 prefix다. ``del:`` lease에
대해서는 인수가 거절되므로, destroy 경로의 claim과 실제 컨테이너 stop 사이에 컨테이너가
재획득되지 않는다.

동기 클라이언트를 쓰는 것은 의도된 선택이다. 이 store는 provider 생성 시점과 백그라운드
스레드에서 구동되고 event loop에서는 절대 구동되지 않는다(계약은 ``base`` 참고).
여기서 ``redis.asyncio``는 잘못된 선택이다.

모든 변경은 Lua 스크립트를 거치므로 읽기와 쓰기 사이에 peer가 끼어들 수 없다. ``SET NX``만으로는
부족하다. 이미 우리가 소유한 키에서 실패하고, Python에서 GET 후 SET으로 대체하면 스크립트가
막아 주는 race가 다시 열린다.
"""

from __future__ import annotations

import logging

from .base import OwnershipBackendError, RenewOutcome, SandboxOwnershipStore

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - optional extra가 없을 때만 도달한다
    # ``redis``는 optional extra다(stream_bridge redis 경로와 동일). 이 모듈은
    # ``sandbox.ownership.type == "redis"``일 때만 ``make_sandbox_ownership_store``에서
    # lazy import되므로, 이 안내는 패키지 없이 redis ownership store를 요청한 경우에만
    # 정확히 노출된다.
    raise ImportError(
        "sandbox.ownership.type is set to 'redis' but the redis package is not installed.\n"
        "Install it with:\n"
        "    cd backend && uv sync --all-packages --extra redis\n"
        "On the next `make dev` the redis extra is auto-detected from config.yaml\n"
        "(sandbox.ownership.type: redis) and reinstalled, so it will not be wiped again.\n"
        "Or switch to sandbox.ownership.type: memory in config.yaml for single-instance deployment."
    ) from None

logger = logging.getLogger(__name__)

_OWN = "own:"
_DEL = "del:"

# 멈춘 Redis가 호출자를 붙잡지 못하도록 모든 store round-trip에 상한을 둔다. teardown
# heartbeat에서 가장 중요하다. heartbeat의 종료와 그 종료가 수행하는 마지막 lease release는
# 유한해야 하며, 그렇지 않으면 블랙홀 연결에 막힌 refresh가 destroy 경로(및 지연된 release)를
# 무한정 붙잡는다. socket timeout이 없으면 redis-py는 영원히 블로킹한다.
_STORE_SOCKET_TIMEOUT_SECONDS = 5.0

# acquire 경로의 인수. thread의 turn이 여기로 라우팅됐으므로 살아 있는 peer의 일반 lease는
# 일부러 덮어쓴다. 다만 진행 중인 teardown은 거절하는데, 그래야 peer가 곧 멈출 컨테이너를
# 넘겨주지 않는다.
_TAKE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current ~= false and string.sub(current, 1, 4) == 'del:' then
    return 0
end
redis.call('SET', KEYS[1], 'own:' .. ARGV[1], 'PX', ARGV[2])
return 1
"""

# adopt/reap 관문: 주인이 없거나 (어느 상태로든) 이미 우리 것일 때만 통과한다.
# ARGV[3]이 기록할 상태를 고른다. '1'은 진행 중인 teardown을 표시한다.
#
# 비-destroy claim은 *우리 자신의* teardown을 되감지 않는다. stop이 이미 진행 중이라
# 취소할 수 없으므로, 마커를 `own:`으로 낮추면 `take()`가 곧 죽을 컨테이너를 넘겨주게 된다.
# 지금은 그렇게 호출하는 곳이 없지만(`for_destroy=false` 호출자는 키가 없거나 주인 없는
# 상태에서 돈다), 그 전제가 계속 참이길 기대하는 대신 계약으로 금지한다.
_CLAIM_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local mine_own = 'own:' .. ARGV[1]
local mine_del = 'del:' .. ARGV[1]
if ARGV[3] == '0' and current == mine_del then
    return 0
end
if current == false or current == mine_own or current == mine_del then
    local value = mine_own
    if ARGV[3] == '1' then
        value = mine_del
    end
    redis.call('SET', KEYS[1], value, 'PX', ARGV[2])
    return 1
end
return 0
"""

# 호출자가 없는 lease(재확립해도 안전)와 peer의 lease(다시 가져오면 #4206 kill)를 구분할 수
# 있도록 3분기다. 둘을 합쳤던 탓에 Redis 재시작이 fleet 전체의 살아 있는 sandbox를 날렸다.
#    1 = renewed, -1 = lapsed/absent, 0 = peer가 보유 중이거나 teardown 중
_RENEW_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current == false then
    return -1
end
if current == 'own:' .. ARGV[1] then
    redis.call('PEXPIRE', KEYS[1], ARGV[2])
    return 1
end
return 0
"""

# 어느 상태든 우리 lease만 지운다. peer의 lease는 절대 지우지 않는다.
_RELEASE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current == 'own:' .. ARGV[1] or current == 'del:' .. ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisOwnershipStore(SandboxOwnershipStore):
    """Redis를 통해 gateway instance 간에 공유되는 ownership lease."""

    supports_cross_process = True

    def __init__(
        self,
        *,
        owner_id: str,
        redis_url: str,
        ttl_seconds: float,
        key_prefix: str = "deerflow:sandbox:owner",
        client: Redis | None = None,
    ) -> None:
        self._owner_id = owner_id
        self._ttl_ms = max(1, int(float(ttl_seconds) * 1000))
        self._key_prefix = key_prefix.rstrip(":")
        # Redis.from_url은 lazy라서 Redis에 닿지 않아도 provider 생성이 막히지 않고,
        # 대신 첫 claim에서 예외가 난다. socket_timeout이 모든 round-trip에 상한을 두므로
        # (_STORE_SOCKET_TIMEOUT_SECONDS 참고) 어떤 store 호출도, 특히 teardown heartbeat의
        # refresh도 무한정 블로킹하지 않는다.
        self._redis = (
            client
            if client is not None
            else Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=_STORE_SOCKET_TIMEOUT_SECONDS,
                socket_connect_timeout=_STORE_SOCKET_TIMEOUT_SECONDS,
            )
        )
        self._owns_client = client is None
        self._take = self._redis.register_script(_TAKE_SCRIPT)
        self._claim = self._redis.register_script(_CLAIM_SCRIPT)
        self._renew = self._redis.register_script(_RENEW_SCRIPT)
        self._release = self._redis.register_script(_RELEASE_SCRIPT)

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def _key(self, sandbox_id: str) -> str:
        return f"{self._key_prefix}:{sandbox_id}"

    def take(self, sandbox_id: str) -> bool:
        try:
            result = self._take(keys=[self._key(sandbox_id)], args=[self._owner_id, self._ttl_ms])
        except RedisError as e:
            raise OwnershipBackendError(f"failed to publish sandbox ownership for {sandbox_id}: {e}") from e
        return bool(result)

    def claim(self, sandbox_id: str, *, for_destroy: bool = False) -> bool:
        try:
            result = self._claim(keys=[self._key(sandbox_id)], args=[self._owner_id, self._ttl_ms, "1" if for_destroy else "0"])
        except RedisError as e:
            raise OwnershipBackendError(f"failed to claim sandbox ownership for {sandbox_id}: {e}") from e
        return bool(result)

    def renew(self, sandbox_id: str) -> RenewOutcome:
        try:
            result = int(self._renew(keys=[self._key(sandbox_id)], args=[self._owner_id, self._ttl_ms]))
        except RedisError as e:
            raise OwnershipBackendError(f"failed to renew sandbox ownership for {sandbox_id}: {e}") from e
        if result == 1:
            return RenewOutcome.RENEWED
        if result == -1:
            return RenewOutcome.LAPSED
        return RenewOutcome.LOST

    def release(self, sandbox_id: str) -> None:
        try:
            self._release(keys=[self._key(sandbox_id)], args=[self._owner_id])
        except RedisError as e:
            raise OwnershipBackendError(f"failed to release sandbox ownership for {sandbox_id}: {e}") from e

    def owner(self, sandbox_id: str) -> str | None:
        try:
            value = self._redis.get(self._key(sandbox_id))
        except RedisError as e:
            raise OwnershipBackendError(f"failed to read sandbox ownership for {sandbox_id}: {e}") from e
        if value is None:
            return None
        # 주입된 클라이언트는 decode_responses를 설정하지 않았을 수 있다.
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        if text.startswith(_OWN) or text.startswith(_DEL):
            return text[4:]
        return text

    def close(self) -> None:
        if not self._owns_client:
            return
        try:
            self._redis.close()
        except Exception as e:  # pragma: no cover - teardown은 best effort다
            logger.warning("Error closing sandbox ownership redis client: %s", e)
