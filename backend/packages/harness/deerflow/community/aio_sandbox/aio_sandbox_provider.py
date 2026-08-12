"""AIO Sandbox Provider — 교체 가능한 backend와 함께 sandbox lifecycle을 관리한다.

이 provider가 조합하는 것:
- SandboxBackend: sandbox를 어떻게 provisioning하는지(로컬 컨테이너 vs 원격/K8s)

provider 자체가 처리하는 것:
- 반복 접근을 빠르게 하기 위한 in-process 캐싱
- idle timeout 관리
- signal 처리를 통한 graceful shutdown
- mount 계산(thread 전용, skills)
"""

import asyncio
import atexit
import contextlib
import hashlib
import logging
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]
    import msvcrt

from deerflow.community.warm_pool_lifecycle import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_REPLICAS,
    WarmPoolLifecycleMixin,
)
from deerflow.community.warm_pool_lifecycle import (
    IDLE_CHECK_INTERVAL as _SHARED_IDLE_CHECK_INTERVAL,
)
from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths, join_host_path
from deerflow.integrations.lark_cli import INTEGRATION_ID as LARK_CLI_INTEGRATION_ID
from deerflow.integrations.lark_cli import LARK_CLI_SANDBOX_CONFIG_DIR, LARK_CLI_SANDBOX_DATA_DIR, LARK_CLI_SANDBOX_LOCKS_DIR, LARK_CLI_SANDBOX_RUNTIME_DIR, ensure_lark_cli_credential_tree, lark_skills_installed
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider

from .aio_sandbox import AioSandbox
from .backend import SandboxBackend, wait_for_sandbox_ready, wait_for_sandbox_ready_async
from .local_backend import LocalContainerBackend
from .ownership import (
    OwnershipBackendError,
    RenewOutcome,
    SandboxOwnershipStore,
    compute_lease_ttl,
    generate_owner_id,
    make_sandbox_ownership_store,
    resolve_ownership_config,
)
from .remote_backend import RemoteSandboxBackend
from .sandbox_info import SandboxInfo

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_IMAGE = "enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest"
DEFAULT_PORT = 8080
DEFAULT_CONTAINER_PREFIX = "deer-flow-sandbox"
IDLE_CHECK_INTERVAL = _SHARED_IDLE_CHECK_INTERVAL
THREAD_LOCK_EXECUTOR_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_THREAD_LOCK_EXECUTOR = ThreadPoolExecutor(max_workers=THREAD_LOCK_EXECUTOR_WORKERS, thread_name_prefix="sandbox-lock-wait")
atexit.register(_THREAD_LOCK_EXECUTOR.shutdown, wait=False, cancel_futures=True)


class SandboxBeingDestroyedError(RuntimeError):
    """peer가 이 컨테이너를 내리는 중이므로 넘겨주면 안 된다.

    ownership lease가 teardown 상태일 때 acquire 경로에서 발생한다. 호출자는 컨테이너를
    tracking에서 제거하고, 곧 밑에서 멈춰버릴 sandbox를 agent에게 넘기는 대신 일반적인
    discover-or-create 경로가 새 것을 provisioning하게 한다.
    """

    def __init__(self, sandbox_id: str) -> None:
        super().__init__(f"sandbox {sandbox_id} is being destroyed by another instance")
        self.sandbox_id = sandbox_id


class SandboxIdentityCollisionError(RuntimeError):
    """결정적으로 계산된 ID가 이미 다른 user/thread용으로 tracking되고 있다."""

    def __init__(
        self,
        sandbox_id: str,
        stored_key: tuple[str, str] | None,
        requested_key: tuple[str, str],
    ) -> None:
        super().__init__(f"sandbox ID collision for {sandbox_id}: tracked identity is {stored_key!r}, requested identity is {requested_key!r}")
        self.sandbox_id = sandbox_id


def _lock_file_exclusive(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return

    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)


def _unlock_file(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return

    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _open_lock_file(lock_path):
    return open(lock_path, "a", encoding="utf-8")


async def _acquire_thread_lock_async(lock: threading.Lock) -> None:
    """polling이나 default executor 없이 threading.Lock을 획득한다."""
    loop = asyncio.get_running_loop()
    acquire_future = loop.run_in_executor(_THREAD_LOCK_EXECUTOR, lock.acquire, True)

    try:
        acquired = await asyncio.shield(acquire_future)
    except asyncio.CancelledError:
        acquire_future.add_done_callback(lambda task: _release_cancelled_lock_acquire(lock, task))
        raise

    if not acquired:
        raise RuntimeError("Failed to acquire sandbox thread lock")


def _release_cancelled_lock_acquire(lock: threading.Lock, task: asyncio.Future[bool]) -> None:
    """대기하던 coroutine이 취소된 뒤에 획득된 lock을 해제한다."""
    if task.cancelled():
        return

    try:
        acquired = task.result()
    except Exception as e:
        logger.warning(f"Cancelled sandbox lock acquisition finished with error: {e}")
        return

    if acquired:
        lock.release()


class AioSandboxProvider(WarmPoolLifecycleMixin[SandboxInfo], SandboxProvider):
    """AIO sandbox를 실행하는 컨테이너를 관리하는 sandbox provider.

    구조:
        SandboxBackend(provisioning 방식)를 조합해서 다음을 지원한다.
        - 로컬 Docker/Apple Container 모드(컨테이너 자동 기동)
        - 원격/K8s 모드(이미 존재하는 sandbox URL에 연결)

    config.yaml의 sandbox 아래 설정 옵션:
        use: deerflow.community.aio_sandbox:AioSandboxProvider
        image: <컨테이너 이미지>
        port: 8080                      # 로컬 컨테이너의 base port
        container_prefix: deer-flow-sandbox
        idle_timeout: 600               # idle timeout(초). 0이면 비활성화
        replicas: 3                     # 동시 sandbox 컨테이너 최대 개수(초과 시 LRU eviction)
        thread_data_mounts: null        # null이면 backend 자동 감지
        mounts:                         # 로컬 컨테이너의 volume mount
          - host_path: /path/on/host
            container_path: /path/in/container
            read_only: false
        environment:                    # 컨테이너 환경 변수
          NODE_ENV: production
          API_KEY: $MY_API_KEY
    """

    # `_held_teardown_lease`가 마지막 lease 해제를 (아직 실행 중인) heartbeat thread에
    # 넘기기 전에 그 thread의 종료를 기다리는 시간. store의 socket timeout이 각 연산을
    # 제한하지만, context 종료 시점이 heartbeat의 마지막 refresh와 겹칠 수 있고 그 뒤의
    # 최종 release까지 기다려야 한다. 순차로 일어나는 5초짜리 연산 두 번보다 크게 잡아,
    # 정상적으로 timeout되는 refresh + release가 여전히 동기적으로 끝나게 한다.
    _TEARDOWN_JOIN_TIMEOUT_SECONDS = 12.0

    def __init__(self):
        self._lock = threading.Lock()
        self._sandboxes: dict[str, AioSandbox] = {}  # sandbox_id -> AioSandbox 인스턴스
        self._sandbox_infos: dict[str, SandboxInfo] = {}  # sandbox_id -> SandboxInfo (destroy용)
        self._thread_sandboxes: dict[tuple[str, str], str] = {}  # (user_id, thread_id) -> sandbox_id
        self._thread_locks: dict[tuple[str, str], threading.Lock] = {}  # (user_id, thread_id) -> 프로세스 내 lock
        self._last_activity: dict[str, float] = {}  # sandbox_id -> 마지막 활동 timestamp
        # warm pool: release되었지만 컨테이너는 아직 살아 있는 sandbox들.
        # sandbox_id -> (SandboxInfo, release timestamp) 매핑.
        # 여기 있는 컨테이너는 cold-start 없이 빠르게 회수하거나, replicas 용량이
        # 소진되면 파괴한다.
        self._warm_pool: dict[str, tuple[SandboxInfo, float]] = {}
        self._active_sandbox_identity: dict[str, tuple[str, str] | None] = {}
        self._warm_pool_identity: dict[str, tuple[str, str] | None] = {}
        # sandbox_id -> reconciliation이 lease 없이 실행 중인 상태로 처음 발견한 시각.
        # recovery grace를 지나야 adoption을 허용한다(_adoptable_after_grace 참고).
        self._unowned_since: dict[str, float] = {}
        # 같은 프로세스 내 배제(exclusion)의 두 축. ownership store는 peer만 배제할 뿐이다
        # — `claim()`과 `take()`는 설계상 우리 자신의 lease에 대해서는 모두 성공한다 —
        # 따라서 `del:`은 이 프로세스의 다른 thread에게는 아무 의미가 없다.
        # _reserve_local_teardown / _acquire_epoch 참고.
        self._local_teardown: set[str] = set()
        self._acquire_epoch: dict[str, int] = {}
        self._acquire_epoch_counter = 0
        self._acquire_inflight: dict[str, int] = {}
        self._shutdown_called = False
        self._idle_checker_stop = threading.Event()
        self._idle_checker_thread: threading.Thread | None = None
        self._renewal_stop = threading.Event()
        self._renewal_thread: threading.Thread | None = None
        # 인스턴스 간 sandbox ownership lease에 쓰는 인스턴스별 id(#4206).
        self._owner_id = generate_owner_id()

        self._config = self._load_config()
        self._ownership_config = resolve_ownership_config(self._config.get("ownership"), stream_bridge=self._config.get("stream_bridge"))
        self._ownership: SandboxOwnershipStore = make_sandbox_ownership_store(self._ownership_config, owner_id=self._owner_id)
        if not self._ownership.supports_cross_process:
            # peer는 이 lease를 볼 수 없으므로 모든 컨테이너가 orphan처럼 보인다. 설정을
            # 하지 않은 multi-worker 배포에서 #4206이 조용히 재발하게 두지 말고 한 번
            # 경고한다.
            logger.warning(
                "Sandbox ownership store cannot coordinate across processes (sandbox.ownership.type: %s). "
                "Safe for a single gateway instance only — multi-worker / load-balanced gateways sharing a "
                "container backend must set sandbox.ownership.type: redis, or peers will adopt and idle-destroy "
                "each other's live sandboxes (#4206).",
                self._ownership_config.type,
            )
        self._backend: SandboxBackend = self._create_backend()

        # shutdown handler 등록
        atexit.register(self.shutdown)
        self._register_signal_handlers()

        # 이전 프로세스 수명 주기에서 남은 orphan 컨테이너를 정리한다
        self._reconcile_orphans()

        # renewal은 idle cleanup과 독립적이다. idle reaper가 꺼져 있어도 owner는 계속
        # 살아 있음을 증명해야 하며, 그렇지 않으면 lease가 만료된 순간 peer가 살아 있는
        # 컨테이너를 가져간다(idle_timeout: 0은 지원되는 설정이다).
        self._start_lease_renewal()

        # 활성화되어 있으면 idle checker를 시작한다
        if self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT) > 0:
            self._start_idle_checker()

    @property
    def uses_thread_data_mounts(self) -> bool:
        """thread의 workspace/uploads/outputs가 mount를 통해 보이는지 여부.

        로컬 컨테이너 backend는 thread 데이터 디렉터리를 bind-mount하므로 gateway가 쓴
        파일이 sandbox 시작 시점에 이미 보인다. 원격 backend는 명시적인 파일 sync가
        필요할 수 있다. gateway와 원격 sandbox가 같은 storage를 공유하는 경우 운영자가
        이 자동 감지를 override할 수 있다.
        """
        override = self._config.get("thread_data_mounts")
        if override is not None:
            return override
        return isinstance(self._backend, LocalContainerBackend)

    # ── Factory 메서드 ────────────────────────────────────────────────────

    def _create_backend(self) -> SandboxBackend:
        """설정에 맞는 backend를 생성한다.

        선택 로직(순서대로 확인):
        1. ``provisioner_url``이 설정됨 → RemoteSandboxBackend(provisioner 모드)
              provisioner가 k3s에 Pod + Service를 동적으로 생성한다.
        2. 기본값 → LocalContainerBackend(로컬 모드)
              로컬 provider가 컨테이너 lifecycle(start/stop)을 직접 관리한다.
        """
        provisioner_url = self._config.get("provisioner_url")
        if provisioner_url:
            logger.info(f"Using remote sandbox backend with provisioner at {provisioner_url}")
            api_key = self._config.get("provisioner_api_key", "")
            return RemoteSandboxBackend(provisioner_url=provisioner_url, api_key=api_key)

        logger.info("Using local container sandbox backend")
        return LocalContainerBackend(
            image=self._config["image"],
            base_port=self._config["port"],
            container_prefix=self._config["container_prefix"],
            config_mounts=self._config["mounts"],
            environment=self._config["environment"],
        )

    # ── 설정 ──────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        """app config에서 sandbox 설정을 읽어온다."""
        config = get_app_config()
        sandbox_config = config.sandbox

        idle_timeout = getattr(sandbox_config, "idle_timeout", None)
        replicas = getattr(sandbox_config, "replicas", None)

        return {
            "image": sandbox_config.image or DEFAULT_IMAGE,
            "port": sandbox_config.port or DEFAULT_PORT,
            "container_prefix": sandbox_config.container_prefix or DEFAULT_CONTAINER_PREFIX,
            "idle_timeout": idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,
            "replicas": replicas if replicas is not None else DEFAULT_REPLICAS,
            "mounts": sandbox_config.mounts or [],
            "thread_data_mounts": getattr(sandbox_config, "thread_data_mounts", None),
            "environment": self._resolve_env_vars(sandbox_config.environment or {}),
            "ownership": getattr(sandbox_config, "ownership", None),
            # redis stream bridge를 쓴다는 것은 multi-instance 배포라는 뜻이고, ownership
            # store도 거기에 맞춰 기본값을 정해야 한다. env var만 보지 말고 bridge의
            # resolver가 읽는 것과 같은 소스를 읽는다.
            "stream_bridge": getattr(config, "stream_bridge", None),
            # 동적 pod 관리를 위한 provisioner URL (예: http://provisioner:8002)
            "provisioner_url": getattr(sandbox_config, "provisioner_url", None) or "",
            "provisioner_api_key": getattr(sandbox_config, "provisioner_api_key", None) or "",
        }

    @staticmethod
    def _resolve_env_vars(env_config: dict[str, str]) -> dict[str, str]:
        """환경 변수 참조($로 시작하는 값)를 해석한다."""
        resolved = {}
        for key, value in env_config.items():
            if isinstance(value, str) and value.startswith("$"):
                env_name = value[1:]
                resolved[key] = os.environ.get(env_name, "")
            else:
                resolved[key] = str(value)
        return resolved

    # ── 인스턴스 간 ownership lease ────────────────────────────────────────

    def _publish_ownership(self, sandbox_id: str) -> None:
        """acquire 경로에서 *sandbox_id*에 대한 책임을 가져온다.

        이 thread를 마지막으로 처리한 인스턴스가 누구든 그로부터 넘겨받는다. 컨테이너는
        (user, thread)마다 결정적이므로 여기로 라우팅된 turn은 정당한 인계다. 이전 owner는
        다음 renewal에서 LOST를 보고받고, 컨테이너를 건드리지 않은 채 tracking만 중단한다.

        의도적으로 fail-open이 **아니다**. 에러를 삼키고 sandbox를 그냥 넘겨주면 사용 중인데도
        주인이 없는 상태가 되고, peer가 이를 orphan으로 보고 회수해버린다 — 바로 이 store가
        막으려는 실패다. 호출자는 이 예외를 전파시켜야 한다.

        intent mark는 round trip **전에** 게시하며, 그 순서가 핵심이다. ``take()``는 반환하기
        전에 인계를 durable하게 만든다 — redis에서는 응답이 아직 전송 중일 때 서버가 이미 SET을
        커밋한 상태다 — 따라서 epoch를 그 뒤에 올리면 store는 이미 컨테이너가 우리 것이라고
        말하는데 guard는 아직 아니라고 읽는 구간이 생긴다. 오래된 ``LOST``를 들고 있는 renewal이
        그 틈으로 그대로 들어와 map을 비우고, 이 호출이 막 돌려주려던 client를 닫아버린다.
        그러면 acquire는 provider가 더 이상 tracking하지 않는 id를 반환하고 ``get()``은
        ``None``을 답한다. guard는 자신이 지키는 전이보다 늦지 않게 보여야 하는데, epoch만으로는
        불가능하다. 전이를 수행하는 호출이 반환된 뒤에야 쓸 수 있기 때문이다.

        그래서 두 mark가 두 축을 나눠 맡고 둘 다 필요하다. intent mark는 "acquire가 진행 중"을,
        epoch는 "당신이 판단한 이후 acquire가 완료됨"을 담당한다.

        Raises:
            SandboxBeingDestroyedError: peer가 이 컨테이너를 내리는 중이므로 agent에게
                넘겨서는 안 된다(destroy → 재acquire 경쟁).
            OwnershipBackendError: ownership을 게시하지 못했다.
        """
        with self._lock:
            self._acquire_inflight[sandbox_id] = self._acquire_inflight.get(sandbox_id, 0) + 1
        try:
            if not self._ownership.take(sandbox_id):
                raise SandboxBeingDestroyedError(sandbox_id)
            with self._lock:
                self._acquire_epoch_counter += 1
                self._acquire_epoch[sandbox_id] = self._acquire_epoch_counter
        finally:
            # set이 아니라 count를 쓴다. 지금은 한 id에 대한 acquire가 thread별 lock으로
            # 직렬화되므로 set이어도 동등하지만, 그건 두 계층 위 호출자에 대한 가정이다.
            # 그 가정이 깨지면 set은 가장 먼저 끝난 쪽이 지워버려 이 구간을 조용히 다시
            # 열어준다. count를 쓰면 그 가정 자체가 없어진다.
            with self._lock:
                remaining = self._acquire_inflight.get(sandbox_id, 0) - 1
                if remaining > 0:
                    self._acquire_inflight[sandbox_id] = remaining
                else:
                    self._acquire_inflight.pop(sandbox_id, None)

    # ── 같은 프로세스 내 배제(store가 제공하지 않는 나머지 절반) ─────────────
    #
    # lease는 *peer*를 배제한다. `claim()`은 설계상 우리 자신의 `own:` lease에 대해 성공하고
    # (destroy 경로가 이미 소유한 것을 claim할 수 있는 이유다), `take()`도 마찬가지다. 따라서
    # 이 프로세스의 reaper thread들(idle checker, renewal, eviction)과 자기 자신의 acquire
    # 경로 사이에서 store는 **배제를 전혀 제공하지 않는다**. 모든 reaper는 `self._lock` 밖에서
    # 판단하므로(store round trip을 모든 acquire를 지키는 lock 안에서 붙들고 있으면 안 된다),
    # 각자가 이미 acquire에 의해 무효화되었을 수도 있는 판단에 따라 행동한다. 아래 두 helper가
    # 그 빠진 절반이며, 방향별로 하나씩이다.
    #
    #   reaping  — 곧 멈추거나 버릴 것이므로 아무도 promote하면 안 된다. 예약(reserve)하고,
    #              모든 promote 경로가 peer의 `del:`을 존중하는 것과 똑같이 그 예약을
    #              존중하게 한다.
    #   forgetting — peer가 정당하게 소유했고 그쪽이 이겨야 하므로, 감지해야 할 것은 promote다.
    #              판단 시점의 acquire epoch를 비교한다.

    def _reserve_local_teardown(self, sandbox_id: str, still_reapable: Callable[[], bool]) -> bool:
        """이 프로세스가 teardown하려고 *sandbox_id*를 예약한다.

        ``still_reapable``은 예약과 **같은** 임계 구역에서 평가되므로, 마지막 확인과 mark 사이로
        acquire가 끼어들 수 없다. 이 짝지음이 핵심이다. 먼저 확인하고 나중에 mark하는 것은 그
        구간을 좁힌 게 아니라 구간 그 자체다.

        그 결과이자 새 호출자가 알아야 할 유일한 규칙: **predicate는 ``self._lock``을 쥔 채로
        실행된다**. 이 lock은 평범한 ``Lock``이므로, lock을 건드리는 predicate는 — 직접이든
        lock을 잡는 provider 메서드를 통해서든 — deadlock을 일으킨다. predicate는 map을 읽는
        값싸고 블로킹하지 않는 연산이어야 한다(``sandbox_id in self._warm_pool``,
        ``_last_activity`` 비교 등). 이 제약을 우회 설계하지 않고 명시만 한 것은 의도적이다.
        이를 허용하려고 lock을 reentrant로 만들면, 눈에 띄는 hang 대신 이 provider 전체에
        조용한 재진입 버그를 얻게 된다.
        """
        with self._lock:
            if sandbox_id in self._local_teardown or not still_reapable():
                return False
            self._local_teardown.add(sandbox_id)
            return True

    def _finish_local_teardown(self, sandbox_id: str) -> None:
        with self._lock:
            self._local_teardown.discard(sandbox_id)

    def _being_torn_down_locally(self, sandbox_id: str) -> bool:
        """*이* 프로세스의 reaper thread가 *sandbox_id*를 붙들고 있는지 여부.

        호출자는 이미 ``self._lock``을 쥐고 있어야 한다.
        """
        return sandbox_id in self._local_teardown

    def _acquire_epoch_of(self, sandbox_id: str) -> int:
        """acquire 세대를 snapshot해서 낡은 판단을 감지할 수 있게 한다.

        ``_publish_ownership``만 이 값을 올린다 — 즉 acquire 경로가 sandbox를 agent에게
        넘기는 길에 lease를 (다시) 가져가는 바로 그 순간이다. ``_refresh_ownership``에서
        만료된 lease를 다시 세우는 경우는 의도적으로 올리지 않는다. 넘겨준 것이 없으므로 그
        id에 대한 reaper의 판단은 여전히 유효하다.
        """
        with self._lock:
            return self._acquire_epoch.get(sandbox_id, 0)

    def _claim_ownership(self, sandbox_id: str, *, for_destroy: bool = False) -> bool:
        """*sandbox_id*의 ownership을 가져오거나 갱신한다.

        claim이 성공해야 컨테이너에 대한 조작이 안전해진다. 우리가 lease를 쥐고 있는 동안
        peer의 claim은 실패한다. ``for_destroy``를 주면 lease에 teardown 표시가 추가되고,
        동시에 진행되는 acquire 쪽 ``take()``가 이를 거부한다 — 삭제된 sandbox별 flock guard가
        메우던 ownership 확인 → 컨테이너 stop 사이의 구간을 이렇게 닫는다.

        backend 에러 시 fail-closed다. ownership을 알 수 없으면 "우리 것이 아님"으로 취급해
        컨테이너를 adopt하지도 destroy하지도 않는다.
        """
        try:
            return self._ownership.claim(sandbox_id, for_destroy=for_destroy)
        except OwnershipBackendError as e:
            logger.warning("Sandbox ownership claim failed for %s (treating as not owned): %s", sandbox_id, e)
            return False

    def _release_ownership(self, sandbox_id: str) -> None:
        try:
            self._ownership.release(sandbox_id)
        except OwnershipBackendError as e:
            # best effort다. lease는 스스로 만료되므로 release 실패는 ownership을 망가뜨리는
            # 대신 재사용을 늦출 뿐이다.
            logger.warning("Failed to release sandbox ownership for %s: %s", sandbox_id, e)

    def _refresh_ownership(self, sandbox_id: str) -> bool:
        """*sandbox_id*의 lease를 계속 유지한다. peer가 가져갔으면 False.

        **만료된(lapsed)** lease는 잃어버린 것으로 취급하지 않고 다시 세운다. 아무도 쥐고
        있지 않으므로 재claim이 안전하고, 이것이 (모든 키가 사라지는) Redis 재시작 때문에
        전체 fleet의 살아 있는 sandbox가 전부 회수되는 일을 막는다. peer가 실제로 쥐고 있는
        lease는 절대 다시 가져오지 않는다 — 그게 #4206 사고다.
        """
        try:
            outcome = self._ownership.renew(sandbox_id)
        except OwnershipBackendError as e:
            # 잃은 게 아니라 알 수 없는 것이다. sandbox를 유지하고 다음 tick에 재시도한다.
            # 정말로 죽은 owner가 lease를 붙들고 있는 시간은 여전히 TTL이 제한한다.
            logger.warning("Could not renew sandbox ownership for %s, will retry: %s", sandbox_id, e)
            return True

        if outcome is RenewOutcome.RENEWED:
            return True
        if outcome is RenewOutcome.LAPSED:
            # 비어 있으니 다시 세운다. 여기는 의도적으로 fail-open인 renewal 경로이므로
            # `_claim_ownership`을 쓸 수 없다. 그 helper는 adopt/reap 호출자를 위해 backend
            # 에러를 False로 바꾸는데, 그러면 이 두 round trip 사이의 장애를 peer의 인계와
            # 혼동해 살아 있는 sandbox를 회수해버린다. 위에서 `renew()` 자체가 답하지 못한
            # 경우와 똑같이, 여기서도 "알 수 없음"은 유지하고 재시도한다는 뜻이다.
            try:
                if self._ownership.claim(sandbox_id):
                    logger.info("Re-established a lapsed ownership lease for %s", sandbox_id)
                    return True
            except OwnershipBackendError as e:
                logger.warning("Could not re-establish lapsed lease for %s, will retry: %s", sandbox_id, e)
                return True
            logger.warning("Lapsed ownership lease for %s was taken by a peer", sandbox_id)
            return False
        return False

    @contextlib.contextmanager
    def _held_teardown_lease(self, sandbox_id: str):
        """컨테이너 stop이 진행되는 내내 *sandbox_id*의 teardown marker를 살려둔다.

        ``claim(..., for_destroy=True)``는 일반 lease TTL로 ``del:`` marker를 쓰고, 평범한
        ``renew()``는 ``own:``만 연장하며 teardown은 의도적으로 ``LOST``로 보고한다. active와
        unhealthy destroy 경로는 ``_renew_owned_leases``가 순회하는 map에서 sandbox를 빼지만,
        warm 경로는 stop이 성공할 때까지 항목을 남겨두므로 ``_forget_lost_sandbox``가 우리
        자신의 marker를 peer의 인계로 오해하지 않도록 별도로 ``_local_teardown``을 존중한다.
        이 heartbeat가 없으면, TTL보다 오래 걸린 컨테이너 stop 도중 marker가 만료되고, 아직
        살아 있는 컨테이너에 대해 peer의 ``take()``가 성공하며, 그 stop이 방금 컨테이너를
        넘겨받은 turn 위로 떨어진다 — ``del:`` 상태가 닫으려던 바로 그 구간이 자기 만료 때문에
        다시 열리는 것이다.

        sandbox별 ``flock``이 공짜로 해주던 일이 이것이다. 쥐고 있는 lock은 만료되지 않는다.
        lease는 만료되므로, 배제가 자신이 지키는 작업보다 오래갈 것이라 가정하지 말고 의도적으로
        붙들고 있어야 한다. 비정상 backend가 아니어도 도달 가능하다. config 스키마는
        ``renewal_interval_seconds``(> 0)와 ``ttl_multiplier``(>= 2)만 제한하므로 합법적인
        설정으로도 TTL이 정상적인 컨테이너 stop보다 짧아질 수 있고,
        ``LocalContainerBackend._stop_container``는 ``subprocess.run``에 ``timeout``을 주지
        않으므로 멈춰버린 daemon은 기본 120초에서도 무한정 블로킹한다.

        TTL을 유한하게 두는 것은 의도적이다. heartbeat는 프로세스와 함께 죽으므로, stop 도중
        크래시한 destroyer도 컨테이너를 영원히 파괴 불가로 표시하는 대신 TTL 하나 뒤에는
        풀어준다.

        마지막 release는 호출자가 아니라 heartbeat 자신의 마지막 동작이다. context가 빠져나갈
        때 아직 진행 중인 refresh ``claim``이 있으면(socket timeout이 제한하지만 호출 중일 수
        있다) 호출자 쪽 release *뒤에* 도착해서, 이미 stop이 끝난 컨테이너에 ``del:`` marker를
        다시 써버린다 — 그러면 새 ``take()``(또는 새로 만든 컨테이너의 롤백)가 TTL까지 묶인다.
        loop가 멈춘 뒤 heartbeat 내부에서 release하면 release가 마지막 refresh보다 반드시 뒤에
        오게 되므로 그 뒤를 따르는 claim이 있을 수 없다.
        """
        stop = threading.Event()

        def beat() -> None:
            interval = self._ownership_config.renewal_interval_seconds
            try:
                while not stop.wait(interval):
                    try:
                        if not self._ownership.claim(sandbox_id, for_destroy=True):
                            # store가 우리 marker를 잃고 *동시에* peer가 그것을 가져간
                            # 경우에만 도달한다(예: stop 도중 flush). stop은 이미
                            # 진행 중이라 되돌릴 수 없으므로, peer의 컨테이너가 흔적도
                            # 없이 죽게 두는 대신 크게 알린다.
                            logger.error(
                                "Lost the teardown exclusion for %s while its container stop was still in flight; a peer may have taken it",
                                sandbox_id,
                            )
                            return
                    except Exception as e:
                        # 의도적으로 넓게 잡는다. refresh가 예외를 던졌다고 heartbeat가
                        # 죽어서, 무한정 실행될 수 있는 stop을 위한 marker를 방치하면 안
                        # 된다. 잃은 게 아니라 알 수 없는 것이다 — marker는 아직 살아 있을
                        # 수 있고 낡은 marker는 TTL이 정리한다. 다음 tick에 재시도한다.
                        logger.warning("Could not refresh the teardown lease for %s, will retry: %s", sandbox_id, e)
            finally:
                # release는 heartbeat 자신이 마지막에 한다. 그래야 진행 중인 refresh가
                # marker가 지워진 뒤에 실행되는 일이 없다. `release()`는 우리 자신의 lease만
                # 지우므로, 위에서 peer가 가져갔다면 안전한 no-op이 된다.
                self._release_ownership(sandbox_id)

        beater = threading.Thread(target=beat, name="sandbox-teardown-lease", daemon=True)
        beater.start()
        try:
            yield
        finally:
            stop.set()
            beater.join(timeout=self._TEARDOWN_JOIN_TIMEOUT_SECONDS)
            if beater.is_alive():
                # 이 예산은 정상적으로 timeout되는 refresh와 마지막 release를 합친 값이다.
                # release는 heartbeat의 몫이고 아직 남아 있다. 여기서 marker를 지우면 이
                # 컨텍스트가 담당하는 바로 그 경쟁이 다시 열리므로 그대로 둔다 — thread가
                # 풀리면 release하거나, TTL이 정리한다.
                logger.warning(
                    "Teardown heartbeat for %s did not exit within %.1fs; its lease release is deferred to that thread",
                    sandbox_id,
                    self._TEARDOWN_JOIN_TIMEOUT_SECONDS,
                )

    # ── 기동 시 reconciliation ─────────────────────────────────────────────

    def _adoptable_after_grace(self, sandbox_id: str, now: float) -> bool:
        """*sandbox_id*가 진짜 orphan이라고 볼 만큼 오래 주인 없이 보였는지 여부.

        lease가 없다는 것은 보통 owner가 죽고 TTL이 지났다는 증거다. 하지만 모든 owner가
        살아서 서비스 중인데도 store가 모든 키를 잃을 수 있다 — persistence 없는 Redis 재시작,
        또는 ``maxmemory`` 압박에 의한 eviction. ``_refresh_ownership``은 이미 그 신호를
        포기로 읽기를 거부한다(``LAPSED``는 넘겨주는 게 아니라 다시 세운다). 여기서 같은
        신호를 "orphan이니 adopt"로 읽으면 다른 경로와 모순된다. 먼저 reconcile한 쪽이 각
        owner의 다음 renewal tick 전 구간에서 살아 있는 컨테이너를 전부 adopt하고, 그 owner의
        renewal은 ``LOST``를 보고받아 자신이 실제로 서비스 중인 sandbox를 놓아버리며,
        adopter가 그것을 idle-destroy한다 — 뒷문으로 들어온 #4206이다.

        lease TTL 하나를 온전히 기다리면 상태 손실이 지워버린 지연이 복원된다. 살아 있는
        owner는 renewal 간격 하나 안에 다시 게시하고 그 간격은 구조상 TTL보다 짧으므로
        (``ttl_multiplier >= 2``), grace 전체 동안 주인 없이 남는 컨테이너는 owner가 정말로
        사라진 것뿐이다.
        """
        if not self._ownership.supports_cross_process:
            # 이 store가 우리에게 보여줄 lease를 peer가 쥐고 있을 수 없으므로, 주인 없는
            # 컨테이너는 살아 있는 peer의 것일 수 없다 — 이 프로세스의 죽은 수명 주기에서
            # 남은 것이다. 단일 인스턴스 배포는 즉시 정리를 유지하고, 이 store에서는 grace가
            # multi-worker 배포에도 도움이 되지 않는다. grace가 있든 없든 peer들의 lease는
            # 서로에게 보이지 않기 때문이다.
            return True

        try:
            current_owner = self._ownership.owner(sandbox_id)
        except OwnershipBackendError as e:
            # 비어 있는 게 아니라 알 수 없는 것이다. _claim_ownership과 마찬가지로 fail closed.
            logger.warning("Could not read sandbox ownership for %s during reconciliation (deferring adoption): %s", sandbox_id, e)
            return False

        if current_owner is not None:
            # 주인이 있다 — peer이거나 이미 우리다. 어느 쪽이든 orphan이 아니며, 살아 있는
            # owner가 다시 게시했다면 낡은 타이머가 그 lease를 넘어 만료되게 두지 말고 grace를
            # 다시 시작해야 한다.
            self._unowned_since.pop(sandbox_id, None)
            return False

        first_seen = self._unowned_since.setdefault(sandbox_id, now)
        return now - first_seen >= compute_lease_ttl(self._ownership_config)

    def _reconcile_orphans(self) -> None:
        """이전 프로세스 수명 주기가 남긴 orphan 컨테이너를 정리한다.

        기동 시(그리고 idle checker를 통해 주기적으로) 우리 prefix에 맞는 실행 중 컨테이너를
        열거하고 **진짜 orphan**만 warm pool로 adopt한다. 이 인스턴스가 ownership lease를
        claim할 수 있을 때만 adopt하므로, multi-instance gateway가 peer의 살아 있는 sandbox를
        adopt한 뒤 idle-destroy하는 일이 생기지 않는다(#4206).

        adopt된 orphan은 새 warm-pool timestamp를 받고, ``idle_timeout`` 안에 아무도 다시
        acquire하지 않으면 idle checker가 파괴한다. 크래시한 프로세스가 남긴 컨테이너도 그
        lease가 만료되면 여전히 이렇게 정리된다.

        주인 없는 컨테이너를 보자마자 adopt하지는 않는다. 먼저 recovery grace 동안 계속 주인
        없이 남아 있어야 하며, 그래야 상태를 잃은 store를 죽은 owner 무리로 오인하지 않는다
        (``_adoptable_after_grace`` 참고).
        """
        try:
            running = self._backend.list_running()
        except Exception as e:
            logger.warning(f"Failed to enumerate running containers during startup reconciliation: {e}")
            return

        # 더 이상 존재하지 않는 컨테이너의 grace 타이머를 버려서, 오래 사는 인스턴스가 파괴된
        # 컨테이너마다 항목을 쌓지 않게 한다. 빈 목록 반환보다 앞에서 실행해 그 경우에도
        # 비워지게 한다.
        running_ids = {info.sandbox_id for info in running}
        self._unowned_since = {sid: seen for sid, seen in self._unowned_since.items() if sid in running_ids}

        if not running:
            return

        current_time = time.time()
        adopted = 0
        skipped_live = 0
        deferred = 0

        for info in running:
            age = current_time - info.created_at if info.created_at > 0 else float("inf")
            if not self._adoptable_after_grace(info.sandbox_id, current_time):
                deferred += 1
                logger.debug("Deferring container %s during reconciliation: owned, or not yet past the recovery grace", info.sandbox_id)
                continue

            # 두 번째로 claim한다. claim이 성공하면 그 컨테이너가 peer의 것이 아님이
            # 증명되고 peer가 잠긴다. 하지만 *우리*에 대해서는 아무것도 말해주지 않는다 —
            # 설계상 우리 자신의 lease에 대해서는 성공한다 — 따라서 아래의 local teardown
            # 확인을 대체하지 못한다. 위의 grace도 마찬가지로 전제 조건일 뿐 대체재가
            # 아니다. atomic한 것은 claim뿐이다.
            if not self._claim_ownership(info.sandbox_id):
                skipped_live += 1
                logger.debug("Skipping container %s during reconciliation: owned by another instance", info.sandbox_id)
                continue

            # 컨테이너당 lock 획득 한 번으로 atomic한 check-and-insert를 한다.
            # "이미 tracking 중인가?" 확인과 warm-pool 삽입 사이의 TOCTOU 구간을 없앤다.
            with self._lock:
                if info.sandbox_id in self._sandboxes or info.sandbox_id in self._warm_pool:
                    continue
                if self._being_torn_down_locally(info.sandbox_id):
                    # adoption도 promote이므로 나머지 세 경로와 같은 예약 확인이 필요하다.
                    # 여기서 teardown 중인 컨테이너는 tracking되지 않은 채 아직 실행 중인데,
                    # 이 루프가 adopt하는 모습이 정확히 그것이다 — 그리고 claim도 grace도
                    # 그것을 배제하지 못한다. `memory`에서는 grace가 아예 건너뛰어지므로
                    # (`supports_cross_process = False`) 막아줄 것이 전혀 없다. adopt하면
                    # stop이 도착하기 직전에 컨테이너를 warm pool에 넣게 되고, 다음 reclaim이
                    # 죽은 항목을 넘겨주게 된다.
                    deferred += 1
                    logger.debug("Deferring container %s during reconciliation: this instance is tearing it down", info.sandbox_id)
                    continue
                self._warm_pool[info.sandbox_id] = (info, current_time)
                self._warm_pool_identity[info.sandbox_id] = None
            self._unowned_since.pop(info.sandbox_id, None)
            adopted += 1
            logger.info(f"Adopted container {info.sandbox_id} into warm pool (age: {age:.0f}s)")

        logger.info(
            "Startup reconciliation complete: %s adopted into warm pool, %s skipped (live peer ownership), %s deferred (owned or within recovery grace), %s total found",
            adopted,
            skipped_live,
            deferred,
            len(running),
        )

    # ── 결정적 ID ─────────────────────────────────────────────────────────

    @staticmethod
    def _effective_acquire_user_id(user_id: str | None) -> str:
        return user_id or get_effective_user_id()

    @staticmethod
    def _thread_key(thread_id: str, user_id: str) -> tuple[str, str]:
        return (user_id, thread_id)

    @staticmethod
    def _deterministic_sandbox_id(thread_id: str, user_id: str) -> str:
        """user/thread 범위에서 결정적인 sandbox ID를 생성한다.

        user_id를 포함하므로, 이전에 만들어진 기본 bucket sandbox가 user 범위 bucket을
        mount해야 하는 auth/channel 실행에 재사용되지 않는다.

        버전이 섞인 롤아웃 중에는 기존 8자 컨테이너가 새 16자 identity로 재사용되지 않는다.
        첫 신버전 acquire가 cold-start하는 동안 기존 컨테이너는 일반 orphan 정리 대상으로
        남는다.
        """
        return hashlib.sha256(f"{user_id}:{thread_id}".encode()).hexdigest()[:16]

    def _assert_active_identity_available_locked(
        self,
        sandbox_id: str,
        requested_key: tuple[str, str],
    ) -> None:
        """활성 상태의 잘린 ID가 다른 identity의 것이면 fail closed 한다."""
        if sandbox_id not in self._sandboxes and sandbox_id not in self._sandbox_infos:
            return

        stored_key = self._active_sandbox_identity.get(sandbox_id)
        if stored_key is None:
            matching_keys = [key for key, mapped_id in self._thread_sandboxes.items() if mapped_id == sandbox_id]
            if len(matching_keys) == 1:
                stored_key = matching_keys[0]
        if stored_key != requested_key:
            raise SandboxIdentityCollisionError(sandbox_id, stored_key, requested_key)

    def _assert_warm_identity_available_locked(
        self,
        sandbox_id: str,
        requested_key: tuple[str, str],
    ) -> None:
        """warm ID가 acquire 도중 테넌트를 바꿨으면 fail closed 한다."""
        if sandbox_id not in self._warm_pool:
            return
        # 기동 시 adopt된 항목은 첫 reclaim 전까지 identity를 알 수 없다.
        stored_key = self._warm_pool_identity.get(sandbox_id)
        if stored_key is not None and stored_key != requested_key:
            raise SandboxIdentityCollisionError(sandbox_id, stored_key, requested_key)

    # ── Mount 관련 helper ─────────────────────────────────────────────────

    def _get_extra_mounts(self, thread_id: str | None, *, user_id: str | None = None) -> list[tuple[str, str, bool]]:
        """sandbox에 붙일 추가 mount를 모두 모은다(thread 전용 + skills)."""
        mounts: list[tuple[str, str, bool]] = []

        if thread_id:
            mounts.extend(self._get_thread_mounts(thread_id, user_id=user_id))
            logger.info(f"Adding thread mounts for thread {thread_id}: {mounts}")

        skills_mounts = self._get_skills_mounts(user_id=user_id)
        if skills_mounts:
            mounts.extend(skills_mounts)
            logger.info(f"Adding skills mounts: {skills_mounts}")

        user_skill_mounts = self._get_user_skill_mounts(user_id=user_id)
        if user_skill_mounts:
            mounts.extend(user_skill_mounts)
            logger.info(f"Adding user skill mounts: {user_skill_mounts}")

        lark_cli_mounts = self._get_lark_cli_runtime_mounts(user_id=user_id)
        if lark_cli_mounts:
            mounts.extend(lark_cli_mounts)
            logger.info(f"Adding Lark CLI runtime mounts: {lark_cli_mounts}")

        return self._dedupe_mounts_by_container_path(mounts)

    @staticmethod
    def _dedupe_mounts_by_container_path(mounts: list[tuple[str, str, bool]]) -> list[tuple[str, str, bool]]:
        """컨테이너 경로마다 첫 번째 mount만 남긴다.

        컨테이너 경로가 중복되면 provisioner가 거부하고, 로컬 Docker 생성도 실패할 수 있다.
        mount helper가 우선순위 순서(thread 데이터, skill 루트, integration skill 루트,
        그다음 integration runtime/credential)로 추가되므로 먼저 온 mount가 이긴다.
        """
        seen: set[str] = set()
        deduped: list[tuple[str, str, bool]] = []
        for host_path, container_path, read_only in mounts:
            if container_path in seen:
                logger.warning(
                    "Skipping duplicate sandbox mount for container path %s from host %s",
                    container_path,
                    host_path,
                )
                continue
            seen.add(container_path)
            deduped.append((host_path, container_path, read_only))
        return deduped

    @staticmethod
    def _get_thread_mounts(thread_id: str, *, user_id: str | None = None) -> list[tuple[str, str, bool]]:
        """thread의 데이터 디렉터리에 대한 volume mount를 만든다.

        디렉터리가 없으면 생성한다(lazy 초기화). mount source는 host_base_dir을 쓰므로,
        Docker socket을 mount한 Docker 안에서 실행할 때(DooD) host Docker daemon이 경로를
        해석할 수 있다.
        """
        paths = get_paths()
        effective_user_id = AioSandboxProvider._effective_acquire_user_id(user_id)
        paths.ensure_thread_dirs(thread_id, user_id=effective_user_id)

        return [
            (paths.host_sandbox_work_dir(thread_id, user_id=effective_user_id), f"{VIRTUAL_PATH_PREFIX}/workspace", False),
            (paths.host_sandbox_uploads_dir(thread_id, user_id=effective_user_id), f"{VIRTUAL_PATH_PREFIX}/uploads", False),
            (paths.host_sandbox_outputs_dir(thread_id, user_id=effective_user_id), f"{VIRTUAL_PATH_PREFIX}/outputs", False),
            # ACP workspace: sandbox 안에서는 읽기 전용이다(lead agent가 결과를 읽고, ACP
            # subprocess는 컨테이너 안이 아니라 host 쪽에서 쓴다).
            (paths.host_acp_workspace_dir(thread_id, user_id=effective_user_id), "/mnt/acp-workspace", True),
        ]

    @staticmethod
    def _get_skills_mounts(*, user_id: str | None = None) -> list[tuple[str, str, bool]]:
        """3분할 skills 레이아웃에 대한 skills 디렉터리 mount 설정을 만든다.

        AIO sandbox용으로 ``LocalSandboxProvider._build_thread_path_mappings``를 그대로
        반영한다. public, 사용자별 custom, legacy(마이그레이션 이전의 global-custom) skill을
        각각 다른 컨테이너 하위 디렉터리에 mount해서, ``Skill.get_container_path()``의
        카테고리 인식 경로가 sandbox 안에서 올바르게 해석되게 한다.

        Docker 안에서 실행할 때(DooD)는 mount source가 ``DEER_FLOW_HOST_BASE_DIR``를 쓰므로
        host Docker daemon이 projection 경로를 해석할 수 있다.
        """
        mounts: list[tuple[str, str, bool]] = []
        try:
            config = get_app_config()
            container_path = config.skills.container_path
            effective_user_id = AioSandboxProvider._effective_acquire_user_id(user_id)
            AioSandboxProvider._ensure_skills_projection(effective_user_id)
            paths = get_paths()
            host_base_dir = str(paths.host_base_dir)

            # 1. public skills: 전역, 읽기 전용 — 정적이며 모든 thread가 공유한다
            mounts.append(
                (
                    join_host_path(host_base_dir, "skills_view", "public"),
                    f"{container_path}/public",
                    True,
                )
            )

            # 2. 사용자별 custom skills: 읽기 전용, thread/user 단위
            host_user_custom = join_host_path(
                host_base_dir,
                "users",
                effective_user_id,
                "skills_view",
                "custom",
            )
            mounts.append(
                (
                    host_user_custom,
                    f"{container_path}/custom",
                    True,
                )
            )

            # 3. legacy 가시성은 projection 내용으로 표현된다. 디렉터리가 비어 있어도 mount를
            # 그대로 유지해서, 나중에 상태가 바뀌어도 sandbox를 다시 만들지 않고 반영되게 한다.
            mounts.append(
                (
                    join_host_path(host_base_dir, "users", effective_user_id, "skills_view", "legacy"),
                    f"{container_path}/legacy",
                    True,
                )
            )
        except Exception as e:
            logger.warning("Could not setup skills mounts: %s", e)

        return mounts

    @staticmethod
    def _ensure_skills_projection(user_id: str):
        """best-effort다. projection 실패가 sandbox acquire를 실패시켜서는 안 된다.

        ``_acquire_internal`` / ``_acquire_internal_async``에서 try/except 없이 (side effect
        목적으로) 직접 호출되고, ``_get_skills_mounts``의 자체 보호 블록 안에서도 호출된다 —
        여기서 예외를 삼키면 guard를 중복하지 않고도 두 호출 지점 모두 안전해진다.
        """
        from deerflow.skills.projection import ensure_skill_projections
        from deerflow.skills.storage import get_or_new_user_skill_storage

        try:
            storage = get_or_new_user_skill_storage(user_id, app_config=get_app_config())
            return ensure_skill_projections(storage)
        except Exception as exc:
            logger.warning("Could not ensure skills projection for user %s: %s", user_id, exc, exc_info=True)
            return None

    @staticmethod
    def _get_user_skill_mounts(*, user_id: str | None = None) -> list[tuple[str, str, bool]]:
        """활성화된 managed integration skill을 AIO sandbox에 mount한다.

        사용자별 custom skill은 이미 ``_get_skills_mounts``가 mount한다. integration 패키지는
        공유되지만 활성화 상태는 사용자별이므로, 이 helper는 공유 루트 원본 대신 사용자의
        projection을 mount한다.
        """
        try:
            config = get_app_config()
            paths = get_paths()
            skills_container_path = config.skills.container_path
            effective_user_id = AioSandboxProvider._effective_acquire_user_id(user_id)
            AioSandboxProvider._ensure_skills_projection(effective_user_id)
            return [
                (
                    join_host_path(
                        str(paths.host_base_dir),
                        "users",
                        effective_user_id,
                        "skills_view",
                        "integrations",
                    ),
                    f"{skills_container_path}/integrations",
                    True,
                ),
            ]
        except Exception as e:
            logger.warning(f"Could not setup user skill mounts: {e}")
            return []

    @staticmethod
    def _lark_integration_active(user_id: str | None = None) -> bool:
        """이 사용자에게 managed Lark skill pack이 설치되어 있는지 여부.

        sandbox가 lark-cli runtime(init container / Gateway 다운로드 mount)을 요청할지를
        결정한다. 로컬 ``sandbox-cli`` 디렉터리 존재 여부와 무관하므로, 원격/K8s는 Gateway 쪽
        다운로드 없이도 사용할 수 있다.
        """
        try:
            effective_user_id = AioSandboxProvider._effective_acquire_user_id(user_id)
            return lark_skills_installed(effective_user_id)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not determine Lark integration state: {e}")
            return False

    @staticmethod
    def _lark_broker_active(user_id: str | None = None) -> bool:
        """이 사용자의 sandbox가 lark-cli broker(Pattern B)를 써야 하는지 여부.

        Lark pack이 설치되어 있고 **동시에** 원격 provisioner가 설정된 broker 이미지를 보고할
        때만 True다. True이면 provisioner가 credential을 sidecar에 두고 sandbox에는 shim만
        주므로, Gateway 쪽 credential mount overlay도 실행하면 안 된다.
        """
        try:
            if not AioSandboxProvider._lark_integration_active(user_id):
                return False
            from deerflow.integrations.lark_cli import sandbox_lark_broker_active

            return sandbox_lark_broker_active()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not determine Lark broker state: {e}")
            return False

    @staticmethod
    def _get_lark_cli_runtime_mounts(*, user_id: str | None = None) -> list[tuple[str, str, bool]]:
        """Settings 인증이 쓰는 사용자별 lark-cli config/data 디렉터리를 mount한다.

        Settings 엔드포인트는 Gateway에서 ``LARKSUITE_CLI_CONFIG_DIR`` / ``DATA_DIR``를
        ``users/{user}/integrations/lark-cli``로 가리킨 채 ``lark-cli``를 실행한다. agent 대화는
        sandbox 안에서 ``lark-cli``를 실행하므로 같은 디렉터리를 컨테이너에 mount하지 않으면
        CLI가 인증되지 않은 별도 프로필을 보게 된다.

        ``config`` 디렉터리는 수명이 긴 Lark ``appSecret``을 담고 있으므로(Gateway의
        ``lark-cli config init``이 쓰며 sandbox 안에서는 절대 쓰지 않는다) **읽기 전용**으로
        mount한다. sandbox 프로세스는 읽기만 하면 되고, 읽기 전용 bind는 침해된 agent가 app
        credential을 변조하거나 교체하지 못하게 막는다. 최신 ``lark-cli``는 ``config/locks``를
        통해 API 호출을 조율하므로, 그 빈 하위 디렉터리만 쓰기 가능하게 덧mount해서 나머지
        ``config``는 쓰기에 노출하지 않는다. ``data`` 디렉터리는 ``lark-cli auth``가 sandbox
        안에서 갱신하는 OAuth 토큰을 담으므로 쓰기 가능한 상태를 유지한다.
        이것은 방어의 한 겹일 뿐이다 — auth-proxy 후속 작업(issue #4338)이 반영되기 전까지 두
        디렉터리 모두 임의의 sandbox 프로세스가 읽을 수 있다. ``backend/AGENTS.md``의 sandbox
        신뢰 경계 설명을 참고한다.
        """
        try:
            paths = get_paths()
            effective_user_id = AioSandboxProvider._effective_acquire_user_id(user_id)
            ensure_lark_cli_credential_tree(effective_user_id, paths=paths)
            config_dir = paths.host_user_integration_config_dir(effective_user_id, LARK_CLI_INTEGRATION_ID)
            mounts = [
                (config_dir, LARK_CLI_SANDBOX_CONFIG_DIR, True),
                (join_host_path(config_dir, "locks"), LARK_CLI_SANDBOX_LOCKS_DIR, False),
                (paths.host_user_integration_data_dir(effective_user_id, LARK_CLI_INTEGRATION_ID), LARK_CLI_SANDBOX_DATA_DIR, False),
            ]
            runtime_dir = paths.base_dir / "integrations" / LARK_CLI_INTEGRATION_ID / "sandbox-cli"
            if runtime_dir.is_dir():
                mounts.append(
                    (
                        join_host_path(str(paths.host_base_dir), "integrations", LARK_CLI_INTEGRATION_ID, "sandbox-cli"),
                        LARK_CLI_SANDBOX_RUNTIME_DIR,
                        True,
                    )
                )
            return mounts
        except Exception as e:
            logger.warning(f"Could not setup Lark CLI runtime mounts: {e}")
            return []

    # ── idle timeout 관리 ─────────────────────────────────────────────────

    def _cleanup_idle_resources(self, idle_timeout: float) -> None:
        """``idle_timeout``초보다 오래 유휴 상태인 AIO 리소스를 정리한다."""
        # 기동 이후 peer의 lease가 만료된 컨테이너를 회수한다(크래시 경로).
        self._reconcile_orphans()
        self._cleanup_idle_sandboxes(idle_timeout)

    # ── ownership lease 갱신 ───────────────────────────────────────────────

    def _start_lease_renewal(self) -> None:
        """이 인스턴스의 lease를 살려두는 daemon thread를 시작한다.

        의도적으로 idle checker에 합치지 않았다. 그 thread는 ``idle_timeout > 0``일 때만
        시작하므로, renewal을 거기에 얹으면 ``idle_timeout: 0`` 배포(지원되는 설정이다.
        "shutdown까지 warm VM 유지")에서 조용히 멈춰 모든 lease가 만료되고 TTL 하나 뒤에
        #4206이 다시 열린다. liveness와 reaping은 같은 스위치를 공유하면 안 된다.
        """
        if self._renewal_thread is not None and self._renewal_thread.is_alive():
            return

        self._renewal_stop.clear()
        self._renewal_thread = threading.Thread(
            target=self._lease_renewal_loop,
            name="sandbox-lease-renewal",
            daemon=True,
        )
        self._renewal_thread.start()
        logger.info(
            "Started sandbox ownership renewal thread (interval: %.1fs, ttl: %.1fs)",
            self._ownership_config.renewal_interval_seconds,
            self._ownership_config.renewal_interval_seconds * self._ownership_config.ttl_multiplier,
        )

    def _stop_lease_renewal(self) -> None:
        self._renewal_stop.set()
        thread = self._renewal_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _lease_renewal_loop(self) -> None:
        interval = self._ownership_config.renewal_interval_seconds
        while not self._renewal_stop.wait(interval):
            try:
                self._renew_owned_leases()
            except Exception:
                logger.exception("Error in sandbox ownership renewal loop")

    def _renew_owned_leases(self) -> None:
        """이 인스턴스가 소유했다고 믿는 모든 컨테이너의 lease를 갱신한다.

        active 항목뿐 아니라 warm 항목도 포함한다. warm 컨테이너도 여전히 우리 것이므로(빠른
        reclaim을 위해 붙들고 있다), lease를 만료시키면 곧 그 thread에 돌려줄 컨테이너를 peer가
        adopt해버린다.

        **peer**가 지금 쥐고 있는 lease만이 컨테이너가 더 이상 우리 것이 아님을 뜻한다. 만료된
        lease는 다시 세운다(``_refresh_ownership`` 참고). 둘을 혼동하면 store가 상태를 잃는
        순간 이 인스턴스의 살아 있는 sandbox가 전부 회수된다.
        """
        with self._lock:
            owned_ids = list(self._sandboxes.keys()) + list(self._warm_pool.keys())

        for sandbox_id in owned_ids:
            # round trip 전에 snapshot한다. `renew()`가 LOST를 답할 즈음이면 이 프로세스의
            # acquire가 이미 lease를 되찾고 그 id를 promote했을 수 있으며, 그 답은 그때 우리가
            # 쥐고 있던 lease에 대한 것이다.
            epoch = self._acquire_epoch_of(sandbox_id)
            if not self._refresh_ownership(sandbox_id):
                logger.warning("Lost sandbox ownership lease for %s; dropping it from this instance", sandbox_id)
                self._forget_lost_sandbox(sandbox_id, expected_epoch=epoch)

    def _forget_lost_sandbox(self, sandbox_id: str, *, expected_epoch: int | None = None) -> None:
        """lease를 더 이상 쥐고 있지 않은 sandbox를 컨테이너는 건드리지 않고 버린다.

        컨테이너는 이제 lease를 쥔 인스턴스의 것이므로, 여기서 멈추면 이 store가 막으려는 바로
        그 인스턴스 간 kill이 된다. 사라지는 것은 우리 host 쪽 handle뿐이다.

        ``expected_epoch``는 "잃었다"는 판단이 lock 밖의 store round trip에서 나온 호출자를
        보호한다. **진행 중인** acquire도 여기 해당한다. 그 ``take()``는 epoch가 아직 기록되지
        않은 상태에서 이미 인계를 durable하게 만들 수 있으므로, epoch만으로는 낡은 판단이
        통과한다(``_publish_ownership`` 참고). 그 구간에서 lease를 되찾은 acquire는 이미
        sandbox를 turn에 넘겼고, reuse 경로에서는 *같은* tracking 중인 client를 넘겨줬으므로
        객체 identity 비교로는 알아챌 수 없다. 그때 버리면 turn 도중 client가 닫히고, agent는
        다음 turn까지 tool call이 실패하는 id를 들고 있게 된다.
        """
        with self._lock:
            # warm teardown은 backend stop이 성공할 때까지 의도적으로 항목을 남겨둔다. 자신의
            # `del:` marker 때문에 `renew()`가 LOST를 보고하지만 그건 peer의 인계가 아니므로
            # 남겨둔 항목을 없애면 안 된다 — 특히 stop이 실패해서 컨테이너가 재시도/reclaim을
            # 위해 살아 있는 경우에 그렇다. 성공 시에는 teardown 경로가 제거한다.
            if sandbox_id in self._local_teardown:
                logger.debug("Not dropping sandbox %s: this instance is tearing it down", sandbox_id)
                return
            # in-flight 확인은 의도적으로 `expected_epoch` 유무와 *무관하게* 동작한다. 지금
            # epoch를 넘기지 않는 호출자(두 개의 `SandboxBeingDestroyedError` 핸들러)는 같은
            # id에 대한 publish와 충돌할 수 없다 — 그들이 실행될 때면 `_publish_ownership`이
            # 이미 mark를 지웠고, 한 id에 대한 acquire는 thread별 lock으로 직렬화된다 —
            # 따라서 현재 동작은 달라지지 않는다. 그럼에도 여기 있는 이유는, "epoch를 주지
            # 않음"이 "guard가 전혀 없음"으로 읽히는 것이 위험한 primitive의 다음 호출자가
            # 잘못 작성되는 방식이기 때문이다. 지금 acquire 중인 id는 절대 버리면 안 된다.
            if sandbox_id in self._acquire_inflight:
                logger.info("Not dropping sandbox %s: an acquire is publishing ownership for it", sandbox_id)
                return
            if expected_epoch is not None and self._acquire_epoch.get(sandbox_id, 0) != expected_epoch:
                logger.info("Not dropping sandbox %s: this instance re-acquired it after the lease check", sandbox_id)
                return

            sandbox = self._sandboxes.pop(sandbox_id, None)
            self._sandbox_infos.pop(sandbox_id, None)
            self._active_sandbox_identity.pop(sandbox_id, None)
            self._last_activity.pop(sandbox_id, None)
            self._warm_pool.pop(sandbox_id, None)
            self._warm_pool_identity.pop(sandbox_id, None)
            self._acquire_epoch.pop(sandbox_id, None)
            for key, mapped_id in list(self._thread_sandboxes.items()):
                if mapped_id == sandbox_id:
                    del self._thread_sandboxes[key]

        # 버리는 host 쪽 HTTP client를 닫는다(#2872). 컨테이너 자체는 새 owner를 위해 계속
        # 살아 있다.
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing sandbox {sandbox_id} after losing its lease: {e}")

    def _cleanup_idle_sandboxes(self, idle_timeout: float) -> None:
        current_time = time.time()
        active_to_destroy = []

        with self._lock:
            # active sandbox는 _last_activity로 추적한다
            for sandbox_id, last_activity in self._last_activity.items():
                idle_duration = current_time - last_activity
                if idle_duration > idle_timeout:
                    active_to_destroy.append(sandbox_id)
                    logger.info(f"Sandbox {sandbox_id} idle for {idle_duration:.1f}s, marking for destroy")

        # active sandbox를 파괴한다(행동 전에 여전히 유휴인지 다시 확인).
        #
        # 재확인은 teardown 예약과 같은 임계 구역에서 일어나야 하므로, 여기서 실행하지 않고
        # predicate로 `_destroy_tracked`에 넘긴다. 여기서 확인하고 나중에 파괴하면 turn이
        # sandbox를 다시 acquire한 뒤 그 밑에서 컨테이너가 멈춰버리는 구간이 남는데, 이 PR에서
        # `destroy()`가 untrack 전에 ownership을 claim하게 되면서 그 구간이 명령어 몇 개에서
        # store round trip 하나로 넓어졌다.
        def still_idle(sandbox_id: str) -> bool:
            last_activity = self._last_activity.get(sandbox_id)
            if last_activity is None:
                # 다른 경로가 이미 release하거나 파괴했다 — 건너뛴다.
                logger.info(f"Sandbox {sandbox_id} already gone before idle destroy, skipping")
                return False
            if (time.time() - last_activity) < idle_timeout:
                # snapshot 이후 다시 acquire되었다(활동 시각 갱신) — 건너뛴다.
                logger.info(f"Sandbox {sandbox_id} was re-acquired before idle destroy, skipping")
                return False
            return True

        for sandbox_id in active_to_destroy:
            try:
                logger.info(f"Destroying idle sandbox {sandbox_id}")
                self._destroy_tracked(sandbox_id, still_reapable=lambda sid=sandbox_id: still_idle(sid))
            except Exception as e:
                logger.error(f"Failed to destroy idle sandbox {sandbox_id}: {e}")

        self._reap_expired_warm(idle_timeout)

    def _reap_expired_warm(self, idle_timeout: float | None = None) -> None:
        """``idle_timeout``보다 오래된 warm 항목을 파괴한다. peer의 살아 있는 컨테이너는 절대 건드리지 않는다."""
        timeout = float(self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT) if idle_timeout is None else idle_timeout)
        if timeout <= 0:
            return

        now = time.time()
        expired: list[tuple[str, SandboxInfo]] = []
        with self._lock:
            for sandbox_id, (entry, timestamp) in self._warm_pool.items():
                if now - timestamp > timeout:
                    expired.append((sandbox_id, entry))

        # 정말로 사라진다는 것을 확인한 뒤에만 warm pool에서 항목을 제거한다. 먼저 제거하면
        # claim이 거부되거나 답을 못 받았을 때 컨테이너를 잃는다. 여전히 실행 중인데 아무도
        # tracking하지 않게 되는 것이다. 제거를 미루기 때문에 예약이 필요하다 — stop이 진행되는
        # 내내 항목이 `_reclaim_warm_pool_sandbox`에 계속 보인다.
        for sandbox_id, entry in expired:
            self._destroy_warm_entry(sandbox_id, entry, reason="idle_timeout", still_reapable=lambda sid=sandbox_id: sid in self._warm_pool)

    def _evict_oldest_warm(self) -> str | None:
        """이 인스턴스가 아직 소유한 가장 오래된 warm 항목을 evict한다."""
        with self._lock:
            if not self._warm_pool:
                return None
            # lock 안에서 오래된 순으로 snapshot한다. ownership 확인은 lock 밖에서 한다.
            # claim은 network round trip일 수 있고 provider lock은 모든 acquire 경로를
            # 지키기 때문이다.
            candidates = [(sandbox_id, entry) for sandbox_id, (entry, _) in sorted(self._warm_pool.items(), key=lambda item: item[1][1])]

        for sandbox_id, entry in candidates:
            # "아직 warm pool에 있는가?"가 reapable 확인이며, 예약과 같은 임계 구역에서
            # 실행되어야 한다 — 여기서 확인하고 나중에 예약하는 것이 바로 reclaim이 끼어드는
            # 구간이다. `_destroy_warm_entry`가 lock 한 번으로 둘 다 처리한다.
            if not self._destroy_warm_entry(sandbox_id, entry, reason="replica_enforcement", still_reapable=lambda sid=sandbox_id: sid in self._warm_pool):
                continue
            return sandbox_id

        return None

    # ── Signal 처리 ───────────────────────────────────────────────────────

    def _register_signal_handlers(self) -> None:
        """graceful shutdown을 위한 signal handler를 등록한다.

        SIGTERM, SIGINT, SIGHUP(터미널 종료)을 처리해서 사용자가 터미널을 닫아도 sandbox
        컨테이너가 정리되게 한다.
        """
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

        def signal_handler(signum, frame):
            self.shutdown()
            if signum == signal.SIGTERM:
                original = self._original_sigterm
            elif hasattr(signal, "SIGHUP") and signum == signal.SIGHUP:
                original = self._original_sighup
            else:
                original = self._original_sigint
            if callable(original):
                original(signum, frame)
            elif original == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal_handler)
        except ValueError:
            logger.debug("Could not register signal handlers (not main thread)")

    # ── Thread locking (프로세스 내) ───────────────────────────────────────

    def _get_thread_lock(self, thread_id: str, user_id: str) -> threading.Lock:
        """특정 user/thread 범위의 in-process lock을 가져오거나 생성한다."""
        key = self._thread_key(thread_id, user_id)
        with self._lock:
            if key not in self._thread_locks:
                self._thread_locks[key] = threading.Lock()
            return self._thread_locks[key]

    def _sandbox_id_for_thread(self, thread_id: str | None, user_id: str | None) -> str:
        """thread sandbox에는 결정적 ID를, 그 외에는 랜덤 ID를 반환한다."""
        return self._deterministic_sandbox_id(thread_id, self._effective_acquire_user_id(user_id)) if thread_id else str(uuid.uuid4())[:8]

    def _reuse_in_process_sandbox(self, thread_id: str | None, *, user_id: str | None = None, post_lock: bool = False) -> str | None:
        """thread용으로 아직 tracking 중인 활성 in-process sandbox가 있으면 재사용한다."""
        if thread_id is None:
            return None

        effective_user_id = self._effective_acquire_user_id(user_id)
        key = self._thread_key(thread_id, effective_user_id)
        with self._lock:
            if key not in self._thread_sandboxes:
                return None

            existing_id = self._thread_sandboxes[key]
            if self._being_torn_down_locally(existing_id):
                # 이 프로세스의 reaper thread가 이 컨테이너를 멈추는 중이다.
                # peer의 `del:` lease와 같은 답을 낸다. 대신 cold-start 한다.
                logger.info("Cached sandbox %s is being destroyed by this instance; not reusing it", existing_id)
                return None
            if existing_id in self._sandboxes:
                info = self._sandbox_infos.get(existing_id)
            else:
                del self._thread_sandboxes[key]
                return None

        alive = self._check_tracked_sandbox_alive(existing_id, info) if info is not None else True
        if alive is False:
            self._drop_unhealthy_sandbox(
                existing_id,
                "in-process cache failed health check",
                expected_info=info,
            )
            return None

        with self._lock:
            if self._thread_sandboxes.get(key) != existing_id:
                return None
            if existing_id not in self._sandboxes:
                self._thread_sandboxes.pop(key, None)
                return None

            suffix = " (post-lock check)" if post_lock else ""
            logger.info(f"Reusing in-process sandbox {existing_id} for user/thread {effective_user_id}/{thread_id}{suffix}")
            self._last_activity[existing_id] = time.time()

        # fail closed다. ownership을 게시하지 못한 sandbox를 넘겨주는 대신
        # OwnershipBackendError를 전파한다.
        try:
            self._publish_ownership(existing_id)
        except SandboxBeingDestroyedError:
            # peer가 이 컨테이너를 멈추는 중이다. 곧 사라질 sandbox를 넘겨주는 대신 이것을
            # 버리고 호출자가 discover-or-create로 새 것을 얻게 한다.
            logger.info("Cached sandbox %s is being destroyed by another instance; not reusing it", existing_id)
            self._forget_lost_sandbox(existing_id)
            return None

        with self._lock:
            if self._being_torn_down_locally(existing_id):
                # 첫 예약 확인은 backend health check와 ownership round trip 이전에
                # 실행됐다. 둘 중 하나가 진행되는 동안 로컬 reaper가 이길 수 있고, 그
                # reaper는 destroy claim이 성공할 때까지 의도적으로 항목을 `_sandboxes`에
                # 남겨둔다. 따라서 map에 들어 있다는 사실만으로는 이 id를 돌려줘도 안전하다는
                # 증명이 되지 않는다.
                logger.info("Cached sandbox %s was reserved for teardown while publishing ownership; not reusing it", existing_id)
                return None
            if existing_id not in self._sandboxes:
                # 게시하는 동안 버려졌다. intent mark는 `_publish_ownership` *내부*의 구간을
                # 닫지만 그 이전의 틈은 닫지 못한다. mark가 설정되기 전까지 renewal의
                # `LOST`는 최신이면서 정확하다 — peer가 정말로 lease를 쥐고 있었다 — 따라서
                # forget이 정당하게 실행되어 이 client를 닫는다. 그래도 id를 반환하면
                # `get()`이 `None`인 sandbox를 돌려주는 셈이 된다. 대신 그냥 흘려보낸다.
                # 호출자가 다시 discover해서 새 client를 만들고, 방금 가져온 lease는 이미
                # 우리 것이다.
                logger.info("Cached sandbox %s was dropped while publishing ownership; falling through to discovery", existing_id)
                return None
        return existing_id

    def _reclaim_warm_pool_sandbox(
        self,
        thread_id: str | None,
        sandbox_id: str,
        *,
        user_id: str | None = None,
        post_lock: bool = False,
    ) -> str | None:
        """가능하면 warm-pool sandbox를 다시 active tracking으로 승격한다."""
        if thread_id is None:
            return None

        effective_user_id = self._effective_acquire_user_id(user_id)
        key = self._thread_key(thread_id, effective_user_id)
        with self._lock:
            if sandbox_id not in self._warm_pool:
                return None
            self._assert_warm_identity_available_locked(sandbox_id, key)
            if self._being_torn_down_locally(sandbox_id):
                # 항목은 stop이 진행되는 내내 의도적으로 `_warm_pool`에 남는다(claim이
                # 거부돼도 컨테이너를 잃지 않기 위해서다). 따라서 pool에 있다는 것만으로는
                # reclaim 가능하다는 뜻이 아니다.
                logger.info("Warm-pool sandbox %s is being destroyed by this instance; not reclaiming it", sandbox_id)
                return None

            info, _ = self._warm_pool[sandbox_id]

        alive = self._check_tracked_sandbox_alive(sandbox_id, info)
        if alive is False:
            self._drop_unhealthy_sandbox(
                sandbox_id,
                "warm-pool cache failed health check",
                expected_info=info,
            )
            return None

        # warm → active 전이 전에 ownership을 게시한다. 여기서 예외가 나도 sandbox가 active로
        # tracking되면서 주인이 없는 상태로 남아서는 안 된다(peer가 orphan으로 보고 turn 도중
        # 회수해버린다). 실패하면 항목은 warm으로 남고 이 인스턴스는 기존 lease를 유지한다.
        try:
            self._publish_ownership(sandbox_id)
        except SandboxBeingDestroyedError:
            logger.info("Warm-pool sandbox %s is being destroyed by another instance; not reclaiming it", sandbox_id)
            self._forget_lost_sandbox(sandbox_id)
            return None

        with self._lock:
            if self._being_torn_down_locally(sandbox_id):
                # 첫 확인이 round trip 이전이었으므로 다시 확인한다. reaper는 우리의
                # `take()` *뒤에* 예약할 수 있고 — warm 항목은 stop이 끝날 때까지 제거가
                # 미뤄지므로 아직 남아 있다 — 그런 다음 `del:`을 claim해서(성공한다. lease는
                # 방금 우리가 가져온 우리 것이다) 컨테이너를 멈춘다. 어느 쪽 제거가 먼저
                # 도착하느냐가 결정하며, 우리 쪽이 먼저면 이미 멈춘 컨테이너에 client를
                # 설치하게 된다.
                logger.info("Warm-pool sandbox %s was claimed for teardown while publishing ownership; not reclaiming it", sandbox_id)
                return None
            self._assert_warm_identity_available_locked(sandbox_id, key)
            warm_item = self._warm_pool.pop(sandbox_id, None)
            if warm_item is None:
                return None
            self._warm_pool_identity.pop(sandbox_id, None)
            info, _ = warm_item
            sandbox = AioSandbox(id=sandbox_id, base_url=info.sandbox_url)
            self._sandboxes[sandbox_id] = sandbox
            self._sandbox_infos[sandbox_id] = info
            self._active_sandbox_identity[sandbox_id] = key
            self._last_activity[sandbox_id] = time.time()
            self._thread_sandboxes[key] = sandbox_id

        suffix = " (post-lock check)" if post_lock else f" at {info.sandbox_url}"
        logger.info(f"Reclaimed warm-pool sandbox {sandbox_id} for user/thread {effective_user_id}/{thread_id}{suffix}")
        return sandbox_id

    def _recheck_cached_sandbox(self, thread_id: str, sandbox_id: str, *, user_id: str) -> str | None:
        """프로세스 간 file lock을 획득한 뒤 in-memory 캐시를 다시 확인한다."""
        return self._reuse_in_process_sandbox(thread_id, user_id=user_id, post_lock=True) or self._reclaim_warm_pool_sandbox(
            thread_id,
            sandbox_id,
            user_id=user_id,
            post_lock=True,
        )

    def _register_discovered_sandbox(self, thread_id: str, info: SandboxInfo, *, user_id: str) -> str:
        """backend를 통해 발견한 sandbox를 tracking에 등록한다.

        Raises:
            SandboxBeingDestroyedError: discovery가 아직 실행 중인 컨테이너를 찾았지만
                peer가 그것을 멈추는 중이다. 삼키지 않고 의도적으로 전파한다. create로
                흘려보내면 아직 제거되지 않은 컨테이너 이름과 충돌하고, 이것을 agent에게
                넘기는 것은 이 store가 막으려는 turn 도중 사망(#4206) 그 자체다. 이 구간은
                peer의 진행 중인 컨테이너 stop이므로, thread의 다음 turn은 아무것도 발견하지
                못하고 깔끔하게 cold-start 한다.
        """
        key = self._thread_key(thread_id, user_id)
        with self._lock:
            if self._being_torn_down_locally(info.sandbox_id):
                # 캐시가 빗나가면 discovery로 흘러오므로, 여기는 reaper 자신의 untrack이
                # 열어주는 경로이기도 하다. `take()`는 reaper의 `del:` claim이 도착한
                # 뒤에야 이것을 거부한다. 그 전까지는 우리 자신의 lease에 대해 성공한다.
                raise SandboxBeingDestroyedError(info.sandbox_id)
            self._assert_active_identity_available_locked(info.sandbox_id, key)
            self._assert_warm_identity_available_locked(info.sandbox_id, key)

        sandbox = AioSandbox(id=info.sandbox_id, base_url=info.sandbox_url)
        # ownership을 먼저 잡아서, 실패해도 tracking되지만 주인 없는 sandbox가 남지 않게 한다.
        # 롤백할 컨테이너는 없지만(우리가 만든 게 아니다) 위에서 만든 host 쪽 HTTP client는
        # 우리 것이라 누수되면 안 된다 — `_register_created_sandbox`와 같은 실패 시 close다.
        try:
            self._publish_ownership(info.sandbox_id)
            with self._lock:
                if self._being_torn_down_locally(info.sandbox_id):
                    # 게시 전 예약 확인은 조기 탈출용일 뿐이다. store round trip 도중
                    # 로컬 reaper가 이 id를 예약할 수 있다. 그 reaper가 이미 멈추기로 한
                    # 컨테이너에 client를 설치하면 안 된다.
                    raise SandboxBeingDestroyedError(info.sandbox_id)
                self._assert_active_identity_available_locked(info.sandbox_id, key)
                self._assert_warm_identity_available_locked(info.sandbox_id, key)
                # active와 warm은 배타적인 상태이고, 그것을 깰 수 있는 것은 이 insert뿐이다.
                # 같은 id의 warm 항목은 그 id가 active가 되는 순간 낡은 것이 된다. 그대로
                # 두면 컨테이너에 reaper가 둘 생긴다 — `_reap_expired_warm`은 warm timestamp로
                # 판단하고 `_last_activity`는 전혀 보지 않으므로, `_sandboxes`가 여전히
                # client를 넘겨주는 동안 agent가 쓰고 있는 컨테이너를 멈춘다.
                self._warm_pool.pop(info.sandbox_id, None)
                self._warm_pool_identity.pop(info.sandbox_id, None)
                self._sandboxes[info.sandbox_id] = sandbox
                self._sandbox_infos[info.sandbox_id] = info
                self._active_sandbox_identity[info.sandbox_id] = key
                self._last_activity[info.sandbox_id] = time.time()
                self._thread_sandboxes[key] = info.sandbox_id
        except (
            OwnershipBackendError,
            SandboxBeingDestroyedError,
            SandboxIdentityCollisionError,
        ):
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing sandbox {info.sandbox_id} after failed ownership publish: {e}")
            raise

        logger.info(f"Discovered existing sandbox {info.sandbox_id} for user/thread {user_id}/{thread_id} at {info.sandbox_url}")
        return info.sandbox_id

    def _register_created_sandbox(self, thread_id: str | None, sandbox_id: str, info: SandboxInfo, *, user_id: str | None = None) -> str:
        """새로 만든 sandbox를 active map에 등록한다."""
        sandbox = AioSandbox(id=sandbox_id, base_url=info.sandbox_url)
        key = (
            self._thread_key(
                thread_id,
                self._effective_acquire_user_id(user_id),
            )
            if thread_id
            else None
        )
        # ownership을 먼저 잡는다. discover 경로와 달리 여기에는 롤백할 것이 있다. 방금 이
        # 컨테이너를 시작했고, 주인 없이 실행 중인 컨테이너야말로 peer의 reconciliation이
        # adopt하는 대상이다. 그대로 흘리면 이 인스턴스가 곧 쓸 컨테이너를 peer에게 넘기는
        # 셈이 된다.
        # 여기서도 SandboxBeingDestroyedError가 날 수 있다. stop 도중 죽은 peer가 TTL이
        # 만료될 때까지 teardown marker를 남기기 때문이다. 두 경우 모두 롤백하지 않으면 방금
        # 시작한 컨테이너가 누수된다.
        try:
            if key is not None:
                with self._lock:
                    self._assert_active_identity_available_locked(sandbox_id, key)
                    self._assert_warm_identity_available_locked(sandbox_id, key)
            self._publish_ownership(sandbox_id)

            with self._lock:
                if key is not None:
                    self._assert_active_identity_available_locked(sandbox_id, key)
                    self._assert_warm_identity_available_locked(sandbox_id, key)
                # discover 경로와 같은 배타성 규칙.
                self._warm_pool.pop(sandbox_id, None)
                self._warm_pool_identity.pop(sandbox_id, None)
                self._sandboxes[sandbox_id] = sandbox
                self._sandbox_infos[sandbox_id] = info
                self._active_sandbox_identity[sandbox_id] = key
                self._last_activity[sandbox_id] = time.time()
                if key is not None:
                    self._thread_sandboxes[key] = sandbox_id
        except (
            OwnershipBackendError,
            SandboxBeingDestroyedError,
            SandboxIdentityCollisionError,
        ):
            logger.error(
                "Could not register new sandbox %s; destroying it rather than leaking an untracked container",
                sandbox_id,
            )
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing sandbox {sandbox_id} during ownership rollback: {e}")
            try:
                self._backend.destroy(info)
            except Exception as e:
                logger.error(
                    "Failed to destroy sandbox %s after registration failure: %s",
                    sandbox_id,
                    e,
                )
            raise

        logger.info(f"Created sandbox {sandbox_id} for thread {thread_id} at {info.sandbox_url}")
        return sandbox_id

    def _check_tracked_sandbox_alive(self, sandbox_id: str, info: SandboxInfo) -> bool | None:
        """tracking 중인 sandbox가 살아 있어 보이는지 반환한다. 알 수 없으면 None."""
        try:
            return self._backend.is_alive(info)
        except Exception as e:
            logger.warning(f"Failed to check sandbox {sandbox_id} health: {e}")
            return None

    def _remove_tracked_sandbox(
        self,
        sandbox_id: str,
        *,
        expected_info: SandboxInfo | None = None,
    ) -> tuple[Sandbox | None, SandboxInfo | None, bool]:
        """in-process tracking map에서 sandbox를 제거한다.

        expected_info를 주면, 현재 tracking 중인 active 또는 warm-pool 항목이 확인 당시의 바로
        그 info 객체일 때만 제거한다. 낡은 health check 결과 때문에 같은 결정적 id로 새로 만든
        sandbox가 지워지는 것을 막는다.
        """
        thread_keys_to_remove: list[tuple[str, str]] = []

        with self._lock:
            active_info = self._sandbox_infos.get(sandbox_id)
            warm_item = self._warm_pool.get(sandbox_id)
            warm_info = warm_item[0] if warm_item is not None else None
            if expected_info is not None and active_info is not expected_info and warm_info is not expected_info:
                return None, None, False

            sandbox = self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            self._active_sandbox_identity.pop(sandbox_id, None)
            thread_keys_to_remove = [key for key, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for key in thread_keys_to_remove:
                del self._thread_sandboxes[key]
            self._last_activity.pop(sandbox_id, None)
            self._acquire_epoch.pop(sandbox_id, None)
            if info is None and sandbox_id in self._warm_pool:
                info, _ = self._warm_pool.pop(sandbox_id)
            else:
                self._warm_pool.pop(sandbox_id, None)
            self._warm_pool_identity.pop(sandbox_id, None)

        return sandbox, info, True

    def _drop_unhealthy_sandbox(self, sandbox_id: str, reason: str, *, expected_info: SandboxInfo | None = None) -> None:
        """health check가 확실히 실패한 뒤 sandbox를 제거하고 파괴한다."""
        # stop만이 아니라 경로 전체를 예약한다. 여기는 untrack을 먼저 하므로, untrack과
        # `del:` claim 사이에 acquire가 캐시를 놓치고 discovery로 흘러가는데 거기서는
        # `take()`가 여전히 우리 자신의 lease에 대해 성공한다.
        if not self._reserve_local_teardown(sandbox_id, lambda: True):
            logger.info(f"Skipped dropping sandbox {sandbox_id}: already being torn down by this instance")
            return
        try:
            self._drop_unhealthy_reserved(sandbox_id, reason, expected_info=expected_info)
        finally:
            self._finish_local_teardown(sandbox_id)

    def _drop_unhealthy_reserved(self, sandbox_id: str, reason: str, *, expected_info: SandboxInfo | None = None) -> None:
        sandbox, info, removed = self._remove_tracked_sandbox(sandbox_id, expected_info=expected_info)
        if not removed:
            logger.info(f"Skipped dropping sandbox {sandbox_id}: tracked info changed after health check")
            return

        if sandbox is not None:
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing unhealthy sandbox {sandbox_id}: {e}")

        if info is not None:
            # 다른 모든 reap 경로와 똑같이 게이트를 건다. 컨테이너는 확실한 health check에
            # 실패했지만, "우리에게 확실히 죽었다"가 그것이 우리 것이라는 증명은 아니다.
            # peer가 이 id 뒤의 컨테이너를 교체했을 수 있고, 그렇다면 멈추는 것은 또다시
            # 인스턴스 간 kill이다.
            if self._claim_ownership(sandbox_id, for_destroy=True):
                try:
                    # 다른 두 stop 경로처럼 marker를 붙들고 있는다. 여기는 claim 전에
                    # untrack하므로 `_renew_owned_leases`도 이 id를 볼 수 없고, marker를
                    # 갱신할 다른 무엇도 없다. heartbeat가 종료 시(성공이든 실패든) marker를
                    # 해제하므로, 늦은 refresh와 경쟁할 호출자 쪽 release가 없다.
                    with self._held_teardown_lease(sandbox_id):
                        self._backend.destroy(info)
                except Exception as e:
                    logger.warning(f"Error destroying unhealthy sandbox {sandbox_id}: {e}")
            else:
                logger.info("Not destroying unhealthy sandbox %s: owned by another instance", sandbox_id)

        logger.warning(f"Dropped unhealthy sandbox {sandbox_id}: {reason}")

    def _active_count_locked(self) -> int:
        """``_lock``을 쥔 상태에서 활성 AIO sandbox 개수를 반환한다."""
        return len(self._sandboxes)

    def _destroy_warm_entry(self, sandbox_id: str, entry: SandboxInfo, *, reason: str, still_reapable: Callable[[], bool]) -> bool:
        """AIO 전용 backend 로깅을 사용해 warm-pool sandbox를 파괴한다.

        destroy용 claim은 **peer**에 대한 배제다. lease에 teardown 표시가 붙어서 다른
        인스턴스의 동시 acquire가 거부되고, 이 판단과 stop 사이에 컨테이너가 다시 acquire될
        수 없다. 이 짝지음이 sandbox별 flock guard를 대체했다. claim이 실패하면 — peer 소유든
        backend 장애든 — fail closed로 파괴하지 않는다.

        하지만 이 프로세스에 대한 배제는 *아니다*. `claim()`은 우리 자신의 `own:` lease에
        대해 성공하므로, 그 전에 실행된 같은 프로세스의 reclaim이 컨테이너를 가져갔다면 이
        stop은 이미 그것을 쓰고 있는 turn 위로 떨어진다. 예약이 그 나머지 절반이고, claim보다
        먼저 잡는다. claim 이후에는 stop이 진행되는 내내 항목이 `_warm_pool`에 그대로 보이므로
        예약이 없으면 reclaim이 여전히 그것을 찾아낸다.

        ``still_reapable``에 무조건 참인 기본값을 주지 않고 필수로 만든 이유는, 새 호출 지점이
        이 문제를 생각하게 만드는 쪽이 안전한 기본값이기 때문이다. 이 시그니처가
        ``WarmPoolLifecycleMixin``의 hook과 의도적으로 다른 것도 그래서다. 이 provider가 mixin
        호출자 두 곳(``_evict_oldest_warm`` / ``_reap_expired_warm``)을 모두 override하므로
        안전하며, 그 override가 사라지면 mixin의 호출이 조용히 구간을 다시 여는 대신 여기서
        요란하게 실패한다.

        Returns:
            컨테이너를 멈췄고 호출자가 warm-pool 항목을 제거해야 하면 ``True``,
            아직 실행 중이면 ``False``.
        """
        if not self._reserve_local_teardown(sandbox_id, still_reapable):
            logger.info("Refusing to destroy warm-pool sandbox %s for %s: reclaimed by this instance", sandbox_id, reason)
            return False

        try:
            if not self._claim_ownership(sandbox_id, for_destroy=True):
                logger.info("Refusing to destroy warm-pool sandbox %s for %s: owned by another instance", sandbox_id, reason)
                return False

            try:
                # marker는 자신이 쓰인 TTL이 아니라 stop보다 오래 살아야 하며, 종료 시
                # heartbeat가 해제한다. stop이 실패한 경우에도 그 해제는 똑같이 중요하다 —
                # 컨테이너는 아마 아직 살아 있으므로, 남은 marker가 그 thread의 재acquire를
                # 막게 된다.
                with self._held_teardown_lease(sandbox_id):
                    self._backend.destroy(entry)
            except Exception as e:
                if reason == "idle_timeout":
                    logger.error(f"Failed to destroy idle warm-pool sandbox {sandbox_id}: {e}")
                elif reason == "replica_enforcement":
                    logger.error(f"Failed to destroy warm-pool sandbox {sandbox_id}: {e}")
                else:
                    logger.error(f"Failed to destroy warm-pool sandbox {sandbox_id} for {reason}: {e}")
                return False

            # 항목 제거를 호출자에게 맡기지 않고 예약 안에서 여기서 처리한다. stop이 끝날 때
            # 예약을 풀고 그 뒤에 제거하면, 컨테이너는 이미 멈췄는데 항목은 아직 `_warm_pool`에
            # 있고 아무 표시도 없는 틈이 생긴다 — 그러면 reclaim이 그것을 집어 죽은 컨테이너를
            # 넘겨준다. 제거는 여전히 *stop*에 대해서는 미뤄지지만(거부되거나 실패한 stop은
            # 항목을 남긴다), 예약에 대해서는 더 이상 미뤄지지 않는다.
            with self._lock:
                current = self._warm_pool.get(sandbox_id)
                if current is not None and current[0] is entry:
                    self._warm_pool.pop(sandbox_id, None)
                    self._warm_pool_identity.pop(sandbox_id, None)
        finally:
            self._finish_local_teardown(sandbox_id)

        if reason == "idle_timeout":
            logger.info(f"Destroyed idle warm-pool sandbox {sandbox_id}")
        elif reason == "replica_enforcement":
            logger.info(f"Destroyed warm-pool sandbox {sandbox_id}")
        else:
            logger.info(f"Destroyed warm-pool sandbox {sandbox_id} for {reason}")
        return True

    # ── 핵심: acquire / get / release / shutdown ──────────────────────────

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """sandbox 환경을 확보하고 그 ID를 반환한다.

        같은 thread_id에 대해서는 여러 turn, 여러 프로세스, 그리고 (공유 storage가 있다면)
        여러 pod에 걸쳐 같은 sandbox_id를 반환한다.

        프로세스 내 lock과 프로세스 간 lock 모두를 사용해 thread-safe하다.

        Args:
            thread_id: thread 전용 설정을 위한 선택적 thread ID.

        Returns:
            확보한 sandbox 환경의 ID.
        """
        effective_user_id = self._effective_acquire_user_id(user_id)
        if thread_id:
            thread_lock = self._get_thread_lock(thread_id, effective_user_id)
            with thread_lock:
                return self._acquire_internal(thread_id, user_id=effective_user_id)
        else:
            return self._acquire_internal(thread_id, user_id=effective_user_id)

    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """event loop를 블로킹하지 않고 sandbox 환경을 확보한다.

        ``acquire()``와 동작이 같되, 블로킹하는 backend 연산을 event loop 밖으로 빼고 새로
        만든 sandbox에는 async 네이티브 readiness polling을 쓴다.
        """
        effective_user_id = self._effective_acquire_user_id(user_id)
        if thread_id:
            thread_lock = self._get_thread_lock(thread_id, effective_user_id)
            await _acquire_thread_lock_async(thread_lock)
            try:
                return await self._acquire_internal_async(thread_id, user_id=effective_user_id)
            finally:
                thread_lock.release()

        return await self._acquire_internal_async(thread_id, user_id=effective_user_id)

    def _acquire_internal(self, thread_id: str | None, *, user_id: str) -> str:
        """2계층 일관성을 갖춘 내부 sandbox 확보 로직.

        Layer 1: 프로세스 내 캐시(가장 빠르며 같은 프로세스의 반복 접근을 담당)
        Layer 2: backend discovery(다른 프로세스가 시작한 컨테이너를 담당. sandbox_id가
                 thread_id에서 결정적으로 나오므로 공유 상태 파일이 필요 없다 — 어떤
                 프로세스든 같은 컨테이너 이름을 유도할 수 있다)
        """
        self._ensure_skills_projection(user_id)
        cached_id = self._reuse_in_process_sandbox(thread_id, user_id=user_id)
        if cached_id is not None:
            return cached_id

        # thread 전용이면 결정적 ID, 익명이면 랜덤 ID
        sandbox_id = self._sandbox_id_for_thread(thread_id, user_id)
        if thread_id:
            key = self._thread_key(thread_id, user_id)
            with self._lock:
                self._assert_active_identity_available_locked(sandbox_id, key)

        # ── Layer 1.5: warm pool(컨테이너가 아직 실행 중이라 cold-start 없음) ──
        reclaimed_id = self._reclaim_warm_pool_sandbox(thread_id, sandbox_id, user_id=user_id)
        if reclaimed_id is not None:
            return reclaimed_id

        # ── Layer 2: backend discovery + create(프로세스 간 lock으로 보호) ──
        # file lock을 써서 같은 thread_id의 sandbox를 만들려고 경쟁하는 두 프로세스를 여기서
        # 직렬화한다. 두 번째 프로세스는 이름 충돌을 만나는 대신 첫 번째가 시작한 컨테이너를
        # discover한다.
        if thread_id:
            return self._discover_or_create_with_lock(thread_id, sandbox_id, user_id=user_id)

        return self._create_sandbox(thread_id, sandbox_id, user_id=user_id)

    async def _acquire_internal_async(self, thread_id: str | None, *, user_id: str) -> str:
        """``_acquire_internal``의 async 버전."""
        await asyncio.to_thread(self._ensure_skills_projection, user_id)
        cached_id = await asyncio.to_thread(self._reuse_in_process_sandbox, thread_id, user_id=user_id)
        if cached_id is not None:
            return cached_id

        # thread 전용이면 결정적 ID, 익명이면 랜덤 ID
        sandbox_id = self._sandbox_id_for_thread(thread_id, user_id)
        if thread_id:
            key = self._thread_key(thread_id, user_id)
            with self._lock:
                self._assert_active_identity_available_locked(sandbox_id, key)

        # ── Layer 1.5: warm pool(컨테이너가 아직 실행 중이라 cold-start 없음) ──
        reclaimed_id = await asyncio.to_thread(self._reclaim_warm_pool_sandbox, thread_id, sandbox_id, user_id=user_id)
        if reclaimed_id is not None:
            return reclaimed_id

        # ── Layer 2: backend discovery + create(프로세스 간 lock으로 보호) ──
        if thread_id:
            return await self._discover_or_create_with_lock_async(thread_id, sandbox_id, user_id=user_id)

        return await self._create_sandbox_async(thread_id, sandbox_id, user_id=user_id)

    def _discover_or_create_with_lock(self, thread_id: str, sandbox_id: str, *, user_id: str | None = None) -> str:
        """프로세스 간 file lock 아래에서 기존 sandbox를 discover하거나 새로 만든다.

        file lock은 여러 프로세스에 걸쳐 같은 thread_id의 동시 sandbox 생성을 직렬화해서
        컨테이너 이름 충돌을 막는다.
        """
        paths = get_paths()
        effective_user_id = self._effective_acquire_user_id(user_id)
        paths.ensure_thread_dirs(thread_id, user_id=effective_user_id)
        lock_path = paths.thread_dir(thread_id, user_id=effective_user_id) / f"{sandbox_id}.lock"

        with open(lock_path, "a", encoding="utf-8") as lock_file:
            locked = False
            try:
                _lock_file_exclusive(lock_file)
                locked = True
                # 기다리는 동안 이 프로세스의 다른 thread가 경쟁에서 이겼을 수 있으므로
                # file lock 아래에서 프로세스 내 캐시를 다시 확인한다.
                cached_id = self._recheck_cached_sandbox(thread_id, sandbox_id, user_id=effective_user_id)
                if cached_id is not None:
                    return cached_id

                # backend discovery: 다른 프로세스가 컨테이너를 만들었을 수 있다.
                discovered = self._backend.discover(sandbox_id)
                if discovered is not None:
                    return self._register_discovered_sandbox(thread_id, discovered, user_id=effective_user_id)

                return self._create_sandbox(thread_id, sandbox_id, user_id=effective_user_id)
            finally:
                if locked:
                    _unlock_file(lock_file)

    async def _discover_or_create_with_lock_async(self, thread_id: str, sandbox_id: str, *, user_id: str | None = None) -> str:
        """``_discover_or_create_with_lock``의 async 버전."""
        paths = get_paths()
        effective_user_id = self._effective_acquire_user_id(user_id)
        await asyncio.to_thread(paths.ensure_thread_dirs, thread_id, user_id=effective_user_id)
        lock_path = paths.thread_dir(thread_id, user_id=effective_user_id) / f"{sandbox_id}.lock"

        lock_file = await asyncio.to_thread(_open_lock_file, lock_path)
        locked = False
        try:
            await asyncio.to_thread(_lock_file_exclusive, lock_file)
            locked = True
            # 기다리는 동안 이 프로세스의 다른 thread가 경쟁에서 이겼을 수 있으므로
            # file lock 아래에서 프로세스 내 캐시를 다시 확인한다.
            cached_id = await asyncio.to_thread(self._recheck_cached_sandbox, thread_id, sandbox_id, user_id=effective_user_id)
            if cached_id is not None:
                return cached_id

            # 로컬 discovery는 Docker를 조회하고 health check를 수행할 수 있어서 backend
            # discovery는 sync다. event loop 밖에서 실행한다.
            discovered = await asyncio.to_thread(self._backend.discover, sandbox_id)
            if discovered is not None:
                # 등록 과정에서 ownership을 게시하는데 이는 블로킹 store IO다(backend에 따라
                # 파일시스템 또는 네트워크) — 이 coroutine의 다른 모든 단계를 offload하는 것과
                # 같은 이유다.
                return await asyncio.to_thread(self._register_discovered_sandbox, thread_id, discovered, user_id=effective_user_id)

            return await self._create_sandbox_async(thread_id, sandbox_id, user_id=effective_user_id)
        finally:
            if locked:
                await asyncio.to_thread(_unlock_file, lock_file)
            await asyncio.to_thread(lock_file.close)

    def _destroy_unready_sandbox(self, sandbox_id: str, info: SandboxInfo) -> None:
        """readiness check에 실패한 갓 만든 컨테이너를 정리한다.

        backend가 컨테이너를 시작했지만 ready에 도달하지 못했으므로
        ``_register_created_sandbox``에 들어간 적이 없고 ownership store에도 lease가 없다.
        readiness timeout 전체(60초) 동안 주인 없이 실행되는데, 그 구간이 바로 peer gateway의
        기동 reconciliation이 adopt하려고 만들어진 구간이다(#4206). claim이 없으면, 아직 ready가
        아닌 Pod을 adopt한 peer 위로 이 인스턴스의 stop이 떨어져 활성 turn을 끊는 인스턴스 간
        kill이 된다(#4248).

        teardown lease를 먼저 claim해서 이 reap 경로도 다른 모든 destroy
        (``_destroy_warm_entry``, ``_drop_unhealthy_reserved``)와 같은 ownership guard를 거치게
        한다. peer가 이미 소유했거나 ownership store가 답하지 못하면 fail closed 한다(컨테이너는
        peer가 자기 reconciliation으로 회수하도록 남겨둔다).

        claim만으로는 인스턴스 **간** 절반만 해결된다. 설계상 우리 자신의 lease에 대해서는
        성공하므로 이 프로세스에 대해서는 아무 말도 해주지 않는다. 같은 프로세스 쪽 절반이
        local teardown 예약이며, 가장 먼저 잡아 경로 전체에 걸쳐 유지한다 — readiness timeout과
        claim 사이에 ``_reconcile_orphans``(idle checker, 60초마다)가 이 컨테이너를 실행 중이고
        tracking되지 않으며 recovery grace를 지난 상태로 보고 ``_warm_pool``에 넣을 수 있다.
        그러면 이어지는 claim은 여전히 성공하고 stop은 이 인스턴스가 방금 adopt한 항목 위로
        떨어져, 다음 reclaim이 넘겨줄 죽은 warm 항목만 남는다. predicate는 그 id가 active와 warm
        map 어디에도 없는지 확인하며, 예약이 그 확인과 teardown mark를 하나의 임계 구역으로
        묶어서 그 사이로 adopt/acquire가 끼어들 수 없게 한다(``_destroy_warm_entry``와 같은
        짝지음).
        """
        if not self._reserve_local_teardown(
            sandbox_id,
            lambda: sandbox_id not in self._sandboxes and sandbox_id not in self._sandbox_infos and sandbox_id not in self._warm_pool,
        ):
            logger.warning(
                "Not destroying unready sandbox %s: adopted or being torn down by this instance",
                sandbox_id,
            )
            return
        try:
            if not self._claim_ownership(sandbox_id, for_destroy=True):
                logger.warning(
                    "Not destroying unready sandbox %s: owned by another instance or ownership unavailable",
                    sandbox_id,
                )
                return
            try:
                with self._held_teardown_lease(sandbox_id):
                    self._backend.destroy(info)
            except Exception as e:
                logger.warning(f"Error destroying unready sandbox {sandbox_id}: {e}")
        finally:
            self._finish_local_teardown(sandbox_id)

    def _create_sandbox(self, thread_id: str | None, sandbox_id: str, *, user_id: str | None = None) -> str:
        """backend를 통해 새 sandbox를 만든다.

        Args:
            thread_id: 선택적 thread ID.
            sandbox_id: 사용할 sandbox ID.

        Returns:
            sandbox_id.

        Raises:
            RuntimeError: sandbox 생성이나 readiness check가 실패한 경우.
        """
        effective_user_id = self._effective_acquire_user_id(user_id)
        extra_mounts = self._get_extra_mounts(thread_id, user_id=effective_user_id)
        provision_lark_cli_runtime = self._lark_integration_active(effective_user_id)
        provision_lark_cli_broker = self._lark_broker_active(effective_user_id)

        # replicas를 강제한다. eviction 예산에 포함되는 것은 warm-pool 컨테이너뿐이다.
        # active sandbox는 살아 있는 thread가 쓰고 있으므로 강제로 멈추면 안 된다.
        replicas, total = self._replica_count()
        if total >= replicas:
            evicted = self._evict_oldest_warm()
            self._log_replicas_soft_cap(replicas, sandbox_id, evicted)

        info = self._backend.create(
            thread_id,
            sandbox_id,
            extra_mounts=extra_mounts or None,
            user_id=effective_user_id,
            provision_lark_cli_runtime=provision_lark_cli_runtime,
            provision_lark_cli_broker=provision_lark_cli_broker,
        )

        # sandbox가 ready가 될 때까지 기다린다
        if not wait_for_sandbox_ready(info.sandbox_url, timeout=60):
            # 컨테이너는 실행 중이지만 주인이 없다. ownership은 이 게이트 이후
            # ``_register_created_sandbox``가 게시한다. 그 사이에 peer가 아직 ready가 아닌
            # Pod을 adopt하지 못하도록, 멈추기 전에 teardown lease를 claim한다(#4248).
            self._destroy_unready_sandbox(sandbox_id, info)
            raise RuntimeError(f"Sandbox {sandbox_id} failed to become ready within timeout at {info.sandbox_url}")

        return self._register_created_sandbox(thread_id, sandbox_id, info, user_id=effective_user_id)

    async def _create_sandbox_async(self, thread_id: str | None, sandbox_id: str, *, user_id: str | None = None) -> str:
        """``_create_sandbox``의 async 버전."""
        effective_user_id = self._effective_acquire_user_id(user_id)
        extra_mounts = await asyncio.to_thread(self._get_extra_mounts, thread_id, user_id=effective_user_id)
        provision_lark_cli_runtime = await asyncio.to_thread(self._lark_integration_active, effective_user_id)
        provision_lark_cli_broker = await asyncio.to_thread(self._lark_broker_active, effective_user_id)

        # replicas를 강제한다. eviction 예산에 포함되는 것은 warm-pool 컨테이너뿐이다.
        # active sandbox는 살아 있는 thread가 쓰고 있으므로 강제로 멈추면 안 된다.
        replicas, total = self._replica_count()
        if total >= replicas:
            evicted = await asyncio.to_thread(self._evict_oldest_warm)
            self._log_replicas_soft_cap(replicas, sandbox_id, evicted)

        info = await asyncio.to_thread(
            self._backend.create,
            thread_id,
            sandbox_id,
            extra_mounts=extra_mounts or None,
            user_id=effective_user_id,
            provision_lark_cli_runtime=provision_lark_cli_runtime,
            provision_lark_cli_broker=provision_lark_cli_broker,
        )

        # event loop를 블로킹하지 않고 sandbox가 ready가 될 때까지 기다린다.
        if not await wait_for_sandbox_ready_async(info.sandbox_url, timeout=60):
            # 컨테이너는 실행 중이지만 주인이 없다. ownership은 이 게이트 이후
            # ``_register_created_sandbox``가 게시한다. 그 사이에 peer가 아직 ready가 아닌
            # Pod을 adopt하지 못하도록, 멈추기 전에 teardown lease를 claim한다(#4248).
            await asyncio.to_thread(self._destroy_unready_sandbox, sandbox_id, info)
            raise RuntimeError(f"Sandbox {sandbox_id} failed to become ready within timeout at {info.sandbox_url}")

        # 등록 과정에서 ownership을 게시하므로(블로킹 store IO) 이 경로의 다른 모든 블로킹
        # 단계와 마찬가지로 offload한다.
        return await asyncio.to_thread(self._register_created_sandbox, thread_id, sandbox_id, info, user_id=effective_user_id)

    def get(self, sandbox_id: str) -> Sandbox | None:
        """ID로 sandbox를 가져온다. 마지막 활동 timestamp를 갱신한다.

        순수한 in-memory 조회로 유지한다. async tool 경로가 event loop 위에서 이것을 직접
        호출하므로(``ensure_sandbox_initialized_async``) ownership store를 건드리면 안 된다 —
        backend에 따라 블로킹 파일시스템 IO 또는 네트워크 IO이기 때문이다. ownership은
        acquire/reclaim 시 event loop 밖에서 게시되고 renewal thread가 갱신한다
        (``_renew_owned_leases`` 참고).

        Args:
            sandbox_id: sandbox의 ID.

        Returns:
            찾으면 sandbox 인스턴스, 없으면 None.
        """
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
            if sandbox is not None:
                self._last_activity[sandbox_id] = time.time()
        return sandbox

    def release(self, sandbox_id: str) -> None:
        """활성 사용 상태의 sandbox를 warm pool로 release한다.

        같은 thread가 다음 turn에 cold-start 없이 빠르게 회수할 수 있도록 컨테이너는 계속
        실행 상태로 둔다. 컨테이너는 replicas 한도 때문에 eviction이 강제되거나 shutdown일
        때만 멈춘다.

        캐시된 ``AioSandbox`` 인스턴스가 소유한 host 쪽 HTTP client는 인스턴스를 버리기 전에
        닫는다(#2872). warm-pool 항목은 ``SandboxInfo``만 저장하므로, 나중에 컨테이너를
        회수하면 새 ``AioSandbox``(와 새 client)를 만든다.

        Args:
            sandbox_id: release할 sandbox의 ID.
        """
        info = None
        sandbox = None
        thread_keys_to_remove: list[tuple[str, str]] = []

        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            thread_keys_to_remove = [key for key, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for key in thread_keys_to_remove:
                del self._thread_sandboxes[key]
            active_identity = self._active_sandbox_identity.pop(sandbox_id, None)
            self._last_activity.pop(sandbox_id, None)
            # warm pool에 넣어둔다 — 컨테이너는 계속 실행된다
            if info and sandbox_id not in self._warm_pool:
                self._warm_pool[sandbox_id] = (info, time.time())
                self._warm_pool_identity[sandbox_id] = thread_keys_to_remove[0] if thread_keys_to_remove else active_identity

        if sandbox is not None:
            # 방어의 한 겹이다. close()는 이미 자기 에러를 삼키므로, 이 guard는 앞으로
            # close()가 잘못 동작할 경우에 대비할 뿐이다. host 쪽 client 정리가 warm pool
            # 적재를 막는 일은 절대 없어야 한다.
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing sandbox {sandbox_id} during release: {e}")

        # 우리가 회수하기 전에 peer가 adopt하고 파괴하지 못하도록 warm 상태에서도 lease를
        # 유지하고, 긴 turn 동안 만료됐다면 다시 세운다. 절대 예외를 던지지 않는다. turn은 이미
        # 끝났으므로 store 문제가 after_agent를 통해 드러나면 안 되고, 실제 보장은 (warm 항목도
        # 담당하는) renewal thread다 — 여기서는 구간을 좁힐 뿐이다.
        if info is not None:
            # renewal thread와 같은 낡음 문제가 있다. refresh는 store round trip이고, 그
            # 사이에 thread의 다음 turn이 이 warm 항목을 회수할 수 있다. 그 사이 아무도 다시
            # acquire하지 않은 경우에만 버린다.
            epoch = self._acquire_epoch_of(sandbox_id)
            if not self._refresh_ownership(sandbox_id):
                logger.warning("Sandbox %s is owned by another instance; releasing it from this warm pool", sandbox_id)
                self._forget_lost_sandbox(sandbox_id, expected_epoch=epoch)

        logger.info(f"Released sandbox {sandbox_id} to warm pool (container still running)")

    def destroy(self, sandbox_id: str) -> None:
        """sandbox를 파괴한다. 컨테이너를 멈추고 모든 리소스를 해제한다.

        release()와 달리 실제로 컨테이너를 멈춘다. 명시적 정리, 용량에 따른 eviction,
        shutdown에 사용한다.

        캐시된 ``AioSandbox`` 인스턴스가 소유한 host 쪽 HTTP client도 backend/컨테이너 파괴와
        함께 닫아서 client/socket 리소스가 누수되지 않게 한다(#2872).

        Args:
            sandbox_id: 파괴할 sandbox의 ID.
        """
        self._destroy_tracked(sandbox_id, still_reapable=lambda: True)

    def _destroy_tracked(self, sandbox_id: str, *, still_reapable: Callable[[], bool]) -> None:
        """호출자가 준 "아직 회수해도 되는가" 게이트를 붙인 ``destroy()``.

        *더 앞서* 파괴를 결정한 호출자(idle checker)는 자기 predicate를 넘겨서, teardown을
        예약하는 바로 그 임계 구역에서 판단이 재검증되게 한다. ``destroy()`` 자신은 상수를
        넘긴다. 명시적 destroy는 지금 내린 판단이기 때문이다.
        """
        if not self._reserve_local_teardown(sandbox_id, still_reapable):
            logger.info("Skipping destroy of sandbox %s: re-acquired by this instance or already being torn down", sandbox_id)
            return

        try:
            self._destroy_reserved(sandbox_id)
        finally:
            self._finish_local_teardown(sandbox_id)

    def _destroy_reserved(self, sandbox_id: str) -> None:
        # untrack보다 claim을 먼저 한다. 순서가 반대면 claim이 거부됐을 때 컨테이너를 잃는다.
        # 여전히 실행 중인데 우리 map 어디에도 없으므로, 여기서는 아무도 회수하거나 reclaim하지
        # 못한다.
        if not self._claim_ownership(sandbox_id, for_destroy=True):
            logger.warning("Refusing to destroy sandbox %s: owned by another instance", sandbox_id)
            return

        sandbox, info, _ = self._remove_tracked_sandbox(sandbox_id)

        if sandbox is not None:
            # 방어의 한 겹이다. close()는 이미 자기 에러를 삼키므로, 이 guard는 앞으로
            # close()가 잘못 동작할 경우에 대비할 뿐이다. host 쪽 client 정리가 컨테이너
            # 파괴를 막는 일은 절대 없어야 한다.
            try:
                sandbox.close()
            except Exception as e:
                logger.warning(f"Error closing sandbox {sandbox_id} during destroy: {e}")

        if info:
            # marker는 자신이 쓰인 TTL이 아니라 stop보다 오래 살아야 하며, 두 결과 모두에서
            # heartbeat가 종료 시 해제한다. stop이 실패하면 컨테이너는 아마 아직 살아 있으므로,
            # 남은 marker가 TTL이 만료될 때까지 자기 thread의 `take()`를 거부하게 된다. 에러는
            # 여전히 `with` 밖으로 전파되며(`shutdown()`이 sandbox별로 로깅한다), 단지 해제가
            # 더 이상 이 메서드의 일이 아닐 뿐이다.
            with self._held_teardown_lease(sandbox_id):
                self._backend.destroy(info)
            logger.info(f"Destroyed sandbox {sandbox_id}")
        else:
            # 멈출 컨테이너가 없어 teardown lease도 잡지 않았다. 위의 claim이 쓴 marker를
            # 지워서, tracking되지 않는 id 때문에 lease가 `del:` 상태로 남지 않게 한다.
            self._release_ownership(sandbox_id)

    def shutdown(self) -> None:
        """모든 sandbox를 종료한다. thread-safe하고 멱등하다."""
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
            sandbox_ids = list(self._sandboxes.keys())
            warm_items = list(self._warm_pool.items())
            self._warm_pool.clear()
            self._warm_pool_identity.clear()

        self._stop_idle_checker()
        # 파괴 전에 renewal을 멈춘다. destroy 경로가 스스로 ownership을 claim하며, 그와
        # 경쟁하는 renewal은 곧 버릴 lease를 다시 게시할 뿐이다.
        self._stop_lease_renewal()

        logger.info(f"Shutting down {len(sandbox_ids)} active + {len(warm_items)} warm-pool sandbox(es)")

        for sandbox_id in sandbox_ids:
            try:
                self.destroy(sandbox_id)
            except Exception as e:
                logger.error(f"Failed to destroy sandbox {sandbox_id} during shutdown: {e}")

        for sandbox_id, (info, _) in warm_items:
            # idle 경로와 마찬가지로 ownership claim과 컨테이너 stop이 함께 가도록
            # _destroy_warm_entry를 거친다. 여기서는 무조건 실행한다. 위의 lock 안에서 항목을
            # 이미 `_warm_pool`에서 제거했으므로, 다른 호출자가 쓰는 pool 소속 predicate는
            # 전부 거부할 것이기 때문이다.
            self._destroy_warm_entry(sandbox_id, info, reason="shutdown", still_reapable=lambda: True)

        try:
            self._ownership.close()
        except Exception as e:
            logger.warning(f"Error closing sandbox ownership store during shutdown: {e}")
