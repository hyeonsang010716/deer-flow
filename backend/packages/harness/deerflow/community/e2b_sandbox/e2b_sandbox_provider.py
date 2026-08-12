"""``E2BSandboxProvider`` — e2b 클라우드용 DeerFlow :class:`SandboxProvider`.

설정은 :class:`SandboxConfig`에서 읽는다. 알 수 없는 provider 필드는 startup 시 보고한다.

.. code-block:: yaml

    sandbox:
      use: deerflow.community.e2b_sandbox:E2BSandboxProvider
      api_key: $E2B_API_KEY            # 필수(또는 E2B_API_KEY 환경 변수)
      template: code-interpreter-v1     # 기본값: e2b code-interpreter 템플릿
      domain: e2b.dev                  # 선택. self-host e2b용
      idle_timeout: 1800               # ``set_timeout``으로 전달
      replicas: 3                      # 최대 capacity(active + warm)
      overflow_policy: wait            # wait | reject | burst (기본값: wait)
      acquire_timeout: 30              # ``wait`` 정책의 대기 시간(초, 기본값: 30)
      burst_limit: 2                   # ``burst`` 정책의 추가 slot 수(기본값: 0)
      ownership:
        type: redis                    # Gateway 간 ownership과 capacity를 공유한다
        redis_url: redis://redis:6379/0
      mounts:                          # sandbox 시작 시 1회 업로드
        - host_path: /data/skills
          container_path: /home/user/skills
          read_only: true
      environment:                     # 생성 시 e2b ``envs``로 전달
        OPENAI_API_KEY: $OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import shlex
import signal
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import Any

from e2b import SandboxQuery
from e2b_code_interpreter import Sandbox as E2BClientSandbox

from deerflow.config import get_app_config
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.sandbox.exceptions import SandboxCapacityExceededError
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider

from ..aio_sandbox.ownership import (
    OwnershipBackendError,
    RenewOutcome,
    SandboxOwnershipStore,
    compute_lease_ttl,
    generate_owner_id,
    make_sandbox_ownership_store,
    resolve_ownership_config,
)
from .capacity import (
    CapacityBackendError,
    ReserveStatus,
    make_e2b_capacity_store,
)
from .e2b_sandbox import DEFAULT_E2B_HOME_DIR, E2BSandbox, _is_sandbox_gone_error

logger = logging.getLogger(__name__)


# ── 기본값 ───────────────────────────────────────────────────────────────
DEFAULT_TEMPLATE = "code-interpreter-v1"  # 공개된 e2b code-interpreter 템플릿
DEFAULT_IDLE_TIMEOUT = 1800  # 30분. ``Sandbox.set_timeout``으로 전달한다.
DEFAULT_REPLICAS = 3
DEFAULT_OVERFLOW_POLICY = "wait"  # wait | reject | burst
DEFAULT_ACQUIRE_TIMEOUT = 30  # wait 정책의 대기 시간(초)
DEFAULT_RECONCILIATION_INTERVAL_SECONDS = 60.0
DEFAULT_RECONCILIATION_GRACE_SECONDS = 120.0
DEFAULT_RECONCILIATION_ORPHAN_TTL_SECONDS = 3600.0
DEFAULT_RECONCILIATION_MAX_PAGES = 10
DEFAULT_RECONCILIATION_MAX_ITEMS = 200
DEFAULT_RECONCILIATION_MAX_SECONDS = 15.0
# E2B SDK의 60초 create 요청 타임아웃의 두 배. ownership lease를 짧게 잡더라도
# 진행 중인 create가 버려진 것처럼 보여서는 안 된다.
MIN_CAPACITY_RESERVATION_SECONDS = 120.0
# ``set_timeout``의 상한(e2b 무료 플랜은 현재 24시간까지만 허용하며,
# 과도한 값은 control plane이 거부한다).
MAX_E2B_TIMEOUT = 24 * 60 * 60

# 어느 gateway 프로세스에서든 ``Sandbox.list(query={...})``로 우리 sandbox를
# 찾을 수 있도록 모든 sandbox에 붙이는 metadata 키.
META_KEY_USER = "deer_flow_user"
META_KEY_THREAD = "deer_flow_thread"
META_KEY_PROVIDER = "deer_flow_provider"
META_KEY_GATEWAY = "deer_flow_gateway"
META_KEY_CREATED_AT = "deer_flow_created_at"
META_KEY_CAPACITY_LEDGER = "deer_flow_capacity_ledger"
META_KEY_CAPACITY_RESERVATION = "deer_flow_capacity_reservation"
META_VAL_PROVIDER = "e2b_sandbox_provider"
E2B_EXTRA_CONFIG_KEYS = frozenset({"api_key", "domain", "home_dir", "template"})


@dataclass
class ReconciliationStats:
    """제한된 범위의 reconciliation 결과. metric/logging에도 그대로 쓴다."""

    discovered: int = 0
    adopted: int = 0
    duplicates: int = 0
    deferred: int = 0
    killed: int = 0
    dead: int = 0
    budget_exhausted: bool = False


class E2BSandboxProvider(SandboxProvider):
    """e2b code-interpreter 클라우드 SDK 기반 sandbox provider."""

    # e2b sandbox는 원격이라 gateway와 공유하는 호스트 파일시스템이 없다.
    # 따라서 프레임워크가 업로드 파일을 명시적으로 동기화해야 한다
    # (AioSandboxProvider의 remote backend도 같은 플래그를 쓴다).
    uses_thread_data_mounts = False
    needs_upload_permission_adjustment = True

    # ── 생성 & 설정 ──────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 활성 sandbox. DeerFlow 쪽 sandbox id(= e2b id)를 키로 쓴다.
        self._sandboxes: dict[str, E2BSandbox] = {}
        # 프로세스 내 빠른 조회를 위한 (user_id, thread_id) -> sandbox id 매핑.
        self._thread_sandboxes: dict[tuple[str, str], str] = {}
        # (user, thread)별 lock. 원격 IO 동안 provider 전역 lock을 잡지 않고
        # acquire()와 release()의 상태 전이를 직렬화한다.
        self._thread_locks: dict[tuple[str, str], threading.Lock] = {}
        # Warm pool: release되었지만 원격 micro-VM이 아직 살아 있는 sandbox.
        # ``OrderedDict``가 삽입/move_to_end 순서로 LRU를 유지한다.
        self._warm_pool: OrderedDict[str, tuple[str, float]] = OrderedDict()
        # 원격 상태를 알 수 없는 eviction. 이후 eviction 시도가 파기를 확인할 때까지
        # 각 id는 transition slot을 계속 점유한다.
        self._eviction_tombstones: set[str] = set()
        # 현재 재연결 또는 종료 중인 ID. tombstone마다 재시도 소유자를 하나만 두어
        # 두 번째 호출이 slot을 해제해 버리는 것을 막는다.
        self._evictions_in_progress: set[str] = set()
        # 추적 중인 lifecycle 상태 사이에 있는 원격 ID. shutdown은 원격 호출 도중에도
        # 정리 대상을 놓치지 않기 위해 이 집합을 쓴다.
        self._remote_ops_in_progress: set[str] = set()
        # discovery는 다른 Gateway가 아직 쓰는 VM을 찾을 수 있다.
        # shutdown은 이 client들을 close만 하고 VM은 파기하지 않는다.
        self._unowned_remote_ops_in_progress: set[str] = set()
        # capacity는 예약했지만 아직 ``_sandboxes``에 커밋되지 않은 진행 중 create.
        # ``_lock``으로 보호한다.
        self._reserved_slots = 0
        self._transitioning_slots = 0
        self._capacity_cond = threading.Condition(self._lock)
        self._shutdown_called = False
        self._owned_sandbox_ids: set[str] = set()
        self._acquire_inflight: set[str] = set()
        self._orphan_first_seen: dict[str, float] = {}
        self._maintenance_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._reconcile_thread: threading.Thread | None = None
        self._owner_id = generate_owner_id()

        self._config = self._load_config()
        acquire_workers = max(4, min(32, self._capacity_limit() + 1))
        self._acquire_executor = ThreadPoolExecutor(
            max_workers=acquire_workers,
            thread_name_prefix="e2b-sandbox-acquire",
        )
        self._ownership_config = resolve_ownership_config(
            self._config.get("ownership"),
            stream_bridge=self._config.get("stream_bridge"),
        )
        self._ownership: SandboxOwnershipStore = make_sandbox_ownership_store(
            self._ownership_config,
            owner_id=self._owner_id,
        )
        self._deployment_capacity = make_e2b_capacity_store(
            self._ownership_config,
            hard_limit=self._capacity_limit(),
        )
        if not self._ownership.supports_cross_process:
            logger.warning("E2B sandbox ownership is process-local. Multi-worker gateways must configure sandbox.ownership.type: redis for safe reconciliation.")

        atexit.register(self.shutdown)
        self._register_signal_handlers()
        self._start_maintenance_threads()

    def _load_config(self) -> dict[str, Any]:
        """``SandboxConfig``(``extra="allow"``)에서 e2b 옵션을 읽는다."""
        sandbox_config = get_app_config().sandbox
        unknown_keys = sorted(set(getattr(sandbox_config, "model_extra", None) or {}) - E2B_EXTRA_CONFIG_KEYS)
        if unknown_keys:
            logger.warning(
                "E2BSandboxProvider: unknown sandbox config fields: %s",
                ", ".join(unknown_keys),
            )

        def _opt(name: str, default: Any = None) -> Any:
            return getattr(sandbox_config, name, default)

        api_key = _opt("api_key") or os.environ.get("E2B_API_KEY")
        if not api_key:
            logger.warning("E2BSandboxProvider: no api_key configured (set sandbox.api_key in config.yaml or the E2B_API_KEY environment variable). The SDK will fail on the first acquire() until this is provided.")

        idle_timeout = _opt("idle_timeout")
        if idle_timeout is None:
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        idle_timeout = max(0, min(int(idle_timeout), MAX_E2B_TIMEOUT))

        replicas = _opt("replicas")
        replicas = DEFAULT_REPLICAS if replicas is None else int(replicas)

        overflow_policy = _opt("overflow_policy") or DEFAULT_OVERFLOW_POLICY
        if overflow_policy not in ("wait", "reject", "burst"):
            logger.warning("E2BSandboxProvider: invalid overflow_policy %r; falling back to %r", overflow_policy, DEFAULT_OVERFLOW_POLICY)
            overflow_policy = DEFAULT_OVERFLOW_POLICY

        acquire_timeout = _opt("acquire_timeout")
        if acquire_timeout is None:
            acquire_timeout = DEFAULT_ACQUIRE_TIMEOUT
        else:
            acquire_timeout = max(1, int(acquire_timeout))

        burst_limit_raw = _opt("burst_limit")
        burst_limit = max(0, int(burst_limit_raw)) if burst_limit_raw is not None else 0
        if overflow_policy == "burst" and burst_limit == 0:
            logger.warning("E2BSandboxProvider: overflow_policy is 'burst' but burst_limit is 0; falling back to 'reject'")
            overflow_policy = "reject"

        return {
            "api_key": api_key,
            "template": _opt("template") or _opt("image") or DEFAULT_TEMPLATE,
            "domain": _opt("domain"),
            "home_dir": _opt("home_dir") or DEFAULT_E2B_HOME_DIR,
            "idle_timeout": idle_timeout,
            "replicas": replicas,
            "overflow_policy": overflow_policy,
            "acquire_timeout": acquire_timeout,
            "burst_limit": burst_limit,
            "mounts": _opt("mounts") or [],
            "environment": self._resolve_env_vars(_opt("environment") or {}),
            "ownership": _opt("ownership"),
            "stream_bridge": getattr(get_app_config(), "stream_bridge", None),
            "reconciliation_interval_seconds": max(
                1.0,
                float(_opt("reconciliation_interval_seconds", DEFAULT_RECONCILIATION_INTERVAL_SECONDS)),
            ),
            "reconciliation_grace_seconds": max(
                0.0,
                float(_opt("reconciliation_grace_seconds", DEFAULT_RECONCILIATION_GRACE_SECONDS)),
            ),
            "reconciliation_orphan_ttl_seconds": max(
                0.0,
                float(_opt("reconciliation_orphan_ttl_seconds", DEFAULT_RECONCILIATION_ORPHAN_TTL_SECONDS)),
            ),
            "reconciliation_max_pages": max(
                1,
                int(_opt("reconciliation_max_pages", DEFAULT_RECONCILIATION_MAX_PAGES)),
            ),
            "reconciliation_max_items": max(
                1,
                int(_opt("reconciliation_max_items", DEFAULT_RECONCILIATION_MAX_ITEMS)),
            ),
            "reconciliation_max_seconds": max(
                0.1,
                float(_opt("reconciliation_max_seconds", DEFAULT_RECONCILIATION_MAX_SECONDS)),
            ),
        }

    @staticmethod
    def _resolve_env_vars(env_config: dict[str, str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for key, value in env_config.items():
            if isinstance(value, str) and value.startswith("$"):
                resolved[key] = os.environ.get(value[1:], "")
            else:
                resolved[key] = "" if value is None else str(value)
        return resolved

    def _get_sandbox_cls(self) -> type[E2BClientSandbox]:
        """e2b SDK의 Sandbox 클래스를 반환한다."""
        return E2BClientSandbox

    # ── 식별자 헬퍼 ──────────────────────────────────────────────────────

    @staticmethod
    def _effective_acquire_user_id(user_id: str | None) -> str:
        return user_id or get_effective_user_id()

    @staticmethod
    def _thread_key(thread_id: str, user_id: str) -> tuple[str, str]:
        return (user_id, thread_id)

    @staticmethod
    def _stable_seed(thread_id: str, user_id: str) -> str:
        return hashlib.sha256(f"{user_id}:{thread_id}".encode()).hexdigest()[:16]

    def _metadata_matches_capacity_ledger(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        """이 ledger와, 태그 도입 이전의 legacy sandbox를 함께 포함시킨다."""
        store = self._deployment_capacity
        if store is None:
            return True
        remote_ledger = metadata.get(META_KEY_CAPACITY_LEDGER)
        return remote_ledger in (None, "", store.key)

    @staticmethod
    def _capacity_reservation_from_metadata(
        metadata: dict[str, Any],
    ) -> str | None:
        token = metadata.get(META_KEY_CAPACITY_RESERVATION)
        return token if isinstance(token, str) and token else None

    # ── signal / shutdown 처리 ───────────────────────────────────────────

    def _register_signal_handlers(self) -> None:
        try:
            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            self._original_sigint = signal.getsignal(signal.SIGINT)
            self._original_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None
        except (ValueError, OSError):
            return

        def _handler(signum, frame):
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

        for sig_name in ("SIGTERM", "SIGINT", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                logger.debug(
                    "Could not register %s handler (likely not running on main thread)",
                    sig_name,
                )

    def _get_thread_lock(self, thread_id: str, user_id: str) -> threading.Lock:
        key = self._thread_key(thread_id, user_id)
        with self._lock:
            lock = self._thread_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._thread_locks[key] = lock
            return lock

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        effective_user_id = self._effective_acquire_user_id(user_id)
        if thread_id:
            with self._get_thread_lock(thread_id, effective_user_id):
                return self._acquire_internal(thread_id, user_id=effective_user_id)
        return self._acquire_internal(thread_id, user_id=effective_user_id)

    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        effective_user_id = self._effective_acquire_user_id(user_id)
        loop = asyncio.get_running_loop()
        acquire = partial(self.acquire, thread_id, user_id=effective_user_id)
        return await loop.run_in_executor(self._acquire_executor, acquire)

    def _acquire_internal(self, thread_id: str | None, *, user_id: str) -> str:
        if thread_id:
            cached = self._reuse_in_process_sandbox(thread_id, user_id=user_id)
            if cached is not None:
                return cached

        if thread_id:
            reclaimed = self._reclaim_warm_pool_sandbox(thread_id, user_id=user_id)
            if reclaimed is not None:
                return reclaimed

        if thread_id:
            discovered = self._discover_remote_sandbox(thread_id, user_id=user_id)
            if discovered is not None:
                return discovered
        return self._create_sandbox(thread_id, user_id=user_id)

    def _reuse_in_process_sandbox(self, thread_id: str, *, user_id: str) -> str | None:
        key = self._thread_key(thread_id, user_id)
        with self._lock:
            sid = self._thread_sandboxes.get(key)
            if sid is None:
                return None
            sandbox = self._sandboxes.get(sid)
            if sandbox is None:
                # 매핑이 이미 사라진 항목을 가리키고 있으니 정리한다.
                self._thread_sandboxes.pop(key, None)
                return None

        # e2b VM이 회수되었다면(control plane의 idle timeout, 수동 pause 등)
        # 캐시 항목을 버린다. 이 사실은 이전 tool 호출에서 ``execute_command``가
        # ``is_dead``를 뒤집었거나, 아래의 명시적 ping으로 알게 된다. 이 검사가
        # 없으면 다음 acquire가 sandbox를 다시 만들 때까지 에이전트가
        # "sandbox not found" 에러로 무한히 맴돈다.
        if sandbox.is_dead or not sandbox.ping():
            logger.warning(
                "In-process e2b sandbox %s is dead (reaped by e2b control plane); evicting cache so acquire() can rebuild a fresh sandbox",
                sid,
            )
            with self._lock:
                self._sandboxes.pop(sid, None)
                self._thread_sandboxes.pop(key, None)
            try:
                sandbox.close()
            except Exception:
                pass
            self._release_deployment_sandbox(sid)
            return None

        try:
            self._refresh_remote_timeout(sandbox.client)
        except Exception as e:  # pragma: no cover - 방어적 처리
            logger.debug("Failed to refresh timeout on reuse: %s", e)
        self._publish_ownership(sid)
        with self._lock:
            self._acquire_inflight.discard(sid)

        logger.info(
            "Reusing in-process e2b sandbox %s for user/thread %s/%s",
            sid,
            user_id,
            thread_id,
        )
        return sid

    def _reclaim_warm_pool_sandbox(self, thread_id: str, *, user_id: str) -> str | None:
        """warm pool의 sandbox를 회수한다. 그동안 transitioning slot을 계속 점유한다.

        warm pool 항목을 꺼내는 즉시 transitioning slot을 잡는다. sandbox가
        ``_sandboxes``에 등록되면 slot은 active로 확정되고, 회수에 실패하면 해제된다.
        전이 도중 provider가 shutdown되면 VM을 kill한다.
        """
        seed = self._stable_seed(thread_id, user_id)
        with self._lock:
            target_id = next(
                (sid for sid, (s, _) in self._warm_pool.items() if s == seed),
                None,
            )
            if target_id is None:
                return None
            self._warm_pool.pop(target_id)
            self._begin_transition_locked()
            self._remote_ops_in_progress.add(target_id)

        try:
            client = self._reconnect_live_client(self._get_sandbox_cls(), target_id)
        except Exception as e:
            logger.warning(
                "Warm-pool e2b sandbox %s failed to reconnect, dropping: %s",
                target_id,
                e,
            )
            self._complete_transition_remote_op(target_id, remote_destroyed=False)
            return None

        if client is None:
            logger.warning(
                "Warm-pool e2b sandbox %s is no longer alive (reaped by control plane); dropping and falling back to create",
                target_id,
            )
            self._complete_transition_remote_op(target_id, remote_destroyed=True)
            return None

        try:
            self._publish_ownership(target_id)
        except Exception:
            self._complete_transition_remote_op(target_id, remote_destroyed=False)
            self._safe_close_client(client)
            raise

        self._refresh_remote_timeout(client)
        bootstrap_error, remote_destroyed = self._bootstrap_or_discard(client, target_id)
        if bootstrap_error is not None:
            self._complete_transition_remote_op(target_id, remote_destroyed=remote_destroyed)
            return None

        discard_after_shutdown = False
        with self._lock:
            if self._shutdown_called:
                logger.info(
                    "Provider shut down during reclaim of sandbox %s; killing VM",
                    target_id,
                )
                discard_after_shutdown = True
            else:
                self._remote_ops_in_progress.discard(target_id)
                self._register_connected_sandbox(target_id, client, thread_id=thread_id, user_id=user_id)
                self._end_transition_locked()

        if discard_after_shutdown:
            if self._claim_ownership(target_id, for_destroy=True):
                self._kill_client(client)
                self._release_ownership(target_id)
            self._safe_close_client(client)
            return None
        logger.info(
            "Reclaimed warm-pool e2b sandbox %s for user/thread %s/%s",
            target_id,
            user_id,
            thread_id,
        )
        return target_id

    def _discover_remote_sandbox(self, thread_id: str, *, user_id: str) -> str | None:
        """이 (user, thread)로 태깅된 실행 중 e2b sandbox를 찾는다.

        다른 gateway 프로세스(또는 재시작 전의 이 프로세스)가 이미 sandbox를
        만들어 두었을 수 있다. e2b sandbox는 서버 쪽 타임아웃이 발동하지 않는 한
        재연결을 넘어 유지된다.
        """
        sandbox_cls = self._get_sandbox_cls()
        seed = self._stable_seed(thread_id, user_id)
        entries, _, _ = self._list_remote_entries(
            {
                META_KEY_PROVIDER: META_VAL_PROVIDER,
                META_KEY_USER: user_id,
                META_KEY_THREAD: thread_id,
            }
        )
        candidates = sorted(
            (
                (sandbox_id, metadata)
                for entry in entries
                if (sandbox_id := self._entry_id(entry)) and (metadata := self._entry_metadata(entry)).get(META_KEY_USER) == user_id and metadata.get(META_KEY_THREAD) == thread_id and self._metadata_matches_capacity_ledger(metadata)
            ),
            key=lambda item: (item[1].get(META_KEY_CREATED_AT, ""), item[0]),
        )
        for target_id, metadata in candidates:
            adopted = self._adopt_remote_candidate(
                sandbox_cls,
                target_id,
                thread_id=thread_id,
                user_id=user_id,
                seed=seed,
                capacity_reservation=(self._capacity_reservation_from_metadata(metadata)),
            )
            if adopted is not None:
                return adopted
        return None

    def _adopt_remote_candidate(
        self,
        sandbox_cls: type[E2BClientSandbox],
        target_id: str,
        *,
        thread_id: str,
        user_id: str,
        seed: str,
        capacity_reservation: str | None = None,
    ) -> str | None:
        """발견한 후보 하나를 adopt한다. peer가 소유한 VM은 건드리지 않는다."""

        try:
            client = self._reconnect_live_client(sandbox_cls, target_id)
        except Exception as e:
            logger.warning(
                "Discovered e2b sandbox %s could not be reconnected: %s",
                target_id,
                e,
            )
            return None

        if client is None:
            logger.warning(
                "Discovered e2b sandbox %s is no longer alive; falling back to create",
                target_id,
            )
            return None

        try:
            self._track_deployment_sandbox(
                target_id,
                reservation_token=capacity_reservation,
            )
            self._reserve_capacity(
                thread_id,
                user_id,
                remote_id=target_id,
                remote_owned=False,
            )
        except SandboxCapacityExceededError as error:
            if error.reason == "shutdown":
                logger.info(
                    "Discovered e2b sandbox %s while the provider is shutting down; not adopting it",
                    target_id,
                )
            else:
                logger.info(
                    "Discovered e2b sandbox %s, but capacity is full; not adopting it",
                    target_id,
                )
            self._safe_close_client(client)
            raise

        with self._lock:
            shutdown_before_ownership = self._shutdown_called
        if shutdown_before_ownership:
            self._complete_reserved_remote_op(target_id, remote_destroyed=False)
            self._safe_close_client(client)
            return None

        try:
            self._publish_ownership(target_id)
        except Exception:
            self._complete_reserved_remote_op(target_id, remote_destroyed=False)
            self._safe_close_client(client)
            raise

        self._refresh_remote_timeout(client)
        bootstrap_error, remote_destroyed = self._bootstrap_or_discard(client, target_id)
        if bootstrap_error is not None:
            self._complete_reserved_remote_op(target_id, remote_destroyed=remote_destroyed)
            return None

        discard_after_shutdown = False
        with self._lock:
            if self._shutdown_called:
                discard_after_shutdown = True
            else:
                self._unowned_remote_ops_in_progress.discard(target_id)
                self._register_connected_sandbox(target_id, client, thread_id=thread_id, user_id=user_id)
                self._commit_capacity()
        if discard_after_shutdown:
            if self._claim_ownership(target_id, for_destroy=True):
                self._kill_client(client)
                self._release_ownership(target_id)
            self._safe_close_client(client)
            return None
        logger.info(
            "Discovered remote e2b sandbox %s for user/thread %s/%s (seed=%s)",
            target_id,
            user_id,
            thread_id,
            seed,
        )
        return target_id

    # ── capacity 예약 ─────────────────────────────────────────────────────
    #
    # sandbox 하나는 *slot* 하나를 점유하며, slot은 정확히 다음 네 상태 중 하나다.
    #
    #   reserved      _reserved_slots             create 진행 중, 아직 _sandboxes에 없음
    #   active        _sandboxes                  thread를 서빙 중
    #   warm          _warm_pool                  release 후 재사용 대기
    #   transitioning _transitioning_slots        상태 사이를 이동 중
    #
    # 총합 = _reserved_slots + len(_sandboxes) + len(_warm_pool) + _transitioning_slots
    #
    # ``transitioning`` 버킷은 sandbox가 ``_sandboxes``(또는 ``_warm_pool``)에서 빠졌지만
    # 아직 목적지에 안착하지 않은 구간을 메운다. 이것이 없으면
    # ``_release_internal``(active → warm), ``_reclaim_warm_pool_sandbox``(warm → active),
    # ``_evict_oldest_warm``(warm → 파기) 모두 일시적으로 점유 slot이 *0*으로 보이고,
    # 동시에 진행되는 acquire가 새 slot을 예약해 설정된 ``replicas``를 넘길 수 있다.

    def _total_capacity_used_locked(self) -> int:
        """reserved + active + warm + transitioning 합을 반환한다(``_lock``을 잡은 상태여야 한다)."""
        return self._reserved_slots + len(self._sandboxes) + len(self._warm_pool) + self._transitioning_slots

    def _capacity_limit(self) -> int:
        """reserved + active + warm + transitioning slot의 절대 상한."""
        replicas = int(self._config["replicas"])
        if self._config["overflow_policy"] == "burst":
            return replicas + int(self._config["burst_limit"])
        return replicas

    def _begin_transition_locked(self) -> None:
        """transitioning 카운터를 증가시킨다(``_lock``을 잡은 상태여야 한다).

        전이가 진행되는 동안에도 slot이 계속 집계되도록, sandbox를 ``_sandboxes``나
        ``_warm_pool``에서 제거하기 전에 호출한다.
        """
        self._transitioning_slots += 1

    def _end_transition_locked(self) -> None:
        """transitioning 카운터를 감소시킨다(``_lock``을 잡은 상태여야 한다).

        전이가 끝난 뒤 호출한다. 즉 slot이 목적지 dict에 안착했거나 sandbox가 파기된 뒤다.
        """
        if self._transitioning_slots > 0:
            self._transitioning_slots -= 1
            self._capacity_cond.notify_all()

    def _free_transitioning_slot(self) -> None:
        """sandbox 파기 후 transitioning slot을 해제한다."""
        with self._lock:
            self._end_transition_locked()

    def _capacity_reservation_max_age_ms(self) -> int:
        configured = compute_lease_ttl(self._ownership_config) + float(self._config["reconciliation_grace_seconds"])
        return int(max(configured, MIN_CAPACITY_RESERVATION_SECONDS) * 1_000)

    def _capacity_error(
        self,
        message: str,
        *,
        reason: str = "capacity",
    ) -> SandboxCapacityExceededError:
        with self._lock:
            return SandboxCapacityExceededError(
                message,
                active=len(self._sandboxes),
                warm=len(self._warm_pool),
                reserved=self._reserved_slots,
                replicas=int(self._config["replicas"]),
                reason=reason,
            )

    def _track_deployment_sandbox(
        self,
        sandbox_id: str,
        *,
        reservation_token: str | None = None,
        required: bool = True,
    ) -> None:
        store = self._deployment_capacity
        if store is None:
            return
        try:
            store.track(
                sandbox_id,
                reservation_token=reservation_token,
            )
        except CapacityBackendError as error:
            if required:
                raise self._capacity_error(
                    f"Deployment-wide E2B capacity is unavailable; cannot safely track sandbox {sandbox_id}",
                    reason="capacity_backend",
                ) from error
            logger.warning(
                "Could not track E2B sandbox %s in deployment capacity; reconciliation will retry: %s",
                sandbox_id,
                error,
            )

    def _release_deployment_sandbox(self, sandbox_id: str) -> None:
        store = self._deployment_capacity
        if store is None:
            return
        try:
            store.release(sandbox_id)
        except CapacityBackendError as error:
            logger.warning(
                "Could not release deployment capacity for destroyed E2B sandbox %s; reconciliation will retry: %s",
                sandbox_id,
                error,
            )

    def _reserve_capacity(
        self,
        thread_id: str | None,
        user_id: str,
        *,
        remote_id: str | None = None,
        remote_owned: bool = True,
    ) -> str | None:
        """새 VM을 위해 로컬 capacity를 먼저, 이어서 배포 전체 capacity를 예약한다."""
        store = self._deployment_capacity
        policy = self._config["overflow_policy"]
        timeout = float(self._config["acquire_timeout"])
        deadline = time.monotonic() + timeout
        token = uuid.uuid4().hex if store is not None and remote_id is None else None

        while True:
            self._reserve_local_capacity(
                thread_id,
                user_id,
                remote_id=remote_id,
                remote_owned=remote_owned,
                deadline=deadline,
            )
            if store is None or remote_id is not None:
                return token

            assert token is not None
            backend_error = None
            try:
                status = store.reserve(token)
            except CapacityBackendError as error:
                backend_error = error
                status = None
            if status is ReserveStatus.GRANTED:
                return token
            self._release_capacity()
            if status is ReserveStatus.FULL and self._evict_oldest_warm() is not None:
                continue

            if policy != "wait":
                if backend_error is not None:
                    raise self._capacity_error(
                        "Deployment-wide E2B capacity is unavailable",
                        reason="capacity_backend",
                    ) from backend_error
                if status is ReserveStatus.NOT_READY:
                    raise self._capacity_error(
                        "Deployment-wide E2B capacity is initializing",
                        reason="capacity_initializing",
                    )
                raise self._capacity_error("Deployment-wide E2B capacity is full")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._capacity_error(f"Timed out after {timeout}s waiting for deployment-wide E2B capacity")
            with self._capacity_cond:
                self._capacity_cond.wait(timeout=min(remaining, 1.0))

    def _reserve_local_capacity(
        self,
        thread_id: str | None,
        user_id: str,
        *,
        remote_id: str | None = None,
        remote_owned: bool = True,
        deadline: float | None = None,
    ) -> None:
        """기존 프로세스 로컬 lifecycle slot을 확보한다."""
        policy = self._config["overflow_policy"]
        timeout = float(self._config["acquire_timeout"])
        deadline = deadline or time.monotonic() + timeout

        while True:
            with self._lock:
                if self._shutdown_called:
                    raise SandboxCapacityExceededError(
                        "Sandbox provider is shutting down; cannot acquire capacity",
                        replicas=int(self._config["replicas"]),
                        retry_after_seconds=30.0,
                        reason="shutdown",
                    )

            with self._lock:
                if self._shutdown_called:
                    raise SandboxCapacityExceededError(
                        "Sandbox provider is shutting down; cannot acquire capacity",
                        replicas=int(self._config["replicas"]),
                        retry_after_seconds=30.0,
                        reason="shutdown",
                    )
                cap = self._capacity_limit()
                if self._total_capacity_used_locked() < cap:
                    self._reserved_slots += 1
                    if remote_id is not None:
                        remote_ops = self._remote_ops_in_progress if remote_owned else self._unowned_remote_ops_in_progress
                        remote_ops.add(remote_id)
                    return

            evicted = self._evict_oldest_warm()
            if evicted is not None:
                with self._lock:
                    if self._shutdown_called:
                        raise SandboxCapacityExceededError(
                            "Sandbox provider shut down while acquiring capacity",
                            replicas=int(self._config["replicas"]),
                            retry_after_seconds=30.0,
                            reason="shutdown",
                        )
                    if self._total_capacity_used_locked() < cap:
                        self._reserved_slots += 1
                        if remote_id is not None:
                            remote_ops = self._remote_ops_in_progress if remote_owned else self._unowned_remote_ops_in_progress
                            remote_ops.add(remote_id)
                        return

            with self._lock:
                if self._shutdown_called:
                    raise SandboxCapacityExceededError(
                        "Sandbox provider is shutting down; cannot acquire capacity",
                        replicas=int(self._config["replicas"]),
                        retry_after_seconds=30.0,
                        reason="shutdown",
                    )
                used = self._total_capacity_used_locked()
                cap = self._capacity_limit()

                if used >= cap:
                    if policy == "reject":
                        raise SandboxCapacityExceededError(
                            f"All {cap} sandbox capacity slots are in use and overflow_policy is 'reject'",
                            active=len(self._sandboxes),
                            warm=len(self._warm_pool),
                            reserved=self._reserved_slots,
                            replicas=int(self._config["replicas"]),
                        )

                    if policy == "burst":
                        raise SandboxCapacityExceededError(
                            f"All {cap} sandbox capacity slots are in use (replicas={self._config['replicas']}, burst={self._config['burst_limit']})",
                            active=len(self._sandboxes),
                            warm=len(self._warm_pool),
                            reserved=self._reserved_slots,
                            replicas=int(self._config["replicas"]),
                        )

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SandboxCapacityExceededError(
                            f"Timed out after {timeout}s waiting for a sandbox capacity slot (replicas={self._config['replicas']}, active={len(self._sandboxes)}, warm={len(self._warm_pool)}, reserved={self._reserved_slots})",
                            active=len(self._sandboxes),
                            warm=len(self._warm_pool),
                            reserved=self._reserved_slots,
                            replicas=int(self._config["replicas"]),
                        )
                    self._capacity_cond.wait(timeout=min(remaining, 1.0))

    def _release_capacity(self) -> None:
        """예약된 slot을 해제한다(create 실패나 파기 시 호출한다)."""
        with self._lock:
            if self._reserved_slots > 0:
                self._reserved_slots -= 1
            self._capacity_cond.notify_all()

    def _complete_transition_remote_op(self, sandbox_id: str, *, remote_destroyed: bool) -> None:
        """이미 transition slot을 점유한 원격 작업을 마무리한다."""
        with self._lock:
            if sandbox_id not in self._remote_ops_in_progress:
                return
            self._remote_ops_in_progress.discard(sandbox_id)
            if self._shutdown_called:
                return
            if remote_destroyed:
                self._end_transition_locked()
            else:
                self._eviction_tombstones.add(sandbox_id)

    def _track_reserved_remote_op(self, sandbox_id: str) -> bool:
        """예약된 원격 ID를 shutdown이 볼 수 있게 등록한다."""
        with self._lock:
            if self._shutdown_called:
                return False
            self._remote_ops_in_progress.add(sandbox_id)
            return True

    def _complete_reserved_remote_op(self, sandbox_id: str, *, remote_destroyed: bool) -> None:
        """예약(reserved) slot을 점유한 원격 작업을 마무리한다."""
        with self._lock:
            tracked = sandbox_id in self._remote_ops_in_progress
            tracked_unowned = sandbox_id in self._unowned_remote_ops_in_progress
            if not tracked and not tracked_unowned:
                return
            self._remote_ops_in_progress.discard(sandbox_id)
            self._unowned_remote_ops_in_progress.discard(sandbox_id)
            if self._shutdown_called:
                return
            if self._reserved_slots > 0:
                self._reserved_slots -= 1
            if remote_destroyed:
                self._capacity_cond.notify_all()
            else:
                self._begin_transition_locked()
                self._eviction_tombstones.add(sandbox_id)
            self._capacity_cond.notify_all()

    def _commit_capacity(self) -> None:
        """예약 slot을 확정된 active slot으로 전환한다.

        예약을 없애고 새로 만든 sandbox가 그 slot을 채운다. ``_sandboxes``에
        삽입하는 것과 같은 임계 구역 안에서 호출해야 한다.
        """
        if self._reserved_slots > 0:
            self._reserved_slots -= 1

    def _create_sandbox(self, thread_id: str | None, *, user_id: str) -> str:
        """새 e2b sandbox를 할당하고 설정된 mount를 채워 넣는다.

        capacity 제한은 :meth:`_reserve_capacity`가 원자적으로 강제한다.
        """
        reservation_token = self._reserve_capacity(thread_id, user_id)

        sandbox_cls = self._get_sandbox_cls()
        metadata: dict[str, str] = {
            META_KEY_PROVIDER: META_VAL_PROVIDER,
            META_KEY_GATEWAY: self._owner_id,
            META_KEY_CREATED_AT: str(time.time()),
        }
        if self._deployment_capacity is not None:
            metadata[META_KEY_CAPACITY_LEDGER] = self._deployment_capacity.key
        if reservation_token is not None:
            metadata[META_KEY_CAPACITY_RESERVATION] = reservation_token
        if thread_id:
            metadata[META_KEY_USER] = user_id
            metadata[META_KEY_THREAD] = thread_id

        create_kwargs: dict[str, Any] = {
            "template": self._config["template"],
            "metadata": metadata,
            **self._common_kwargs(),
        }
        if self._config["idle_timeout"] > 0:
            create_kwargs["timeout"] = self._config["idle_timeout"]
        if self._config["environment"]:
            create_kwargs["envs"] = self._config["environment"]

        try:
            client = sandbox_cls.create(**create_kwargs)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error("Failed to create e2b sandbox: %s", e)
            self._release_capacity()
            raise

        sandbox_id: str = getattr(client, "sandbox_id", None) or str(uuid.uuid4())[:8]
        self._track_deployment_sandbox(
            sandbox_id,
            reservation_token=reservation_token,
            required=False,
        )
        if not self._track_reserved_remote_op(sandbox_id):
            kill_error = self._kill_client(client)
            cleanup_confirmed = kill_error is None
            self._safe_close_client(client)
            if kill_error is not None:
                with self._lock:
                    self._remote_ops_in_progress.add(sandbox_id)
                try:
                    retry_client = self._reconnect_client(sandbox_cls, sandbox_id)
                except Exception as reconnect_error:
                    logger.warning(
                        "Failed to reconnect e2b sandbox %s after shutdown cleanup failed: %s",
                        sandbox_id,
                        reconnect_error,
                    )
                else:
                    retry_error = self._kill_client(retry_client)
                    self._safe_close_client(retry_client)
                    if retry_error is None:
                        cleanup_confirmed = True
                        with self._lock:
                            self._remote_ops_in_progress.discard(sandbox_id)
                    else:
                        logger.warning(
                            "Failed to kill e2b sandbox %s after reconnecting during shutdown: %s",
                            sandbox_id,
                            retry_error,
                        )
            if cleanup_confirmed:
                message = f"Sandbox provider shut down during sandbox creation; cleaned up remote sandbox {sandbox_id}"
            else:
                message = f"Sandbox provider shut down during sandbox creation; could not confirm cleanup for remote sandbox {sandbox_id}"
            raise SandboxCapacityExceededError(
                message,
                replicas=int(self._config["replicas"]),
                retry_after_seconds=30.0,
                reason="shutdown",
            )

        try:
            self._publish_ownership(sandbox_id)
        except Exception:
            remote_destroyed = False
            if self._claim_ownership(sandbox_id, for_destroy=True):
                remote_destroyed = self._kill_client(client) is None
                self._release_ownership(sandbox_id)
            self._safe_close_client(client)
            self._complete_reserved_remote_op(sandbox_id, remote_destroyed=remote_destroyed)
            raise

        # DeerFlow의 가상 경로 구조(/mnt/user-data/...)를 e2b VM 안에 실제로 만든다.
        # e2b 템플릿에서 /mnt는 root 소유라, 이 단계가 없으면 에이전트가 내보내는
        # 셸 명령(LocalSandbox/AioSandbox와 동일한 /mnt/user-data prefix를 쓴다)이
        # PermissionError로 실패한다. :class:`E2BSandbox`의 경로 매핑 설명 참고.
        bootstrap_error, remote_destroyed = self._bootstrap_or_discard(client, sandbox_id)
        if bootstrap_error is not None:
            self._complete_reserved_remote_op(sandbox_id, remote_destroyed=remote_destroyed)
            raise RuntimeError(f"Failed to bootstrap e2b sandbox {sandbox_id}") from bootstrap_error

        # 1회성 mount 업로드. e2b에는 호스트 bind-mount가 없으므로 sandbox 시작 시
        # ``host_path``의 파일을 ``container_path``로 복사한다.
        try:
            self._apply_mounts(client, user_id=user_id)
        except Exception as e:
            logger.warning("Failed to apply some mounts to e2b sandbox %s: %s", sandbox_id, e)

        sandbox = E2BSandbox(id=sandbox_id, client=client, home_dir=self._config["home_dir"])

        # 원자적으로 커밋한다. bootstrap이나 mount 도중 provider가 shutdown되었다면,
        # 다음 shutdown이 보지 못할 ``_sandboxes``에 넣지 말고 VM을 kill한다.
        should_kill = False
        with self._lock:
            if self._shutdown_called:
                should_kill = True
            else:
                self._remote_ops_in_progress.discard(sandbox_id)
                self._commit_capacity()
                self._sandboxes[sandbox_id] = sandbox
                if thread_id:
                    self._thread_sandboxes[self._thread_key(thread_id, user_id)] = sandbox_id

        if should_kill:
            if self._claim_ownership(sandbox_id, for_destroy=True):
                self._kill_client(client)
                self._release_ownership(sandbox_id)
            self._safe_close_client(client)
            raise SandboxCapacityExceededError(
                f"Sandbox provider shut down during sandbox creation; killed remote sandbox {sandbox_id}",
                replicas=int(self._config["replicas"]),
                retry_after_seconds=30.0,
                reason="shutdown",
            )

        replicas = self._config["replicas"]
        logger.info(
            "Created e2b sandbox %s for user/thread %s/%s (template=%s, replicas=%s)",
            sandbox_id,
            user_id,
            thread_id,
            self._config["template"],
            replicas,
        )
        return sandbox_id

    def _common_kwargs(self) -> dict[str, Any]:
        """``Sandbox.create``, ``Sandbox.connect``, ``Sandbox.list``가 공유하는 kwargs."""
        kwargs: dict[str, Any] = {}
        if self._config["api_key"]:
            kwargs["api_key"] = self._config["api_key"]
        if self._config["domain"]:
            kwargs["domain"] = self._config["domain"]
        return kwargs

    @staticmethod
    def _entry_id(entry: Any) -> str | None:
        value = getattr(entry, "sandbox_id", None)
        if value is None and isinstance(entry, dict):
            value = entry.get("sandbox_id")
        return str(value) if value else None

    @staticmethod
    def _entry_metadata(entry: Any) -> dict[str, Any]:
        value = getattr(entry, "metadata", None)
        if value is None and isinstance(entry, dict):
            value = entry.get("metadata")
        return value if isinstance(value, dict) else {}

    def _list_remote_entries(
        self,
        metadata: dict[str, str],
    ) -> tuple[list[Any], bool, bool]:
        """E2B 엔트리를 나열하고 budget 소진 여부와 목록의 완전성을 함께 보고한다."""
        sandbox_cls = self._get_sandbox_cls()
        try:
            query = SandboxQuery(metadata=metadata)
            result = sandbox_cls.list(query=query, **self._common_kwargs())  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("E2B reconciliation list failed: %s", e)
            return [], False, False

        max_pages = int(self._config["reconciliation_max_pages"])
        max_items = int(self._config["reconciliation_max_items"])
        deadline = time.monotonic() + float(self._config["reconciliation_max_seconds"])
        entries: list[Any] = []
        exhausted = False
        complete = True

        if hasattr(result, "next_items") and hasattr(result, "has_next"):
            for page_number in range(max_pages):
                if time.monotonic() >= deadline or len(entries) >= max_items:
                    exhausted = True
                    break
                try:
                    page = result.next_items()
                except Exception as e:
                    logger.warning("E2B reconciliation paginator failed: %s", e)
                    complete = False
                    break
                if not page:
                    break
                room = max_items - len(entries)
                entries.extend(list(page)[:room])
                if len(page) > room:
                    exhausted = True
                    break
                if not getattr(result, "has_next", False):
                    break
                if page_number + 1 >= max_pages:
                    exhausted = True
        else:
            try:
                all_entries = list(result or [])
            except TypeError:
                logger.warning("E2B Sandbox.list returned non-iterable %s", type(result).__name__)
                return [], False, False
            entries = all_entries[:max_items]
            exhausted = len(all_entries) > max_items

        if time.monotonic() >= deadline:
            exhausted = True
        if exhausted:
            complete = False
        return entries, exhausted, complete

    def _publish_ownership(self, sandbox_id: str) -> None:
        """sandbox를 로컬에 노출하기 전에 acquire 쪽 ownership을 publish한다."""
        with self._lock:
            self._acquire_inflight.add(sandbox_id)
        try:
            if not self._ownership.take(sandbox_id):
                raise RuntimeError(f"E2B sandbox {sandbox_id} is being destroyed")
        except Exception:
            with self._lock:
                self._acquire_inflight.discard(sandbox_id)
            raise
        with self._lock:
            self._owned_sandbox_ids.add(sandbox_id)

    def _claim_ownership(self, sandbox_id: str, *, for_destroy: bool = False) -> bool:
        """소유자 없는 sandbox를 배타적으로 claim한다. store 오류 시 fail-closed로 동작한다."""
        try:
            claimed = self._ownership.claim(sandbox_id, for_destroy=for_destroy)
        except OwnershipBackendError as e:
            logger.warning("E2B ownership claim failed for %s: %s", sandbox_id, e)
            return False
        if claimed:
            with self._lock:
                self._owned_sandbox_ids.add(sandbox_id)
        return claimed

    def _release_ownership(self, sandbox_id: str) -> None:
        try:
            self._ownership.release(sandbox_id)
        except OwnershipBackendError as e:
            logger.warning("Failed to release E2B ownership for %s: %s", sandbox_id, e)
        with self._lock:
            self._owned_sandbox_ids.discard(sandbox_id)
            self._acquire_inflight.discard(sandbox_id)

    def _forget_local_sandbox(self, sandbox_id: str) -> None:
        """peer가 가져간 lease를 잊는다. 원격 VM은 건드리지 않는다."""
        with self._lock:
            if sandbox_id in self._acquire_inflight:
                return
            sandbox = self._sandboxes.pop(sandbox_id, None)
            self._warm_pool.pop(sandbox_id, None)
            self._owned_sandbox_ids.discard(sandbox_id)
            for key, sid in list(self._thread_sandboxes.items()):
                if sid == sandbox_id:
                    self._thread_sandboxes.pop(key, None)
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception:
                pass

    def _refresh_owned_leases(self) -> None:
        with self._lock:
            sandbox_ids = list(self._owned_sandbox_ids)
        for sandbox_id in sandbox_ids:
            try:
                outcome = self._ownership.renew(sandbox_id)
            except OwnershipBackendError as e:
                logger.warning("Could not renew E2B ownership for %s; will retry: %s", sandbox_id, e)
                continue
            if outcome is RenewOutcome.RENEWED:
                continue
            if outcome is RenewOutcome.LAPSED:
                try:
                    if self._ownership.claim(sandbox_id):
                        continue
                except OwnershipBackendError as e:
                    logger.warning("Could not re-establish E2B ownership for %s: %s", sandbox_id, e)
                    continue
            logger.info("E2B sandbox %s ownership moved to a peer; forgetting local client", sandbox_id)
            self._forget_local_sandbox(sandbox_id)

    def _start_maintenance_threads(self) -> None:
        def renew() -> None:
            interval = self._ownership_config.renewal_interval_seconds
            while not self._maintenance_stop.wait(interval):
                self._refresh_owned_leases()

        def reconcile() -> None:
            interval = float(self._config["reconciliation_interval_seconds"])
            while not self._maintenance_stop.is_set():
                try:
                    self._reconcile_remote_sandboxes()
                except Exception:
                    logger.exception("Periodic E2B sandbox reconciliation failed")
                if self._maintenance_stop.wait(interval):
                    break

        self._lease_thread = threading.Thread(target=renew, name="e2b-lease-renewal", daemon=True)
        self._reconcile_thread = threading.Thread(target=reconcile, name="e2b-reconciliation", daemon=True)
        self._lease_thread.start()
        self._reconcile_thread.start()

    def _reserve_reconciliation_capacity(
        self,
        sandbox_id: str,
        *,
        reservation_token: str | None = None,
    ) -> bool:
        """maintenance를 막지 않으면서 adopt용 로컬 slot 하나를 예약한다."""
        try:
            self._track_deployment_sandbox(
                sandbox_id,
                reservation_token=reservation_token,
            )
        except SandboxCapacityExceededError as error:
            logger.warning("Could not track discovered E2B sandbox %s: %s", sandbox_id, error)
            return False
        with self._lock:
            if self._shutdown_called or self._total_capacity_used_locked() >= self._capacity_limit():
                return False
            self._reserved_slots += 1
            self._unowned_remote_ops_in_progress.add(sandbox_id)
            self._acquire_inflight.add(sandbox_id)
            return True

    def _reconcile_remote_sandboxes(self, *, now: float | None = None) -> ReconciliationStats:
        """정규(canonical) E2B sandbox를 adopt하고 중복/고아 sandbox를 안전하게 회수한다."""
        observed_at = time.monotonic() if now is None else now
        deadline = time.monotonic() + float(self._config["reconciliation_max_seconds"])
        stats = ReconciliationStats()
        capacity_revision = None
        capacity_store = self._deployment_capacity
        if capacity_store is not None:
            try:
                capacity_revision = capacity_store.revision()
            except CapacityBackendError as error:
                logger.warning(
                    "Could not read E2B capacity before reconciliation: %s",
                    error,
                )

        entries, stats.budget_exhausted, inventory_complete = self._list_remote_entries({META_KEY_PROVIDER: META_VAL_PROVIDER})
        entries = [entry for entry in entries if self._metadata_matches_capacity_ledger(self._entry_metadata(entry))]
        stats.discovered = len(entries)

        if capacity_store is not None and capacity_revision is not None:
            records = {sandbox_id: self._capacity_reservation_from_metadata(self._entry_metadata(entry)) for entry in entries if (sandbox_id := self._entry_id(entry))}
            try:
                applied = capacity_store.reconcile(
                    expected_revision=capacity_revision,
                    remote_sandboxes=records,
                    complete=inventory_complete,
                    reservation_max_age_ms=self._capacity_reservation_max_age_ms(),
                )
                if not applied:
                    logger.debug("E2B capacity inventory became stale during reconciliation; retrying on the next pass")
            except CapacityBackendError as error:
                logger.warning(
                    "Could not apply E2B capacity inventory: %s",
                    error,
                )

        groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
        orphans: list[tuple[str, dict[str, Any]]] = []
        present_ids: set[str] = set()

        for entry in entries:
            sandbox_id = self._entry_id(entry)
            if not sandbox_id:
                continue
            present_ids.add(sandbox_id)
            metadata = self._entry_metadata(entry)
            user_id = metadata.get(META_KEY_USER)
            thread_id = metadata.get(META_KEY_THREAD)
            if isinstance(user_id, str) and user_id and isinstance(thread_id, str) and thread_id:
                groups.setdefault((user_id, thread_id), []).append((sandbox_id, metadata))
            else:
                orphans.append((sandbox_id, metadata))

        for (user_id, thread_id), candidates in groups.items():
            candidates.sort(key=lambda item: (item[1].get(META_KEY_CREATED_AT, ""), item[0]))
            with self._lock:
                local_id = self._thread_sandboxes.get((user_id, thread_id))
            if local_id:
                candidates.sort(key=lambda item: item[0] != local_id)

            live: list[tuple[str, dict[str, Any], E2BClientSandbox]] = []
            for sandbox_id, metadata in candidates:
                if time.monotonic() >= deadline:
                    stats.budget_exhausted = True
                    break
                try:
                    client = self._reconnect_live_client(self._get_sandbox_cls(), sandbox_id)
                except Exception as e:
                    logger.debug("Could not probe E2B sandbox %s during reconciliation: %s", sandbox_id, e)
                    continue
                if client is None:
                    stats.dead += 1
                    continue
                live.append((sandbox_id, metadata, client))

            if not live:
                continue
            stats.duplicates += max(0, len(live) - 1)
            canonical_id, canonical_metadata, canonical_client = live[0]
            with self._lock:
                already_local = canonical_id in self._sandboxes
            if already_local:
                self._safe_close_client(canonical_client)
            elif not self._reserve_reconciliation_capacity(
                canonical_id,
                reservation_token=self._capacity_reservation_from_metadata(canonical_metadata),
            ):
                self._safe_close_client(canonical_client)
                stats.deferred += 1
            elif not self._claim_ownership(canonical_id):
                self._complete_reserved_remote_op(canonical_id, remote_destroyed=False)
                with self._lock:
                    self._acquire_inflight.discard(canonical_id)
                self._safe_close_client(canonical_client)
                stats.deferred += 1
            else:
                bootstrap_error, remote_destroyed = self._bootstrap_or_discard(canonical_client, canonical_id)
                if bootstrap_error is not None:
                    self._complete_reserved_remote_op(canonical_id, remote_destroyed=remote_destroyed)
                    with self._lock:
                        self._acquire_inflight.discard(canonical_id)
                    stats.deferred += 1
                else:
                    discard_after_shutdown = False
                    with self._lock:
                        if self._shutdown_called:
                            discard_after_shutdown = True
                        else:
                            self._owned_sandbox_ids.add(canonical_id)
                            self._unowned_remote_ops_in_progress.discard(canonical_id)
                            self._register_connected_sandbox(
                                canonical_id,
                                canonical_client,
                                thread_id=thread_id,
                                user_id=user_id,
                            )
                            self._commit_capacity()
                    if discard_after_shutdown:
                        if self._claim_ownership(canonical_id, for_destroy=True):
                            self._kill_client(canonical_client)
                            self._release_ownership(canonical_id)
                        self._safe_close_client(canonical_client)
                    else:
                        stats.adopted += 1

            for sandbox_id, _metadata, client in live[1:]:
                first_seen = self._orphan_first_seen.setdefault(sandbox_id, observed_at)
                if observed_at - first_seen < float(self._config["reconciliation_grace_seconds"]):
                    self._safe_close_client(client)
                    stats.deferred += 1
                    continue
                if not self._claim_ownership(sandbox_id, for_destroy=True):
                    self._safe_close_client(client)
                    stats.deferred += 1
                    continue
                if error := self._kill_client(client):
                    logger.warning("Failed to kill duplicate E2B sandbox %s: %s", sandbox_id, error)
                    self._release_ownership(sandbox_id)
                    stats.deferred += 1
                else:
                    stats.killed += 1
                    self._orphan_first_seen.pop(sandbox_id, None)
                    self._forget_local_sandbox(sandbox_id)
                    self._release_ownership(sandbox_id)
                self._safe_close_client(client)

        for sandbox_id, metadata in orphans:
            if time.monotonic() >= deadline:
                stats.budget_exhausted = True
                break
            first_seen = self._orphan_first_seen.setdefault(sandbox_id, observed_at)
            created_at = metadata.get(META_KEY_CREATED_AT)
            try:
                age = time.time() - float(created_at) if created_at is not None else observed_at - first_seen
            except (TypeError, ValueError):
                age = observed_at - first_seen
            if age < float(self._config["reconciliation_orphan_ttl_seconds"]):
                stats.deferred += 1
                continue
            if not self._claim_ownership(sandbox_id, for_destroy=True):
                stats.deferred += 1
                continue
            try:
                client = self._reconnect_client(self._get_sandbox_cls(), sandbox_id)
            except Exception:
                self._release_ownership(sandbox_id)
                continue
            if error := self._kill_client(client):
                logger.warning("Failed to kill orphan E2B sandbox %s: %s", sandbox_id, error)
                self._release_ownership(sandbox_id)
                stats.deferred += 1
            else:
                stats.killed += 1
                self._orphan_first_seen.pop(sandbox_id, None)
                self._forget_local_sandbox(sandbox_id)
                self._release_ownership(sandbox_id)
            self._safe_close_client(client)

        for sandbox_id in set(self._orphan_first_seen) - present_ids:
            self._orphan_first_seen.pop(sandbox_id, None)
        logger.info(
            "E2B reconciliation: discovered=%d adopted=%d duplicates=%d deferred=%d killed=%d dead=%d budget_exhausted=%s",
            stats.discovered,
            stats.adopted,
            stats.duplicates,
            stats.deferred,
            stats.killed,
            stats.dead,
            stats.budget_exhausted,
        )
        return stats

    def _reconnect_client(self, sandbox_cls: type[E2BClientSandbox], sandbox_id: str) -> E2BClientSandbox:
        """id로 기존 e2b sandbox에 연결한다. kwargs는 항상 동일하게 맞춘다."""
        return sandbox_cls.connect(sandbox_id, **self._common_kwargs())  # type: ignore[attr-defined]

    def _reconnect_live_client(
        self,
        sandbox_cls: type[E2BClientSandbox],
        sandbox_id: str,
    ) -> E2BClientSandbox | None:
        """*sandbox_id*에 재연결하되, 이미 회수된 VM의 client는 거부한다.

        E2B control plane이 VM을 회수한 뒤에도 ``Sandbox.connect``는 성공할 수 있다.
        ``None``을 반환하기 전에 호스트 쪽 client를 닫아, 두 acquire 경로 모두에서
        연결이 새지 않게 한다.
        """
        try:
            client = self._reconnect_client(sandbox_cls, sandbox_id)
        except Exception as error:
            if _is_sandbox_gone_error(error):
                self._release_deployment_sandbox(sandbox_id)
            raise
        if self._client_alive(client):
            return client
        self._safe_close_client(client)
        self._release_deployment_sandbox(sandbox_id)
        return None

    def _register_connected_sandbox(
        self,
        sandbox_id: str,
        client: E2BClientSandbox,
        *,
        thread_id: str | None,
        user_id: str,
    ) -> None:
        """재연결된 live sandbox를 해당 thread 소유로 등록한다.

        호출자는 ``self._lock``을 잡고 있어야 한다.
        """
        sandbox = E2BSandbox(id=sandbox_id, client=client, home_dir=self._config["home_dir"])
        self._sandboxes[sandbox_id] = sandbox
        self._warm_pool.pop(sandbox_id, None)
        if thread_id:
            self._thread_sandboxes[self._thread_key(thread_id, user_id)] = sandbox_id
        self._acquire_inflight.discard(sandbox_id)

    def _refresh_remote_timeout(self, client: E2BClientSandbox) -> None:
        """설정된 idle timeout을 e2b control plane에 반영한다."""
        idle_timeout = int(self._config["idle_timeout"])
        if idle_timeout <= 0:
            return
        set_timeout = getattr(client, "set_timeout", None)
        if not callable(set_timeout):
            return
        try:
            set_timeout(idle_timeout)
        except Exception as e:  # pragma: no cover - 방어적 처리
            logger.debug("Failed to set timeout on e2b sandbox: %s", e)

    @staticmethod
    def _client_alive(client: E2BClientSandbox) -> bool:
        """방금 재연결한 e2b client에 대한 best-effort liveness probe.

        일부 SDK 버전에서는 pause/만료된 sandbox에도 ``Sandbox.connect``가 성공하고,
        실패는 첫 명령에서야 드러난다. 여기서 간단한 ``true`` 셸 명령을 보내
        실패를 acquire 경로에서 일으킨다. 그러면 조용히 새 sandbox 생성으로 넘어갈 수
        있고, tool 호출 도중 에이전트가 혼란스러운 "sandbox not found" 스택 트레이스를
        보는 일이 없다.

        명령이 성공하면 ``True``, "sandbox not found / paused" 에러가 나면 ``False``를
        반환한다. 그 밖의 일시적 오류는 살아 있는 것으로 처리해, 네트워크가 한 번
        끊겼다고 캐시를 날려 버리지 않는다.
        """
        try:
            client.commands.run("true")
            return True
        except Exception as e:
            if _is_sandbox_gone_error(e):
                return False
            logger.debug("e2b client liveness probe non-fatal error: %s", e)
            return True

    @staticmethod
    def _safe_close_client(client: E2BClientSandbox | None) -> None:
        """*client*의 호스트 쪽 HTTP client를 예외 없이 닫는다.

        e2b VM이 이미 도달 불가(pause/만료)임을 알고 gateway 프로세스의 소켓만
        반납하려는 정리 경로에서 쓴다. 모든 예외는 debug 레벨로 남기고 삼킨다.
        """
        if client is None:
            return
        for attr in ("close", "_transport"):
            target = getattr(client, attr, None)
            if target is None:
                continue
            close = target if callable(target) else getattr(target, "close", None)
            if not callable(close):
                continue
            try:
                close()
                return
            except Exception as e:  # pragma: no cover - 방어적 처리
                logger.debug("e2b client close raised: %s", e)
                return

    def _bootstrap_or_discard(self, client: E2BClientSandbox, sandbox_id: str) -> tuple[Exception | None, bool]:
        """sandbox를 bootstrap하고, 정리 과정에서 VM을 파기했는지 함께 보고한다."""
        try:
            self._bootstrap_sandbox_paths(client)
        except Exception as e:
            logger.exception("Failed to bootstrap e2b sandbox %s. Discarding the unusable sandbox.", sandbox_id)
            remote_destroyed = False
            if self._claim_ownership(sandbox_id, for_destroy=True):
                kill_error = self._kill_client(client)
                remote_destroyed = kill_error is None
                if kill_error:
                    logger.warning("Failed to kill e2b sandbox %s after bootstrap failure: %s", sandbox_id, kill_error)
                self._release_ownership(sandbox_id)
            else:
                logger.info(
                    "Not killing E2B sandbox %s after bootstrap failure because a peer owns it",
                    sandbox_id,
                )
            self._safe_close_client(client)
            return e, remote_destroyed
        return None, True

    def _bootstrap_sandbox_paths(self, client: E2BClientSandbox) -> None:
        """DeerFlow의 가상 경로 구조를 e2b VM 안에 실제로 만든다.

        local/docker sandbox는 ``/mnt/user-data/{workspace,uploads,outputs}``와
        ``/mnt/acp-workspace``를 쓰기 가능한 디렉터리로 노출하고, 에이전트 프롬프트
        (특히 lead-agent 시스템 프롬프트)는 모델에게 결과물을 그곳에 쓰라고 지시한다.
        반면 e2b의 기본 ``code-interpreter`` 템플릿은 비특권 ``user``(uid 1000)로
        실행되고 ``/mnt``는 ``root`` 소유라, 에이전트가 내는
        ``mkdir -p /mnt/user-data/...``는 ``Permission denied``로 실패한다.

        sandbox 시작 시 한 번에 다음을 처리해 이를 해결한다.

        1. ``/home/user/{workspace,uploads,outputs}``를 실제 쓰기 가능한 backing
           디렉터리로 만든다(에이전트 HOME 아래라 권한 문제가 없다).
        2. ``sudo``로 ``/mnt/user-data``를 ``/home/user``에, ``/mnt/acp-workspace``를
           ``/home/user/acp-workspace``에 symlink한다. 그러면 문서화된 ``/mnt/...``
           경로를 쓰는 명령이 그대로 동작하고, :class:`E2BSandbox._resolve_path`가
           이미 매핑하는 것과 같은 물리 위치에 떨어진다.
        3. symlink(필요하면 ``/mnt`` 자체)의 소유권을 조정해, symlink 대상으로의
           이후 쓰기가 성공하게 한다.

        e2b code-interpreter 템플릿은 ``user``를 비밀번호 없는 ``sudo`` 그룹에 넣는다.
        커스텀 템플릿도 동등한 권한을 제공해야 한다. bootstrap이 실패하면 sandbox는
        쓸 수 없으므로, provider는 반쯤 동작하는 VM을 돌려주지 않고 폐기한다.
        """
        # 커스텀 템플릿이 HOME을 옮길 수 있으므로 설정된 ``home_dir``을 쓴다.
        home_dir = self._config["home_dir"].rstrip("/") or "/home/user"
        bootstrap_script = (
            f"set -e; "
            f"mkdir -p {shlex.quote(home_dir)}/workspace "
            f"{shlex.quote(home_dir)}/uploads "
            f"{shlex.quote(home_dir)}/outputs "
            f"{shlex.quote(home_dir)}/acp-workspace; "
            # /mnt/user-data -> $home_dir
            f"if [ ! -e /mnt/user-data ] || [ -L /mnt/user-data ]; then "
            f"  sudo ln -sfn {shlex.quote(home_dir)} /mnt/user-data; "
            f"fi; "
            # /mnt/acp-workspace -> $home_dir/acp-workspace
            f"if [ ! -e /mnt/acp-workspace ] || [ -L /mnt/acp-workspace ]; then "
            f"  sudo ln -sfn {shlex.quote(home_dir)}/acp-workspace /mnt/acp-workspace; "
            f"fi; "
            # /mnt/skills는 여기서 건드리지 않는다. 선택적인 ``mounts`` 설정이
            # _apply_mounts로 내용을 업로드하며 필요할 때 디렉터리를 만든다.
            # 여기서는 /mnt 자체를 통과할 수 있게만 보장한다.
            f"sudo chmod a+rx /mnt 2>/dev/null || true; "
            f"echo BOOTSTRAP_OK"
        )

        try:
            result = client.commands.run(bootstrap_script)
        except Exception as e:
            raise RuntimeError("e2b bootstrap script raised") from e

        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "exit_code", 0)
        if exit_code not in (0, None) or "BOOTSTRAP_OK" not in stdout:
            raise RuntimeError(f"e2b bootstrap script failed with exit code {exit_code}; stderr={stderr.strip()}")

    def _skill_projection_mounts(self, user_id: str) -> list[tuple[Path, str, bool]]:
        """best-effort로 동작한다. projection 실패가 설정된 mount까지 날려서는 안 된다.

        Local/AIO의 ``_ensure_skills_projection``과 달리, 예전에는 이 함수가
        configured-mounts 루프가 돌기 전에 ``_apply_mounts`` 밖으로 바로 예외를 던져서
        projection이 한 번 삐끗하면 운영자가 설정한 mount까지 함께 유실됐다
        (``create()``의 바깥 warning에만 걸리고 mount는 하나도 적용되지 않았다).
        여기서 예외를 삼켜 두 mount 소스를 독립적으로 유지하며, 다른 두 provider와
        동작을 맞춘다.
        """
        from deerflow.skills.projection import ensure_skill_projections
        from deerflow.skills.storage import get_or_new_user_skill_storage

        try:
            config = get_app_config()
            storage = get_or_new_user_skill_storage(user_id, app_config=config)
            projection = ensure_skill_projections(storage)
            container_root = config.skills.container_path.rstrip("/")
            return [
                (projection.public, f"{container_root}/public", True),
                (projection.custom, f"{container_root}/custom", True),
                (projection.legacy, f"{container_root}/legacy", True),
                (projection.integrations, f"{container_root}/integrations", True),
            ]
        except Exception as exc:
            logger.warning("Could not ensure skills projection for user %s: %s", user_id, exc, exc_info=True)
            return []

    def _apply_mounts(self, client: E2BClientSandbox, *, user_id: str | None = None) -> None:
        effective_user_id = user_id or get_effective_user_id()
        projection_mounts = self._skill_projection_mounts(effective_user_id)
        configured_mounts = self._config.get("mounts") or []
        skills_root = get_app_config().skills.container_path.rstrip("/")

        mounts: list[tuple[Path, str, bool]] = list(projection_mounts)
        for mount in configured_mounts:
            try:
                host_path = Path(getattr(mount, "host_path", "") or "")
                container_path = (getattr(mount, "container_path", "") or "").rstrip("/")
                read_only = bool(getattr(mount, "read_only", False))
            except AttributeError:
                host_path = Path(mount.get("host_path", ""))
                container_path = (mount.get("container_path", "") or "").rstrip("/")
                read_only = bool(mount.get("read_only", False))

            if container_path == skills_root or container_path.startswith(skills_root + "/"):
                logger.warning("Skipping e2b mount that conflicts with managed skills projection: %s", container_path)
                continue
            mounts.append((host_path, container_path, read_only))

        for host_path, container_path, read_only in mounts:
            if not host_path.exists():
                logger.warning("Skipping e2b mount: host_path %s does not exist", host_path)
                continue
            if not container_path.startswith("/"):
                logger.warning(
                    "Skipping e2b mount: container_path %s must be absolute",
                    container_path,
                )
                continue

            try:
                make_dir = getattr(client.files, "make_dir", None)
                if callable(make_dir):
                    make_dir(container_path)
            except Exception as e:
                logger.debug("make_dir(%s) failed (continuing): %s", container_path, e)

            try:
                self._upload_tree(client, host_path, container_path, read_only)
            except Exception as e:
                logger.warning("Failed to upload mount %s -> %s: %s", host_path, container_path, e)

    # ── 출력물 미러링 ───────────────────────────────────────────────────
    _SYNC_BACK_SUBDIRS = ("outputs", "workspace")
    _SYNC_MANIFEST_NAME = ".e2b-output-sync.json"

    # release 시점의 output sync 1회 pass에 대한 총량 상한. 파일 단위
    # ``_MAX_DOWNLOAD_SIZE`` 상한 위에 얹는다. 파일 단위 상한은 artifact 하나를
    # 제한하고, 이 값들은 pass 전체를 제한해 병적인 outputs 트리(파일 수천 개,
    # 상한 미만이지만 합치면 수 GB, 또는 느린 VM)가 release 다운로드를 무한정
    # 늘리지 못하게 한다. 상한에 걸리면 pass를 조기 종료하고 무엇을 빠뜨렸는지 로그로
    # 남기며, manifest는 정리하지 않고 둔다. 그래야 도달하지 못한 파일이 잊히지 않고
    # 다음 release에서 다시 처리된다.
    _MAX_SYNC_TOTAL_BYTES = 512 * 1024 * 1024  # pass당 총 다운로드 바이트
    _MAX_SYNC_FILES = 2000  # pass당 다운로드 파일 수
    _SYNC_DEADLINE_SECONDS = 120  # pass당 wall-clock 예산

    @staticmethod
    def _load_sync_manifest(manifest_path: Path, sandbox_id: str) -> tuple[dict[str, dict[str, int]], bool]:
        """직전 output sync에서 확인된 remote/host 버전 정보를 읽어 온다."""
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}, False
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("e2b sync: failed to load manifest %s: %s", manifest_path, e)
            return {}, True

        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("files"), dict):
            logger.warning("e2b sync: ignoring invalid manifest %s", manifest_path)
            return {}, True
        if data.get("sandbox_id") != sandbox_id:
            logger.debug("e2b sync: ignoring manifest from another sandbox %s", manifest_path)
            return {}, True

        files: dict[str, dict[str, int]] = {}
        for key, value in data["files"].items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            required = ("remote_size", "remote_mtime_ns", "host_size", "host_mtime_ns")
            if all(isinstance(value.get(field), int) for field in required):
                files[key] = {field: value[field] for field in required}
        return files, False

    @staticmethod
    def _write_sync_manifest(
        manifest_path: Path,
        sandbox_id: str,
        files: dict[str, dict[str, int]],
    ) -> None:
        """호스트 파일 기록이 끝난 뒤 output sync 버전 정보를 원자적으로 저장한다."""
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
            tmp_path.write_text(
                json.dumps({"version": 1, "sandbox_id": sandbox_id, "files": files}, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(manifest_path)
        except OSError as e:
            logger.warning("e2b sync: failed to write manifest %s: %s", manifest_path, e)

    def _sync_outputs_to_host(
        self,
        sandbox: E2BSandbox,
        *,
        thread_id: str,
        user_id: str,
    ) -> None:
        """에이전트 artifact를 e2b VM에서 호스트의 thread 디렉터리로 미러링한다.

        DeerFlow의 ``/api/threads/{tid}/artifacts/...`` 엔드포인트는 호스트 쪽
        thread별 ``user-data/`` 트리를 기준으로 파일을 찾는다
        (:meth:`Paths.sandbox_outputs_dir` 참고). LocalSandbox는 경로 매핑으로 그곳에
        직접 쓰기 때문에 local provider에서는 엔드포인트가 그냥 동작한다. e2b VM에는
        공유 호스트 파일시스템이 없으므로 release 시점에 artifact를 명시적으로 끌어온다.

        호스트 쪽 대응 파일이 없거나, 크기가 다르거나, 원격 수정 시각이 다른 파일을
        미러링한다. thread 로컬 manifest가 매 기록 후 원격 버전과 실제 호스트 metadata를
        저장한다. 이렇게 해야 수정 시각을 반올림하는 호스트 파일시스템에서 잘못된
        갱신 판정이 나지 않는다.

        원격 파일이 source of truth다. 크기나 수정 시각이 다르면 다음 sync가 호스트 쪽
        수정 내용을 덮어쓴다.

        실패는 WARNING으로 남기되 예외로 올리지 않는다. artifact 다운로드는 sandbox
        lifecycle에 치명적이지 않고, 하위 e2b SDK 오류는 다른 곳에서 이미 로깅한다.
        """
        from deerflow.config.paths import get_paths  # 순환 import를 피하기 위한 lazy import

        client = sandbox.client
        if client is None:
            logger.debug("Skip output sync: e2b client already closed for sandbox %s", sandbox.id)
            return

        home_dir = sandbox.home_dir.rstrip("/") or "/home/user"
        paths = get_paths()

        thread_dir = paths.thread_dir(thread_id, user_id=user_id)
        thread_root = thread_dir / "user-data"
        host_targets: dict[str, Path] = {sub: thread_root / sub for sub in self._SYNC_BACK_SUBDIRS}
        manifest_path = thread_dir / self._SYNC_MANIFEST_NAME
        remote_sandbox_id = sandbox.sandbox_id
        manifest, manifest_dirty = self._load_sync_manifest(manifest_path, remote_sandbox_id)

        # sync 대상 디렉터리의 모든 파일을 크기, 수정 시각, 경로와 함께 나열하는 셸 명령
        # 하나를 만든다. 특이한 파일명도 안전하게 파싱하도록 NUL로 구분한다. 미러링하는
        # 하위 디렉터리 수와 무관하게 왕복은 한 번으로 끝난다.
        #
        # 나열은 *물리* 경로인 /home/user 기준으로 한다(bootstrap symlink
        # /mnt/user-data -> /home/user가 투명하게 따라간다). 그리고 각 결과를
        # ``E2BSandbox.download_file`` 호출 전에 /mnt/user-data prefix로 되돌린다.
        # 이 메서드는 경로가 ``VIRTUAL_PATH_PREFIX``(/mnt/user-data) 아래인지 보안 검사를
        # 강제하고, 내부적으로 ``_resolve_path``로 다시 /home/user로 해석하기 때문이다.
        find_targets = " ".join(shlex.quote(f"{home_dir}/{sub}") for sub in self._SYNC_BACK_SUBDIRS)
        list_cmd = f'for d in {find_targets}; do   [ -d "$d" ] && find "$d" -type f -printf \'%s\\t%T@\\t%p\\0\' 2>/dev/null; done'

        try:
            result = client.commands.run(list_cmd)
        except Exception as e:
            logger.warning("e2b sync: list command failed: %s", e)
            if _is_sandbox_gone_error(e):
                with sandbox._lock:
                    sandbox._dead = True
            return

        stdout = getattr(result, "stdout", "") or ""
        if not stdout:
            if manifest or manifest_dirty:
                self._write_sync_manifest(manifest_path, remote_sandbox_id, {})
            return

        synced = 0
        skipped = 0
        seen_manifest_keys: set[str] = set()
        from .e2b_sandbox import _MAX_DOWNLOAD_SIZE

        # 이번 pass의 총량 예산(_MAX_SYNC_* 클래스 상수 참고).
        downloaded_bytes = 0
        downloaded_files = 0
        truncated_reason: str | None = None
        deadline = time.monotonic() + self._SYNC_DEADLINE_SECONDS

        for entry in stdout.split("\0"):
            if time.monotonic() >= deadline:
                truncated_reason = f"time budget {self._SYNC_DEADLINE_SECONDS}s"
                break
            entry = entry.strip()
            if not entry:
                continue
            try:
                size_str, remote_mtime_str, remote_path = entry.split("\t", 2)
                remote_size = int(size_str)
                remote_mtime_ns = int(Decimal(remote_mtime_str) * 1_000_000_000)
            except (InvalidOperation, ValueError):
                logger.debug("e2b sync: unparseable entry %r", entry)
                continue

            # 호스트 쪽 상대 경로를 계산하기 위해 이 파일이 어느 하위 디렉터리에
            # 속하는지 판별한다. remote_path는 절대 경로다.
            # 예: /home/user/outputs/foo/bar.pdf
            sub_match: tuple[str, Path, str] | None = None
            for sub, host_root in host_targets.items():
                prefix = f"{home_dir}/{sub}/"
                if remote_path == f"{home_dir}/{sub}":
                    continue
                if remote_path.startswith(prefix):
                    rel = remote_path[len(prefix) :]
                    virtual_path = f"/mnt/user-data/{sub}/{rel}"
                    sub_match = (sub, host_root / rel, virtual_path)
                    break
            if sub_match is None:
                continue
            _sub, host_path, virtual_path = sub_match
            manifest_key = host_path.relative_to(thread_root).as_posix()
            seen_manifest_keys.add(manifest_key)

            if remote_size > _MAX_DOWNLOAD_SIZE:
                logger.warning(
                    "e2b sync: skipping oversize artefact %s (%d bytes > %d cap)",
                    remote_path,
                    remote_size,
                    _MAX_DOWNLOAD_SIZE,
                )
                skipped += 1
                continue

            try:
                host_stat = host_path.stat()
                entry = manifest.get(manifest_key)
                if entry == {
                    "remote_size": remote_size,
                    "remote_mtime_ns": remote_mtime_ns,
                    "host_size": host_stat.st_size,
                    "host_mtime_ns": host_stat.st_mtime_ns,
                }:
                    skipped += 1
                    continue
            except OSError:
                pass

            if downloaded_files >= self._MAX_SYNC_FILES:
                truncated_reason = f"file count cap {self._MAX_SYNC_FILES}"
                break
            if downloaded_bytes + remote_size > self._MAX_SYNC_TOTAL_BYTES:
                truncated_reason = f"total byte budget {self._MAX_SYNC_TOTAL_BYTES}"
                break

            try:
                data = sandbox.download_file(virtual_path)
            except Exception as e:
                logger.warning(
                    "e2b sync: failed to download %s from sandbox %s: %s",
                    virtual_path,
                    sandbox.id,
                    e,
                )
                continue
            # 아래의 호스트 쪽 쓰기 성공 여부와 무관하게 다운로드를 예산에 반영한다.
            # 제한하려는 자원은 원격 왕복 그 자체다.
            downloaded_files += 1
            downloaded_bytes += remote_size

            try:
                host_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = host_path.with_name(host_path.name + ".e2bsync.tmp")
                tmp_path.write_bytes(data)
                os.utime(tmp_path, ns=(remote_mtime_ns, remote_mtime_ns))
                tmp_path.replace(host_path)
                host_stat = host_path.stat()
                manifest[manifest_key] = {
                    "remote_size": remote_size,
                    "remote_mtime_ns": remote_mtime_ns,
                    "host_size": host_stat.st_size,
                    "host_mtime_ns": host_stat.st_mtime_ns,
                }
                manifest_dirty = True
                synced += 1
            except OSError as e:
                logger.warning("e2b sync: failed to write %s on host: %s", host_path, e)

        # 잘린 pass는 모든 원격 파일을 확인하지 못했으므로 ``seen_manifest_keys``가
        # 불완전하다. 여기서 "stale" 항목을 정리하면 단지 도달하지 못했을 뿐인 파일을
        # 잊어버린다. 정리를 건너뛰고 다음 release가 처리하게 둔다
        # (새로 다운로드한 항목은 아래에서 그대로 기록된다).
        stale_keys = set(manifest) - seen_manifest_keys
        if stale_keys and truncated_reason is None:
            for key in stale_keys:
                manifest.pop(key)
            manifest_dirty = True

        if manifest_dirty:
            self._write_sync_manifest(manifest_path, remote_sandbox_id, manifest)

        if truncated_reason is not None:
            logger.warning(
                "e2b sync: sandbox=%s thread=%s truncated (%s); downloaded=%d files/%d bytes this pass, remaining artefacts deferred to next release",
                sandbox.id,
                thread_id,
                truncated_reason,
                downloaded_files,
                downloaded_bytes,
            )

        if synced or skipped:
            logger.info(
                "e2b sync: sandbox=%s thread=%s synced=%d skipped=%d",
                sandbox.id,
                thread_id,
                synced,
                skipped,
            )

    @staticmethod
    def _upload_tree(
        client: E2BClientSandbox,
        src: Path,
        dest_dir: str,
        read_only: bool,
    ) -> None:
        """``src``를 sandbox 안의 ``dest_dir``로 재귀적으로 업로드한다."""
        if src.is_file():
            target = f"{dest_dir}/{src.name}"
            with src.open("rb") as fh:
                client.files.write(target, fh.read())
            if read_only:
                try:
                    client.commands.run(f"chmod a-w {shlex.quote(target)}")
                except Exception:
                    pass
            return

        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src).as_posix()
            target = f"{dest_dir}/{rel}"
            try:
                make_dir = getattr(client.files, "make_dir", None)
                if callable(make_dir):
                    parent = target.rsplit("/", 1)[0]
                    if parent and parent != dest_dir:
                        make_dir(parent)
            except Exception:
                pass
            with path.open("rb") as fh:
                client.files.write(target, fh.read())
        if read_only:
            try:
                client.commands.run(f"chmod -R a-w {shlex.quote(dest_dir)}")
            except Exception:
                pass

    def _evict_oldest_warm(self) -> str | None:
        """가장 오래된 warm 항목을 evict한다. 그동안 transitioning slot을 점유한다.

        warm 항목을 꺼내면서 transitioning slot을 잡는다. control plane이 VM 소멸을
        확인할 때까지 slot은 계속 점유 상태로 남는다.
        """
        with self._lock:
            retryable = self._eviction_tombstones - self._evictions_in_progress
            if retryable:
                evict_id = next(iter(retryable))
                self._evictions_in_progress.add(evict_id)
            elif self._warm_pool:
                evict_id, (_, _) = self._warm_pool.popitem(last=False)
                self._eviction_tombstones.add(evict_id)
                self._evictions_in_progress.add(evict_id)
                self._begin_transition_locked()
            else:
                return None

        if not self._claim_ownership(evict_id, for_destroy=True):
            logger.info("Deferring warm-pool eviction for %s because a peer owns it", evict_id)
            self._forget_local_sandbox(evict_id)
            with self._lock:
                if evict_id in self._evictions_in_progress:
                    self._evictions_in_progress.discard(evict_id)
                    self._eviction_tombstones.discard(evict_id)
                    self._end_transition_locked()
            return evict_id
        try:
            client = self._reconnect_live_client(self._get_sandbox_cls(), evict_id)
        except Exception as e:
            logger.warning(
                "Evicted warm-pool e2b sandbox %s could not be reconnected for kill: %s",
                evict_id,
                e,
            )
            with self._lock:
                if evict_id in self._evictions_in_progress:
                    self._evictions_in_progress.discard(evict_id)
                    if not self._shutdown_called:
                        self._eviction_tombstones.add(evict_id)
            self._release_ownership(evict_id)
            return None

        if client is None:
            with self._lock:
                if evict_id in self._evictions_in_progress:
                    self._evictions_in_progress.discard(evict_id)
                    self._eviction_tombstones.discard(evict_id)
                    self._end_transition_locked()
            self._release_ownership(evict_id)
            logger.info("Evicted warm-pool e2b sandbox %s was already gone", evict_id)
            return evict_id

        if error := self._kill_client(client):
            logger.warning("Failed to kill evicted e2b sandbox %s: %s", evict_id, error)
            self._safe_close_client(client)
            with self._lock:
                if evict_id in self._evictions_in_progress:
                    self._evictions_in_progress.discard(evict_id)
                    if not self._shutdown_called:
                        self._eviction_tombstones.add(evict_id)
            self._release_ownership(evict_id)
            return None

        self._safe_close_client(client)
        with self._lock:
            if evict_id in self._evictions_in_progress:
                self._evictions_in_progress.discard(evict_id)
                self._eviction_tombstones.discard(evict_id)
                self._end_transition_locked()
        self._release_ownership(evict_id)
        logger.info("Evicted warm-pool e2b sandbox %s", evict_id)
        return evict_id

    def get(self, sandbox_id: str) -> Sandbox | None:
        with self._lock:
            return self._sandboxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        """클라우드 VM은 살려 둔 채 sandbox를 warm pool에 넣는다.

        e2b sandbox에는 서버가 강제하는 타임아웃이 있다. 여기서 이를 갱신해
        warm pool 항목이 release 이후에도 최소 ``idle_timeout`` 구간 동안 유효하게 한다.
        """
        with self._lock:
            thread_key = next(
                (key for key, sid in self._thread_sandboxes.items() if sid == sandbox_id),
                None,
            )

        if thread_key is None:
            self._release_internal(sandbox_id)
            return

        user_id, thread_id = thread_key
        with self._get_thread_lock(thread_id, user_id):
            self._release_internal(sandbox_id)

    def _release_internal(self, sandbox_id: str) -> None:
        """thread 전이 lock을 잡은 상태에서 release 하나를 완료한다.

        sandbox가 ``_sandboxes``에서 제거되는 순간 active slot은 *transitioning* slot이
        되고, output sync와 timeout 갱신 동안에도 계속 집계된다. 전이는 VM이 목적지에
        안착하거나 파기가 끝날 때 종료된다. shutdown 중이라면 ``_warm_pool``에 넣지 않고
        VM을 kill한다.
        """
        sandbox: E2BSandbox | None = None
        seed: str | None = None
        removed_keys: list[tuple[str, str]] = []
        transition_slot_held = False

        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            if sandbox is None:
                return
            self._begin_transition_locked()
            transition_slot_held = True
            removed_keys = [key for key, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for key in removed_keys:
                self._thread_sandboxes.pop(key, None)
            if removed_keys:
                user_id, thread_id = removed_keys[0]
                seed = self._stable_seed(thread_id, user_id)

        # E2BSandbox.close()는 client 참조를 지운다. release와 경쟁하는 shutdown도
        # 원격 VM을 kill할 수 있도록 이 참조를 미리 잡아 둔다.
        client = sandbox.client

        try:
            if sandbox.is_dead:
                logger.info(
                    "Releasing dead e2b sandbox %s; skipping output sync and warm pool, killing remote VM",
                    sandbox_id,
                )
                self._kill_and_close(sandbox)
                return

            sync_failed_due_to_dead_vm = False
            if seed is not None and removed_keys:
                user_id_sync, thread_id_sync = removed_keys[0]
                try:
                    self._sync_outputs_to_host(sandbox, thread_id=thread_id_sync, user_id=user_id_sync)
                except Exception as e:  # pragma: no cover - 방어적 처리
                    logger.warning(
                        "Failed to mirror e2b sandbox %s outputs to host: %s",
                        sandbox_id,
                        e,
                    )
                if sandbox.is_dead:
                    sync_failed_due_to_dead_vm = True

            if sync_failed_due_to_dead_vm:
                logger.info(
                    "Sandbox %s was reaped during release; not parking in warm pool",
                    sandbox_id,
                )
                self._kill_and_close(sandbox)
                return

            try:
                self._refresh_remote_timeout(client)
            except Exception as e:
                logger.debug("Failed to refresh timeout during release: %s", e)

            with self._lock:
                should_kill = self._shutdown_called
                if not should_kill:
                    self._warm_pool[sandbox_id] = (seed or "", time.time())
                    self._warm_pool.move_to_end(sandbox_id)
                    self._end_transition_locked()
                    transition_slot_held = False
                    logger.info("Released e2b sandbox %s to warm pool", sandbox_id)

            if should_kill:
                logger.info(
                    "Provider shut down during release of sandbox %s; killing instead of parking in warm pool",
                    sandbox_id,
                )
                if self._claim_ownership(sandbox_id, for_destroy=True):
                    if error := self._kill_client(client):
                        logger.debug("Failed to kill e2b sandbox %s during release: %s", sandbox_id, error)
                    self._release_ownership(sandbox_id)
                self._safe_close_client(client)
                return

            try:
                sandbox.close()
            except Exception as e:
                logger.warning("Error closing e2b sandbox %s during release: %s", sandbox_id, e)
        finally:
            if transition_slot_held:
                self._free_transitioning_slot()

    def _kill_and_close(self, sandbox: E2BSandbox) -> None:
        if not self._claim_ownership(sandbox.id, for_destroy=True):
            logger.info("Not killing E2B sandbox %s because a peer owns it", sandbox.id)
            try:
                sandbox.close()
            except Exception:
                pass
            return
        if error := self._kill_client(getattr(sandbox, "_client", None)):
            logger.debug(
                "kill() on e2b sandbox %s raised (probably already gone): %s",
                sandbox.id,
                error,
            )
        self._release_ownership(sandbox.id)
        try:
            sandbox.close()
        except Exception:
            pass

    def _kill_client(
        self,
        client: E2BClientSandbox | None,
    ) -> Exception | None:
        """원격 VM을 kill하고, 호출자가 로깅할 수 있도록 예외를 반환한다."""
        if client is None:
            return RuntimeError("Cannot confirm remote VM destruction without a client")
        sandbox_id = getattr(client, "sandbox_id", None)
        try:
            kill = getattr(client, "kill", None)
            if not callable(kill):
                return RuntimeError("Cannot confirm remote VM destruction without a callable kill method")
            kill()
        except Exception as e:
            return e
        if sandbox_id is not None:
            self._release_deployment_sandbox(sandbox_id)
        return None

    def reset(self) -> None:
        """추적 중인 E2B VM을 파기하고, 분리된 이 provider를 사용 불가 상태로 만든다."""
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
        self._maintenance_stop.set()
        for thread in (self._lease_thread, self._reconcile_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(5.0, float(self._config["reconciliation_max_seconds"]) + 1.0))
        with self._lock:
            active = list(self._sandboxes.items())
            warm_ids = list(self._warm_pool.keys() | self._eviction_tombstones | self._remote_ops_in_progress)
            owned_ids = set(self._owned_sandbox_ids)
            self._sandboxes.clear()
            self._warm_pool.clear()
            self._eviction_tombstones.clear()
            self._evictions_in_progress.clear()
            self._remote_ops_in_progress.clear()
            self._unowned_remote_ops_in_progress.clear()
            self._thread_sandboxes.clear()
            self._thread_locks.clear()
            self._owned_sandbox_ids.clear()
            self._acquire_inflight.clear()
            self._orphan_first_seen.clear()
            self._reserved_slots = 0
            self._transitioning_slots = 0
            self._capacity_cond.notify_all()

        executor = getattr(self, "_acquire_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        logger.info(
            "Shutting down E2BSandboxProvider: %d active + %d warm sandboxes",
            len(active),
            len(warm_ids),
        )

        for sandbox_id, sandbox in active:
            if sandbox_id in owned_ids and self._claim_ownership(sandbox_id, for_destroy=True):
                if error := self._kill_client(sandbox.client):
                    logger.warning(
                        "Failed to kill active e2b sandbox %s during shutdown: %s",
                        sandbox_id,
                        error,
                    )
                self._release_ownership(sandbox_id)
            try:
                sandbox.close()
            except Exception:
                pass

        sandbox_cls = self._get_sandbox_cls()
        for sandbox_id in warm_ids:
            if sandbox_id not in owned_ids:
                continue
            if not self._claim_ownership(sandbox_id, for_destroy=True):
                continue
            try:
                client = self._reconnect_client(sandbox_cls, sandbox_id)
            except Exception as e:
                logger.warning(
                    "Failed to reconnect warm-pool e2b sandbox %s for shutdown: %s",
                    sandbox_id,
                    e,
                )
                self._release_ownership(sandbox_id)
                continue
            if error := self._kill_client(client):
                logger.warning(
                    "Failed to kill warm-pool e2b sandbox %s during shutdown: %s",
                    sandbox_id,
                    error,
                )
            self._release_ownership(sandbox_id)
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        try:
            self._ownership.close()
        except Exception as e:
            logger.warning("Failed to close E2B ownership store: %s", e)
        if self._deployment_capacity is not None:
            try:
                self._deployment_capacity.close()
            except Exception as e:
                logger.warning(
                    "Failed to close E2B deployment capacity store: %s",
                    e,
                )
