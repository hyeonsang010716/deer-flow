"""``TenkiSandboxProvider`` — Tenki를 백엔드로 쓰는 DeerFlow :class:`SandboxProvider`.

`Tenki <https://tenki.cloud>`_ 클라우드 sandbox를 DeerFlow sandbox 백엔드로 통합한다.
각 sandbox는 표준 base image에서 만들어진 격리된 클라우드 microVM이다. provider는
``(user, thread)`` 마다 하나를 만들어 프로세스 안에서 재사용하고, 반납된 sandbox는
빠른 재확보를 위해 warm pool(공용 :class:`WarmPoolLifecycleMixin` 기반)에 넣어 둔다.

설정은 :class:`SandboxConfig`(``extra="allow"``)에서 읽으므로, 모델에 선언되지 않은
Tenki 키도 ``config.yaml`` 의 ``sandbox:`` 아래에 둘 수 있다. 전체 목록은 이 패키지의
``__init__`` docstring을 참고한다.

Tenki SDK는 지연 import한다(``_import_client``). 그래야 harness와 다른 provider들이
``tenki-sandbox`` 없이도 설치된다. 이 provider를 선택했을 때만 의존성이 필요하다.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import shlex
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from deerflow.config import get_app_config
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.sandbox_provider import SandboxProvider

from ..warm_pool_lifecycle import WarmPoolLifecycleMixin
from .sandbox import DEFAULT_TENKI_HOME_DIR, TenkiSandbox

if TYPE_CHECKING:
    from tenki_sandbox import Client

logger = logging.getLogger(__name__)

_SANDBOX_NAME_PREFIX = "deer-flow-tenki-"
# Tenki는 최대 수명(기본 약 30분)이 지나면 sandbox를 종료하는데, 그러면 오래 도는 thread의
# 상태가 대화 중간에 조용히 사라진다. 여기서는 DeerFlow가 lifecycle을 소유하고
# warm pool의 idle_timeout이 미사용 sandbox를 회수하므로, 리서치 실행보다 넉넉히 긴 수명을 요청한다.
# sandbox.max_duration(초)으로 덮어쓸 수 있고, 0이면 Tenki 계정 기본값을 쓴다.
DEFAULT_MAX_DURATION = 4 * 60 * 60
# bootstrap은 best-effort이고 scope별 acquire lock을 쥔 채 실행되므로 exec에 상한을 둔다.
# 여기서 멈추면(예: sudo가 걸리면) 해당 scope의 acquire가 무한정 지연되고 이후 acquire가 뒤에 쌓인다.
# timeout이 나면 기존 경고 경로로 떨어진다.
_BOOTSTRAP_TIMEOUT = 30.0


def _bootstrap_script(home_dir: str) -> str:
    """sandbox 안에 DeerFlow의 virtual path 레이아웃을 만든다.

    Tenki sandbox는 권한 없는 ``tenki`` 사용자로 돌고 ``/mnt`` 는 root 소유라서
    ``mkdir -p /mnt/user-data/...`` 가 Permission denied로 실패한다. ``community/e2b_sandbox``
    와 마찬가지로 쓰기 가능한 HOME 아래에 실제 디렉터리를 만들고, 문서화된 ``/mnt/...``
    경로를 쓰는 명령도 동작하도록 ``sudo`` 로 ``/mnt/user-data`` symlink를 best-effort로 건다.
    sudo가 없으면 symlink 단계는 건너뛴다. 파일 API는 :meth:`TenkiSandbox._resolve_path`
    의 home remap 덕분에 계속 동작한다.

    ``sudo -n``(비대화형)은 의도적이다. tenki 사용자의 sudoers 항목이 비밀번호를 요구하면
    tty를 붙이는 exec에서 맨 ``sudo`` 는 비밀번호 프롬프트에서 멈추고, 이 호출의 exec timeout과
    맞물려 acquire 전체를 지연시킨다. ``-n`` 은 즉시 실패하고 기존 ``|| true`` 가 이를 삼킨다.
    """
    home = shlex.quote(home_dir)
    return (
        f"mkdir -p {home}/workspace {home}/uploads {home}/outputs; "
        f"if command -v sudo >/dev/null 2>&1; then "
        f"  if [ ! -e /mnt/user-data ] || [ -L /mnt/user-data ]; then "
        f"    sudo -n ln -sfn {home} /mnt/user-data 2>/dev/null || true; "
        f"  fi; "
        f"fi; "
        f"echo BOOTSTRAP_OK"
    )


def _import_client() -> type[Client]:
    """Tenki의 ``Client`` 를 지연 import한다.

    모듈 import 시점에서 빼두어야 harness와 다른 provider들이 Tenki 없이도 설치된다.
    이 provider를 선택했을 때만 의존성이 필요하다.
    """
    try:
        from tenki_sandbox import Client
    except ImportError as e:  # pragma: no cover - depends on the optional dependency
        raise ImportError("TenkiSandboxProvider requires the optional 'tenki-sandbox' dependency. Install it with: pip install 'deerflow-harness[tenki]' or pip install tenki-sandbox.") from e
    return Client


class TenkiSandboxProvider(WarmPoolLifecycleMixin[TenkiSandbox], SandboxProvider):
    """DeerFlow sandbox 하나를 Tenki 클라우드 microVM 하나로 실행한다."""

    uses_thread_data_mounts = False
    needs_upload_permission_adjustment = True
    _idle_checker_thread_name = "tenki-idle-reaper"

    @staticmethod
    def _sandbox_id(thread_id: str, user_id: str) -> str:
        """user/thread scope에서 결정적으로 만든 sandbox ID.

        user_id를 포함하므로 한 사용자의 bucket용 sandbox를 같은 thread_id를 가진 다른
        사용자의 thread가 재확보할 수 없다. warm pool은 이 id만으로 키를 잡기 때문에
        (``_reclaim_warm_pool`` 이 full-seed fallback 없이 직접 조회한다) 멀티테넌트
        호스팅 gateway에서 해시 충돌이 나면 남의 대기 sandbox를 가져갈 수 있다.
        community/e2b_sandbox와 동일하게 64비트를 써서 그 확률을 무시할 수준으로 낮춘다.
        """
        return hashlib.sha256(f"{user_id}:{thread_id}".encode()).hexdigest()[:16]

    # ── Provider 생명주기 ─────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sandboxes: dict[str, TenkiSandbox] = {}
        self._thread_sandboxes: dict[tuple[str, str], str] = {}
        self._warm_pool: dict[str, tuple[TenkiSandbox, float]] = {}
        self._acquire_locks: dict[str, threading.Lock] = {}
        self._idle_checker_stop = threading.Event()
        self._idle_checker_thread: threading.Thread | None = None
        self._shutdown_called = False
        self._client: Client | None = None
        self._config = self._load_config()
        atexit.register(self.shutdown)
        self._start_idle_checker()

    def _load_config(self) -> dict[str, Any]:
        sandbox_config = get_app_config().sandbox

        def _opt(name: str, default: Any = None) -> Any:
            return getattr(sandbox_config, name, default)

        api_key = _opt("api_key")
        replicas = _opt("replicas")
        idle_timeout = _opt("idle_timeout")
        max_duration = _opt("max_duration")
        environment = dict(_opt("environment") or {})
        # 잘못된 키(예: "bad-key")는 즉시 실패시킨다. 호출별 env는 execute_command에서
        # 같은 POSIX 이름 검사를 거치지만, 이 정적 설정 env는 모든 명령에 병합되므로
        # 검사하지 않으면 create/exec 시점에 알아보기 힘든 SDK 에러로만 드러난다.
        _validate_extra_env(environment)
        return {
            "max_duration": float(max_duration if max_duration is not None else DEFAULT_MAX_DURATION),
            # 기본은 off(SDK 기본값). warm pool sandbox는 turn 사이에도 계속 떠 있으므로
            # host pinning은 pause/resume까지 쓰는 배포에서만 의미가 있다.
            # 그래서 값을 정해주지 않고 옵션으로 노출만 한다.
            "sticky": bool(_opt("sticky", False)),
            "api_key": api_key,  # None이면 SDK가 TENKI_API_KEY / TENKI_AUTH_TOKEN으로 fallback
            "base_url": _opt("base_url"),
            "image": _opt("image"),  # None이면 Tenki 계정 기본 base image
            "home_dir": _opt("home_dir") or DEFAULT_TENKI_HOME_DIR,
            "project_id": _opt("project_id"),
            "workspace_id": _opt("workspace_id"),
            "cpu_cores": _opt("cpu_cores"),
            "memory_mb": _opt("memory_mb"),
            "environment": environment,
            "replicas": replicas if replicas is not None else self.DEFAULT_REPLICAS,
            "idle_timeout": idle_timeout if idle_timeout is not None else self.DEFAULT_IDLE_TIMEOUT,
        }

    def _get_client(self) -> Client:
        with self._lock:
            if self._client is not None:
                return self._client
        client_cls = _import_client()
        client = client_cls(auth_token=self._config["api_key"], base_url=self._config["base_url"])
        with self._lock:
            if self._client is None:
                self._client = client
            return self._client

    def _resolve_scope(self) -> tuple[str | None, str | None]:
        """(project_id, workspace_id)를 반환한다. 모호하지 않으면 자동 선택한다.

        Tenki의 ``create`` 는 project scope를 요구한다. 설정에 값이 없으면 계정에
        workspace와 project가 정확히 하나씩일 때만 그것을 고르고, 아니면 선택지를 담아
        예외를 던져 운영자가 ``project_id`` 를 지정하게 한다.
        """
        project_id = self._config["project_id"]
        workspace_id = self._config["workspace_id"]
        if project_id is not None:
            return project_id, workspace_id

        identity = self._get_client().who_am_i()
        workspaces = list(identity.workspaces or [])
        if workspace_id is not None:
            workspaces = [w for w in workspaces if w.id == workspace_id]
        workspace = self._require_single(workspaces, "workspace", "workspace_id")
        project = self._require_single(list(workspace.projects or []), "project", "project_id")
        return project.id, workspace.id

    @staticmethod
    def _require_single(items: list[Any], kind: str, param: str) -> Any:
        if len(items) != 1:
            raise ValueError(f"Could not auto-select a Tenki {kind}; set sandbox.{param} in config.yaml. Found: {[(getattr(i, 'id', None), getattr(i, 'name', None)) for i in items]}")
        return items[0]

    @staticmethod
    def _thread_key(thread_id: str, user_id: str | None) -> tuple[str, str]:
        return (user_id or "", thread_id)

    @classmethod
    def _sandbox_name(cls, sandbox_id: str) -> str:
        return f"{_SANDBOX_NAME_PREFIX}{sandbox_id}"

    def _lock_for_sandbox(self, sandbox_id: str) -> threading.Lock:
        with self._lock:
            lock = self._acquire_locks.get(sandbox_id)
            if lock is None:
                lock = threading.Lock()
                self._acquire_locks[sandbox_id] = lock
            return lock

    def _start_idle_checker(self) -> None:
        """활성화된 경우에만 idle 정리를 시작한다. idle_timeout=0이면 비활성 상태로 둔다."""
        if self._config["idle_timeout"] <= 0:
            return
        super()._start_idle_checker()

    def _active_count_locked(self) -> int:
        return len(self._sandboxes)

    def _destroy_warm_entry(self, sandbox_id: str, entry: TenkiSandbox, *, reason: str) -> None:
        self._close_quietly(entry, context=f"warm pool, reason={reason}")

    def _invalidate_sandbox(self, sandbox_id: str, reason: str) -> None:
        """명령 경로에서 복구 불가능한 실패가 나면 sandbox를 파괴하고 등록을 해제한다."""
        to_close: TenkiSandbox | None = None
        with self._lock:
            active = self._sandboxes.pop(sandbox_id, None)
            warm_entry = self._warm_pool.pop(sandbox_id, None)
            for key in [k for k, sid in self._thread_sandboxes.items() if sid == sandbox_id]:
                self._thread_sandboxes.pop(key, None)
            to_close = active or (warm_entry[0] if warm_entry is not None else None)

        if to_close is None:
            logger.warning("Tenki sandbox %s failed terminally but was not tracked: %s", sandbox_id, reason)
            return
        logger.warning("Invalidating Tenki sandbox %s after terminal failure: %s", sandbox_id, reason)
        self._close_quietly(to_close, context="terminal failure")

    # ── 확보(acquire) / 반납(release) ──────────────────────────────────────

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        if thread_id is None:
            sandbox_id = str(uuid.uuid4())[:8]
            sandbox = self._create_sandbox(sandbox_id)
            with self._lock:
                self._sandboxes[sandbox.id] = sandbox
            return sandbox.id

        key = self._thread_key(thread_id, user_id)
        sandbox_id = self._sandbox_id(thread_id, user_id or "")
        acquire_lock = self._lock_for_sandbox(sandbox_id)
        with acquire_lock:
            with self._lock:
                existing = self._thread_sandboxes.get(key)
                if existing is not None and existing in self._sandboxes:
                    return existing

            reclaimed = self._reclaim_warm_pool(sandbox_id)
            if reclaimed is not None:
                with self._lock:
                    self._thread_sandboxes[key] = reclaimed
                return reclaimed

            sandbox = self._create_sandbox(sandbox_id)
            with self._lock:
                self._sandboxes[sandbox.id] = sandbox
                self._thread_sandboxes[key] = sandbox.id
            return sandbox.id

    def _create_sandbox(self, sandbox_id: str) -> TenkiSandbox:
        # replica soft cap을 적용한다. active + warm이 한도에 닿으면 가장 오래된 warm sandbox를 evict한다.
        replicas, total = self._replica_count()
        if total >= replicas:
            evicted = self._evict_oldest_warm()
            self._log_replicas_soft_cap(replicas, sandbox_id, evicted)

        client = self._get_client()
        project_id, workspace_id = self._resolve_scope()
        create_kwargs: dict[str, Any] = {
            "name": self._sandbox_name(sandbox_id),
            "project_id": project_id,
            "workspace_id": workspace_id,
            "sticky": self._config["sticky"],
            # readiness 대기는 create() 안이 아니라 아래에서 직접 한다.
            # create(wait=True)는 session handle이 아직 SDK 내부에만 있는 상태로 예외를 던지므로,
            # readiness 실패 시 이 provider가 영영 보지도 종료하지도 못하는
            # 과금 중인 microVM이 새어 나간다.
            "wait": False,
        }
        if self._config["max_duration"] > 0:
            create_kwargs["max_duration"] = self._config["max_duration"]
        for key in ("image", "cpu_cores", "memory_mb"):
            if self._config[key] is not None:
                create_kwargs[key] = self._config[key]
        if self._config["environment"]:
            create_kwargs["env"] = self._config["environment"]

        remote = client.create(**create_kwargs)
        try:
            remote.wait_ready()
        except Exception:
            self._terminate_orphan(sandbox_id, remote)
            raise

        # 쓰기 가능한 HOME 아래에 DeerFlow의 virtual path 레이아웃을 만든다.
        # best-effort다. 실패해도 파일 API는 home remap으로 계속 동작한다.
        try:
            result = remote.exec("sh", "-lc", _bootstrap_script(self._config["home_dir"]), timeout=_BOOTSTRAP_TIMEOUT)
            if result.exit_code not in (0, None) or "BOOTSTRAP_OK" not in (result.stdout_text or ""):
                logger.warning(
                    "Tenki bootstrap for %s exited code=%s stderr=%s",
                    sandbox_id,
                    result.exit_code,
                    (result.stderr_text or "").strip(),
                )
        except Exception as e:
            logger.warning("Tenki bootstrap for %s raised: %s", sandbox_id, e)
        logger.info("Created Tenki sandbox %s (name=%s)", sandbox_id, self._sandbox_name(sandbox_id))
        return TenkiSandbox(
            sandbox_id,
            remote,
            default_env=self._config["environment"],
            home_dir=self._config["home_dir"],
            on_terminal_failure=self._invalidate_sandbox,
        )

    @staticmethod
    def _terminate_orphan(sandbox_id: str, remote: Any) -> None:
        """생성됐지만 adapter에 전달되지 못한 microVM을 종료한다."""
        try:
            remote.close()
            logger.warning("Terminated Tenki sandbox %s after it failed to become ready", sandbox_id)
        except Exception as e:
            logger.error("Leaked Tenki sandbox %s (id=%s): could not terminate after readiness failure: %s", sandbox_id, getattr(remote, "id", "?"), e)

    @staticmethod
    def _close_quietly(sandbox: TenkiSandbox, *, context: str) -> None:
        """호출자가 실패에 대응할 방법이 없는 자리에서 sandbox를 닫는다."""
        try:
            sandbox.close()
        except Exception as e:
            logger.warning("Error closing Tenki sandbox %s (%s): %s", sandbox.id, context, e)

    def get(self, sandbox_id: str) -> Sandbox | None:
        with self._lock:
            return self._sandboxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        """sandbox를 warm pool로 반납한다. microVM은 계속 실행 상태로 둔다.

        sandbox는 ``_sandboxes`` 에서 ``_warm_pool`` 로 옮겨지고 ``_thread_sandboxes``
        항목은 정리된다. 이미 shutdown이 시작된 경우가 아니면 종료하지 않는다.
        """
        close_sandbox: TenkiSandbox | None = None
        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            for key in [k for k, sid in self._thread_sandboxes.items() if sid == sandbox_id]:
                self._thread_sandboxes.pop(key, None)
            if sandbox is None:
                return
            if self._shutdown_called:
                close_sandbox = sandbox
            else:
                self._warm_pool[sandbox_id] = (sandbox, time.time())

        if close_sandbox is not None:
            self._close_quietly(close_sandbox, context="released during shutdown")
            logger.info("Closed released Tenki sandbox %s because shutdown is in progress", sandbox_id)
        else:
            logger.info("Released Tenki sandbox %s to warm pool (microVM still running)", sandbox_id)

    def _reclaim_warm_pool(self, sandbox_id: str) -> str | None:
        """liveness health check를 거친 뒤 id로 warm pool sandbox를 재확보한다.

        성공하면 sandbox_id를, 항목이 없거나 health check가 실패하면 None을 반환한다.
        죽은 항목은 파괴한다.
        """
        with self._lock:
            if sandbox_id not in self._warm_pool:
                return None
            sandbox, _ = self._warm_pool[sandbox_id]

        try:
            result = sandbox.execute_command("echo ok", timeout=10)
            healthy = "ok" in result
        except Exception as e:
            logger.warning("Tenki warm-pool sandbox %s health check error: %s", sandbox_id, e)
            healthy = False

        if not healthy:
            with self._lock:
                warm_entry = self._warm_pool.pop(sandbox_id, None)
            if warm_entry is not None:
                self._destroy_warm_entry(sandbox_id, warm_entry[0], reason="health_check_failed")
            return None

        with self._lock:
            warm_entry = self._warm_pool.pop(sandbox_id, None)
            if warm_entry is None:
                return None  # 다른 thread와 경합해서 밀렸다
            self._sandboxes[sandbox_id] = warm_entry[0]

        logger.info("Reclaimed warm-pool Tenki sandbox %s", sandbox_id)
        return sandbox_id

    def reset(self) -> None:
        """추적 중인 sandbox를 이 인스턴스의 warm pool에 넣어 나중에 정리되게 한다.

        ``reset_sandbox_provider()`` 는 싱글턴을 버리고 이 메서드를 호출해서 다음 생성 때
        설정 변경이 반영되게 한다. 실제 teardown은 ``shutdown()`` 의 몫이다. reset은 실행 중인
        microVM을 살려두되 고아로 만들지 않고 이 인스턴스의 idle reaper와 atexit shutdown에
        계속 보이게 한다.
        """
        with self._lock:
            now = time.time()
            for sandbox_id, sandbox in self._sandboxes.items():
                self._warm_pool.setdefault(sandbox_id, (sandbox, now))
            self._sandboxes.clear()
            self._thread_sandboxes.clear()
            self._acquire_locks.clear()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        self._stop_idle_checker()

        with self._lock:
            active = list(self._sandboxes.values())
            warm = [sandbox for sandbox, _ in self._warm_pool.values()]
            self._sandboxes.clear()
            self._warm_pool.clear()
            self._thread_sandboxes.clear()
            self._acquire_locks.clear()

        for sandbox in active + warm:
            self._close_quietly(sandbox, context="shutdown")
