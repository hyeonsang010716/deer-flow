import logging
import threading
from collections import OrderedDict
from pathlib import Path

from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

# ``local_sandbox_provider._singleton``에 직접 접근하는 예전 caller/테스트와의 하위
# 호환을 위해 남겨 둔 모듈 수준 alias. 새 코드는 provider 인스턴스 속성
# (``_generic_sandbox`` / ``_thread_sandboxes``)을 읽는다.
_singleton: LocalSandbox | None = None

# ``acquire``에서 생성되는 per-thread mapping이 예약해야 하는 가상 prefix.
# ``config.yaml``의 custom mount는 이들과 겹칠 수 없다.
_USER_DATA_VIRTUAL_PREFIX = "/mnt/user-data"
_ACP_WORKSPACE_VIRTUAL_PREFIX = "/mnt/acp-workspace"

# 메모리에 유지하는 per-thread LocalSandbox 인스턴스 수의 기본 상한.
# 캐시된 인스턴스 하나는 저렴하지만(PathMapping 리스트와 reverse resolve에 쓰는
# agent 작성 경로 집합을 가진 작은 Python 객체), 장수 gateway에서는 서로 다른
# thread_id 수가 무한하다. 상한을 넘으면 least-recently-used 항목을 버린다. 해당
# thread의 다음 ``acquire(thread_id)``는 sandbox를 다시 만들 뿐이며, 그 대가로 누적된
# ``_agent_written_paths``를 잃는다(read_file은 reverse resolution 없이 동작하며,
# 이는 새로 시작한 run과 동일한 동작이다).
DEFAULT_MAX_CACHED_THREAD_SANDBOXES = 256


class LocalSandboxProvider(SandboxProvider):
    """per-thread 경로 스코핑을 갖는 로컬 파일시스템 sandbox provider.

    이전 버전의 이 provider는 리터럴 id ``"local"``로 식별되는 프로세스 전역
    ``LocalSandbox`` 하나를 반환했다. 그 singleton은 공개 ``Sandbox`` API 경계에서
    문서화된 ``/mnt/user-data/...`` 계약을 지킬 수 없었다. 대응하는 host 디렉터리가
    thread별(``{base_dir}/users/{user_id}/threads/{thread_id}/user-data/``)이기 때문이다.

    이제 provider는 ``thread_id``마다 새 ``LocalSandbox``를 만들고, 그 ``path_mappings``에
    ``/mnt/user-data/{workspace,uploads,outputs}``와 ``/mnt/acp-workspace``의 thread 스코프
    항목을 포함한다. :class:`AioSandboxProvider`가 그 경로들을 docker 컨테이너에
    bind-mount하는 방식과 같다. 레거시 ``acquire()`` / ``acquire(None)`` 호출은 thread
    context가 없는 caller(및 테스트)를 위해 여전히 id ``"local"``인 일반 singleton을
    반환한다.

    Thread-safety: ``acquire``, ``get``, ``reset``은 여러 thread에서 호출될 수 있으므로
    (Gateway tool dispatch, subagent worker pool, 백그라운드 memory updater 등) 모든 cache
    상태 변경은 provider 전역 :class:`threading.Lock`으로 직렬화한다.
    :class:`AioSandboxProvider`와 같은 패턴이다.

    메모리 상한: ``_thread_sandboxes``는 ``max_cached_threads``
    (기본값 :data:`DEFAULT_MAX_CACHED_THREAD_SANDBOXES`)로 제한된 LRU cache다. 상한을
    넘으면 다음 ``acquire``에서 least-recently-used 항목이 evict된다. evict된 thread의
    다음 ``acquire``는 새 sandbox를 다시 만들며, 잃는 것은 ``_agent_written_paths``
    reverse-resolve 힌트뿐이라 read_file 출력이 완만하게 degrade된다.
    """

    uses_thread_data_mounts = True
    needs_upload_permission_adjustment = False

    def __init__(self, max_cached_threads: int = DEFAULT_MAX_CACHED_THREAD_SANDBOXES):
        """정적 path mapping으로 로컬 sandbox provider를 초기화한다.

        Args:
            max_cached_threads: LRU cache에 유지하는 per-thread sandbox 수의 상한.
                초과하면 다음 ``acquire``에서 least-recently-used 항목이 evict된다.
        """
        self._path_mappings = self._setup_path_mappings()
        self._generic_sandbox: LocalSandbox | None = None
        self._thread_sandboxes: OrderedDict[tuple[str, str], LocalSandbox] = OrderedDict()
        self._max_cached_threads = max_cached_threads
        self._lock = threading.Lock()

    def _setup_path_mappings(self) -> list[PathMapping]:
        """이 provider가 만드는 모든 sandbox가 공유하는 정적 path mapping을 구성한다.

        정적 mapping은 **public** skills 디렉터리와 ``config.yaml``의 custom mount를
        다룬다. 둘 다 프로세스 전역이며 모든 thread에서 동일하다. thread별
        ``/mnt/user-data/...``, ``/mnt/acp-workspace``, ``/mnt/skills/custom`` mapping은
        ``thread_id``와 유효 ``user_id``에 의존하므로 :meth:`_build_thread_path_mappings`
        안에서 추가한다.

        Returns:
            정적 path mapping 리스트
        """
        mappings: list[PathMapping] = []

        # skills mapping: public + legacy + custom으로 나눠 mount한다
        try:
            from deerflow.config import get_app_config

            config = get_app_config()
            container_path = config.skills.container_path
            projection = self._ensure_skills_projection()

            # public skills: 전역, read-only — 정적이며 모든 thread가 공유한다
            public_skills_path = projection.public
            if public_skills_path.exists():
                mappings.append(
                    PathMapping(
                        container_path=f"{container_path}/public",
                        local_path=str(public_skills_path),
                        read_only=True,
                    )
                )

            # NOTE: legacy skills mount는 여기 포함하지 않는다. 아직 per-user custom
            # skill이 없는 사용자에게만 노출되어야 하기 때문이다(그런 사용자에게만
            # SkillCategory.LEGACY를 노출하는 ``UserScopedSkillStorage._iter_skill_files``와
            # 동일). 모든 사용자에게 포함하면 per-user custom skill이 있는 사용자도
            # ``read_file("/mnt/skills/legacy/<name>/SKILL.md")``로 목록 계층이 없다고 한
            # 내용을 읽을 수 있다. PR #3889 리뷰 피드백 참고 — legacy mount는 이제
            # user_id를 알게 된 뒤 ``_build_thread_path_mappings``에서 만든다.

            # NOTE: custom skills mount도 여기 포함하지 않는다. per-user이므로
            # ``_build_thread_path_mappings`` 안에서 thread마다 동적으로 만들어야 한다.
            # 이전에 init 시점의 ``get_effective_user_id()``를 묶어 두던 정적 mount는
            # 잘못이었다. 이후 모든 사용자의 sandbox가 ``/mnt/skills/custom``을 init 시점
            # 사용자의 디렉터리로 resolve하게 되기 때문이다.

            # sandbox config의 custom mount 매핑
            _RESERVED_CONTAINER_PREFIXES = [
                f"{container_path}/public",
                f"{container_path}/custom",
                f"{container_path}/integrations",
                f"{container_path}/legacy",
                _ACP_WORKSPACE_VIRTUAL_PREFIX,
                _USER_DATA_VIRTUAL_PREFIX,
            ]
            sandbox_config = config.sandbox
            if sandbox_config and sandbox_config.mounts:
                for mount in sandbox_config.mounts:
                    host_path = Path(mount.host_path)
                    container_path = mount.container_path.rstrip("/") or "/"

                    if not host_path.is_absolute():
                        logger.warning(
                            "Mount host_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        continue

                    if not container_path.startswith("/"):
                        logger.warning(
                            "Mount container_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        continue

                    # 예약된 container 경로와 충돌하는 mount는 거부한다
                    if any(container_path == p or container_path.startswith(p + "/") for p in _RESERVED_CONTAINER_PREFIXES):
                        logger.warning(
                            "Mount container_path conflicts with reserved prefix, skipping: %s",
                            mount.container_path,
                        )
                        continue
                    # mapping을 추가하기 전에 host 경로가 존재하는지 확인한다.
                    #
                    # ``host_path``는 이 provider를 실행하는 프로세스의 파일시스템을
                    # 기준으로 resolve된다. ``make dev``에서는 host 머신이지만
                    # ``make up``에서는 ``deer-flow-gateway`` 컨테이너이므로, gateway
                    # 이미지에 bind-mount되지 않은 host 경로는 여기서 존재하지 않는다.
                    # 조용히 건너뛰면 디버깅 비용이 큰 silent failure가 된다(sandbox
                    # skill/tool이 설정된 mount 대신 빈 디렉터리를 읽는다). 그래서
                    # ERROR로 올리고 조치 가능한 안내를 함께 남긴다. #3244 참고.
                    if host_path.exists():
                        mappings.append(
                            PathMapping(
                                container_path=container_path,
                                local_path=str(host_path.resolve()),
                                read_only=mount.read_only,
                            )
                        )
                    else:
                        logger.error(
                            "sandbox.mounts entry %s -> %s ignored: host_path %s does not exist from the "
                            "perspective of the gateway process. In Docker deployments (make up / docker-compose), "
                            "this path must also be bind-mounted into the gateway container — add a matching "
                            "volume entry under services.gateway.volumes in docker/docker-compose.yaml (and use "
                            "the in-container path here), or run in local mode (make dev) where the gateway sees "
                            "the host filesystem directly.",
                            mount.host_path,
                            mount.container_path,
                            mount.host_path,
                        )
        except Exception as e:
            # config 로딩이 실패해도 실패시키지 않고 로그만 남긴다
            logger.warning("Could not setup path mappings: %s", e, exc_info=True)

        return mappings

    @staticmethod
    def _effective_acquire_user_id(user_id: str | None) -> str:
        from deerflow.runtime.user_context import get_effective_user_id

        return user_id or get_effective_user_id()

    @staticmethod
    def _thread_key(thread_id: str, user_id: str) -> tuple[str, str]:
        return (user_id, thread_id)

    @staticmethod
    def _ensure_skills_projection(user_id: str | None = None):
        """Best-effort: projection 실패가 sandbox acquire를 실패시켜서는 안 된다.

        주변 skill-mount 설정과 동일한 방식이다. 그쪽도 acquire 전체를 실패시키는 대신
        늘 로그를 남기고 계속 진행해 왔다(예: test double에 config.yaml이 없는 경우).
        caller는 ``None``을 받고 이번 acquire의 skill mount를 건너뛴다. 근본 조건이
        해소되면 이후 acquire에서 projection이 스스로 복구된다.
        """
        from deerflow.config import get_app_config
        from deerflow.skills.projection import ensure_skill_projections
        from deerflow.skills.storage import get_or_new_skill_storage, get_or_new_user_skill_storage

        try:
            config = get_app_config()
            if user_id is None:
                storage = get_or_new_skill_storage(app_config=config)
            else:
                storage = get_or_new_user_skill_storage(user_id, app_config=config)
            return ensure_skill_projections(storage)
        except Exception as exc:
            logger.warning("Could not ensure skills projection for user %s: %s", user_id, exc, exc_info=True)
            return None

    @staticmethod
    def _append_public_skill_mapping(mappings: list[PathMapping], projection) -> None:
        if projection is None:
            return
        try:
            from deerflow.config import get_app_config

            container_path = get_app_config().skills.container_path.rstrip("/")
            public_container_path = f"{container_path}/public"
            if any(mapping.container_path.rstrip("/") == public_container_path for mapping in mappings):
                return
            mappings.append(
                PathMapping(
                    container_path=public_container_path,
                    local_path=str(projection.public),
                    read_only=True,
                )
            )
        except Exception as exc:
            logger.warning("Could not append public skill mapping: %s", exc, exc_info=True)

    @staticmethod
    def _sandbox_id_for_thread(thread_id: str, user_id: str) -> str:
        return f"local:{user_id}:{thread_id}"

    @staticmethod
    def _key_from_sandbox_id(sandbox_id: str) -> tuple[str, str] | None:
        if not sandbox_id.startswith("local:"):
            return None
        value = sandbox_id[len("local:") :]
        user_id, separator, thread_id = value.partition(":")
        if not separator or not user_id or not thread_id:
            return None
        return (user_id, thread_id)

    @staticmethod
    def _build_thread_path_mappings(thread_id: str, *, user_id: str | None = None, skill_projection=None) -> list[PathMapping]:
        """/mnt/user-data, /mnt/acp-workspace, /mnt/skills/custom에 대한 per-thread path
        mapping을 만든다.

        명시적으로 resolve된 user id가 주어지면 그것을 쓰고, 레거시 caller에 대해서는
        :func:`get_effective_user_id`로 폴백한다. custom skills는 사용자별로 read-only
        mount한다. agent가 custom skill을 sandbox 안이 아니라 host 파일시스템에서
        ``skill_manage_tool``로 작성하기 때문이다.
        """
        from deerflow.config import get_app_config
        from deerflow.config.paths import get_paths

        paths = get_paths()
        effective_user_id = LocalSandboxProvider._effective_acquire_user_id(user_id)
        paths.ensure_thread_dirs(thread_id, user_id=effective_user_id)

        mappings = [
            # 부모 디렉터리를 묶는 mapping. ``ls /mnt/user-data`` 등 부모 수준 작업이
            # AIO 안에서와 똑같이 동작하게 한다(AIO에서는 부모 디렉터리가 실제로
            # 존재하고 세 개의 하위 디렉터리를 담는다). ``_find_path_mapping``이
            # container_path 길이로 정렬하므로 ``/mnt/user-data/workspace/...``에는
            # 아래의 더 긴 하위 경로 mapping이 여전히 우선한다.
            PathMapping(
                container_path=_USER_DATA_VIRTUAL_PREFIX,
                local_path=str(paths.sandbox_user_data_dir(thread_id, user_id=effective_user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/workspace",
                local_path=str(paths.sandbox_work_dir(thread_id, user_id=effective_user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/uploads",
                local_path=str(paths.sandbox_uploads_dir(thread_id, user_id=effective_user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/outputs",
                local_path=str(paths.sandbox_outputs_dir(thread_id, user_id=effective_user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=_ACP_WORKSPACE_VIRTUAL_PREFIX,
                local_path=str(paths.acp_workspace_dir(thread_id, user_id=effective_user_id)),
                read_only=False,
            ),
        ]

        # per-user 카테고리 mount는 sandbox 수명 내내 유지된다. 활성화된 항목만 담은
        # 내용물이 이 안정적인 root 아래에서 바뀐다.
        try:
            config = get_app_config()
            skills_container_path = config.skills.container_path
            projection = skill_projection if skill_projection is not None else LocalSandboxProvider._ensure_skills_projection(effective_user_id)

            if projection is not None:
                mappings.extend(
                    [
                        PathMapping(
                            container_path=f"{skills_container_path}/custom",
                            local_path=str(projection.custom),
                            read_only=True,
                        ),
                        PathMapping(
                            container_path=f"{skills_container_path}/legacy",
                            local_path=str(projection.legacy),
                            read_only=True,
                        ),
                        PathMapping(
                            container_path=f"{skills_container_path}/integrations",
                            local_path=str(projection.integrations),
                            read_only=True,
                        ),
                    ]
                )
        except Exception as exc:
            logger.warning("Could not setup per-thread skills projection mounts: %s", exc, exc_info=True)

        return mappings

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """*thread_id*로 스코프된 sandbox id(또는 일반 singleton)를 반환한다.

        - ``thread_id=None``은 thread context가 없는 caller(예: 레거시 테스트, 스크립트)를
          위해 id ``"local"``인 레거시 singleton을 유지한다.
        - ``thread_id="abc"``는 id가 ``"local:abc"``이고 ``path_mappings``가
          ``/mnt/user-data/...``를 그 thread의 host 디렉터리로 resolve하는 per-thread
          ``LocalSandbox``를 만든다.

        동시 호출에서도 thread-safe하다. cache 확인과 삽입을 ``self._lock``으로 보호하므로
        같은 ``thread_id``로 경쟁하는 두 caller는 항상 같은 LocalSandbox 인스턴스를 본다.
        """
        global _singleton

        if thread_id is None:
            skill_projection = self._ensure_skills_projection()
            with self._lock:
                if self._generic_sandbox is None:
                    mappings = list(self._path_mappings)
                    self._append_public_skill_mapping(mappings, skill_projection)
                    self._generic_sandbox = LocalSandbox("local", path_mappings=mappings)
                    _singleton = self._generic_sandbox
                return self._generic_sandbox.id

        effective_user_id = self._effective_acquire_user_id(user_id)
        # drift를 자가 복구하기 위해 cache hit를 포함한 모든 acquire에서 실행한다.
        # manifest가 최신이면 저렴하다(메타데이터 순회 약 3-4 ms). 마지막 확인 이후 다른
        # worker가 이 사용자의 skill을 변경했다면 cross-process projection lock 아래에서
        # 전체 rebuild가 발생하고(로컬 측정 약 400 ms), 그 사용자의 동시 acquire와 변경이
        # 직렬화된다. 편집 빈도의 이벤트라면 감수할 만하다.
        skill_projection = self._ensure_skills_projection(effective_user_id)
        key = self._thread_key(thread_id, effective_user_id)

        # lock 아래의 fast path.
        with self._lock:
            cached = self._thread_sandboxes.get(key)
            if cached is not None:
                # 자주 쓰는 thread가 eviction을 견디도록 most-recently used로 표시한다.
                self._thread_sandboxes.move_to_end(key)
        if cached is not None:
            return cached.id

        # ``_build_thread_path_mappings``는 파일시스템을 건드리므로
        # (``ensure_thread_dirs``) I/O 동안에는 lock을 놓는다.
        new_mappings = list(self._path_mappings)
        self._append_public_skill_mapping(new_mappings, skill_projection)
        new_mappings += self._build_thread_path_mappings(
            thread_id,
            user_id=effective_user_id,
            skill_projection=skill_projection,
        )

        with self._lock:
            # lock 없이 I/O를 한 뒤 재확인한다. mapping을 계산하는 동안 다른 caller가
            # cache를 채웠을 수 있다.
            cached = self._thread_sandboxes.get(key)
            if cached is None:
                cached = LocalSandbox(self._sandbox_id_for_thread(thread_id, effective_user_id), path_mappings=new_mappings)
                self._thread_sandboxes[key] = cached
                self._evict_until_within_cap_locked()
            else:
                self._thread_sandboxes.move_to_end(key)
            return cached.id

    def _evict_until_within_cap_locked(self) -> None:
        """상한을 넘으면 캐시된 thread sandbox를 LRU 순서로 evict한다.

        caller는 반드시 ``self._lock``을 잡고 있어야 한다.
        """
        while len(self._thread_sandboxes) > self._max_cached_threads:
            evicted_key, _ = self._thread_sandboxes.popitem(last=False)
            logger.info(
                "Evicting LocalSandbox cache entry for user/thread %s/%s (cap=%d)",
                evicted_key[0],
                evicted_key[1],
                self._max_cached_threads,
            )

    def get(self, sandbox_id: str) -> Sandbox | None:
        if sandbox_id == "local":
            with self._lock:
                generic = self._generic_sandbox
            if generic is None:
                self.acquire()
                with self._lock:
                    return self._generic_sandbox
            return generic
        if isinstance(sandbox_id, str) and sandbox_id.startswith("local:"):
            key = self._key_from_sandbox_id(sandbox_id)
            if key is None:
                return None
            with self._lock:
                cached = self._thread_sandboxes.get(key)
                if cached is not None:
                    # ``get``으로 thread를 건드리면(tools.py가 tool call마다 sandbox를
                    # 조회할 때 사용) LRU 순서에서 승격되므로, 부하 상황에서도 활성
                    # thread가 evict되지 않는다.
                    self._thread_sandboxes.move_to_end(key)
                return cached
        return None

    def release(self, sandbox_id: str) -> None:
        # LocalSandbox에는 해제할 자원이 없다. 캐시된 인스턴스를 유지해
        # ``_agent_written_paths``(읽기 시 agent가 작성한 파일 내용을 reverse-resolve하는
        # 데 사용)가 턴 사이에 살아남게 한다. 캐시 항목을 버리는 경로는 ``acquire``의
        # LRU eviction과 명시적인 ``reset()`` / ``shutdown()``뿐이다.
        #
        # 참고: 이 메서드는 thread 안에서 여러 턴에 걸쳐 sandbox를 재사용할 수 있도록
        # SandboxMiddleware가 의도적으로 호출하지 않는다.
        pass

    def reset(self) -> None:
        """캐시된 LocalSandbox 인스턴스를 전부 버린다.

        ``reset_sandbox_provider()``가 이를 호출해 config / mount 변경이 다음
        ``acquire()``에 반영되게 한다. 모듈 수준 ``_singleton`` alias도 함께 리셋해,
        거기에 접근하는 예전 caller/테스트가 새 상태를 보게 한다.
        """
        global _singleton
        with self._lock:
            self._generic_sandbox = None
            self._thread_sandboxes.clear()
            _singleton = None

    def shutdown(self) -> None:
        # LocalSandboxProvider는 캐시된 ``LocalSandbox`` 인스턴스 외에 추가 자원이 없으므로
        # shutdown은 ``reset``과 같은 정리 경로를 쓴다.
        self.reset()
