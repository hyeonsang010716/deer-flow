import hashlib
import logging
import os
import re
import shutil
from pathlib import Path, PureWindowsPath

from deerflow.config.runtime_paths import runtime_home
from deerflow.utils.thread_id import validate_thread_id

# sandbox 안에서 에이전트가 보게 되는 가상 경로 prefix
VIRTUAL_PATH_PREFIX = "/mnt/user-data"

_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_SAFE_INTEGRATION_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_UNSAFE_USER_ID_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]")
_SAFE_USER_ID_DIGEST_HEX_LEN = 16

logger = logging.getLogger(__name__)


def _default_local_base_dir() -> Path:
    """호출자 프로젝트의 쓰기 가능한 DeerFlow 상태 디렉터리를 반환한다."""
    return runtime_home()


def _validate_thread_id(thread_id: str) -> str:
    """파일시스템 경로에 쓰기 전에 thread ID를 검증한다."""
    return validate_thread_id(thread_id)


def _validate_user_id(user_id: str) -> str:
    """파일시스템 경로에 쓰기 전에 user ID를 검증한다."""
    if not _SAFE_USER_ID_RE.match(user_id):
        raise ValueError(f"Invalid user_id {user_id!r}: only alphanumeric characters, hyphens, and underscores are allowed.")
    return user_id


def _validate_integration_id(integration_id: str) -> str:
    """파일시스템 경로에 쓰기 전에 integration ID를 검증한다."""
    if not _SAFE_INTEGRATION_ID_RE.match(integration_id):
        raise ValueError(f"Invalid integration_id {integration_id!r}: only alphanumeric characters, dots, hyphens, and underscores are allowed.")
    # ``some.integration`` 같은 이름을 위해 점을 허용하므로, ``.``/``..`` 단독 경로 요소는
    # 거부한다. 그래야 나중에 ``_join_host_path(..., integration_id, ...)``로 integration
    # 네임스페이스를 벗어날 수 없다.
    if integration_id in {".", ".."}:
        raise ValueError(f"Invalid integration_id {integration_id!r}: '.' and '..' are not allowed.")
    return integration_id


def make_safe_user_id(raw: str) -> str:
    """외부 identity를 user-id 문자 집합(``[A-Za-z0-9_-]``)으로 정규화한다.

    IM channel id(Feishu/Slack/Telegram)에는 :func:`_validate_user_id`가 거부하는 문자가
    들어갈 수 있다. 이미 안전한 id는 그대로 통과시키고, 변환으로 정보가 손실되는 경우에는
    짧은 digest를 덧붙여 서로 다른 입력이 같은 저장소 버킷을 공유하지 않게 한다.
    """
    if not raw:
        raise ValueError("user_id must be a non-empty string.")
    sanitized = _UNSAFE_USER_ID_CHAR_RE.sub("-", raw)
    if sanitized == raw:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_SAFE_USER_ID_DIGEST_HEX_LEN]
    return f"{sanitized}-{digest}"


def _legacy_safe_user_id(raw: str, sanitized: str) -> str:
    """``raw``에 대해 이전 SHA-1 digest 방식이 만들던 버킷 이름을 반환한다."""
    digest = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:_SAFE_USER_ID_DIGEST_HEX_LEN]
    return f"{sanitized}-{digest}"


def _join_host_path(base: str, *parts: str) -> str:
    """host 파일시스템 경로 조각을 원래 표기 방식을 유지한 채 이어붙인다.

    Windows의 Docker Desktop은 bind mount source가 Windows 경로 형태로 유지되기를
    기대한다(예: ``C:\\repo\\backend\\.deer-flow``). POSIX host에서 ``Path(base) / ...``를
    쓰면 구분자가 섞인 경로로 잘못 바뀔 수 있어, 이 헬퍼가 원래 표기를 보존한다.
    """
    if not parts:
        return base

    if re.match(r"^[A-Za-z]:[\\/]", base) or base.startswith("\\\\") or "\\" in base:
        result = PureWindowsPath(base)
        for part in parts:
            result /= part
        return str(result)

    result = Path(base)
    for part in parts:
        result /= part
    return str(result)


def join_host_path(base: str, *parts: str) -> str:
    """host 파일시스템 경로 조각을 원래 표기 방식을 유지한 채 이어붙인다."""
    return _join_host_path(base, *parts)


class Paths:
    """
    DeerFlow 애플리케이션 데이터의 경로 설정을 한곳에 모은 클래스.

    디렉터리 구조(host 기준):
        {base_dir}/
        ├── memory.json
        ├── USER.md          <-- 전역 user 프로필(모든 에이전트에 주입)
        ├── agents/
        │   └── {agent_name}/
        │       ├── config.yaml
        │       ├── SOUL.md  <-- 에이전트 성격/정체성(lead prompt와 함께 주입)
        │       └── memory.json
        └── threads/
            └── {thread_id}/
                └── user-data/         <-- sandbox 안에서 /mnt/user-data/로 mount
                    ├── workspace/     <-- /mnt/user-data/workspace/
                    ├── uploads/       <-- /mnt/user-data/uploads/
                    └── outputs/       <-- /mnt/user-data/outputs/

    base_dir 결정 순서:
        1. 생성자 인자 `base_dir`
        2. DEER_FLOW_HOME 환경변수
        3. 호출자 프로젝트 fallback: `{project_root}/.deer-flow`
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None

    @property
    def host_base_dir(self) -> Path:
        """Docker volume mount source로 쓰이는 host 기준 base dir.

        Docker socket을 mount한 채 Docker 안에서 실행하면(DooD) Docker daemon은 host에서
        돌면서 mount 경로를 host 파일시스템 기준으로 해석한다. 이 컨테이너의 base_dir에
        대응하는 host 경로를 DEER_FLOW_HOST_BASE_DIR에 지정해야 sandbox 컨테이너의 volume
        mount가 제대로 동작한다.

        환경변수가 없으면(네이티브/로컬 실행) base_dir로 fallback한다.
        """
        if env := os.getenv("DEER_FLOW_HOST_BASE_DIR"):
            return Path(env)
        return self.base_dir

    def _host_base_dir_str(self) -> str:
        """bind mount용으로 host base dir을 원시 문자열로 반환한다."""
        if env := os.getenv("DEER_FLOW_HOST_BASE_DIR"):
            return env
        return str(self.base_dir)

    @property
    def base_dir(self) -> Path:
        """모든 애플리케이션 데이터의 루트 디렉터리."""
        if self._base_dir is not None:
            return self._base_dir

        if env_home := os.getenv("DEER_FLOW_HOME"):
            return Path(env_home).resolve()

        return _default_local_base_dir()

    @property
    def memory_file(self) -> Path:
        """저장된 memory 파일 경로: `{base_dir}/memory.json`."""
        return self.base_dir / "memory.json"

    @property
    def user_md_file(self) -> Path:
        """전역 user 프로필 파일 경로: `{base_dir}/USER.md`."""
        return self.base_dir / "USER.md"

    @property
    def agents_dir(self) -> Path:
        """user 격리 이전의 공유 custom agent legacy 루트: `{base_dir}/agents/`.

        새 코드는 :meth:`user_agents_dir`을 써야 한다. 이 property는 아직
        ``migrate_user_isolation.py``를 돌리지 않은 설치본을 위한 읽기 전용 fallback으로만
        남아 있다.
        """
        return self.base_dir / "agents"

    def agent_dir(self, name: str) -> Path:
        """user 격리가 없던 legacy agent 디렉터리: `{base_dir}/agents/{name}/`."""
        return self.agents_dir / name.lower()

    def agent_memory_file(self, name: str) -> Path:
        """legacy agent memory 파일: `{base_dir}/agents/{name}/memory.json`."""
        return self.agent_dir(name) / "memory.json"

    def user_dir(self, user_id: str) -> Path:
        """특정 user의 디렉터리: `{base_dir}/users/{user_id}/`."""
        return self.base_dir / "users" / _validate_user_id(user_id)

    def prepare_user_dir_for_raw_id(self, raw_user_id: str) -> str:
        """안전한 user ID를 반환하고, 이 ID의 legacy unsafe-id 버킷을 이관한다.

        이전 리비전은 안전하지 않은 외부 user ID에 SHA-1을 썼고 지금은 SHA-256을 쓴다.
        legacy 버킷 이름은 동일한 raw ID로 다시 계산하므로 이 user 자신의 옛 버킷만
        옮겨진다. sanitize된 prefix가 같은 다른 raw ID는 legacy digest가 달라 건드리지 않는다.
        """
        safe_user_id = make_safe_user_id(raw_user_id)
        sanitized = _UNSAFE_USER_ID_CHAR_RE.sub("-", raw_user_id)
        if safe_user_id == raw_user_id:
            return safe_user_id

        users_dir = self.base_dir / "users"
        target_dir = users_dir / safe_user_id
        legacy_dir = users_dir / _legacy_safe_user_id(raw_user_id, sanitized)
        try:
            if target_dir.exists() or not legacy_dir.is_dir():
                return safe_user_id
            legacy_dir.rename(target_dir)
            logger.info("Migrated legacy unsafe-id user directory to the current digest format")
        except OSError:
            logger.exception("Failed to migrate legacy unsafe-id user directory")
        return safe_user_id

    def user_memory_file(self, user_id: str) -> Path:
        """user별 memory 파일: `{base_dir}/users/{user_id}/memory.json`."""
        return self.user_dir(user_id) / "memory.json"

    def user_agents_dir(self, user_id: str) -> Path:
        """해당 user의 custom agent 루트: `{base_dir}/users/{user_id}/agents/`."""
        return self.user_dir(user_id) / "agents"

    def user_agent_dir(self, user_id: str, agent_name: str) -> Path:
        """user별 agent 디렉터리: `{base_dir}/users/{user_id}/agents/{name}/`."""
        return self.user_agents_dir(user_id) / agent_name.lower()

    def user_agent_memory_file(self, user_id: str, agent_name: str) -> Path:
        """user별 agent memory: `{base_dir}/users/{user_id}/agents/{name}/memory.json`."""
        return self.user_agent_dir(user_id, agent_name) / "memory.json"

    def user_skills_dir(self, user_id: str) -> Path:
        """해당 user의 custom skill 루트: `{base_dir}/users/{user_id}/skills/`."""
        return self.user_dir(user_id) / "skills"

    def user_custom_skills_dir(self, user_id: str) -> Path:
        """user별 custom skill 디렉터리: `{base_dir}/users/{user_id}/skills/custom/`.

        전역 ``{base_dir}/skills/custom/``을 대체하는 user 단위 위치다. custom skill은 여기에
        기록되고, public skill은 전역 ``{base_dir}/skills/public/``에 읽기 전용으로 남는다.
        """
        return self.user_skills_dir(user_id) / "custom"

    def integration_skills_dir(self) -> Path:
        """전역으로 설치된 managed integration skill.

        구조: ``{base_dir}/integrations/skills/{provider}/{skill}/``. 패키지 내용은 공유되며
        읽기 전용이고, 자격 증명과 활성 상태는 ``users/{user_id}`` 아래에 user 단위로 남는다.
        """
        return self.base_dir / "integrations" / "skills"

    @property
    def skills_view_dir(self) -> Path:
        """sandbox에 노출되는 전역 skill projection: ``{base_dir}/skills_view/``."""
        return self.base_dir / "skills_view"

    @property
    def public_skills_view_dir(self) -> Path:
        """sandbox에 노출되는 활성 public skill."""
        return self.skills_view_dir / "public"

    def user_skills_view_dir(self, user_id: str) -> Path:
        """user별 sandbox 노출 skill projection 루트."""
        return self.user_dir(user_id) / "skills_view"

    def user_custom_skills_view_dir(self, user_id: str) -> Path:
        """한 user의 sandbox에 노출되는 활성 custom skill."""
        return self.user_skills_view_dir(user_id) / "custom"

    def user_legacy_skills_view_dir(self, user_id: str) -> Path:
        """한 user의 sandbox에 노출되는 활성 legacy skill."""
        return self.user_skills_view_dir(user_id) / "legacy"

    def user_integration_skills_view_dir(self, user_id: str) -> Path:
        """한 user의 sandbox에 노출되는 활성 managed integration skill."""
        return self.user_skills_view_dir(user_id) / "integrations"

    def thread_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        thread 데이터의 host 경로.

        *user_id*가 주어지면:
            `{base_dir}/users/{user_id}/threads/{thread_id}/`
        아니면(legacy 구조):
            `{base_dir}/threads/{thread_id}/`

        이 디렉터리에는 sandbox 안에서 `/mnt/user-data/`로 mount되는 `user-data/`
        하위 디렉터리가 들어 있다.

        Raises:
            ValueError: `thread_id`나 `user_id`에 directory traversal을 일으킬 수 있는
                        위험한 문자(경로 구분자나 `..`)가 들어 있는 경우.
        """
        if user_id is not None:
            return self.user_dir(user_id) / "threads" / _validate_thread_id(thread_id)
        return self.base_dir / "threads" / _validate_thread_id(thread_id)

    def sandbox_work_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        에이전트 workspace 디렉터리의 host 경로.
        Host: `{base_dir}/threads/{thread_id}/user-data/workspace/`
        Sandbox: `/mnt/user-data/workspace/`
        """
        return self.thread_dir(thread_id, user_id=user_id) / "user-data" / "workspace"

    def sandbox_uploads_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        user가 업로드한 파일의 host 경로.
        Host: `{base_dir}/threads/{thread_id}/user-data/uploads/`
        Sandbox: `/mnt/user-data/uploads/`
        """
        return self.thread_dir(thread_id, user_id=user_id) / "user-data" / "uploads"

    def sandbox_outputs_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        에이전트가 생성한 artifact의 host 경로.
        Host: `{base_dir}/threads/{thread_id}/user-data/outputs/`
        Sandbox: `/mnt/user-data/outputs/`
        """
        return self.thread_dir(thread_id, user_id=user_id) / "user-data" / "outputs"

    def acp_workspace_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        특정 thread의 ACP workspace host 경로.
        Host: `{base_dir}/threads/{thread_id}/acp-workspace/`
        Sandbox: `/mnt/acp-workspace/`

        thread마다 격리된 ACP workspace를 가지므로 동시에 진행되는 session끼리 서로의
        ACP 에이전트 출력을 읽을 수 없다.
        """
        return self.thread_dir(thread_id, user_id=user_id) / "acp-workspace"

    def sandbox_user_data_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
        """
        user-data 루트의 host 경로.
        Host: `{base_dir}/threads/{thread_id}/user-data/`
        Sandbox: `/mnt/user-data/`
        """
        return self.thread_dir(thread_id, user_id=user_id) / "user-data"

    def host_thread_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """thread 디렉터리의 host 경로. Windows 경로 표기를 보존한다."""
        if user_id is not None:
            return _join_host_path(self._host_base_dir_str(), "users", _validate_user_id(user_id), "threads", _validate_thread_id(thread_id))
        return _join_host_path(self._host_base_dir_str(), "threads", _validate_thread_id(thread_id))

    def host_sandbox_user_data_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """thread의 user-data 루트 host 경로."""
        return _join_host_path(self.host_thread_dir(thread_id, user_id=user_id), "user-data")

    def host_sandbox_work_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """workspace mount source의 host 경로."""
        return _join_host_path(self.host_sandbox_user_data_dir(thread_id, user_id=user_id), "workspace")

    def host_sandbox_uploads_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """uploads mount source의 host 경로."""
        return _join_host_path(self.host_sandbox_user_data_dir(thread_id, user_id=user_id), "uploads")

    def host_sandbox_outputs_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """outputs mount source의 host 경로."""
        return _join_host_path(self.host_sandbox_user_data_dir(thread_id, user_id=user_id), "outputs")

    def host_acp_workspace_dir(self, thread_id: str, *, user_id: str | None = None) -> str:
        """ACP workspace mount source의 host 경로."""
        return _join_host_path(self.host_thread_dir(thread_id, user_id=user_id), "acp-workspace")

    def host_user_custom_skills_dir(self, user_id: str) -> str:
        """user의 custom skill 디렉터리 host 경로. Windows 경로 표기를 보존한다."""
        return _join_host_path(self._host_base_dir_str(), "users", _validate_user_id(user_id), "skills", "custom")

    def host_integration_skills_dir(self) -> str:
        """전역 설치된 managed integration skill의 host 경로."""
        return _join_host_path(self._host_base_dir_str(), "integrations", "skills")

    def host_user_integration_config_dir(self, user_id: str, integration_id: str) -> str:
        """user의 managed integration 런타임 config 디렉터리 host 경로."""
        return _join_host_path(self._host_base_dir_str(), "users", _validate_user_id(user_id), "integrations", _validate_integration_id(integration_id), "config")

    def host_user_integration_data_dir(self, user_id: str, integration_id: str) -> str:
        """user의 managed integration 런타임 data 디렉터리 host 경로."""
        return _join_host_path(self._host_base_dir_str(), "users", _validate_user_id(user_id), "integrations", _validate_integration_id(integration_id), "data")

    def ensure_thread_dirs(self, thread_id: str, *, user_id: str | None = None) -> None:
        """thread에 필요한 표준 sandbox 디렉터리를 모두 만든다.

        디렉터리는 0o777로 만든다. sandbox 컨테이너가 host backend 프로세스와 다른 UID로
        실행될 수 있어서, 그래야 volume mount된 경로에 "Permission denied" 없이 쓸 수 있다.
        Path.mkdir(mode=...)는 프로세스 umask의 영향을 받아 의도한 권한이 되지 않으므로
        chmod()를 따로 호출한다.

        ACP workspace 디렉터리도 포함한다. 그래야 첫 ACP 에이전트 호출 전에도 sandbox
        컨테이너의 ``/mnt/acp-workspace``로 volume mount할 수 있다.
        """
        for d in [
            self.sandbox_work_dir(thread_id, user_id=user_id),
            self.sandbox_uploads_dir(thread_id, user_id=user_id),
            self.sandbox_outputs_dir(thread_id, user_id=user_id),
            self.acp_workspace_dir(thread_id, user_id=user_id),
        ]:
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o777)

    def delete_thread_dir(self, thread_id: str, *, user_id: str | None = None) -> None:
        """thread에 저장된 데이터를 모두 삭제한다.

        멱등하게 동작한다. thread 디렉터리가 없으면 그냥 넘어간다.
        """
        thread_dir = self.thread_dir(thread_id, user_id=user_id)
        if thread_dir.exists():
            shutil.rmtree(thread_dir)

    def resolve_virtual_path(self, thread_id: str, virtual_path: str, *, user_id: str | None = None) -> Path:
        """sandbox 가상 경로를 실제 host 파일시스템 경로로 변환한다.

        Args:
            thread_id: thread ID.
            virtual_path: sandbox 안에서 보이는 가상 경로. 예:
                          ``/mnt/user-data/outputs/report.pdf``.
                          비교 전에 앞쪽 슬래시는 제거한다.
            user_id: user 단위 경로 해석을 위한 user ID(선택).

        Returns:
            변환된 절대 host 파일시스템 경로.

        Raises:
            ValueError: 경로가 예상한 가상 prefix로 시작하지 않거나 path traversal 시도가
                        감지된 경우.
        """
        stripped = virtual_path.lstrip("/")
        prefix = VIRTUAL_PATH_PREFIX.lstrip("/")

        # prefix 혼동을 막기 위해 경로 세그먼트 경계까지 정확히 일치해야 한다
        # (예: "mnt/user-dataX/..." 같은 경로는 거부한다).
        if stripped != prefix and not stripped.startswith(prefix + "/"):
            raise ValueError(f"Path must start with /{prefix}")

        relative = stripped[len(prefix) :].lstrip("/")
        base = self.sandbox_user_data_dir(thread_id, user_id=user_id).resolve()
        actual = (base / relative).resolve()

        try:
            actual.relative_to(base)
        except ValueError:
            raise ValueError("Access denied: path traversal detected")

        return actual


# ── Singleton ────────────────────────────────────────────────────────────

_paths: Paths | None = None


def get_paths() -> Paths:
    """전역 Paths singleton을 반환한다(지연 초기화)."""
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths


def resolve_path(path: str) -> Path:
    """*path*를 절대 ``Path``로 변환한다.

    상대 경로는 애플리케이션 base 디렉터리를 기준으로 해석한다. 절대 경로는 정규화만 거쳐
    그대로 반환한다.
    """
    p = Path(path)
    if not p.is_absolute():
        p = get_paths().base_dir / path
    return p.resolve()
