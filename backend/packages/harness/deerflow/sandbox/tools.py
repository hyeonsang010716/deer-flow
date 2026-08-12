import asyncio
import json
import logging
import os
import posixpath
import re
import shlex
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from langchain.tools import tool

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.runtime.secret_context import read_active_secrets
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.sandbox.exceptions import (
    SandboxError,
    SandboxNotFoundError,
    SandboxRuntimeError,
)
from deerflow.sandbox.file_operation_lock import get_file_operation_lock
from deerflow.sandbox.overwrite import unwrap_sandbox
from deerflow.sandbox.path_patterns import build_output_mask_pattern
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import get_sandbox_provider
from deerflow.sandbox.search import GrepMatch
from deerflow.sandbox.security import LOCAL_HOST_BASH_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![:\w])(?<!:/)/(?:[^\s\"'`;&|<>()]+)")
# identifier 형태의 placeholder 하나만 담은 ``{...}`` 블록(예: REST 템플릿의 ``{id}``,
# f-string의 ``{port}``). ``{passwd,shadow}``나 ``{,.bak}`` 같은 bash brace expansion은
# 매칭되지 않는다(쉼표/점/빈 내부).
_IDENTIFIER_BRACE_BLOCK_PATTERN = re.compile(r"\{([^{}]*)\}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE_URL_PATTERN = re.compile(r"\bfile://\S+", re.IGNORECASE)
_URL_WITH_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_URL_IN_COMMAND_PATTERN = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'`;&|<>()]+", re.IGNORECASE)
_DOTDOT_PATH_SEGMENT_PATTERN = re.compile(r"(?:^|[/\\=])\.\.(?:$|[/\\])")
_LOCAL_BASH_SYSTEM_PATH_PREFIXES = (
    "/bin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/sbin/",
    "/opt/homebrew/bin/",
    "/dev/",
)

_DEFAULT_SKILLS_CONTAINER_PATH = DEFAULT_SKILLS_CONTAINER_PATH
_ACP_WORKSPACE_VIRTUAL_PATH = "/mnt/acp-workspace"
_DEFAULT_GLOB_MAX_RESULTS = 200
_MAX_GLOB_MAX_RESULTS = 1000
_DEFAULT_GREP_MAX_RESULTS = 100
_MAX_GREP_MAX_RESULTS = 500
_DEFAULT_WRITE_FILE_ERROR_MAX_CHARS = 2000

# append가 아닌 단일 write_file 호출이 받아들이는 최대 바이트 수(issue #3189).
# 한 번에 너무 큰 쓰기는 LLM streaming chunk-gap timeout과 상관관계가 있다. 모델이 하나의
# 연속 stream으로 뱉어야 하는 tool call JSON payload가 안전 구간을 넘어서기 때문이다.
# 80 KB ≈ 20K token으로, factory 기본값인 240s stream_chunk_timeout 아래로 충분한 여유가 있다.
# 배포 환경은 env var DEERFLOW_WRITE_FILE_MAX_BYTES로 덮어쓸 수 있고, 0(또는 음수)으로
# 두면 이 guard를 완전히 끈다.
_WRITE_FILE_CONTENT_MAX_BYTES = 80 * 1024
_WRITE_FILE_MAX_BYTES_ENV = "DEERFLOW_WRITE_FILE_MAX_BYTES"
_LOCAL_BASH_CWD_COMMANDS = {"cd", "pushd"}
_LOCAL_BASH_COMMAND_WRAPPERS = {"command", "builtin"}
_LOCAL_BASH_COMMAND_PREFIX_KEYWORDS = {"!", "{", "case", "do", "elif", "else", "for", "if", "select", "then", "time", "until", "while"}
_LOCAL_BASH_COMMAND_END_KEYWORDS = {"}", "done", "esac", "fi"}
_LOCAL_BASH_ROOT_PATH_COMMANDS = {
    "awk",
    "cat",
    "cp",
    "du",
    "find",
    "grep",
    "head",
    "less",
    "ln",
    "ls",
    "more",
    "mv",
    "rm",
    "sed",
    "tail",
    "tar",
}
_SHELL_COMMAND_SEPARATORS = {";", "&&", "||", "|", "|&", "&", "(", ")"}
_SHELL_REDIRECTION_OPERATORS = {
    "<",
    ">",
    "<<",
    ">>",
    "<<<",
    "<>",
    ">&",
    "<&",
    "&>",
    "&>>",
    ">|",
}


def _get_skills_container_path() -> str:
    """config에서 skills container path를 가져오고, 실패하면 기본값으로 fallback한다.

    첫 config 로드가 성공하면 결과를 캐시한다. config 로드가 실패하면 캐시하지 *않고*
    기본값을 반환하므로, 나중에 config가 준비되면 다음 호출이 실제 값을 가져올 수 있다.
    """
    cached = getattr(_get_skills_container_path, "_cached", None)
    if cached is not None:
        return cached
    try:
        from deerflow.config import get_app_config

        value = get_app_config().skills.container_path
        _get_skills_container_path._cached = value  # type: ignore[attr-defined]
        return value
    except Exception:
        return _DEFAULT_SKILLS_CONTAINER_PATH


def _get_skills_host_path() -> str | None:
    """config에서 skills의 host 파일시스템 경로를 가져온다.

    skills 디렉터리가 없거나 config를 로드할 수 없으면 None을 반환한다. 성공한 조회만
    캐시하고 실패는 다음 호출에서 재시도하므로, 일시적으로 접근 불가능한 skills 디렉터리가
    skills 접근을 영구히 막지 않는다.
    """
    cached = getattr(_get_skills_host_path, "_cached", None)
    if cached is not None:
        return cached
    try:
        from deerflow.config import get_app_config

        config = get_app_config()
        skills_path = config.skills.get_skills_path()
        if skills_path.exists():
            value = str(skills_path)
            _get_skills_host_path._cached = value  # type: ignore[attr-defined]
            return value
    except Exception:
        pass
    return None


def _is_skills_path(path: str) -> bool:
    """경로가 skills container path 아래에 있는지 확인한다."""
    skills_prefix = _get_skills_container_path()
    return path == skills_prefix or path.startswith(f"{skills_prefix}/")


def _extract_skill_name_from_skills_path(path: str) -> str | None:
    """virtual skills 경로에서 skill 이름을 추출한다.

    /mnt/skills/public/bootstrap/SKILL.md → "bootstrap"
    /mnt/skills/custom/my-skill/SKILL.md → "my-skill"
    /mnt/skills/legacy/my-skill/references/... → "my-skill"
    /mnt/skills/integrations/lark-cli/lark-doc/SKILL.md → "lark-doc"
    /mnt/skills/public/bootstrap/ → "bootstrap"
    인식 가능한 skill 이름 패턴이 경로에 없으면 None을 반환한다.
    """
    skills_prefix = _get_skills_container_path()
    if not _is_skills_path(path):
        return None
    # skills prefix(예: "/mnt/skills/")를 떼어낸다.
    relative = path[len(skills_prefix) :].lstrip("/")
    if not relative:
        return None
    # 기대하는 패턴: "public/<name>/...", "custom/<name>/...",
    # "legacy/<name>/...", "integrations/<provider>/<name>/..."
    # 또는 "<name>/..."(직접 skill 접근). 빈 segment는 버리므로 디렉터리 항목
    # (`ls`가 디렉터리에 대해 내보내는 "public/")도 빈 skill 이름이 되지 않고
    # category root로 인식된다.
    parts = [part for part in relative.split("/") if part]
    if len(parts) >= 2 and parts[0] in ("public", "custom", "legacy"):
        return parts[1]
    if len(parts) >= 3 and parts[0] == "integrations":
        return parts[2]
    if len(parts) == 1 and parts[0] in ("public", "custom", "legacy", "integrations"):
        # /mnt/skills/custom 같은 category root — skill 경로가 아니다.
        return None
    if len(parts) == 2 and parts[0] == "integrations":
        # /mnt/skills/integrations/lark-cli 같은 provider root.
        return None
    if len(parts) >= 1:
        # /mnt/skills/my-skill/SKILL.md 같은 직접 경로
        return parts[0]
    return None


def _is_disabled_skill_path(path: str, *, user_id: str | None = None) -> bool:
    """경로가 비활성화된 skill에 속하는지 확인한다.

    PUBLIC skill의 enabled 상태는 전역 ``extensions_config.json``에서 읽는다.
    CUSTOM / LEGACY skill의 enabled 상태는 per-user ``_skill_states.json``에서 읽으므로,
    같은 이름의 custom skill을 가진 두 사용자가 서로 독립적으로 토글할 수 있다.

    skills 경로가 아니거나 해당 skill이 활성화되어 있으면 False를 반환한다.
    """
    skill_name = _extract_skill_name_from_skills_path(path)
    if skill_name is None:
        return False
    try:
        from deerflow.runtime.user_context import get_effective_user_id
        from deerflow.skills.storage import get_or_new_user_skill_storage

        # 경로에서 category를 판별한다.
        skills_prefix = _get_skills_container_path()
        relative = path[len(skills_prefix) :].lstrip("/")
        if relative.startswith("public/"):
            category = "public"
        elif relative.startswith("custom/"):
            category = "custom"
        elif relative.startswith("legacy/"):
            category = "legacy"
        elif relative.startswith("integrations/"):
            category = "integrations"
        else:
            # storage에서 추론을 시도한다.
            effective_uid = user_id or get_effective_user_id()
            storage = get_or_new_user_skill_storage(effective_uid)
            all_skills = storage.load_skills(enabled_only=False)
            matching = next((s for s in all_skills if s.name == skill_name), None)
            if matching is None:
                return False  # skill이 존재하지 않으므로 비활성 skill 경로가 아니다
            category = matching.category.value

        if category == "public":
            from deerflow.config.extensions_config import ExtensionsConfig

            ext_config = ExtensionsConfig.from_file()
            return not ext_config.is_skill_enabled(skill_name, category)
        else:
            # CUSTOM / LEGACY: per-user 상태를 사용한다.
            effective_uid = user_id or get_effective_user_id()
            storage = get_or_new_user_skill_storage(effective_uid)
            return not storage.get_skill_enabled_state(skill_name)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        # 접근 제어 검사는 fail closed여야 한다. enabled 상태를 판별할 수 없으면
        # (_skill_states.json 손상, 쓰기 중 race, config 누락) 비활성 skill의 파일을
        # 조용히 내주는 대신 접근을 거부한다. PR #3889의 리뷰 피드백 참고.
        logger.warning("Failed to determine enabled state, denying access: %s", exc)
        return True


def _drop_disabled_skill_paths(paths: list[str], *, user_id: str | None = None) -> list[str]:
    """비활성화된 skill에 속하는 경로를 걸러낸다.

    ``_is_disabled_skill_path``는 *요청된* 경로만 막는다. ``read_file``에는 그것으로
    충분하지만 하위로 내려가는 도구에는 부족하다. ``ls``, ``glob``, ``grep``은 받은 경로가
    아닌 다른 경로를 반환하므로, 비활성 skill 위쪽 어디든 root로 잡으면 그 파일이 노출된다.
    여기서 결과에도 동일한 검사를 적용한다.

    enabled 상태 조회는 매번 ``extensions_config.json``(또는 per-user skill 상태)을 다시
    읽으므로, 판정 결과를 skill 단위로 메모이즈한다. 100건이 매칭된 grep이 config를 100번
    읽어서는 안 된다.
    """
    skills_prefix = _get_skills_container_path()
    verdicts: dict[tuple[str, str], bool] = {}
    kept: list[str] = []
    for path in paths:
        skill_name = _extract_skill_name_from_skills_path(path)
        if skill_name is None:
            kept.append(path)
            continue
        # 첫 segment(category, 또는 직접 레이아웃에서는 skill 자체)와 이름을 합치면
        # _is_disabled_skill_path와 동일한 방식으로 skill이 식별되므로, 같은 key를
        # 가진 경로는 같은 판정을 공유한다.
        category = path[len(skills_prefix) :].lstrip("/").split("/")[0]
        key = (category, skill_name)
        if key not in verdicts:
            verdicts[key] = _is_disabled_skill_path(path, user_id=user_id)
        if not verdicts[key]:
            kept.append(path)
    return kept


def _resolve_skills_path(path: str) -> str:
    """virtual skills 경로를 host 파일시스템 경로로 해석한다.

    경고: per-user custom skill(``/mnt/skills/custom/...``)의 경우 이 함수는 contextvar의
    ``get_effective_user_id()``를 사용하는데, 이는 sandbox PathMapping의 user_id(acquire
    시점에 ``resolve_runtime_user_id``로 설정)와 다를 수 있다. local sandbox 모드에서는
    이 함수 대신 sandbox의 PathMapping이 skills 경로를 해석해야 한다. 이 함수는 output
    masking(``mask_local_paths_in_output``)과 sandbox가 아닌 코드 경로를 위해 남겨둔다.

    Args:
        path: virtual skills 경로(예: /mnt/skills/public/bootstrap/SKILL.md)

    Returns:
        해석된 host 경로.

    Raises:
        FileNotFoundError: skills 디렉터리가 설정되지 않았거나 존재하지 않을 때.
    """
    skills_container = _get_skills_container_path()
    skills_host = _get_skills_host_path()
    if skills_host is None:
        raise FileNotFoundError(f"Skills directory not available for path: {path}")

    if path == skills_container:
        return skills_host

    relative = path[len(skills_container) :].lstrip("/")

    # per-user custom skill과 전역으로 관리되는 integration skill.
    # ``skill_manage_tool``은 custom skill을 per-user 디렉터리에 쓰고,
    # ``LocalSandboxProvider._build_thread_path_mappings``는 ``/mnt/skills/custom``을
    # 같은 per-user 디렉터리에 마운트한다. 이 분기가 없으면
    # ``_resolve_skills_path("/mnt/skills/custom")``이 전역 ``{skills_host}/custom/``,
    # 즉 저장소 수준의 ``skills/custom/``으로 매핑된다. 이는 비어 있거나 legacy skill만
    # 들어 있을 수 있는 완전히 다른 디렉터리다.
    if relative == "custom" or relative.startswith("custom/"):
        from deerflow.config.paths import get_paths
        from deerflow.runtime.user_context import get_effective_user_id

        user_id = get_effective_user_id()
        paths = get_paths()
        user_custom_dir = paths.user_custom_skills_dir(user_id)
        custom_relative = relative[len("custom") :].lstrip("/")
        if custom_relative:
            return str(user_custom_dir / custom_relative)
        return str(user_custom_dir)

    if relative == "integrations" or relative.startswith("integrations/"):
        from deerflow.config.paths import get_paths

        paths = get_paths()
        integrations_dir = paths.integration_skills_dir()
        integrations_relative = relative[len("integrations") :].lstrip("/")
        if not integrations_relative:
            return str(integrations_dir)
        # 심층 방어: sandbox 호출자에 대해서는 상위에서 _reject_path_traversal이 이미
        # 돌지만, 해석된 경로가 전역 integration 디렉터리 안에 머무는지 확인해서
        # 문자열상의 ``../``가 여기서 빠져나가지 못하게 한다.
        resolved = (integrations_dir / integrations_relative).resolve()
        if not resolved.is_relative_to(integrations_dir.resolve()):
            raise PermissionError("Access denied: path traversal detected")
        return str(integrations_dir / integrations_relative)

    return _join_path_preserving_style(skills_host, relative)


def _is_acp_workspace_path(path: str) -> bool:
    """경로가 ACP workspace virtual path 아래에 있는지 확인한다."""
    return path == _ACP_WORKSPACE_VIRTUAL_PATH or path.startswith(f"{_ACP_WORKSPACE_VIRTUAL_PATH}/")


def _get_custom_mounts():
    """sandbox config에서 custom volume mount 목록을 가져온다.

    첫 config 로드가 성공하면 결과를 캐시한다. config 로드가 실패하면 캐시하지 *않고*
    빈 list를 반환하므로, 나중에 config가 준비되면 다음 호출이 실제 값을 가져올 수 있다.
    """
    cached = getattr(_get_custom_mounts, "_cached", None)
    if cached is not None:
        return cached
    try:
        from pathlib import Path

        from deerflow.config import get_app_config

        config = get_app_config()
        mounts = []
        if config.sandbox and config.sandbox.mounts:
            # host_path가 실제로 존재하는 mount만 포함한다. 마찬가지로
            # host_path.exists()로 거르는 LocalSandboxProvider._setup_path_mappings()와
            # 동작을 맞춘다.
            mounts = [m for m in config.sandbox.mounts if Path(m.host_path).exists()]
        _get_custom_mounts._cached = mounts  # type: ignore[attr-defined]
        return mounts
    except Exception:
        # config 로드가 실패하면 캐시하지 않고 빈 list를 반환해서, config가 준비된 뒤의
        # 호출이 다시 시도할 수 있게 한다.
        return []


def _is_custom_mount_path(path: str) -> bool:
    """경로가 custom mount의 container_path 아래에 있는지 확인한다."""
    for mount in _get_custom_mounts():
        if path == mount.container_path or path.startswith(f"{mount.container_path}/"):
            return True
    return False


def _get_custom_mount_for_path(path: str):
    """이 경로에 맞는 mount config를 가져온다(가장 긴 prefix 우선)."""
    best = None
    for mount in _get_custom_mounts():
        if path == mount.container_path or path.startswith(f"{mount.container_path}/"):
            if best is None or len(mount.container_path) > len(best.container_path):
                best = mount
    return best


def _extract_thread_id_from_thread_data(thread_data: "ThreadDataState | None") -> str | None:
    """workspace_path를 살펴서 thread_data에서 thread_id를 추출한다.

    workspace_path는 ``{base_dir}/threads/{thread_id}/user-data/workspace`` 형태이므로
    ``Path(workspace_path).parent.parent.name``이 thread_id가 된다.
    """
    if thread_data is None:
        return None
    workspace_path = thread_data.get("workspace_path")
    if not workspace_path:
        return None
    try:
        # {base_dir}/threads/{thread_id}/user-data/workspace → parent.parent = threads/{thread_id}로 이어진다.
        return Path(workspace_path).parent.parent.name
    except Exception:
        return None


def _get_acp_workspace_host_path(thread_id: str | None = None) -> str | None:
    """ACP workspace의 host 파일시스템 경로를 가져온다.

    *thread_id*가 주어지면 per-thread workspace
    ``{base_dir}/threads/{thread_id}/acp-workspace/``를 반환한다(캐시하지 않는다 —
    이 디렉터리는 ``invoke_acp_agent_tool``이 필요할 때 만든다).

    *thread_id*가 ``None``이면 전역 ``{base_dir}/acp-workspace/``로 fallback하며,
    그 결과는 첫 해석 성공 후 캐시한다. 디렉터리가 없으면 ``None``을 반환한다.
    """
    if thread_id is not None:
        try:
            from deerflow.config.paths import get_paths
            from deerflow.runtime.user_context import get_effective_user_id

            host_path = get_paths().acp_workspace_dir(thread_id, user_id=get_effective_user_id())
            if host_path.exists():
                return str(host_path)
        except Exception:
            pass
        return None

    cached = getattr(_get_acp_workspace_host_path, "_cached", None)
    if cached is not None:
        return cached
    try:
        from deerflow.config.paths import get_paths

        host_path = get_paths().base_dir / "acp-workspace"
        if host_path.exists():
            value = str(host_path)
            _get_acp_workspace_host_path._cached = value  # type: ignore[attr-defined]
            return value
    except Exception:
        pass
    return None


def _resolve_acp_workspace_path(path: str, thread_id: str | None = None) -> str:
    """virtual ACP workspace 경로를 host 파일시스템 경로로 해석한다.

    Args:
        path: virtual 경로(예: /mnt/acp-workspace/hello_world.py)
        thread_id: per-thread workspace 해석에 쓰는 현재 thread ID.
                   ``None``이면 전역 workspace로 fallback한다.

    Returns:
        해석된 host 경로.

    Raises:
        FileNotFoundError: ACP workspace 디렉터리가 없을 때.
        PermissionError: path traversal이 탐지될 때.
    """
    _reject_path_traversal(path)

    host_path = _get_acp_workspace_host_path(thread_id)
    if host_path is None:
        raise FileNotFoundError(f"ACP workspace directory not available for path: {path}")

    if path == _ACP_WORKSPACE_VIRTUAL_PATH:
        return host_path

    relative = path[len(_ACP_WORKSPACE_VIRTUAL_PATH) :].lstrip("/")
    resolved = _join_path_preserving_style(host_path, relative)

    if "/" in host_path and "\\" not in host_path:
        base_path = posixpath.normpath(host_path)
        candidate_path = posixpath.normpath(resolved)
        try:
            if posixpath.commonpath([base_path, candidate_path]) != base_path:
                raise PermissionError("Access denied: path traversal detected")
        except ValueError:
            raise PermissionError("Access denied: path traversal detected") from None
        return resolved

    resolved_path = Path(resolved).resolve()
    try:
        resolved_path.relative_to(Path(host_path).resolve())
    except ValueError:
        raise PermissionError("Access denied: path traversal detected")

    return str(resolved_path)


def _get_mcp_allowed_paths() -> list[str]:
    """MCP config에서 filesystem server용 허용 경로 목록을 가져온다."""
    allowed_paths = []
    try:
        from deerflow.config.extensions_config import get_extensions_config

        extensions_config = get_extensions_config()

        for _, server in extensions_config.mcp_servers.items():
            if not server.enabled:
                continue

            # filesystem server만 검사한다.
            args = server.args or []
            # args에 server-filesystem 패키지가 있는지 확인한다.
            has_filesystem = any("server-filesystem" in arg for arg in args)
            if not has_filesystem:
                continue
            # config에 있는 허용 파일시스템 경로를 꺼낸다.
            for arg in args:
                if not arg.startswith("-") and arg.startswith("/"):
                    allowed_paths.append(arg.rstrip("/") + "/")

    except Exception:
        pass

    return allowed_paths


def _get_tool_config_int(name: str, key: str, default: int) -> int:
    try:
        tool_config = get_app_config().get_tool_config(name)
        if tool_config is not None and key in tool_config.model_extra:
            value = tool_config.model_extra.get(key)
            if isinstance(value, int):
                return value
    except Exception:
        pass
    return default


def _clamp_max_results(value: int, *, default: int, upper_bound: int) -> int:
    if value <= 0:
        return default
    return min(value, upper_bound)


def _resolve_max_results(name: str, requested: int, *, default: int, upper_bound: int) -> int:
    requested_max_results = _clamp_max_results(requested, default=default, upper_bound=upper_bound)
    configured_max_results = _clamp_max_results(
        _get_tool_config_int(name, "max_results", default),
        default=default,
        upper_bound=upper_bound,
    )
    return min(requested_max_results, configured_max_results)


def _resolve_local_read_path(path: str, thread_data: ThreadDataState) -> str:
    validate_local_tool_path(path, thread_data, read_only=True)
    if _is_skills_path(path) or _is_acp_workspace_path(path):
        # skills와 ACP workspace 경로는 sandbox의 PathMapping(acquire 시점의 user_id 사용)이
        # 해석한다. contextvar의 get_effective_user_id()를 쓰는
        # _resolve_skills_path / _resolve_acp_workspace_path는 sandbox mapping의 user_id와
        # 다를 수 있으므로 여기서 쓰지 않는다.
        return path
    return _resolve_and_validate_user_data_path(path, thread_data)


def _format_glob_results(root_path: str, matches: list[str], truncated: bool) -> str:
    if not matches:
        return f"No files matched under {root_path}"

    lines = [f"Found {len(matches)} paths under {root_path}"]
    if truncated:
        lines[0] += f" (showing first {len(matches)})"
    lines.extend(f"{index}. {path}" for index, path in enumerate(matches, start=1))
    if truncated:
        lines.append("Results truncated. Narrow the path or pattern to see fewer matches.")
    return "\n".join(lines)


def _format_grep_results(root_path: str, matches: list[GrepMatch], truncated: bool) -> str:
    if not matches:
        return f"No matches found under {root_path}"

    lines = [f"Found {len(matches)} matches under {root_path}"]
    if truncated:
        lines[0] += f" (showing first {len(matches)})"
    lines.extend(f"{match.path}:{match.line_number}: {match.line}" for match in matches)
    if truncated:
        lines.append("Results truncated. Narrow the path or add a glob filter.")
    return "\n".join(lines)


def _path_variants(path: str) -> set[str]:
    return {path, path.replace("\\", "/"), path.replace("/", "\\")}


def _path_separator_for_style(path: str) -> str:
    return "\\" if "\\" in path and "/" not in path else "/"


def _join_path_preserving_style(base: str, relative: str) -> str:
    if not relative:
        return base
    separator = _path_separator_for_style(base)
    normalized_relative = relative.replace("\\" if separator == "/" else "/", separator).lstrip("/\\")
    stripped_base = base.rstrip("/\\")
    return f"{stripped_base}{separator}{normalized_relative}"


def _sanitize_error(error: Exception, runtime: Runtime | None = None) -> str:
    """host 파일시스템 경로가 새어 나가지 않도록 오류 메시지를 정제한다.

    local sandbox 모드에서는 오류 문자열에 담긴 해석된 host 경로를 다시 virtual 경로로
    마스킹해서, 사용자에게 보이는 출력이 host 디렉터리 구조를 절대 드러내지 않게 한다.
    """
    msg = f"{type(error).__name__}: {error}"
    if runtime is not None and is_local_sandbox(runtime):
        thread_data = get_thread_data(runtime)
        msg = mask_local_paths_in_output(msg, thread_data)
    return msg


def _truncate_write_file_error_detail(detail: str, max_chars: int) -> str:
    """write_file 오류 상세를 앞뒤를 남기고 가운데에서 잘라낸다."""
    if max_chars == 0:
        return detail
    if len(detail) <= max_chars:
        return detail
    total = len(detail)
    marker_max_len = len(f"\n... [write_file error truncated: {total} chars skipped] ...\n")
    kept = max(0, max_chars - marker_max_len)
    if kept == 0:
        return detail[:max_chars]
    head_len = kept // 2
    tail_len = kept - head_len
    skipped = total - kept
    marker = f"\n... [write_file error truncated: {skipped} chars skipped] ...\n"
    return f"{detail[:head_len]}{marker}{detail[-tail_len:] if tail_len > 0 else ''}"


def _format_write_file_error(
    requested_path: str,
    error: Exception,
    runtime: Runtime | None = None,
    *,
    max_chars: int = _DEFAULT_WRITE_FILE_ERROR_MAX_CHARS,
) -> str:
    """write_file 실패에 대해 길이가 제한되고 정제된 오류 문자열을 반환한다."""
    header = f"Error: Failed to write file '{requested_path}'"
    detail = _sanitize_error(error, runtime)
    if max_chars == 0:
        return f"{header}: {detail}"
    detail_budget = max_chars - len(header) - 2
    if detail_budget <= 0:
        return _truncate_write_file_error_detail(f"{header}: {detail}", max_chars)
    return f"{header}: {_truncate_write_file_error_detail(detail, detail_budget)}"


def replace_virtual_path(path: str, thread_data: ThreadDataState | None) -> str:
    """virtual /mnt/user-data 경로를 실제 thread data 경로로 치환한다.

    매핑:
        /mnt/user-data/workspace/* -> thread_data['workspace_path']/*
        /mnt/user-data/uploads/* -> thread_data['uploads_path']/*
        /mnt/user-data/outputs/* -> thread_data['outputs_path']/*

    Args:
        path: virtual path prefix를 포함할 수 있는 경로.
        thread_data: 실제 경로들을 담은 thread data.

    Returns:
        virtual prefix가 실제 경로로 치환된 경로.
    """
    if thread_data is None:
        return path

    mappings = _thread_virtual_to_actual_mappings(thread_data)
    if not mappings:
        return path

    # segment 경계를 검사하면서 가장 긴 prefix부터 치환한다.
    for virtual_base, actual_base in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        if path == virtual_base:
            return actual_base
        if path.startswith(f"{virtual_base}/"):
            rest = path[len(virtual_base) :].lstrip("/")
            result = _join_path_preserving_style(actual_base, rest)
            if path.endswith("/") and not result.endswith(("/", "\\")):
                result += _path_separator_for_style(actual_base)
            return result

    return path


def _thread_virtual_to_actual_mappings(thread_data: ThreadDataState) -> dict[str, str]:
    """thread에 대한 virtual → 실제 경로 매핑을 만든다."""
    mappings: dict[str, str] = {}

    workspace = thread_data.get("workspace_path")
    uploads = thread_data.get("uploads_path")
    outputs = thread_data.get("outputs_path")

    if workspace:
        mappings[f"{VIRTUAL_PATH_PREFIX}/workspace"] = workspace
    if uploads:
        mappings[f"{VIRTUAL_PATH_PREFIX}/uploads"] = uploads
    if outputs:
        mappings[f"{VIRTUAL_PATH_PREFIX}/outputs"] = outputs

    # 알려진 디렉터리가 모두 같은 부모를 공유하면 virtual root도 매핑한다.
    actual_dirs = [Path(p) for p in (workspace, uploads, outputs) if p]
    if actual_dirs:
        common_parent = str(Path(actual_dirs[0]).parent)
        if all(str(path.parent) == common_parent for path in actual_dirs):
            mappings[VIRTUAL_PATH_PREFIX] = common_parent

    return mappings


def _thread_actual_to_virtual_mappings(thread_data: ThreadDataState) -> dict[str, str]:
    """output masking에 쓸 실제 → virtual 매핑을 만든다."""
    return {actual: virtual for virtual, actual in _thread_virtual_to_actual_mappings(thread_data).items()}


@lru_cache(maxsize=512)
def _compiled_mask_patterns(sources: tuple[tuple[str, str], ...]) -> tuple[tuple[re.Pattern[str], str, str], ...]:
    """host→virtual masking 패턴을 source 집합마다 한 번만 컴파일한다.

    ``sources``는 ``(host_base, virtual_base)`` 쌍의 순서 있는 tuple이다(skills, ACP
    workspace, 그다음 host 경로 길이 기준 내림차순으로 정렬된 per-thread user-data 매핑).
    패턴은 config에서 안정적인 입력과 per-thread 입력에서만 파생되므로, 매 호출마다
    ``re.escape`` + ``re.compile`` + ``Path.resolve``(syscall)로 다시 만들지 않고 캐시해서
    재사용한다. ``mask_local_paths_in_output``은 glob/grep 매치마다 한 번씩 실행되므로,
    이 캐시가 없으면 매치마다 같은 패턴을 다시 컴파일하게 된다.
    """
    # segment 경계와 경로 tail 규칙은 ``LocalSandbox._reverse_output_patterns``와 공유한다.
    # 그 규칙을 소유한 ``deerflow.sandbox.path_patterns``를 참고. 두 사본이 다시 어긋나지
    # 않도록 한곳에 모아둔 것이다(#4035가 한쪽을 고치면서 다른 쪽을 놓쳤고, #4053이 나머지를
    # 고쳤다).
    #
    # 이 지점이 다르게 하는 것은 ``separator_agnostic=True`` 하나다. 여기의 base는
    # Windows 스타일 표기까지 만들어내는 ``_path_variants``에서 오고, 이 레이어가 구분자를
    # 통제하지 못하는 출력과 매칭되기 때문이다.
    compiled: list[tuple[re.Pattern[str], str, str]] = []
    for host_base, virtual_base in sources:
        seen: set[str] = set()
        # ``_path_variants(raw) | _path_variants(resolved)``와 같은 base 집합이며,
        # 캐시된 tuple이 안정적이도록 결정적으로 정렬한다(한 host의 variant들은 같은
        # virtual로 매핑되고 치환 후에도 겹치지 않으므로, source 내부 순서는 결과에
        # 영향을 주지 않는다).
        for root in (str(Path(host_base)), str(Path(host_base).resolve())):
            for variant in sorted(_path_variants(root)):
                if variant in seen:
                    continue
                seen.add(variant)
                compiled.append((build_output_mask_pattern(variant, separator_agnostic=True), variant, virtual_base))
    return tuple(compiled)


def mask_local_paths_in_output(output: str, thread_data: ThreadDataState | None) -> str:
    """local sandbox 출력의 host 절대 경로를 virtual 경로로 마스킹한다.

    user-data 경로(per-thread), skills 경로(전역 + per-user custom + 관리형 integration),
    ACP workspace 경로(per-thread)를 처리한다.
    """
    # (host_base, virtual_base) source 목록을 순서대로 구성한다. 순서는 원래 구현을
    # 그대로 유지한다: skills, per-user custom/integration skills, ACP workspace,
    # 그다음 user-data 매핑(host 경로가 긴 것부터). custom mount의 host 경로는
    # LocalSandbox._reverse_resolve_paths_in_output()이 마스킹한다.
    sources: list[tuple[str, str]] = []

    skills_host = _get_skills_host_path()
    if skills_host:
        sources.append((skills_host, _get_skills_container_path()))

    # per-user custom skills: 사용자의 custom skills 디렉터리 아래 host 경로를 다시
    # /mnt/skills/custom으로 마스킹한다. sandbox의 _reverse_resolve_path가 자체 작업에
    # 대해 처리하지만, mask_local_paths_in_output은 sandbox 해석을 우회한 출력에 host
    # 경로가 나타나는 엣지 케이스를 위한 안전망 역할을 한다.
    try:
        from deerflow.config.paths import get_paths
        from deerflow.runtime.user_context import get_effective_user_id

        user_id = get_effective_user_id()
        user_custom_dir = get_paths().user_custom_skills_dir(user_id)
        integrations_dir = get_paths().integration_skills_dir()
        if user_custom_dir.exists():
            skills_container = _get_skills_container_path()
            sources.append((str(user_custom_dir), f"{skills_container}/custom"))
        if integrations_dir.exists():
            skills_container = _get_skills_container_path()
            sources.append((str(integrations_dir), f"{skills_container}/integrations"))
    except Exception:
        pass

    acp_host = _get_acp_workspace_host_path(_extract_thread_id_from_thread_data(thread_data))
    if acp_host:
        sources.append((acp_host, _ACP_WORKSPACE_VIRTUAL_PATH))

    if thread_data is not None:
        mappings = _thread_actual_to_virtual_mappings(thread_data)
        for actual_base, virtual_base in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
            sources.append((actual_base, virtual_base))

    if not sources:
        return output

    result = output
    for pattern, base, virtual in _compiled_mask_patterns(tuple(sources)):

        def replace_match(match: re.Match, _base: str = base, _virtual: str = virtual) -> str:
            matched_path = match.group(0)
            if matched_path == _base:
                return _virtual
            relative = matched_path[len(_base) :].lstrip("/\\")
            return f"{_virtual}/{relative}" if relative else _virtual

        result = pattern.sub(replace_match, result)

    return result


def _reject_path_traversal(path: str) -> None:
    """directory traversal을 막기 위해 '..' segment가 포함된 경로를 거부한다."""
    # 슬래시로 정규화한 뒤 '..' segment를 검사한다.
    normalised = path.replace("\\", "/")
    for segment in normalised.split("/"):
        if segment == "..":
            raise PermissionError("Access denied: path traversal detected")


def validate_local_tool_path(path: str, thread_data: ThreadDataState | None, *, read_only: bool = False) -> None:
    """virtual 경로가 local sandbox 접근에 허용되는지 검증한다.

    이 함수는 보안 gate다 — *path*에 접근해도 되는지 검사하고 위반이면 raise한다.
    virtual 경로를 host 경로로 해석하지는 **않는다**. 해석은 호출자가
    ``resolve_and_validate_user_data_path`` 또는 ``_resolve_skills_path``로 처리한다.

    허용되는 virtual 경로 계열:
      - ``/mnt/user-data/*``  — 항상 허용(읽기 + 쓰기)
      - ``/mnt/skills/*``     — *read_only*가 True일 때만 허용
      - ``/mnt/acp-workspace/*`` — *read_only*가 True일 때만 허용
      - custom mount 경로(config.yaml) — mount별 ``read_only`` 플래그를 따른다

    Args:
        path: 검증할 virtual 경로.
        thread_data: thread data(local sandbox에서는 반드시 있어야 한다).
        read_only: True면 skills와 ACP workspace 경로를 허용한다.

    Raises:
        SandboxRuntimeError: thread data가 없을 때.
        PermissionError: 경로가 허용되지 않거나 traversal을 포함할 때.
    """
    if thread_data is None:
        raise SandboxRuntimeError("Thread data not available for local sandbox")

    _reject_path_traversal(path)

    # skills 경로 — 읽기 전용 접근만 허용
    if _is_skills_path(path):
        if not read_only:
            raise PermissionError(f"Write access to skills path is not allowed: {path}")
        return

    # ACP workspace 경로 — 읽기 전용 접근만 허용
    if _is_acp_workspace_path(path):
        if not read_only:
            raise PermissionError(f"Write access to ACP workspace is not allowed: {path}")
        return

    # user-data 경로
    if path.startswith(f"{VIRTUAL_PATH_PREFIX}/"):
        return

    # custom mount 경로 — read_only 설정을 따른다
    if _is_custom_mount_path(path):
        mount = _get_custom_mount_for_path(path)
        if mount and mount.read_only and not read_only:
            raise PermissionError(f"Write access to read-only mount is not allowed: {path}")
        return

    raise PermissionError(f"Only paths under {VIRTUAL_PATH_PREFIX}/, {_get_skills_container_path()}/, {_ACP_WORKSPACE_VIRTUAL_PATH}/, or configured mount paths are allowed")


def _validate_resolved_user_data_path(resolved: Path, thread_data: ThreadDataState) -> None:
    """해석된 host 경로가 허용된 per-thread root 안에 머무는지 확인한다.

    경로가 workspace/uploads/outputs를 벗어나면 PermissionError를 raise한다.
    """
    allowed_roots = [
        Path(p).resolve()
        for p in (
            thread_data.get("workspace_path"),
            thread_data.get("uploads_path"),
            thread_data.get("outputs_path"),
        )
        if p is not None
    ]

    if not allowed_roots:
        raise SandboxRuntimeError("No allowed local sandbox directories configured")

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return
        except ValueError:
            continue

    raise PermissionError("Access denied: path traversal detected")


def _resolve_and_validate_user_data_path(path: str, thread_data: ThreadDataState) -> str:
    """/mnt/user-data virtual 경로를 해석하고 허용 범위 안에 머무는지 검증한다.

    해석된 host 경로 문자열을 반환한다.
    """
    resolved_str = replace_virtual_path(path, thread_data)
    resolved = Path(resolved_str).resolve()
    _validate_resolved_user_data_path(resolved, thread_data)
    return str(resolved)


def _is_non_file_url_token(token: str) -> bool:
    """경로로 해석하면 안 되는 URL token이면 True를 반환한다."""
    values = [token]
    if "=" in token:
        values.append(token.split("=", 1)[1])

    for value in values:
        match = _URL_WITH_SCHEME_PATTERN.match(value)
        if match and not value.lower().startswith("file://"):
            return True
    return False


def _non_file_url_spans(command: str) -> list[tuple[int, int]]:
    spans = []
    for match in _URL_IN_COMMAND_PATTERN.finditer(command):
        if not match.group().lower().startswith("file://"):
            spans.append(match.span())
    return spans


def _is_in_spans(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _has_dotdot_path_segment(token: str) -> bool:
    if _is_non_file_url_token(token):
        return False
    return bool(_DOTDOT_PATH_SEGMENT_PATTERN.search(token))


def _split_shell_tokens(command: str) -> list[str]:
    try:
        normalized = command.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ; ")
        lexer = shlex.shlex(normalized, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        # 잘못된 따옴표는 나중에 shell이 거부한다. 문법 오류를 보안 메시지로 바꾸는 대신
        # 검증을 best-effort로 유지한다.
        return command.split()


def _is_shell_command_separator(token: str) -> bool:
    return token in _SHELL_COMMAND_SEPARATORS


def _is_shell_redirection_operator(token: str) -> bool:
    return token in _SHELL_REDIRECTION_OPERATORS


def _is_shell_assignment(token: str) -> bool:
    name, separator, _ = token.partition("=")
    if not separator or not name:
        return False
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _is_allowed_local_bash_absolute_path(path: str, allowed_paths: list[str], *, allow_system_paths: bool) -> bool:
    # MCP filesystem server의 허용 경로인지 확인한다.
    if any(path.startswith(allowed_path) or path == allowed_path.rstrip("/") for allowed_path in allowed_paths):
        _reject_path_traversal(path)
        return True

    if path == VIRTUAL_PATH_PREFIX or path.startswith(f"{VIRTUAL_PATH_PREFIX}/"):
        _reject_path_traversal(path)
        return True

    # skills container 경로 허용(sandbox로 넘기기 전에 tools.py가 해석한다)
    if _is_skills_path(path):
        _reject_path_traversal(path)
        return True

    # ACP workspace 경로 허용(path traversal 검사만 수행)
    if _is_acp_workspace_path(path):
        _reject_path_traversal(path)
        return True

    # custom mount container 경로 허용
    if _is_custom_mount_path(path):
        _reject_path_traversal(path)
        return True

    if allow_system_paths and any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in _LOCAL_BASH_SYSTEM_PATH_PREFIXES):
        return True

    return False


def _next_cd_target(tokens: list[str], start_index: int) -> tuple[str | None, int]:
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if _is_shell_command_separator(token):
            return None, index
        if _is_shell_redirection_operator(token):
            index += 2
            continue
        if token == "--":
            index += 1
            continue
        if token in {"-L", "-P", "-e", "-@"}:
            index += 1
            continue
        if token.startswith("-") and token != "-":
            index += 1
            continue
        return token, index + 1
    return None, index


def _validate_local_bash_cwd_target(command_name: str, target: str | None, allowed_paths: list[str]) -> None:
    if target is None or target == "-":
        raise PermissionError(f"Unsafe working directory change in command: {command_name}. Use paths under {VIRTUAL_PATH_PREFIX}")
    if target.startswith(("$", "`")):
        raise PermissionError(f"Unsafe working directory change in command: {command_name} {target}. Use paths under {VIRTUAL_PATH_PREFIX}")
    if target.startswith("~"):
        raise PermissionError(f"Unsafe working directory change in command: {command_name} {target}. Use paths under {VIRTUAL_PATH_PREFIX}")
    if target.startswith("/"):
        _reject_path_traversal(target)
        if not _is_allowed_local_bash_absolute_path(target, allowed_paths, allow_system_paths=False):
            raise PermissionError(f"Unsafe working directory change in command: {command_name} {target}. Use paths under {VIRTUAL_PATH_PREFIX}")


def _validate_local_bash_root_path_args(command_name: str, tokens: list[str], start_index: int) -> None:
    if command_name not in _LOCAL_BASH_ROOT_PATH_COMMANDS:
        return

    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if _is_shell_command_separator(token):
            return
        if _is_shell_redirection_operator(token):
            index += 2
            continue
        if token == "/" and not _is_non_file_url_token(token):
            raise PermissionError(f"Unsafe absolute paths in command: /. Use paths under {VIRTUAL_PATH_PREFIX}")
        index += 1


def _validate_local_bash_shell_tokens(command: str, allowed_paths: list[str]) -> None:
    """절대 경로 스캔이 놓친 상대 경로 이탈을 보수적으로 거부한다."""
    if re.search(r"\$\([^)]*\b(?:cd|pushd)\b", command):
        raise PermissionError(f"Unsafe working directory change in command substitution. Use paths under {VIRTUAL_PATH_PREFIX}")

    tokens = _split_shell_tokens(command)

    for token in tokens:
        if _is_shell_command_separator(token) or _is_shell_redirection_operator(token):
            continue
        if _has_dotdot_path_segment(token):
            raise PermissionError("Access denied: path traversal detected")

    at_command_start = True
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if _is_shell_command_separator(token):
            at_command_start = True
            index += 1
            continue

        if _is_shell_redirection_operator(token):
            index += 1
            continue

        if at_command_start and _is_shell_assignment(token):
            index += 1
            continue

        command_name = token.rsplit("/", 1)[-1]
        if at_command_start and command_name in _LOCAL_BASH_COMMAND_PREFIX_KEYWORDS | _LOCAL_BASH_COMMAND_END_KEYWORDS:
            index += 1
            continue

        if not at_command_start:
            index += 1
            continue

        at_command_start = False
        if command_name in _LOCAL_BASH_COMMAND_WRAPPERS and index + 1 < len(tokens):
            wrapped_name = tokens[index + 1].rsplit("/", 1)[-1]
            if wrapped_name in _LOCAL_BASH_CWD_COMMANDS:
                target, next_index = _next_cd_target(tokens, index + 2)
                _validate_local_bash_cwd_target(wrapped_name, target, allowed_paths)
                index = next_index
                continue
            _validate_local_bash_root_path_args(wrapped_name, tokens, index + 2)

        if command_name not in _LOCAL_BASH_CWD_COMMANDS:
            _validate_local_bash_root_path_args(command_name, tokens, index + 1)
            index += 1
            continue

        target, next_index = _next_cd_target(tokens, index + 1)
        _validate_local_bash_cwd_target(command_name, target, allowed_paths)
        index = next_index


def resolve_and_validate_user_data_path(path: str, thread_data: ThreadDataState) -> str:
    """/mnt/user-data virtual 경로를 해석하고 허용 범위 안에 머무는지 검증한다."""
    return _resolve_and_validate_user_data_path(path, thread_data)


def _braces_are_identifier_placeholders_only(fragment: str) -> bool:
    """모든 ``{...}`` 블록이 identifier placeholder 하나일 때만 True를 반환한다.

    identifier만 담은 블록(``{id}``, ``{port}``)은 REST 템플릿과 f-string에서 나온 텍스트다.
    반면 bash brace expansion(``{passwd,shadow}``, ``{,.bak}``, ``{etc,var}``)은 runtime에
    실제 host 경로를 만들어내므로 예외로 두면 안 된다. 짝이 맞지 않거나 비었거나 중첩된
    중괄호도 거부한다(모든 ``{``/``}``는 균형 잡힌 단일 placeholder 블록에 속해야 한다).

    ``${VAR}`` shell 변수 확장(예: ``/home/${USER}/.ssh/id_rsa``)도 runtime에 실제 host
    경로로 확장되므로, 안쪽 이름이 identifier 형태여도 ``${``가 어디든 있으면 이 fragment는
    자격을 잃는다.
    """
    if "${" in fragment:
        return False
    blocks = _IDENTIFIER_BRACE_BLOCK_PATTERN.findall(fragment)
    # 모든 중괄호는 균형 잡힌 ``{...}`` 블록의 일부여야 한다(짝 없는/중첩된 중괄호 금지).
    if fragment.count("{") != len(blocks) or fragment.count("}") != len(blocks):
        return False
    return all(_IDENTIFIER_PATTERN.fullmatch(inner) for inner in blocks)


def _is_non_path_literal_fragment(fragment: str) -> bool:
    """``/segment`` 매치가 경로가 아니라 사실상 텍스트일 때 True를 반환한다.

    절대 경로 스캔은 raw command 문자열 전체에 대해 돌기 때문에, 문자열 리터럴, f-string,
    템플릿 안에 있는 ``/segment`` 시퀀스도 매칭한다(예: ``python -c "print(f'/端口{port}')"``
    또는 ``/devices/{id}/port`` 같은 REST 템플릿). 비ASCII 문자나 identifier 하나짜리
    ``{placeholder}`` 중괄호는 명령이 실제로 열 host 파일시스템 경로에는 나타나지 않으므로,
    그런 fragment를 텍스트로 처리하면 이런 오탐이 사라진다.

    bash brace expansion(``cat /etc/{passwd,shadow}``)은 의도적으로 예외 처리하지 않는다.
    runtime에 평범한 host 경로로 확장되므로, identifier 하나짜리 placeholder인 중괄호만
    텍스트로 취급한다(:func:`_braces_are_identifier_placeholders_only` 참고).

    이 guard는 보안 경계가 아니라 best-effort다(:func:`validate_local_bash_command_paths`
    참고). ``/etc/passwd`` 같은 순수 ASCII host 경로에는 이런 표식이 없으므로 여전히 거부된다.
    """
    if any(ord(ch) > 127 for ch in fragment):
        return True
    if "{" in fragment or "}" in fragment:
        return _braces_are_identifier_placeholders_only(fragment)
    return False


def validate_local_bash_command_paths(command: str, thread_data: ThreadDataState | None) -> None:
    """local sandbox bash 명령의 절대 경로를 검증한다.

    이 검증은 명시적인 ``sandbox.allow_host_bash: true`` opt-in에 대한 best-effort guard일
    뿐이다. 안전한 sandbox 경계가 아니며 host 파일시스템으로부터의 격리로 취급하면 안 된다.

    local 모드에서 명령은 사용자 데이터 접근에 /mnt/user-data 아래 virtual 경로를 써야 한다.
    /mnt/skills 아래 skills 경로, /mnt/acp-workspace 아래 ACP workspace 경로, config.yaml에
    설정된 custom mount container 경로는 허용된다(path traversal 검사만 하며, bash 명령에
    대한 쓰기 차단은 여기서 강제하지 않는다).
    실행 파일과 device 참조(예: /bin/sh, /dev/null)를 위해 흔한 시스템 경로 prefix의 작은
    allowlist를 유지한다.
    """
    if thread_data is None:
        raise SandboxRuntimeError("Thread data not available for local sandbox")

    # 절대 경로 정규식은 우회하면서 로컬 파일 유출은 가능한 file:// URL을 차단한다.
    file_url_match = _FILE_URL_PATTERN.search(command)
    if file_url_match:
        raise PermissionError(f"Unsafe file:// URL in command: {file_url_match.group()}. Use paths under {VIRTUAL_PATH_PREFIX}")

    unsafe_paths: list[str] = []
    allowed_paths = _get_mcp_allowed_paths()
    _validate_local_bash_shell_tokens(command, allowed_paths)
    url_spans = _non_file_url_spans(command)

    for match in _ABSOLUTE_PATH_PATTERN.finditer(command):
        if _is_in_spans(match.start(), url_spans):
            continue
        absolute_path = match.group()
        if _is_non_path_literal_fragment(absolute_path):
            continue
        if _is_allowed_local_bash_absolute_path(absolute_path, allowed_paths, allow_system_paths=True):
            continue

        unsafe_paths.append(absolute_path)

    if unsafe_paths:
        unsafe = ", ".join(sorted(dict.fromkeys(unsafe_paths)))
        raise PermissionError(f"Unsafe absolute paths in command: {unsafe}. Use paths under {VIRTUAL_PATH_PREFIX}")


def replace_virtual_paths_in_command(command: str, thread_data: ThreadDataState | None) -> str:
    """local sandbox를 위해 명령 문자열 안의 /mnt/user-data virtual 경로를 치환한다.

    skills 경로(/mnt/skills)와 ACP workspace 경로(/mnt/acp-workspace)는 여기서 치환하지
    않는다 — 실행 시점에 LocalSandbox._resolve_paths_in_command()가 PathMapping으로
    해석하며, 이는 sandbox acquire에서 얻은 올바른 user_id를 쓴다. _resolve_skills_path /
    _resolve_acp_workspace_path로 미리 해석하면 contextvar의 get_effective_user_id()를
    쓰게 되는데, 이는 sandbox mapping의 user_id와 다를 수 있다.

    Args:
        command: virtual 경로를 포함할 수 있는 명령 문자열.
        thread_data: 실제 경로들을 담은 thread data.

    Returns:
        user-data virtual 경로가 치환된 명령.
    """
    result = command

    # skills, ACP workspace, custom mount 경로는 LocalSandbox._resolve_paths_in_command()가
    # PathMapping으로 해석한다.

    # user-data 경로를 치환한다.
    if VIRTUAL_PATH_PREFIX in result and thread_data is not None:
        # segment 경계 lookahead가 있어야 virtual root가 prefix만 공유하는 형제 경로
        # (``/mnt/user-data-backup`` 안의 ``/mnt/user-data``) 안에서 매칭되지 않는다.
        # 뒤쪽 group은 무언가를 소비하려면 ``/``가 필요하므로, lookahead가 없으면 맨 root가
        # 그대로 매칭되어 형제 경로가 thread의 host 디렉터리로 재작성된다 — mount 계약을
        # 벗어난 실제 경로가 되는 것이다. #4035(reverse pattern), #4053(masking pattern)과
        # 같은 결함을 이 방향으로 옮긴 것이다.
        #
        # 문자 클래스는 ``LocalSandbox._command_pattern``이 아니라 ``_content_pattern``의
        # 것을 따른다. virtual root 뒤에는 ``:``(PATH 스타일 연결)나 ``,``가 정당하게 올 수
        # 있는데, shell 지향 클래스는 이를 거부한다 — 그쪽으로 좁히면 지금 변환되는 경로가
        # 변환되지 않는다. ``$``는 명령이 정확히 root에서 끝나는 경우를 커버한다.
        pattern = re.compile(rf"{re.escape(VIRTUAL_PATH_PREFIX)}(?=/|$|[^\w./-])(/[^\s\"';&|<>()]*)?")

        def replace_user_data_match(match: re.Match) -> str:
            return replace_virtual_path(match.group(0), thread_data).replace("\\", "/")

        result = pattern.sub(replace_user_data_match, result)

    return result


def _apply_cwd_prefix(command: str, thread_data: ThreadDataState | None) -> str:
    """상대 경로가 thread workspace를 기준으로 잡히도록 'cd <workspace> &&'를 앞에 붙인다.

    Args:
        command: 실행할 bash 명령.
        thread_data: workspace 경로를 담은 thread data.

    Returns:
        workspace_path가 있으면 'cd <workspace> &&'가 붙은 명령을, 없으면 원래 명령을
        그대로 반환한다.
    """
    if thread_data and (workspace := thread_data.get("workspace_path")):
        return f"cd {shlex.quote(workspace)} && {command}"
    return command


def get_thread_data(runtime: Runtime | None) -> ThreadDataState | None:
    """runtime state에서 thread_data를 추출한다."""
    if runtime is None:
        return None
    if runtime.state is None:
        return None
    return runtime.state.get("thread_data")


def is_local_sandbox(runtime: Runtime | None) -> bool:
    """현재 sandbox가 local sandbox인지 확인한다.

    thread context 없이 acquire했을 때의 일반 id ``"local"``과, thread를 알게 된 뒤
    :meth:`LocalSandboxProvider.acquire`가 만드는 per-thread id 형식
    ``"local:{user_id}:{thread_id}"``를 모두 받아들인다.
    """
    if runtime is None:
        return False
    if runtime.state is None:
        return False
    # 읽기 전용 분류다. id를 매칭하기만 하므로 fork로 복원된 wrapper는 여기서 버려도
    # 안전하다(이 경로에서는 아무것도 release하지 않는다).
    sandbox_state, _ = unwrap_sandbox(runtime.state.get("sandbox"))
    if sandbox_state is None:
        return False
    sandbox_id = sandbox_state.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        return False
    return sandbox_id == "local" or sandbox_id.startswith("local:")


def sandbox_from_runtime(runtime: Runtime | None = None) -> Sandbox:
    """tool runtime에서 sandbox 인스턴스를 추출한다.

    DEPRECATED: lazy 초기화를 지원하려면 ensure_sandbox_initialized()를 사용한다.
    이 함수는 sandbox가 이미 초기화되었다고 가정하며, 아니면 오류를 raise한다.

    Raises:
        SandboxRuntimeError: runtime을 쓸 수 없거나 sandbox state가 없을 때.
        SandboxNotFoundError: 주어진 ID의 sandbox를 찾을 수 없을 때.
    """
    if runtime is None:
        raise SandboxRuntimeError("Tool runtime not available")
    if runtime.state is None:
        raise SandboxRuntimeError("Tool runtime state not available")
    # 읽기 전용 조회다. provider 항목만 해석하며, 소유권(release)은 wrapping된 state에
    # 대한 after_agent의 short-circuit이 계속 가진다.
    sandbox_state, _ = unwrap_sandbox(runtime.state.get("sandbox"))
    if sandbox_state is None:
        raise SandboxRuntimeError("Sandbox state not initialized in runtime")
    sandbox_id = sandbox_state.get("sandbox_id")
    if sandbox_id is None:
        raise SandboxRuntimeError("Sandbox ID not found in state")
    sandbox = get_sandbox_provider().get(sandbox_id)
    if sandbox is None:
        raise SandboxNotFoundError(f"Sandbox with ID '{sandbox_id}' not found", sandbox_id=sandbox_id)

    if runtime.context is not None:
        runtime.context["sandbox_id"] = sandbox_id  # 이후 단계에서 쓰도록 sandbox_id를 context에 보장한다
    return sandbox


def ensure_sandbox_initialized(runtime: Runtime | None = None) -> Sandbox:
    """sandbox가 초기화되어 있는지 보장하고, 필요하면 lazy하게 acquire한다.

    첫 호출에서 provider로부터 sandbox를 acquire해 runtime state에 저장한다. 이후 호출은
    기존 sandbox를 반환한다.

    thread 안전성은 provider의 내부 락 메커니즘이 보장한다.

    Args:
        runtime: state와 context를 담은 tool runtime.

    Returns:
        초기화된 sandbox 인스턴스.

    Raises:
        SandboxRuntimeError: runtime을 쓸 수 없거나 thread_id가 없을 때.
        SandboxNotFoundError: sandbox acquire에 실패했을 때.
    """
    if runtime is None:
        raise SandboxRuntimeError("Tool runtime not available")

    if runtime.state is None:
        raise SandboxRuntimeError("Tool runtime state not available")

    # state에 sandbox가 이미 있는지 확인한다.
    # fork_restored를 버려도 안전하다. after_agent가 context 기반 release 분기보다 먼저
    # 아직 wrapping된 state에서 short-circuit하므로, 이 재사용 경로는 부모 sandbox를
    # 절대 release하지 않는다.
    sandbox_state, _ = unwrap_sandbox(runtime.state.get("sandbox"))
    if sandbox_state is not None:
        sandbox_id = sandbox_state.get("sandbox_id")
        if sandbox_id is not None:
            sandbox = get_sandbox_provider().get(sandbox_id)
            if sandbox is not None:
                if runtime.context is not None:
                    runtime.context["sandbox_id"] = sandbox_id  # after_agent에서 release할 수 있도록 sandbox_id를 context에 보장한다
                return sandbox
            # sandbox가 이미 release되었으므로 아래로 내려가 새로 acquire한다.

    # lazy acquire: thread_id를 얻어 sandbox를 acquire한다.
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id is None:
        thread_id = runtime.config.get("configurable", {}).get("thread_id") if runtime.config else None
    if thread_id is None:
        raise SandboxRuntimeError("Thread ID not available in runtime context")

    provider = get_sandbox_provider()
    sandbox_id = provider.acquire(thread_id, user_id=resolve_runtime_user_id(runtime))

    # runtime state를 갱신한다 — 이 값은 tool call 간에 유지된다.
    runtime.state["sandbox"] = {"sandbox_id": sandbox_id}

    # sandbox를 조회해 반환한다.
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        raise SandboxNotFoundError("Sandbox not found after acquisition", sandbox_id=sandbox_id)

    if runtime.context is not None:
        runtime.context["sandbox_id"] = sandbox_id  # after_agent에서 release할 수 있도록 sandbox_id를 context에 보장한다
    return sandbox


async def ensure_sandbox_initialized_async(runtime: Runtime | None = None) -> Sandbox:
    """tool runtime을 위한 ``ensure_sandbox_initialized``의 async 대응 함수.

    lazy sandbox acquire를 async provider hook에 유지하므로, async tool 실행 중에 AIO
    sandbox 시작과 readiness 폴링이 동기 ``provider.acquire()``로 떨어지지 않는다.
    """
    if runtime is None:
        raise SandboxRuntimeError("Tool runtime not available")

    if runtime.state is None:
        raise SandboxRuntimeError("Tool runtime state not available")

    # 위 동기 경로와 동일하게 버린다. after_agent가 아직 wrapping된 state에서 먼저
    # short-circuit하므로, 재사용 경로는 절대 release하지 않는다.
    sandbox_state, _ = unwrap_sandbox(runtime.state.get("sandbox"))
    if sandbox_state is not None:
        sandbox_id = sandbox_state.get("sandbox_id")
        if sandbox_id is not None:
            sandbox = get_sandbox_provider().get(sandbox_id)
            if sandbox is not None:
                if runtime.context is not None:
                    runtime.context["sandbox_id"] = sandbox_id
                return sandbox

    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id is None:
        thread_id = runtime.config.get("configurable", {}).get("thread_id") if runtime.config else None
    if thread_id is None:
        raise SandboxRuntimeError("Thread ID not available in runtime context")

    provider = get_sandbox_provider()
    sandbox_id = await provider.acquire_async(thread_id, user_id=resolve_runtime_user_id(runtime))

    runtime.state["sandbox"] = {"sandbox_id": sandbox_id}

    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        raise SandboxNotFoundError("Sandbox not found after acquisition", sandbox_id=sandbox_id)

    if runtime.context is not None:
        runtime.context["sandbox_id"] = sandbox_id
    return sandbox


async def _run_sync_tool_after_async_sandbox_init(
    func: Callable[..., str] | None,
    runtime: Runtime,
    *args: object,
) -> str:
    """async provider로 lazy 초기화한 뒤, 동기 tool 본문을 별도 thread에서 실행한다."""
    try:
        await ensure_sandbox_initialized_async(runtime)
    except SandboxError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Unexpected error initializing sandbox: {_sanitize_error(e, runtime)}"

    if func is None:
        return "Error: Tool implementation not available"

    return await asyncio.to_thread(func, runtime, *args)


def ensure_thread_directories_exist(runtime: Runtime | None) -> None:
    """thread data 디렉터리(workspace, uploads, outputs)가 존재하도록 보장한다.

    sandbox 도구를 처음 쓸 때 lazy하게 호출된다.
    local sandbox에서는 파일시스템에 디렉터리를 만든다.
    aio 같은 다른 sandbox에서는 디렉터리가 이미 container에 마운트되어 있다.

    Args:
        runtime: state와 context를 담은 tool runtime.
    """
    if runtime is None:
        return

    # local sandbox에서만 디렉터리를 만든다.
    if not is_local_sandbox(runtime):
        return

    thread_data = get_thread_data(runtime)
    if thread_data is None:
        return

    # 디렉터리가 이미 만들어졌는지 확인한다.
    if runtime.state.get("thread_directories_created"):
        return

    # 세 디렉터리를 만든다.
    import os

    for key in ["workspace_path", "uploads_path", "outputs_path"]:
        path = thread_data.get(key)
        if path:
            os.makedirs(path, exist_ok=True)

    # 중복 작업을 피하기 위해 생성 완료로 표시한다.
    runtime.state["thread_directories_created"] = True


_SECRET_REDACTION = "[redacted]"

# 이보다 짧은 값은 bash 출력에서 마스킹하지 않는다. 짧은 secret 값(2글자 지역 코드,
# 숫자 id, PIN)은 exit code, timestamp, 크기, 경로 등 tool 출력의 무관한 바이트를 갈기갈기
# 찢어서 모델이 다시 읽는 결과를 망가뜨린다. 이 정도로 짧은 값을 마스킹하는 것은 실질적인
# 유출 방지보다 노이즈에 가깝다. secret은 여전히 subprocess에 주입되며 출력 마스킹만
# 건너뛴다.
_MIN_MASK_LENGTH = 8


def mask_secret_values(output: str, injected_env: dict[str, str] | None) -> str:
    """bash 출력이 context로 다시 들어가기 전에 주입된 secret 값을 가린다.

    skill script는 request-scoped secret을 env var로 받는다(#3861). script가 그 값을
    출력하면(디버깅, ``set -x``, 오류 덤프) 값이 tool 결과로 흘러 들어가고, 결국 prompt와
    trace까지 도달한다. 이것이 skill 고유의 다섯 번째 유출 지점이다(MCP tool과 달리 bash
    tool은 subprocess stdout을 반환한다). 비어 있지 않은 각 secret 값을 마스킹 마커로
    치환한다. 어떤 값이 다른 값의 부분 문자열일 때 일부만 드러나지 않도록 긴 값부터
    처리한다. ``_MIN_MASK_LENGTH``보다 짧은 값은 건너뛴다 — 3글자 토큰을 마스킹하는 것은
    실제 secret을 보호하기보다 무관한 출력을 망가뜨릴 가능성이 크다.
    """
    if not injected_env or not output:
        return output
    for value in sorted((v for v in injected_env.values() if v and len(v) >= _MIN_MASK_LENGTH), key=len, reverse=True):
        output = output.replace(value, _SECRET_REDACTION)
    return output


def _truncate_bash_output(output: str, max_chars: int) -> str:
    """bash 출력을 앞뒤(50/50)를 남기고 가운데에서 잘라낸다.

    bash 출력은 양쪽 끝 어디에나 오류가 있을 수 있으므로(stderr/stdout 순서가 비결정적),
    양쪽 끝을 균등하게 남긴다.

    반환 문자열은 truncation 마커를 포함해 max_chars를 넘지 않는다. max_chars=0을 넘기면
    truncation을 끄고 전체 출력을 그대로 반환한다.
    """
    if max_chars == 0:
        return output
    if len(output) <= max_chars:
        return output
    total_len = len(output)
    # 최악의 경우 마커 길이를 정확히 계산한다. 건너뛴 문자 수는 최대 total_len이므로
    # 이 값이 빡빡한 상한이 된다.
    marker_max_len = len(f"\n... [middle truncated: {total_len} chars skipped] ...\n")
    kept = max(0, max_chars - marker_max_len)
    if kept == 0:
        return output[:max_chars]
    head_len = kept // 2
    tail_len = kept - head_len
    skipped = total_len - kept
    marker = f"\n... [middle truncated: {skipped} chars skipped] ...\n"
    return f"{output[:head_len]}{marker}{output[-tail_len:] if tail_len > 0 else ''}"


def _truncate_read_file_output(output: str, max_chars: int) -> str:
    """read_file 출력을 앞부분을 남기고 뒤에서 잘라낸다.

    소스 코드와 문서는 위에서 아래로 읽으며, 앞부분에 가장 많은 맥락(import, 클래스 정의,
    함수 시그니처)이 담긴다.

    반환 문자열은 truncation 마커를 포함해 max_chars를 넘지 않는다. max_chars=0을 넘기면
    truncation을 끄고 전체 출력을 그대로 반환한다.
    """
    if max_chars == 0:
        return output
    if len(output) <= max_chars:
        return output
    total = len(output)
    # 최악의 경우 마커 길이를 정확히 계산한다. 두 숫자 필드가 모두 최대값(전체 문자 수)일
    # 때이므로 이 값이 빡빡한 상한이 된다.
    marker_max_len = len(f"\n... [truncated: showing first {total} of {total} chars. Use start_line/end_line to read a specific range] ...")
    kept = max(0, max_chars - marker_max_len)
    if kept == 0:
        return output[:max_chars]
    marker = f"\n... [truncated: showing first {kept} of {total} chars. Use start_line/end_line to read a specific range] ..."
    return f"{output[:kept]}{marker}"


def _truncate_ls_output(output: str, max_chars: int) -> str:
    """ls 출력을 앞부분을 남기고 뒤에서 잘라낸다.

    디렉터리 목록은 위에서 아래로 읽으며, 앞부분이 가장 관련 있는 구조를 보여준다.

    반환 문자열은 truncation 마커를 포함해 max_chars를 넘지 않는다. max_chars=0을 넘기면
    truncation을 끄고 전체 출력을 그대로 반환한다.
    """
    if max_chars == 0:
        return output
    if len(output) <= max_chars:
        return output
    total = len(output)
    marker_max_len = len(f"\n... [truncated: showing first {total} of {total} chars. Use a more specific path to see fewer results] ...")
    kept = max(0, max_chars - marker_max_len)
    if kept == 0:
        return output[:max_chars]
    marker = f"\n... [truncated: showing first {kept} of {total} chars. Use a more specific path to see fewer results] ..."
    return f"{output[:kept]}{marker}"


# IM channel 플랫폼 user id(Feishu open_id, Slack Uxxx, ...)를 sandbox 명령에 노출하는
# 고정 env var. skill이 현재 최종 사용자의 channel identity를 기준으로 동작할 수 있게
# 한다(#3914). secret이 아니라 식별자다.
CHANNEL_USER_ID_ENV = "DEERFLOW_CHANNEL_USER_ID"

_CHANNEL_USER_ID_CONTEXT_KEY = "channel_user_id"

# Gateway는 내부적으로 인증된 channel 요청에서만 이 identity를 받아들이지만, 임베디드
# runtime은 여전히 context를 직접 구성할 수 있다. 값을 방어적으로 제한한다. 실제 플랫폼
# id는 수십 글자 수준이며, 이를 넘는 값은 손상된 것이므로 모든 sandbox 명령 문자열을
# 부풀리게 두어서는 안 된다.
_CHANNEL_USER_ID_MAX_LEN = 256


def _is_windows() -> bool:
    return os.name == "nt"


def _channel_identity_prefix(runtime: Runtime) -> str | None:
    """channel-user-id env var를 설정하거나 지우는 명령 prefix를 만든다.

    IM이 아닌 run(context에 ``channel_user_id`` key가 없음)에는 ``None``을 반환해서 명령을
    그대로 둔다. IM run에는 항상 prefix를 내보낸다:

    - 유효한 id(길이 제한 안의 비어 있지 않은 str) → ``export VAR=<quoted>; ``
    - 쓸 수 없는 id(빈 값 / str이 아님 / 길이 초과) → ``unset VAR; ``

    이 id는 ``execute_command(env=...)`` 채널이 아니라 명령 문자열에 태우도록 의도적으로
    설계했다. 비어 있지 않은 ``env``는 ``AioSandbox``를 ``bash.exec`` API(호출마다 새 session,
    image >= 1.9.3 필요)로 전환시키는데, 그 경로는 request-scoped secret 전용이다. 모든 IM
    명령에 명시적인 ``export`` 또는 ``unset``을 내보내면 **AIO shell의 session 의미론에
    의존하지 않고** 호출 단위 identity가 정확해진다. AIO의 env 없는 경로는 영속 shell
    session을 재사용하므로(클래스 락의 이유, #1433), 아무 처리 없는 명령은 공유된
    group chat sandbox에서 앞선 발신자가 export한 낡은 값을 그대로 읽을 수 있다.
    ``unset``은 길이/타입 guard가 열어둘 뻔한 구멍을 막는다 — id가 버려진 발신자가 이전
    발신자의 값을 물려받는 상황이다. 값은 secret이 아니라 식별자이므로 감사에 노출되는
    명령 문자열에 남겨도 문제없다.
    """
    context = getattr(runtime, "context", None)
    if not isinstance(context, dict) or _CHANNEL_USER_ID_CONTEXT_KEY not in context:
        return None
    channel_user_id = context.get(_CHANNEL_USER_ID_CONTEXT_KEY)
    if isinstance(channel_user_id, str) and 0 < len(channel_user_id) <= _CHANNEL_USER_ID_MAX_LEN:
        return f"export {CHANNEL_USER_ID_ENV}={shlex.quote(channel_user_id)}; "
    return f"unset {CHANNEL_USER_ID_ENV}; "


def _github_env_from_runtime(runtime: Runtime) -> dict[str, str] | None:
    """GitHub App installation token을 실어 나르는 호출 단위 env overlay를 만든다.

    GitHub channel은 ``ChannelManager``(app 레이어)에서 짧은 수명의 installation token을
    발급하고 ``run_context``를 통해 넘겨서 ``runtime.context["github_token"]``에 도달하게
    한다. 여기서는 이를 agent의 bash에 ``GH_TOKEN``(``gh`` CLI가 읽는 이름)과
    ``GITHUB_TOKEN``(관례적인 이름) 양쪽으로 노출한다. token이 없으면 ``None``을 반환하므로
    GitHub가 아닌 run은 이전과 동일하게 동작한다.

    ``runtime.context["github_token"]``의 값은 다음 둘 중 하나다:

    * ``str`` — 캡처된 token. 테스트와 refresh가 필요 없는 오래된 코드 경로가 쓰는
      단순한 형태다.
    * ``str``을 반환하는 인자 없는 동기 callable — 기반 installation token의 1시간 TTL이
      만료에 가까워지면 투명하게 재발급하는 provider다. provider의 캐시 로직은 app 쪽에
      있으며(캐시 + 여유 시간 의미론은 ``app.gateway.github.app_auth.mint_installation_token``
      참고), harness는 호출만 한다.

    callable 경로 덕분에 긴 자율 run이 60분짜리 installation token 수명을 넘겨 살아남는다.
    bash 호출마다 provider에 다시 물어보고, provider는 약 55분까지는 캐시된 token을 주다가
    이후 새로 발급한다. 이것이 없으면 몇 시간짜리 리팩터링을 하는 coder agent가 작업을
    거의 다 끝내놓고 마지막 ``git push``에서 401을 받는다.

    token은 여전히 불투명한 데이터로서 harness/app 경계를 넘는다 — harness는 app 레이어의
    발급 코드를 절대 import하지 않으므로 ``tests/test_harness_boundary.py``가 강제하는
    의존성 방화벽이 유지된다.
    """
    context = runtime.context if runtime.context is not None else None
    value = context.get("github_token") if context else None
    if callable(value):
        try:
            token = value()
        except Exception:
            logger.warning("github_token provider raised; skipping env overlay", exc_info=True)
            return None
    else:
        token = value
    if not isinstance(token, str) or not token:
        return None
    return {"GH_TOKEN": token, "GITHUB_TOKEN": token}


_LARK_CLI_COMMAND_RE = re.compile(r"(?<![A-Za-z0-9_.-])lark-cli(?![A-Za-z0-9_.-])")


def _lark_cli_env_from_runtime(runtime: Runtime, command: str, *, sandbox_paths: bool) -> dict[str, str] | None:
    """Settings 페이지의 Lark 인증을 sandbox의 ``lark-cli`` 명령에 노출한다.

    Settings는 DeerFlow의 per-user integration config/data 디렉터리 아래에서 ``lark-cli``를
    인가한다. agent 대화는 sandbox를 통해 ``lark-cli``를 호출하므로, lark 명령이 같은
    디렉터리를 받지 못하면 무관한 미인증 프로필을 보게 된다. 실제로 ``lark-cli``를 부르는
    명령에만 적용해서, 평범한 bash 호출이 AIO를 env를 실어 나르는 실행 경로로 전환시키지
    않게 한다.

    broker 모드(Pattern B, issue #4338)에서는 sidecar가 credential을 소유하므로, overlay는
    broker URL과 runtime PATH만 담는다 — config/data 디렉터리는 sandbox에 절대 주입되지
    않는다.
    """
    if not _LARK_CLI_COMMAND_RE.search(command):
        return None
    try:
        from deerflow.integrations.lark_cli import lark_cli_env_overlay, sandbox_lark_broker_active

        broker = sandbox_paths and sandbox_lark_broker_active()
        return lark_cli_env_overlay(resolve_runtime_user_id(runtime), sandbox_paths=sandbox_paths, broker=broker)
    except Exception:
        logger.warning("Could not build Lark CLI env overlay; running command without managed auth", exc_info=True)
        return None


@tool("bash", parse_docstring=True)
def bash_tool(runtime: Runtime, description: str, command: str) -> str:
    """Linux 환경에서 bash 명령을 실행한다.


    - Python 코드를 실행할 때는 `python`을 사용하라.
    - `/mnt/user-data/workspace/.venv`의 thread 전용 virtual environment를 우선 사용하라.
    - Python 패키지 설치는 (virtual environment 안에서) `python -m pip`로 하라.
    - web server처럼 오래 실행되는 프로세스는 ALWAYS 출력을 리다이렉트해 background로 실행하라.
      예: `your-command > /mnt/user-data/workspace/server.log 2>&1 &`. 그다음 로그 파일을 확인하거나
      포트를 폴링하라. foreground로 실행한 장기 프로세스는 command timeout으로 종료될 때까지 해당 턴을
      차단한다.

    Args:
        description: 이 명령을 실행하는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        command: 실행할 bash 명령. 파일과 디렉터리는 항상 절대 경로로 지정하라.
    """
    try:
        sandbox = ensure_sandbox_initialized(runtime)
        # 활성 skill에 대해 해석된 request-scoped secret(#3861)과, GitHub channel이 넘겨준
        # 짧은 수명의 GitHub App installation token. 둘 다 호출 단위 env로 subprocess에
        # 주입되며 명령 문자열에는 절대 넣지 않는다.
        injected_env = read_active_secrets(getattr(runtime, "context", None)) or None
        identity_prefix = _channel_identity_prefix(runtime)
        github_env = _github_env_from_runtime(runtime)
        lark_cli_env = _lark_cli_env_from_runtime(runtime, command, sandbox_paths=not is_local_sandbox(runtime))
        if github_env:
            injected_env = {**(injected_env or {}), **github_env}
        if lark_cli_env:
            injected_env = {**(injected_env or {}), **lark_cli_env}
        if is_local_sandbox(runtime):
            if not is_host_bash_allowed():
                return f"Error: {LOCAL_HOST_BASH_DISABLED_MESSAGE}"
            ensure_thread_directories_exist(runtime)
            thread_data = get_thread_data(runtime)
            validate_local_bash_command_paths(command, thread_data)
            command = replace_virtual_paths_in_command(command, thread_data)
            command = _apply_cwd_prefix(command, thread_data)
            # POSIX 전용. Windows local sandbox는 `export`가 유효한 문법이 아닌
            # PowerShell/cmd.exe로 실행될 수 있다.
            if identity_prefix and not _is_windows():
                command = identity_prefix + command
            try:
                from deerflow.config.app_config import get_app_config

                sandbox_cfg = get_app_config().sandbox
                max_chars = sandbox_cfg.bash_output_max_chars if sandbox_cfg else 20000
                command_timeout = sandbox_cfg.bash_command_timeout if sandbox_cfg else None
            except Exception:
                max_chars = 20000
                command_timeout = None
            output = sandbox.execute_command(command, env=injected_env, timeout=command_timeout)
            return _truncate_bash_output(
                mask_secret_values(mask_local_paths_in_output(output, thread_data), injected_env),
                max_chars,
            )
        ensure_thread_directories_exist(runtime)
        command = f"cd {VIRTUAL_PATH_PREFIX}/workspace; {command}"
        if identity_prefix:
            command = identity_prefix + command
        try:
            from deerflow.config.app_config import get_app_config

            sandbox_cfg = get_app_config().sandbox
            max_chars = sandbox_cfg.bash_output_max_chars if sandbox_cfg else 20000
        except Exception:
            max_chars = 20000
        return _truncate_bash_output(mask_secret_values(sandbox.execute_command(command, env=injected_env), injected_env), max_chars)
    except SandboxError as e:
        return f"Error: {e}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Unexpected error executing command: {_sanitize_error(e, runtime)}"


async def _bash_tool_async(runtime: Runtime, description: str, command: str) -> str:
    return await _run_sync_tool_after_async_sandbox_init(bash_tool.func, runtime, description, command)


bash_tool.coroutine = _bash_tool_async


@tool("ls", parse_docstring=True)
def ls_tool(runtime: Runtime, description: str, path: str) -> str:
    """디렉터리 내용을 최대 2단계 깊이까지 tree 형식으로 나열한다.

    Args:
        description: 이 디렉터리를 나열하는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        path: 나열할 디렉터리의 **절대** 경로.
    """
    try:
        user_id = resolve_runtime_user_id(runtime)
        # 비활성화된 skill 디렉터리 접근을 차단한다.
        if _is_disabled_skill_path(path, user_id=user_id):
            skill_name = _extract_skill_name_from_skills_path(path) or "unknown"
            return f"Error: Skill '{skill_name}' is disabled. Access to its files is blocked. Enable the skill in settings before using it."
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        requested_path = path
        thread_data = None
        if is_local_sandbox(runtime):
            thread_data = get_thread_data(runtime)
            validate_local_tool_path(path, thread_data, read_only=True)
            if _is_skills_path(path) or _is_acp_workspace_path(path):
                # skills와 ACP workspace 경로는 sandbox의 PathMapping(acquire 시점의
                # user_id 사용)이 해석한다. contextvar의 get_effective_user_id()를 쓰는
                # _resolve_skills_path / _resolve_acp_workspace_path는 sandbox mapping의
                # user_id와 다를 수 있으므로 여기서 쓰지 않는다.
                pass
            elif not _is_custom_mount_path(path):
                path = _resolve_and_validate_user_data_path(path, thread_data)
            # custom mount 경로와 skills/ACP 경로는 LocalSandbox._resolve_path()가 해석한다.
        children = sandbox.list_dir(path)
        if not children:
            return "(empty)"
        output = "\n".join(children)
        if thread_data is not None:
            output = mask_local_paths_in_output(output, thread_data)
        # 위 gate는 `path` 자체만 막는다. 목록은 하위로 내려가므로, 비활성 skill 위쪽을
        # root로 잡으면 여전히 그 파일이 노출된다.
        entries = _drop_disabled_skill_paths(output.splitlines(), user_id=user_id)
        if not entries:
            return "(empty)"
        output = "\n".join(entries)
        try:
            from deerflow.config.app_config import get_app_config

            sandbox_cfg = get_app_config().sandbox
            max_chars = sandbox_cfg.ls_output_max_chars if sandbox_cfg else 20000
        except Exception:
            max_chars = 20000
        return _truncate_ls_output(output, max_chars)
    except SandboxError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: Directory not found: {requested_path}"
    except PermissionError:
        return f"Error: Permission denied: {requested_path}"
    except Exception as e:
        return f"Error: Unexpected error listing directory: {_sanitize_error(e, runtime)}"


async def _ls_tool_async(runtime: Runtime, description: str, path: str) -> str:
    return await _run_sync_tool_after_async_sandbox_init(ls_tool.func, runtime, description, path)


ls_tool.coroutine = _ls_tool_async


@tool("glob", parse_docstring=True)
def glob_tool(
    runtime: Runtime,
    description: str,
    pattern: str,
    path: str,
    include_dirs: bool = False,
    max_results: int = _DEFAULT_GLOB_MAX_RESULTS,
) -> str:
    """root 디렉터리 아래에서 glob pattern과 일치하는 파일이나 디렉터리를 찾는다.

    Args:
        description: 이 경로들을 검색하는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        pattern: root 경로 기준 상대 glob pattern. 예: `**/*.py`.
        path: 검색 대상이 되는 **절대** root 디렉터리.
        include_dirs: 일치하는 디렉터리도 함께 반환할지 여부. 기본값은 False.
        max_results: 반환할 최대 경로 수. 기본값은 200.
    """
    try:
        user_id = resolve_runtime_user_id(runtime)
        # 비활성화된 skill 디렉터리 접근을 차단한다.
        if _is_disabled_skill_path(path, user_id=user_id):
            skill_name = _extract_skill_name_from_skills_path(path) or "unknown"
            return f"Error: Skill '{skill_name}' is disabled. Access to its files is blocked. Enable the skill in settings before using it."
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        requested_path = path
        effective_max_results = _resolve_max_results(
            "glob",
            max_results,
            default=_DEFAULT_GLOB_MAX_RESULTS,
            upper_bound=_MAX_GLOB_MAX_RESULTS,
        )
        thread_data = None
        if is_local_sandbox(runtime):
            thread_data = get_thread_data(runtime)
            if thread_data is None:
                raise SandboxRuntimeError("Thread data not available for local sandbox")
            path = _resolve_local_read_path(path, thread_data)
        matches, truncated = sandbox.glob(path, pattern, include_dirs=include_dirs, max_results=effective_max_results)
        if thread_data is not None:
            matches = [mask_local_paths_in_output(match, thread_data) for match in matches]
        # 위 gate는 `path` 자체만 막는다. 검색은 하위로 내려가므로, 비활성 skill 위쪽을
        # root로 잡으면 여전히 그 파일이 드러난다.
        matches = _drop_disabled_skill_paths(matches, user_id=user_id)
        return _format_glob_results(requested_path, matches, truncated)
    except SandboxError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: Directory not found: {requested_path}"
    except NotADirectoryError:
        return f"Error: Path is not a directory: {requested_path}"
    except PermissionError:
        return f"Error: Permission denied: {requested_path}"
    except Exception as e:
        return f"Error: Unexpected error searching paths: {_sanitize_error(e, runtime)}"


async def _glob_tool_async(
    runtime: Runtime,
    description: str,
    pattern: str,
    path: str,
    include_dirs: bool = False,
    max_results: int = _DEFAULT_GLOB_MAX_RESULTS,
) -> str:
    return await _run_sync_tool_after_async_sandbox_init(
        glob_tool.func,
        runtime,
        description,
        pattern,
        path,
        include_dirs,
        max_results,
    )


glob_tool.coroutine = _glob_tool_async


@tool("grep", parse_docstring=True)
def grep_tool(
    runtime: Runtime,
    description: str,
    pattern: str,
    path: str,
    glob: str | None = None,
    literal: bool = False,
    case_sensitive: bool = False,
    max_results: int = _DEFAULT_GREP_MAX_RESULTS,
) -> str:
    """텍스트 파일 하나 또는 root 디렉터리 아래 파일들에서 일치하는 줄을 검색한다.

    Args:
        description: 파일 내용을 검색하는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        pattern: 검색할 문자열 또는 regex pattern.
        path: 검색 대상 파일 또는 root 디렉터리의 **절대** 경로.
        glob: 후보 파일을 걸러낼 선택적 glob filter. 예: `**/*.py`.
        literal: `pattern`을 일반 문자열로 취급할지 여부. 기본값은 False.
        case_sensitive: 대소문자를 구분해 매칭할지 여부. 기본값은 False.
        max_results: 반환할 최대 일치 줄 수. 기본값은 100.
    """
    try:
        user_id = resolve_runtime_user_id(runtime)
        # 비활성화된 skill 디렉터리 접근을 차단한다.
        if _is_disabled_skill_path(path, user_id=user_id):
            skill_name = _extract_skill_name_from_skills_path(path) or "unknown"
            return f"Error: Skill '{skill_name}' is disabled. Access to its files is blocked. Enable the skill in settings before using it."
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        requested_path = path
        effective_max_results = _resolve_max_results(
            "grep",
            max_results,
            default=_DEFAULT_GREP_MAX_RESULTS,
            upper_bound=_MAX_GREP_MAX_RESULTS,
        )
        thread_data = None
        if is_local_sandbox(runtime):
            thread_data = get_thread_data(runtime)
            if thread_data is None:
                raise SandboxRuntimeError("Thread data not available for local sandbox")
            path = _resolve_local_read_path(path, thread_data)
        matches, truncated = sandbox.grep(
            path,
            pattern,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_results=effective_max_results,
        )
        if thread_data is not None:
            matches = [
                GrepMatch(
                    path=mask_local_paths_in_output(match.path, thread_data),
                    line_number=match.line_number,
                    line=match.line,
                )
                for match in matches
            ]
        # 위 gate는 `path` 자체만 막는다. 검색은 하위로 내려가므로, 비활성 skill 위쪽을
        # root로 잡으면 여전히 그 파일 내용이 드러난다.
        allowed = set(_drop_disabled_skill_paths([match.path for match in matches], user_id=user_id))
        matches = [match for match in matches if match.path in allowed]
        return _format_grep_results(requested_path, matches, truncated)
    except SandboxError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: Directory not found: {requested_path}"
    except NotADirectoryError:
        return f"Error: Path is not a directory: {requested_path}"
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    except PermissionError:
        return f"Error: Permission denied: {requested_path}"
    except Exception as e:
        return f"Error: Unexpected error searching file contents: {_sanitize_error(e, runtime)}"


async def _grep_tool_async(
    runtime: Runtime,
    description: str,
    pattern: str,
    path: str,
    glob: str | None = None,
    literal: bool = False,
    case_sensitive: bool = False,
    max_results: int = _DEFAULT_GREP_MAX_RESULTS,
) -> str:
    return await _run_sync_tool_after_async_sandbox_init(
        grep_tool.func,
        runtime,
        description,
        pattern,
        path,
        glob,
        literal,
        case_sensitive,
        max_results,
    )


grep_tool.coroutine = _grep_tool_async


def read_current_file_content(runtime: Runtime | None, path: str) -> str:
    """read_file의 해석 규칙을 그대로 써서 ``path``의 현재 전체 내용을 읽는다.

    ``read_file_tool``과 ``ReadBeforeWriteMiddleware``(issue #3857)가 공유하므로, gate가
    read 도구가 보게 될 바이트를 그대로 해시한다. 파일이 없으면 ``FileNotFoundError``를
    raise하고, 다른 sandbox 오류는 호출자에게 전파된다.
    """
    sandbox = ensure_sandbox_initialized(runtime)
    ensure_thread_directories_exist(runtime)
    if is_local_sandbox(runtime):
        thread_data = get_thread_data(runtime)
        validate_local_tool_path(path, thread_data, read_only=True)
        if _is_skills_path(path):
            path = _resolve_skills_path(path)
        elif _is_acp_workspace_path(path):
            path = _resolve_acp_workspace_path(path, _extract_thread_id_from_thread_data(thread_data))
        elif not _is_custom_mount_path(path):
            path = _resolve_and_validate_user_data_path(path, thread_data)
        # custom mount 경로는 LocalSandbox._resolve_path()가 해석한다.
    return sandbox.read_file(path)


@tool("read_file", parse_docstring=True)
def read_file_tool(
    runtime: Runtime,
    description: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """텍스트 파일의 내용을 읽는다. 소스 코드, 설정 파일, 로그 등 텍스트 기반 파일을 확인할 때 사용하라.

    Args:
        description: 이 파일을 읽는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        path: 읽을 파일의 **절대** 경로.
        start_line: 선택적 시작 줄 번호(1부터 시작, 포함). 생략하면 첫 줄부터 읽는다.
        end_line: 선택적 종료 줄 번호(1부터 시작, 포함). 생략하면 마지막 줄까지 읽는다.
    """
    try:
        # 비활성화된 skill 파일 접근을 차단한다.
        if _is_disabled_skill_path(path, user_id=resolve_runtime_user_id(runtime)):
            skill_name = _extract_skill_name_from_skills_path(path) or "unknown"
            return f"Error: Skill '{skill_name}' is disabled. Access to its files is blocked. Enable the skill in settings before using it."
        if start_line is not None and start_line < 1:
            return "(start_line must be >= 1)"
        effective_start = start_line or 1
        if end_line is not None and end_line < 1:
            return "(end_line must be >= 1)"
        if end_line is not None and effective_start > end_line:
            return "(start_line > end_line — no lines in range)"

        requested_path = path
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        use_line_range = start_line is not None or end_line is not None
        if use_line_range:
            if is_local_sandbox(runtime):
                thread_data = get_thread_data(runtime)
                validate_local_tool_path(path, thread_data, read_only=True)
                if _is_skills_path(path):
                    path = _resolve_skills_path(path)
                elif _is_acp_workspace_path(path):
                    path = _resolve_acp_workspace_path(path, _extract_thread_id_from_thread_data(thread_data))
                elif not _is_custom_mount_path(path):
                    path = _resolve_and_validate_user_data_path(path, thread_data)
                # custom mount 경로는 LocalSandbox._resolve_path()가 해석한다.
            content = sandbox.read_file(path, start_line=start_line, end_line=end_line)
        else:
            content = read_current_file_content(runtime, path)
        if not content:
            if start_line is not None and start_line > 1:
                return "(start_line exceeds file length)"
            return "(empty)"
        try:
            from deerflow.config.app_config import get_app_config

            sandbox_cfg = get_app_config().sandbox
            max_chars = sandbox_cfg.read_file_output_max_chars if sandbox_cfg else 50000
        except Exception:
            max_chars = 50000
        return _truncate_read_file_output(content, max_chars)
    except SandboxError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: File not found: {requested_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {requested_path}"
    except IsADirectoryError:
        return f"Error: Path is a directory, not a file: {requested_path}"
    except UnicodeDecodeError:
        return (
            f"Error: cannot read '{requested_path}' as text — it appears to be a binary file "
            "(e.g. .xlsx, .pdf, or an image). read_file only supports UTF-8 text. Use bash with a "
            "suitable library instead (pandas/openpyxl for spreadsheets), or view_image for images."
        )
    except Exception as e:
        return f"Error: Unexpected error reading file: {_sanitize_error(e, runtime)}"


async def _read_file_tool_async(
    runtime: Runtime,
    description: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    return await _run_sync_tool_after_async_sandbox_init(read_file_tool.func, runtime, description, path, start_line, end_line)


read_file_tool.coroutine = _read_file_tool_async


def _effective_write_file_max_bytes() -> int:
    """append가 아닌 write_file 호출에 적용되는 현재 크기 상한을 반환한다.

    ``DEERFLOW_WRITE_FILE_MAX_BYTES``를 import 시점이 아니라 호출 시점에 읽으므로,
    테스트나 runtime 조정이 재시작 없이 반영된다. 값이 없거나 잘못되었으면 기본값으로
    fallback한다. 0 이하 값은 이 guard를 끈다.
    """
    raw = os.environ.get(_WRITE_FILE_MAX_BYTES_ENV)
    if raw is None:
        return _WRITE_FILE_CONTENT_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        return _WRITE_FILE_CONTENT_MAX_BYTES


@tool("write_file", parse_docstring=True)
def write_file_tool(
    runtime: Runtime,
    description: str,
    path: str,
    content: str,
    append: bool = False,
) -> str:
    """파일에 텍스트 내용을 쓴다. 기본적으로 대상 파일을 덮어쓰며, append=True로 지정하면 기존 내용을 지우지 않고 끝에 이어 붙인다.

    READ-BEFORE-WRITE (issue #3857): 대상 파일이 이미 존재하면(append=True인 경우 포함)
    먼저 read_file로 그 파일의 CURRENT 버전을 읽어야 한다. 모든 쓰기는 이전 읽기를
    무효화하므로, 연속 수정 사이에는 다시 읽어야 한다 — 해당 구간만 범위 지정해
    읽어도 충분하다. 이 검사를 통과하지 못한 쓰기는 에러로 거부된다.

    크기 정책 (issue #3189):
    append가 아닌 단일 write_file 호출은 UTF-8 기준 80 KB를 초과해서는 안 된다.
    한 번에 너무 큰 내용을 쓰면 LLM streaming chunk-gap timeout과 연결된다. 모델이
    하나의 연속 스트림으로 내보내야 하는 tool call JSON payload가 안전 범위를 넘기
    때문이다. 더 큰 문서에는 아래 전략 중 ONE을 사용하라(write_file은 초과 payload를
    실행 가능한 안내와 함께 거부한다):

      1. INCREMENTAL EDIT (수정 작업에 권장): 최초 쓰기 이후에는 `str_replace`로
         필요한 구간만 정밀하게 갱신하라. Claude Code의 Write+Edit과 OpenAI Codex의
         apply_patch가 쓰는 것과 같은 패턴이며, 각 tool call의 payload를 작게 유지한다.
      2. APPEND-IN-CHUNKS (새로 작성하는 긴 문서용): 문서를 각각 80 KB보다 충분히
         작은 구간으로 나눠라. 첫 호출은 append=False로 파일을 만들고, 이후 호출은
         append=True를 쓴다. 80 KB 상한은 append=True 호출에는 적용되지 NOT.

    운영자는 env var `DEERFLOW_WRITE_FILE_MAX_BYTES`로 이 상한을 덮어쓸 수 있다
    (0이면 guard 자체를 비활성화). 상한을 올리면 streaming timeout 위험이 있다.

    Args:
        description: 이 파일에 쓰는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        path: 쓸 대상 파일의 **절대** 경로. ALWAYS PROVIDE THIS PARAMETER SECOND.
        content: 파일에 쓸 내용. ALWAYS PROVIDE THIS PARAMETER THIRD.
        append: append 모드 여부. True면 덮어쓰지 않고 파일 끝에 이어 붙인다. 기본값은 False.
    """
    if not append:
        max_bytes = _effective_write_file_max_bytes()
        if max_bytes > 0:
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > max_bytes:
                return (
                    f"Error: write_file content ({content_bytes} bytes) exceeds the "
                    f"{max_bytes}-byte single-call limit. Split the content into smaller "
                    "pieces: either (a) write the first section now, then use `str_replace` "
                    "for further edits, or (b) call write_file again with append=True "
                    "carrying the next section. See SIZE POLICY in the tool docstring "
                    "or issue #3189 for the rationale."
                )
    try:
        requested_path = path
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        if is_local_sandbox(runtime):
            thread_data = get_thread_data(runtime)
            validate_local_tool_path(path, thread_data)
            if not _is_custom_mount_path(path):
                path = _resolve_and_validate_user_data_path(path, thread_data)
            # custom mount 경로는 LocalSandbox._resolve_path()가 해석한다.
        with get_file_operation_lock(sandbox, path):
            sandbox.write_file(path, content, append)
        return "OK"
    except SandboxError as e:
        return _format_write_file_error(requested_path, e, runtime)
    except PermissionError:
        return _truncate_write_file_error_detail(
            f"Error: Permission denied writing to file: {requested_path}",
            _DEFAULT_WRITE_FILE_ERROR_MAX_CHARS,
        )
    except IsADirectoryError:
        return _truncate_write_file_error_detail(
            f"Error: Path is a directory, not a file: {requested_path}",
            _DEFAULT_WRITE_FILE_ERROR_MAX_CHARS,
        )
    except OSError as e:
        return _format_write_file_error(requested_path, e, runtime)
    except Exception as e:
        return _format_write_file_error(requested_path, e, runtime)


async def _write_file_tool_async(
    runtime: Runtime,
    description: str,
    path: str,
    content: str,
    append: bool = False,
) -> str:
    return await _run_sync_tool_after_async_sandbox_init(write_file_tool.func, runtime, description, path, content, append)


write_file_tool.coroutine = _write_file_tool_async


@tool("str_replace", parse_docstring=True)
def str_replace_tool(
    runtime: Runtime,
    description: str,
    path: str,
    old_str: str,
    new_str: str,
    replace_all: bool = False,
) -> str:
    """파일 안의 substring을 다른 substring으로 교체한다.
    `replace_all`이 False(기본값)이면, 교체할 substring이 파일 안에 **정확히 한 번만** 나타나야 한다.

    READ-BEFORE-WRITE (issue #3857): 먼저 read_file로 그 파일의 CURRENT 버전을 읽어야
    한다. 모든 쓰기는 이전 읽기를 무효화한다.

    Args:
        description: 이 substring을 교체하는 이유를 짧게 설명한다. ALWAYS PROVIDE THIS PARAMETER FIRST.
        path: substring을 교체할 파일의 **절대** 경로. ALWAYS PROVIDE THIS PARAMETER SECOND.
        old_str: 교체할 substring. ALWAYS PROVIDE THIS PARAMETER THIRD.
        new_str: 새 substring. ALWAYS PROVIDE THIS PARAMETER FOURTH.
        replace_all: substring이 나타나는 모든 위치를 교체할지 여부. False이면 첫 번째 위치만 교체된다. 기본값은 False.
    """
    try:
        sandbox = ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        requested_path = path
        if is_local_sandbox(runtime):
            thread_data = get_thread_data(runtime)
            validate_local_tool_path(path, thread_data)
            if not _is_custom_mount_path(path):
                path = _resolve_and_validate_user_data_path(path, thread_data)
            # custom mount 경로는 LocalSandbox._resolve_path()가 해석한다.
        with get_file_operation_lock(sandbox, path):
            content = sandbox.read_file(path)
            if not old_str:
                # 아무것도 바꾸지 않는 편집이다. str.replace("", new_str)는 모든 문자
                # 경계에 new_str을 끼워 넣으므로 아래로 흘려보낼 수 없다.
                return "OK"
            if not content or old_str not in content:
                return f"Error: String to replace not found in file: {requested_path}"
            if replace_all:
                content = content.replace(old_str, new_str)
            else:
                content = content.replace(old_str, new_str, 1)
            sandbox.write_file(path, content)
        return "OK"
    except SandboxError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: File not found: {requested_path}"
    except PermissionError:
        return f"Error: Permission denied accessing file: {requested_path}"
    except Exception as e:
        return f"Error: Unexpected error replacing string: {_sanitize_error(e, runtime)}"


async def _str_replace_tool_async(
    runtime: Runtime,
    description: str,
    path: str,
    old_str: str,
    new_str: str,
    replace_all: bool = False,
) -> str:
    return await _run_sync_tool_after_async_sandbox_init(
        str_replace_tool.func,
        runtime,
        description,
        path,
        old_str,
        new_str,
        replace_all,
    )


str_replace_tool.coroutine = _str_replace_tool_async
