"""MCP server와 skill을 함께 다루는 통합 extensions 설정."""

import json
import logging
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deerflow.config.runtime_paths import existing_project_file
from deerflow.constants import DEFAULT_MCP_SESSION_INIT_TIMEOUT

logger = logging.getLogger(__name__)


def normalize_mcp_transport_alias(data: Any) -> Any:
    """``type``이 없으면 MCP 스펙의 ``transport``를 ``type``으로 승격한다."""
    if isinstance(data, dict):
        transport = data.get("transport")
        if transport and not data.get("type"):
            return {**data, "type": transport}
    return data


class McpRoutingConfig(BaseModel):
    """MCP 도구 선호도를 나타내는 soft routing 힌트."""

    mode: Literal["off", "prefer"] = Field(
        default="off",
        description="Whether to emit prompt hints preferring this MCP tool for matching requests.",
    )
    priority: int = Field(
        default=0,
        description="Ordering key for routing hints. Higher values are rendered first.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Operator-authored keywords that describe when this MCP tool should be preferred.",
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("priority")
    @classmethod
    def _clamp_priority(cls, value: int) -> int:
        if value < 0:
            logger.warning("MCP routing priority %s is below 0; clamping to 0.", value)
            return 0
        if value > 100:
            logger.warning("MCP routing priority %s is above 100; clamping to 100.", value)
            return 100
        return value


class McpToolOverride(BaseModel):
    """도구 단위 MCP 설정 override."""

    routing: McpRoutingConfig = Field(default_factory=McpRoutingConfig)
    model_config = ConfigDict(extra="allow")


class McpOAuthConfig(BaseModel):
    """MCP server의 OAuth 설정(HTTP/SSE transport 전용)."""

    enabled: bool = Field(default=True, description="Whether OAuth token injection is enabled")
    token_url: str = Field(description="OAuth token endpoint URL")
    grant_type: Literal["client_credentials", "refresh_token"] = Field(
        default="client_credentials",
        description="OAuth grant type",
    )
    client_id: str | None = Field(default=None, description="OAuth client ID")
    client_secret: str | None = Field(default=None, description="OAuth client secret")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token (for refresh_token grant)")
    scope: str | None = Field(default=None, description="OAuth scope")
    audience: str | None = Field(default=None, description="OAuth audience (provider-specific)")
    token_field: str = Field(default="access_token", description="Field name containing access token in token response")
    token_type_field: str = Field(default="token_type", description="Field name containing token type in token response")
    expires_in_field: str = Field(default="expires_in", description="Field name containing expiry (seconds) in token response")
    default_token_type: str = Field(default="Bearer", description="Default token type when missing in token response")
    refresh_skew_seconds: int = Field(default=60, description="Refresh token this many seconds before expiry")
    extra_token_params: dict[str, str] = Field(default_factory=dict, description="Additional form params sent to token endpoint")
    model_config = ConfigDict(extra="allow")


class McpServerConfig(BaseModel):
    """MCP server 하나에 대한 설정."""

    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")
    type: str = Field(default="stdio", description="Transport type: 'stdio', 'sse', or 'http'")
    command: str | None = Field(default=None, description="Command to execute to start the MCP server (for stdio type)")
    args: list[str] = Field(default_factory=list, description="Arguments to pass to the command (for stdio type)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the MCP server")
    url: str | None = Field(default=None, description="URL of the MCP server (for sse or http type)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers to send (for sse or http type)")
    oauth: McpOAuthConfig | None = Field(default=None, description="OAuth configuration (for sse or http type)")
    description: str = Field(default="", description="Human-readable description of what this MCP server provides")
    routing: McpRoutingConfig = Field(default_factory=McpRoutingConfig, description="Soft routing hints for tools from this MCP server")
    tools: dict[str, McpToolOverride] = Field(default_factory=dict, description="Per-original-tool MCP configuration overrides")
    tool_name_prefix: bool = Field(
        default=True,
        description="Whether to prefix discovered tool names with the MCP server name to avoid cross-server collisions",
    )
    tool_call_timeout: float | None = Field(
        default=None,
        description="Timeout in seconds for individual stdio MCP tool calls. HTTP/SSE servers use transport-level timeouts. None means no timeout.",
    )
    session_init_timeout: float | None = Field(
        default=DEFAULT_MCP_SESSION_INIT_TIMEOUT,
        description=(
            "Timeout in seconds for MCP server bring-up: tool discovery (subprocess spawn + initialize + tools/list) "
            "and persistent stdio session initialization. Defaults to DEFAULT_MCP_SESSION_INIT_TIMEOUT so a hung "
            "server cannot block agent construction indefinitely. None means no timeout."
        ),
    )
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _accept_transport_alias(cls, data: Any) -> Any:
        """MCP 스펙의 ``transport`` 필드를 ``type``의 alias로 받아들인다.

        공식 MCP 설정 스키마는 transport 방식(``stdio``/``sse``/``http``)을
        ``transport``로 표기한다. 예전 버전은 ``type``만 인식해서 ``transport``만
        지정한 원격 SSE/HTTP server가 기본값인 ``stdio``로 잘못 처리됐다.
        이 validator가 둘을 정규화해 어느 표기든 동작하게 하며, 둘 다 있으면
        ``type``이 우선한다.
        """
        return normalize_mcp_transport_alias(data)


def resolve_effective_mcp_routing(server_config: McpServerConfig | None, original_tool_name: str) -> dict[str, Any]:
    """MCP 도구 하나에 대해 server 단위 routing과 도구 단위 override를 병합한다."""
    if server_config is None:
        return McpRoutingConfig().model_dump(mode="json")

    effective = server_config.routing.model_dump(mode="json")
    override = server_config.tools.get(original_tool_name)
    if override is not None and "routing" in override.model_fields_set:
        effective.update(override.routing.model_dump(mode="json", exclude_unset=True))
    return effective


class SkillStateConfig(BaseModel):
    """skill 하나의 상태 설정."""

    enabled: bool = Field(default=True, description="Whether this skill is enabled")


class ExtensionsConfig(BaseModel):
    """MCP server와 skill을 함께 다루는 통합 설정."""

    middlewares: list[str] = Field(
        default_factory=list,
        description="AgentMiddleware class paths loaded into the lead-agent middleware chain. Each entry uses 'module.path:ClassName'.",
    )
    mcp_servers: dict[str, McpServerConfig] = Field(
        default_factory=dict,
        description="Map of MCP server name to configuration",
        alias="mcpServers",
    )
    skills: dict[str, SkillStateConfig] = Field(
        default_factory=dict,
        description="Map of skill name to state configuration",
    )
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def to_file_dict(self) -> dict[str, Any]:
        """공개 extensions_config.json 형태로 직렬화한다."""
        return self.model_dump(by_alias=True)

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path | None:
        """extensions 설정 파일 경로를 결정한다.

        우선순위:
        1. `config_path` 인자가 있으면 그것을 쓴다.
        2. `DEER_FLOW_EXTENSIONS_CONFIG_PATH` 환경변수가 있으면 그것을 쓴다.
        3. 없으면 호출자 프로젝트 루트에서 `extensions_config.json`, 그 다음 `mcp_config.json`을 찾는다.
        4. 하위 호환을 위해 legacy backend/저장소 루트 기본 위치도 찾는다.
        5. 그래도 못 찾으면 None을 반환한다(extensions는 선택 사항).

        Args:
            config_path: extensions 설정 파일 경로(선택).

        Returns:
            위 순서로 찾은 extensions 설정 파일 경로.

            명시적 `config_path` 인자나 설정된 `DEER_FLOW_EXTENSIONS_CONFIG_PATH`는
            "이 파일을 반드시 쓰라"는 운영자의 단언이다. 따라서 두 모드에서 파일이
            없으면 "설정 없음"으로 격하하지 않고 ``FileNotFoundError``를 던진다
            (아래 Raises 참고). 잘못된 Docker mount, 오타, 삭제된 운영 설정은 모든
            MCP server와 skill이 사라진 채 조용히 기동하는 대신 눈에 띄는 오류로
            드러나야 한다.

            fallback *탐색* 모드(인자도 환경변수도 없는 경우)만 아무것도 못 찾았을 때
            ``None``을 반환한다. 이 경우는 애초에 extensions를 설정한 적이 없다는
            뜻이고, 일부 호출자(예: `deerflow.mcp.cache`의 MCP tools-cache staleness
            검사)가 정상적인 신호로 의존하는 "extensions는 선택 사항" 케이스다.

        Raises:
            FileNotFoundError: `config_path`가 주어졌거나
                `DEER_FLOW_EXTENSIONS_CONFIG_PATH`가 설정됐는데 해당 경로가
                존재하지 않는 경우.
        """
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Extensions config file specified by param `config_path` not found at {path}")
            return path
        elif env_path := os.getenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH"):
            path = Path(env_path)
            if not path.exists():
                raise FileNotFoundError(f"Extensions config file specified by environment variable `DEER_FLOW_EXTENSIONS_CONFIG_PATH` not found at {path}")
            return path
        else:
            project_config = existing_project_file(("extensions_config.json", "mcp_config.json"))
            if project_config is not None:
                return project_config

            backend_dir = Path(__file__).resolve().parents[4]
            repo_root = backend_dir.parent
            for path in (
                backend_dir / "extensions_config.json",
                repo_root / "extensions_config.json",
                backend_dir / "mcp_config.json",
                repo_root / "mcp_config.json",
            ):
                if path.exists():
                    return path

            # extensions는 선택 사항이다. 위의 명시적 config_path/환경변수 분기와 달리
            # 여기서 아무것도 못 찾는 것은 정상이므로 예외 대신 None을 반환한다.
            return None

    @classmethod
    def from_file(cls, config_path: str | None = None) -> "ExtensionsConfig":
        """JSON 파일에서 extensions 설정을 읽는다.

        자세한 경로 결정 규칙은 `resolve_config_path`를 참고한다.

        Args:
            config_path: extensions 설정 파일 경로.

        Returns:
            ExtensionsConfig: 읽어들인 설정. 파일이 없으면 빈 설정.
        """
        resolved_path = cls.resolve_config_path(config_path)
        if resolved_path is None:
            # extensions 설정 파일이 없으면 빈 설정을 반환한다.
            return cls(mcp_servers={}, skills={})

        try:
            with open(resolved_path, encoding="utf-8") as f:
                config_data = json.load(f)
            config_data = cls.resolve_env_variables(config_data)
            return cls.model_validate(config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extensions config file at {resolved_path} is not valid JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load extensions config from {resolved_path}: {e}") from e

    @classmethod
    def resolve_env_variables(cls, config: Any) -> Any:
        """설정 안의 환경변수를 재귀적으로 치환한다.

        환경변수는 `os.getenv`로 해석한다. 예: $OPENAI_API_KEY

        Args:
            config: 환경변수를 치환할 설정.

        Returns:
            환경변수가 치환된 설정.
        """
        if isinstance(config, str):
            if not config.startswith("$"):
                return config
            env_value = os.getenv(config[1:])
            if env_value is None:
                # 치환되지 않은 placeholder는 빈 문자열로 둔다. 그래야 downstream
                # 소비자(예: MCP server)가 "$VAR" 리터럴을 실제 환경값으로 받지 않는다.
                return ""
            return env_value

        if isinstance(config, dict):
            return {key: cls.resolve_env_variables(value) for key, value in config.items()}

        if isinstance(config, list):
            return [cls.resolve_env_variables(item) for item in config]

        if isinstance(config, tuple):
            return tuple(cls.resolve_env_variables(item) for item in config)

        return config

    def get_enabled_mcp_servers(self) -> dict[str, McpServerConfig]:
        """활성화된 MCP server만 반환한다.

        Returns:
            활성화된 MCP server 딕셔너리.
        """
        return {name: config for name, config in self.mcp_servers.items() if config.enabled}

    def is_skill_enabled(self, skill_name: str, skill_category: str) -> bool:
        """skill이 활성화됐는지 확인한다.

        Args:
            skill_name: skill 이름.
            skill_category: skill 카테고리(public, custom, legacy).

        Returns:
            활성화됐으면 True, 아니면 False.

        Note:
            모든 카테고리(public, custom, legacy)가 extensions_config의
            활성/비활성 상태를 따른다. 명시적 항목이 없으면 활성으로 본다.
        """
        skill_config = self.skills.get(skill_name)
        if skill_config is None:
            # 모든 skill 카테고리는 기본값이 활성이다.
            return skill_category in ("public", "custom", "legacy", "integrations")
        return skill_config.enabled


_extensions_config: ExtensionsConfig | None = None


def _fsync_directory_best_effort(directory: Path) -> None:
    """플랫폼이 지원하는 경우 디렉터리 엔트리 변경을 디스크에 반영한다."""
    if os.name == "nt":
        return

    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(directory_fd)
    except OSError:
        logger.debug("Could not fsync extensions config directory: %s", directory, exc_info=True)
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            logger.debug("Could not close extensions config directory: %s", directory, exc_info=True)


def atomic_write_extensions_config(path: Path, data: dict[str, Any]) -> None:
    """잘리거나 반쯤 쓰인 파일이 노출되지 않도록 extensions 설정을 기록한다."""
    path = Path(path)
    target_path = path.resolve(strict=False) if path.is_symlink() else path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(target_path.stat().st_mode)
    except FileNotFoundError:
        pass

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, indent=2)
            if existing_mode is not None:
                temporary_path.chmod(existing_mode)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, target_path)
        _fsync_directory_best_effort(target_path.parent)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove temporary extensions config file: %s",
                    temporary_path,
                    exc_info=True,
                )


def get_extensions_config() -> ExtensionsConfig:
    """extensions 설정 인스턴스를 반환한다.

    캐시된 singleton을 돌려준다. 파일에서 다시 읽으려면 `reload_extensions_config()`를,
    캐시를 비우려면 `reset_extensions_config()`를 쓴다.

    Returns:
        캐시된 ExtensionsConfig 인스턴스.
    """
    global _extensions_config
    if _extensions_config is None:
        _extensions_config = ExtensionsConfig.from_file()
    return _extensions_config


#: 모든 writer의 ``extensions_config.json`` read-modify-write 사이클을 직렬화한다.
#: skills router(skill 활성/비활성)와 MCP router(server 설정 갱신)가 모두 이 파일을
#: 읽고 변경을 병합한 뒤 다시 쓴다. RMW가 event loop에서 inline으로 돌 때는 암묵적으로
#: 직렬화됐지만, 한쪽이 RMW를 worker thread로 넘기는 순간 loop가 read->write 구간에
#: 다른 writer를 끼워 넣을 수 있고, 나중 쓰기가 앞선 변경을 조용히 덮어쓴다.
#:
#: ``asyncio.Lock``이 아니라 ``threading.Lock``이며, RMW를 수행하는 worker *안에서*
#: 획득해야 한다. ``await asyncio.to_thread(...)``를 감싼 asyncio lock은 대기 중인
#: task만 보호한다. 그 task가 취소되면 context manager는 즉시 lock을 풀지만 worker
#: thread는 계속 쓰기 때문에 두 번째 writer가 들어온다. worker가 직접 소유하면 쓰기와
#: reload가 실제로 끝날 때까지 유지된다. 또한 event loop 종속성이 없어 서로 다른 loop의
#: writer끼리도 배제된다.
extensions_config_write_lock = threading.Lock()


def reload_extensions_config(config_path: str | None = None) -> ExtensionsConfig:
    """extensions 설정을 파일에서 다시 읽고 캐시를 갱신한다.

    설정 파일이 바뀌었을 때 애플리케이션을 재시작하지 않고 변경을 반영하는 용도다.

    Args:
        config_path: extensions 설정 파일 경로(선택). 없으면 기본 경로 결정 전략을 쓴다.

    Returns:
        새로 읽어들인 ExtensionsConfig 인스턴스.
    """
    global _extensions_config
    _extensions_config = ExtensionsConfig.from_file(config_path)
    return _extensions_config


def reset_extensions_config() -> None:
    """캐시된 extensions 설정 인스턴스를 비운다.

    singleton 캐시를 지워서 다음 `get_extensions_config()` 호출이 파일에서 다시 읽게
    한다. 테스트나 설정 전환 시 유용하다.
    """
    global _extensions_config
    _extensions_config = None


def set_extensions_config(config: ExtensionsConfig) -> None:
    """extensions 설정 인스턴스를 직접 주입한다.

    테스트용 custom/mock 설정을 넣을 때 쓴다.

    Args:
        config: 사용할 ExtensionsConfig 인스턴스.
    """
    global _extensions_config
    _extensions_config = config
