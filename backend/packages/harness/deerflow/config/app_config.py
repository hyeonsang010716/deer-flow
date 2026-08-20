import logging
import os
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from deerflow.config.acp_config import ACPAgentConfig, load_acp_config_from_dict
from deerflow.config.agent_storage_config import AgentStorageConfig
from deerflow.config.agents_api_config import AgentsApiConfig, load_agents_api_config_from_dict
from deerflow.config.auth_config import AuthAppConfig
from deerflow.config.authorization_config import AuthorizationConfig, load_authorization_config_from_dict
from deerflow.config.channel_connections_config import ChannelConnectionsConfig
from deerflow.config.checkpointer_config import CheckpointerConfig, load_checkpointer_config_from_dict
from deerflow.config.database_config import DatabaseConfig
from deerflow.config.dedupe_storage_config import DedupeStorageConfig
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.file_signature import ConfigSignature as _ConfigSignature
from deerflow.config.file_signature import get_config_signature as _get_config_signature
from deerflow.config.guardrails_config import GuardrailsConfig, load_guardrails_config_from_dict
from deerflow.config.input_polish_config import InputPolishConfig
from deerflow.config.loop_detection_config import LoopDetectionConfig
from deerflow.config.memory_config import MemoryConfig, load_memory_config_from_dict
from deerflow.config.model_config import ModelConfig
from deerflow.config.read_before_write_config import ReadBeforeWriteConfig
from deerflow.config.reload_boundary import format_field_description
from deerflow.config.run_events_config import RunEventsConfig
from deerflow.config.run_ownership_config import RunOwnershipConfig
from deerflow.config.runtime_paths import existing_project_file
from deerflow.config.safety_finish_reason_config import SafetyFinishReasonConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.scheduler_config import SchedulerConfig
from deerflow.config.skill_evolution_config import SkillEvolutionConfig
from deerflow.config.skill_scan_config import SkillScanConfig
from deerflow.config.skills_config import SkillsConfig
from deerflow.config.stream_bridge_config import StreamBridgeConfig, load_stream_bridge_config_from_dict
from deerflow.config.subagents_config import SubagentsAppConfig, load_subagents_config_from_dict
from deerflow.config.suggestions_config import SuggestionsConfig
from deerflow.config.summarization_config import SummarizationConfig, load_summarization_config_from_dict
from deerflow.config.title_config import TitleConfig, load_title_config_from_dict
from deerflow.config.token_budget_config import TokenBudgetConfig
from deerflow.config.token_usage_config import TokenUsageConfig
from deerflow.config.tool_config import ToolConfig, ToolGroupConfig
from deerflow.config.tool_output_config import ToolOutputConfig
from deerflow.config.tool_progress_config import ToolProgressConfig
from deerflow.config.tool_search_config import ToolSearchConfig, load_tool_search_config_from_dict
from deerflow.extensions.loader import ExtensionSpec

load_dotenv()

logger = logging.getLogger(__name__)


CONFIG_FILE_DATABASE_DEFAULTS = {
    "backend": "sqlite",
    "sqlite_dir": ".deer-flow/data",
}


class CircuitBreakerConfig(BaseModel):
    """LLM Circuit Breaker 설정."""

    failure_threshold: int = Field(default=5, description="Number of consecutive failures before tripping the circuit")
    recovery_timeout_sec: int = Field(default=60, description="Time in seconds before attempting to recover the circuit")


class LlmCallConfig(BaseModel):
    """LLM 호출 실행 설정(동시성 / rate shaping).

    *실패하는* provider를 다루는 :class:`CircuitBreakerConfig` 나 모델 endpoint를 기술하는
    :class:`ModelConfig` 와는 다르다. 여기 옵션들은 LLM 호출을 동시에 몇 개 돌릴지와
    retry/backoff 루프의 동작을 정한다. 동시성을 제한하면 요청 rate의 *기울기*가 제한되는데,
    provider의 burst-rate(``limit_burst_rate``) 제한이 반응하는 게 바로 그 기울기다.
    """

    max_concurrent_calls: int = Field(
        default=0,
        ge=0,
        description=(
            "Process-wide cap on concurrently in-flight LLM calls. 0 disables "
            "the cap (default, preserving existing behavior). Set to a positive "
            "int to smooth provider burst-rate (limit_burst_rate) spikes by "
            "bounding the request-rate slope at the morning peak. Per-process, "
            "not per-cluster: with GATEWAY_WORKERS > 1 the aggregate cap is "
            "effectively max_concurrent_calls * GATEWAY_WORKERS (and a "
            "multi-node rollout multiplies it further), so size the per-process "
            "value accordingly and pair it with an nginx limit_req at the ingress "
            "for a true cluster-wide slope cap. Startup-only: the cap is captured "
            "at the first LLM run and frozen for the process lifetime, so editing "
            "it in config.yaml takes effect only after a gateway restart (the "
            "other llm_call.* knobs remain hot-reloadable). Freezing avoids the "
            "downscale/config-freshness races a runtime-mutable cap would "
            "introduce on a process-wide, cross-loop limiter."
        ),
    )
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        description="Max LLM call attempts (1 = no retry) for retriable transient errors.",
    )
    retry_base_delay_ms: int = Field(
        default=1000,
        ge=0,
        description="Base (ms) for the decorrelated-jitter retry backoff; seeds the first retry delay.",
    )
    retry_cap_delay_ms: int = Field(
        default=8000,
        ge=0,
        description="Hard cap (ms) on any single retry backoff delay.",
    )
    burst_retry_base_delay_ms: int = Field(
        default=5000,
        ge=0,
        description=(
            "Base (ms) for the backoff when the provider returns a burst-rate "
            "(limit_burst_rate) 429. Higher than retry_base_delay_ms so the "
            "single burst retry lands after the throttle window subsides. "
            "Ignored when the provider sends Retry-After (honored verbatim)."
        ),
    )


class LoggingEnhanceConfig(BaseModel):
    """요청 trace 로깅 향상 설정."""

    enabled: bool = Field(default=False, description="Enable request-level trace ids in Gateway response headers and log records.")
    format: Literal["text", "json"] = Field(default="text", description="Enhanced log output format.")


class LoggingConfig(BaseModel):
    """로깅 설정."""

    enhance: LoggingEnhanceConfig = Field(default_factory=LoggingEnhanceConfig, description="Request trace correlation logging settings.")


def is_trace_correlation_enabled(config: Any) -> bool:
    """*config* 에 ``logging.enhance.enabled`` 가 켜져 있으면 ``True``를 반환한다.

    요청 trace 상관관계 게이트의 단일 진실 공급원이다. Gateway ``TraceMiddleware`` 와 내장
    ``DeerFlowClient`` 가 공유하므로, ``deerflow_trace_id``(Langfuse metadata)를 언제 내보내는지와
    요청 단위 trace id를 아예 바인딩할지에 대해 두 진입점이 어긋날 수 없다. ``getattr`` 체인으로
    ``logging.enhance.enabled`` 를 노출하는 객체면 무엇이든 받는다(``AppConfig``, 테스트용
    ``SimpleNamespace`` 등). 중간 속성이 없으면 조용히 ``False``로 떨어진다.
    """
    logging_config = getattr(config, "logging", None)
    enhance = getattr(logging_config, "enhance", None)
    return bool(getattr(enhance, "enabled", False))


def _legacy_config_candidates() -> tuple[Path, ...]:
    """monorepo 호환을 위해 소스 트리의 config.yaml 후보 경로를 반환한다."""
    backend_dir = Path(__file__).resolve().parents[4]
    repo_root = backend_dir.parent
    return (backend_dir / "config.yaml", repo_root / "config.yaml")


def logging_level_from_config(name: str | None) -> int:
    """``config.yaml`` 의 ``log_level`` 문자열을 :mod:`logging` 레벨 상수로 매핑한다."""
    mapping = logging.getLevelNamesMapping()
    return mapping.get((name or "info").strip().upper(), logging.INFO)


def apply_logging_level(name: str | None) -> None:
    """*name* 을 로깅 레벨로 해석해 ``deerflow``/``app`` logger 계층에 적용한다.

    서드파티 라이브러리(uvicorn, sqlalchemy 등)의 로그 양에 영향을 주지 않도록 ``deerflow`` 와
    ``app`` logger 레벨만 바꾼다. root handler 레벨은 낮추기만 하고 올리지 않는다. 그래야 설정된
    logger의 메시지가 걸러지지 않고 통과하면서도, 서드파티 로그 출력을 위해 의도적으로 엄격하게
    잡아 둔 handler 임계값은 유지된다.
    """
    level = logging_level_from_config(name)
    for logger_name in ("deerflow", "app"):
        logging.getLogger(logger_name).setLevel(level)
    for handler in logging.root.handlers:
        if level < handler.level:
            handler.setLevel(level)


class AppConfig(BaseModel):
    """DeerFlow 애플리케이션 설정."""

    log_level: str = Field(
        default="info",
        description=format_field_description(
            "log_level",
            field_doc="Logging level for deerflow and app modules (debug/info/warning/error); third-party libraries are not affected.",
        ),
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description=format_field_description(
            "logging",
            field_doc="Structured logging and request trace correlation settings.",
        ),
    )
    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig, description="Token usage tracking configuration")
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig, description="Token Budget tracking and limits configuration.")
    plugins: list[ExtensionSpec] = Field(
        default_factory=list,
        description=format_field_description(
            "plugins",
            field_doc=(
                "Extension packages to load at startup, in order. Each entry names an install "
                "entry point as 'module.path:install' and carries its own private config block. "
                "Distinct from the `extensions` field above, which configures MCP servers, skills "
                "and config-declared middlewares and is backed by the HTTP-writable "
                "extensions_config.json."
            ),
        ),
    )
    max_recursion_limit: int = Field(
        default=1000,
        ge=1,
        description="Hard server-side ceiling for a client-supplied run recursion_limit. Client values above this are clamped; prevents runaway LangGraph super-steps (LLM cost / DoS).",
    )
    models: list[ModelConfig] = Field(default_factory=list, description="Available models")
    sandbox: SandboxConfig = Field(
        description=format_field_description(
            "sandbox",
            field_doc="Sandbox provider configuration (local filesystem or Docker-based aio sandbox).",
        ),
    )
    tools: list[ToolConfig] = Field(default_factory=list, description="Available tools")
    tool_groups: list[ToolGroupConfig] = Field(default_factory=list, description="Available tool groups")
    skills: SkillsConfig = Field(default_factory=SkillsConfig, description="Skills configuration")
    skill_scan: SkillScanConfig = Field(default_factory=SkillScanConfig, description="Native deterministic skill safety scanning configuration")
    skill_evolution: SkillEvolutionConfig = Field(default_factory=SkillEvolutionConfig, description="Agent-managed skill evolution configuration")
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig, description="Extensions configuration (MCP servers and skills state)")
    tool_output: ToolOutputConfig = Field(default_factory=ToolOutputConfig, description="Tool output budget protection configuration")
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig, description="Tool search / deferred loading configuration")
    title: TitleConfig = Field(default_factory=TitleConfig, description="Automatic title generation configuration")
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig, description="Conversation summarization configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory subsystem configuration")
    agents_api: AgentsApiConfig = Field(default_factory=AgentsApiConfig, description="Custom-agent management API configuration")
    acp_agents: dict[str, ACPAgentConfig] = Field(default_factory=dict, description="ACP-compatible agent configuration")
    subagents: SubagentsAppConfig = Field(default_factory=SubagentsAppConfig, description="Subagent runtime configuration")
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig, description="Guardrail middleware configuration")
    authorization: AuthorizationConfig = Field(default_factory=AuthorizationConfig, description="Fine-grained resource authorization configuration (RBAC and beyond)")
    input_polish: InputPolishConfig = Field(default_factory=InputPolishConfig, description="Pre-send input polishing configuration.")
    suggestions: SuggestionsConfig = Field(default_factory=SuggestionsConfig, description="Follow-up suggestions configuration.")
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="LLM circuit breaker configuration")
    llm_call: LlmCallConfig = Field(default_factory=LlmCallConfig, description="LLM call execution configuration (concurrency / rate shaping)")
    channel_connections: ChannelConnectionsConfig = Field(
        default_factory=ChannelConnectionsConfig,
        description=format_field_description(
            "channel_connections",
            field_doc="User-facing IM channel connection configuration.",
        ),
    )
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig, description="Loop detection middleware configuration")
    tool_progress: ToolProgressConfig = Field(default_factory=ToolProgressConfig, description="Tool progress state machine middleware configuration")
    read_before_write: ReadBeforeWriteConfig = Field(default_factory=ReadBeforeWriteConfig, description="Read-before-write file gate middleware configuration")
    safety_finish_reason: SafetyFinishReasonConfig = Field(default_factory=SafetyFinishReasonConfig, description="Provider safety-filter finish_reason interception middleware configuration")
    auth: AuthAppConfig = Field(default_factory=AuthAppConfig, description="Authentication configuration (local + OIDC SSO)")
    model_config = ConfigDict(extra="allow")
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description=format_field_description(
            "database",
            field_doc="Unified database backend for run/feedback metadata (memory, sqlite, or postgres).",
        ),
    )
    run_events: RunEventsConfig = Field(
        default_factory=RunEventsConfig,
        description=format_field_description(
            "run_events",
            field_doc="Run-event store backend (memory for dev, db for production queries, jsonl for lightweight single-node persistence).",
        ),
    )
    agent_storage: AgentStorageConfig = Field(
        default_factory=AgentStorageConfig,
        description=format_field_description(
            "agent_storage",
            field_doc="Custom agent definition storage backend ('file' for today's per-user on-disk layout, 'db' to share definitions across nodes via the SQL persistence layer).",
        ),
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=format_field_description(
            "scheduler",
            field_doc="Scheduled task runtime configuration (background poller for one-time and cron agent runs).",
        ),
    )
    checkpointer: CheckpointerConfig | None = Field(
        default=None,
        description=format_field_description(
            "checkpointer",
            field_doc="LangGraph state-persistence checkpointer configuration.",
        ),
    )
    stream_bridge: StreamBridgeConfig | None = Field(
        default=None,
        description=format_field_description(
            "stream_bridge",
            field_doc="Stream bridge connecting agent workers to SSE endpoints.",
        ),
    )
    run_ownership: RunOwnershipConfig = Field(
        default_factory=RunOwnershipConfig,
        description=format_field_description(
            "run_ownership",
            field_doc="Run ownership and lease configuration for multi-worker deployments.",
        ),
    )
    dedupe_storage: DedupeStorageConfig = Field(
        default_factory=DedupeStorageConfig,
        description=format_field_description(
            "dedupe_storage",
            field_doc="Inbound webhook dedupe storage backend (memory / postgres / auto) for cross-pod redelivery dedup. See issue #4120.",
        ),
    )

    # 이름 -> config 조회 테이블. validation 이후 ``_build_name_indexes`` 가 (재)구축한다.
    # 덕분에 ``get_model_config`` / ``get_tool_config`` / ``get_tool_group_config`` 가
    # 호출마다 O(n) ``next(...)`` 스캔을 하지 않고 O(1)로 동작한다.
    # private attr이므로 직렬화 대상에서 제외된다.
    _models_by_name: dict[str, ModelConfig] = PrivateAttr(default_factory=dict)
    _tools_by_name: dict[str, ToolConfig] = PrivateAttr(default_factory=dict)
    _tool_groups_by_name: dict[str, ToolGroupConfig] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_null_config_sections(cls, data: Any) -> Any:
        """값이 null인 설정 섹션을 없는 것으로 취급해 기본값이 적용되게 한다.

        최상위 YAML 키 아래 항목을 전부 주석 처리하면(예: 리스트인 ``models:`` 나 객체인 ``memory:``.
        ``config.example.yaml`` 전반이 이런 형태로 배포된다) PyYAML이 값을 ``None``으로 파싱한다.
        이 처리가 없으면 문서화된 ``cp config.example.yaml config.yaml`` 첫 실행 흐름이 해당 섹션에서
        ``Input should be a valid list`` / ``valid dictionary`` 같은 불친절한 pydantic 에러로 죽는다.

        ``None``을 제거하면 각 필드가 기본값으로 떨어진다. 리스트 섹션은 ``default_factory=list`` 로
        ``[]`` 가 되고, 객체 섹션은 기본 설정을 받는다. 기존의 리스트 전용 처리를 기본값이 있는 모든
        섹션으로 일반화한 것이다. ``database`` 섹션은 별개이며 여전히 ``from_file`` 안의
        ``_apply_database_defaults`` 가 담당한다. 그쪽은 null 보정을 넘어 구체적인 기본값까지 적용한다.
        기본값이 없는 필수 섹션(``sandbox``)은 null일 때 의도적으로 계속 에러를 낸다.
        떨어질 곳이 없기 때문이다.
        """
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path:
        """설정 파일 경로를 해석한다.

        우선순위:
        1. `config_path` 인자가 있으면 그것을 쓴다.
        2. `DEER_FLOW_CONFIG_PATH` 환경변수가 있으면 그것을 쓴다.
        3. 없으면 호출자 프로젝트 루트를 찾는다.
        4. 마지막으로 monorepo 호환을 위해 레거시 backend/저장소 루트 기본 경로를 찾는다.
        """
        if config_path:
            path = Path(config_path)
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by param `config_path` not found at {path}")
            return path
        elif os.getenv("DEER_FLOW_CONFIG_PATH"):
            path = Path(os.getenv("DEER_FLOW_CONFIG_PATH"))
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by environment variable `DEER_FLOW_CONFIG_PATH` not found at {path}")
            return path
        else:
            project_config = existing_project_file(("config.yaml",))
            if project_config is not None:
                return project_config

            for path in _legacy_config_candidates():
                if path.exists():
                    return path
            raise FileNotFoundError("`config.yaml` file not found in the project root or legacy backend/repository root locations")

    @classmethod
    def from_file(cls, config_path: str | None = None) -> Self:
        """YAML 파일에서 설정을 로드한다.

        자세한 내용은 `resolve_config_path` 를 참고한다.

        Args:
            config_path: 설정 파일 경로.

        Returns:
            AppConfig: 로드된 설정.
        """
        resolved_path = cls.resolve_config_path(config_path)
        with open(resolved_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # 처리 전에 설정 버전을 확인한다
        cls._check_config_version(config_data, resolved_path)

        config_data = cls.resolve_env_variables(config_data)
        cls._apply_database_defaults(config_data)

        # circuit_breaker 설정이 있으면 로드한다
        if "circuit_breaker" in config_data:
            config_data["circuit_breaker"] = config_data["circuit_breaker"]

        # extensions 설정은 별도 파일이므로 따로 로드하되, config.yaml에 있는 extension 필드는 보존한다.
        # config.yaml이 필드를 명시적으로 선언하면 그쪽이 이긴다.
        # 그 값들은 AppConfig 본체의 hot-reload 계약에 속하기 때문이다.
        yaml_extensions = config_data.get("extensions")
        extensions_config = ExtensionsConfig.from_file()
        extensions_data = extensions_config.model_dump(by_alias=True)
        if isinstance(yaml_extensions, Mapping):
            yaml_extensions_config = ExtensionsConfig.model_validate(yaml_extensions)
            extensions_data.update(yaml_extensions_config.model_dump(by_alias=True, exclude_unset=True))
        config_data["extensions"] = extensions_data

        result = cls.model_validate(config_data)
        if not result.models:
            logger.warning(
                "No models are configured in %s. Add at least one entry under `models:` (see the commented examples in config.example.yaml) or run `make setup`.",
                resolved_path,
            )
        acp_agents = cls._validate_acp_agents(config_data.get("acp_agents", {}))
        cls._apply_singleton_configs(result, acp_agents)
        return result

    @classmethod
    def _validate_acp_agents(
        cls,
        config_data: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, ACPAgentConfig]:
        if config_data is None:
            config_data = {}
        return {name: ACPAgentConfig(**cfg) for name, cfg in config_data.items()}

    @classmethod
    def _apply_singleton_configs(cls, config: Self, acp_agents: dict[str, ACPAgentConfig]) -> None:
        from deerflow.config.checkpointer_config import get_checkpointer_config

        previous_checkpointer_config = get_checkpointer_config()

        load_title_config_from_dict(config.title.model_dump())
        load_summarization_config_from_dict(config.summarization.model_dump())
        load_memory_config_from_dict(config.memory.model_dump())
        load_agents_api_config_from_dict(config.agents_api.model_dump())
        load_subagents_config_from_dict(config.subagents.model_dump())
        load_tool_search_config_from_dict(config.tool_search.model_dump())
        load_guardrails_config_from_dict(config.guardrails.model_dump())
        load_authorization_config_from_dict(config.authorization.model_dump())
        load_checkpointer_config_from_dict(config.checkpointer.model_dump() if config.checkpointer is not None else None)
        load_stream_bridge_config_from_dict(config.stream_bridge.model_dump() if config.stream_bridge is not None else None)
        load_acp_config_from_dict({name: agent.model_dump() for name, agent in acp_agents.items()})

        if previous_checkpointer_config != config.checkpointer:
            # 이 런타임 싱글턴들은 backend를 checkpointer 설정에서 가져온다.
            # 두 provider 모두 get_app_config를 import하므로 순환을 피하려고 import를 지역에 둔다.
            #
            # 통합 ``database`` 섹션은 의도적으로 여기서 다루지 않는다.
            # ``database`` 는 재시작이 필요한 필드다(reload_boundary.STARTUP_ONLY_FIELDS).
            # ``init_engine_from_config()`` 는 시작 시 ORM engine을 한 번 만들고 config.yaml 수정에도
            # 다시 만들지 않는다. 운영 중 ``database``/``postgres_schema`` 가 바뀌었을 때 동기
            # checkpointer/store 싱글턴만 리셋하면 배포가 반쪽만 마이그레이션된다. 새 checkpoint/store
            # 테이블은 새 schema에 생기는데 ORM row는 계속 옛 schema로 들어가고, 에러도 드러나지 않는다.
            # 문서화된 재시작을 요구해야 배포가 일관된 상태로 유지된다.
            from deerflow.runtime.checkpointer import reset_checkpointer
            from deerflow.runtime.store import reset_store

            reset_checkpointer()
            reset_store()

    @classmethod
    def _apply_database_defaults(cls, config_data: dict[str, Any]) -> None:
        """섹션이 없을 때 영속성 관련 config.yaml 기본값을 적용한다."""
        database_config = config_data.get("database")
        if database_config is None:
            database_config = {}
            config_data["database"] = database_config
        if not isinstance(database_config, dict):
            return
        for key, value in CONFIG_FILE_DATABASE_DEFAULTS.items():
            database_config.setdefault(key, value)

    @classmethod
    def _check_config_version(cls, config_data: dict, config_path: Path) -> None:
        """사용자의 config.yaml이 config.example.yaml보다 오래됐는지 확인한다.

        사용자 config_version이 예제보다 낮으면 경고를 남긴다.
        config_version이 없으면 버전 0(버전 관리 이전)으로 취급한다.
        """
        try:
            user_version = int(config_data.get("config_version", 0))
        except (TypeError, ValueError):
            user_version = 0

        # config.yaml이 있는 디렉터리와 상위 디렉터리를 훑어 config.example.yaml을 찾는다
        example_path = None
        search_dir = config_path.parent
        for _ in range(5):  # 최대 5단계까지 탐색
            candidate = search_dir / "config.example.yaml"
            if candidate.exists():
                example_path = candidate
                break
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent
        if example_path is None:
            return

        try:
            with open(example_path, encoding="utf-8") as f:
                example_data = yaml.safe_load(f)
            raw = example_data.get("config_version", 0) if example_data else 0
            try:
                example_version = int(raw)
            except (TypeError, ValueError):
                example_version = 0
        except Exception:
            return

        if user_version < example_version:
            logger.warning(
                "Your config.yaml (version %d) is outdated — the latest version is %d. Run `make config-upgrade` to merge new fields into your config.",
                user_version,
                example_version,
            )

    @classmethod
    def resolve_env_variables(cls, config: Any) -> Any:
        """설정 안의 환경변수를 재귀적으로 해석한다.

        환경변수는 `os.getenv` 로 해석한다. 예: $OPENAI_API_KEY

        Args:
            config: 환경변수를 해석할 설정.

        Returns:
            환경변수가 해석된 설정.
        """
        if isinstance(config, str):
            if config.startswith("$"):
                env_value = os.getenv(config[1:])
                if env_value is None:
                    raise ValueError(f"Environment variable {config[1:]} not found for config value {config}")
                return env_value
            return config
        elif isinstance(config, dict):
            return {k: cls.resolve_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls.resolve_env_variables(item) for item in config]
        return config

    @model_validator(mode="after")
    def _build_name_indexes(self) -> "AppConfig":
        """``get_*_config`` 를 O(1)로 만들기 위한 이름 -> config 조회 테이블을 만든다.

        ``get_tool_config`` 는 community 도구 호출마다 2~3번, ``get_model_config`` 는 agent를
        만들 때마다 여러 번 호출되므로 기존 O(n) ``next(...)`` 스캔은 hot path에 있었다.
        여기서 다시 만들어, 설정 reload(새 ``AppConfig`` 를 생성한다)가 테이블도 갱신하게 한다.
        ``setdefault`` 는 이름이 중복되면 첫 항목을 유지하므로 기존 ``next(...)`` 의
        first-match 의미가 그대로 보존된다.
        """
        models_by_name: dict[str, ModelConfig] = {}
        for model in self.models:
            models_by_name.setdefault(model.name, model)
        tools_by_name: dict[str, ToolConfig] = {}
        for tool in self.tools:
            tools_by_name.setdefault(tool.name, tool)
        tool_groups_by_name: dict[str, ToolGroupConfig] = {}
        for group in self.tool_groups:
            tool_groups_by_name.setdefault(group.name, group)
        self._models_by_name = models_by_name
        self._tools_by_name = tools_by_name
        self._tool_groups_by_name = tool_groups_by_name
        return self

    def get_model_config(self, name: str) -> ModelConfig | None:
        """이름으로 모델 설정을 가져온다.

        Args:
            name: 설정을 가져올 모델 이름.

        Returns:
            찾으면 모델 설정, 없으면 None.
        """
        return self._models_by_name.get(name)

    def get_tool_config(self, name: str) -> ToolConfig | None:
        """이름으로 도구 설정을 가져온다.

        Args:
            name: 설정을 가져올 도구 이름.

        Returns:
            찾으면 도구 설정, 없으면 None.
        """
        return self._tools_by_name.get(name)

    def get_tool_group_config(self, name: str) -> ToolGroupConfig | None:
        """이름으로 tool group 설정을 가져온다.

        Args:
            name: 설정을 가져올 tool group 이름.

        Returns:
            찾으면 tool group 설정, 없으면 None.
        """
        return self._tool_groups_by_name.get(name)


# 아직 명시적인 ``AppConfig`` 전달 방식으로 옮기지 못한 코드 경로를 위한 호환 싱글턴 계층.
# 새 조립 지점에서는 ``AppConfig`` 를 한 번 만들어 직접 내려주는 방식을 택한다.
_app_config: AppConfig | None = None
_app_config_path: Path | None = None
_app_config_mtime: float | None = None
_app_config_signature: _ConfigSignature | None = None
_app_config_is_custom = False
_current_app_config: ContextVar[AppConfig | None] = ContextVar("deerflow_current_app_config", default=None)
_current_app_config_stack: ContextVar[tuple[AppConfig | None, ...]] = ContextVar("deerflow_current_app_config_stack", default=())


def _get_config_mtime(config_path: Path) -> float | None:
    """설정 파일이 있으면 수정 시각을 반환한다."""
    try:
        return config_path.stat().st_mtime
    except OSError:
        return None


def _load_and_cache_app_config(config_path: str | None = None) -> AppConfig:
    """디스크에서 설정을 로드하고 캐시 메타데이터를 갱신한다."""
    global _app_config, _app_config_path, _app_config_mtime, _app_config_signature, _app_config_is_custom

    resolved_path = AppConfig.resolve_config_path(config_path)
    _app_config = AppConfig.from_file(str(resolved_path))
    _app_config_path = resolved_path
    _app_config_mtime = _get_config_mtime(resolved_path)
    _app_config_signature = _get_config_signature(resolved_path)
    _app_config_is_custom = False
    return _app_config


def get_app_config() -> AppConfig:
    """DeerFlow 설정 인스턴스를 반환한다.

    캐시된 싱글턴을 반환하며, 설정 파일 경로나 내용 signature가 바뀌면 자동으로 다시 로드한다.
    강제 reload는 `reload_app_config()`, 캐시 비우기는 `reset_app_config()` 를 쓴다.
    """
    global _app_config, _app_config_path, _app_config_mtime, _app_config_signature

    runtime_override = _current_app_config.get()
    if runtime_override is not None:
        return runtime_override

    if _app_config is not None and _app_config_is_custom:
        return _app_config

    resolved_path = AppConfig.resolve_config_path()
    current_mtime = _get_config_mtime(resolved_path)
    current_signature = _get_config_signature(resolved_path)

    should_reload = _app_config is None or _app_config_path != resolved_path or _app_config_signature != current_signature
    if should_reload:
        if _app_config_path == resolved_path and _app_config_mtime is not None and current_mtime is not None and _app_config_mtime != current_mtime:
            logger.info(
                "Config file has been modified (mtime: %s -> %s), reloading AppConfig",
                _app_config_mtime,
                current_mtime,
            )
        elif _app_config_path == resolved_path and _app_config_signature != current_signature:
            logger.info("Config file content signature changed, reloading AppConfig")
        _load_and_cache_app_config(str(resolved_path))
    return _app_config


def reload_app_config(config_path: str | None = None) -> AppConfig:
    """파일에서 설정을 다시 읽고 캐시된 인스턴스를 갱신한다.

    설정 파일이 수정됐고 애플리케이션을 재시작하지 않고 변경을 반영하고 싶을 때 쓴다.

    Args:
        config_path: 선택적 설정 파일 경로. 없으면 기본 해석 전략을 쓴다.

    Returns:
        새로 로드된 AppConfig 인스턴스.
    """
    return _load_and_cache_app_config(config_path)


def reset_app_config() -> None:
    """캐시된 설정 인스턴스를 초기화한다.

    싱글턴 캐시를 비워서 다음 `get_app_config()` 호출이 파일에서 다시 읽게 한다.
    테스트나 서로 다른 설정을 전환할 때 유용하다.
    """
    global _app_config, _app_config_path, _app_config_mtime, _app_config_signature, _app_config_is_custom
    _app_config = None
    _app_config_path = None
    _app_config_mtime = None
    _app_config_signature = None
    _app_config_is_custom = False


def set_app_config(config: AppConfig) -> None:
    """커스텀 설정 인스턴스를 지정한다.

    테스트 목적으로 커스텀이나 mock 설정을 주입할 수 있게 한다.

    Args:
        config: 사용할 AppConfig 인스턴스.
    """
    global _app_config, _app_config_path, _app_config_mtime, _app_config_signature, _app_config_is_custom
    _app_config = config
    _app_config_path = None
    _app_config_mtime = None
    _app_config_signature = None
    _app_config_is_custom = True


def peek_current_app_config() -> AppConfig | None:
    """활성화된 런타임 범위 AppConfig 오버라이드가 있으면 반환한다."""
    return _current_app_config.get()


def push_current_app_config(config: AppConfig) -> None:
    """현재 실행 컨텍스트에 런타임 범위 AppConfig 오버라이드를 push한다."""
    stack = _current_app_config_stack.get()
    _current_app_config_stack.set(stack + (_current_app_config.get(),))
    _current_app_config.set(config)


def pop_current_app_config() -> None:
    """현재 실행 컨텍스트의 가장 최근 런타임 범위 AppConfig 오버라이드를 pop한다."""
    stack = _current_app_config_stack.get()
    if not stack:
        _current_app_config.set(None)
        return
    previous = stack[-1]
    _current_app_config_stack.set(stack[:-1])
    _current_app_config.set(previous)
