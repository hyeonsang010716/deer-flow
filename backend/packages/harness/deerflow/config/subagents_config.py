"""config.yaml에서 읽어들이는 subagent 시스템 설정."""

import logging

from pydantic import BaseModel, Field

from deerflow.config.token_budget_config import TokenBudgetConfig

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN = 6
MIN_TOTAL_SUBAGENTS_PER_RUN = 1
MAX_TOTAL_SUBAGENTS_PER_RUN = 50
MIN_CONCURRENT_SUBAGENT_CALLS = 1
MAX_CONCURRENT_SUBAGENT_CALLS = 4


def clamp_subagent_concurrency(value: int) -> int:
    """응답 단위 task 호출 동시성을 middleware가 강제하는 범위로 클램프한다."""
    return max(MIN_CONCURRENT_SUBAGENT_CALLS, min(MAX_CONCURRENT_SUBAGENT_CALLS, value))


def clamp_total_subagents_per_run(value: int) -> int:
    """run 단위 task 위임 총량을 middleware가 강제하는 범위로 클램프한다."""
    return max(MIN_TOTAL_SUBAGENTS_PER_RUN, min(MAX_TOTAL_SUBAGENTS_PER_RUN, value))


def default_subagent_token_budget(*, summarization_enabled: bool = False) -> TokenBudgetConfig:
    """subagent의 run 단위 기본 token budget(#3875 Phase 2 → Phase 3 연동).

    비정상적인 token 소모를 막는 backstop이 실제로 작동하도록 기본 활성화한다(umbrella
    #3857 4번: backstop은 존재만 해서는 안 되고 작동해야 한다). ``max_tokens``는 **subagent
    summarization 활성 여부와 연동된다**(#3875 Phase 3 리뷰 지적).

    - ``summarization_enabled=True``(Phase 3가 컨텍스트가 비정상적으로 커지기 전에 압축):
      **1M**. 더 빡빡한 상한이지만 정상적인 심층 조사는 여전히 커버하면서 비정상 run은
      더 일찍 잡는다.
    - ``summarization_enabled=False``: **2M**(Phase 2의 상한). Phase 2 docstring이 지적했듯
      summarization 없이 ``max_turns=150``으로 도는 정상적인 심층 조사 run은 누적 입력이
      1M을 넘길 수 있어, 압축 없이 1M 상한을 두면 조기에 잘린다. 여기서 2M을 유지해 여유를
      남기고, 더 빡빡한 1M은 그것을 정당화하는 압축이 실제로 돌 때만 적용한다.

    모델 수준 ``default_factory``(``SubagentsAppConfig.token_budget``)는 형제 최상위 필드인
    ``summarization.enabled``를 읽을 수 없어 압축 없음 기준인 2M을 쓴다. 빌더
    (``build_subagent_runtime_middlewares``)가 ``get_token_budget_for(...,
    summarization_enabled=...)``로 다시 계산해 실제 스위치를 반영한다. 사용자가 지정한
    ``token_budget``(전역이든 agent별이든)은 스위치와 무관하게 항상 우선한다. 튜닝 가능 항목.
    """
    max_tokens = 1_000_000 if summarization_enabled else 2_000_000
    return TokenBudgetConfig(enabled=True, max_tokens=max_tokens, warn_threshold=0.7)


class SubagentOverrideConfig(BaseModel):
    """agent 단위 설정 override."""

    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Timeout in seconds for this subagent (None = use global default)",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Maximum turns for this subagent (None = use global or builtin default)",
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Model name for this subagent (None = inherit from parent agent)",
    )
    skills: list[str] | None = Field(
        default=None,
        description="Skill names whitelist for this subagent (None = inherit all enabled skills, [] = no skills)",
    )
    token_budget: TokenBudgetConfig | None = Field(
        default=None,
        description="Per-run token budget override for this subagent (None = use the global subagents.token_budget default). Symmetric with timeout_seconds/max_turns.",
    )


class CustomSubagentConfig(BaseModel):
    """config.yaml에 선언한 사용자 정의 subagent 타입."""

    description: str = Field(
        description="When the lead agent should delegate to this subagent",
    )
    system_prompt: str = Field(
        description="System prompt that guides the subagent's behavior",
    )
    tools: list[str] | None = Field(
        default=None,
        description="Tool names whitelist (None = inherit all tools from parent)",
    )
    disallowed_tools: list[str] | None = Field(
        default_factory=lambda: ["task", "ask_clarification", "present_files"],
        description="Tool names to deny",
    )
    skills: list[str] | None = Field(
        default=None,
        description="Skill names whitelist (None = inherit all enabled skills, [] = no skills)",
    )
    model: str = Field(
        default="inherit",
        description="Model to use - 'inherit' uses parent's model",
    )
    max_turns: int = Field(
        default=50,
        ge=1,
        description="Maximum number of agent turns before stopping",
    )
    timeout_seconds: int = Field(
        default=900,
        ge=1,
        description="Maximum execution time in seconds",
    )


class SubagentsAppConfig(BaseModel):
    """subagent 시스템 설정."""

    timeout_seconds: int = Field(
        default=1800,
        ge=1,
        description="Default timeout in seconds for built-in subagents (default: 1800 = 30 minutes); custom agents use their own timeout_seconds unless given a per-agent override",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Optional default max-turn override for all subagents (None = keep builtin defaults)",
    )
    max_total_per_run: int = Field(
        default=DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN,
        ge=MIN_TOTAL_SUBAGENTS_PER_RUN,
        le=MAX_TOTAL_SUBAGENTS_PER_RUN,
        description="Default total number of subagent delegations allowed in one lead-agent run. This is a deterministic backstop against repeated legal-sized task batches. Valid range: 1-50.",
    )
    token_budget: TokenBudgetConfig = Field(
        default_factory=default_subagent_token_budget,
        description="Default per-run token budget for subagents — a cost-ceiling backstop that engages by default (#3875 Phase 2). Set enabled: false to disable, or override per agent via agents.<name>.token_budget.",
    )
    agents: dict[str, SubagentOverrideConfig] = Field(
        default_factory=dict,
        description="Per-agent configuration overrides keyed by agent name",
    )
    custom_agents: dict[str, CustomSubagentConfig] = Field(
        default_factory=dict,
        description="User-defined subagent types keyed by agent name",
    )

    # 사용자가 ``token_budget``을 명시하지 않아 default_factory로 채워졌을 때 True.
    # ``get_token_budget_for``가 상한을 ``summarization.enabled``에 다시 연동해도 되는지
    # 판단하는 데 쓴다(#3875 Phase 3). 사용자가 지정한 budget은 항상 그대로 존중한다.
    # ``__init__``이 ``model_fields_set``으로 설정하며, app-config reload 경로에서도
    # 유지된다(재구성 전에 기본값 ``token_budget``을 제거한다 —
    # ``load_subagents_config_from_dict`` 참고).
    _token_budget_is_default: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        self._token_budget_is_default = "token_budget" not in self.model_fields_set

    def get_timeout_for(self, agent_name: str) -> int:
        """특정 agent의 실효 timeout을 반환한다.

        Args:
            agent_name: subagent 이름.

        Returns:
            초 단위 timeout. agent별 override가 있으면 그것을, 없으면 전역 기본값을 쓴다.
        """
        override = self.agents.get(agent_name)
        if override is not None and override.timeout_seconds is not None:
            return override.timeout_seconds
        return self.timeout_seconds

    def get_model_for(self, agent_name: str) -> str | None:
        """특정 agent의 model override를 반환한다.

        Args:
            agent_name: subagent 이름.

        Returns:
            override된 model 이름, 없으면 None(subagent가 부모의 model을 상속한다).
        """
        override = self.agents.get(agent_name)
        if override is not None and override.model is not None:
            return override.model
        return None

    def get_max_turns_for(self, agent_name: str, builtin_default: int) -> int:
        """특정 agent의 실효 max_turns를 반환한다."""
        override = self.agents.get(agent_name)
        if override is not None and override.max_turns is not None:
            return override.max_turns
        if self.max_turns is not None:
            return self.max_turns
        return builtin_default

    def get_skills_for(self, agent_name: str) -> list[str] | None:
        """특정 agent의 skills override를 반환한다.

        Args:
            agent_name: subagent 이름.

        Returns:
            override된 skill 이름 whitelist, 없으면 None(subagent가 활성화된 모든 skill을
            상속한다).
        """
        override = self.agents.get(agent_name)
        if override is not None and override.skills is not None:
            return override.skills
        return None

    def get_token_budget_for(
        self,
        agent_name: str,
        *,
        summarization_enabled: bool = False,
    ) -> TokenBudgetConfig:
        """특정 agent의 실효 token-budget 설정을 반환한다.

        custom agent 자신의 값을 유지하는 ``max_turns``/``timeout_seconds``와 달리, token
        budget은 명시적으로 끄지 않는 한 모든 subagent에 걸려야 하는 안전 backstop이다. 따라서
        agent별 override가 있으면 그것이 이기고, 없으면 전역 기본값이 built-in과 custom agent에
        똑같이 적용된다(#3875 Phase 2 / umbrella #3857 4번).

        ``summarization_enabled``는 **기본** 상한을 subagent summarization 활성 여부와
        연동한다(#3875 Phase 3 리뷰): 압축이 돌면 1M, 아니면 2M. 오직 기본값에만 영향을 준다 —
        명시적으로 설정된 ``token_budget``(전역이든 agent별이든)이 항상 우선하므로, 값을 고정한
        배포가 summarization 스위치를 바꿨다고 조용히 달라지지 않는다.
        """
        override = self.agents.get(agent_name)
        if override is not None and override.token_budget is not None:
            return override.token_budget
        # 호출자가 기본값을 쓰는 경우(전역 token_budget을 명시하지 않은 경우)에만 다시 계산한다.
        # 사용자가 지정한 전역 값은 그대로 존중한다.
        if self._token_budget_is_default:
            return default_subagent_token_budget(summarization_enabled=summarization_enabled)
        return self.token_budget


_subagents_config: SubagentsAppConfig = SubagentsAppConfig()


def get_subagents_app_config() -> SubagentsAppConfig:
    """현재 subagents 설정을 반환한다."""
    return _subagents_config


def load_subagents_config_from_dict(config_dict: dict) -> None:
    """dict에서 subagents 설정을 읽어들인다."""
    global _subagents_config
    # app-config reload 경로(app_config.py)는 ``config.subagents.model_dump()``로 왕복하는데,
    # 이때 기본값 ``token_budget``도 dict에 직렬화된다. 그 dict로 재구성하면
    # ``model_fields_set``에 ``token_budget``이 들어가 ``_token_budget_is_default``가 False가
    # 되고, ``get_token_budget_for``의 summarization 연동 재계산이 깨진다(#3875 Phase 3).
    # 값이 여전히 압축 없음 기준 기본값과 같으면 키를 제거해, 재구성 시 default_factory가
    # 동작하고 "사용자가 설정하지 않았다"는 신호가 보존되게 한다.
    tb = config_dict.get("token_budget")
    if tb is not None and tb == default_subagent_token_budget(summarization_enabled=False).model_dump():
        config_dict = {k: v for k, v in config_dict.items() if k != "token_budget"}
    _subagents_config = SubagentsAppConfig(**config_dict)

    overrides_summary = {}
    for name, override in _subagents_config.agents.items():
        parts = []
        if override.timeout_seconds is not None:
            parts.append(f"timeout={override.timeout_seconds}s")
        if override.max_turns is not None:
            parts.append(f"max_turns={override.max_turns}")
        if override.model is not None:
            parts.append(f"model={override.model}")
        if override.skills is not None:
            parts.append(f"skills={override.skills}")
        if parts:
            overrides_summary[name] = ", ".join(parts)

    custom_agents_names = list(_subagents_config.custom_agents.keys())

    if overrides_summary or custom_agents_names:
        logger.info(
            "Subagents config loaded: default timeout=%ss, default max_turns=%s, per-agent overrides=%s, custom_agents=%s",
            _subagents_config.timeout_seconds,
            _subagents_config.max_turns,
            overrides_summary or "none",
            custom_agents_names or "none",
        )
