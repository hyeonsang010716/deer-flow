"""subagent 설정 정의."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig


@dataclass
class SubagentConfig:
    """subagent 설정.

    Attributes:
        name: subagent의 고유 식별자.
        description: Claude가 이 subagent에 위임해야 하는 상황.
        system_prompt: subagent의 동작을 이끄는 system prompt.
        tools: 허용할 tool 이름 목록(선택). None이면 모든 tool을 상속한다.
        disallowed_tools: 거부할 tool 이름 목록(선택).
        skills: 발견 및 활성화를 허용할 skill 이름 목록(선택). None이면 enabled된 모든 skill을
                쓸 수 있고, 빈 리스트면 이 subagent에서 skill이 비활성화된다. skill 본문과
                allowed-tools 정책은 runtime에 활성화/로드된 뒤에만 적용된다.
        model: 사용할 model. 'inherit'이면 부모의 model을 쓴다.
        max_turns: 중단 전 최대 agent turn 수. 전역 ``subagents.max_turns``가 설정되지 않은
            한 built-in agent는 여기 값을 쓴다(general-purpose=150, bash=60).
        timeout_seconds: 최후의 fallback 실행 시간 상한. built-in agent의 실효 제한은 registry가
            덧씌우는 전역 ``subagents.timeout_seconds``(기본 1800 = 30분)이며, 여기의 900은
            다른 전역 값이 없을 때만 적용된다.
    """

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900


def _default_model_name(app_config: "AppConfig") -> str:
    if not app_config.models:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")
    return app_config.models[0].name


def resolve_subagent_model_name(config: SubagentConfig, parent_model: str | None, *, app_config: "AppConfig | None" = None) -> str:
    """subagent가 실제로 사용할 model 이름을 해석한다."""
    if config.model != "inherit":
        return config.model

    if parent_model is not None:
        return parent_model

    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()
    return _default_model_name(app_config)
