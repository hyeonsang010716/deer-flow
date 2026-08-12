"""도구 호출 전 authorization 설정."""

from pydantic import BaseModel, Field


class GuardrailProviderConfig(BaseModel):
    """guardrail provider 설정."""

    use: str = Field(description="Class path (e.g. 'deerflow.guardrails.builtin:AllowlistProvider')")
    config: dict = Field(default_factory=dict, description="Provider-specific settings passed as kwargs")


class GuardrailsConfig(BaseModel):
    """도구 호출 전 authorization 설정.

    활성화하면 모든 도구 호출이 실행 전에 설정된 provider를 거친다. provider는 도구
    이름, 인자, 에이전트의 passport 참조를 받아 허용/거부를 결정한다.
    """

    enabled: bool = Field(default=False, description="Enable guardrail middleware")
    fail_closed: bool = Field(default=True, description="Block tool calls if provider errors")
    passport: str | None = Field(default=None, description="OAP passport path or hosted agent ID")
    provider: GuardrailProviderConfig | None = Field(default=None, description="Guardrail provider configuration")


_guardrails_config: GuardrailsConfig | None = None


def get_guardrails_config() -> GuardrailsConfig:
    """guardrails 설정을 반환한다. 아직 로드되지 않았으면 기본값을 쓴다."""
    global _guardrails_config
    if _guardrails_config is None:
        _guardrails_config = GuardrailsConfig()
    return _guardrails_config


def load_guardrails_config_from_dict(data: dict) -> GuardrailsConfig:
    """dict에서 guardrails 설정을 읽는다(AppConfig 로딩 중 호출된다)."""
    global _guardrails_config
    _guardrails_config = GuardrailsConfig.model_validate(data)
    return _guardrails_config


def reset_guardrails_config() -> None:
    """캐시된 설정 인스턴스를 비운다. 테스트에서 singleton 누수를 막는 데 쓴다."""
    global _guardrails_config
    _guardrails_config = None
