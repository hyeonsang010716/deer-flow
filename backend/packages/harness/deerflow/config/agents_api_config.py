"""커스텀 agent 관리 API 설정."""

from pydantic import BaseModel, Field


class AgentsApiConfig(BaseModel):
    """커스텀 agent 및 사용자 프로필 관리 route 설정."""

    enabled: bool = Field(
        default=False,
        description=("Whether to expose the custom-agent management API over HTTP. When disabled, the gateway rejects read/write access to custom agent SOUL.md, config, and USER.md prompt-management routes."),
    )


_agents_api_config: AgentsApiConfig = AgentsApiConfig()


def get_agents_api_config() -> AgentsApiConfig:
    """현재 agents API 설정을 반환한다."""
    return _agents_api_config


def set_agents_api_config(config: AgentsApiConfig) -> None:
    """agents API 설정을 지정한다."""
    global _agents_api_config
    _agents_api_config = config


def load_agents_api_config_from_dict(config_dict: dict) -> None:
    """dict에서 agents API 설정을 읽어 들인다."""
    global _agents_api_config
    _agents_api_config = AgentsApiConfig(**config_dict)
