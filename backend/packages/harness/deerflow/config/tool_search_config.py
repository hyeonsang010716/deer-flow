"""tool_search를 통한 deferred tool 로딩 설정."""

from pydantic import BaseModel, Field, field_validator

AUTO_PROMOTE_TOP_K_MIN = 1
AUTO_PROMOTE_TOP_K_MAX = 5


def clamp_auto_promote_top_k(value: int) -> int:
    """전역 MCP routing auto-promote 범위를 PR2의 허용 범위로 자른다."""
    return max(AUTO_PROMOTE_TOP_K_MIN, min(AUTO_PROMOTE_TOP_K_MAX, int(value)))


class ToolSearchConfig(BaseModel):
    """tool_search를 통한 deferred tool 로딩 설정.

    켜면 MCP tool이 agent context에 직접 로드되지 않는다. 대신 system prompt에 이름만
    나열되고, runtime에 tool_search tool로 발견할 수 있다.
    """

    enabled: bool = Field(
        default=False,
        description="Defer tools and enable tool_search",
    )
    auto_promote_top_k: int = Field(
        default=3,
        description="Maximum number of deferred MCP tool schemas auto-promoted from routing metadata per model call",
    )

    @field_validator("auto_promote_top_k")
    @classmethod
    def _clamp_auto_promote_top_k(cls, value: int) -> int:
        return clamp_auto_promote_top_k(value)


_tool_search_config: ToolSearchConfig | None = None


def get_tool_search_config() -> ToolSearchConfig:
    """tool search 설정을 반환한다. 필요하면 AppConfig에서 읽어 들인다."""
    global _tool_search_config
    if _tool_search_config is None:
        _tool_search_config = ToolSearchConfig()
    return _tool_search_config


def load_tool_search_config_from_dict(data: dict) -> ToolSearchConfig:
    """dict에서 tool search 설정을 읽어 들인다(AppConfig 로딩 중에 호출된다)."""
    global _tool_search_config
    _tool_search_config = ToolSearchConfig.model_validate(data)
    return _tool_search_config
