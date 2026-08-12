"""대화 summarization 설정."""

from typing import Literal

from pydantic import BaseModel, Field

ContextSizeType = Literal["fraction", "tokens", "messages"]
DEFAULT_SKILL_FILE_READ_TOOL_NAMES: tuple[str, ...] = ("read_file", "read", "view", "cat")


class ContextSize(BaseModel):
    """trigger 또는 keep 파라미터에 쓰는 context 크기 명세."""

    type: ContextSizeType = Field(description="Type of context size specification")
    value: int | float = Field(description="Value for the context size specification")

    def to_tuple(self) -> tuple[ContextSizeType, int | float]:
        """SummarizationMiddleware가 기대하는 tuple 형식으로 변환한다."""
        return (self.type, self.value)


class SummarizationConfig(BaseModel):
    """대화 자동 summarization 설정."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable automatic conversation summarization",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name to use for summarization. None = summarize with the model the run "
        "actually executes with (the lead run's model, a subagent's own model, or a thread's "
        "custom-agent model), not config.models[0]. When set, that model generates and the run's "
        "own model is used as a fallback if the configured summary provider fails.",
    )
    trigger: ContextSize | list[ContextSize] | None = Field(
        default=None,
        description="One or more thresholds that trigger summarization. When any threshold is met, summarization runs. "
        "Examples: {'type': 'messages', 'value': 50} triggers at 50 messages, "
        "{'type': 'tokens', 'value': 4000} triggers at 4000 tokens, "
        "{'type': 'fraction', 'value': 0.8} triggers at 80% of model's max input tokens",
    )
    keep: ContextSize = Field(
        default_factory=lambda: ContextSize(type="messages", value=20),
        description="Context retention policy after summarization. Specifies how much history to preserve. "
        "Examples: {'type': 'messages', 'value': 20} keeps 20 messages, "
        "{'type': 'tokens', 'value': 3000} keeps 3000 tokens, "
        "{'type': 'fraction', 'value': 0.3} keeps 30% of model's max input tokens",
    )
    trim_tokens_to_summarize: int | None = Field(
        default=4000,
        description="Maximum tokens to keep when preparing messages for summarization. Pass null to skip trimming.",
    )
    summary_prompt: str | None = Field(
        default=None,
        description="Custom prompt template for generating summaries. If not provided, uses the default LangChain prompt.",
    )
    skill_file_read_tool_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SKILL_FILE_READ_TOOL_NAMES),
        description="Tool names treated as skill-file reads when capturing loaded skills into the durable skill_context channel.",
    )


# 전역 설정 인스턴스
_summarization_config: SummarizationConfig = SummarizationConfig()


def get_summarization_config() -> SummarizationConfig:
    """현재 summarization 설정을 반환한다."""
    return _summarization_config


def set_summarization_config(config: SummarizationConfig) -> None:
    """summarization 설정을 지정한다."""
    global _summarization_config
    _summarization_config = config


def load_summarization_config_from_dict(config_dict: dict) -> None:
    """dict에서 summarization 설정을 읽어 들인다."""
    global _summarization_config
    _summarization_config = SummarizationConfig(**config_dict)
