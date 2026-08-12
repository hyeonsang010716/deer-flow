"""thread 제목 자동 생성 설정."""

from pydantic import BaseModel, Field


class TitleConfig(BaseModel):
    """thread 제목 자동 생성 설정."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable automatic title generation",
    )
    max_words: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of words in the generated title",
    )
    max_chars: int = Field(
        default=60,
        ge=10,
        le=200,
        description="Maximum number of characters in the generated title",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name to use for LLM title generation (None = use local fallback title)",
    )
    prompt_template: str = Field(
        default=("Generate a concise title (max {max_words} words) for this conversation.\nUser: {user_msg}\nAssistant: {assistant_msg}\n\nReturn ONLY the title, no quotes, no explanation."),
        description="Prompt template for LLM title generation when model_name is set",
    )


# 전역 설정 인스턴스
_title_config: TitleConfig = TitleConfig()


def get_title_config() -> TitleConfig:
    """현재 title 설정을 반환한다."""
    return _title_config


def set_title_config(config: TitleConfig) -> None:
    """title 설정을 지정한다."""
    global _title_config
    _title_config = config


def load_title_config_from_dict(config_dict: dict) -> None:
    """dict에서 title 설정을 읽어 들인다."""
    global _title_config
    _title_config = TitleConfig(**config_dict)


def reset_title_config() -> None:
    """title 설정을 원래의 ``TitleConfig()`` 기본값으로 되돌린다.

    테스트가 private한 ``_title_config`` 모듈 속성을 직접 건드리지 않도록 공개 API로 둔다.
    ``AppConfig.from_file()``이 :func:`load_title_config_from_dict`를 호출해 싱글턴을 영구히
    변경하므로, 케이스 사이에 깨끗한 상태가 필요한 테스트는 이 함수를 호출해야 한다.
    """
    global _title_config
    _title_config = TitleConfig()
