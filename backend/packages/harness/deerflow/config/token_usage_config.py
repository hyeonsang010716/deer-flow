from pydantic import BaseModel, Field


class TokenUsageConfig(BaseModel):
    """token 사용량 추적 설정."""

    enabled: bool = Field(default=True, description="Enable token usage tracking middleware")
