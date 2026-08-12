"""네이티브 skill 안전성 검사 설정."""

from pydantic import BaseModel, Field


class SkillScanConfig(BaseModel):
    """결정적 SkillScan analyzer 설정."""

    enabled: bool = Field(
        default=True,
        description="Whether native deterministic SkillScan analyzers run before the LLM skill scanner.",
    )
