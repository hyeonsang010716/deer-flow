from pydantic import BaseModel, Field


class McpTasksConfig(BaseModel):
    """프로토콜 중립 MCP task poller의 기동 설정."""

    enabled: bool = Field(default=False)
    poll_interval_seconds: int = Field(default=5, ge=1, le=300)
    lease_seconds: int = Field(default=120, ge=5, le=3600)
    max_concurrent_polls: int = Field(default=8, ge=1, le=64)
