from __future__ import annotations

from typing import Protocol

from deerflow.mcp.tasks.models import TaskReference, TaskSnapshot, TaskSubmission, TaskSubmitRequest


class McpTaskDriver(Protocol):
    """protocol 중립 task runtime이 쓰는 transport/protocol adapter."""

    async def submit(self, request: TaskSubmitRequest) -> TaskSubmission: ...

    async def get_status(self, task: TaskReference) -> TaskSnapshot: ...

    async def cancel(self, task: TaskReference) -> TaskSnapshot: ...


class McpTaskDriverRegistry:
    """Gateway 시작 시 구성되는 프로세스 로컬 driver 카탈로그."""

    def __init__(self) -> None:
        self._drivers: dict[str, McpTaskDriver] = {}

    def register(self, name: str, driver: McpTaskDriver) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("driver name must not be empty")
        if normalized in self._drivers:
            raise ValueError(f"MCP task driver {normalized!r} is already registered")
        self._drivers[normalized] = driver

    def get(self, name: str) -> McpTaskDriver | None:
        return self._drivers.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._drivers))
