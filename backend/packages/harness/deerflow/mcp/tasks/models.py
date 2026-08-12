from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """장시간 실행되는 MCP 작업의 protocol 중립 lifecycle state."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


POLLABLE_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.SUBMITTED,
        TaskStatus.WORKING,
    }
)
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)
ATTENTION_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.INPUT_REQUIRED,
        *TERMINAL_TASK_STATUSES,
    }
)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """task driver가 반환하는 정규화된 상태 응답 하나."""

    status: TaskStatus
    result: Any | None = None
    error: str | None = None
    input_required: dict[str, Any] | None = None
    poll_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TaskStatus):
            object.__setattr__(self, "status", TaskStatus(self.status))
        if self.poll_after_seconds is not None and self.poll_after_seconds <= 0:
            raise ValueError("poll_after_seconds must be positive")
        if self.status == TaskStatus.INPUT_REQUIRED and self.input_required is None:
            raise ValueError("input_required status requires an input_required payload")

    @property
    def is_pollable(self) -> bool:
        return self.status in POLLABLE_TASK_STATUSES

    @property
    def needs_attention(self) -> bool:
        return self.status in ATTENTION_TASK_STATUSES


@dataclass(frozen=True, slots=True)
class TaskReference:
    """작업을 시작한 Agent run이 끝난 뒤에도 driver가 필요로 하는 안정적인 데이터."""

    local_task_id: str
    user_id: str
    thread_id: str
    server_name: str
    remote_task_id: str
    driver_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TaskReference:
        return cls(
            local_task_id=record["id"],
            user_id=record["user_id"],
            thread_id=record["thread_id"],
            server_name=record["server_name"],
            remote_task_id=record["remote_task_id"],
            driver_data=dict(record.get("driver_data") or {}),
        )


@dataclass(frozen=True, slots=True)
class TaskSubmitRequest:
    """MCP tool wrapper가 driver에 넘기는 protocol 중립 요청."""

    user_id: str
    thread_id: str
    run_id: str | None
    tool_call_id: str | None
    server_name: str
    task_name: str
    arguments: dict[str, Any]
    driver_data: dict[str, Any] = field(default_factory=dict)
    local_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskSubmission:
    """영속적인 remote handle과 그 초기 정규화 state."""

    remote_task_id: str
    snapshot: TaskSnapshot
    driver_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.remote_task_id.strip():
            raise ValueError("remote_task_id must not be empty")
