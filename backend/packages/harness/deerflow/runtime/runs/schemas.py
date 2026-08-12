"""run 상태와 disconnect mode enum."""

from enum import StrEnum


class ThreadOperationKind(StrEnum):
    """thread에 대한 배타적 admission을 점유하는 operation의 종류."""

    run = "run"
    checkpoint_write = "checkpoint_write"
    artifact_write = "artifact_write"


class RunStatus(StrEnum):
    """단일 run의 lifecycle 상태."""

    pending = "pending"
    running = "running"
    success = "success"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


class DisconnectMode(StrEnum):
    """SSE consumer가 연결을 끊었을 때의 동작."""

    cancel = "cancel"
    continue_ = "continue"
