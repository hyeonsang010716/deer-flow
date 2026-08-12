"""구조화된 오류 정보를 담는 sandbox 관련 예외."""


class SandboxError(Exception):
    """모든 sandbox 관련 오류의 기본 예외."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class SandboxNotFoundError(SandboxError):
    """sandbox를 찾을 수 없거나 사용할 수 없을 때 발생한다."""

    def __init__(self, message: str = "Sandbox not found", sandbox_id: str | None = None):
        details = {"sandbox_id": sandbox_id} if sandbox_id else None
        super().__init__(message, details)
        self.sandbox_id = sandbox_id


class SandboxRuntimeError(SandboxError):
    """sandbox runtime을 사용할 수 없거나 설정이 잘못되었을 때 발생한다."""

    pass


class SandboxCommandError(SandboxError):
    """sandbox에서 명령 실행이 실패했을 때 발생한다."""

    def __init__(self, message: str, command: str | None = None, exit_code: int | None = None):
        details = {}
        if command:
            details["command"] = command[:100] + "..." if len(command) > 100 else command
        if exit_code is not None:
            details["exit_code"] = exit_code
        super().__init__(message, details)
        self.command = command
        self.exit_code = exit_code


class SandboxFileError(SandboxError):
    """sandbox에서 파일 작업이 실패했을 때 발생한다."""

    def __init__(self, message: str, path: str | None = None, operation: str | None = None):
        details = {}
        if path:
            details["path"] = path
        if operation:
            details["operation"] = operation
        super().__init__(message, details)
        self.path = path
        self.operation = operation


class SandboxPermissionError(SandboxFileError):
    """파일 작업 중 권한 오류가 발생했을 때 발생한다."""

    pass


class SandboxFileNotFoundError(SandboxFileError):
    """파일이나 디렉터리를 찾을 수 없을 때 발생한다."""

    pass


class SandboxCapacityExceededError(SandboxError):
    """sandbox provider에 남은 용량이 없을 때 발생한다.

    reason은 용량이 찬 경우와 provider 종료를 구분한다. retry 일정은 호출자가 정하며,
    DeerFlow는 자동으로 재시도하지 않는다.
    """

    CODE = "SANDBOX_CAPACITY_EXCEEDED"

    def __init__(
        self,
        message: str = "All sandbox replica slots are in use",
        *,
        active: int = 0,
        warm: int = 0,
        reserved: int = 0,
        replicas: int = 0,
        retry_after_seconds: float = 5.0,
        reason: str = "capacity",
    ) -> None:
        details: dict[str, object] = {
            "code": self.CODE,
            "reason": reason,
            "replicas": replicas,
            "retryable": True,
            "retry_after_seconds": retry_after_seconds,
        }
        if active:
            details["active"] = active
        if warm:
            details["warm"] = warm
        if reserved:
            details["reserved"] = reserved
        super().__init__(message, details)
        self.active = active
        self.warm = warm
        self.reserved = reserved
        self.replicas = replicas
        self.retry_after_seconds = retry_after_seconds
        self.reason = reason
