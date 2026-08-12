import os
import threading

from pydantic import BaseModel, Field

_config_lock = threading.Lock()


class LangSmithTracingConfig(BaseModel):
    """LangSmith tracing 설정."""

    enabled: bool = Field(...)
    api_key: str | None = Field(...)
    project: str = Field(...)
    endpoint: str = Field(...)

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.api_key)

    def validate(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("LangSmith tracing is enabled but LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is not set.")


class LangfuseTracingConfig(BaseModel):
    """Langfuse tracing 설정."""

    enabled: bool = Field(...)
    public_key: str | None = Field(...)
    secret_key: str | None = Field(...)
    host: str = Field(...)

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.public_key) and bool(self.secret_key)

    def validate(self) -> None:
        if not self.enabled:
            return
        missing: list[str] = []
        if not self.public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not self.secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        if missing:
            raise ValueError(f"Langfuse tracing is enabled but required settings are missing: {', '.join(missing)}")


# monocle_apptrace가 지원하는 exporter 목록을 수동으로 복제한 것. 로컬에 두어서 오타가 나면
# 알 수 없는 upstream 에러 대신 시작 시점에 명확한 메시지로 실패하게 한다.
# monocle_apptrace 버전이 올라가며 exporter가 추가되거나 이름이 바뀌면 이 tuple을 갱신한다.
_MONOCLE_EXPORTERS = ("file", "console", "okahu", "s3", "blob", "gcs")


class MonocleTracingConfig(BaseModel):
    """Monocle telemetry 설정."""

    enabled: bool = Field(...)
    exporters: str = Field(...)
    okahu_api_key: str | None = Field(...)

    @property
    def is_enabled(self) -> bool:
        # 형제 클래스들의 is_configured와 달리 여기서는 credential을 확인하지 않는다.
        # 그건 exporter에 따라 다르고, Gateway 시작 시 실행되는 validate()가 담당한다.
        return self.enabled

    @property
    def exporter_list(self) -> list[str]:
        """설정된 exporter 목록. 한 번만 파싱해서 검증과 설정이 어긋나지 않게 한다."""
        return [e.strip() for e in self.exporters.split(",") if e.strip()]

    def validate(self) -> None:
        if not self.enabled:
            return
        selected = self.exporter_list
        unknown = [e for e in selected if e not in _MONOCLE_EXPORTERS]
        if unknown:
            raise ValueError(f"MONOCLE_EXPORTERS has unknown exporter(s): {', '.join(unknown)}. Allowed: {', '.join(_MONOCLE_EXPORTERS)}.")
        if "okahu" in selected and not self.okahu_api_key:
            raise ValueError("Monocle 'okahu' exporter is selected but OKAHU_API_KEY is not set.")


class TracingConfig(BaseModel):
    """지원하는 provider들의 tracing 설정."""

    langsmith: LangSmithTracingConfig = Field(...)
    langfuse: LangfuseTracingConfig = Field(...)
    monocle: MonocleTracingConfig = Field(...)

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled_providers)

    @property
    def explicitly_enabled_providers(self) -> list[str]:
        enabled: list[str] = []
        if self.langsmith.enabled:
            enabled.append("langsmith")
        if self.langfuse.enabled:
            enabled.append("langfuse")
        return enabled

    @property
    def enabled_providers(self) -> list[str]:
        enabled: list[str] = []
        if self.langsmith.is_configured:
            enabled.append("langsmith")
        if self.langfuse.is_configured:
            enabled.append("langfuse")
        return enabled

    def validate_enabled(self) -> None:
        self.langsmith.validate()
        self.langfuse.validate()


_tracing_config: TracingConfig | None = None


_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _env_flag_preferred(*names: str) -> bool:
    """존재하고 비어 있지 않은 첫 환경 변수의 boolean 값을 반환한다."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip().lower() in _TRUTHY_VALUES
    return False


def _first_env_value(*names: str) -> str | None:
    """후보 이름들 중 비어 있지 않은 첫 환경 변수 값을 반환한다."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def get_tracing_config() -> TracingConfig:
    """환경 변수로부터 현재 tracing 설정을 얻는다."""
    global _tracing_config
    if _tracing_config is not None:
        return _tracing_config
    with _config_lock:
        if _tracing_config is not None:
            return _tracing_config
        _tracing_config = TracingConfig(
            langsmith=LangSmithTracingConfig(
                enabled=_env_flag_preferred("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"),
                api_key=_first_env_value("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
                project=_first_env_value("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT") or "deer-flow",
                endpoint=_first_env_value("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
            ),
            langfuse=LangfuseTracingConfig(
                enabled=_env_flag_preferred("LANGFUSE_TRACING"),
                public_key=_first_env_value("LANGFUSE_PUBLIC_KEY"),
                secret_key=_first_env_value("LANGFUSE_SECRET_KEY"),
                host=_first_env_value("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com",
            ),
            monocle=MonocleTracingConfig(
                enabled=_env_flag_preferred("MONOCLE_TRACING"),
                exporters=_first_env_value("MONOCLE_EXPORTERS") or "file",
                okahu_api_key=_first_env_value("OKAHU_API_KEY"),
            ),
        )
        return _tracing_config


def get_enabled_tracing_providers() -> list[str]:
    """활성화되어 있고 설정이 완전한 tracing provider를 반환한다."""
    return get_tracing_config().enabled_providers


def get_explicitly_enabled_tracing_providers() -> list[str]:
    """설정이 불완전하더라도 config에서 명시적으로 켜진 tracing provider를 반환한다."""
    return get_tracing_config().explicitly_enabled_providers


def validate_enabled_tracing_providers() -> None:
    """명시적으로 켜진 provider가 완전히 설정되었는지 검증한다."""
    get_tracing_config().validate_enabled()


def is_tracing_enabled() -> bool:
    """tracing provider 중 켜져 있고 완전히 설정된 것이 있는지 확인한다."""
    return get_tracing_config().is_configured


def is_monocle_tracing_enabled() -> bool:
    """Monocle OTel observability가 켜져 있는지 여부(``MONOCLE_TRACING``로 제어).

    Monocle은 run 단위 LangChain callback이 아니라 시작 시 활성화되는 프로세스 전역
    instrumentor이므로 :func:`get_enabled_tracing_providers`와 분리해 둔다.
    """
    return get_tracing_config().monocle.is_enabled


def reset_tracing_config() -> None:
    """캐시된 :class:`TracingConfig`를 버려서 다음 호출이 다시 만들게 한다.

    테스트가 private 모듈 속성 ``_tracing_config``에 직접 손대지 않도록 공개 API로 둔다.
    나중에 내부 이름이 바뀌면 속성을 직접 변경하는 호출자는 조용히 깨진다.
    """
    global _tracing_config
    with _config_lock:
        _tracing_config = None
