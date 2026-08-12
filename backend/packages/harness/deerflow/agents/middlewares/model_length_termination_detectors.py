"""provider가 보내는 모델 길이 종료 신호를 감지하는 detector.

provider마다 "응답이 output token 한도에 걸렸다"를 서로 다른 필드와 값으로 알린다. 그런
provider별 세부사항은 여기에 모아 두어, ``ModelLengthFinishReasonMiddleware``는 어느 provider가
어떤 표기를 쓰는지가 아니라 언제 run을 표시할지에만 집중하게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class ModelLengthTermination:
    """감지된 모델 출력 길이 상한."""

    detector: str
    reason_field: str
    reason_value: str
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelLengthTerminationDetector(Protocol):
    """provider 길이 상한 감지를 위한 전략 인터페이스."""

    name: str

    def detect(self, message: AIMessage) -> ModelLengthTermination | None:
        """*message*가 출력 길이 절단을 나타내면 감지 결과를 반환한다."""
        ...


def _get_metadata_value(message: AIMessage, field_name: str) -> str | None:
    """흔한 LangChain provider 필드에서 문자열 metadata 값을 읽는다."""
    for container_name in ("response_metadata", "additional_kwargs"):
        container = getattr(message, container_name, None) or {}
        if not isinstance(container, dict):
            continue
        value = container.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


class OpenAICompatibleLengthDetector:
    """OpenAI 호환 ``finish_reason == "length"`` 신호."""

    name = "openai_compatible_length"

    def __init__(self, finish_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = finish_reasons if finish_reasons is not None else ("length",)
        self._finish_reasons: frozenset[str] = frozenset(r.lower() for r in configured)

    def detect(self, message: AIMessage) -> ModelLengthTermination | None:
        value = _get_metadata_value(message, "finish_reason")
        if value is None or value.lower() not in self._finish_reasons:
            return None
        return ModelLengthTermination(
            detector=self.name,
            reason_field="finish_reason",
            reason_value=value,
        )


class AnthropicMaxTokensDetector:
    """Anthropic ``stop_reason == "max_tokens"`` 신호."""

    name = "anthropic_max_tokens"

    def __init__(self, stop_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = stop_reasons if stop_reasons is not None else ("max_tokens",)
        self._stop_reasons: frozenset[str] = frozenset(r.lower() for r in configured)

    def detect(self, message: AIMessage) -> ModelLengthTermination | None:
        value = _get_metadata_value(message, "stop_reason")
        if value is None or value.lower() not in self._stop_reasons:
            return None
        return ModelLengthTermination(
            detector=self.name,
            reason_field="stop_reason",
            reason_value=value,
        )


class GeminiMaxTokensDetector:
    """Gemini / Vertex AI ``finish_reason == "MAX_TOKENS"`` 신호."""

    name = "gemini_max_tokens"

    def __init__(self, finish_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = finish_reasons if finish_reasons is not None else ("MAX_TOKENS",)
        self._finish_reasons: frozenset[str] = frozenset(r.upper() for r in configured)

    def detect(self, message: AIMessage) -> ModelLengthTermination | None:
        value = _get_metadata_value(message, "finish_reason")
        if value is None or value.upper() not in self._finish_reasons:
            return None
        return ModelLengthTermination(
            detector=self.name,
            reason_field="finish_reason",
            reason_value=value,
        )


def default_detectors() -> list[ModelLengthTerminationDetector]:
    """provider 길이 상한 신호에 쓰이는 기본 detector 집합."""
    return [
        OpenAICompatibleLengthDetector(),
        AnthropicMaxTokensDetector(),
        GeminiMaxTokensDetector(),
    ]


__all__ = [
    "AnthropicMaxTokensDetector",
    "GeminiMaxTokensDetector",
    "ModelLengthTermination",
    "ModelLengthTerminationDetector",
    "OpenAICompatibleLengthDetector",
    "default_detectors",
]
