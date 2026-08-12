"""provider 측 safety termination 신호를 감지하는 detector들.

LLM provider마다 "안전상의 이유로 이 응답을 중단했다"를 서로 다른 필드와 값으로 알린다. 이
모듈은 작은 전략 인터페이스와, DeerFlow가 현재 지원하는 주요 provider를 커버하는 내장 detector
세 개를 정의한다. 새 provider(Wenxin, Hunyuan, Bedrock adapter, 사내 gateway 등)는
``SafetyTerminationDetector``를 구현하고 ``config.yaml: safety_finish_reason.detectors``로
연결해 추가할 수 있다.

이 detector들을 사용하는 middleware는 ``safety_finish_reason_middleware.py``에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class SafetyTermination:
    """감지된 safety 관련 termination 신호.

    Attributes:
        detector: 이 결과를 만든 detector의 이름. 운영자가 어떤 provider 규칙이 발동했는지
            볼 수 있도록 observability 용도로 쓴다.
        reason_field: 신호를 실은 메시지 metadata 필드
            (예: ``finish_reason``, ``stop_reason``).
        reason_value: 그 필드의 실제 값
            (예: ``content_filter``, ``refusal``, ``SAFETY``).
        extras: 하위 소비자에게 도움이 될 수 있는 provider별 metadata
            (예: Azure OpenAI content_filter_results, Gemini safety_ratings).
            detector는 이를 채워도 되고 생략해도 된다.
    """

    detector: str
    reason_field: str
    reason_value: str
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SafetyTerminationDetector(Protocol):
    """provider safety termination 감지를 위한 전략 인터페이스."""

    name: str

    def detect(self, message: AIMessage) -> SafetyTermination | None:
        """*message*가 provider safety termination을 나타내면 SafetyTermination을,
        아니면 ``None``을 반환한다.

        구현은 부수효과가 없어야 하고 metadata가 없거나 타입이 이상해도 견뎌야 한다.
        detector는 모든 model 응답마다 실행되기 때문이다.
        """
        ...


def _get_metadata_value(message: AIMessage, field_name: str) -> str | None:
    """``response_metadata`` 또는 ``additional_kwargs``에서 문자열 값을 읽는다.

    LangChain provider adapter는 provider stop 신호를 어디에 담는지 일관적이지 않다. 최신
    adapter 대부분은 ``response_metadata``를 쓰지만, 일부 legacy / passthrough 경로는 여전히
    ``additional_kwargs``로 노출한다. 두 곳을 그 순서로 확인하고 문자열 값만 받아들인다.
    Pydantic enum이나 dict는 무시해서 잘못된 입력에도 예외를 던지지 않는다.
    """
    for container_name in ("response_metadata", "additional_kwargs"):
        container = getattr(message, container_name, None) or {}
        if not isinstance(container, dict):
            continue
        value = container.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


class OpenAICompatibleContentFilterDetector:
    """OpenAI 호환 content_filter 신호.

    OpenAI, Azure OpenAI, Moonshot/Kimi, DeepSeek, Mistral, vLLM,
    Qwen(OpenAI 호환 모드) 및 OpenAI ``finish_reason`` 관례를 따르는 모든 adapter를 다룬다.

    일부 중국 provider는 ``sensitive``나 ``violation`` 같은 다른 토큰을 쓰는 자체 OpenAI 호환
    gateway를 제공한다. config의 ``finish_reasons`` kwarg로 집합을 확장한다.
    """

    name = "openai_compatible_content_filter"

    def __init__(self, finish_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = finish_reasons if finish_reasons is not None else ("content_filter",)
        self._finish_reasons: frozenset[str] = frozenset(r.lower() for r in configured)

    def detect(self, message: AIMessage) -> SafetyTermination | None:
        value = _get_metadata_value(message, "finish_reason")
        if value is None or value.lower() not in self._finish_reasons:
            return None

        extras: dict[str, Any] = {}
        # Azure OpenAI는 구조화된 content_filter_results 블록을 준다. 운영자가 다시 추적하지
        # 않고도 *무엇이* 필터링됐는지 볼 수 있도록 그대로 전달한다.
        response_metadata = getattr(message, "response_metadata", None) or {}
        if isinstance(response_metadata, dict):
            filter_results = response_metadata.get("content_filter_results")
            if filter_results:
                extras["content_filter_results"] = filter_results

        return SafetyTermination(
            detector=self.name,
            reason_field="finish_reason",
            reason_value=value,
            extras=extras,
        )


class AnthropicRefusalDetector:
    """Anthropic ``stop_reason == "refusal"`` 신호.

    Anthropic 모델은 safety 거부를 ``finish_reason``이 아니라 전용 ``stop_reason``으로
    노출한다. 참고:
    https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals
    """

    name = "anthropic_refusal"

    def __init__(self, stop_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = stop_reasons if stop_reasons is not None else ("refusal",)
        self._stop_reasons: frozenset[str] = frozenset(r.lower() for r in configured)

    def detect(self, message: AIMessage) -> SafetyTermination | None:
        value = _get_metadata_value(message, "stop_reason")
        if value is None or value.lower() not in self._stop_reasons:
            return None
        return SafetyTermination(
            detector=self.name,
            reason_field="stop_reason",
            reason_value=value,
        )


class GeminiSafetyDetector:
    """Gemini / Vertex AI의 safety 관련 finish reason.

    Gemini는 OpenAI와 같은 ``finish_reason`` 필드를 쓰되 대문자 열거형 체계를 사용한다. 기본
    집합은 "콘텐츠/이미지가 safety, blocklist, recitation, PII 필터에 걸려 모델이 멈췄다"를
    뜻하는 모든 Gemini finish_reason을 다룬다. 즉 함께 반환된 tool_calls가 잘렸거나 신뢰할 수
    없을 가능성이 큰 경우들이다. 전체 enum:
    https://docs.cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.types.Candidate.FinishReason

    기본 집합에서 의도적으로 **제외**한 값:
    - ``STOP``                       — 정상 종료.
    - ``MAX_TOKENS``                 — safety가 아니라 출력 길이 절단
                                       (content_filter와 근본 실패 양상은 같지만
                                       이슈 #3028의 범위 밖이다. 필요하면
                                       별도로 노출한다).
    - ``LANGUAGE`` / ``NO_IMAGE``    — safety와 무관한 능력 불일치. 어차피
                                       tool_calls가 없다.
    - ``MALFORMED_FUNCTION_CALL`` /
      ``UNEXPECTED_TOOL_CALL``       — tool-call 프로토콜 오류. 여기서도
                                       tool_calls를 신뢰할 수 없지만 실패
                                       범주가 safety 필터링과 다르다.
                                       observability 기록을 정직하게
                                       유지하려면 전용 detector에서 다룬다.
    - ``OTHER`` / ``IMAGE_OTHER`` /
      ``FINISH_REASON_UNSPECIFIED``  — 기본 활성화하기에는 너무 광범위하다.
                                       provider가 이를 남용하면
                                       ``finish_reasons=``로 opt-in한다.
    """

    name = "gemini_safety"

    _DEFAULT_FINISH_REASONS = (
        # 텍스트 safety
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "RECITATION",
        # 이미지 safety (멀티모달 생성)
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    )

    def __init__(self, finish_reasons: list[str] | tuple[str, ...] | None = None) -> None:
        configured = finish_reasons if finish_reasons is not None else self._DEFAULT_FINISH_REASONS
        self._finish_reasons: frozenset[str] = frozenset(r.upper() for r in configured)

    def detect(self, message: AIMessage) -> SafetyTermination | None:
        value = _get_metadata_value(message, "finish_reason")
        if value is None or value.upper() not in self._finish_reasons:
            return None

        extras: dict[str, Any] = {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        if isinstance(response_metadata, dict):
            # Gemini는 카테고리별 점수를 safety_ratings 아래에 노출한다.
            ratings = response_metadata.get("safety_ratings")
            if ratings:
                extras["safety_ratings"] = ratings

        return SafetyTermination(
            detector=self.name,
            reason_field="finish_reason",
            reason_value=value,
            extras=extras,
        )


def default_detectors() -> list[SafetyTerminationDetector]:
    """커스텀 detector가 설정되지 않았을 때 쓰는 내장 detector 집합."""
    return [
        OpenAICompatibleContentFilterDetector(),
        AnthropicRefusalDetector(),
        GeminiSafetyDetector(),
    ]


__all__ = [
    "AnthropicRefusalDetector",
    "GeminiSafetyDetector",
    "OpenAICompatibleContentFilterDetector",
    "SafetyTermination",
    "SafetyTerminationDetector",
    "default_detectors",
]
