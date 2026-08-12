"""SafetyFinishReasonMiddleware 설정.

GuardrailsConfig와 같은 형태다. detector는 ``deerflow.reflection.resolve_variable``로
클래스 경로를 통해 로드한다(``guardrails.provider``가 쓰는 것과 같은 loader). 덕분에 코어
코드를 고치지 않고 custom provider detector를 끼워 넣을 수 있다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SafetyDetectorConfig(BaseModel):
    """``safety_finish_reason.detectors`` 아래의 detector 항목 하나."""

    use: str = Field(
        description=("Class path of a SafetyTerminationDetector implementation (e.g. 'deerflow.agents.middlewares.safety_termination_detectors:OpenAICompatibleContentFilterDetector')."),
    )
    config: dict = Field(
        default_factory=dict,
        description="Constructor kwargs passed to the detector class.",
    )


class SafetyFinishReasonConfig(BaseModel):
    """SafetyFinishReasonMiddleware 설정.

    provider가 안전 관련 종료를 알리면서도(예: OpenAI ``finish_reason='content_filter'``)
    도구 호출을 함께 반환한 AIMessage를 가로채, 잘려나간 인자가 실행되지 않도록 그 도구
    호출을 제거한다.
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for the SafetyFinishReasonMiddleware.",
    )
    detectors: list[SafetyDetectorConfig] | None = Field(
        default=None,
        description=(
            "Custom detector list. Leave unset (None) to use the built-in "
            "set covering OpenAI-compatible content_filter, Anthropic "
            "refusal, and Gemini SAFETY/BLOCKLIST/PROHIBITED_CONTENT/SPII/"
            "RECITATION. Provide a non-null list to fully override."
        ),
    )
