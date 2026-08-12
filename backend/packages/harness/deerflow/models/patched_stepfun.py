"""StepFun reasoning 모델용으로 패치한 ChatOpenAI adapter.

StepFun은 streaming delta와 비streaming 응답 양쪽에서 ``reasoning``(deepseek 방식이면
``reasoning_content``)을 반환한다. 표준 ``ChatOpenAI``는 이 비표준 필드를 무시하므로
reasoning 내용이 조용히 사라진다. 이 adapter는 모든 응답 경로에서 reasoning을 잡아내고,
멀티턴 tool call 대화에서 과거 assistant 메시지에 다시 실어 보낸다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from deerflow.models.assistant_payload_replay import (
    restore_assistant_payloads,
    restore_reasoning_content,
)

_MISSING = object()


def _extract_reasoning(value: Any) -> str | object:
    """dict 또는 Pydantic 객체에서 reasoning 내용을 반환한다.

    StepFun은 reasoning을 ``reasoning``(기본) 또는 ``reasoning_content``(deepseek 방식)로
    반환할 수 있다. 두 필드를 모두 확인한다.
    """
    if isinstance(value, Mapping):
        # reasoning_content(deepseek 방식)를 먼저 보고, 그다음 reasoning(기본)을 본다
        for field in ("reasoning_content", "reasoning"):
            if field in value and value[field] is not None:
                return value[field]
        return _MISSING

    # Pydantic / SDK 객체 속성
    for field in ("reasoning_content", "reasoning"):
        attr = getattr(value, field, _MISSING)
        if attr is not _MISSING and attr is not None:
            return attr

    # 일부 SDK 버전은 추가 필드를 model_extra에 저장한다
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        for field in ("reasoning_content", "reasoning"):
            if field in model_extra and model_extra[field] is not None:
                return model_extra[field]

    return _MISSING


def _with_reasoning_content(message: AIMessage | AIMessageChunk, reasoning: str) -> AIMessage | AIMessageChunk:
    """additional_kwargs에 reasoning_content를 담은 *message*의 사본을 반환한다."""
    additional_kwargs = dict(message.additional_kwargs)
    if additional_kwargs.get("reasoning_content") != reasoning:
        additional_kwargs["reasoning_content"] = reasoning
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def _get_typed_choice_message(response: Any, index: int) -> Any:
    """가능하면 *index* 위치의 SDK 타입 choice 메시지를 추출한다."""
    choices = getattr(response, "choices", None)
    if choices is None:
        return None
    try:
        return choices[index].message
    except (AttributeError, IndexError, TypeError):
        return None


class PatchedChatStepFun(ChatOpenAI):
    """StepFun 모델의 reasoning을 완전히 지원하는 ChatOpenAI.

    streaming과 비streaming 응답 양쪽에서 ``reasoning`` / ``reasoning_content``를 잡아내고,
    멀티턴 tool call 대화에서 과거 assistant 메시지에 다시 실어 보낸다.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "STEPFUN_API_KEY", "openai_api_key": "STEPFUN_API_KEY"}

    # --- request payload 재전달 ---

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """과거 assistant 메시지에 ``reasoning_content``를 복원한다."""
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        restore_assistant_payloads(
            payload.get("messages", []),
            original_messages,
            restore_reasoning_content,
        )

        return payload

    # --- streaming reasoning 수집 ---

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """streaming delta에서 ``reasoning`` / ``reasoning_content``를 수집한다."""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning = _extract_reasoning(delta)
            if reasoning is not _MISSING and isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk = ChatGenerationChunk(
                    message=_with_reasoning_content(generation_chunk.message, reasoning),
                    generation_info=generation_chunk.generation_info,
                )

        return generation_chunk

    # --- 비streaming reasoning 수집 ---

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """비streaming 응답에서 ``reasoning`` / ``reasoning_content``를 추출한다."""
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices", [])

        patched_generations: list[ChatGeneration] | None = None
        for index, generation in enumerate(result.generations):
            choice = choices[index] if index < len(choices) else {}
            choice_message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
            reasoning = _extract_reasoning(choice_message)

            if reasoning is _MISSING and not isinstance(response, dict):
                reasoning = _extract_reasoning(_get_typed_choice_message(response, index))

            message = generation.message
            if reasoning is not _MISSING and isinstance(message, AIMessage):
                if patched_generations is None:
                    patched_generations = list(result.generations)
                patched_generations[index] = ChatGeneration(
                    message=_with_reasoning_content(message, reasoning),
                    generation_info=generation.generation_info,
                )

        return ChatResult(
            generations=patched_generations or result.generations,
            llm_output=result.llm_output,
        )
