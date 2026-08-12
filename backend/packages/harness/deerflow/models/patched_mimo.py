"""Xiaomi MiMo의 reasoning_content 재전송을 위한 ChatOpenAI 패치 adapter.

MiMo의 OpenAI 호환 API는 thinking mode에서 ``reasoning_content``를 반환하며, 멀티턴 agent
대화에서는 그 값을 과거 assistant 메시지에 다시 실어 보내야 한다. 기본
``langchain_openai.ChatOpenAI``는 이 provider 전용 필드를 버리므로, tool call이 대화 히스토리에
들어오면 HTTP 400 오류가 날 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from deerflow.models.assistant_payload_replay import restore_assistant_payloads, restore_reasoning_content

_MISSING = object()


def _extract_reasoning_content(value: Any) -> str | object:
    """dict나 Pydantic 객체에서 reasoning_content를 꺼낸다. 빈 문자열도 보존한다."""
    if isinstance(value, Mapping):
        if "reasoning_content" in value and value["reasoning_content"] is not None:
            return value["reasoning_content"]
        return _MISSING

    reasoning = getattr(value, "reasoning_content", _MISSING)
    if reasoning is not _MISSING and reasoning is not None:
        return reasoning

    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping) and "reasoning_content" in model_extra and model_extra["reasoning_content"] is not None:
        return model_extra["reasoning_content"]

    return _MISSING


def _with_reasoning_content(message: AIMessage | AIMessageChunk, reasoning: str) -> AIMessage | AIMessageChunk:
    additional_kwargs = dict(message.additional_kwargs)
    if additional_kwargs.get("reasoning_content") != reasoning:
        additional_kwargs["reasoning_content"] = reasoning
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def _get_typed_choice_message(response: Any, index: int) -> Any:
    choices = getattr(response, "choices", None)
    if choices is None:
        return None
    try:
        return choices[index].message
    except (AttributeError, IndexError, TypeError):
        return None


class PatchedChatMiMo(ChatOpenAI):
    """MiMo thinking mode를 위해 ``reasoning_content``를 보존하는 ChatOpenAI."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "MIMO_API_KEY", "openai_api_key": "MIMO_API_KEY"}

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        restore_assistant_payloads(
            payload.get("messages", []),
            original_messages,
            restore_reasoning_content,
        )

        return payload

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
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
            reasoning = _extract_reasoning_content(delta)
            if reasoning is not _MISSING and isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk = ChatGenerationChunk(
                    message=_with_reasoning_content(generation_chunk.message, reasoning),
                    generation_info=generation_chunk.generation_info,
                )

        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices", [])

        patched_generations: list[ChatGeneration] | None = None
        for index, generation in enumerate(result.generations):
            choice = choices[index] if index < len(choices) else {}
            choice_message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
            reasoning = _extract_reasoning_content(choice_message)
            if reasoning is _MISSING and not isinstance(response, dict):
                reasoning = _extract_reasoning_content(_get_typed_choice_message(response, index))

            message = generation.message
            if reasoning is not _MISSING and isinstance(message, AIMessage):
                if patched_generations is None:
                    patched_generations = list(result.generations)
                patched_generations[index] = ChatGeneration(
                    message=_with_reasoning_content(message, reasoning),
                    generation_info=generation.generation_info,
                )

        return ChatResult(generations=patched_generations or result.generations, llm_output=result.llm_output)
