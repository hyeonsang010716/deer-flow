"""provider별 assistant message 필드를 복원하는 헬퍼.

일부 provider adapter는 LangChain이 원본 ``AIMessage``에는 보관하지만 request payload를
직렬화할 때 버리는 필드를 유지해야 한다. 이 모듈은 assistant message 매칭 로직을 공유하되,
어떤 필드를 복원할지는 각 provider가 결정하게 한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

AssistantPayloadRestorer = Callable[[dict[str, Any], AIMessage], None]


def restore_assistant_payloads(
    payload_messages: Sequence[dict[str, Any]],
    original_messages: Sequence[BaseMessage],
    restore: AssistantPayloadRestorer,
) -> None:
    """직렬화된 assistant payload에 provider별 필드를 복원한다."""
    if len(payload_messages) == len(original_messages):
        for payload_msg, orig_msg in zip(payload_messages, original_messages):
            if payload_msg.get("role") == "assistant" and isinstance(orig_msg, AIMessage):
                restore(payload_msg, orig_msg)
        return

    ai_messages = [m for m in original_messages if isinstance(m, AIMessage)]
    assistant_payloads = [m for m in payload_messages if m.get("role") == "assistant"]
    used_ai_indexes: set[int] = set()

    for ordinal, payload_msg in enumerate(assistant_payloads):
        ai_msg = _match_ai_message(payload_msg, ai_messages, used_ai_indexes, ordinal)
        if ai_msg is not None:
            restore(payload_msg, ai_msg)


def restore_additional_kwargs_field(payload_msg: dict[str, Any], orig_msg: AIMessage, field_name: str) -> None:
    """provider별 ``additional_kwargs`` 필드를 payload message로 복사한다."""
    value = orig_msg.additional_kwargs.get(field_name)
    if value is not None:
        payload_msg[field_name] = value


def restore_reasoning_content(payload_msg: dict[str, Any], orig_msg: AIMessage) -> None:
    """provider의 reasoning content를 직렬화된 assistant payload로 복사한다."""
    restore_additional_kwargs_field(payload_msg, orig_msg, "reasoning_content")


def _match_ai_message(
    payload_msg: dict[str, Any],
    ai_messages: Sequence[AIMessage],
    used_ai_indexes: set[int],
    fallback_ordinal: int,
) -> AIMessage | None:
    payload_key = _assistant_signature(payload_msg)
    if payload_key is not None:
        matches = [index for index, ai_msg in enumerate(ai_messages) if index not in used_ai_indexes and _ai_signature(ai_msg) == payload_key]
        if len(matches) == 1:
            used_ai_indexes.add(matches[0])
            return ai_messages[matches[0]]

    fallback_index = _next_unused_index_at_or_after(len(ai_messages), used_ai_indexes, fallback_ordinal)
    if fallback_index is not None:
        used_ai_indexes.add(fallback_index)
        return ai_messages[fallback_index]

    return None


def _next_unused_index_at_or_after(count: int, used_ai_indexes: set[int], start: int) -> int | None:
    """``start`` 이상에서 아직 쓰이지 않은 다음 AI index를 반환한다.

    payload의 서수부터 앞으로 훑으면 기존 동작의 위치 편향을 유지하면서도, 직렬화 과정에서
    메시지가 빠지거나 순서가 바뀌어 정확한 서수 index가 이미 점유된 경우에도 복구할 수 있다.
    앞쪽 index로 되돌아가지는 않는다. 그 메시지들은 이미 버려진 payload 항목에 대응할 수 있기
    때문이다.
    """
    if count == 0 or start >= count:
        return None
    for index in range(start, count):
        if index not in used_ai_indexes:
            return index
    return None


def _assistant_signature(payload_msg: dict[str, Any]) -> tuple[str, str] | None:
    return _signature(
        payload_msg.get("content"),
        _tool_call_ids(payload_msg.get("tool_calls") or []),
    )


def _ai_signature(message: AIMessage) -> tuple[str, str] | None:
    tool_calls = message.tool_calls or message.additional_kwargs.get("tool_calls") or []
    return _signature(message.content, _tool_call_ids(tool_calls))


def _signature(content: Any, tool_call_ids: tuple[str, ...]) -> tuple[str, str] | None:
    if content in (None, "") and not tool_call_ids:
        return None
    return (_stable_repr(content), "|".join(tool_call_ids))


def _stable_repr(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _tool_call_ids(tool_calls: Sequence[Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                ids.append(call_id)
    return tuple(ids)
