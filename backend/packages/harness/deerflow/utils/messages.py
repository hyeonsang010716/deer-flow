from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage

ORIGINAL_USER_CONTENT_KEY = "original_user_content"
SUMMARY_MESSAGE_NAME = "summary"


def message_content_to_text(content: Any) -> str:
    """LangChain message content 형태들에서 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def message_to_text(message: Any, *, text_attribute_fallback: bool = False) -> str:
    """메시지 전체(``BaseMessage`` 또는 dict 형태)에서 표시용 텍스트를 추출한다.

    ``content``를 속성(``BaseMessage``) 또는 mapping 키(``run_events`` row는 dict)에서
    읽은 뒤, 섞여 있는 ``content`` 형태를 순회한다: 평범한 문자열, 구분자 없이 이어붙이는
    문자열 / ``{"text": ...}`` / 중첩 ``{"content": ...}`` 블록 list, 또는 ``text``/``content``
    키를 가진 mapping. ``text_attribute_fallback=True``이면 content에서 아무것도 나오지
    않을 때 ``message.text``로 fallback한다(``RunJournal._message_text``와 동일).

    raw ``content``를 받아 list 블록을 개행으로 잇는 :func:`message_content_to_text`와 달리,
    이 함수는 구분자 없는 결합과 여러 호출 지점이 각각 재구현하던 더 넓은 형태 처리를 유지한다.
    """
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    nested = block.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
        return "".join(parts)
    if isinstance(content, Mapping):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    if text_attribute_fallback:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
    return ""


def get_original_user_content_text(content: Any, additional_kwargs: Mapping[str, Any] | None) -> str:
    """middleware 이전의 사용자 텍스트가 있으면 그것을, 없으면 content 텍스트를 반환한다."""
    original_content = (additional_kwargs or {}).get(ORIGINAL_USER_CONTENT_KEY)
    if isinstance(original_content, str):
        return original_content
    return message_content_to_text(content)


def restore_original_human_message(message: HumanMessage) -> HumanMessage:
    """모델용으로 sanitize된 human message의 UI 노출용 사본을 만든다.

    input middleware는 의도적으로 원본 사용자 텍스트를 ``additional_kwargs``에 남겨두고,
    모델에 보내는 텍스트만 transport wrapper와 추가 컨텍스트로 교체한다. run-event 히스토리는
    실제로 모델에 전달되는 메시지를 변경하지 않으면서 원본 텍스트를 남겨야 한다.

    섞인 content는 sanitization middleware가 이미 단일 텍스트 블록으로 정규화한다. 방어적
    호환을 위해, 현재 텍스트 블록이 여러 개면 첫 번째 텍스트 위치로 합치고 텍스트가 아닌
    블록은 값과 상대 순서를 그대로 유지한다.
    """
    original_content = message.additional_kwargs.get(ORIGINAL_USER_CONTENT_KEY)
    if not isinstance(original_content, str):
        return message

    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs.pop(ORIGINAL_USER_CONTENT_KEY, None)

    content = message.content
    if isinstance(content, str):
        restored_content: str | list = original_content
    elif isinstance(content, list):
        restored_content = []
        restored_text = False
        for block in content:
            is_string_text = isinstance(block, str)
            is_mapping_text = isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
            if not is_string_text and not is_mapping_text:
                restored_content.append(block)
                continue
            if restored_text:
                continue
            if is_mapping_text:
                restored_content.append({**block, "text": original_content})
            else:
                restored_content.append(original_content)
            restored_text = True
        if not restored_text:
            restored_content.insert(0, {"type": "text", "text": original_content})
    else:
        restored_content = original_content

    return message.model_copy(
        update={
            # Pydantic은 ``deep=True``일 때 원본 모델을 deep copy하지만, ``update``로 넘긴
            # 값은 복사하지 않고 그대로 적용한다. 중첩된 image/file 블록과 metadata까지
            # 포함해서, 저장/UI용 사본을 모델용 메시지와 완전히 분리한다.
            "content": deepcopy(restored_content),
            "additional_kwargs": deepcopy(additional_kwargs),
        },
        deep=True,
    )


def is_real_user_message(message: object) -> bool:
    """``message``가 실제 사용자가 작성한 HumanMessage인지 반환한다.

    middleware가 주입한 숨김 HumanMessage와 summarization 마커는 slash-skill 활성화나
    MCP routing 같은 사용자 의도 기반 기능을 촉발해서는 안 된다.
    """
    if not isinstance(message, HumanMessage):
        return False
    if message.name == SUMMARY_MESSAGE_NAME:
        return False
    if message.additional_kwargs.get("hide_from_ui"):
        return False
    return True
