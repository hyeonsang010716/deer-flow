"""mem0 쓰기 경로용 메시지 필터링 — DeerMem의 ``filter_messages_for_memory`` 규칙을
그대로 옮긴 독립 사본이다. 이식성 규칙상 백엔드 폴더끼리 import할 수 없어 로직을
공유하지 않고 복제했다.

유지: 사용자에게 보이는 입력, 형식이 올바른 사람의 clarification 답변, 최종 assistant
응답. 제거: 프레임워크 내부의 ``hide_from_ui`` 메시지, tool call AI 메시지, tool 출력,
빈 턴과 업로드만 있는 턴.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import copy
from typing import Any

_UPLOAD_BLOCK_RE = re.compile(r"<(?P<tag>uploaded_files|current_uploads)>[\s\S]*?</(?P=tag)>\n*", re.IGNORECASE)


def extract_message_text(message: Any) -> str:
    """메시지 content(str 또는 content block 리스트)에서 평문 텍스트를 뽑는다."""
    content = getattr(message, "content", "")
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return str(content)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _is_human_clarification_response(additional_kwargs: Any) -> bool:
    """hidden 메시지에 실려 온 사용자 작성 clarification 답변인지 구조적으로 확인한다.
    DeerMem의 host 비의존 fallback과 동일한 규칙이다."""
    if not isinstance(additional_kwargs, Mapping):
        return False
    raw = additional_kwargs.get("human_input_response")
    if not isinstance(raw, Mapping):
        return False
    if raw.get("version") != 1 or raw.get("kind") != "human_input_response":
        return False
    if _non_empty_str(raw.get("source")) is None or _non_empty_str(raw.get("request_id")) is None or _non_empty_str(raw.get("value")) is None:
        return False
    response_kind = raw.get("response_kind")
    if response_kind == "text":
        return True
    if response_kind == "option":
        return _non_empty_str(raw.get("option_id")) is not None
    return False


def filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """사용자 입력과 최종 assistant 응답만 남긴다."""
    filtered: list[Any] = []
    skip_next_ai = False
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if additional_kwargs.get("hide_from_ui") and not _is_human_clarification_response(additional_kwargs):
                continue
            text = extract_message_text(msg)
            if "<uploaded_files>" in text.lower() or "<current_uploads>" in text.lower():
                stripped = _UPLOAD_BLOCK_RE.sub("", text).strip()
                if not stripped:
                    # 업로드만 있는 턴. 뒤따르는 AI 확인 응답에는 사용자 내용이 없다.
                    skip_next_ai = True
                    continue
                clean_msg = copy(msg)
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                filtered.append(msg)
                skip_next_ai = False
        elif msg_type == "ai":
            if getattr(msg, "tool_calls", None):
                continue
            if skip_next_ai:
                skip_next_ai = False
                continue
            filtered.append(msg)
    return filtered
