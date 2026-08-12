"""LangChain message 객체를 OpenAI Chat Completions 형식으로 바꾸는 순수 함수 모음.

LangChain message 타입을 OpenAI 호환 dict로 변환하는 유틸리티다. 현재 RunJournal에는 연결되어
있지 않지만(RunJournal은 message.model_dump()를 직접 쓴다), OpenAI wire format이 필요한
consumer가 쓸 수 있다.
"""

from __future__ import annotations

import json
from typing import Any

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def langchain_to_openai_message(message: Any) -> dict:
    """LangChain BaseMessage 하나를 OpenAI message dict로 변환한다.

    처리 대상:
    - HumanMessage → {"role": "user", "content": "..."}
    - AIMessage (텍스트만) → {"role": "assistant", "content": "..."}
    - AIMessage (tool_calls 포함) → {"role": "assistant", "content": null, "tool_calls": [...]}
    - AIMessage (텍스트 + tool_calls) → content와 tool_calls 모두 존재
    - AIMessage (list content / multimodal) → content를 list 그대로 보존
    - SystemMessage → {"role": "system", "content": "..."}
    - ToolMessage → {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    msg_type = getattr(message, "type", "")
    role = _ROLE_MAP.get(msg_type, msg_type)
    content = getattr(message, "content", "")

    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": getattr(message, "tool_call_id", ""),
            "content": content,
        }

    if role == "assistant":
        tool_calls = getattr(message, "tool_calls", None) or []
        result: dict = {"role": "assistant"}

        if tool_calls:
            openai_tool_calls = []
            for tc in tool_calls:
                args = tc.get("args", {})
                openai_tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(args) if not isinstance(args, str) else args,
                        },
                    }
                )
            # 텍스트 content가 없으면 OpenAI 스펙대로 content를 null로 둔다
            result["content"] = content if (isinstance(content, list) and content) or (isinstance(content, str) and content) else None
            result["tool_calls"] = openai_tool_calls
        else:
            result["content"] = content

        return result

    # user / system / 알 수 없는 타입
    return {"role": role, "content": content}


def _infer_finish_reason(message: Any) -> str:
    """AIMessage에서 OpenAI finish_reason을 추론한다.

    tool_calls가 있으면 "tool_calls"를, 없으면 response_metadata.finish_reason을 찾고,
    그것도 없으면 "stop"을 반환한다.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return "tool_calls"
    resp_meta = getattr(message, "response_metadata", None) or {}
    if isinstance(resp_meta, dict):
        finish = resp_meta.get("finish_reason")
        if finish:
            return finish
    return "stop"


def langchain_to_openai_completion(message: Any) -> dict:
    """AIMessage와 그 metadata를 OpenAI completion 응답 dict로 변환한다.

    Returns:
        {
            "id": message.id,
            "model": message.response_metadata.get("model_name"),
            "choices": [{"index": 0, "message": <openai_message>, "finish_reason": <inferred>}],
            "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...} or None,
        }
    """
    resp_meta = getattr(message, "response_metadata", None) or {}
    model_name = resp_meta.get("model_name") if isinstance(resp_meta, dict) else None

    openai_msg = langchain_to_openai_message(message)
    finish_reason = _infer_finish_reason(message)

    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata is not None:
        input_tokens = usage_metadata.get("input_tokens", 0) or 0
        output_tokens = usage_metadata.get("output_tokens", 0) or 0
        usage: dict | None = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    else:
        usage = None

    return {
        "id": getattr(message, "id", None),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": openai_msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def langchain_messages_to_openai(messages: list) -> list[dict]:
    """LangChain BaseMessage 리스트를 OpenAI message dict 리스트로 변환한다."""
    return [langchain_to_openai_message(m) for m in messages]
