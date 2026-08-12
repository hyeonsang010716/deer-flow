"""``DeerFlowClient`` streaming과 view-state reducer를 잇는 runtime bridge.

두 계층 모두 Textual에 의존하지 않는다:

* :func:`translate` — 순수 함수. ``StreamEvent`` 하나를 0개 이상의 reducer action으로 바꾼다.
* :func:`stream_actions` — ``client.stream()``을 구동하며 앞뒤가 감싸인 action 시퀀스
  (``RunStarted`` … 변환된 action들 … ``RunEnded``)를 내보낸다. model 에러는 crash 대신
  ``AssistantError`` row로 바꾼다.

Textual app은 :func:`stream_actions`를 worker thread에서 실행하고, 내보내진 각 action을 UI
thread에서 reducer에 적용한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from .view_state import (
    Action,
    AssistantDelta,
    AssistantError,
    RunEnded,
    RunStarted,
    ThreadTitle,
    ToolResult,
    ToolStarted,
)


class _StreamEventLike(Protocol):
    type: str
    data: dict


class _ClientLike(Protocol):
    def stream(self, message: str, *, thread_id: str | None = None, **kwargs: Any) -> Iterator[Any]:
        """*message*에 대한 streaming event를 내보낸다(``DeerFlowClient.stream`` 참고)."""


def translate(event: _StreamEventLike) -> list[Action]:
    """``StreamEvent`` 하나를 reducer action으로 변환한다. 순수 함수다."""
    if event.type == "messages-tuple":
        return _translate_message(event.data)
    if event.type == "end":
        usage = event.data.get("usage") if isinstance(event.data, dict) else None
        return [RunEnded(usage=usage)]
    if event.type == "values" and isinstance(event.data, dict):
        title = event.data.get("title")
        if isinstance(title, str) and title.strip():
            return [ThreadTitle(title=title.strip())]
        return []
    # "custom" event는 점진적으로 렌더링하지 않는다.
    return []


def _translate_message(data: Any) -> list[Action]:
    if not isinstance(data, dict):
        return []

    message_type = data.get("type")
    actions: list[Action] = []

    if message_type == "ai":
        text = _extract_text(data.get("content"))
        if text:
            actions.append(AssistantDelta(id=_as_str(data.get("id")), text=text))
        for tool_call in data.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            actions.append(
                ToolStarted(
                    tool_call_id=_as_str(tool_call.get("id")),
                    tool_name=_as_str(tool_call.get("name")),
                    args=tool_call.get("args") or {},
                )
            )
    elif message_type == "tool":
        is_error = bool(data.get("is_error")) or data.get("status") == "error"
        actions.append(
            ToolResult(
                tool_call_id=_as_str(data.get("tool_call_id")),
                content=_extract_text(data.get("content")),
                is_error=is_error,
                tool_name=_as_str(data.get("name")),
            )
        )

    return actions


def _as_str(value: Any) -> str:
    # provider stream chunk는 id/name을 명시적으로 ``None``으로 담을 수 있다(키 자체는 있으므로
    # ``.get(k, "")``가 None을 반환하고, ``str(None) == "None"``이라는 truthy 값이 되어 하류의
    # 빈 id 가드를 무력화한다).
    return "" if value is None else str(value)


def stream_actions(client: _ClientLike, message: str, *, thread_id: str | None = None, **kwargs: Any) -> Iterator[Action]:
    """agent run 하나에 대해 앞뒤가 감싸인 action stream을 내보낸다.

    항상 ``RunStarted``로 시작해 ``RunEnded``로 끝난다. 에러가 나도 마찬가지이며, 그 경우
    ``AssistantError`` row가 먼저 나온다.
    """
    yield RunStarted()
    try:
        for event in client.stream(message, thread_id=thread_id, **kwargs):
            yield from translate(event)
            if event.type == "end":
                return  # RunEnded는 translate()가 이미 내보냈다
        yield RunEnded()
    except Exception as exc:  # noqa: BLE001 - model/runtime 에러를 UI에 그대로 드러낸다
        yield AssistantError(str(exc) or exc.__class__.__name__)
        yield RunEnded()


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)
