"""LangChain / LangGraph 객체의 표준 직렬화.

LangChain 메시지 객체, Pydantic 모델, LangGraph state dict를 순수 JSON 직렬화 가능한 Python
구조로 변환하는 단일 기준점을 제공한다.

소비처: ``deerflow.runtime.runs.worker``(SSE 발행)와 ``app.gateway.routers.threads``(REST 응답).
"""

from __future__ import annotations

from typing import Any


def serialize_lc_object(obj: Any) -> Any:
    """LangChain 객체를 JSON 직렬화 가능한 dict로 재귀 변환한다."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_lc_object(item) for item in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / 그 이전 객체
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Interrupt는 __slots__ 클래스라 model_dump/dict/__dict__가 없다. 그대로 두면 str()까지
    # 내려가 잘못된 payload가 만들어진다.
    try:
        from langgraph.types import Interrupt
    except ImportError:
        pass
    else:
        if isinstance(obj, Interrupt):
            return serialize_lc_object(
                {
                    "value": obj.value,
                    "id": getattr(obj, "id", None),
                }
            )
    # 최후의 수단
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 내부 키를 제거하며 channel value를 직렬화한다.

    ``__pregel_*`` 키만 제거한다. ``__interrupt__``는 LangGraph SDK가 values chunk에서 interrupt
    이벤트를 감지할 수 있도록 의도적으로 남긴다(issue #3595 참고).
    """
    result: dict[str, Any] = {}
    for key, value in channel_values.items():
        if key.startswith("__pregel_"):
            continue
        result[key] = serialize_lc_object(value)
    return result


def strip_data_url_image_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*hide_from_ui* 메시지에서 ``data:`` 스킴 ``image_url`` block을 제거한다.

    history와 run-wait endpoint는 checkpoint에 저장된 메시지를 frontend에 반환한다.
    ``ViewImageMiddleware``는 전체 base64 이미지 payload를 ``hide_from_ui`` human 메시지에
    저장하는데, 이는 내부 모델 context이므로 네트워크로 내보내면 안 된다(응답이 거대해지고 UI
    상으로는 가치가 없다).

    URL이 ``data:``로 시작하는 ``image_url`` 타입 content block만 제거한다. 텍스트 block,
    ``https://`` 이미지 URL, 숨김이 아닌 메시지는 건드리지 않아 메시지 순서와 개수가 유지된다.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue

        # UI에서 숨김으로 명시된 메시지만 건드린다.
        additional_kwargs = msg.get("additional_kwargs")
        if not (isinstance(additional_kwargs, dict) and additional_kwargs.get("hide_from_ui") is True):
            result.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        # data: 스킴을 가진 image_url block을 걸러낸다.
        filtered = [block for block in content if not (isinstance(block, dict) and block.get("type") == "image_url" and isinstance(block.get("image_url"), dict) and str(block["image_url"].get("url", "")).startswith("data:"))]
        result.append({**msg, "content": filtered})
    return result


def serialize_channel_values_for_api(channel_values: dict[str, Any]) -> dict[str, Any]:
    """channel value를 직렬화하고 메시지에서 base64 이미지 데이터를 제거한다.

    :func:`serialize_channel_values`와 :func:`strip_data_url_image_blocks`를 묶은 편의 wrapper다.
    ``data:`` 스킴 base64 이미지 payload가 절대 네트워크로 나가지 않도록, channel value를
    frontend에 반환하는 모든 REST endpoint에서 이 함수를 쓴다.
    """
    result = serialize_channel_values(channel_values)
    if isinstance(result.get("messages"), list):
        result["messages"] = strip_data_url_image_blocks(result["messages"])
    return result


def serialize_messages_tuple(obj: Any) -> Any:
    """messages 모드의 tuple ``(chunk, metadata)``를 직렬화한다."""
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj
        return [serialize_lc_object(chunk), metadata if isinstance(metadata, dict) else {}]
    return serialize_lc_object(obj)


def serialize(obj: Any, *, mode: str = "") -> Any:
    """mode별 처리를 적용해 LangChain 객체를 직렬화한다.

    * ``messages`` — obj는 ``(message_chunk, metadata_dict)``
    * ``values`` — obj는 전체 state dict. ``__pregel_*`` 키를 제거하고 hide_from_ui 메시지에서
      base64 ``data:`` 이미지 block을 걸러낸다
    * 그 외 — 재귀적인 ``model_dump()`` / ``dict()`` fallback
    """
    if mode == "messages":
        return serialize_messages_tuple(obj)
    if mode == "values":
        # ``values`` snapshot은 전체 state를 frontend로 streaming하므로, REST endpoint와 동일하게
        # base64 이미지 payload를 제거해야 한다.
        return serialize_channel_values_for_api(obj) if isinstance(obj, dict) else serialize_lc_object(obj)
    return serialize_lc_object(obj)
