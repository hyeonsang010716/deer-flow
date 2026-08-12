"""DeerFlow custom stream event 호환 헬퍼."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)

StreamWriter = Callable[[Any], None]


def _event_name(payload: dict[str, Any]) -> str | None:
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type:
        return event_type
    logger.debug("Custom stream payload has no non-empty string 'type'; skipping callback dispatch")
    return None


def emit_custom_event(payload: dict[str, Any], *, writer: StreamWriter) -> None:
    """LangGraph의 custom stream과 callback API로 이벤트를 하나 내보낸다.

    writer가 여전히 주된 호환 경로다. callback dispatch는 best-effort이므로, 선택적인
    ``astream_events`` consumer가 기존 DeerFlow run을 망가뜨릴 수 없다.
    """

    writer(payload)
    event_name = _event_name(payload)
    if event_name is None:
        return
    try:
        dispatch_custom_event(event_name, payload)
    except GraphBubbleUp:
        raise
    except Exception:
        logger.debug("Failed to dispatch custom callback event %s", event_name, exc_info=True)


async def aemit_custom_event(payload: dict[str, Any], *, writer: StreamWriter) -> None:
    """:func:`emit_custom_event`의 async 버전."""

    writer(payload)
    event_name = _event_name(payload)
    if event_name is None:
        return
    try:
        await adispatch_custom_event(event_name, payload)
    except GraphBubbleUp:
        raise
    except Exception:
        logger.debug("Failed to dispatch async custom callback event %s", event_name, exc_info=True)
