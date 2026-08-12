"""summarization이 state에서 메시지를 제거하기 전에 발화하는 hook 모음."""

from __future__ import annotations

from deerflow.agents.memory import get_memory_manager
from deerflow.agents.middlewares.summarization_middleware import SummarizationEvent
from deerflow.config.memory_config import get_memory_config
from deerflow.runtime.user_context import resolve_runtime_user_id


def memory_flush_hook(event: SummarizationEvent) -> None:
    """요약 직전의 메시지를 memory 큐로 flush한다.

    backend 중립적인 얇은 진입점이다. 여기에는 ``enabled`` + ``thread_id`` 게이트와
    ``user_id`` 해석만 둔다. 필터링, human/AI 검증, correction/reinforcement 탐지는
    backend가(``manager.add_nowait``를 통해) 담당한다.
    """
    if not get_memory_config().enabled or not event.thread_id:
        return

    user_id = resolve_runtime_user_id(event.runtime)
    get_memory_manager().add_nowait(
        event.thread_id,
        list(event.messages_to_summarize),
        agent_name=event.agent_name,
        user_id=user_id,
    )
