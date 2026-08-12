"""memory 메커니즘을 담당하는 middleware."""

import asyncio
import logging
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.memory import get_memory_manager
from deerflow.config.memory_config import get_memory_config
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY, get_current_trace_id, normalize_trace_id

if TYPE_CHECKING:
    from deerflow.config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryMiddlewareState(AgentState):
    """`ThreadState` 스키마와 호환된다."""

    pass


class MemoryMiddleware(AgentMiddleware[MemoryMiddlewareState]):
    """에이전트 실행이 끝난 뒤 대화를 memory 갱신 큐에 넣는 middleware.

    동작은 다음과 같다.
    1. 에이전트 실행마다 대화를 memory 갱신 큐에 넣는다.
    2. 사용자 입력과 최종 assistant 응답만 포함한다(tool call은 무시한다).
    3. 큐는 debounce로 여러 갱신을 묶는다.
    4. memory는 LLM 요약을 통해 비동기로 갱신된다.
    """

    state_schema = MemoryMiddlewareState

    def __init__(self, agent_name: str | None = None, *, memory_config: "MemoryConfig | None" = None):
        """MemoryMiddleware를 초기화한다.

        Args:
            agent_name: 지정하면 에이전트별로 memory를 저장한다. None이면 전역 memory를 쓴다.
            memory_config: 명시적인 memory config. 생략하면 기존 전역 config로 대체한다.
        """
        super().__init__()
        self._agent_name = agent_name
        self._memory_config = memory_config

    def _resolve_add_args(self, state: MemoryMiddlewareState, runtime: Runtime) -> tuple[str, list, str, str | None] | None:
        """manager를 호출하지 않고 쓰기 요청 하나를 구성한다."""
        config = self._memory_config or get_memory_config()
        if not config.enabled:
            return None

        # thread ID를 먼저 runtime context에서 찾고, 없으면 LangGraph의 configurable metadata로 대체한다
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            config_data = get_config()
            thread_id = config_data.get("configurable", {}).get("thread_id")
        if not thread_id:
            logger.debug("No thread_id in context, skipping memory update")
            return None

        # state에서 메시지를 가져온다
        messages = state.get("messages", [])
        if not messages:
            logger.debug("No messages in state, skipping memory update")
            return None

        # 요청 context가 살아 있는 enqueue 시점에 user_id를 캡처한다. threading.Timer는
        # ContextVar 값이 전파되지 않는 다른 thread에서 실행되므로, user_id를
        # ConversationContext에 명시적으로 저장해야 한다.
        user_id = resolve_runtime_user_id(runtime)
        runtime_context = runtime.context if isinstance(runtime.context, dict) else {}
        trace_id = normalize_trace_id(runtime_context.get(DEERFLOW_TRACE_METADATA_KEY))
        if trace_id is None:
            try:
                config_data = get_config()
            except RuntimeError:
                config_data = {}
            config_metadata = config_data.get("metadata", {}) if isinstance(config_data.get("metadata"), dict) else {}
            trace_id = normalize_trace_id(config_metadata.get(DEERFLOW_TRACE_METADATA_KEY))
        if trace_id is None:
            trace_id = get_current_trace_id()

        return thread_id, messages, user_id, trace_id

    @override
    def after_agent(self, state: MemoryMiddlewareState, runtime: Runtime) -> dict | None:
        """에이전트가 끝난 뒤 대화를 memory 갱신 큐에 넣는다."""
        add_args = self._resolve_add_args(state, runtime)
        if add_args is None:
            return None
        thread_id, messages, user_id, trace_id = add_args

        # 원본 메시지를 manager에 넘긴다. backend가 사용자 + 최종 AI 턴만 걸러 내고 검증한 뒤
        # 정정/강화 여부를 판별하고 큐에 넣는다.
        get_memory_manager().add(
            thread_id,
            messages,
            agent_name=self._agent_name,
            user_id=user_id,
            trace_id=trace_id,
        )

        return None

    @override
    async def aafter_agent(self, state: MemoryMiddlewareState, runtime: Runtime) -> dict | None:
        """LangGraph의 async 실행 경로에서는 manager의 async 경계를 사용한다."""
        add_args = self._resolve_add_args(state, runtime)
        if add_args is None:
            return None
        thread_id, messages, user_id, trace_id = add_args
        manager = await asyncio.to_thread(get_memory_manager)
        await manager.aadd(
            thread_id,
            messages,
            agent_name=self._agent_name,
            user_id=user_id,
            trace_id=trace_id,
        )
        return None
