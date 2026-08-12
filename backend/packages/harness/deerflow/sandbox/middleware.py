import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace as dc_replace
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deerflow.agents.thread_state import SandboxStateField, ThreadDataState
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.sandbox import get_sandbox_provider
from deerflow.sandbox.overwrite import unwrap_sandbox

logger = logging.getLogger(__name__)


class SandboxMiddlewareState(AgentState):
    """`ThreadState` schema와 호환된다."""

    sandbox: SandboxStateField
    thread_data: NotRequired[ThreadDataState | None]


class SandboxMiddleware(AgentMiddleware[SandboxMiddlewareState]):
    """sandbox 환경을 만들어 agent에 할당한다.

    생명주기 관리:
    - lazy_init=True(기본): 첫 tool call 때 sandbox를 획득한다
    - lazy_init=False: 첫 agent 호출(before_agent) 때 sandbox를 획득한다
    - sandbox는 같은 thread의 여러 턴에 걸쳐 재사용된다
    - 불필요한 재생성을 피하려고 agent 호출마다 sandbox를 release하지 않는다
    - 정리는 애플리케이션 종료 시 SandboxProvider.shutdown()에서 이뤄진다
    """

    state_schema = SandboxMiddlewareState

    def __init__(self, lazy_init: bool = True):
        """sandbox middleware를 초기화한다.

        Args:
            lazy_init: True면 sandbox 획득을 첫 tool call까지 미룬다. False면
                      before_agent()에서 즉시 획득한다. 성능을 위해 기본값은 True다.
        """
        super().__init__()
        self._lazy_init = lazy_init

    def _acquire_sandbox(self, thread_id: str, *, user_id: str) -> str:
        provider = get_sandbox_provider()
        sandbox_id = provider.acquire(thread_id, user_id=user_id)
        logger.info(f"Acquiring sandbox {sandbox_id}")
        return sandbox_id

    async def _acquire_sandbox_async(self, thread_id: str, *, user_id: str) -> str:
        provider = get_sandbox_provider()
        sandbox_id = await provider.acquire_async(thread_id, user_id=user_id)
        logger.info(f"Acquiring sandbox {sandbox_id}")
        return sandbox_id

    async def _release_sandbox_async(self, sandbox_id: str) -> None:
        await asyncio.to_thread(get_sandbox_provider().release, sandbox_id)

    @override
    def before_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        # lazy_init이 켜져 있으면 획득을 건너뛴다
        if self._lazy_init:
            return super().before_agent(state, runtime)

        # 즉시 초기화(원래 동작)
        if "sandbox" not in state or state["sandbox"] is None:
            thread_id = (runtime.context or {}).get("thread_id")
            if thread_id is None:
                return super().before_agent(state, runtime)
            sandbox_id = self._acquire_sandbox(thread_id, user_id=resolve_runtime_user_id(runtime))
            logger.info(f"Assigned sandbox {sandbox_id} to thread {thread_id}")
            return {"sandbox": {"sandbox_id": sandbox_id}}
        return super().before_agent(state, runtime)

    @override
    async def abefore_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        # lazy_init이 켜져 있으면 획득을 건너뛴다
        if self._lazy_init:
            return await super().abefore_agent(state, runtime)

        # 즉시 초기화(원래 동작). 단 blocking sandbox 시작/폴링이 event loop 밖에서 돌도록
        # async provider hook을 쓴다.
        if "sandbox" not in state or state["sandbox"] is None:
            thread_id = (runtime.context or {}).get("thread_id")
            if thread_id is None:
                return await super().abefore_agent(state, runtime)
            sandbox_id = await self._acquire_sandbox_async(thread_id, user_id=resolve_runtime_user_id(runtime))
            logger.info(f"Assigned sandbox {sandbox_id} to thread {thread_id}")
            return {"sandbox": {"sandbox_id": sandbox_id}}
        return await super().abefore_agent(state, runtime)

    @override
    def after_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        sandbox, fork_restored = unwrap_sandbox(state.get("sandbox"))
        if sandbox is not None:
            sandbox_id = sandbox["sandbox_id"]
            if fork_restored:
                # 감싸인 값은 부모 thread의 sandbox 상태를 재현한 것이므로, 여기서 release하면
                # 부모의 warm sandbox가 회수된다.
                logger.info(f"Not releasing fork-restored sandbox {sandbox_id}")
                return None
            logger.info(f"Releasing sandbox {sandbox_id}")
            get_sandbox_provider().release(sandbox_id)
            return None

        if (runtime.context or {}).get("sandbox_id") is not None:
            sandbox_id = runtime.context.get("sandbox_id")
            logger.info(f"Releasing sandbox {sandbox_id} from context")
            get_sandbox_provider().release(sandbox_id)
            return None

        # release할 sandbox가 없다
        return super().after_agent(state, runtime)

    @override
    async def aafter_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        sandbox, fork_restored = unwrap_sandbox(state.get("sandbox"))
        if sandbox is not None:
            sandbox_id = sandbox["sandbox_id"]
            if fork_restored:
                # 감싸인 값은 부모 thread의 sandbox 상태를 재현한 것이므로, 여기서 release하면
                # 부모의 warm sandbox가 회수된다.
                logger.info(f"Not releasing fork-restored sandbox {sandbox_id}")
                return None
            logger.info(f"Releasing sandbox {sandbox_id}")
            await self._release_sandbox_async(sandbox_id)
            return None

        if (runtime.context or {}).get("sandbox_id") is not None:
            sandbox_id = runtime.context.get("sandbox_id")
            logger.info(f"Releasing sandbox {sandbox_id} from context")
            await self._release_sandbox_async(sandbox_id)
            return None

        # release할 sandbox가 없다
        return await super().aafter_agent(state, runtime)

    # ------------------------------------------------------------------
    # tool-call wrapper: lazy하게 획득한 sandbox 상태를 Command(update=...)로 graph state에
    # 반영한다.
    #
    # 배경:
    #   ``deerflow.sandbox.tools``의 ``ensure_sandbox_initialized*``는
    #   ``runtime.state["sandbox"]``를 직접 변경한다. 그 변경은 현재 tool 호출에 국한되며
    #   LangGraph의 channel reducer가 인식하지 못하므로, 이후 graph step과 하위
    #   소비자(``ToolOutputBudgetMiddleware``, subagent ``task_tool`` 등)가 sandbox id를 볼 수
    #   없다. tool call을 감싸면 handler 전후의 state snapshot을 비교해 새로 일어난 lazy init을
    #   감지하고 ``Command``로 제대로 된 state 업데이트를 낼 수 있다.
    # ------------------------------------------------------------------

    @staticmethod
    def _read_sandbox_id_from_state(state: object) -> str | None:
        if not isinstance(state, dict):
            return None
        sandbox_state = state.get("sandbox")
        if not isinstance(sandbox_state, dict):
            return None
        sandbox_id = sandbox_state.get("sandbox_id")
        return sandbox_id if isinstance(sandbox_id, str) else None

    @staticmethod
    def _attach_sandbox_update(result: ToolMessage | Command, sandbox_id: str) -> ToolMessage | Command:
        """``sandbox.sandbox_id``가 반영되도록 ``result``를 감싸거나 병합한다.

        - ``ToolMessage`` -> ``Command(update={"sandbox": ..., "messages": [msg]})``
        - dict update를 가진 ``Command`` -> ``sandbox`` 키를 병합하고 기존 필드
          (``messages``, ``goto``, ``graph``, ``resume`` 등)는 모두 보존한다.
        - dict가 아니거나 None인 update를 가진 ``Command`` -> 알 수 없는 update 형태에서 데이터가
          조용히 유실되지 않도록 그대로 둔다.
        """
        sandbox_update = {"sandbox": {"sandbox_id": sandbox_id}}

        if isinstance(result, ToolMessage):
            return Command(update={**sandbox_update, "messages": [result]})

        existing_update = result.update
        if isinstance(existing_update, dict):
            merged_update = {**existing_update, **sandbox_update}
            return dc_replace(result, update=merged_update)
        return result

    @staticmethod
    def _read_sandbox_id_from_request(request: ToolCallRequest) -> str | None:
        """runtime.state에서 sandbox_id를 읽는다(ensure_sandbox_initialized가 쓰는 위치)."""
        runtime = request.runtime
        if runtime is None or runtime.state is None:
            return None
        return SandboxMiddleware._read_sandbox_id_from_state(runtime.state)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        prev_sandbox_id = self._read_sandbox_id_from_request(request)
        result = handler(request)
        if prev_sandbox_id is not None:
            return result
        curr_sandbox_id = self._read_sandbox_id_from_request(request)
        if curr_sandbox_id is None:
            return result
        return self._attach_sandbox_update(result, curr_sandbox_id)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        prev_sandbox_id = self._read_sandbox_id_from_request(request)
        result = await handler(request)
        if prev_sandbox_id is not None:
            return result
        curr_sandbox_id = self._read_sandbox_id_from_request(request)
        if curr_sandbox_id is None:
            return result
        return self._attach_sandbox_update(result, curr_sandbox_id)
