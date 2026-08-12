import logging
from datetime import UTC, datetime
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import Paths, get_paths
from deerflow.runtime.user_context import resolve_runtime_user_id

logger = logging.getLogger(__name__)


class ThreadDataMiddlewareState(AgentState):
    """`ThreadState` 스키마와 호환된다."""

    thread_data: NotRequired[ThreadDataState | None]


class ThreadDataMiddleware(AgentMiddleware[ThreadDataMiddlewareState]):
    """thread 실행마다 thread 데이터 디렉터리를 만든다.

    다음 디렉터리 구조를 만든다.

    - {base_dir}/threads/{thread_id}/user-data/workspace
    - {base_dir}/threads/{thread_id}/user-data/uploads
    - {base_dir}/threads/{thread_id}/user-data/outputs

    생명주기 관리:

    - lazy_init=True(기본값): 경로만 계산하고 디렉터리는 필요할 때 만든다.
    - lazy_init=False: before_agent()에서 디렉터리를 즉시 만든다.
    """

    state_schema = ThreadDataMiddlewareState

    def __init__(self, base_dir: str | None = None, lazy_init: bool = True):
        """미들웨어를 초기화한다.

        Args:
            base_dir: thread 데이터의 기준 디렉터리. 기본값은 Paths가 해석한 경로다.
            lazy_init: True면 디렉터리 생성을 필요할 때까지 미룬다.
                      False면 before_agent()에서 즉시 만든다.
                      성능을 위해 기본값은 True다.
        """
        super().__init__()
        self._paths = Paths(base_dir) if base_dir else get_paths()
        self._lazy_init = lazy_init

    def _get_thread_paths(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """thread 데이터 디렉터리 경로를 반환한다.

        Args:
            thread_id: thread ID.
            user_id: 사용자별 경로 격리를 위한 선택적 user ID.

        Returns:
            workspace_path, uploads_path, outputs_path를 담은 dict.
        """
        return {
            "workspace_path": str(self._paths.sandbox_work_dir(thread_id, user_id=user_id)),
            "uploads_path": str(self._paths.sandbox_uploads_dir(thread_id, user_id=user_id)),
            "outputs_path": str(self._paths.sandbox_outputs_dir(thread_id, user_id=user_id)),
        }

    def _create_thread_directories(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """thread 데이터 디렉터리를 생성한다.

        Args:
            thread_id: thread ID.
            user_id: 사용자별 경로 격리를 위한 선택적 user ID.

        Returns:
            생성된 디렉터리 경로를 담은 dict.
        """
        self._paths.ensure_thread_dirs(thread_id, user_id=user_id)
        return self._get_thread_paths(thread_id, user_id=user_id)

    @override
    def before_agent(self, state: ThreadDataMiddlewareState, runtime: Runtime) -> dict | None:
        context = runtime.context or {}
        thread_id = context.get("thread_id")
        if thread_id is None:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")

        if thread_id is None:
            raise ValueError("Thread ID is required in runtime context or config.configurable")

        user_id = resolve_runtime_user_id(runtime)

        if self._lazy_init:
            # 지연 초기화: 경로만 계산하고 디렉터리는 만들지 않는다.
            paths = self._get_thread_paths(thread_id, user_id=user_id)
        else:
            # 즉시 초기화: 디렉터리를 바로 만든다.
            paths = self._create_thread_directories(thread_id, user_id=user_id)
            logger.debug("Created thread data directories for thread %s", thread_id)

        messages = list(state.get("messages", []))
        last_message = messages[-1] if messages else None

        if last_message and isinstance(last_message, HumanMessage):
            messages[-1] = HumanMessage(
                content=last_message.content,
                id=last_message.id,
                name=last_message.name or "user-input",
                additional_kwargs={**last_message.additional_kwargs, "run_id": context.get("run_id"), "timestamp": datetime.now(UTC).isoformat()},
            )

        return {
            "thread_data": {
                **paths,
            },
            "messages": messages,
        }
