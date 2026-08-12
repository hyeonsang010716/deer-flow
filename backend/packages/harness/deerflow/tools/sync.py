"""동기 agent 경로에서 async tool을 호출하기 위한 유틸리티."""

import asyncio
import atexit
import concurrent.futures
import contextvars
import functools
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# async 환경에서 sync tool을 호출할 때 공용으로 쓰는 thread pool.
_SYNC_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="tool-sync")

atexit.register(lambda: _SYNC_TOOL_EXECUTOR.shutdown(wait=False))


def _get_runnable_config_param(func: Callable[..., Any]) -> str | None:
    """LangChain RunnableConfig를 기대하는 coroutine 파라미터 이름을 반환한다."""
    if isinstance(func, functools.partial):
        func = func.func

    try:
        type_hints = get_type_hints(func)
    except Exception:
        return None

    for name, type_ in type_hints.items():
        if type_ is RunnableConfig:
            return name
    return None


def make_sync_tool_wrapper(coro: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """async tool coroutine에 대한 동기 wrapper를 만든다.

    Args:
        coro: LangChain tool을 뒷받침하는 async callable.
        tool_name: 에러 로그에 쓰는 tool 이름.

    Returns:
        ``BaseTool.func``에 넣을 수 있는 sync callable.

    Notes:
        ``coro``가 ``RunnableConfig`` 파라미터를 선언하면 이 wrapper는
        ``config: RunnableConfig``를 노출해 LangChain이 runtime config를 주입할 수 있게 하고,
        이를 coroutine에서 탐지한 config 파라미터로 전달한다. ``invoke_acp_agent``처럼 config에
        의존하는 현재 DeerFlow tool들이 여기에 해당한다.

        이 wrapper는 의도적으로 동적 함수 시그니처를 만들지 않는다. 앞으로 사용자에게 노출되는 일반
        인자 이름이 ``config``이면서 ``RunnableConfig`` 파라미터는 ``run_config`` 같은 다른 이름으로
        가진 async tool이 생기면, LangChain이 주입하는 ``config`` 인자와 충돌할 수 있다. 그런
        시그니처를 쓰기 전에 사용자 노출 필드 이름을 바꾸거나 이 헬퍼를 확장한다.
    """
    config_param = _get_runnable_config_param(coro)

    def run_coroutine(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop is not None and loop.is_running():
                context = contextvars.copy_context()
                future = _SYNC_TOOL_EXECUTOR.submit(context.run, lambda: asyncio.run(coro(*args, **kwargs)))
                return future.result()
            return asyncio.run(coro(*args, **kwargs))
        except Exception as e:
            logger.error("Error invoking tool %r via sync wrapper: %s", tool_name, e, exc_info=True)
            raise

    if config_param:

        def sync_wrapper(*args: Any, config: RunnableConfig = None, **kwargs: Any) -> Any:
            if config is not None or config_param not in kwargs:
                kwargs[config_param] = config
            return run_coroutine(*args, **kwargs)

        return sync_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return run_coroutine(*args, **kwargs)

    return sync_wrapper
