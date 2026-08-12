"""extension middleware의 실패를 사용자 run에서 격리한다.

extension middleware는 LangChain 호출 체인 안에서 실행되므로, 처리되지 않은 예외는 사용자
run을 중단시킨다. 기여된 middleware는 모두 wrapping 되어, 관찰 실패가 diagnostic으로 격하되고
호출은 그대로 통과한다. 격리 복구가 model 요청이나 tool 부작용을 한 번 더 만들지 않도록
downstream handler를 추적한다. handler 이전에 발생한 extension 실패는 handler를 한 번
호출하고, handler 이후 실패는 이미 잡아둔 결과를 반환하며, handler 자체의 실패는 graph의 error
정책이 계속 담당한다.

wrapper는 네 개의 wrap-call hook만이 아니라 inner middleware의 인터페이스 전체를 그대로
반영해야 한다. LangChain은 wrapper를 들여다봐서 기능을 발견하기 때문이다. hook 참여 여부는
class 레벨 identity 검사(`m.__class__.before_model is not AgentMiddleware.before_model`)로,
tools/state_schema/transformers는 인스턴스 속성으로 확인한다. lifecycle 반영은 양방향 모두
정확히 일치한다. LangChain은 sync/async wrap 쌍을 하나의 기능으로 취급해서 한쪽만 있어도 두
실행 경로를 모두 연결하므로, inner가 한쪽만 구현했으면 wrapper가 조용한 pass-through 짝을
공급한다. 그러지 않으면 격리가 fail open 하기도 전에 base class가 ``NotImplementedError``를
raise 한다.

첫 버전의 기여는 전부 관찰용이므로 fail-open이다. 앞으로 개입(결정)하는 기여가 생긴다면
fail closed 해야 하며, 이 wrapper를 명시적으로 opt out 해야 한다.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.errors import GraphBubbleUp

from deerflow.extensions.loader import Diagnostic

logger = logging.getLogger(__name__)

_UNSAFE_GRAPH_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def graph_safe_middleware_name(value: str) -> str:
    """LangGraph 노드 이름으로 쓸 수 있게 middleware identity를 정규화한다."""
    return _UNSAFE_GRAPH_NAME.sub("_", value)


_WRAP_HOOKS = ("wrap_model_call", "awrap_model_call", "wrap_tool_call", "awrap_tool_call")
_WRAP_HOOK_PAIRS = (
    ("wrap_model_call", "awrap_model_call"),
    ("wrap_tool_call", "awrap_tool_call"),
)
_LIFECYCLE_HOOKS = (
    "before_agent",
    "abefore_agent",
    "before_model",
    "abefore_model",
    "after_model",
    "aafter_model",
    "after_agent",
    "aafter_agent",
)


def _implemented_hooks(inner: AgentMiddleware) -> frozenset[str]:
    """``inner``가 실제로 재정의한 hook을 LangChain 자신의 class 레벨 identity 검사로 판별한다.
    인스턴스 레벨 속성은 factory에게 보이지 않으므로 여기서도 보이지 않는다."""
    return frozenset(hook for hook in (*_WRAP_HOOKS, *_LIFECYCLE_HOOKS) if getattr(type(inner), hook, None) is not getattr(AgentMiddleware, hook, None))


def _make_sync_wrap_delegate(hook: str):
    def delegate(self: IsolatedMiddleware, request: Any, handler: Callable[[Any], Any]) -> Any:
        return self._invoke_sync(hook, getattr(self._inner, hook), request, handler)

    return delegate


def _make_async_wrap_delegate(hook: str):
    async def delegate(self: IsolatedMiddleware, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await self._invoke_async(hook, getattr(self._inner, hook), request, handler)

    return delegate


def _make_sync_wrap_passthrough():
    def delegate(self: IsolatedMiddleware, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(request)

    return delegate


def _make_async_wrap_passthrough():
    async def delegate(self: IsolatedMiddleware, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(request)

    return delegate


def _make_sync_lifecycle_delegate(hook: str):
    def delegate(self: IsolatedMiddleware, state: Any, runtime: Any) -> Any:
        return self._invoke_lifecycle_sync(hook, state, runtime)

    return delegate


def _make_async_lifecycle_delegate(hook: str):
    async def delegate(self: IsolatedMiddleware, state: Any, runtime: Any) -> Any:
        return await self._invoke_lifecycle_async(hook, state, runtime)

    return delegate


# async 변형은 명시적으로 판별한다. startswith("a")로 하면 sync after_model/after_agent까지
# 걸리기 때문이다.
_ASYNC_HOOKS = frozenset(hook for hook in (*_WRAP_HOOKS, *_LIFECYCLE_HOOKS) if hook[1:].startswith(("wrap", "before", "after")))


def _delegate_for(hook: str):
    if hook in _WRAP_HOOKS:
        return _make_async_wrap_delegate(hook) if hook in _ASYNC_HOOKS else _make_sync_wrap_delegate(hook)
    return _make_async_lifecycle_delegate(hook) if hook in _ASYNC_HOOKS else _make_sync_lifecycle_delegate(hook)


_subclass_cache: dict[frozenset[str], type[IsolatedMiddleware]] = {}
_subclass_cache_lock = threading.Lock()


def _wrapper_subclass(hooks: frozenset[str]) -> type[IsolatedMiddleware]:
    """``hooks``와 필요한 wrap-hook pass-through 짝을 정의하는, 캐시된 IsolatedMiddleware
    subclass를 반환한다.

    middleware 단위가 아니라 hook 집합 단위다. 구현한 hook 조합이 같은 inner middleware는
    subclass 하나를 공유한다.
    """
    with _subclass_cache_lock:
        subclass = _subclass_cache.get(hooks)
        if subclass is None:
            namespace = {hook: _delegate_for(hook) for hook in hooks}
            for sync_hook, async_hook in _WRAP_HOOK_PAIRS:
                if sync_hook in hooks and async_hook not in hooks:
                    namespace[async_hook] = _make_async_wrap_passthrough()
                elif async_hook in hooks and sync_hook not in hooks:
                    namespace[sync_hook] = _make_sync_wrap_passthrough()
            subclass = type(IsolatedMiddleware.__name__, (IsolatedMiddleware,), namespace)
            _subclass_cache[hooks] = subclass
        return subclass


class IsolatedMiddleware(AgentMiddleware):
    """extension middleware 하나를 감싸서 그 실패가 run을 망가뜨리지 못하게 한다.

    인스턴스화하면 inner middleware가 구현한 hook만 정확히 정의한 캐시된 subclass가 반환된다.
    그래서 LangChain의 class 레벨 기능 검사는 wrapper에서 inner middleware와 동일한 인터페이스를
    본다.
    """

    def __new__(cls, inner: AgentMiddleware, source: str, on_error: Callable[[Diagnostic], None], *, name: str | None = None):
        if cls is IsolatedMiddleware:
            cls = _wrapper_subclass(_implemented_hooks(inner))
        return super().__new__(cls)

    def __init__(
        self,
        inner: AgentMiddleware,
        source: str,
        on_error: Callable[[Diagnostic], None],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._source = source
        self._on_error = on_error
        if name is None:
            inner_name = getattr(inner, "name", type(inner).__name__)
            name = f"extension:{source}:{inner_name}"
        self._name = graph_safe_middleware_name(name)
        # LangChain이 middleware 인스턴스에서 읽는 선언적 기여 속성을 그대로 반영한다
        # (factory.py: m.tools, m.state_schema, m.transformers). state_schema는 base에서는
        # class 속성이지만 여기서는 인스턴스별이어야 한다. 캐시된 subclass는 schema가 서로
        # 다른 middleware들 사이에서 공유되기 때문이다.
        self.tools = getattr(inner, "tools", [])
        self.transformers = getattr(inner, "transformers", ())
        self.state_schema = getattr(inner, "state_schema", AgentMiddleware.state_schema)

    @property
    def name(self) -> str:
        """이 격리된 기여의 안정적인 graph/trace identity."""
        return self._name

    @property
    def inner(self) -> AgentMiddleware:
        """감싸진 middleware. 순서 검사와 테스트에서 쓴다."""
        return self._inner

    @property
    def source(self) -> str:
        """이 middleware가 온 extension. provenance map이 읽는다."""
        return self._source

    def _report(self, hook: str, exc: Exception) -> None:
        message = f"{type(self._inner).__name__}.{hook} failed and was skipped: {exc}"
        logger.exception("Extension %s: %s", self._source, message)
        try:
            self._on_error(Diagnostic.error(self._source, message))
        except Exception:  # pragma: no cover - 보고는 절대 raise 하면 안 된다
            logger.exception("Extension %s: diagnostic reporting failed", self._source)

    def _invoke_sync(
        self,
        hook: str,
        inner_hook: Callable[[Any, Callable[[Any], Any]], Any],
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        handler_called = False
        handler_succeeded = False
        handler_result: Any = None
        handler_error: BaseException | None = None
        handler_error_traceback: TracebackType | None = None
        duplicate_call_error: RuntimeError | None = None

        def tracked_handler(inner_request: Any) -> Any:
            nonlocal handler_called, duplicate_call_error
            nonlocal handler_error, handler_error_traceback
            nonlocal handler_result, handler_succeeded
            if handler_called:
                duplicate_call_error = RuntimeError(f"{type(self._inner).__name__}.{hook} called the downstream handler more than once")
                raise duplicate_call_error
            handler_called = True
            handler_error = None
            handler_error_traceback = None
            handler_succeeded = False
            try:
                # 계약의 첫 조각은 관찰용이다. 기여된 wrapper는 request를 들여다볼 수는 있지만,
                # host의 policy/authorization 레이어가 실행된 뒤에 새 request로 갈아끼울 수는
                # 없다.
                handler_result = handler(request)
            except BaseException as exc:
                handler_error = exc
                handler_error_traceback = exc.__traceback__
                raise
            else:
                handler_succeeded = True
                return handler_result

        try:
            inner_hook(request, tracked_handler)
            if handler_error is not None:
                raise handler_error.with_traceback(handler_error_traceback)
            if duplicate_call_error is not None:
                raise duplicate_call_error
            if not handler_called:
                raise RuntimeError(f"{type(self._inner).__name__}.{hook} did not call the downstream handler")
            return handler_result
        except GraphBubbleUp as exc:
            if handler_error is not None:
                if handler_error is exc:
                    raise
                raise handler_error.with_traceback(handler_error_traceback) from None
            if handler_succeeded:
                self._report(hook, duplicate_call_error or exc)
                return handler_result
            raise
        except Exception as exc:
            if handler_error is not None:
                if handler_error is exc:
                    raise
                raise handler_error.with_traceback(handler_error_traceback) from None
            self._report(hook, exc)
            if handler_succeeded:
                return handler_result
            return handler(request)

    async def _invoke_async(
        self,
        hook: str,
        inner_hook: Callable[
            [Any, Callable[[Any], Awaitable[Any]]],
            Awaitable[Any],
        ],
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        handler_called = False
        handler_succeeded = False
        handler_result: Any = None
        handler_error: BaseException | None = None
        handler_error_traceback: TracebackType | None = None
        duplicate_call_error: RuntimeError | None = None

        async def tracked_handler(inner_request: Any) -> Any:
            nonlocal handler_called, duplicate_call_error
            nonlocal handler_error, handler_error_traceback
            nonlocal handler_result, handler_succeeded
            if handler_called:
                duplicate_call_error = RuntimeError(f"{type(self._inner).__name__}.{hook} called the downstream handler more than once")
                raise duplicate_call_error
            handler_called = True
            handler_error = None
            handler_error_traceback = None
            handler_succeeded = False
            try:
                handler_result = await handler(request)
            except BaseException as exc:
                handler_error = exc
                handler_error_traceback = exc.__traceback__
                raise
            else:
                handler_succeeded = True
                return handler_result

        try:
            await inner_hook(request, tracked_handler)
            if handler_error is not None:
                raise handler_error.with_traceback(handler_error_traceback)
            if duplicate_call_error is not None:
                raise duplicate_call_error
            if not handler_called:
                raise RuntimeError(f"{type(self._inner).__name__}.{hook} did not call the downstream handler")
            return handler_result
        except GraphBubbleUp as exc:
            if handler_error is not None:
                if handler_error is exc:
                    raise
                raise handler_error.with_traceback(handler_error_traceback) from None
            if handler_succeeded:
                self._report(hook, duplicate_call_error or exc)
                return handler_result
            raise
        except Exception as exc:
            if handler_error is not None:
                if handler_error is exc:
                    raise
                raise handler_error.with_traceback(handler_error_traceback) from None
            self._report(hook, exc)
            if handler_succeeded:
                return handler_result
            return await handler(request)

    def _invoke_lifecycle_sync(self, hook: str, state: Any, runtime: Any) -> Any:
        """lifecycle hook에는 넘어갈 handler가 없다. 관찰이 실패했을 때의 fail-open 격하는
        state를 아예 갱신하지 않는 것이다."""
        try:
            return getattr(self._inner, hook)(state, runtime)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self._report(hook, exc)
            return None

    async def _invoke_lifecycle_async(self, hook: str, state: Any, runtime: Any) -> Any:
        try:
            return await getattr(self._inner, hook)(state, runtime)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self._report(hook, exc)
            return None
