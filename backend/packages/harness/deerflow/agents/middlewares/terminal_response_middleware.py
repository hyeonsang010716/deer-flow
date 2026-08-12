"""도구를 사용한 lead-agent 턴이 사용자에게 보이는 assistant 응답으로 끝나도록 보장한다."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares._bounded_dict import BoundedDict

_RECOVERY_PROMPT = (
    "<system_reminder>\n"
    "Your previous response after the tool execution was empty. Review the tool results "
    "already present in the conversation and provide a concise, user-visible final response. "
    "Do not call another tool unless it is strictly necessary.\n"
    "</system_reminder>"
)

_FALLBACK_CONTENT = "The model completed the tool run but returned no final response, including after one automatic retry. Please try again or use a different model."

_TOOL_CALL_FINISH_REASONS = {"tool_calls", "function_call"}


def _has_visible_content(message: AIMessage) -> bool:
    """AI 메시지에 사용자에게 보이는 텍스트가 있는지 반환한다."""
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                return True
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    return False


def _has_tool_call_intent_or_error(message: AIMessage) -> bool:
    """도구 라우팅과 잘못된 tool-call 처리는 이 가드의 대상에서 제외한다."""
    if message.tool_calls or getattr(message, "invalid_tool_calls", None):
        return True
    additional_kwargs = message.additional_kwargs or {}
    if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
        return True
    response_metadata = message.response_metadata or {}
    return response_metadata.get("finish_reason") in _TOOL_CALL_FINISH_REASONS


def _tool_result_in_current_turn(messages: list[Any]) -> bool:
    """가장 최근의 실제 사용자 메시지 뒤에 도구 결과가 있는지 반환한다."""
    latest_user_index = -1
    for index, message in enumerate(messages):
        if not isinstance(message, HumanMessage):
            continue
        if (message.additional_kwargs or {}).get("hide_from_ui"):
            continue
        latest_user_index = index
    # 범위: #4027은 대화형의 도구 실행 이후 턴을 다룬다. 실제 HumanMessage가 없는 scheduled/내부 호출은
    # 과거의 임의 도구 사용에서 추론할 것이 아니라 별도의 terminal-success 불변식이 필요하다.
    if latest_user_index == -1:
        return False
    return any(isinstance(message, ToolMessage) for message in messages[latest_user_index + 1 :])


class TerminalResponseMiddleware(AgentMiddleware[AgentState]):
    """도구 실행 후 빈 응답을 한 번 재시도하고, 그래도 비면 보이는 error fallback을 남긴다."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._retry_counts: BoundedDict[tuple[str, str], int] = BoundedDict(1000)
        self._pending_prompts: BoundedDict[tuple[str, str], bool] = BoundedDict(1000)

    @staticmethod
    def _key(runtime: Runtime) -> tuple[str, str]:
        context = getattr(runtime, "context", None)
        if isinstance(context, dict):
            thread_id = str(context.get("thread_id") or "unknown-thread")
            run_id = str(context.get("run_id") or context.get("run_attempt_id") or id(runtime))
            return thread_id, run_id
        # 테스트나 custom 임베딩을 위한 방어적 fallback이다. 실제 Gateway run은 항상 Runtime.context에
        # thread_id와 run_id를 제공한다.
        return "unknown-thread", str(id(runtime))

    def _clear(self, runtime: Runtime) -> None:
        key = self._key(runtime)
        with self._lock:
            self._retry_counts.pop(key, None)
            self._pending_prompts.pop(key, None)

    def _clear_other_runs(self, runtime: Runtime) -> None:
        thread_id, run_id = self._key(runtime)
        with self._lock:
            stale = [key for key in self._retry_counts if key[0] == thread_id and key[1] != run_id]
            for key in stale:
                self._retry_counts.pop(key, None)
                self._pending_prompts.pop(key, None)

    def _apply(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        last = messages[-1]
        if _has_visible_content(last) or _has_tool_call_intent_or_error(last):
            return None
        if not _tool_result_in_current_turn(messages):
            return None

        key = self._key(runtime)
        with self._lock:
            # 복구 예산은 빈 메시지마다가 아니라 run당 한 번이다. 재시도가 또 다른 도구를 호출하더라도
            # 예산을 갱신해 빈 응답 -> 재시도 -> 도구의 무한 루프를 만들면 안 된다.
            retry_count = self._retry_counts.get(key, 0)
            if retry_count == 0:
                self._retry_counts[key] = 1
                self._pending_prompts[key] = True

        if retry_count == 0:
            # 다음 model 호출은 새 메시지 id를 받는다. 복구가 성공했을 때 이 빈 종료 메시지가 checkpoint
            # 이력이나 이후 model context에 남지 않도록 지금 제거한다.
            message_updates = [RemoveMessage(id=last.id)] if last.id else []
            return {"messages": message_updates, "jump_to": "model"}

        additional_kwargs = dict(last.additional_kwargs or {})
        additional_kwargs.update(
            {
                "deerflow_error_fallback": True,
                "error_reason": "Model returned an empty terminal response after one retry",
            }
        )
        fallback = last.model_copy(
            update={
                "content": _FALLBACK_CONTENT,
                "additional_kwargs": additional_kwargs,
            }
        )
        return {"messages": [fallback]}

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        key = self._key(request.runtime)
        with self._lock:
            pending = key in self._pending_prompts
            self._pending_prompts.pop(key, None)
        if not pending:
            return request
        reminder = HumanMessage(
            content=_RECOVERY_PROMPT,
            name="terminal_response_recovery",
            additional_kwargs={"hide_from_ui": True},
        )
        return request.override(messages=[*request.messages, reminder])

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_other_runs(runtime)
        # 이전 호출이 Command(goto=END)로 after_agent를 건너뛸 수 있다. 여기서 같은 run id를 초기화해
        # resume이 재시도 1회 예산으로 새로 시작하게 한다. 내부 jump_to=model 루프는 before_agent를
        # 다시 실행하지 않는다.
        self._clear(runtime)
        return None

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_other_runs(runtime)
        self._clear(runtime)
        return None

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._augment_request(request))

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear(runtime)
        return None

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear(runtime)
        return None
