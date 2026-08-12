"""TodoListMiddleware에 context 유실 감지와 조기 종료 방지를 더한 미들웨어.

메시지 이력이 잘리면(예: SummarizationMiddleware에 의해) 원래의 `write_todos` 도구 호출과 그
ToolMessage가 활성 context window 밖으로 밀려날 수 있다. 이 미들웨어는 그 상황을 감지해 reminder
메시지를 주입하고, 모델이 남은 todo 목록을 계속 인지하게 한다.

또한 미완료 todo가 남아 있는 동안 agent가 루프를 빠져나가지 못하게 막는다. 모델이 도구 호출 없이 최종
응답을 냈지만 todo가 아직 끝나지 않았다면, 다음 model 요청용 reminder를 큐에 넣고 model 노드로 되돌아가
작업을 이어가게 한다. 이 완료 reminder는 graph state에 사용자에게 보이는 일반 메시지로 저장되지 않고
``wrap_model_call``을 통해 주입된다.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.todo import Todo
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState


def _todos_in_messages(messages: list[Any]) -> bool:
    """*messages* 안의 AIMessage 중 write_todos 도구 호출을 담은 것이 있으면 True를 반환한다."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "write_todos":
                    return True
    return False


def _reminder_in_messages(messages: list[Any]) -> bool:
    """*messages*에 todo_reminder HumanMessage가 이미 있으면 True를 반환한다."""
    for msg in messages:
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "todo_reminder":
            return True
    return False


def _format_todos(todos: list[Todo]) -> str:
    """Todo 항목 목록을 사람이 읽을 수 있는 문자열로 포맷한다."""
    lines: list[str] = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")
        lines.append(f"- [{status}] {content}")
    return "\n".join(lines)


def _format_completion_reminder(todos: list[Todo]) -> str:
    """미완료 todo 항목에 대한 완료 reminder를 포맷한다."""
    incomplete = [t for t in todos if t.get("status") != "completed"]
    incomplete_text = "\n".join(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in incomplete)
    return (
        "<system_reminder>\n"
        "You have incomplete todo items that must be finished before giving your final response:\n\n"
        f"{incomplete_text}\n\n"
        "Please continue working on these tasks. Call `write_todos` to mark items as completed "
        "as you finish them, and only respond when all items are done.\n"
        "</system_reminder>"
    )


_TOOL_CALL_FINISH_REASONS = {"tool_calls", "function_call"}


def _has_tool_call_intent_or_error(message: AIMessage) -> bool:
    """AIMessage가 깔끔한 최종 답변이 아니면 True를 반환한다.

    todo 완료 reminder는 모델이 순수한 최종 응답을 냈을 때만 발동해야 한다. provider와 도구 파싱 세부는
    LangChain 버전과 integration에 따라 계속 바뀌었으므로, 호출부에서 특정 필드 하나를 확인하지 말고
    모든 tool-intent/error 신호를 이 헬퍼 뒤에 모아 둔다.
    """
    if message.tool_calls:
        return True

    if getattr(message, "invalid_tool_calls", None):
        return True

    # 하위/provider 호환: 일부 integration은 구조화된 tool_calls가 비어 있어도 raw 또는 legacy tool-call
    # 의도를 additional_kwargs에 남긴다. 이 헬퍼를 바꾸면 대응하는 sentinel 테스트
    # `TestToolCallIntentOrError.test_langchain_ai_message_tool_fields_are_explicitly_handled`도 갱신한다.
    # LangChain 업그레이드 후 그 테스트가 실패하면 이 헬퍼를 다시 검토해, 새로운 tool-call/error 필드가
    # 조용히 깔끔한 최종 답변으로 취급되지 않게 한다.
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
        return True

    response_metadata = getattr(message, "response_metadata", {}) or {}
    return response_metadata.get("finish_reason") in _TOOL_CALL_FINISH_REASONS


class TodoMiddleware(TodoListMiddleware):
    """TodoListMiddleware에 `write_todos` context 유실 감지를 더한다.

    원래의 `write_todos` 도구 호출이 메시지 이력에서 잘려 나가면(예: 요약 이후) 모델은 현재 todo 목록을
    인지하지 못한다. 이 미들웨어는 `before_model` / `abefore_model`에서 그 공백을 감지하고 reminder
    메시지를 주입해 모델이 진행 상황을 계속 추적하게 한다.
    """

    state_schema = ThreadState

    @override
    def before_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """write_todos가 context window에서 사라졌을 때 todo 목록 reminder를 주입한다."""
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        if not todos:
            return None

        messages = state.get("messages") or []
        if _todos_in_messages(messages):
            # write_todos가 아직 context에 보이므로 할 일이 없다.
            return None

        if _reminder_in_messages(messages):
            # reminder를 이미 주입했고 아직 잘려 나가지 않았다.
            return None

        # state에는 todo 목록이 있지만 원래의 write_todos 호출이 사라졌다.
        # 모델이 계속 인지하도록 HumanMessage 형태의 reminder를 주입한다.
        formatted = _format_todos(todos)
        reminder = HumanMessage(
            name="todo_reminder",
            additional_kwargs={"hide_from_ui": True},
            content=(
                "<system_reminder>\n"
                "Your todo list from earlier is no longer visible in the current context window, "
                "but it is still active. Here is the current state:\n\n"
                f"{formatted}\n\n"
                "Continue tracking and updating this todo list as you work. "
                "Call `write_todos` whenever the status of any item changes.\n"
                "</system_reminder>"
            ),
        )
        return {"messages": [reminder]}

    @override
    async def abefore_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """before_model의 async 버전."""
        return self.before_model(state, runtime)

    # agent의 종료를 허용하기 전까지 보낼 수 있는 완료 reminder의 최대 횟수.
    # agent가 더 진전하지 못할 때 무한 루프에 빠지는 것을 막는다.
    _MAX_COMPLETION_REMINDERS = 2
    # 오래 사는 미들웨어 인스턴스에서 run별 reminder 기록이 무한히 늘지 않도록 하는 상한.
    _MAX_COMPLETION_REMINDER_KEYS = 4096

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._pending_completion_reminders: dict[tuple[str, str], list[str]] = {}
        self._completion_reminder_counts: dict[tuple[str, str], int] = {}
        self._completion_reminder_touch_order: dict[tuple[str, str], int] = {}
        self._completion_reminder_next_order = 0

    @staticmethod
    def _get_thread_id(runtime: Runtime) -> str:
        context = getattr(runtime, "context", None)
        thread_id = context.get("thread_id") if context else None
        return str(thread_id) if thread_id else "default"

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        context = getattr(runtime, "context", None)
        run_id = context.get("run_id") if context else None
        return str(run_id) if run_id else "default"

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        return self._get_thread_id(runtime), self._get_run_id(runtime)

    def _touch_completion_reminder_key_locked(self, key: tuple[str, str]) -> None:
        self._completion_reminder_next_order += 1
        self._completion_reminder_touch_order[key] = self._completion_reminder_next_order

    def _completion_reminder_keys_locked(self) -> set[tuple[str, str]]:
        keys = set(self._pending_completion_reminders)
        keys.update(self._completion_reminder_counts)
        keys.update(self._completion_reminder_touch_order)
        return keys

    def _drop_completion_reminder_key_locked(self, key: tuple[str, str]) -> None:
        self._pending_completion_reminders.pop(key, None)
        self._completion_reminder_counts.pop(key, None)
        self._completion_reminder_touch_order.pop(key, None)

    def _prune_completion_reminder_state_locked(self, protected_key: tuple[str, str]) -> None:
        keys = self._completion_reminder_keys_locked()
        overflow = len(keys) - self._MAX_COMPLETION_REMINDER_KEYS
        if overflow <= 0:
            return

        candidates = [key for key in keys if key != protected_key]
        candidates.sort(key=lambda key: self._completion_reminder_touch_order.get(key, 0))
        for key in candidates[:overflow]:
            self._drop_completion_reminder_key_locked(key)

    def _queue_completion_reminder(self, runtime: Runtime, reminder: str) -> None:
        key = self._pending_key(runtime)
        with self._lock:
            self._pending_completion_reminders.setdefault(key, []).append(reminder)
            self._completion_reminder_counts[key] = self._completion_reminder_counts.get(key, 0) + 1
            self._touch_completion_reminder_key_locked(key)
            self._prune_completion_reminder_state_locked(protected_key=key)

    def _completion_reminder_count_for_runtime(self, runtime: Runtime) -> int:
        key = self._pending_key(runtime)
        with self._lock:
            return self._completion_reminder_counts.get(key, 0)

    def _drain_completion_reminders(self, runtime: Runtime) -> list[str]:
        key = self._pending_key(runtime)
        with self._lock:
            reminders = self._pending_completion_reminders.pop(key, [])
            if reminders or key in self._completion_reminder_counts:
                self._touch_completion_reminder_key_locked(key)
            return reminders

    def _clear_other_run_completion_reminders(self, runtime: Runtime) -> None:
        thread_id, current_run_id = self._pending_key(runtime)
        with self._lock:
            for key in self._completion_reminder_keys_locked():
                if key[0] == thread_id and key[1] != current_run_id:
                    self._drop_completion_reminder_key_locked(key)

    def _clear_current_run_completion_reminders(self, runtime: Runtime) -> None:
        key = self._pending_key(runtime)
        with self._lock:
            self._drop_completion_reminder_key_locked(key)

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_other_run_completion_reminders(runtime)
        return None

    @override
    async def abefore_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_other_run_completion_reminders(runtime)
        return None

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """todo 항목이 남아 있을 때 agent가 조기 종료하는 것을 막는다.

        부모 클래스의 병렬 ``write_todos`` 호출 검사에 더해, 미완료 todo가 남은 상태에서 도구 호출이 없는
        model 응답을 가로챈다. reminder ``HumanMessage``를 주입하고 model 노드로 되돌아가 agent가 todo
        목록을 계속 처리하게 한다.

        ``_MAX_COMPLETION_REMINDERS``(기본값 2) 재시도 상한이 있어, agent가 더 진전하지 못할 때 무한
        루프에 빠지지 않는다.
        """
        # 1. 부모 클래스 로직(병렬 write_todos 감지)을 유지한다.
        base_result = super().after_model(state, runtime)
        if base_result is not None:
            return base_result

        # 2. agent가 정상적으로 종료하려 할 때만 개입한다. 도구 호출 의도나 tool-call 파싱 오류는
        # todo reminder로 가리지 말고 도구 경로에서 처리해야 한다.
        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai or _has_tool_call_intent_or_error(last_ai):
            return None

        # 3. 모든 todo가 완료됐거나 todo가 없으면 종료를 허용한다.
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        if not todos or all(t.get("status") == "completed" for t in todos):
            return None

        # 4. 무한 재개입 루프를 막기 위해 reminder 상한을 적용한다.
        if self._completion_reminder_count_for_runtime(runtime) >= self._MAX_COMPLETION_REMINDERS:
            return None

        # 5. 다음 model 요청용 reminder를 큐에 넣고 되돌아간다. 이 제어용 prompt를 일반 HumanMessage로
        # 저장하면 사용자에게 보이는 메시지 스트림과 저장된 대화 기록에 새어 나가므로 저장하지 않는다.
        self._queue_completion_reminder(runtime, _format_completion_reminder(todos))
        return {"jump_to": "model"}

    @override
    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """after_model의 async 버전."""
        return self.after_model(state, runtime)

    @staticmethod
    def _format_pending_completion_reminders(reminders: list[str]) -> str:
        return "\n\n".join(dict.fromkeys(reminders))

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        reminders = self._drain_completion_reminders(request.runtime)
        if not reminders:
            return request
        new_messages = [
            *request.messages,
            HumanMessage(
                content=self._format_pending_completion_reminders(reminders),
                name="todo_completion_reminder",
                additional_kwargs={"hide_from_ui": True},
            ),
        ]
        return request.override(messages=new_messages)

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
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_current_run_completion_reminders(runtime)
        return None

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_current_run_completion_reminders(runtime)
        return None
