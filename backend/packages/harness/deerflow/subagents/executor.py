"""Subagent 실행 엔진."""

import asyncio
import atexit
import logging
import os
import threading
import uuid
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.errors import GraphRecursionError

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.authz.principal import normalize_authz_attributes
from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model
from deerflow.runtime.user_context import DEFAULT_USER_ID
from deerflow.skills.types import Skill
from deerflow.subagents.config import SubagentConfig, resolve_subagent_model_name
from deerflow.subagents.step_events import capture_new_step_messages
from deerflow.subagents.token_collector import SubagentTokenCollector
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY
from deerflow.tracing import build_tracing_callbacks, inject_langfuse_metadata
from deerflow.utils.messages import message_content_to_text

if TYPE_CHECKING:
    # runtime에서는 _build_initial_state 안에서 지연 import한다. tool_search를 즉시
    # import하면 tools/builtins/__init__ -> task_tool ->
    # `from deerflow.subagents import SubagentExecutor` 경로로 아직 초기화 중인 이
    # 패키지에 재진입한다. 여기서는 타입 전용이므로 annotation만 정확하게 유지한다.
    from deerflow.tools.builtins.tool_search import DeferredToolSetup

logger = logging.getLogger(__name__)


_previous_shutdown_isolated_subagent_loop = globals().get("_shutdown_isolated_subagent_loop")
if callable(_previous_shutdown_isolated_subagent_loop):
    atexit.unregister(_previous_shutdown_isolated_subagent_loop)
    _previous_shutdown_isolated_subagent_loop()


class SubagentStatus(Enum):
    """subagent 실행 상태."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in {
            type(self).COMPLETED,
            type(self).FAILED,
            type(self).CANCELLED,
            type(self).TIMED_OUT,
        }


@dataclass
class SubagentResult:
    """subagent 실행 결과.

    Attributes:
        task_id: 이 실행의 고유 식별자.
        trace_id: 분산 tracing용 trace ID(부모와 subagent 로그를 연결한다).
        status: 실행의 현재 상태.
        result: 최종 결과 메시지(완료된 경우).
        error: 에러 메시지(실패한 경우).
        stop_reason: guardrail cap이 run을 조기 종료시킨 이유
            (``token_capped`` / ``turn_capped`` / ``loop_capped``), 정상 종료면
            ``None``. cap이 걸린 run도 상태는 평범하게 유지한다 — 쓸 만한 출력을
            냈으면 ``completed``(부분 결과는 ``result``에 남는다), 아니면
            ``failed`` — 그리고 cap을 여기에 실어서 lead가 "완료"와 "cap 걸림"을
            구분할 수 있게 한다(#3875 Phase 2).
        started_at: 실행 시작 시각.
        completed_at: 실행 완료 시각.
        ai_messages: 실행 중 생성된 완전한 AI 메시지 목록(dict 형태).
    """

    task_id: str
    trace_id: str
    status: SubagentStatus
    result: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    ai_messages: list[dict[str, Any]] | None = None
    token_usage_records: list[dict[str, int | str | None]] = field(default_factory=list)
    usage_reported: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        """가변 기본값을 초기화한다."""
        if self.ai_messages is None:
            self.ai_messages = []

    def update_token_usage_records(self, records: list[dict[str, int | str | None]]) -> None:
        """실행 중인 동안 collector의 최신 누적 snapshot을 공개한다."""
        with self._state_lock:
            if not self.status.is_terminal:
                self.token_usage_records = list(records)

    def try_set_terminal(
        self,
        status: SubagentStatus,
        *,
        result: str | None = None,
        error: str | None = None,
        stop_reason: str | None = None,
        completed_at: datetime | None = None,
        ai_messages: list[dict[str, Any]] | None = None,
        token_usage_records: list[dict[str, int | str | None]] | None = None,
    ) -> bool:
        """terminal 상태를 정확히 한 번만 설정한다.

        백그라운드 timeout/취소와 실행 worker가 같은 result holder를 두고 경쟁할 수
        있다. 먼저 도착한 terminal 전이가 이기고, 뒤늦은 terminal 쓰기는 status나
        payload 필드를 바꾸면 안 된다.
        """
        if not status.is_terminal:
            raise ValueError(f"Status {status} is not terminal")

        with self._state_lock:
            if self.status.is_terminal:
                return False

            if result is not None:
                self.result = result
            if error is not None:
                self.error = error
            if stop_reason is not None:
                self.stop_reason = stop_reason
            if ai_messages is not None:
                self.ai_messages = ai_messages
            if token_usage_records is not None:
                self.token_usage_records = token_usage_records
            self.completed_at = completed_at or datetime.now()
            self.status = status
            return True


def _extract_final_result(final_state: Any, *, trace_id: str, name: str) -> str:
    """stream된 subagent state에서 사람이 읽을 수 있는 결과 문자열을 추출한다.

    대화에서 마지막 ``AIMessage``를 찾아 공용 :func:`message_content_to_text`
    헬퍼로 문자열화한다. AIMessage가 없으면 타입을 가리지 않고 마지막 메시지로
    fallback한다. 추출할 것이 없으면 — 공용 헬퍼가 빈 문자열을 돌려준 경우 포함 —
    sentinel 문자열(``"No response generated"``)을 반환하므로, 호출자가 결과 없음과
    정상적으로 비어 있는 결과를 혼동하지 않는다.

    정상 완료 경로와 max-turns 경로 양쪽에서 쓴다(#3875 Phase 2). ``recursion_limit``
    이 run을 중간에 중단시키면 ``final_state``에는 limit이 걸리기 직전에 stream된
    마지막 chunk가 들어 있으므로, 부분 작업을 버리지 않고 복구한다.
    """
    if final_state is None:
        logger.warning(f"[trace={trace_id}] Subagent {name} no final state")
        return "No response generated"

    messages = final_state.get("messages", [])
    logger.info(f"[trace={trace_id}] Subagent {name} final messages count: {len(messages)}")

    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if last_ai_message is not None:
        text = message_content_to_text(last_ai_message.content)
        return text if text else "No response generated"

    if messages:
        last_message = messages[-1]
        logger.warning(f"[trace={trace_id}] Subagent {name} no AIMessage found, using last message: {type(last_message)}")
        raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        text = message_content_to_text(raw_content)
        return text if text else "No response generated"

    logger.warning(f"[trace={trace_id}] Subagent {name} no messages in final state")
    return "No response generated"


def _extract_llm_error_fallback(final_state: Any) -> str | None:
    """terminal LLM fallback 메시지의 사용자 노출용 에러를 반환한다.

    ``LLMErrorHandlingMiddleware``는 provider 예외를 표시가 붙은 ``AIMessage``로
    변환해 graph가 깔끔하게 종료되도록 한다. 하지만 graph가 깔끔히 끝났다고 해서
    작업이 성공한 것은 아니다. subagent 호출자는 그 구조적 marker를 기존의 실패
    terminal 상태로 번역받아야 한다.

    권위를 갖는 것은 마지막 assistant 메시지뿐이고, 전체가 아니라 tail만 훑는 것도
    의도적이다. subagent는 부모의 ``thread_id``를 공유하고(``_aexecute``의
    ``run_config`` 참고), LangGraph는 ``stream_mode="values"``로 부모 메시지 히스토리
    전체를 replay하므로 ``final_state``에는 이전 부모 턴이 남긴 *오래된* fallback
    marker가 들어 있을 수 있다. lead-agent run 경로는 모든 메시지를 훑기 때문에
    ``pre_existing_message_ids``로 그런 오래된 marker를 가려야 한다
    (``runtime/runs/worker.py::_extract_llm_error_fallback_message``). 여기서는 그럴
    필요가 없다. fallback ``AIMessage``에는 ``tool_calls``가 없으므로 항상 run을
    종료시키고, subagent는 최소한 자기 자신의 terminal assistant 메시지를 항상
    덧붙인다 — 따라서 마지막 ``AIMessage``는 결코 오래된 부모 히스토리 marker가 아니다.
    모든 메시지를 훑도록 "고치지" 말 것. worker.py가 막고 있는 stale-marker 오탐이
    다시 생긴다.

    marker 없이 에러처럼 보이는 메시지 텍스트는 평범한 출력으로 취급한다.
    """
    if final_state is None:
        return None

    for message in reversed(final_state.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        metadata = message.additional_kwargs
        if metadata.get("deerflow_error_fallback") is not True:
            return None

        content = message_content_to_text(message.content).strip()
        if content:
            return content

        # 방어적 처리: ``_build_error_fallback_message``는 항상 비어 있지 않은 사용자
        # 노출 ``content``를 설정한다(``error_detail``도 ``_extract_error_detail``을
        # 통해 설정되며, 실패 시 예외 클래스명으로 fallback한다). 아래 분기는 빈
        # fallback을 내보내는 미래의 middleware에 대비한 안전장치일 뿐이다.
        detail = metadata.get("error_detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return "LLM request failed"

    return None


# 백그라운드 task 결과를 담는 전역 저장소
_background_tasks: dict[str, SubagentResult] = {}
_background_tasks_lock = threading.Lock()

# 백그라운드 task 스케줄링 및 오케스트레이션용 thread pool
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# 이미 실행 중인 부모 loop에서 시작된 격리 subagent 실행에 쓰는 상시 event loop.
# 오래 사는 loop 하나를 재사용하면 실행마다 새 loop를 만들었다가 거기에 묶인 async
# 리소스를 닫아야 하는 문제를 피할 수 있다.
_isolated_subagent_loop: asyncio.AbstractEventLoop | None = None
_isolated_subagent_loop_thread: threading.Thread | None = None
_isolated_subagent_loop_started: threading.Event | None = None
_isolated_subagent_loop_lock = threading.Lock()


def _run_isolated_subagent_loop(
    loop: asyncio.AbstractEventLoop,
    started_event: threading.Event,
) -> None:
    """상시 격리 subagent loop를 전용 daemon thread에서 실행한다."""
    asyncio.set_event_loop(loop)
    loop.call_soon(started_event.set)
    try:
        loop.run_forever()
    finally:
        started_event.clear()


def _shutdown_isolated_subagent_loop() -> None:
    """상시 격리 subagent loop를 중지하고 닫는다."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started

    with _isolated_subagent_loop_lock:
        loop = _isolated_subagent_loop
        thread = _isolated_subagent_loop_thread
        _isolated_subagent_loop = None
        _isolated_subagent_loop_thread = None
        _isolated_subagent_loop_started = None

    if loop is None:
        return

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1)

    thread_stopped = thread is None or not thread.is_alive()
    loop_stopped = not loop.is_running()

    if not loop.is_closed():
        if thread_stopped and loop_stopped:
            loop.close()
        else:
            logger.warning(
                "Skipping close of isolated subagent loop because shutdown did not complete within timeout (thread_alive=%s, loop_running=%s)",
                thread is not None and thread.is_alive(),
                loop.is_running(),
            )


atexit.register(_shutdown_isolated_subagent_loop)


def _get_isolated_subagent_loop() -> asyncio.AbstractEventLoop:
    """격리 subagent 실행에 쓰는 상시 event loop를 반환한다."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started
    with _isolated_subagent_loop_lock:
        thread_is_alive = _isolated_subagent_loop_thread is not None and _isolated_subagent_loop_thread.is_alive()
        loop_is_usable = _isolated_subagent_loop is not None and not _isolated_subagent_loop.is_closed() and _isolated_subagent_loop.is_running() and thread_is_alive

        if not loop_is_usable:
            loop = asyncio.new_event_loop()
            started_event = threading.Event()
            thread = threading.Thread(
                target=_run_isolated_subagent_loop,
                args=(loop, started_event),
                name="subagent-persistent-loop",
                daemon=True,
            )
            thread.start()
            if not started_event.wait(timeout=5):
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=1)
                loop.close()
                raise RuntimeError("Timed out starting isolated subagent event loop")
            _isolated_subagent_loop = loop
            _isolated_subagent_loop_thread = thread
            _isolated_subagent_loop_started = started_event

        if _isolated_subagent_loop is None:
            raise RuntimeError("Isolated subagent event loop is not initialized")
        return _isolated_subagent_loop


def _submit_to_isolated_loop_in_context(
    context: Context,
    coro_factory: Callable[[], Coroutine[Any, Any, SubagentResult]],
) -> Future[SubagentResult]:
    """ContextVar state를 보존한 채 격리 loop에 coroutine을 제출한다."""
    return context.run(
        lambda: asyncio.run_coroutine_threadsafe(
            coro_factory(),
            _get_isolated_subagent_loop(),
        )
    )


def _copy_isolated_subagent_context() -> Context:
    """loop에 묶인 부모 graph callback을 제외하고 주변 context를 복사한다.

    LangGraph는 현재 runnable config를 ``ContextVar``에 보관한다. 상시 subagent
    loop로 넘어갈 때도 checkpoint lineage, runtime metadata, 사용자 identity,
    tracing context는 유지되어야 한다. LangGraph는 상속된 callback과 명시적
    callback을 병합하므로 subagent collector만 넘기는 것으로는 부족하다. 부모
    ``RunJournal``처럼 loop에 묶인 애플리케이션 callback이 격리 loop에서 그대로
    실행되기 때문이다. framework streaming callback은 namespace가 붙은 자식 token
    frame이 계속 부모 stream에 도달하도록 의도적으로 남긴다.
    """
    context = copy_context()
    inherited_config = context.get(var_child_runnable_config)
    if inherited_config is None or "callbacks" not in inherited_config:
        return context

    callbacks = inherited_config.get("callbacks")
    if isinstance(callbacks, BaseCallbackManager):
        isolated_callbacks = callbacks.copy()
        isolated_callbacks.handlers = [handler for handler in callbacks.handlers if not getattr(handler, "deerflow_loop_bound", False)]
        isolated_callbacks.inheritable_handlers = [handler for handler in callbacks.inheritable_handlers if not getattr(handler, "deerflow_loop_bound", False)]
    elif isinstance(callbacks, (list, tuple)):
        isolated_callbacks = [handler for handler in callbacks if not getattr(handler, "deerflow_loop_bound", False)]
    elif getattr(callbacks, "deerflow_loop_bound", False):
        isolated_callbacks = None
    else:
        isolated_callbacks = callbacks

    isolated_config = inherited_config.copy()
    if isolated_callbacks:
        isolated_config["callbacks"] = isolated_callbacks
    else:
        isolated_config.pop("callbacks", None)
    context.run(var_child_runnable_config.set, isolated_config)
    return context


def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """subagent 설정에 따라 tool을 필터링한다.

    Args:
        all_tools: 사용 가능한 전체 tool 목록.
        allowed: 선택적 tool 이름 allowlist. 주어지면 이 tool들만 포함한다.
        disallowed: 선택적 tool 이름 denylist. 이 tool들은 항상 제외한다.

    Returns:
        필터링된 tool 목록.
    """
    filtered = all_tools

    # allowlist가 지정된 경우 적용
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]

    # denylist 적용
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


class SubagentExecutor:
    """subagent를 실행하는 executor."""

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        app_config: AppConfig | None = None,
        parent_model: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        user_role: str | None = None,
        oauth_provider: str | None = None,
        oauth_id: str | None = None,
        run_id: str | None = None,
        channel_user_id: str | None = None,
        is_internal: bool = False,
        authz_attributes: Mapping[str, Any] | None = None,
        deerflow_trace_id: str | None = None,
        extensions: Any | None = None,
    ):
        """executor를 초기화한다.

        Args:
            config: subagent 설정.
            tools: 사용 가능한 전체 tool 목록(필터링된다).
            app_config: 해석된 AppConfig. None이면 ``_create_agent``가
                ``get_app_config()``로 fallback한다(lead-agent factory와 같은 패턴).
            parent_model: 상속에 쓸 부모 agent의 모델 이름.
            sandbox_state: 부모 agent에서 받은 sandbox state.
            thread_data: 부모 agent에서 받은 thread data.
            thread_id: sandbox 작업에 쓰는 Thread ID.
            trace_id: 분산 tracing용으로 부모에게서 받은 Trace ID.
            user_id: 부모 tool의 runtime context에서 캡처한 User ID.
                None이면 tracing 계층이 DEFAULT_USER_ID로 fallback한다.
            user_role: 인증된 사용자의 role. subagent의 GuardrailMiddleware가 위임된
                호출에 role 기반 정책을 적용할 수 있도록 전파한다.
            oauth_provider: SSO로 인증한 경우의 외부 identity provider.
            oauth_id: 외부 identity provider에서의 subject id.
            run_id: 부모 run id. 위임된 guardrail 판정이 lead agent와 같은 run에
                귀속되도록 한다.
            deerflow_trace_id: Langfuse metadata 상관관계를 위해 부모 run에서 전파된
                DeerFlow 요청 수준 correlation id.
            extensions: ``task_tool`` dispatch 시점에 캡처한 부모 run의 불변
                ``LoadedExtensions`` snapshot. None이면(embedded client, 단독
                LangGraph Server) ``_aexecute``가 프로세스 전역 singleton으로
                fallback한다.
        """
        self.config = config
        self.app_config = app_config
        self.parent_model = parent_model
        # config.yaml 로드가 필요 없을 때만 즉시 해석한다. 그 외에는 이미 app_config를
        # 로드하는 _create_agent로 미뤄서, config 파일 없이도 단위 테스트가 executor를
        # 생성할 수 있게 한다.
        if config.model != "inherit" or parent_model is not None or app_config is not None:
            self.model_name: str | None = resolve_subagent_model_name(config, parent_model, app_config=app_config)
        else:
            self.model_name = None
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        # 주어지지 않으면 trace_id를 생성한다(최상위 호출용)
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.user_id = user_id
        # 부모 runtime context에서 전파된 guardrail 귀속 정보.
        self.user_role = user_role
        self.oauth_provider = oauth_provider
        self.oauth_id = oauth_id
        self.run_id = run_id
        # task_tool dispatch 시점에 캡처한 IM 채널 발신자 identity. 그룹 채팅은 여러
        # 발신자가 하나의 thread를 공유하므로, 위임된 bash 명령은 아무것도 내보내지
        # 않는 대신 dispatch한 턴의 id를 export해야 한다.
        self.channel_user_id = channel_user_id
        # 부모 runtime context에서 전파된 authorization identity.
        # subagent의 GuardrailMiddleware가 lead와 동일한 provenance를 보도록
        # is_internal은 (False라도) 무조건 기록한다.
        self.is_internal = is_internal
        self.authz_attributes = normalize_authz_attributes(authz_attributes)
        self.deerflow_trace_id = deerflow_trace_id
        # 부모 run의 extension snapshot. 실행 시점에 singleton을 읽는 대신 여기서
        # 묶어두는 것이 하나의 run을 단일 extension 세대에 고정하는 방법이다.
        # lead run 시작과 이 subagent 실행 사이에 동시 실행된
        # ``set_loaded_extensions()``가 위임된 작업 밑에서 세대를 바꿔치기하면 안 된다.
        self.extensions = extensions

        self._base_tools = _filter_tools(
            tools,
            config.tools,
            config.disallowed_tools,
        )
        self.tools = self._base_tools
        # prompt를 만들 때 쓰는 것과 동일한 사용자별·config 필터링된 registry에서
        # 채운다. runtime skill activation/policy middleware가 정확히 이 집합을
        # 받으므로 subagent가 공개되지 않은 skill을 활성화할 수 없다.
        self._available_skill_names: set[str] = set()
        # ``consume_stop_reason``을 노출하는 guard middleware들(현재
        # ``TokenBudgetMiddleware``와 ``LoopDetectionMiddleware``). ``_create_agent``에서
        # 수집해서 ``_aexecute``가 run 종료 후 각각을 읽고 어떤 cap이 걸렸는지
        # (token_capped / loop_capped) lead에 전달할 수 있게 한다(#3875 Phase 2).
        # v2 contract가 cap 사유를 둘 이상 규정하므로 첫 번째만이 아니라 모든 guard를
        # 확인해야 하며, 그래서 list로 모은다.
        self._stop_reason_middlewares: list[Any] = []

        logger.info(f"[trace={self.trace_id}] SubagentExecutor initialized: {config.name} with {len(self.tools)} tools")

    def _create_agent(
        self,
        tools: list[BaseTool] | None = None,
        *,
        deferred_setup: "DeferredToolSetup | None" = None,
        extensions=None,
    ):
        """agent 인스턴스를 생성한다.

        ``deferred_setup``(``_build_initial_state``에서 조립된다)은 deferred MCP tool
        이름과 catalog hash를 담고 있어, subagent도 lead agent와 동일한
        DeferredToolFilterMiddleware를 갖게 한다. ``None``이면 아무 일도 하지 않는다.
        """
        app_config = self.app_config or get_app_config()
        if self.model_name is None:
            self.model_name = resolve_subagent_model_name(self.config, self.parent_model, app_config=app_config)
        model = create_chat_model(name=self.model_name, thinking_enabled=False, app_config=app_config, attach_tracing=False)

        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # lead agent와 공용 middleware 구성을 재사용한다. ``agent_name``이 있으면
        # builder가 agent별 token_budget 오버라이드를 해석할 수 있다.
        mcp_routing_middleware = None
        if deferred_setup is not None and deferred_setup.deferred_names:
            from deerflow.tools.builtins.tool_search import build_mcp_routing_middleware

            mcp_routing_middleware = build_mcp_routing_middleware(
                tools if tools is not None else self.tools,
                deferred_setup,
                top_k=app_config.tool_search.auto_promote_top_k,
            )
        middleware_kwargs = {
            "app_config": app_config,
            "model_name": self.model_name,
            "lazy_init": True,
            "deferred_setup": deferred_setup,
            "agent_name": self.config.name,
            "available_skills": self._available_skill_names,
            "user_id": self.user_id or DEFAULT_USER_ID,
        }
        if extensions is not None:
            middleware_kwargs["extensions"] = extensions
        authz_provider = getattr(self, "_authz_provider", None)
        if authz_provider is not None:
            middleware_kwargs["authorization_provider"] = authz_provider
        if mcp_routing_middleware is not None:
            middleware_kwargs["mcp_routing_middleware"] = mcp_routing_middleware
        middlewares = build_subagent_runtime_middlewares(**middleware_kwargs)
        # ``consume_stop_reason``을 노출하는 guard middleware를 모두 수집해서
        # (TokenBudgetMiddleware, LoopDetectionMiddleware) _aexecute가 run 후 각각을
        # 읽고 어떤 cap이 걸렸는지 전달할 수 있게 한다. ``hasattr`` 기반 duck typing이라
        # 이 파일은 middleware 클래스를 import할 필요가 없다. ``next(...)``가 아니라
        # list로 모으므로 모든 guard가 검사되고, 나중에 추가된 guard도 자동으로 잡힌다.
        self._stop_reason_middlewares = [m for m in middlewares if hasattr(m, "consume_stop_reason")]

        # 일부 LLM API가 여러 개의 SystemMessage를 지원하지 않으므로, system_prompt는
        # 초기 state 메시지에 포함시킨다(_build_initial_state 참고).
        return create_agent(
            model=model,
            tools=tools if tools is not None else self.tools,
            middleware=middlewares,
            system_prompt=None,
            state_schema=ThreadState,
            checkpointer=False,
        )

    def _consume_guard_stop_reason(self) -> str | None:
        """직전 run 중에 설정된 guard cap stop reason을 꺼내서 반환한다.

        ``consume_stop_reason``을 노출하는 모든 guard middleware(:meth:`_create_agent`
        에서 수집)를 확인하고 ``None``이 아닌 첫 번째 사유를 반환한다 — token budget
        hard stop이면 ``"token_capped"``, loop 감지로 중단됐으면 ``"loop_capped"``,
        아니면 ``None``. 각 guard의 cap은 예외를 던지지 않고(run은 최종 답변과 함께
        완료된다) 따라서 executor는 이 경로로만 완료가 실제로 cap 걸렸는지 알 수 있다.
        보통 run당 많아야 하나의 guard만 발동하지만, 전부 확인해야 contract의 cap 어휘
        전체가 도달 가능해진다.
        """
        for mw in self._stop_reason_middlewares:
            reason = mw.consume_stop_reason(self.run_id)
            if reason is not None:
                return reason
        return None

    async def _load_skills(self) -> list[Skill]:
        """config.skills를 기준으로 활성화된 skill 메타데이터를 로드한다."""
        if self.config.skills is not None and len(self.config.skills) == 0:
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} skills=[] — skipping skill loading")
            return []

        try:
            from deerflow.skills.storage import get_or_new_user_skill_storage

            storage_kwargs = {"app_config": self.app_config} if self.app_config is not None else {}
            storage = await asyncio.to_thread(
                get_or_new_user_skill_storage,
                self.user_id or DEFAULT_USER_ID,
                **storage_kwargs,
            )
            # event loop를 막지 않도록 asyncio.to_thread를 쓴다(LangGraph ASGI 요구사항)
            all_skills = await asyncio.to_thread(storage.load_skills, enabled_only=True)
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} loaded {len(all_skills)} enabled skills from disk")
        except Exception:
            logger.exception(f"[trace={self.trace_id}] Failed to load skills for subagent {self.config.name}")
            raise

        if not all_skills:
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} no enabled skills found")
            return []

        # config.skills whitelist로 필터링
        if self.config.skills is not None:
            allowed = set(self.config.skills)
            return [s for s in all_skills if s.name in allowed]
        return all_skills

    async def _build_initial_state(self, task: str) -> tuple[dict[str, Any], list[BaseTool], "DeferredToolSetup"]:
        """agent 실행을 위한 초기 state를 만든다.

        Args:
            task: 작업 설명.

        Returns:
            ``(state, final_tools, deferred_setup)``. ``final_tools``는 인가된 tool
            목록이며, 각 deferral 모드가 적용되면 discovery 헬퍼가 덧붙는다.
            ``deferred_setup``은 ``_create_agent``가 소비하므로 agent 빌드와 주입된
            ``<available-deferred-tools>`` 섹션이 하나의 catalog/hash를 공유한다.
        """
        # 지연 import: 이 모듈 상단의 TYPE_CHECKING 주석 참고. tool_search를 import하면
        # tools/builtins/__init__이 실행되어 초기화 중인 이 패키지에 재진입한다.
        from deerflow.tools.builtins.tool_search import assemble_deferred_tools, get_deferred_tools_prompt_section, get_mcp_routing_hints_prompt_section

        # skill은 명시적 slash 활성화나 read_file 로드 전까지는 탐색 가능한 메타데이터일
        # 뿐이다. allowed-tools 선언은 여기서 즉시가 아니라 SkillToolPolicyMiddleware가
        # 동적으로 적용한다.
        skills = await self._load_skills()
        self._available_skill_names = {skill.name for skill in skills}

        resolved_app_config = self.app_config or get_app_config()

        from deerflow.skills.describe import build_skill_search_setup, get_skill_index_prompt_section

        skill_setup = build_skill_search_setup(
            skills,
            enabled=resolved_app_config.skills.deferred_discovery,
            container_base_path=resolved_app_config.skills.container_path,
        )

        # authorization Layer 1 적용: deferred 조립 전에 tool을 걸러서 거부된 tool이
        # DeferredToolCatalog에 절대 들어가지 못하게 한다.
        from deerflow.authz.tool_filter import apply_tool_authorization

        authz_context = {
            "user_id": self.user_id,
            "user_role": self.user_role,
            "oauth_provider": self.oauth_provider,
            "oauth_id": self.oauth_id,
            "channel_user_id": self.channel_user_id,
            "is_internal": self.is_internal,
            "authz_attributes": self.authz_attributes,
        }
        authorization_candidates = [*self._base_tools]
        if skill_setup.describe_skill_tool is not None:
            authorization_candidates.append(skill_setup.describe_skill_tool)
        configured_tool_ids = {id(tool) for tool in self._base_tools}
        authorized_tools, self._authz_provider = apply_tool_authorization(
            authorization_candidates,
            context=authz_context,
            app_config=resolved_app_config,
        )
        configured_tools = [tool for tool in authorized_tools if id(tool) in configured_tool_ids]
        late_tools = [tool for tool in authorized_tools if id(tool) not in configured_tool_ids]

        # subagent의 이름 allow/deny 및 authorization 필터를 거친 뒤 deferred
        # tool_search를 조립한다. lead 경로와 동일하게 맞춰서 subagent도 전체 MCP
        # schema를 바인딩하지 않게 한다.
        # 생성된 tool_search 헬퍼는 의도적으로 subagent의 이름 수준 allow/deny
        # (config.tools / disallowed_tools) 대상이 아니다. 그 catalog 자체가 이미
        # 필터링된 목록에서 만들어지기 때문이다. 활성 skill 정책은 이후 middleware가
        # schema 가시성과 실행 양쪽에 적용하므로, promotion이 활성 skill의 권한을
        # 넓힐 수는 없다.
        final_tools, deferred_setup = assemble_deferred_tools(
            configured_tools,
            enabled=resolved_app_config.tool_search.enabled,
        )
        final_tools.extend(late_tools)

        # system prompt와 skill discovery 메타데이터를 하나의 SystemMessage로 합친다.
        # SKILL.md 본문 전체는 활성화될 때만 로드한다. 일부 LLM API는 여러 개의
        # SystemMessage를 "System message must be at the beginning."으로 거부한다.
        system_parts: list[str] = []
        if self.config.system_prompt:
            system_parts.append(self.config.system_prompt)
        if skills:
            if skill_setup.skill_names:
                skills_section = get_skill_index_prompt_section(
                    skill_names=skill_setup.skill_names,
                    container_base_path=resolved_app_config.skills.container_path,
                )
            else:
                # legacy discovery 모드에서는 lead agent의 메타데이터 렌더러를
                # 재사용해 두 agent 유형이 같은 skill catalog를 기술하게 한다.
                from deerflow.agents.lead_agent.prompt import get_skills_prompt_section

                skills_section = await asyncio.to_thread(
                    get_skills_prompt_section,
                    self._available_skill_names,
                    app_config=resolved_app_config,
                    user_id=self.user_id or DEFAULT_USER_ID,
                )
            if skills_section:
                system_parts.append(skills_section)
        # deferred MCP tool의 이름만 prompt에 적는다. schema는 tool_search가 promote할
        # 때까지 감춰둔다. 빈 집합이면 "" 이 되어 아무것도 덧붙지 않는다.
        deferred_section = get_deferred_tools_prompt_section(deferred_names=deferred_setup.deferred_names)
        if deferred_section:
            system_parts.append(deferred_section)
        mcp_routing_hints_section = get_mcp_routing_hints_prompt_section(authorized_tools, deferred_names=deferred_setup.deferred_names)
        if mcp_routing_hints_section:
            system_parts.append(mcp_routing_hints_section)

        messages: list[Any] = []
        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))

        # 그다음 실제 task
        messages.append(HumanMessage(content=task))

        state: dict[str, Any] = {
            "messages": messages,
        }

        # 부모의 sandbox와 thread data를 그대로 전달
        if self.sandbox_state is not None:
            state["sandbox"] = self.sandbox_state
        if self.thread_data is not None:
            state["thread_data"] = self.thread_data

        return state, final_tools, deferred_setup

    async def _aexecute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """task를 비동기로 실행한다.

        Args:
            task: subagent에 줄 작업 설명.
            result_holder: 실행 중 갱신할, 미리 생성된 선택적 result 객체.

        Returns:
            실행 결과를 담은 SubagentResult.
        """
        if result_holder is not None:
            # 주어진 result holder를 사용한다(실시간 갱신이 필요한 비동기 실행용)
            result = result_holder
        else:
            # 동기 실행용으로 새 result를 만든다
            task_id = str(uuid.uuid4())[:8]
            result = SubagentResult(
                task_id=task_id,
                trace_id=self.trace_id,
                status=SubagentStatus.RUNNING,
                started_at=datetime.now(),
            )
        from deerflow.extensions import get_loaded_extensions

        loaded_extensions = self.extensions if self.extensions is not None else get_loaded_extensions()
        task_store = None
        if loaded_extensions.needs_task_store:
            from deerflow_extension_api import ExtensionData

            task_store = ExtensionData(result.task_id)
        ai_messages = result.ai_messages
        if ai_messages is None:
            ai_messages = []
            result.ai_messages = ai_messages
        # stream된 AI 메시지의 O(1) 중복 검출. ``stream_mode="values"``는 super-step마다
        # 전체 state를 다시 내보내므로 같은 마지막 메시지가 chunk마다 재검사된다.
        # id를 키로 하는 set을 쓰면 append-only ``ai_messages`` 목록을 다시 훑는 대신
        # (chunk당 O(n) -> run 전체로 O(n^2), deep-research subagent는 max_turns=150에
        # 도달한다) 이 검사를 O(1)로 유지할 수 있다.
        seen_message_ids: set[str] = {mid for msg in ai_messages if (mid := msg.get("id"))}
        # append-only 메시지 히스토리에 대한 cursor. ``values`` 모드 chunk마다 새로
        # 추가된 tail만 다시 훑게 한다(capture_new_step_messages 참고).
        processed_message_count = 0

        collector: SubagentTokenCollector | None = None
        try:
            state, final_tools, deferred_setup = await self._build_initial_state(task)
            agent = self._create_agent(
                final_tools,
                deferred_setup=deferred_setup,
                extensions=loaded_extensions,
            )

            # subagent LLM 호출용 token collector
            collector_caller = f"subagent:{self.config.name}"
            collector = SubagentTokenCollector(caller=collector_caller)

            # checkpoint 좌표(thread_id/checkpoint_ns 등)를 자식 config에 넣지 않는다.
            # LangGraph가 주변 부모 run에서 그 좌표를 상속하므로 이 실행이 subgraph
            # namespace를 유지한다. 비즈니스 소비자는 대신 아래 ``context``로
            # thread_id를 받는다.
            run_config: RunnableConfig = {
                "recursion_limit": self.config.max_turns,
                "callbacks": [collector],
                "tags": [collector_caller],
            }

            # tracing callback을 graph 수준에 주입해서, subagent run 하나가 모든
            # node / LLM / tool 호출을 자식 span으로 갖는 trace 하나를 만들게 한다.
            # lead agent 패턴과 동일하다. graph 수준 tracing + 모델의
            # attach_tracing=False 조합이 trace 이중 집계를 막는다.
            tracing_callbacks = build_tracing_callbacks()
            if tracing_callbacks:
                existing_callbacks = list(run_config.get("callbacks") or [])
                run_config["callbacks"] = [*existing_callbacks, *tracing_callbacks]

            # tracing용으로 subagent 이름을 정규화해서 lead-agent 이름 형태(소문자,
            # 하이픈만)에 맞춘다. 공용 헬퍼가 없어 인라인으로 처리한다 —
            # runtime/runs/naming.py는 lead-agent run만 다룬다.
            if self.config.name:
                normalized_name = self.config.name.strip().lower().replace("_", "-")
                assistant_id = f"subagent:{normalized_name}"
            else:
                assistant_id = "subagent"

            # Langfuse trace 속성 메타데이터를 주입해서 subagent trace가 부모 thread에
            # 연결되고 올바른 session/user ID를 갖게 한다.
            inject_langfuse_metadata(
                run_config,
                thread_id=self.thread_id,
                user_id=self.user_id,
                assistant_id=assistant_id,
                model_name=self.model_name,
                environment=os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT"),
                deerflow_trace_id=self.deerflow_trace_id,
            )

            context: dict[str, Any] = {}
            if self.thread_id:
                context["thread_id"] = self.thread_id
            if self.app_config is not None:
                context["app_config"] = self.app_config
            # guardrail 귀속 정보를 전파해서 위임된 tool 호출이 부모 run의 identity로
            # 평가되게 한다(role 기반 정책, audit). user_id는 해석된 tracing id를
            # 재사용하며, 모든 인증/IM 경로에서 부모 context 값과 같다.
            context["user_id"] = self.user_id
            context["user_role"] = self.user_role
            context["oauth_provider"] = self.oauth_provider
            context["oauth_id"] = self.oauth_id
            context["run_id"] = self.run_id
            if task_store is not None:
                from deerflow_extension_api import EXTENSION_TASK_STORE_KEY

                context[EXTENSION_TASK_STORE_KEY] = task_store
            if self.channel_user_id:
                context["channel_user_id"] = self.channel_user_id
            # authorization identity: is_internal은 (False라도) 무조건 기록하고,
            # attributes는 write-back 시 다시 복사한다.
            context["is_internal"] = self.is_internal
            context["authz_attributes"] = dict(self.authz_attributes)
            if self.deerflow_trace_id:
                context[DEERFLOW_TRACE_METADATA_KEY] = self.deerflow_trace_id
            context["is_subagent"] = True

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution with max_turns={self.config.max_turns}")

            # 실시간 갱신을 받기 위해 invoke 대신 stream을 쓴다.
            # 덕분에 AI 메시지를 생성되는 대로 수집할 수 있다.
            final_state = None

            # 사전 확인: streaming 시작 전에 이미 취소됐다면 즉시 빠져나온다
            if result.cancel_event.is_set():
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled before streaming")
                result.try_set_terminal(
                    SubagentStatus.CANCELLED,
                    error="Cancelled by user",
                    token_usage_records=collector.snapshot_records(),
                )
                return result

            async for chunk in agent.astream(state, config=run_config, context=context, stream_mode="values"):  # type: ignore[arg-type]
                # 협조적 취소: 부모가 중지를 요청했는지 확인한다.
                # 취소는 astream 반복 경계에서만 감지되므로, 한 반복 안에서 오래
                # 걸리는 tool 호출은 다음 chunk가 나올 때까지 중단되지 않는다.
                if result.cancel_event.is_set():
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled by parent")
                    result.try_set_terminal(
                        SubagentStatus.CANCELLED,
                        error="Cancelled by user",
                        token_usage_records=collector.snapshot_records(),
                    )
                    return result

                final_state = chunk
                result.update_token_usage_records(collector.snapshot_records())

                # 직전 chunk 이후 추가된 step 메시지를 모두 캡처한다(assistant 턴과
                # tool 출력 둘 다). 모델이 한 턴에 여러 tool 호출을 내면 하나의
                # super-step이 여러 ToolMessage를 추가할 수 있으므로, messages[-1]만
                # 캡처하면 마지막을 제외한 출력이 전부 사라진다(#3779).
                # 중복 제거와 직렬화는 capture_step_message에 있다.
                messages = chunk.get("messages", [])
                previous_count = len(ai_messages)
                processed_message_count = capture_new_step_messages(messages, ai_messages, seen_message_ids, processed_message_count)
                if len(ai_messages) > previous_count:
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} captured {len(ai_messages) - previous_count} step message(s); total #{len(ai_messages)}")

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} completed async execution")
            token_usage_records = collector.snapshot_records()
            llm_error = _extract_llm_error_fallback(final_state)
            if llm_error is not None:
                result.try_set_terminal(
                    SubagentStatus.FAILED,
                    error=llm_error,
                    token_usage_records=token_usage_records,
                )
            else:
                final_result = _extract_final_result(final_state, trace_id=self.trace_id, name=self.config.name)
                # guard hard-stop(token budget 또는 loop 감지)은 예외를 던지지 않는다
                # — tool_calls를 제거해서 run이 최종 답변과 함께 완료되게 한다.
                # 각 guard의 ``consume_stop_reason``이 그 일이 있었는지 알려주므로,
                # 완료된 결과에 cap 사유(token_capped / loop_capped)를 표시해 lead에
                # 전달할 수 있다(#3875 Phase 2). 사유를 pop하므로 실제로 소비하는
                # 분기에 두어야 한다 — fallback에는 tool_calls가 없으니 FAILED 분기에
                # guard hard-stop이 함께 발생했을 수는 없다.
                stop_reason = self._consume_guard_stop_reason()
                result.try_set_terminal(
                    SubagentStatus.COMPLETED,
                    result=final_result,
                    stop_reason=stop_reason,
                    token_usage_records=token_usage_records,
                )

        except GraphRecursionError:
            # run_config의 ``recursion_limit``은 위에서 ``self.config.max_turns``로
            # 설정된다. 여기에 걸렸다는 것은 subagent가 턴 예산을 소진했다는 뜻이다.
            # 전용 status enum(v1 contract 소비자를 깨뜨린다) 대신 추가형
            # ``stop_reason`` 채널로 보낸다(#3875 Phase 2). run이 쓸 만한 부분 결과를
            # stream했다면 ``completed``로, 아니면 ``failed``로 노출한다. 어느 쪽이든
            # lead는 결과 텍스트를 파싱하지 않고도 "예산 소진"과 "고장난 subagent"를
            # 구분할 수 있다.
            #
            # 이번 run에서 이미 guard가 발동했다면 그 stop reason을 우선한다.
            # token budget / loop hard-stop은 최종 답변을 강제하려고 tool_calls를
            # 제거하는데, 그 답변이 나오기 전 다음 super-step에서 ``recursion_limit``이
            # 걸렸다면 실제 제약은 턴 예산이 아니라 guard였다. 여기서 guard를 확인하면
            # (위 정상 완료 경로와 동일한 조회) 두 경로가 일관되고, 사유를 pop하므로
            # dict에 고아로 남지 않는다.
            max_turns = self.config.max_turns
            logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} reached max_turns={max_turns} (GraphRecursionError); recovering partial result")
            records = collector.snapshot_records() if collector is not None else None
            stop_reason = self._consume_guard_stop_reason() or "turn_capped"

            # 처리된 LLM provider 실패(#4042)는 진짜 부분 출력과 마찬가지로 terminal
            # ``AIMessage``에 비어 있지 않은 사용자 노출 텍스트를 싣는다. 따라서 여기서도
            # 확인해야 하며, 그러지 않으면 아래 raw 텍스트 스캔과 구분되지 않아 완료된
            # 작업으로 잘못 분류된다. 그 스캔으로 넘어가기 전에, 위 정상 완료 경로가
            # 쓰는 것과 동일한 marker를 확인한다.
            llm_error = _extract_llm_error_fallback(final_state)
            if llm_error is not None:
                result.try_set_terminal(
                    SubagentStatus.FAILED,
                    error=llm_error,
                    stop_reason=stop_reason,
                    token_usage_records=records,
                )
            else:
                messages = (final_state or {}).get("messages", [])
                usable_partial: str | None = None
                for m in reversed(messages):
                    if isinstance(m, AIMessage):
                        text = message_content_to_text(m.content).strip()
                        if text:
                            usable_partial = text
                        break
                if usable_partial is not None:
                    result.try_set_terminal(
                        SubagentStatus.COMPLETED,
                        result=usable_partial,
                        stop_reason=stop_reason,
                        token_usage_records=records,
                    )
                else:
                    result.try_set_terminal(
                        SubagentStatus.FAILED,
                        error=f"Reached max_turns={max_turns}",
                        stop_reason=stop_reason,
                        token_usage_records=records,
                    )

        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            result.try_set_terminal(
                SubagentStatus.FAILED,
                error=str(e),
                token_usage_records=collector.snapshot_records() if collector is not None else None,
            )

        return result

    def _execute_in_isolated_loop(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """상시 격리 event loop에서 subagent를 실행한다.

        호출자가 이미 event loop 안에서 돌고 있을 때 동기 ``execute()`` 경로가 이
        메서드를 쓴다. ``execute()``가 동기 API이므로 실제 coroutine이 오래 사는 격리
        loop에서 도는 동안 호출자를 블로킹한다. 그 loop를 재사용하면 공유 async
        client가 실행마다 닫히는 단명 loop에 묶이지 않는다.
        """
        future: Future[SubagentResult] | None = None
        parent_context = _copy_isolated_subagent_context()
        try:
            future = _submit_to_isolated_loop_in_context(
                parent_context,
                lambda: self._aexecute(task, result_holder),
            )
            return future.result(timeout=self.config.timeout_seconds)
        except FuturesTimeoutError:
            if result_holder is not None:
                result_holder.cancel_event.set()
            if future is not None:
                future.cancel()
            raise
        except Exception:
            if future is None:
                logger.debug(
                    f"[trace={self.trace_id}] Failed to submit subagent {self.config.name} to the isolated event loop",
                    exc_info=True,
                )
            else:
                logger.debug(
                    f"[trace={self.trace_id}] Subagent {self.config.name} failed while executing on the isolated event loop",
                    exc_info=True,
                )
            raise

    def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """task를 동기로 실행한다(비동기 실행을 감싼 래퍼).

        새 event loop에서 비동기 실행을 돌리므로 thread pool 안에서도 비동기 tool(MCP
        tool 등)을 쓸 수 있다.

        이미 실행 중인 event loop 안에서 호출되면(예: 부모 agent가 async인 경우)
        httpx client 같은 공유 async primitive와의 event loop 충돌을 피하려고 상시
        격리 loop를 동기적으로 기다린다.

        Args:
            task: subagent에 줄 작업 설명.
            result_holder: 실행 중 갱신할, 미리 생성된 선택적 result 객체.

        Returns:
            실행 결과를 담은 SubagentResult.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                logger.debug(f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, using isolated loop")
                return self._execute_in_isolated_loop(task, result_holder)

            # 표준 경로: 실행 중인 event loop가 없으므로 asyncio.run을 쓴다
            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
            # result가 없으면 에러를 담은 result를 생성한다
            if result_holder is not None:
                result = result_holder
            else:
                result = SubagentResult(
                    task_id=str(uuid.uuid4())[:8],
                    trace_id=self.trace_id,
                    status=SubagentStatus.RUNNING,
                )
            result.try_set_terminal(SubagentStatus.FAILED, error=str(e))
            return result

    def execute_async(self, task: str, task_id: str | None = None) -> str:
        """백그라운드에서 task 실행을 시작한다.

        Args:
            task: subagent에 줄 작업 설명.
            task_id: 사용할 선택적 task ID. 주어지지 않으면 무작위 UUID를 생성한다.

        Returns:
            나중에 상태를 조회할 때 쓸 수 있는 Task ID.
        """
        # 주어진 task_id를 쓰거나 새로 생성한다
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        # 초기 pending result 생성
        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, task_id={task_id}, timeout={self.config.timeout_seconds}s")

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        parent_context = _copy_isolated_subagent_context()

        # scheduler pool에 제출
        def run_task():
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                # 실행을 상시 격리 loop에 직접 제출해서, 백그라운드 경로가
                # execute()를 통해 임시 loop를 만들지 않게 한다.
                execution_future = _submit_to_isolated_loop_in_context(
                    parent_context,
                    lambda: self._aexecute(task, result_holder),
                )
                try:
                    # timeout을 두고 실행 완료를 기다린다
                    execution_future.result(timeout=self.config.timeout_seconds)
                except FuturesTimeoutError:
                    logger.error(f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after {self.config.timeout_seconds}s")
                    # 협조적 취소를 알리고 future를 취소한다
                    result_holder.cancel_event.set()
                    result_holder.try_set_terminal(
                        SubagentStatus.TIMED_OUT,
                        error=f"Execution timed out after {self.config.timeout_seconds} seconds",
                    )
                    execution_future.cancel()
            except Exception as e:
                logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
                with _background_tasks_lock:
                    task_result = _background_tasks[task_id]
                task_result.try_set_terminal(SubagentStatus.FAILED, error=str(e))

        _scheduler_pool.submit(run_task)
        return task_id


MAX_CONCURRENT_SUBAGENTS = 3


def request_cancel_background_task(task_id: str) -> None:
    """실행 중인 백그라운드 task에 중지를 알린다.

    task의 cancel_event를 설정하며, ``_aexecute``가 ``agent.astream()`` 반복 중에
    협조적으로 확인한다. 덕분에 ``Future.cancel()``로 강제 종료할 수 없는 subagent
    thread도 다음 반복 경계에서 멈출 수 있다.

    Args:
        task_id: 취소할 task ID.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for background task %s", task_id)


def get_background_task_result(task_id: str) -> SubagentResult | None:
    """백그라운드 task의 결과를 가져온다.

    Args:
        task_id: execute_async가 반환한 task ID.

    Returns:
        찾으면 SubagentResult, 없으면 None.
    """
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def list_background_tasks() -> list[SubagentResult]:
    """모든 백그라운드 task를 나열한다.

    Returns:
        모든 SubagentResult 인스턴스 목록.
    """
    with _background_tasks_lock:
        return list(_background_tasks.values())


def cleanup_background_task(task_id: str) -> None:
    """완료된 task를 백그라운드 task 목록에서 제거한다.

    task_tool이 polling을 마치고 결과를 반환한 뒤 호출해야 한다. 완료된 task가 쌓여
    메모리 누수가 생기는 것을 막는다.

    백그라운드 executor가 아직 task 항목을 갱신 중일 때의 race를 피하려고 terminal
    상태(COMPLETED/FAILED/TIMED_OUT)인 task만 제거한다.

    Args:
        task_id: 제거할 task ID.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            # 정리할 것이 없다. 이미 제거됐을 수 있다.
            logger.debug("Requested cleanup for unknown background task %s", task_id)
            return

        # 백그라운드 executor가 아직 task 항목을 갱신 중일 때의 race를 피하려고
        # terminal 상태인 task만 정리한다.
        if result.status.is_terminal or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
        else:
            logger.debug(
                "Skipping cleanup for non-terminal background task %s (status=%s)",
                task_id,
                result.status.value if hasattr(result.status, "value") else result.status,
            )
