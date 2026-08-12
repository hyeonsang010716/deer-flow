"""subagent에게 작업을 위임하는 task 도구."""

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain.tools import InjectedToolCallId, tool
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

from deerflow.authz.principal import normalize_authz_attributes
from deerflow.config import get_app_config
from deerflow.extensions import resolve_run_extensions
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.sandbox.security import LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config
from deerflow.subagents.config import resolve_subagent_model_name
from deerflow.subagents.executor import (
    SubagentStatus,
    cleanup_background_task,
    get_background_task_result,
    request_cancel_background_task,
)
from deerflow.subagents.status_contract import (
    SubagentStatusValue,
    SubagentStopReasonValue,
    format_subagent_result_message,
    make_subagent_additional_kwargs,
)
from deerflow.tools.types import Runtime
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY, get_current_trace_id, normalize_trace_id
from deerflow.utils.custom_events import aemit_custom_event

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# subagent token usage를 tool_call_id로 캐시해 두면 TokenUsageMiddleware가
# 이를 호출을 유발한 AIMessage의 usage_metadata에 다시 기록할 수 있다.
_subagent_usage_cache: dict[str, dict[str, int]] = {}


def _token_usage_cache_enabled(app_config: "AppConfig | None") -> bool:
    if app_config is None:
        try:
            app_config = get_app_config()
        except FileNotFoundError:
            return False
    return bool(getattr(getattr(app_config, "token_usage", None), "enabled", False))


def _cache_subagent_usage(tool_call_id: str, usage: dict | None, *, enabled: bool = True) -> None:
    if enabled and usage:
        _subagent_usage_cache[tool_call_id] = usage


def pop_cached_subagent_usage(tool_call_id: str) -> dict | None:
    return _subagent_usage_cache.pop(tool_call_id, None)


def _is_subagent_terminal(result: Any) -> bool:
    """background subagent 결과를 정리해도 안전한지 반환한다."""
    return result.status in {SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED, SubagentStatus.TIMED_OUT} or getattr(result, "completed_at", None) is not None


async def _await_subagent_terminal(task_id: str, max_polls: int) -> Any | None:
    """background subagent가 terminal 상태에 도달하거나 poll 횟수가 소진될 때까지 polling한다."""
    for _ in range(max_polls):
        result = get_background_task_result(task_id)
        if result is None:
            return None
        if _is_subagent_terminal(result):
            return result
        await asyncio.sleep(5)
    return None


async def _deferred_cleanup_subagent_task(task_id: str, trace_id: str, max_polls: int) -> None:
    """취소된 subagent를 안전하게 제거할 수 있을 때까지 계속 polling한다."""
    cleanup_poll_count = 0
    while True:
        result = get_background_task_result(task_id)
        if result is None:
            return
        if _is_subagent_terminal(result):
            cleanup_background_task(task_id)
            return
        if cleanup_poll_count >= max_polls:
            logger.warning(f"[trace={trace_id}] Deferred cleanup for task {task_id} timed out after {cleanup_poll_count} polls")
            return
        await asyncio.sleep(5)
        cleanup_poll_count += 1


def _log_cleanup_failure(cleanup_task: asyncio.Task[None], *, trace_id: str, task_id: str) -> None:
    if cleanup_task.cancelled():
        return

    exc = cleanup_task.exception()
    if exc is not None:
        logger.error(f"[trace={trace_id}] Deferred cleanup failed for task {task_id}: {exc}")


def _schedule_deferred_subagent_cleanup(task_id: str, trace_id: str, max_polls: int) -> None:
    logger.debug(f"[trace={trace_id}] Scheduling deferred cleanup for cancelled task {task_id}")
    cleanup_task = asyncio.create_task(_deferred_cleanup_subagent_task(task_id, trace_id, max_polls))
    cleanup_task.add_done_callback(lambda task: _log_cleanup_failure(task, trace_id=trace_id, task_id=task_id))


def _find_usage_recorder(runtime: Any) -> Any | None:
    """runtime config에서 ``record_external_llm_usage_records``를 가진 callback handler를 찾는다.

    LangChain은 ``config["callbacks"]``를 세 가지 형태로 넘길 수 있다.

    - ``None``(등록된 callback 없음): recorder 없음.
    - 평범한 ``list[BaseCallbackHandler]``: 그대로 순회한다.
    - ``BaseCallbackManager`` 인스턴스(예: async tool 실행 시의 ``AsyncCallbackManager``):
      manager는 순회할 수 없으므로 먼저 ``.handlers``를 꺼낸다.

    그 외의 형태(예: list로 감싸지 않고 실수로 넘어온 단일 handler 객체)는 안전하게 순회할 수
    없으므로 예외를 던지지 않고 "recorder 없음"으로 처리한다.
    """
    if runtime is None:
        return None
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return None
    callbacks = config.get("callbacks")
    if isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.handlers
    if not callbacks:
        return None
    if not isinstance(callbacks, list):
        return None
    for cb in callbacks:
        if hasattr(cb, "record_external_llm_usage_records"):
            return cb
    return None


def _summarize_usage(records: list[dict] | None) -> dict | None:
    """token usage 레코드를 SSE 이벤트용 compact dict로 요약한다."""
    if not records:
        return None
    return {
        "input_tokens": sum(r.get("input_tokens", 0) or 0 for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) or 0 for r in records),
        "total_tokens": sum(r.get("total_tokens", 0) or 0 for r in records),
    }


def _report_subagent_usage(runtime: Any, result: Any) -> None:
    """가능한 경우 subagent token usage를 부모 RunJournal에 보고한다.

    각 subagent task는 한 번만 보고되어야 한다(usage_reported로 보호한다).
    """
    if getattr(result, "usage_reported", True):
        return
    records = getattr(result, "token_usage_records", None) or []
    if not records:
        return
    journal = _find_usage_recorder(runtime)
    if journal is None:
        logger.debug("No usage recorder found in runtime callbacks — subagent token usage not recorded")
        return
    try:
        journal.record_external_llm_usage_records(records)
        result.usage_reported = True
    except Exception:
        logger.warning("Failed to report subagent token usage", exc_info=True)


def _get_runtime_app_config(runtime: Any) -> "AppConfig | None":
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        app_config = context.get("app_config")
        if app_config is not None:
            return cast("AppConfig", app_config)
    return None


def _merge_skill_allowlists(parent: list[str] | None, child: list[str] | None) -> list[str] | None:
    """부모 정책 아래에서 실제로 적용될 subagent skill allowlist를 반환한다."""
    if parent is None:
        return child
    if child is None:
        return list(parent)

    parent_set = set(parent)
    return [skill for skill in child if skill in parent_set]


def _task_result_command(
    *,
    tool_call_id: str,
    status: SubagentStatusValue,
    result: str | None = None,
    error: str | None = None,
    stop_reason: SubagentStopReasonValue | None = None,
    model_name: str | None = None,
    usage: dict[str, int] | None = None,
) -> Command:
    content, metadata_error = format_subagent_result_message(status, result=result, error=error, stop_reason=stop_reason)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name="task",
                    additional_kwargs=make_subagent_additional_kwargs(
                        status,
                        result=result,
                        error=metadata_error,
                        stop_reason=stop_reason,
                        model_name=model_name,
                        token_usage=usage,
                    ),
                )
            ]
        }
    )


@tool("task", parse_docstring=True)
async def task_tool(
    runtime: Runtime,
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str | Command:
    """범위가 한정된 작업을 자체 context를 가진 전문 subagent에게 위임한다.

    기대 이득이 위임 오버헤드를 명확히 초과할 때만 위임하라.
    유효한 이득은 다음과 같다:
    - 독립적인 병렬 작업으로 얻는 실질적인 wall-clock 시간 절감
    - 전문화된 tool, skill, model, 도메인 지시문
    - 비정상적으로 context를 많이 소모하는 한정된 조사에 대한 context 격리

    내장 subagent 타입:
    - **general-purpose**: 한정된 탐색과 실행을 수행하는 범용 agent. 해당 작업에 명확한
      전문성 이득이나 context 격리 이득이 있을 때, 또는 실제로 병렬 실행이 가능한 여러
      독립적이고 겹치지 않는 작업 중 하나일 때 사용하라.
    - **bash**: bash 명령 실행 전문 agent. host bash가 명시적으로 허용되었거나
      `AioSandboxProvider` 같은 격리된 shell sandbox를 사용할 때만 사용할 수 있다. 명확한
      context 격리 이득이나 독립적 병렬 이득이 있는 한정된 shell 작업에만 사용하라.
      일상적인 git, build, test, deploy 작업은 위임 사유로 충분하지 않다.

    추가 custom subagent 타입은 config.yaml의 `subagents.custom_agents` 아래에 정의할 수
    있다. 각 custom 타입은 자체 system prompt, tools, skills, model, timeout 설정을 가질 수
    있다. 알 수 없는 subagent_type을 전달하면 오류 메시지가 사용 가능한 모든 타입을 나열한다.

    이 tool을 사용해야 할 때:
    - 병렬로 실행하면 wall-clock 시간이 실질적으로 줄어드는 독립적인 작업
    - 직접 수행 경로에서는 쓸 수 없는 능력을 전문 subagent가 제공할 때
    - 그대로 두면 중요한 parent context를 밀어낼 한정된 탐색

    이 tool을 사용하면 안 되는 때:
    - 단지 작업이 복잡하거나, 여러 단계이거나, 장황하거나, 큰 repo를 다룬다는 이유만으로는 안 된다
    - 의존 관계가 있는 단계들을 여러 병렬 subagent로 쪼개지 마라. 그 체인은 하나로 묶어 두고,
      전문성 이득이나 context 격리 이득이 명확히 클 때만 하나의 한정된 작업으로 위임하라
    - 파일이 겹치거나, 가변 상태를 공유하거나, 외부 side effect가 있는 병렬 작업
    - 사용자 상호작용이나 clarification이 필요한 작업

    위임 판단에 포함해야 하는 비용:
    - 동일한 repository 탐색을 여러 context에서 반복하는 비용
    - 반환된 결과를 조율, 검증, 종합하는 비용
    - parent가 직접 tool로 더 저렴하게 끝낼 수 있는 모든 작업

    Args:
        description: 로깅/표시용 짧은(3-5 단어) 작업 설명. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: subagent에게 줄 작업 설명. 무엇을 해야 하는지 구체적이고 명확하게 적어라. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: 사용할 subagent의 타입. ALWAYS PROVIDE THIS PARAMETER THIRD.
    """
    runtime_app_config = _get_runtime_app_config(runtime)
    cache_token_usage = _token_usage_cache_enabled(runtime_app_config)
    available_subagent_names = get_available_subagent_names(app_config=runtime_app_config) if runtime_app_config is not None else get_available_subagent_names()

    # subagent 설정을 가져온다
    config = get_subagent_config(subagent_type, app_config=runtime_app_config) if runtime_app_config is not None else get_subagent_config(subagent_type)
    if config is None:
        available = ", ".join(available_subagent_names)
        error = f"Unknown subagent type '{subagent_type}'. Available: {available}"
        return _task_result_command(
            tool_call_id=tool_call_id,
            status="failed",
            error=error,
        )
    if subagent_type == "bash":
        host_bash_allowed = is_host_bash_allowed(runtime_app_config) if runtime_app_config is not None else is_host_bash_allowed()
        if not host_bash_allowed:
            return _task_result_command(
                tool_call_id=tool_call_id,
                status="failed",
                error=LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE,
            )

    # config override를 구성한다
    overrides: dict = {}

    # skill은 SubagentExecutor가 세션 단위로 로드한다(Codex 패턴과 동일하게, 각 subagent가
    # 자신의 config에 따라 skill을 로드해 conversation item으로 주입한다).
    # 더 이상 여기서 system_prompt에 덧붙이지 않는다.

    # runtime에서 부모 context를 추출한다
    sandbox_state = None
    thread_data = None
    thread_id = None
    parent_model = None
    trace_id = None
    user_id = None
    deerflow_trace_id = None
    metadata: dict = {}

    if runtime is not None:
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            thread_id = runtime.config.get("configurable", {}).get("thread_id")

        # configurable에서 부모 model을 가져온다
        metadata = runtime.config.get("metadata", {})
        parent_model = metadata.get("model_name")

        # 분산 tracing용 trace_id를 가져오거나 생성한다
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    # tracing용 user_id를 가져온다(표준 resolution 순서를 따른다)
    user_id = resolve_runtime_user_id(runtime)

    # 인증된 runtime context를 전파해서, 위임된 tool call도 lead agent와 동일한
    # identity/attribution으로 GuardrailMiddleware의 평가를 받게 한다. 값은
    # inject_authenticated_user_context가 쓴 server-side context에서 온다(run_id는 run
    # worker가 쓴다). 값이 없으면(예: internal-auth 실행) None으로 남겨 guardrail 동작을
    # 그대로 유지한다. 이게 없으면 role 기반 정책이 subagent에 위임된 tool call을 조용히
    # 잘못 귀속시킨다(user_role=None).
    parent_context = runtime.context if runtime is not None else None
    parent_context = parent_context if isinstance(parent_context, dict) else {}
    user_role = parent_context.get("user_role")
    oauth_provider = parent_context.get("oauth_provider")
    oauth_id = parent_context.get("oauth_id")
    run_id = parent_context.get("run_id")
    # IM-channel 발신자 identity. group chat은 여러 발신자가 하나의 thread를 공유하므로,
    # 위임된 bash 명령은 이번 turn을 발생시킨 발신자의 channel_user_id가 필요하다.
    channel_user_id = parent_context.get("channel_user_id")
    # authorization identity를 전파한다: is_internal(엄격한 bool)과
    # authz_attributes(검증된 Mapping, 복사본). 둘 다 user_role/oauth와 동일한 server-side
    # 출처를 따른다 — inject_authenticated_user_context 참고.
    is_internal = parent_context.get("is_internal") is True
    authz_attributes = normalize_authz_attributes(parent_context.get("authz_attributes"))
    # run worker가 발행한 해당 run의 불변 extension snapshot. 그 경로 밖(embedded client,
    # standalone LangGraph Server)에서는 None으로 남고, 이때 executor는 process-singleton
    # fallback을 사용한다.
    run_extensions = resolve_run_extensions(parent_context)
    deerflow_trace_id = normalize_trace_id(parent_context.get(DEERFLOW_TRACE_METADATA_KEY)) or normalize_trace_id(metadata.get(DEERFLOW_TRACE_METADATA_KEY)) or get_current_trace_id()

    parent_available_skills = metadata.get("available_skills")
    if parent_available_skills is not None:
        overrides["skills"] = _merge_skill_allowlists(list(parent_available_skills), config.skills)

    if overrides:
        config = replace(config, **overrides)

    # 사용 가능한 tool을 가져온다(중첩을 막기 위해 task tool은 제외)
    # 순환 의존을 피하려고 lazy import한다
    from deerflow.tools import get_available_tools

    # 부모 agent의 tool_groups를 물려받아 subagent도 같은 제약을 지키게 한다
    parent_tool_groups = metadata.get("tool_groups")
    resolved_app_config = runtime_app_config
    if config.model == "inherit" and parent_model is None and resolved_app_config is None:
        resolved_app_config = get_app_config()
    effective_model = resolve_subagent_model_name(config, parent_model, app_config=resolved_app_config)

    # subagent에는 subagent tool을 활성화하지 않는다(재귀적 중첩 방지).
    # 또한 subagent에는 list_uploaded_files를 주지 않는다. subagent는 독립적인 ThreadState를
    # 가져서 runtime.state["uploaded_files"]가 없고, 그러면 현재 run의 파일 제외가 동작하지
    # 않기 때문이다.
    available_tools_kwargs = {
        "model_name": effective_model,
        "groups": parent_tool_groups,
        "subagent_enabled": False,
        "include_upload_tool": False,
    }
    if resolved_app_config is not None:
        available_tools_kwargs["app_config"] = resolved_app_config
    tools = get_available_tools(**available_tools_kwargs)

    # executor를 생성한다
    executor_kwargs = {
        "config": config,
        "tools": tools,
        "parent_model": parent_model,
        "sandbox_state": sandbox_state,
        "thread_data": thread_data,
        "thread_id": thread_id,
        "trace_id": trace_id,
        "user_id": user_id,
        "user_role": user_role,
        "oauth_provider": oauth_provider,
        "oauth_id": oauth_id,
        "run_id": run_id,
        "channel_user_id": channel_user_id,
        "is_internal": is_internal,
        "authz_attributes": authz_attributes,
        "deerflow_trace_id": deerflow_trace_id,
    }
    if resolved_app_config is not None:
        executor_kwargs["app_config"] = resolved_app_config
    if run_extensions is not None:
        executor_kwargs["extensions"] = run_extensions
    executor = SubagentExecutor(**executor_kwargs)

    # background 실행을 시작한다(blocking을 막기 위해 항상 async로 실행)
    # 추적을 쉽게 하려고 tool_call_id를 task_id로 사용한다
    task_id = executor.execute_async(prompt, task_id=tool_call_id)

    # backend에서 task 완료를 polling한다(LLM이 직접 poll할 필요가 없어진다)
    poll_count = 0
    last_status = None
    last_message_count = 0  # 이미 보낸 AI message 개수를 추적한다
    # polling timeout: 실행 timeout + 60초 버퍼, 5초마다 확인한다
    max_poll_count = (config.timeout_seconds + 60) // 5

    logger.info(f"[trace={trace_id}] Started background task {task_id} (subagent={subagent_type}, timeout={config.timeout_seconds}s, polling_limit={max_poll_count} polls)")

    writer = get_stream_writer()
    # Task Started 메시지를 보낸다
    await aemit_custom_event(
        {
            "type": "task_started",
            "task_id": task_id,
            "description": description,
            "model_name": effective_model,
        },
        writer=writer,
    )

    try:
        while True:
            result = get_background_task_result(task_id)

            if result is None:
                logger.error(f"[trace={trace_id}] Task {task_id} not found in background tasks")
                await aemit_custom_event(
                    {"type": "task_failed", "task_id": task_id, "error": "Task disappeared from background tasks"},
                    writer=writer,
                )
                cleanup_background_task(task_id)
                error = f"Task {task_id} disappeared from background tasks"
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="failed",
                    error=error,
                )

            # 디버깅을 위해 status 변경을 로깅한다
            if result.status != last_status:
                logger.info(f"[trace={trace_id}] Task {task_id} status: {result.status.value}")
                last_status = result.status

            # collector는 누적 레코드를 발행한다. 실시간 진행 상황과 terminal 이벤트가 같은
            # snapshot을 재사용해야, frontend가 task별 합계를 더하지 않고 교체할 수 있다.
            usage = _summarize_usage(getattr(result, "token_usage_records", None))

            # 새 AI message를 확인하고 task_running 이벤트를 보낸다
            ai_messages = result.ai_messages or []
            current_message_count = len(ai_messages)
            if current_message_count > last_message_count:
                # 새 message마다 task_running 이벤트를 보낸다
                for i in range(last_message_count, current_message_count):
                    message = ai_messages[i]
                    await aemit_custom_event(
                        {
                            "type": "task_running",
                            "task_id": task_id,
                            "message": message,
                            "message_index": i + 1,  # 표시용 1부터 시작하는 index
                            "total_messages": current_message_count,
                            "usage": usage,
                            "model_name": effective_model,
                        },
                        writer=writer,
                    )
                    logger.info(f"[trace={trace_id}] Task {task_id} sent message #{i + 1}/{current_message_count}")
                last_message_count = current_message_count

            # task가 완료/실패/timeout되었는지 확인한다
            if result.status == SubagentStatus.COMPLETED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                await aemit_custom_event(
                    {
                        "type": "task_completed",
                        "task_id": task_id,
                        "result": result.result,
                        "usage": usage,
                        "model_name": effective_model,
                    },
                    writer=writer,
                )
                logger.info(f"[trace={trace_id}] Task {task_id} completed after {poll_count} polls")
                cleanup_background_task(task_id)
                # run이 조기 종료되었지만 최종 답변은 만들어낸 경우 stop_reason이 guardrail
                # cap(token_capped / turn_capped)을 담는다 — 작업 결과는 정상 성공과
                # 마찬가지로 result_brief에 남는다.
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="completed",
                    result=result.result,
                    stop_reason=result.stop_reason,
                    model_name=effective_model,
                    usage=usage,
                )
            elif result.status == SubagentStatus.FAILED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                await aemit_custom_event(
                    {
                        "type": "task_failed",
                        "task_id": task_id,
                        "error": result.error,
                        "usage": usage,
                        "model_name": effective_model,
                    },
                    writer=writer,
                )
                logger.error(f"[trace={trace_id}] Task {task_id} failed: {result.error}")
                cleanup_background_task(task_id)
                # 쓸 만한 출력 없이 turn cap에 걸린 run은 failed + stop_reason=turn_capped로
                # 드러난다. 이 cap 표시가 있어야 lead가 "예산 소진"과 "고장 난 subagent"를
                # 구분할 수 있다.
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="failed",
                    error=result.error,
                    stop_reason=result.stop_reason,
                    model_name=effective_model,
                    usage=usage,
                )
            elif result.status == SubagentStatus.CANCELLED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                await aemit_custom_event(
                    {
                        "type": "task_cancelled",
                        "task_id": task_id,
                        "error": result.error,
                        "usage": usage,
                        "model_name": effective_model,
                    },
                    writer=writer,
                )
                logger.info(f"[trace={trace_id}] Task {task_id} cancelled: {result.error}")
                cleanup_background_task(task_id)
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="cancelled",
                    error=result.error,
                    model_name=effective_model,
                    usage=usage,
                )
            elif result.status == SubagentStatus.TIMED_OUT:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                await aemit_custom_event(
                    {
                        "type": "task_timed_out",
                        "task_id": task_id,
                        "error": result.error,
                        "usage": usage,
                        "model_name": effective_model,
                    },
                    writer=writer,
                )
                logger.warning(f"[trace={trace_id}] Task {task_id} timed out: {result.error}")
                cleanup_background_task(task_id)
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="timed_out",
                    error=result.error,
                    model_name=effective_model,
                    usage=usage,
                )

            # 아직 실행 중이므로 다음 poll 전까지 대기한다
            await asyncio.sleep(5)
            poll_count += 1

            # 안전망으로서의 polling timeout(thread pool timeout이 동작하지 않는 경우 대비).
            # 실행 timeout + 60초 버퍼로 잡고 5초 간격으로 poll한다.
            # background task가 멈춰버리는 엣지 케이스를 잡아낸다.
            if poll_count > max_poll_count:
                timeout_minutes = config.timeout_seconds // 60
                logger.error(f"[trace={trace_id}] Task {task_id} polling timed out after {poll_count} polls (should have been caught by thread pool timeout)")
                _report_subagent_usage(runtime, result)
                usage = _summarize_usage(getattr(result, "token_usage_records", None))
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                await aemit_custom_event(
                    {
                        "type": "task_timed_out",
                        "task_id": task_id,
                        "usage": usage,
                        "model_name": effective_model,
                    },
                    writer=writer,
                )
                # task가 여전히 background에서 실행 중일 수 있다. 협조적 취소를 요청하고,
                # background thread가 terminal 상태에 도달하면 _background_tasks에서 항목을
                # 제거하도록 지연 cleanup을 예약한다.
                request_cancel_background_task(task_id)
                _schedule_deferred_subagent_cleanup(task_id, trace_id, max_poll_count)
                message = f"Task polling timed out after {timeout_minutes} minutes. This may indicate the background task is stuck. Status: {result.status.value}"
                return _task_result_command(
                    tool_call_id=tool_call_id,
                    status="polling_timed_out",
                    error=message,
                    model_name=effective_model,
                    usage=usage,
                )
    except asyncio.CancelledError:
        # background subagent thread에 협조적으로 중단하라고 알린다.
        request_cancel_background_task(task_id)

        # 부모 worker가 get_completion_data()를 저장하기 전에 최종 token usage snapshot이
        # 부모 RunJournal에 보고되도록, subagent가 terminal 상태에 도달할 때까지
        # (shield한 채) 기다린다.
        terminal_result = None
        try:
            terminal_result = await asyncio.shield(_await_subagent_terminal(task_id, max_poll_count))
        except asyncio.CancelledError:
            pass

        # timeout이 났더라도 subagent가 수집한 내용은 그대로 보고한다.
        final_result = terminal_result or get_background_task_result(task_id)
        if final_result is not None:
            _report_subagent_usage(runtime, final_result)
        if final_result is not None and _is_subagent_terminal(final_result):
            cleanup_background_task(task_id)
        else:
            _schedule_deferred_subagent_cleanup(task_id, trace_id, max_poll_count)
        _subagent_usage_cache.pop(tool_call_id, None)
        raise
    except Exception:
        _subagent_usage_cache.pop(tool_call_id, None)
        raise
