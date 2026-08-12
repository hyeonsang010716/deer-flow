"""Lead agent를 만드는 factory.

불변식 — tracing callback 위치
==============================

tracing callback(Langfuse, LangSmith)은 :func:`_make_lead_agent`의 **graph 호출 루트**에
붙인다(``config["callbacks"]``에 덧붙이는 ``build_tracing_callbacks()`` 블록 참고).
이 모듈 안의 모든 ``create_chat_model(...)`` 호출, 그리고 이 graph에서 도달 가능한 모든
middleware(예: ``TitleMiddleware``)의 호출은 반드시 ``attach_tracing=False``를 넘겨야 한다.

이 플래그를 빠뜨리면 span이 중복 생성되고(하나는 graph 루트, 하나는 model 루트), Langfuse
handler의 ``propagate_attributes`` 경로가 동작하지 않아 ``session_id``/``user_id``가 trace에
도달하지 못한다. 현재 해당하는 곳은 다섯 군데다. bootstrap agent, 기본 agent, summarization
middleware, ``TitleMiddleware`` 내부의 async 경로, 그리고 ``skill_manage`` tool에서 도달하는
skill 보안 scanner(``skills/security_scanner.py``의 ``scan_skill_content``). 마지막 것은
용도가 둘이다. ``tools/skill_manage_tool.py``의 ``_scan_or_raise``가 graph 안의 choke point로
플래그를 넘기고, 독립 호출자는 기본값을 유지한다. graph 안에서 ``create_chat_model``을
새로 호출하면 이 목록에 추가하고 플래그를 넘겨야 한다.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.configured_extensions import load_configured_extension_middlewares
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.agents.middlewares.model_length_finish_reason_middleware import ModelLengthFinishReasonMiddleware
from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware
from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware, create_summarization_middleware
from deerflow.agents.middlewares.terminal_response_middleware import TerminalResponseMiddleware
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.agents.middlewares.todo_middleware import TodoMiddleware
from deerflow.agents.middlewares.token_usage_middleware import TokenUsageMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares
from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware
from deerflow.agents.thread_state import get_thread_state_schema, normalize_middleware_state_schemas
from deerflow.authz.principal import build_principal_from_context
from deerflow.authz.provider import AuthzDecision, AuthzRequest
from deerflow.authz.runtime import resolve_authorization_provider
from deerflow.authz.tool_filter import apply_tool_authorization
from deerflow.config.agents_config import load_agent_config, validate_agent_name
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.config.memory_config import should_use_memory_tools
from deerflow.config.subagents_config import DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN
from deerflow.models import create_chat_model
from deerflow.runtime.checkpoint_mode import (
    INTERNAL_CHECKPOINT_MODE_KEY,
    freeze_checkpoint_channel_mode,
    freeze_checkpoint_snapshot_frequency,
    frozen_checkpoint_channel_mode,
    inject_checkpoint_mode,
)
from deerflow.skills.types import Skill
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)

_BOOTSTRAP_SKILL_NAMES = {"bootstrap"}
_NON_INTERACTIVE_DISABLED_TOOL_NAMES = frozenset({"ask_clarification"})

# inbound 메시지가 신뢰할 수 없는 외부 작성자(GitHub 저장소의 아무나 등)에게서 오는 channel
# 목록. 이런 run context는 ``update_agent`` 같은 관리자급 tool에 안전하지 않다.
# 해당 gate는 :func:`_make_lead_agent`에 있고, channel 이름 자체는
# ``ChannelManager._resolve_run_params``가 ``run_context``에 넣어 준다.
_WEBHOOK_CHANNELS: frozenset[str] = frozenset({"github"})


def _default_max_total_subagents(app_config: object) -> int:
    subagents_config = getattr(app_config, "subagents", None)
    return getattr(subagents_config, "max_total_per_run", DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN)


def _resolve_runtime_option(cfg: dict, key: str, agent_value, default):
    """``request > agent config > 기본값`` 우선순위로 runtime 옵션을 결정한다.

    ``cfg.get(key)``가 아니라 ``key in cfg``를 쓰는 이유는 "request가 필드를 생략했다"와
    "request가 falsy 값을 넣었다"를 구분하기 위해서다. 덕분에 request가 준
    ``thinking_enabled: false``가 agent 기본값으로 넘어가지 않고 그대로 존중된다.
    ``agent_value``는 ``None``이 아닐 때만 쓴다. custom agent가 설정하지 않은 필드는
    "덮어쓰지 말라"는 뜻이다(issue #4336).
    """
    if key in cfg:
        return cfg[key]
    if agent_value is not None:
        return agent_value
    return default


def _append_memory_tools_without_name_conflicts(tools: list) -> None:
    """이름이 겹치는 무관한 tool을 버리지 않으면서 memory tool을 덧붙인다."""
    from deerflow.agents.memory.tools import get_memory_tools

    existing_names = {getattr(tool, "name", None) for tool in tools}
    for memory_tool in get_memory_tools():
        if memory_tool.name in existing_names:
            logger.warning("Memory tool name %r already exists and was skipped.", memory_tool.name)
            continue
        tools.append(memory_tool)
        existing_names.add(memory_tool.name)


def _get_runtime_config(config: RunnableConfig) -> dict:
    """레거시 configurable 옵션과 LangGraph runtime context를 병합한다."""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _resolve_model_name(requested_model_name: str | None = None, *, app_config: AppConfig | None = None) -> str:
    """runtime model 이름을 안전하게 결정한다. 유효하지 않으면 기본값으로 넘어간다. 설정된 model이 없으면 None을 반환한다."""
    app_config = app_config or get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def _authorize_model_name(
    model_name: str,
    *,
    context: Mapping[str, Any],
    app_config: AppConfig,
) -> str:
    """결정된 model 이름에 ``model:use`` 인가를 적용한다.

    ``authorization.enabled``가 false면 아무것도 하지 않고 *model_name*을 그대로 반환한다.
    활성화된 경우 ``authorize("model", "use")``로 provider 정책을 확인해, runtime 경로와
    Gateway ``get_model`` 라우트가 같은 action 단위 contract를 강제하도록 한다
    (``list``와 ``use``를 구분하는 custom provider에서 중요하다). 거부되면
    ``filter_resources``가 허용한 첫 model로 부드럽게 fallback한다
    (RFC §9: "run을 깨뜨리지 않도록 에러 대신 허용된 기본값으로 넘어간다").
    허용된 model이 하나도 없고 ``fail_closed``가 true면 ``ValueError``를 던진다
    (기존 "설정된 model 없음" contract와 동일). fail-open이면 원래 이름을 반환한다.

    ``apply_tool_authorization``의 Principal/provider 패턴을 그대로 따라, tool 경로와
    model 경로가 하나의 identity 소스를 공유하게 한다.
    """
    authz_config = app_config.authorization
    if authz_config.enabled is not True:
        return model_name

    provider = resolve_authorization_provider(authz_config)
    if provider is None:
        return model_name

    principal = build_principal_from_context(context, default_role=authz_config.default_role)
    all_names = [m.name for m in app_config.models]

    # 결정된 model을 action 단위 ``model:use`` 정책으로 확인한다. Gateway ``get_model``
    # 라우트도 ``authorize("model", "use")``를 확인하므로 동작이 일치한다. ``action``을
    # 무시하는 내장 RBAC provider에서는 소속 확인과 같다. ``list``와 ``use``를 구분하는
    # custom provider에서는 ``filter_resources``로는 보이지만 ``use``는 거부된 model이
    # runtime에서 조용히 선택되는 일을 막는다.
    try:
        decision = provider.authorize(AuthzRequest(principal=principal, resource="model", action="use", target=model_name))
        if not isinstance(decision, AuthzDecision):
            raise TypeError("AuthorizationProvider.authorize must return AuthzDecision")
        if decision.allow:
            return model_name
    except Exception:
        logger.warning("Authorization provider failed while checking model:use for '%s'", model_name, exc_info=True)
        if authz_config.fail_closed:
            raise ValueError("No models are authorized for the current role (authorization provider error).")
        return model_name

    # 거부된 경우의 부드러운 fallback. ``filter_resources``가 보인다고 한 것 중
    # ``authorize("model", "use")``도 통과하는 첫 model을 고른다. ``action``을 무시하는
    # 내장 RBAC provider에서는 보이는 첫 이름을 고르는 것과 같다. ``list``와 ``use``를
    # 구분하는 custom provider에서는 fallback이 실제로 사용 가능함을 보장한다.
    try:
        allowed_names = provider.filter_resources(principal, "model", all_names)
        if not isinstance(allowed_names, list) or any(not isinstance(n, str) for n in allowed_names):
            raise TypeError("AuthorizationProvider.filter_resources must return list[str]")
    except Exception:
        logger.warning("Authorization provider failed while resolving allowed models", exc_info=True)
        if authz_config.fail_closed:
            raise ValueError("No models are authorized for the current role (authorization provider error).")
        return model_name

    for candidate in allowed_names:
        if candidate == model_name:
            continue  # 위에서 이미 거부됐다
        try:
            cb_decision = provider.authorize(AuthzRequest(principal=principal, resource="model", action="use", target=candidate))
            if isinstance(cb_decision, AuthzDecision) and cb_decision.allow:
                logger.warning(
                    "Model '%s' is not authorized for the current role; fallback to '%s'.",
                    model_name,
                    candidate,
                )
                return candidate
        except Exception:
            logger.warning(
                "Authorization provider failed while checking model:use fallback for '%s'",
                candidate,
                exc_info=True,
            )
            if authz_config.fail_closed:
                raise ValueError("No models are authorized for the current role (authorization provider error).")
            return model_name
    if authz_config.fail_closed:
        raise ValueError("No models are authorized for the current role.")
    logger.warning("No models are authorized for the current role; fail_open allows '%s'.", model_name)
    return model_name


def _create_summarization_middleware(*, app_config: AppConfig | None = None, run_model_name: str | None = None) -> DeerFlowSummarizationMiddleware | None:
    """config로부터 summarization middleware를 만들어 설정한다.

    ``run_model_name``은 결정된 run model이며, ``model_name: null`` summarization과 명시적
    summary model fallback의 기준이 된다. 덕분에 ``config.models[0]`` 대신 custom agent의
    model이 쓰인다.
    """
    return create_summarization_middleware(app_config=app_config, run_model_name=run_model_name)


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """TodoList middleware를 만들어 설정한다.

    Args:
        is_plan_mode: TodoList middleware가 붙는 plan mode를 켤지 여부.

    Returns:
        plan mode가 켜져 있으면 TodoMiddleware 인스턴스, 아니면 None.
    """
    if not is_plan_mode:
        return None

    # DeerFlow 스타일에 맞춘 커스텀 프롬프트
    system_prompt = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly

**When to Use:**
This tool is designed for complex objectives that require systematic tracking:
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks needing careful planning and execution
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated list)
- The plan may need revisions based on intermediate results

**When NOT to Use:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational or informational requests
- Simple tool calls where the approach is obvious

**Best Practices:**
- Break down complex tasks into smaller, actionable steps
- Use clear, descriptive task names
- Remove tasks that become irrelevant
- Add new tasks discovered during implementation
- Don't be afraid to revise the todo list as you learn more

**Task Management:**
Writing todos takes time and tokens - use it when helpful for managing complex problems, not for simple requests.
</todo_list_system>
"""

    tool_description = """Use this tool to create and manage a structured task list for complex work sessions.

**IMPORTANT: Only use this tool for complex tasks (3+ steps). For simple requests, just do the work directly.**

## When to Use

Use this tool in these scenarios:
1. **Complex multi-step tasks**: When a task requires 3 or more distinct steps or actions
2. **Non-trivial tasks**: Tasks requiring careful planning or multiple operations
3. **User explicitly requests todo list**: When the user directly asks you to track tasks
4. **Multiple tasks**: When users provide a list of things to be done
5. **Dynamic planning**: When the plan may need updates based on intermediate results

## When NOT to Use

Skip this tool when:
1. The task is straightforward and takes less than 3 steps
2. The task is trivial and tracking provides no benefit
3. The task is purely conversational or informational
4. It's clear what needs to be done and you can just do it

## How to Use

1. **Starting a task**: Mark it as `in_progress` BEFORE beginning work
2. **Completing a task**: Mark it as `completed` IMMEDIATELY after finishing
3. **Updating the list**: Add new tasks, remove irrelevant ones, or update descriptions as needed
4. **Multiple updates**: You can make several updates at once (e.g., complete one task and start the next)

## Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (can have multiple if tasks run in parallel)
- `completed`: Task finished successfully

## Task Completion Requirements

**CRITICAL: Only mark a task as completed when you have FULLY accomplished it.**

Never mark a task as completed if:
- There are unresolved issues or errors
- Work is partial or incomplete
- You encountered blockers preventing completion
- You couldn't find necessary resources or dependencies
- Quality standards haven't been met

If blocked, keep the task as `in_progress` and create a new task describing what needs to be resolved.

## Best Practices

- Create specific, actionable items
- Break complex tasks into smaller, manageable steps
- Use clear, descriptive task names
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
- Remove tasks that are no longer relevant
- **IMPORTANT**: When you write the todo list, mark your first task(s) as `in_progress` immediately
- **IMPORTANT**: Unless all tasks are completed, always have at least one task `in_progress` to show progress

Being proactive with task management demonstrates thoroughness and ensures all requirements are completed successfully.

**Remember**: If you only need a few tool calls to complete a task and it's clear what to do, it's better to just do the task directly and NOT use this tool at all.
"""

    return TodoMiddleware(system_prompt=system_prompt, tool_description=tool_description)


# ThreadDataMiddleware는 thread_id를 확보하기 위해 SandboxMiddleware보다 앞에 와야 한다.
# UploadsMiddleware는 thread_id를 쓰므로 ThreadDataMiddleware 뒤에 와야 한다.
# DanglingToolCallMiddleware는 model이 히스토리를 보기 전에 누락된 ToolMessage를 채운다.
# SummarizationMiddleware는 다른 처리 전에 context를 줄이도록 앞쪽에 둔다.
# TodoListMiddleware는 todo 관리를 위해 ClarificationMiddleware보다 앞에 둔다.
# TitleMiddleware는 첫 대화가 끝난 뒤 제목을 생성한다.
# MemoryMiddleware는 대화를 memory 업데이트 큐에 넣는다(TitleMiddleware 뒤).
# ViewImageMiddleware는 LLM 호출 전에 이미지 정보를 주입하도록 ClarificationMiddleware 앞에 둔다.
# ToolErrorHandlingMiddleware는 tool 예외를 ToolMessage로 바꾸도록 ClarificationMiddleware 앞에 둔다.
# ClarificationMiddleware는 model 호출 뒤 clarification 요청을 가로채야 하므로 마지막에 둔다.
def build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None = None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    available_skills: set[str] | None = None,
    app_config: AppConfig | None = None,
    deferred_setup=None,
    mcp_routing_middleware: AgentMiddleware | None = None,
    user_id: str | None = None,
    authorization_provider=None,
    extensions=None,
):
    """runtime 설정에 따라 lead agent의 middleware chain을 구성한다.

    lead agent middleware 전체 구성의 공개 진입점이다. ``make_lead_agent``와 embedded
    ``DeerFlowClient``(같은 chain이 필요한 lead agent 변형)가 사용한다. 이름은 그대로 둔다.
    모듈 경계를 넘어 import되므로 이름이나 시그니처를 바꾸면 ``client.py``까지 영향이 간다.

    Args:
        config: is_plan_mode 같은 configurable 옵션이 담긴 runtime 설정.
        model_name: 결정된 runtime model 이름. vision 전용 middleware를 켤지 결정한다.
        agent_name: 주어지면 MemoryMiddleware가 agent별 memory 저장소를 쓴다.
        custom_middlewares: chain에 주입할 커스텀 middleware 목록(선택).
        app_config: 명시적 AppConfig. 생략하면 ``get_app_config()``로 넘어간다.
        deferred_setup: ``tool_search``가 켜져 있을 때 ``DeferredToolFilterMiddleware``를
            붙이는 deferred MCP tool setup(선택).
        mcp_routing_middleware: deferred filter가 돌기 전에 deferred MCP schema를
            자동 promote하는 PR2 middleware(선택).
        user_id: 사용자 범위 skill 로딩에 쓰는 실효 user ID. ``SkillActivationMiddleware``로
            전달되어 사용자별 custom skill을 해석하게 한다.
        authorization_provider: 조립 시점 필터링에서 이미 해석된 provider. 실행 시점
            authorization middleware가 재사용한다.
        extensions: middleware 기여를 최종 stack에 병합할 로드된 extension. 기본값은
            프로세스 전역 집합이다.

    Returns:
        middleware 인스턴스 목록.
    """
    resolved_app_config = app_config or get_app_config()
    runtime_middleware_kwargs = {
        "app_config": resolved_app_config,
        "lazy_init": True,
    }
    if authorization_provider is not None:
        runtime_middleware_kwargs["authorization_provider"] = authorization_provider
    if authorization_provider is not None and deferred_setup is not None:
        runtime_middleware_kwargs["deferred_setup"] = deferred_setup
    middlewares = build_lead_runtime_middlewares(**runtime_middleware_kwargs)

    # prefix-cache 재사용을 위해 system prompt를 완전히 정적으로 유지하도록, 현재 날짜(및
    # 선택적으로 memory)를 항상 첫 HumanMessage에 <system-reminder>로 주입한다.
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware

    middlewares.append(DynamicContextMiddleware(agent_name=agent_name, app_config=resolved_app_config))

    # 사용자가 /skill-name으로 턴을 시작하면 SKILL.md 전체를 결정론적으로 로드한다. 기본
    # system prompt는 메타데이터만 담은 채로 두면서, model의 관련성 추측보다 사용자의 명시적
    # 활성화를 우선한다.
    from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware

    slash_source_owner_token = secrets.token_urlsafe(24)
    middlewares.append(
        SkillActivationMiddleware(
            available_skills=available_skills,
            app_config=resolved_app_config,
            user_id=user_id,
            slash_source_owner_token=slash_source_owner_token,
        )
    )

    # 활성화된 skill은 발견 가능한 메타데이터일 뿐이다. allowed-tools는 명시적 slash 활성화
    # 또는 실제 skill 파일 로드 이후 runtime에 적용한다.
    from deerflow.agents.middlewares.skill_tool_policy_middleware import SkillToolPolicyMiddleware

    middlewares.append(
        SkillToolPolicyMiddleware(
            available_skills=available_skills,
            app_config=resolved_app_config,
            user_id=user_id,
            slash_source_owner_token=slash_source_owner_token,
        )
    )

    # summarization이 압축하기 전에 완료된 task 위임과 로드된 skill 파일을 붙잡아 두고,
    # durable context channel(summary + ledger + skills)을 model 호출에 주입한다.
    from deerflow.agents.middlewares.durable_context_middleware import DurableContextMiddleware

    middlewares.append(
        DurableContextMiddleware(
            skills_container_path=resolved_app_config.skills.container_path,
            skill_file_read_tool_names=resolved_app_config.summarization.skill_file_read_tool_names,
        )
    )

    # 활성화돼 있으면 summarization middleware를 추가한다
    summarization_middleware = _create_summarization_middleware(app_config=resolved_app_config, run_model_name=model_name)
    if summarization_middleware is not None:
        middlewares.append(summarization_middleware)

    # plan mode가 켜져 있으면 TodoList middleware를 추가한다
    cfg = _get_runtime_config(config)
    is_plan_mode = cfg.get("is_plan_mode", False)
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)

    # token_usage 추적이 켜져 있으면 TokenUsageMiddleware를 추가한다
    if resolved_app_config.token_usage.enabled:
        middlewares.append(TokenUsageMiddleware())

    # TitleMiddleware를 추가한다
    middlewares.append(TitleMiddleware(app_config=resolved_app_config))

    # TitleMiddleware 뒤에 MemoryMiddleware를 추가한다. tool mode는 보통 건너뛰지만,
    # 대화 추출 방식 backend는 수동 쓰기를 명시적으로 유지할 수 있다.
    if should_use_memory_tools(resolved_app_config.memory):
        from deerflow.agents.memory.manager import backend_requires_passive_writes_in_tool_mode

        if backend_requires_passive_writes_in_tool_mode(resolved_app_config.memory.manager_class):
            middlewares.append(MemoryMiddleware(agent_name=agent_name, memory_config=resolved_app_config.memory))
    else:
        if resolved_app_config.memory.mode == "tool" and not resolved_app_config.memory.enabled:
            logger.warning("memory.mode is 'tool' but memory.enabled is false; memory tools will not be registered.")
        middlewares.append(MemoryMiddleware(agent_name=agent_name, memory_config=resolved_app_config.memory))

    # 현재 model이 vision을 지원할 때만 ViewImageMiddleware를 추가한다.
    # 오래된 config 값을 피하려고 make_lead_agent가 결정한 runtime model_name을 쓴다.
    model_config = resolved_app_config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        middlewares.append(ViewImageMiddleware())

    # deferred filter가 이번 model 호출에서 숨길 schema를 정하기 전에, PR1 routing
    # 메타데이터로 deferred MCP schema를 자동 promote한다.
    if mcp_routing_middleware is not None:
        middlewares.append(mcp_routing_middleware)

    # tool_search가 promote하기 전까지 deferred tool schema를 model 바인딩에서 숨긴다.
    # lead의 deferred 집합과 catalog hash는 빌드 시점 MCP catalog 전체에서 온다.
    # SkillToolPolicyMiddleware는 별도로 runtime에 활성 skill 기준으로 model 가시성,
    # tool_search 결과, 실행을 필터링한다.
    if deferred_setup is not None and deferred_setup.deferred_names:
        from deerflow.agents.middlewares.deferred_tool_filter_middleware import DeferredToolFilterMiddleware

        middlewares.append(DeferredToolFilterMiddleware(deferred_setup.deferred_names, deferred_setup.catalog_hash))
        from deerflow.agents.middlewares.mcp_routing_middleware import assert_mcp_routing_before_deferred_filter

        assert_mcp_routing_before_deferred_filter(middlewares)

    # request가 provider에 닿기 전에 모든 SystemMessage를 맨 앞의 하나로 합친다. 엄격한
    # backend(vLLM, SGLang, Qwen, Anthropic)는 맨 앞이 아닌 SystemMessage를 거부한다.
    # system_message_coalescing_middleware.py 참고.
    from deerflow.agents.middlewares.system_message_coalescing_middleware import SystemMessageCoalescingMiddleware

    middlewares.append(SystemMessageCoalescingMiddleware())

    # 병렬 task 호출이 넘칠 때 잘라내도록 SubagentLimitMiddleware를 추가한다
    subagent_enabled = cfg.get("subagent_enabled", False)
    effective_max_subagents_per_run: int | None = None
    if subagent_enabled:
        max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
        max_total_subagents = cfg.get("max_total_subagents", _default_max_total_subagents(resolved_app_config))
        effective_max_subagents_per_run = max_total_subagents
        middlewares.append(SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents, max_total=max_total_subagents))

    # LoopDetectionMiddleware — 반복되는 tool 호출 loop를 감지하고 끊는다
    loop_detection_config = resolved_app_config.loop_detection
    if loop_detection_config.enabled:
        middlewares.append(LoopDetectionMiddleware.from_config(loop_detection_config))

    # TokenBudgetMiddleware - run 단위 token 한도를 강제한다
    token_budget_config = resolved_app_config.token_budget
    if token_budget_config.enabled:
        from deerflow.agents.middlewares.token_budget_middleware import TokenBudgetMiddleware

        middlewares.append(TokenBudgetMiddleware.from_config(token_budget_config))

    # ClarificationMiddleware 앞에 커스텀 middleware를 주입한다
    if custom_middlewares:
        middlewares.extend(custom_middlewares)

    configured_middlewares = load_configured_extension_middlewares(resolved_app_config)
    if configured_middlewares:
        middlewares.extend(configured_middlewares)

    # provider는 tool 실행 뒤 빈 AIMessage를 반환할 수 있다. 최종 응답을 한 번 재시도하고,
    # 그래도 비어 있으면 보이는 error fallback을 남긴다. LangChain의 no-tool-call 라우터가
    # 조용히 성공으로 run을 끝내게 두지 않는다.
    middlewares.append(TerminalResponseMiddleware())

    # provider가 최종 assistant 응답을 model 출력 한도에서 잘라낼 수도 있다. assistant
    # 내용은 그대로 두되 run 수준 stop_reason을 남겨, Gateway 소비자가 길이 제한으로 잘린
    # 완료와 정상 완료를 구분할 수 있게 한다.
    middlewares.append(ModelLengthFinishReasonMiddleware())

    # SafetyFinishReasonMiddleware — provider가 안전 사유로 응답을 종료했을 때 tool 실행을
    # 막는다. LangChain의 after_model이 역순으로 실행되므로, terminal-response 및
    # custom/configured middleware 뒤에 등록해야 Safety가 가장 먼저 돈다. 비워진 tool_calls는
    # 남은 accounting/terminal guard를 지나가되 불필요한 경보를 울리지 않는다.
    safety_config = resolved_app_config.safety_finish_reason
    if safety_config.enabled:
        middlewares.append(SafetyFinishReasonMiddleware.from_config(safety_config))

    # ClarificationMiddleware는 항상 마지막이어야 한다
    middlewares.append(ClarificationMiddleware())

    # extension 기여는 전체 stack이 완성된 이 시점에서만 병합한다.
    # build_lead_runtime_middlewares() 안에서 하면 MODEL_PHYSICAL 기여가 위에서 덧붙인
    # lead 전용 middleware보다 위에 놓여, observer가 보는 "최종 request"의 의미가 달라진다.
    from deerflow_extension_api import AgentScope

    from deerflow.extensions import get_agent_build_extensions
    from deerflow.extensions.stack import compose_with_extensions

    resolved_extensions = extensions if extensions is not None else get_agent_build_extensions()
    if not resolved_extensions.has_middleware_contributors:
        return compose_with_extensions(middlewares, AgentScope.LEAD, None, resolved_extensions)

    from deerflow_extension_api import AgentBuildContext

    from deerflow.extensions.policy import project_host_policy

    return compose_with_extensions(
        middlewares,
        AgentScope.LEAD,
        AgentBuildContext(
            scope=AgentScope.LEAD,
            agent_name=agent_name,
            model_name=model_name,
            policy=project_host_policy(
                resolved_app_config,
                token_budget_config=token_budget_config,
                max_subagents_per_run=effective_max_subagents_per_run,
            ),
        ),
        resolved_extensions,
    )


def _available_skill_names(agent_config, is_bootstrap: bool) -> set[str] | None:
    if is_bootstrap:
        return set(_BOOTSTRAP_SKILL_NAMES)
    if agent_config and agent_config.skills is not None:
        return set(agent_config.skills)
    return None


def _load_enabled_available_skills(available_skills: set[str] | None, *, app_config: AppConfig, user_id: str | None = None) -> list[Skill]:
    try:
        from deerflow.agents.lead_agent.prompt import get_enabled_skills_for_config

        skills = get_enabled_skills_for_config(app_config, user_id=user_id)
    except Exception:
        logger.exception("Failed to load enabled skills")
        raise

    if available_skills is None:
        return skills
    return [skill for skill in skills if skill.name in available_skills]


def make_lead_agent(config: RunnableConfig):
    """LangGraph graph factory. 시그니처는 LangGraph Server와 호환되게 유지한다."""
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    if not isinstance(runtime_app_config, AppConfig):
        runtime_app_config = get_app_config()
    # mode 선택 우선순위. test_checkpoint_mode.py가 이를 고정한다.
    # - 첫 freeze: 프로세스 mode는 app config가 소유한다. client가 준 configurable key는
    #   무시하므로, LangGraph에 직접 보낸 request가 갓 뜬 프로세스를 재설정하거나 죽일 수 없다.
    # - freeze된 뒤: 내부에서 주입한 key(run worker/gateway)나 app config가 freeze된 mode와
    #   일치해야 한다. ``freeze_checkpoint_channel_mode``는 불일치 시 fail-closed로 막으므로
    #   위조된 key도 config.yaml 변경도 프로세스를 조용히 재설정할 수 없다.
    frozen_mode = frozen_checkpoint_channel_mode()
    if frozen_mode is None:
        requested_mode = runtime_app_config.database.checkpoint_channel_mode
    else:
        requested_mode = (config.get("configurable", {}) or {}).get(
            INTERNAL_CHECKPOINT_MODE_KEY,
            runtime_app_config.database.checkpoint_channel_mode,
        )
    mode = freeze_checkpoint_channel_mode(requested_mode)
    # snapshot 주기는 mode와 함께 움직인다. 재시작이 필요하고, app config에서 freeze되며,
    # 의도적으로 client가 주입할 수 없다. 위조된 configurable key가 channel table을 다시
    # 컴파일하게 두어서도 안 된다.
    freeze_checkpoint_snapshot_frequency(runtime_app_config.database.checkpoint_delta.snapshot_frequency)
    inject_checkpoint_mode(config, mode)
    return _make_lead_agent(config, app_config=runtime_app_config)


def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    # 순환 의존을 피하려는 lazy import
    from deerflow.tools import get_available_tools
    from deerflow.tools.builtins import setup_agent, update_agent
    from deerflow.tools.builtins.tool_search import assemble_deferred_tools, build_mcp_routing_middleware, get_mcp_routing_hints_prompt_section

    cfg = _get_runtime_config(config)
    resolved_app_config = app_config
    mode = (config.get("configurable", {}) or {}).get(
        INTERNAL_CHECKPOINT_MODE_KEY,
        resolved_app_config.database.checkpoint_channel_mode,
    )

    # 사용자 범위 factory 입력 전부에 대해 권위 있는 identity 하나를 결정한다.
    # Agent Server의 예약된 auth 필드가 client가 준 일반 context/configurable 값보다 우선한다.
    # embedded Gateway 경로는 context.user_id를 쓴다.
    from deerflow.runtime.user_context import resolve_config_user_id

    resolved_user_id = resolve_config_user_id(config)

    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    is_plan_mode = cfg.get("is_plan_mode", False)
    subagent_enabled = cfg.get("subagent_enabled", False)
    max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
    max_total_subagents = cfg.get("max_total_subagents", _default_max_total_subagents(resolved_app_config))
    is_bootstrap = cfg.get("is_bootstrap", False)
    non_interactive = bool(cfg.get("non_interactive", False))
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name, user_id=resolved_user_id) if not is_bootstrap else None
    available_skills = _available_skill_names(agent_config, is_bootstrap)
    # agent config의 custom agent model. 없으면 None이라 _resolve_model_name이 기본값을 고른다
    agent_model_name = agent_config.model if agent_config and agent_config.model else None

    # thinking/reasoning 우선순위: request > custom agent 기본값 > runtime 기본값
    # (issue #4336). falsy와 미설정을 구분하는 방식은 ``_resolve_runtime_option`` 참고.
    agent_thinking = getattr(agent_config, "thinking_enabled", None) if agent_config else None
    agent_reasoning = getattr(agent_config, "reasoning_effort", None) if agent_config else None
    thinking_enabled = bool(_resolve_runtime_option(cfg, "thinking_enabled", agent_thinking, True))
    reasoning_effort = _resolve_runtime_option(cfg, "reasoning_effort", agent_reasoning, None)

    # 결정된 model 프로파일 위에 얹는 agent별 sampling override(temperature/max_tokens,
    # issue #4336). agent가 아무것도 설정하지 않았으면 None이다.
    agent_model_settings = getattr(agent_config, "model_settings", None) if agent_config else None
    agent_model_overrides = agent_model_settings.model_dump(exclude_none=True) if agent_model_settings else None

    # 최종 model 이름 결정: request → agent config → 전역 기본값. 알 수 없는 이름은 fallback한다
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    # Phase 3: model:use 인가를 강제한다. 거부되면 run을 죽이는 대신 허용된 첫 model로
    # 부드럽게 fallback한다(RFC §9).
    model_name = _authorize_model_name(model_name, context=cfg, app_config=resolved_app_config)

    model_config = resolved_app_config.get_model_config(model_name)

    if model_config is None:
        raise ValueError("No chat model could be resolved. Please configure at least one model in config.yaml or provide a valid 'model_name'/'model' in the request.")
    if thinking_enabled and not model_config.supports_thinking:
        logger.warning(f"Thinking mode is enabled but model '{model_name}' does not support it; fallback to non-thinking mode.")
        thinking_enabled = False

    logger.info(
        "Create Agent(%s) -> thinking_enabled: %s, reasoning_effort: %s, model_name: %s, is_plan_mode: %s, subagent_enabled: %s, max_concurrent_subagents: %s, max_total_subagents: %s",
        agent_name or "default",
        thinking_enabled,
        reasoning_effort,
        model_name,
        is_plan_mode,
        subagent_enabled,
        max_concurrent_subagents,
        max_total_subagents,
    )

    # LangSmith trace 태깅용 run 메타데이터를 주입한다
    if "metadata" not in config:
        config["metadata"] = {}

    config["metadata"].update(
        {
            "agent_name": agent_name or "default",
            "model_name": model_name or "default",
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "is_plan_mode": is_plan_mode,
            "subagent_enabled": subagent_enabled,
            "tool_groups": agent_config.tool_groups if agent_config else None,
            "available_skills": sorted(available_skills) if available_skills is not None else None,
        }
    )

    # tracing callback을 graph 호출 루트에 주입한다. 그래야 LangGraph run 하나가 모든
    # node/LLM/tool 호출을 자식 span으로 갖는 trace 하나를 만들고, Langfuse handler가
    # ``on_chain_start(parent_run_id=None)``을 보고 ``config["metadata"]``의
    # ``langfuse_session_id``/``langfuse_user_id``를 trace로 전파한다. 루트에 붙이지 않으면
    # model이 중첩 observation이 되고 handler가 ``langfuse_*`` key를 떼어낸다.
    tracing_callbacks = build_tracing_callbacks()
    if tracing_callbacks:
        existing = config.get("callbacks") or []
        if not isinstance(existing, list):
            existing = list(existing)
        config["callbacks"] = [*existing, *tracing_callbacks]

    enabled_skills = _load_enabled_available_skills(available_skills, app_config=resolved_app_config, user_id=resolved_user_id)

    # skill search setup(deferred skill discovery)을 만든다.
    # skills.deferred_discovery가 제어하며 tool_search.enabled와는 무관하다.
    from deerflow.skills.describe import build_skill_search_setup

    skill_search_enabled = resolved_app_config.skills.deferred_discovery
    container_base_path = resolved_app_config.skills.container_path

    if is_bootstrap:
        # custom agent 최초 생성 흐름을 위한 최소 프롬프트의 전용 bootstrap agent.
        # custom agent 자신의 config가 생기기 전에도 생성이 결정론적으로 남도록 bootstrap
        # skill 집합은 의도적으로 좁게 유지한다.
        bootstrap_skills = [s for s in enabled_skills if s.name in _BOOTSTRAP_SKILL_NAMES]
        skill_setup = build_skill_search_setup(
            bootstrap_skills,
            enabled=skill_search_enabled,
            container_base_path=container_base_path,
        )
        raw_tools = get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled, app_config=resolved_app_config) + [setup_agent]
        configured_tools = raw_tools
        if non_interactive:
            configured_tools = [tool for tool in configured_tools if tool.name not in _NON_INTERACTIVE_DISABLED_TOOL_NAMES]
        authorization_candidates = [*configured_tools]
        if skill_setup.describe_skill_tool:
            authorization_candidates.append(skill_setup.describe_skill_tool)
        if should_use_memory_tools(resolved_app_config.memory):
            _append_memory_tools_without_name_conflicts(authorization_candidates)
        configured_tool_ids = {id(tool) for tool in configured_tools}
        authorized_tools, _authz_provider = apply_tool_authorization(
            authorization_candidates,
            context=cfg,
            app_config=resolved_app_config,
        )
        configured_tools = [tool for tool in authorized_tools if id(tool) in configured_tool_ids]
        late_tools = [tool for tool in authorized_tools if id(tool) not in configured_tool_ids]
        final_tools, setup = assemble_deferred_tools(configured_tools, enabled=resolved_app_config.tool_search.enabled)
        final_tools.extend(late_tools)
        mcp_routing_middleware = build_mcp_routing_middleware(
            final_tools,
            setup,
            top_k=resolved_app_config.tool_search.auto_promote_top_k,
        )
        return create_agent(
            model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, app_config=resolved_app_config, attach_tracing=False),
            tools=final_tools,
            middleware=normalize_middleware_state_schemas(
                build_middlewares(
                    config,
                    model_name=model_name,
                    available_skills=set(_BOOTSTRAP_SKILL_NAMES),
                    app_config=resolved_app_config,
                    deferred_setup=setup,
                    mcp_routing_middleware=mcp_routing_middleware,
                    user_id=resolved_user_id,
                    authorization_provider=_authz_provider,
                ),
                mode,
            ),
            system_prompt=apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                max_total_subagents=max_total_subagents,
                available_skills=set(_BOOTSTRAP_SKILL_NAMES),
                app_config=resolved_app_config,
                deferred_names=setup.deferred_names,
                user_id=resolved_user_id,
                skill_names=skill_setup.skill_names or None,
            ),
            state_schema=get_thread_state_schema(mode),
        )

    # custom agent는 update_agent로 자신의 SOUL.md/config를 갱신할 수 있다.
    # 기본 agent(agent_name 없음)에게는 이 tool이 보이지 않는다.
    # agent가 쓸 수 있는 skill로 skill search setup을 만든다. 같은 allowlist를 runtime 정책
    # resolver가 강제하므로, describe_skill이 이 custom agent가 활성화할 수 없는 skill을
    # 노출할 수 없다.
    skill_setup = build_skill_search_setup(
        enabled_skills,
        enabled=skill_search_enabled,
        container_base_path=container_base_path,
    )
    #
    # webhook channel(현재는 ``github``뿐)이 촉발한 run에서는 ``update_agent``를 뺀다.
    # webhook 프롬프트는 임의의 외부 작성자에게서 온다. 설정된 저장소에 글을 쓸 수 있고
    # ``@<bot>``만 적으면 누구나 trigger gate를 통과한다. 여기서 이 tool을 노출하면 그
    # 작성자가 agent의 ``tool_groups``/``SOUL.md``/``model``을 바꿀 수 있고, 그 변경은 이후
    # 모든 run에 남는다. 자기 변경은 운영자가 신뢰하는 표면(채팅 UI, HTTP API)의 몫이지
    # webhook fan-out의 몫이 아니다.
    #
    # channel 이름은 ``ChannelManager._resolve_run_params``가 ``run_context``에 넣는다.
    # bootstrap과 직접 호출은 이를 설정하지 않으므로 그쪽에서는 ``update_agent``가 남는다.
    channel_name = cfg.get("channel_name")
    is_webhook_channel = channel_name in _WEBHOOK_CHANNELS
    extra_tools = [update_agent] if agent_name and not is_webhook_channel else []
    # 기본 lead agent (동작 변경 없음)
    raw_tools = get_available_tools(model_name=model_name, groups=agent_config.tool_groups if agent_config else None, subagent_enabled=subagent_enabled, app_config=resolved_app_config)
    configured_tools = raw_tools + extra_tools
    if non_interactive:
        configured_tools = [tool for tool in configured_tools if tool.name not in _NON_INTERACTIVE_DISABLED_TOOL_NAMES]
    authorization_candidates = [*configured_tools]
    if skill_setup.describe_skill_tool:
        authorization_candidates.append(skill_setup.describe_skill_tool)
    if should_use_memory_tools(resolved_app_config.memory):
        _append_memory_tools_without_name_conflicts(authorization_candidates)
    configured_tool_ids = {id(tool) for tool in configured_tools}
    authorized_tools, _authz_provider = apply_tool_authorization(
        authorization_candidates,
        context=cfg,
        app_config=resolved_app_config,
    )
    configured_tools = [tool for tool in authorized_tools if id(tool) in configured_tool_ids]
    late_tools = [tool for tool in authorized_tools if id(tool) not in configured_tool_ids]
    final_tools, setup = assemble_deferred_tools(configured_tools, enabled=resolved_app_config.tool_search.enabled)
    final_tools.extend(late_tools)
    mcp_routing_middleware = build_mcp_routing_middleware(
        final_tools,
        setup,
        top_k=resolved_app_config.tool_search.auto_promote_top_k,
    )
    mcp_routing_hints_section = get_mcp_routing_hints_prompt_section(authorized_tools, deferred_names=setup.deferred_names)
    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort, app_config=resolved_app_config, attach_tracing=False, model_overrides=agent_model_overrides),
        tools=final_tools,
        middleware=normalize_middleware_state_schemas(
            build_middlewares(
                config,
                model_name=model_name,
                agent_name=agent_name,
                available_skills=available_skills,
                app_config=resolved_app_config,
                deferred_setup=setup,
                mcp_routing_middleware=mcp_routing_middleware,
                user_id=resolved_user_id,
                authorization_provider=_authz_provider,
            ),
            mode,
        ),
        system_prompt=apply_prompt_template(
            subagent_enabled=subagent_enabled,
            max_concurrent_subagents=max_concurrent_subagents,
            max_total_subagents=max_total_subagents,
            agent_name=agent_name,
            available_skills=available_skills,
            app_config=resolved_app_config,
            deferred_names=setup.deferred_names,
            mcp_routing_hints_section=mcp_routing_hints_section,
            user_id=resolved_user_id,
            skill_names=skill_setup.skill_names or None,
        ),
        state_schema=get_thread_state_schema(mode),
    )
