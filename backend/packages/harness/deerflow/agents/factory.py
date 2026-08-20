"""인자만으로 DeerFlow agent를 만드는 factory.

``create_deerflow_agent``는 평범한 Python 인자만 받는다. YAML 파일도, 전역 singleton도
쓰지 않는다. 원시 ``langchain.agents.create_agent``와 config 기반 애플리케이션 factory인
``make_lead_agent`` 사이에 있는 SDK 수준 진입점이다.

Note: factory 조립 자체는 config를 읽지 않지만, 주입되는 일부 runtime 컴포넌트(예: subagent용
``task_tool``)는 호출 시점에 전역 config를 읽을 수 있다. 완전한 config-free runtime은
Phase 2 목표다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from deerflow.agents.features import RuntimeFeatures
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.tools.builtins import ask_clarification_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from deerflow.config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TodoMiddleware 프롬프트 (최소 SDK 버전)
# ---------------------------------------------------------------------------

_TODO_SYSTEM_PROMPT = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly
</todo_list_system>
"""

_TODO_TOOL_DESCRIPTION = "Use this tool to create and manage a structured task list for complex work sessions.  Only use for complex tasks (3+ steps)."


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def create_deerflow_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    features: RuntimeFeatures | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
    plan_mode: bool = False,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
) -> CompiledStateGraph:
    """평범한 Python 인자만으로 DeerFlow agent를 만든다.

    factory 조립 자체는 config 파일을 읽지 않는다. 주입되는 일부 runtime 컴포넌트(예:
    ``task_tool``)는 호출 시점에 전역 config에 의존할 수 있다. 완전한 config-free runtime은
    Phase 2 로드맵을 참고한다.

    Parameters
    ----------
    model:
        chat model 인스턴스.
    tools:
        사용자가 제공한 tool. feature가 주입하는 tool은 자동으로 덧붙는다.
    system_prompt:
        system message. ``None``이면 최소한의 기본값을 쓴다.
    middleware:
        **전체 대체** — 주면 이 리스트가 그대로 쓰인다.
        *features*, *extra_middleware*와 함께 쓸 수 없다.
    features:
        선언적 feature 플래그. *middleware*와 함께 쓸 수 없다.
    extra_middleware:
        ``@Next``/``@Prev`` 위치 지정으로 자동 조립된 chain에 끼워 넣는 추가 middleware.
        *middleware*와 함께 쓸 수 없다.
    plan_mode:
        작업 추적용 TodoMiddleware를 활성화한다.
    state_schema:
        LangGraph state 타입. 기본값은 ``ThreadState``다.
    checkpointer:
        선택적 persistence backend.
    name:
        agent 이름. 이를 사용하는 middleware(예: ``MemoryMiddleware``)에 전달된다.

    Raises
    ------
    ValueError
        *middleware*와 *features*/*extra_middleware*를 함께 준 경우.
    """
    if middleware is not None and features is not None:
        raise ValueError("Cannot specify both 'middleware' and 'features'.  Use one or the other.")
    if middleware is not None and extra_middleware:
        raise ValueError("Cannot use 'extra_middleware' with 'middleware' (full takeover).")
    if extra_middleware:
        for mw in extra_middleware:
            if not isinstance(mw, AgentMiddleware):
                raise TypeError(f"extra_middleware items must be AgentMiddleware instances, got {type(mw).__name__}")

    effective_tools: list[BaseTool] = list(tools or [])
    effective_state = ThreadState if state_schema is None else state_schema

    if middleware is not None:
        effective_middleware = list(middleware)
    else:
        feat = features or RuntimeFeatures()
        effective_middleware, extra_tools = _assemble_from_features(
            feat,
            name=name,
            plan_mode=plan_mode,
            extra_middleware=extra_middleware or [],
        )
        # tool 이름 기준으로 중복을 제거한다. 사용자가 준 tool이 우선이다.
        existing_names = {t.name for t in effective_tools}
        for t in extra_tools:
            if t.name not in existing_names:
                effective_tools.append(t)
                existing_names.add(t.name)

    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )


# ---------------------------------------------------------------------------
# 내부: feature 기반 middleware 조립
# ---------------------------------------------------------------------------


def _assemble_from_features(
    feat: RuntimeFeatures,
    *,
    name: str = "default",
    plan_mode: bool = False,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """*feat*으로부터 순서가 정해진 middleware chain과 추가 tool을 만든다.

    middleware 순서는 ``make_lead_agent``와 같다(middleware 14개).

      0-2. sandbox 인프라 (ThreadData → Uploads → Sandbox)
      3.   DanglingToolCallMiddleware (항상)
      4.   GuardrailMiddleware (guardrail 기능)
      5.   ToolErrorHandlingMiddleware (항상)
      6.   SummarizationMiddleware (summarization 기능)
      7.   TodoMiddleware (plan_mode 인자)
      8.   TitleMiddleware (auto_title 기능)
      9.   MemoryMiddleware (memory 기능)
      10.  ViewImageMiddleware (vision 기능)
      11.  SubagentLimitMiddleware (subagent 기능)
      12.  LoopDetectionMiddleware (loop_detection 기능)
      13.  ClarificationMiddleware (항상 마지막)

    순서 결정은 두 단계로 나뉜다.

      1. 내장 chain은 고정된 순서로 덧붙인다.
      2. 추가 middleware는 @Next/@Prev로 끼워 넣는다.

    각 feature 값은 다음과 같이 처리한다.

      - ``False``: 건너뛴다.
      - ``True``: 내장 기본 middleware를 만든다(``summarization``과 ``guardrail``은 불가.
        이 둘은 직접 만든 인스턴스가 필요하다).
      - ``AgentMiddleware`` 인스턴스: 그대로 사용한다(커스텀 대체).
    """
    chain: list[AgentMiddleware] = []
    extra_tools: list[BaseTool] = []

    # --- [0-2] sandbox 인프라 ---
    if feat.sandbox is not False:
        if isinstance(feat.sandbox, AgentMiddleware):
            chain.append(feat.sandbox)
        else:
            from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
            from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware
            from deerflow.sandbox.middleware import SandboxMiddleware

            chain.append(ThreadDataMiddleware(lazy_init=True))
            chain.append(UploadsMiddleware())
            chain.append(SandboxMiddleware(lazy_init=True))

    # --- [3] DanglingToolCall (항상) ---
    chain.append(DanglingToolCallMiddleware())

    # --- [4] Guardrail ---
    if feat.guardrail is not False:
        if isinstance(feat.guardrail, AgentMiddleware):
            chain.append(feat.guardrail)
        else:
            raise ValueError("guardrail=True requires a custom AgentMiddleware instance (no built-in GuardrailMiddleware yet)")

    # --- [5] ToolErrorHandling (항상) ---
    chain.append(ToolErrorHandlingMiddleware())

    # --- [6] Summarization ---
    if feat.summarization is not False:
        if isinstance(feat.summarization, AgentMiddleware):
            chain.append(feat.summarization)
        else:
            raise ValueError("summarization=True requires a custom AgentMiddleware instance (SummarizationMiddleware needs a model argument)")

    # --- [7] TodoMiddleware (plan_mode) ---
    if plan_mode:
        from deerflow.agents.middlewares.todo_middleware import TodoMiddleware

        chain.append(TodoMiddleware(system_prompt=_TODO_SYSTEM_PROMPT, tool_description=_TODO_TOOL_DESCRIPTION))

    # --- [8] Auto Title ---
    if feat.auto_title is not False:
        if isinstance(feat.auto_title, AgentMiddleware):
            chain.append(feat.auto_title)
        else:
            from deerflow.agents.middlewares.title_middleware import TitleMiddleware

            chain.append(TitleMiddleware())

    # --- [9] Memory ---
    if feat.memory is not False:
        if isinstance(feat.memory, AgentMiddleware):
            chain.append(feat.memory)
        else:
            from deerflow.config.memory_config import get_memory_config, should_use_memory_tools

            memory_cfg: MemoryConfig = feat.memory_config or get_memory_config()
            if should_use_memory_tools(memory_cfg):
                from deerflow.agents.memory.manager import backend_requires_passive_writes_in_tool_mode
                from deerflow.agents.memory.tools import get_memory_tools

                existing_names = {tool.name for tool in extra_tools}
                for memory_tool in get_memory_tools():
                    if memory_tool.name in existing_names:
                        logger.warning("Memory tool name %r already exists and was skipped.", memory_tool.name)
                        continue
                    extra_tools.append(memory_tool)
                    existing_names.add(memory_tool.name)
                if backend_requires_passive_writes_in_tool_mode(memory_cfg.manager_class):
                    from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

                    chain.append(MemoryMiddleware(agent_name=name, memory_config=memory_cfg))
            else:
                if memory_cfg.mode == "tool" and not memory_cfg.enabled:
                    logger.warning("memory.mode is 'tool' but memory.enabled is false; memory tools will not be registered.")
                from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

                chain.append(MemoryMiddleware(agent_name=name, memory_config=memory_cfg))

    # --- [10] Vision ---
    if feat.vision is not False:
        if isinstance(feat.vision, AgentMiddleware):
            chain.append(feat.vision)
        else:
            from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

            chain.append(ViewImageMiddleware())

        if feat.sandbox is not False:
            from deerflow.tools.builtins import view_image_tool

            extra_tools.append(view_image_tool)

    # --- [11] Subagent ---
    if feat.subagent is not False:
        if isinstance(feat.subagent, AgentMiddleware):
            chain.append(feat.subagent)
        else:
            from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware

            chain.append(SubagentLimitMiddleware())
        from deerflow.tools.builtins import task_tool

        extra_tools.append(task_tool)

    # --- [12] LoopDetection ---
    if feat.loop_detection is not False:
        if isinstance(feat.loop_detection, AgentMiddleware):
            chain.append(feat.loop_detection)
        else:
            from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
            from deerflow.config.loop_detection_config import LoopDetectionConfig

            chain.append(LoopDetectionMiddleware.from_config(LoopDetectionConfig()))

    # --- [13] TokenBudget ---
    if feat.token_budget is not False:
        if isinstance(feat.token_budget, AgentMiddleware):
            chain.append(feat.token_budget)
        else:
            from deerflow.agents.middlewares.token_budget_middleware import TokenBudgetMiddleware
            from deerflow.config.token_budget_config import TokenBudgetConfig

            chain.append(TokenBudgetMiddleware.from_config(TokenBudgetConfig()))

    # --- [14] Clarification (내장 middleware 중 항상 마지막) ---
    chain.append(ClarificationMiddleware())
    extra_tools.append(ask_clarification_tool)

    # --- @Next/@Prev로 extra_middleware를 끼워 넣는다 ---
    if extra_middleware:
        _insert_extra(chain, extra_middleware)
        # 불변식: ClarificationMiddleware는 항상 마지막이어야 한다.
        # @Next(ClarificationMiddleware)가 이를 끝에서 밀어낼 수 있다.
        clar_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
        if clar_idx != len(chain) - 1:
            chain.append(chain.pop(clar_idx))

    return chain, extra_tools


# ---------------------------------------------------------------------------
# 내부: @Next/@Prev를 사용한 추가 middleware 삽입
# ---------------------------------------------------------------------------


def _insert_extra(chain: list[AgentMiddleware], extras: list[AgentMiddleware]) -> None:
    """``@Next``/``@Prev`` anchor를 사용해 추가 middleware를 *chain*에 끼워 넣는다.

    알고리즘:
      1. 검증: @Next와 @Prev를 동시에 가진 middleware는 없어야 한다.
      2. 충돌 탐지: 두 추가 middleware가 같은 anchor를 가리키면(방향이 같든 반대든) 에러.
      3. anchor 없는 항목은 ClarificationMiddleware 앞에 넣는다.
      4. anchor 있는 항목은 반복적으로 넣는다(추가 middleware끼리의 anchor도 지원).
      5. 모든 라운드를 돌아도 anchor를 찾지 못하면 에러.
    """
    next_targets: dict[type, type] = {}
    prev_targets: dict[type, type] = {}

    anchored: list[tuple[AgentMiddleware, str, type]] = []
    unanchored: list[AgentMiddleware] = []

    for mw in extras:
        next_anchor = getattr(type(mw), "_next_anchor", None)
        prev_anchor = getattr(type(mw), "_prev_anchor", None)

        if next_anchor and prev_anchor:
            raise ValueError(f"{type(mw).__name__} cannot have both @Next and @Prev")

        if next_anchor:
            if next_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {next_targets[next_anchor].__name__} both @Next({next_anchor.__name__})")
            if next_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Next({next_anchor.__name__}) and {prev_targets[next_anchor].__name__} @Prev({next_anchor.__name__}) — use cross-anchoring between extras instead")
            next_targets[next_anchor] = type(mw)
            anchored.append((mw, "next", next_anchor))
        elif prev_anchor:
            if prev_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {prev_targets[prev_anchor].__name__} both @Prev({prev_anchor.__name__})")
            if prev_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Prev({prev_anchor.__name__}) and {next_targets[prev_anchor].__name__} @Next({prev_anchor.__name__}) — use cross-anchoring between extras instead")
            prev_targets[prev_anchor] = type(mw)
            anchored.append((mw, "prev", prev_anchor))
        else:
            unanchored.append(mw)

    # anchor 없는 항목은 ClarificationMiddleware 앞에 넣는다.
    clarification_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
    for mw in unanchored:
        chain.insert(clarification_idx, mw)
        clarification_idx += 1

    # anchor 있는 항목은 반복 삽입한다(외부 middleware끼리의 anchor도 지원).
    pending = list(anchored)
    max_rounds = len(pending) + 1
    for _ in range(max_rounds):
        if not pending:
            break
        remaining = []
        for mw, direction, anchor in pending:
            idx = next(
                (i for i, m in enumerate(chain) if isinstance(m, anchor)),
                None,
            )
            if idx is None:
                remaining.append((mw, direction, anchor))
                continue
            if direction == "next":
                chain.insert(idx + 1, mw)
            else:
                chain.insert(idx, mw)
        if len(remaining) == len(pending):
            names = [type(m).__name__ for m, _, _ in remaining]
            anchor_types = {a for _, _, a in remaining}
            remaining_types = {type(m) for m, _, _ in remaining}
            circular = anchor_types & remaining_types
            if circular:
                raise ValueError(f"Circular dependency among extra middlewares: {', '.join(t.__name__ for t in circular)}")
            raise ValueError(f"Cannot resolve positions for {', '.join(names)} — anchors {', '.join(a.__name__ for _, _, a in remaining)} not found in chain")
        pending = remaining
