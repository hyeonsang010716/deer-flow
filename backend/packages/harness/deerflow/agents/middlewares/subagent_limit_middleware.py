"""subagent 도구 호출 상한을 강제하는 미들웨어."""

import logging
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.tool_call_metadata import clone_ai_message_with_tool_calls
from deerflow.config.subagents_config import (
    DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN,
    MAX_CONCURRENT_SUBAGENT_CALLS,
    MAX_TOTAL_SUBAGENTS_PER_RUN,
    MIN_CONCURRENT_SUBAGENT_CALLS,
    MIN_TOTAL_SUBAGENTS_PER_RUN,
    clamp_subagent_concurrency,
    clamp_total_subagents_per_run,
)
from deerflow.subagents.executor import MAX_CONCURRENT_SUBAGENTS

logger = logging.getLogger(__name__)

# max_concurrent_subagents의 유효 범위
MIN_SUBAGENT_LIMIT = MIN_CONCURRENT_SUBAGENT_CALLS
MAX_SUBAGENT_LIMIT = MAX_CONCURRENT_SUBAGENT_CALLS
DEFAULT_MAX_TOTAL_SUBAGENTS = DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN
MIN_SUBAGENT_TOTAL_LIMIT = MIN_TOTAL_SUBAGENTS_PER_RUN
MAX_SUBAGENT_TOTAL_LIMIT = MAX_TOTAL_SUBAGENTS_PER_RUN

_TOTAL_LIMIT_STOP_MSG = (
    "[SUBAGENT LIMIT REACHED] The subagent delegation limit for this run has been reached. "
    "Continue using the subagent results already collected, execute remaining simple work "
    "directly, or summarize the remaining work instead of launching more subagents."
)


def _clamp_subagent_limit(value: int) -> int:
    """subagent 상한을 유효 범위 [1, 4]로 clamp한다."""
    return clamp_subagent_concurrency(value)


def _clamp_total_subagent_limit(value: int) -> int:
    """전체 subagent 상한을 유한한 양수 범위로 clamp한다."""
    return clamp_total_subagents_per_run(value)


def _append_text(content: Any, text: str) -> Any:
    if content is None:
        return text
    if isinstance(content, str):
        if content:
            return f"{content}\n\n{text}"
        return text
    if isinstance(content, list):
        return [*content, {"type": "text", "text": f"\n\n{text}"}]
    return f"{content}\n\n{text}"


def _delegation_id(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    entry_id = entry.get("id")
    return str(entry_id) if entry_id else None


def _delegation_run_id(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    run_id = entry.get("run_id")
    return str(run_id) if run_id else None


def _runtime_run_id(runtime: Runtime | None) -> str | None:
    context = getattr(runtime, "context", None)
    if not isinstance(context, dict):
        return None
    run_id = context.get("run_id")
    return str(run_id) if run_id else None


def _count_prior_delegations(delegations: object, *, run_id: str | None) -> int:
    if not isinstance(delegations, list):
        return 0
    ids = set()
    for entry in delegations:
        if run_id is not None and _delegation_run_id(entry) != run_id:
            continue
        delegation_id = _delegation_id(entry)
        if delegation_id is not None:
            ids.add(delegation_id)
    return len(ids)


class SubagentLimitMiddleware(AgentMiddleware[AgentState]):
    """한 model 응답 또는 run에서 초과된 'task' 도구 호출을 잘라낸다.

    LLM이 한 응답에서 max_concurrent를 넘는 병렬 task 도구 호출을 만들면, 앞의 max_concurrent개만 남기고
    나머지를 버린다. 또한 현재 run_id가 태깅된 durable delegation ledger 항목을 세어 run 전체 상한도
    강제한다. 덕분에 한 run 안에서 계획 체크포인트가 반복되어도 규격에 맞는 배치를 무한히 실행할 수 없다.
    prompt 기반 제한보다 신뢰할 수 있는 방식이다.

    Args:
        max_concurrent: 허용되는 동시 subagent 호출 수. 기본값은 MAX_CONCURRENT_SUBAGENTS(3)이며 [1, 4]로 clamp된다.
        max_total: run 전체에서 허용되는 subagent 호출 수. 기본값은 6이며 [1, 50]으로 clamp된다.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_SUBAGENTS, max_total: int = DEFAULT_MAX_TOTAL_SUBAGENTS):
        super().__init__()
        self.max_concurrent = _clamp_subagent_limit(max_concurrent)
        self.max_total = _clamp_total_subagent_limit(max_total)

    def _truncate_task_calls(self, state: AgentState, runtime: Runtime | None = None) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None

        # task 도구 호출 개수를 센다.
        task_indices = [i for i, tc in enumerate(tool_calls) if tc.get("name") == "task"]
        if not task_indices:
            return None

        run_id = _runtime_run_id(runtime)
        if run_id is None:
            logger.warning("Subagent limit middleware received no run_id; counting all thread delegations as prior usage. Pass run_id in runtime context to enforce the total cap per run.")
        prior_delegation_count = _count_prior_delegations(state.get("delegations"), run_id=run_id)
        remaining_total = max(0, self.max_total - prior_delegation_count)
        allowed_task_calls = min(self.max_concurrent, remaining_total)

        if len(task_indices) <= allowed_task_calls:
            return None

        # 버릴 인덱스 집합을 만든다(상한을 넘긴 task 호출).
        indices_to_drop = set(task_indices[allowed_task_calls:])
        truncated_tool_calls = [tc for i, tc in enumerate(tool_calls) if i not in indices_to_drop]
        dropped_count = len(indices_to_drop)
        logger.warning(
            "Truncated %s excess task tool call(s) from model response (concurrent limit: %s; total limit: %s; prior delegations: %s)",
            dropped_count,
            self.max_concurrent,
            self.max_total,
            prior_delegation_count,
        )

        # run 전체 상한을 소진하면 stop_reason을 남겨, worker가 loop_capped / token_capped /
        # safety_capped와 함께 이 capped 완료를 드러내도록 한다(#4176).
        if remaining_total == 0 and isinstance(getattr(runtime, "context", None), dict):
            runtime.context["stop_reason"] = "subagent_limit_capped"

        # 잘라낸 tool_calls로 AIMessage를 교체한다(같은 id면 교체로 처리된다).
        content = _append_text(last_msg.content, _TOTAL_LIMIT_STOP_MSG) if remaining_total == 0 else None
        updated_msg = clone_ai_message_with_tool_calls(last_msg, truncated_tool_calls, content=content)
        return {"messages": [updated_msg]}

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._truncate_task_calls(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._truncate_task_calls(state, runtime)
