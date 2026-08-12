"""thread 범위의 goal 상태와 evaluator 헬퍼.

Gateway run과 얇은 API 표면이 사용하는 Claude Code 스타일 goal 루프 primitive를 구현한다.
harness가 FastAPI app을 import하지 않고도 run을 평가하고 이어갈 수 있도록 의도적으로
``deerflow`` 안에 둔다.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import threading
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import empty_checkpoint, uuid6

import deerflow.utils.llm_text as llm_text
from deerflow.agents.goal_state import GoalBlocker, GoalEvaluation, GoalState
from deerflow.models import create_chat_model
from deerflow.tracing import inject_langfuse_metadata
from deerflow.utils.messages import message_to_text
from deerflow.utils.time import now_iso

logger = logging.getLogger(__name__)

DEFAULT_MAX_GOAL_CONTINUATIONS = 8
DEFAULT_MAX_NO_PROGRESS_CONTINUATIONS = 2
MAX_GOAL_OBJECTIVE_CHARS = 4000
MAX_GOAL_REASON_CHARS = 1000
MAX_GOAL_EVIDENCE_CHARS = 1000
MAX_GOAL_CONVERSATION_CHARS = 12000
MAX_GOAL_CONVERSATION_MESSAGES = 30

GOAL_BLOCKERS: set[GoalBlocker] = {
    "none",
    "missing_evidence",
    "needs_user_input",
    "run_failed",
    "external_wait",
    "goal_not_met_yet",
}
CONTINUABLE_GOAL_BLOCKERS: set[GoalBlocker] = {"goal_not_met_yet"}

GOAL_CLEAR_ALIASES = frozenset({"clear", "reset", "off"})

_extract_response_text = llm_text.extract_response_text
_strip_markdown_code_fence = llm_text.strip_markdown_code_fence
_strip_think_blocks = llm_text.strip_think_blocks

_goal_locks_guard = threading.Lock()
_goal_locks_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


class GoalWriteConflict(RuntimeError):
    """goal write가 오래된 checkpoint를 기준으로 할 때 raise된다."""


@asynccontextmanager
async def goal_thread_lock(thread_id: str) -> AsyncIterator[None]:
    """현재 event loop 안에서 goal의 read-modify-write 순서를 직렬화한다."""
    loop = asyncio.get_running_loop()
    with _goal_locks_guard:
        locks = _goal_locks_by_loop.get(loop)
        if locks is None:
            locks = {}
            _goal_locks_by_loop[loop] = locks
        lock = locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[thread_id] = lock

    async with lock:
        yield


class GoalCommand(NamedTuple):
    """``/goal`` slash 명령 인자 문자열을 파싱한 의도."""

    kind: Literal["status", "clear", "set"]
    objective: str = ""


def parse_goal_command(args: str) -> GoalCommand:
    """``/goal`` 명령의 인자 문자열을 의도로 파싱한다.

    TUI와 IM channel 표면이 공유하므로 세 갈래 의미가 한곳에 모인다. 비어 있으면 활성 goal을
    보여주고, ``clear``/``reset``/``off``는 goal을 지우며, 그 외에는 (공백을 정리한) 해당
    objective로 goal을 설정한다. frontend는 ``input-box-helpers.ts``에 같은 내용의 TypeScript
    사본을 유지한다.
    """
    stripped = args.strip()
    if not stripped:
        return GoalCommand("status")
    if stripped.lower() in GOAL_CLEAR_ALIASES:
        return GoalCommand("clear")
    return GoalCommand("set", stripped)


def normalize_goal_objective(objective: str) -> str:
    """사용자가 준 goal 텍스트를 정규화하고 검증한다."""
    normalized = " ".join(objective.strip().split())
    if not normalized:
        raise ValueError("Goal objective must not be empty.")
    if len(normalized) > MAX_GOAL_OBJECTIVE_CHARS:
        raise ValueError(f"Goal objective must be at most {MAX_GOAL_OBJECTIVE_CHARS} characters.")
    return normalized


def build_goal_state(
    objective: str,
    *,
    max_continuations: int = DEFAULT_MAX_GOAL_CONTINUATIONS,
    max_no_progress_continuations: int = DEFAULT_MAX_NO_PROGRESS_CONTINUATIONS,
    now: str | None = None,
) -> GoalState:
    """thread를 위한 새 active goal 상태를 만든다."""
    objective = normalize_goal_objective(objective)
    capped_max = max(0, min(int(max_continuations), DEFAULT_MAX_GOAL_CONTINUATIONS))
    timestamp = now or now_iso()
    return GoalState(
        objective=objective,
        status="active",
        created_at=timestamp,
        updated_at=timestamp,
        continuation_count=0,
        max_continuations=capped_max,
        no_progress_count=0,
        max_no_progress_continuations=max(0, int(max_no_progress_continuations)),
    )


def parse_goal_evaluation_response(text: str) -> GoalEvaluation:
    """evaluator의 JSON 객체 응답을 파싱한다."""
    candidate = _strip_markdown_code_fence(_strip_think_blocks(text))
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Goal evaluator response did not contain a JSON object.")
    try:
        payload = json.loads(candidate[start : end + 1])
    except Exception as exc:
        raise ValueError("Goal evaluator response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Goal evaluator JSON must be an object.")
    satisfied = payload.get("satisfied")
    if not isinstance(satisfied, bool):
        raise ValueError("Goal evaluator JSON must include boolean 'satisfied'.")
    reason = _normalize_evaluation_text(payload.get("reason"), max_chars=MAX_GOAL_REASON_CHARS)
    evidence_summary = _normalize_evaluation_text(payload.get("evidence_summary"), max_chars=MAX_GOAL_EVIDENCE_CHARS)
    blocker = _normalize_goal_blocker(payload.get("blocker"), satisfied=satisfied)
    return GoalEvaluation(
        satisfied=satisfied,
        blocker=blocker,
        reason=reason,
        evidence_summary=evidence_summary,
    )


def _normalize_evaluation_text(value: object, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_chars]


def _normalize_goal_blocker(value: object, *, satisfied: bool) -> GoalBlocker:
    if satisfied:
        return "none"
    if isinstance(value, str) and value in GOAL_BLOCKERS and value != "none":
        return value
    return "missing_evidence"


def _message_type(message: Any) -> str | None:
    value = getattr(message, "type", None)
    if value is None and isinstance(message, dict):
        value = message.get("type") or message.get("role")
    if value == "assistant":
        return "ai"
    if value == "user":
        return "human"
    return str(value) if value else None


def _additional_kwargs(message: Any) -> dict[str, Any]:
    value = getattr(message, "additional_kwargs", None)
    if value is None and isinstance(message, dict):
        value = message.get("additional_kwargs")
    return dict(value) if isinstance(value, dict) else {}


def _is_visible_message(message: Any) -> bool:
    if _additional_kwargs(message).get("hide_from_ui") is True:
        return False
    return _message_type(message) in {"human", "ai"}


def has_visible_assistant_evidence(messages: list[Any]) -> bool:
    """evaluator가 확인할 수 있는 visible AI 응답이 하나라도 있으면 true를 반환한다."""
    return any(_is_visible_message(message) and _message_type(message) == "ai" and bool(message_to_text(message).strip()) for message in messages)


def visible_conversation_signature(messages: list[Any]) -> str:
    """visible한 evaluator 근거에 대한 안정적이고 가벼운 signature를 반환한다."""
    visible = []
    for message in messages:
        if not _is_visible_message(message):
            continue
        visible.append(
            {
                "role": _message_type(message),
                "text": message_to_text(message).strip(),
            }
        )
    return json.dumps(visible[-MAX_GOAL_CONVERSATION_MESSAGES:], ensure_ascii=False, sort_keys=True)


def format_visible_conversation(messages: list[Any]) -> str:
    """goal 평가에 쓸 사용자에게 보이는 대화 근거를 반환한다."""
    lines: list[str] = []
    visible = [message for message in messages if _is_visible_message(message)]
    for message in visible[-MAX_GOAL_CONVERSATION_MESSAGES:]:
        text = message_to_text(message).strip()
        if not text:
            continue
        role = "User" if _message_type(message) == "human" else "Assistant"
        lines.append(f"{role}: {text}")
    conversation = "\n\n".join(lines)
    if len(conversation) > MAX_GOAL_CONVERSATION_CHARS:
        conversation = conversation[-MAX_GOAL_CONVERSATION_CHARS:]
    return conversation


def create_goal_evaluator_model(
    *,
    model_name: str | None = None,
    app_config: Any | None = None,
) -> Any:
    """goal evaluator가 사용하는 non-thinking chat model을 만든다.

    evaluator는 메인 graph run이 이미 끝난 뒤 ``runtime/runs/worker.py``에서 실행된다.
    graph root에 ``build_tracing_callbacks()``를 붙이고 이중 부착을 피하려고
    ``attach_tracing=False``를 넘기는 ``make_lead_agent``/``DeerFlowClient.stream``과 달리,
    여기에는 evaluator의 model 호출이 tracing을 물려받을 graph root가 없다. 따라서 다른
    독립 non-graph caller(``oneshot_llm.run_oneshot_llm``, ``MemoryUpdater``)처럼 자체
    model 수준 tracing callback을 붙여야 한다.
    """
    return create_chat_model(
        name=model_name,
        thinking_enabled=False,
        app_config=app_config,
        attach_tracing=True,
    )


def _resolve_environment() -> str | None:
    return os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT")


async def evaluate_goal_completion(
    goal: GoalState,
    messages: list[Any],
    *,
    model: Any | None = None,
    model_name: str | None = None,
    app_config: Any | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
    deerflow_trace_id: str | None = None,
) -> GoalEvaluation:
    """작은 non-thinking 모델에게 활성 goal이 충족됐는지 묻는다.

    ``thread_id``/``user_id``/``deerflow_trace_id``는 Langfuse trace metadata로만 전달된다
    (``oneshot_llm.run_oneshot_llm``과 같은 방식). 메인 graph 밖의 독립 model 호출이므로,
    graph root callback이 값을 끌어올려 주길 기대하지 않고 Langfuse session/user 귀속을 직접
    주입해야 한다. PR #2944(메인 graph), PR #3902(memory_agent/suggest_agent)와 같은 수정이다.
    """
    conversation = format_visible_conversation(messages)
    if not conversation or not has_visible_assistant_evidence(messages):
        return GoalEvaluation(
            satisfied=False,
            blocker="missing_evidence",
            reason="No visible assistant evidence is available yet.",
            evidence_summary="",
        )

    system_instruction = (
        "You are a strict completion evaluator for an AI coding assistant.\n"
        "Decide whether the active goal is fully satisfied using ONLY the visible conversation evidence.\n"
        "Do not assume files, commands, tests, or external state changed unless the conversation explicitly shows it.\n"
        "If the visible evidence is too weak to prove progress, fail closed with blocker missing_evidence.\n"
        "Use blocker needs_user_input when the assistant is waiting on the user, run_failed when the turn failed, "
        "external_wait when work is waiting on an outside system, goal_not_met_yet when useful autonomous work can continue, "
        "and none only when satisfied is true.\n"
        'Output exactly one JSON object: {"satisfied": boolean, "blocker": string, "reason": string, "evidence_summary": string}.'
    )
    user_content = f"Active goal:\n{goal['objective']}\n\nVisible conversation evidence:\n{conversation}\n\nIs the active goal fully satisfied?"

    if model is None:
        model = create_goal_evaluator_model(model_name=model_name, app_config=app_config)
    invoke_config: dict[str, Any] = {"run_name": "goal_evaluator"}
    inject_langfuse_metadata(
        invoke_config,
        thread_id=thread_id,
        user_id=user_id,
        assistant_id="goal_evaluator",
        model_name=model_name,
        environment=_resolve_environment(),
        deerflow_trace_id=deerflow_trace_id,
    )
    response = await model.ainvoke(
        [SystemMessage(content=system_instruction), HumanMessage(content=user_content)],
        config=invoke_config,
    )
    return parse_goal_evaluation_response(_extract_response_text(response.content))


def should_continue_goal(goal: GoalState, evaluation: GoalEvaluation, *, no_progress_count: int | None = None) -> bool:
    """숨겨진 continuation 턴을 한 번 더 실행해야 하는지 반환한다."""
    if evaluation["satisfied"]:
        return False
    if evaluation["blocker"] not in CONTINUABLE_GOAL_BLOCKERS:
        return False
    if int(goal.get("continuation_count", 0)) >= int(goal.get("max_continuations", DEFAULT_MAX_GOAL_CONTINUATIONS)):
        return False
    current_no_progress = int(goal.get("no_progress_count", 0) if no_progress_count is None else no_progress_count)
    max_no_progress = int(goal.get("max_no_progress_continuations", DEFAULT_MAX_NO_PROGRESS_CONTINUATIONS))
    return current_no_progress < max_no_progress


def latest_visible_assistant_signature(messages: list[Any]) -> str:
    """가장 최근의 visible assistant 근거에 대한 안정적인 signature를 반환한다.

    "no progress" breaker는 evaluator의 자유 서술 ``reason``/``evidence_summary``가 아니라
    agent가 실제로 만들어낸 것, 즉 가장 최근 사용자에게 보이는 assistant 메시지의 텍스트를
    키로 쓴다(LLM은 매 턴 그 서술을 바꿔 쓰므로 바이트 단위로 반복되는 일이 거의 없다).
    continuation이 새로운 visible assistant 출력을 추가하지 않으면 signature가 그대로여서
    breaker가 정체된 턴을 인식할 수 있다.
    """
    for message in reversed(messages):
        if not _is_visible_message(message) or _message_type(message) != "ai":
            continue
        text = message_to_text(message).strip()
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ""


def compute_goal_progress_key(evaluation: GoalEvaluation, *, evidence_signature: str = "") -> str:
    """진전 없는 평가가 반복되는 것을 감지하는 데 쓰는 안정적인 키를 반환한다.

    타입이 지정된 ``blocker``와 visible assistant 근거의 signature를 키로 삼으므로,
    evaluator가 자유 서술 ``reason``/``evidence_summary``를 바꿔 써도 정체된 goal을 감지한다.
    """
    return json.dumps(
        {
            "satisfied": evaluation["satisfied"],
            "blocker": evaluation["blocker"],
            "evidence_signature": evidence_signature,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def compute_no_progress_count(goal: GoalState, evaluation: GoalEvaluation, *, evidence_signature: str = "") -> int:
    """visible 근거가 나아가지 않았으면 반복 카운트를 증가시킨다."""
    if evaluation["satisfied"]:
        return 0
    progress_key = compute_goal_progress_key(evaluation, evidence_signature=evidence_signature)
    previous = goal.get("last_evaluation", {})
    if isinstance(previous, dict) and previous.get("progress_key") == progress_key:
        return int(goal.get("no_progress_count", 0)) + 1
    return 0


def make_goal_continuation_message(goal: GoalState, evaluation: GoalEvaluation) -> HumanMessage:
    """agent에게 작업을 계속하라고 요청하는 숨겨진 user 메시지를 만든다."""
    content = (
        "<goal_continuation>\n"
        f"Active goal: {goal['objective']}\n"
        f"Evaluator result: not satisfied. Blocker: {evaluation['blocker']}. Reason: {evaluation['reason'] or 'No reason provided.'}\n"
        f"Visible evidence: {evaluation.get('evidence_summary') or 'No evidence summary provided.'}\n"
        "Continue working toward the active goal. Use the available tools and conversation context. "
        "Do not ask the user to continue unless you are genuinely blocked.\n"
        "</goal_continuation>"
    )
    return HumanMessage(
        content=content,
        additional_kwargs={
            "hide_from_ui": True,
            "deerflow_goal_continuation": True,
        },
    )


async def _call_checkpointer_method(checkpointer: Any, async_name: str, sync_name: str, *args: Any, **kwargs: Any) -> Any:
    async_method = getattr(checkpointer, async_name, None)
    if async_method is not None:
        result = async_method(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result
    sync_method = getattr(checkpointer, sync_name, None)
    if sync_method is None:
        raise AttributeError(f"Missing checkpointer method: {async_name}/{sync_name}")
    # 동기 checkpointer 호출을 offload해 그 blocking IO가 event loop에서 실행되지 않게 한다
    # (backend/AGENTS.md의 blocking-IO gate).
    result = await asyncio.to_thread(sync_method, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _next_channel_version(checkpointer: Any, current_version: Any) -> Any:
    get_next_version = getattr(checkpointer, "get_next_version", None)
    if callable(get_next_version):
        return get_next_version(current_version, None)
    if isinstance(current_version, int):
        return current_version + 1
    return 1


async def ensure_thread_checkpoint(checkpointer: Any, thread_id: str) -> None:
    """*thread_id*에 checkpoint가 없으면 빈 root checkpoint를 만든다."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", config)
    if checkpoint_tuple is not None:
        return
    metadata = {
        "step": -1,
        "source": "input",
        "writes": None,
        "parents": {},
        "created_at": now_iso(),
    }
    await _call_checkpointer_method(checkpointer, "aput", "put", config, empty_checkpoint(), metadata, {})


def _checkpoint_id_from_tuple(checkpoint_tuple: Any) -> str | None:
    config = getattr(checkpoint_tuple, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    checkpoint_id = configurable.get("checkpoint_id") if isinstance(configurable, dict) else None
    if isinstance(checkpoint_id, str):
        return checkpoint_id
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("id"), str):
        return checkpoint["id"]
    return None


async def read_thread_goal(checkpointer: Any, thread_id: str) -> GoalState | None:
    """checkpoint 상태에서 최신 thread goal을 읽는다."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", config)
    if checkpoint_tuple is None:
        return None
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
    raw_goal = channel_values.get("goal") if isinstance(channel_values, dict) else None
    return copy.deepcopy(raw_goal) if isinstance(raw_goal, dict) else None


async def write_thread_goal(
    checkpointer: Any,
    thread_id: str,
    goal: GoalState | None,
    *,
    as_node: str = "goal",
    create_if_missing: bool = False,
    expected_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """thread goal을 설정하거나 지운 새 checkpoint를 쓴다.

    갱신된 channel value를 반환한다.
    """
    if create_if_missing:
        await ensure_thread_checkpoint(checkpointer, thread_id)

    read_config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", read_config)
    if checkpoint_tuple is None:
        raise LookupError(f"Thread {thread_id} checkpoint not found")
    if expected_checkpoint_id is not None and _checkpoint_id_from_tuple(checkpoint_tuple) != expected_checkpoint_id:
        raise GoalWriteConflict(f"Thread {thread_id} goal checkpoint changed while preparing write")

    checkpoint: dict[str, Any] = dict(getattr(checkpoint_tuple, "checkpoint", {}) or {})
    metadata: dict[str, Any] = dict(getattr(checkpoint_tuple, "metadata", {}) or {})
    channel_values: dict[str, Any] = dict(checkpoint.get("channel_values", {}) or {})

    if goal is None:
        channel_values.pop("goal", None)
    else:
        channel_values["goal"] = copy.deepcopy(goal)

    channel_versions = dict(checkpoint.get("channel_versions", {}) or {})
    current_version = channel_versions.get("goal")
    next_version = _next_channel_version(checkpointer, current_version)
    channel_versions["goal"] = next_version

    checkpoint["channel_values"] = channel_values
    checkpoint["channel_versions"] = channel_versions
    checkpoint["id"] = str(uuid6())
    metadata["updated_at"] = now_iso()
    metadata["source"] = "update"
    metadata["step"] = metadata.get("step", 0) + 1
    metadata["writes"] = {as_node: {"goal": goal}}

    write_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            # 새 checkpoint의 부모를 그것이 파생된 checkpoint로 지정한다.
            # 이렇게 하지 않으면 saver가 부모 없는 checkpoint를 저장해 Delta-channel replay의
            # 조상 연결이 끊기고(full 모드에서도 history 순회가 잘린다).
            "checkpoint_id": _checkpoint_id_from_tuple(checkpoint_tuple),
        }
    }
    await _call_checkpointer_method(checkpointer, "aput", "put", write_config, checkpoint, metadata, {"goal": next_version})
    return channel_values


def attach_goal_evaluation(
    goal: GoalState,
    evaluation: GoalEvaluation,
    *,
    run_id: str,
    continuation_count: int | None = None,
    no_progress_count: int | None = None,
    stand_down_reason: str | None = None,
    evidence_signature: str = "",
) -> GoalState:
    """최신 evaluator 결과를 붙인 goal 사본을 반환한다."""
    next_goal = copy.deepcopy(goal)
    if continuation_count is not None:
        next_goal["continuation_count"] = continuation_count
    if no_progress_count is not None:
        next_goal["no_progress_count"] = no_progress_count
    next_goal["updated_at"] = now_iso()
    next_goal["last_evaluation"] = {
        "satisfied": evaluation["satisfied"],
        "blocker": evaluation["blocker"],
        "reason": evaluation["reason"],
        "evidence_summary": evaluation.get("evidence_summary", ""),
        "run_id": run_id,
        "evaluated_at": next_goal["updated_at"],
        "progress_key": compute_goal_progress_key(evaluation, evidence_signature=evidence_signature),
    }
    if stand_down_reason:
        next_goal["last_evaluation"]["stand_down_reason"] = stand_down_reason
    return next_goal
