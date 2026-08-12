import copy
import uuid
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Annotated, Any, NotRequired, TypedDict, cast, get_type_hints

from langchain.agents import AgentState
from langchain_core.messages import (
    AnyMessage,
    BaseMessageChunk,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langgraph.channels import DeltaChannel
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import deerflow.checkpoint_patches as _checkpoint_patches  # noqa: F401 - import-time saver fixes
from deerflow.agents.goal_state import GoalState
from deerflow.config.database_config import DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY, CheckpointChannelMode
from deerflow.subagents.status_contract import SUBAGENT_STATUS_VALUES


def _resolve_snapshot_frequency(snapshot_frequency: int | None) -> int:
    """유효한 cadence를 결정한다. 명시값이 있으면 그것, 없으면 프로세스에 frozen된 값,
    그것도 없으면 기본값을 쓴다. lazy import인 이유는 ``deerflow.runtime.__init__``이
    ``checkpoint_state``를 거쳐 이 모듈에 닿기 때문에 top-level import가 순환이 되기 때문이다."""
    if snapshot_frequency is not None:
        return snapshot_frequency
    from deerflow.runtime.checkpoint_mode import resolve_checkpoint_snapshot_frequency

    return resolve_checkpoint_snapshot_frequency()


class SandboxState(TypedDict):
    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]


class ViewedImageData(TypedDict):
    """열람한 이미지 파일의 metadata.

    checkpoint state에는 가벼운 metadata만 저장하고, 실제 이미지 바이트는 모델이 필요로 할 때
    디스크에서 그때그때 읽는다. 큰 base64 payload가 모든 checkpoint에 중복 저장되는 것을
    막는다(#4138 참고).
    """

    mime_type: str
    size: int
    actual_path: str


def merge_sandbox(existing: SandboxState | None, new: SandboxState | None) -> SandboxState | None:
    """sandbox state용 reducer. 멱등한 쓰기만 허용한다.

    여러 sandbox 도구가 같은 graph step에서 lazy 초기화되면서 Command(update=...)로 같은
    sandbox_id를 낼 수 있다. LangGraph는 그 공유 state 키에 대한 명시적 reducer를 요구한다.
    같은 thread에서 sandbox id가 다르다면 lifecycle/격리 버그이므로, 하나를 조용히 고르지 않고
    fail closed 한다.
    """
    if new is None:
        return existing
    if existing is None:
        return new

    existing_id = existing.get("sandbox_id")
    new_id = new.get("sandbox_id")
    if existing_id == new_id:
        return existing
    raise ValueError(f"Conflicting sandbox state updates: {existing_id!r} != {new_id!r}")


SandboxStateField = Annotated[NotRequired[SandboxState | None], merge_sandbox]


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """artifacts 목록용 reducer. artifact를 합치고 중복을 제거한다."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    # 순서를 유지하면서 중복을 제거하기 위해 dict.fromkeys를 쓴다
    return list(dict.fromkeys(existing + new))


def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """viewed_images dict용 reducer. 이미지 dict를 병합한다.

    특수 케이스: new가 빈 dict {}이면 기존 이미지를 모두 지운다. middleware가 처리 후
    viewed_images state를 비울 수 있게 하기 위해서다.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # 특수 케이스: 빈 dict는 열람 이미지 전체 삭제를 뜻한다
    if len(new) == 0:
        return {}
    # dict를 병합한다. 키가 겹치면 new 값이 기존 값을 덮어쓴다
    return {**existing, **new}


def merge_todos(existing: list | None, new: list | None) -> list | None:
    """todos 목록용 reducer. 마지막 non-None 값을 유지한다.

    의미:
    - `new`가 None이면(노드가 todos를 건드리지 않음) `existing`을 보존한다.
    - `new`가 주어지면(빈 list라도) 명시적 갱신으로 보고 `existing`을 이긴다.
    """
    if new is None:
        return existing
    return new


def merge_goal(existing: GoalState | None, new: GoalState | None) -> GoalState | None:
    """goal state용 reducer. 노드가 건드리지 않으면 기존 값을 보존한다."""
    if new is None:
        return existing
    return new


class PromotedTools(TypedDict):
    catalog_hash: str
    names: list[str]


def merge_promoted(existing: PromotedTools | None, new: PromotedTools | None) -> PromotedTools | None:
    """catalog hash로 범위가 정해지는 deferred-tool promotion용 reducer.

    - new가 None/빈 값 -> existing 보존(노드가 promotion을 건드리지 않음).
    - catalog_hash가 바뀜 -> 통째로 교체하고 낡은 이름을 버린다(catalog가 바뀐 뒤에도 저장된
      맨이름이 다른 도구를 노출하는 것을 막는다).
    - catalog_hash가 같음 -> 이름을 합집합으로 모으고 중복을 제거하며 순서를 유지한다.
    """
    if not new:
        return existing
    if existing is None or existing.get("catalog_hash") != new["catalog_hash"]:
        return {
            "catalog_hash": new["catalog_hash"],
            "names": list(dict.fromkeys(new["names"])),
        }
    return {
        "catalog_hash": existing["catalog_hash"],
        "names": list(dict.fromkeys(existing["names"] + new["names"])),
    }


TERMINAL_STATUSES: frozenset[str] = frozenset(SUBAGENT_STATUS_VALUES)
_DELEGATION_LEDGER_MAX_ENTRIES = 50


class DelegationEntry(TypedDict):
    id: str
    run_id: NotRequired[str]
    description: str
    subagent_type: str
    status: str
    result_brief: NotRequired[str]
    result_sha256: NotRequired[str]
    result_ref: NotRequired[str]
    # guardrail cap이 run을 일찍 끝낸 이유(#3875 Phase 2): token_capped / turn_capped /
    # loop_capped. status는 completed/failed 그대로이고, 이 필드가 capped run과 정상 run을
    # 구분하는 추가 신호다.
    stop_reason: NotRequired[str]
    created_at: str


def merge_delegations(existing: list[DelegationEntry] | None, new: list[DelegationEntry] | None) -> list[DelegationEntry]:
    """delegation ledger용 reducer.

    - new가 None/빈 값 -> existing 보존.
    - 항목을 덧붙이되, 같은 id는 최신 버전으로 교체하고 처음 등장한 순서를 유지한다.
    - terminal status는 non-terminal status로 절대 덮어쓰지 않는다.
    """
    if not new:
        return existing or []

    by_id: dict[str, DelegationEntry] = {}
    order: list[str] = []
    for entry in [*(existing or []), *new]:
        entry_id = entry["id"]
        previous = by_id.get(entry_id)
        if previous is not None and previous["status"] in TERMINAL_STATUSES and entry["status"] not in TERMINAL_STATUSES:
            continue
        if entry_id not in by_id:
            order.append(entry_id)
        elif previous.get("created_at"):
            entry = {**entry, "created_at": previous["created_at"]}
            if previous.get("run_id") and not entry.get("run_id"):
                entry["run_id"] = previous["run_id"]
        by_id[entry_id] = entry
    merged = [by_id[entry_id] for entry_id in order]
    if len(merged) > _DELEGATION_LEDGER_MAX_ENTRIES:
        merged = merged[-_DELEGATION_LEDGER_MAX_ENTRIES:]
    return merged


_SKILL_CONTEXT_MAX_ENTRIES = 8
_SKILL_DESCRIPTION_MAX_CHARS = 500


class SkillEntry(TypedDict):
    name: str
    path: str
    description: str
    loaded_at: int


def _normalize_skill_entry(entry: Mapping[str, object]) -> SkillEntry:
    """skill_context를 state에 다시 저장하기 전에 레거시 payload 키를 버린다."""
    description = entry.get("description")
    loaded_at = entry.get("loaded_at")
    return {
        "name": str(entry.get("name") or ""),
        "path": str(entry["path"]),
        "description": " ".join(description.split())[:_SKILL_DESCRIPTION_MAX_CHARS] if isinstance(description, str) else "",
        "loaded_at": loaded_at if isinstance(loaded_at, int) else 0,
    }


def merge_skill_context(existing: list[SkillEntry] | None, new: list[SkillEntry] | None) -> list[SkillEntry]:
    """skill-context channel용 reducer.

    - new가 None/빈 값 -> existing 보존.
    - 레거시 항목은 reference로 정규화하고, 본문을 그대로 담은 키는 버린다.
    - ``path``로 중복을 제거한다. 나중에 읽으면 최신성이 갱신되고 reference가 교체된다.
    - 가장 최근에 읽은 항목만 남겨 개수를 제한한다. compaction 후 message index가 초기화되므로
      ``loaded_at``은 참고용일 뿐이다.
    """
    normalized_existing = [_normalize_skill_entry(entry) for entry in existing or []]
    if not new:
        return normalized_existing

    by_path: dict[str, SkillEntry] = {}
    order: list[str] = []
    for entry in normalized_existing:
        path = entry["path"]
        if path not in by_path:
            order.append(path)
        by_path[path] = entry

    for entry in (_normalize_skill_entry(entry) for entry in new):
        path = entry["path"]
        if path in by_path:
            order.remove(path)
        order.append(path)
        by_path[path] = entry

    merged = [by_path[path] for path in order]
    if len(merged) > _SKILL_CONTEXT_MAX_ENTRIES:
        merged = merged[-_SKILL_CONTEXT_MAX_ENTRIES:]
    return merged


class ThreadState(AgentState):
    sandbox: SandboxStateField
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list[str], merge_artifacts]
    todos: Annotated[list | None, merge_todos]
    goal: Annotated[GoalState | None, merge_goal]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # image_path -> metadata (base64 없음)
    promoted: Annotated[PromotedTools | None, merge_promoted]
    delegations: Annotated[list[DelegationEntry], merge_delegations]
    skill_context: Annotated[list[SkillEntry], merge_skill_context]
    summary_text: NotRequired[str | None]


def _normalize_messages(value: Any) -> list[AnyMessage]:
    values = value if isinstance(value, list) else [value]
    messages = [message_chunk_to_message(cast(BaseMessageChunk, message)) for message in convert_to_messages(values)]
    for message in messages:
        if message.id is None:
            message.id = str(uuid.uuid4())
    return messages


def _index_messages(
    messages: list[AnyMessage | None],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    latest_position: dict[str, int] = {}
    positions_by_id: dict[str, list[int]] = {}
    for position, message in enumerate(messages):
        if message is None:
            continue
        message_id = cast(str, message.id)
        latest_position[message_id] = position
        positions_by_id.setdefault(message_id, []).append(position)
    return latest_position, positions_by_id


def _raise_null_write(has_messages: bool) -> None:
    # ``add_messages(left, None)``은 누적된 message list가 비어 있지 않으면 ``left``만,
    # 비어 있으면 ``right``만 보고한다.
    received = "left" if has_messages else "right"
    raise ValueError(f"Must specify non-null arguments for both 'left' and 'right'. Only received: '{received}'.")


def merge_message_writes(state: list[AnyMessage], writes: Sequence[Any]) -> list[AnyMessage]:
    """DeltaChannel 쓰기를 ``add_messages`` 의미대로 선형 시간에 접는다.

    LangGraph의 비공개 ``_messages_delta_reducer``도 선형이지만, 공개 reducer의 coercion,
    ID, 삭제, ``REMOVE_ALL_MESSAGES`` 동작을 온전히 보존하지는 않는다.
    """
    if not writes:
        return list(state)
    if writes[0] is None:
        _raise_null_write(bool(state))

    messages: list[AnyMessage | None] = _normalize_messages(state)
    latest_position, positions_by_id = _index_messages(messages)

    for write in writes:
        if write is None:
            _raise_null_write(bool(latest_position))
        normalized_write = _normalize_messages(write)
        remove_all_idx = None
        for position, message in enumerate(normalized_write):
            if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES:
                remove_all_idx = position

        if remove_all_idx is not None:
            messages = list(normalized_write[remove_all_idx + 1 :])
            latest_position, positions_by_id = _index_messages(messages)
            continue

        ids_to_remove: set[str] = set()
        for message in normalized_write:
            message_id = cast(str, message.id)
            existing_position = latest_position.get(message_id)
            if existing_position is not None:
                if isinstance(message, RemoveMessage):
                    ids_to_remove.add(message_id)
                else:
                    ids_to_remove.discard(message_id)
                    messages[existing_position] = message
                continue

            if isinstance(message, RemoveMessage):
                raise ValueError(f"Attempting to delete a message with an ID that doesn't exist ('{message_id}')")

            position = len(messages)
            messages.append(message)
            latest_position[message_id] = position
            positions_by_id[message_id] = [position]

        for message_id in ids_to_remove:
            for position in positions_by_id.pop(message_id):
                messages[position] = None
            del latest_position[message_id]

    return [message for message in messages if message is not None]


def delta_messages_field(snapshot_frequency: int = DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY) -> Any:
    """주어진 cadence의 ``DeltaChannel``을 붙인 messages 필드 annotation."""
    return Annotated[
        list[AnyMessage],
        DeltaChannel(merge_message_writes, snapshot_frequency=snapshot_frequency),
    ]


DELTA_MESSAGES_FIELD = delta_messages_field()


class DeltaThreadState(ThreadState):
    messages: DELTA_MESSAGES_FIELD


THREAD_STATE_REDUCER_FIELDS = frozenset(
    {
        "messages",
        "sandbox",
        "artifacts",
        "todos",
        "goal",
        "viewed_images",
        "promoted",
        "delegations",
        "skill_context",
    }
)


def get_thread_state_schema(mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> type:
    if mode != "delta":
        return ThreadState
    return _delta_thread_state_schema(_resolve_snapshot_frequency(snapshot_frequency))


@cache
def _delta_thread_state_schema(snapshot_frequency: int) -> type:
    """cadence를 키로 하는 delta thread schema. 기본값에서는 정적 ``DeltaThreadState``의
    identity를 유지해 기존 타입 검사가 계속 통과하게 한다."""
    if snapshot_frequency == DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY:
        return DeltaThreadState
    annotations = get_type_hints(ThreadState, include_extras=True)
    annotations["messages"] = delta_messages_field(snapshot_frequency)
    return TypedDict(
        f"DeltaThreadState_f{snapshot_frequency}",
        annotations,
        total=getattr(ThreadState, "__total__", True),
    )


def adapt_state_schema_for_mode(schema: type, mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> type:
    if mode == "full":
        return schema
    return _adapt_state_schema_for_delta(schema, _resolve_snapshot_frequency(snapshot_frequency))


@cache
def _adapt_state_schema_for_delta(schema: type, snapshot_frequency: int) -> type:
    annotations = get_type_hints(schema, include_extras=True)
    annotations["messages"] = delta_messages_field(snapshot_frequency)
    return TypedDict(
        f"Delta{schema.__module__.replace('.', '_')}_{schema.__name__}_f{snapshot_frequency}",
        annotations,
        total=getattr(schema, "__total__", True),
    )


def normalize_middleware_state_schemas(middleware: Sequence[Any], mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> list[Any]:
    if mode == "full":
        return list(middleware)
    resolved_frequency = _resolve_snapshot_frequency(snapshot_frequency)
    normalized = []
    for item in middleware:
        schema = getattr(item, "state_schema", None)
        if schema is None:
            normalized.append(item)
            continue
        adapted = copy.copy(item)
        adapted.state_schema = adapt_state_schema_for_mode(schema, mode, resolved_frequency)
        normalized.append(adapted)
    return normalized
