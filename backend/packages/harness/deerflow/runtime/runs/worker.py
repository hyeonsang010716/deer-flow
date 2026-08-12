"""background agent 실행.

agent graph를 ``asyncio.Task`` 안에서 실행하며, 생성되는 이벤트를
:class:`StreamBridge`에 publish한다.

``graph.astream(stream_mode=[...])``를 사용한다. ``values`` 모드에서는 올바른
전체 state snapshot을, ``updates``에서는 제대로 된 ``{node: writes}``를,
``messages`` 모드에서는 ``(chunk, metadata)`` 튜플을 준다.

참고: ``events`` 모드는 gateway가 거부한다. ``graph.astream_events()``가 필요한데
이것은 ``values`` snapshot을 동시에 만들 수 없기 때문이다. JS 오픈소스 LangGraph
API 서버는 Python public API에 노출되지 않은 내부 checkpoint callback으로 이를
우회한다.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
import sys
import threading
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, cast

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.types import Overwrite

from deerflow.agents.goal_state import GoalEvaluation, GoalState
from deerflow.config.app_config import AppConfig
from deerflow.config.database_config import CheckpointChannelMode
from deerflow.constants import TOOL_RESULTS_DIRNAME
from deerflow.runtime.checkpoint_mode import (
    aensure_checkpoint_mode_compatible,
    inject_checkpoint_mode,
)
from deerflow.runtime.checkpoint_state import (
    CheckpointStateAccessor,
    build_state_mutation_graph,
    graph_reducer_channels,
    graph_state_schema,
    graph_writable_channels,
)
from deerflow.runtime.context_keys import CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY
from deerflow.runtime.goal import (
    DEFAULT_MAX_GOAL_CONTINUATIONS,
    DEFAULT_MAX_NO_PROGRESS_CONTINUATIONS,
    GoalWriteConflict,
    _call_checkpointer_method,
    _is_visible_message,
    _message_type,
    attach_goal_evaluation,
    compute_no_progress_count,
    create_goal_evaluator_model,
    evaluate_goal_completion,
    goal_thread_lock,
    latest_visible_assistant_signature,
    make_goal_continuation_message,
    read_thread_goal,
    should_continue_goal,
    visible_conversation_signature,
    write_thread_goal,
)
from deerflow.runtime.serialization import serialize
from deerflow.runtime.stream_bridge import StreamBridge
from deerflow.runtime.stream_modes import normalize_stream_modes, to_langgraph_stream_modes
from deerflow.runtime.user_context import get_effective_user_id, resolve_runtime_user_id
from deerflow.trace_context import (
    DEERFLOW_TRACE_METADATA_KEY,
    is_trace_id_from_request_header,
    resolve_deerflow_trace_id,
)
from deerflow.tracing import inject_langfuse_metadata
from deerflow.utils.messages import message_to_text
from deerflow.workspace_changes import capture_workspace_snapshot, get_changed_output_paths, record_workspace_changes
from deerflow.workspace_changes.types import WorkspaceSnapshot

from .manager import RunManager, RunRecord, RunStartOutcome
from .naming import resolve_root_run_name
from .schemas import RunStatus

logger = logging.getLogger(__name__)

_checkpoint_locks_guard = threading.Lock()
_checkpoint_locks_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


@asynccontextmanager
async def _checkpoint_thread_lock(thread_id: str) -> AsyncIterator[None]:
    """goal 명령을 막지 않으면서 한 thread의 checkpoint 변경을 직렬화한다."""
    loop = asyncio.get_running_loop()
    with _checkpoint_locks_guard:
        locks = _checkpoint_locks_by_loop.get(loop)
        if locks is None:
            locks = {}
            _checkpoint_locks_by_loop[loop] = locks
        lock = locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[thread_id] = lock

    async with lock:
        yield


_DELIVERY_RECEIPT_RETRY_DELAYS_SECONDS = (0.1, 0.5)


async def _persist_delivery_receipt(
    event_store: Any,
    *,
    thread_id: str,
    run_id: str,
    content: dict[str, Any],
) -> bool:
    """짧은 유한 retry로 terminal receipt를 저장한다.

    이 coroutine이 도는 동안에도 소유 worker는 실제 terminal 결과를 알고 있고 lease를
    갱신한다. 여기서 retry하면 일시적인 event store 실패를 흡수하면서, 성공한 run을
    orphan recovery에 넘기지 않는다. orphan recovery는 terminal status도 상세 receipt도
    복원할 수 없기 때문이다.
    """
    attempts = len(_DELIVERY_RECEIPT_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(attempts):
        try:
            await event_store.put_if_absent(
                thread_id=thread_id,
                run_id=run_id,
                event_type="run.delivery",
                category="outputs",
                content=content,
            )
            return True
        except Exception:
            if attempt == attempts - 1:
                logger.warning(
                    "Failed to persist delivery receipt for run %s after %d attempts; applying terminal delivery semantics without a receipt",
                    run_id,
                    attempts,
                    exc_info=True,
                )
                return False
            delay = _DELIVERY_RECEIPT_RETRY_DELAYS_SECONDS[attempt]
            logger.warning(
                "Failed to persist delivery receipt for run %s (attempt %d/%d); retrying in %.1fs",
                run_id,
                attempt + 1,
                attempts,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)

    return False  # pragma: no cover - loop always returns


_DELIVERY_INCOMPLETE_ERROR = "Artifact delivery incomplete: no produced output artifact was presented"
_DELIVERY_RECEIPT_FAILED_ERROR = "Artifact delivery verification failed: terminal delivery receipt could not be persisted"


def _empty_delivery_content() -> dict[str, Any]:
    return {"presented": 0, "paths": [], "by_tool": {}}


def _presented_path_covers_output(presented_path: str, produced_path: str) -> bool:
    presented_path = presented_path.rstrip("/")
    return bool(presented_path) and (produced_path == presented_path or produced_path.startswith(f"{presented_path}/"))


def _delivery_content_with_outputs(
    content: dict[str, Any],
    produced_paths: list[str],
) -> dict[str, Any]:
    """이 run이 output을 생성하거나 수정했다면 delivery 판정을 덧붙인다."""
    if not produced_paths:
        return content

    presented_paths = content.get("by_tool", {}).get("present_files", [])
    matched_paths = [produced_path for produced_path in produced_paths if any(_presented_path_covers_output(presented_path, produced_path) for presented_path in presented_paths)]
    satisfied = bool(matched_paths)
    return {
        **content,
        "verification": {
            "source": "outputs_changed",
            "requirement": "present_files_matches_produced_output",
        },
        "produced_paths": produced_paths,
        "presented_paths": presented_paths,
        "matched_paths": matched_paths,
        "stage": "presented" if satisfied else ("mismatched" if presented_paths else "not_started"),
        "satisfied": satisfied,
    }


def _delivery_error(content: dict[str, Any]) -> str | None:
    """변경된 output이 하나도 제시되지 않았다면 terminal error를 반환한다."""
    if not content.get("produced_paths") or content.get("satisfied") is True:
        return None
    return _DELIVERY_INCOMPLETE_ERROR


def _workspace_excluded_dir_names(app_config: AppConfig | None) -> frozenset[str]:
    """이 배포에서 workspace snapshot이 건너뛰어야 하는 디렉터리 이름들.

    tool-output budget middleware는 크기를 초과한 tool 출력을 outputs 아래 storage
    subdir(기본값 ``.tool-results``)로 빼낸다. 이 파일들은 budget preview에서
    ``read_file``로 참조되는 process feedback이지 산출물이 아니다. 이를 produced
    artifact로 세면, 실제 artifact를 제시하지 않은 채 tool 출력만 externalize한 run이
    delivery 검증에 실패하게 된다. 기본 이름은 scanner 자체가 제외하고, 커스텀
    ``tool_output.storage_subdir``(``ToolOutputConfig``가 단일 세그먼트 이름으로
    강제하므로 scanner의 디렉터리명 pruning이 항상 일치한다)은 여기서 snapshot
    capture로 넘겨 before/after diff가 일관되게 유지되도록 한다.
    """
    storage_subdir = app_config.tool_output.storage_subdir if app_config is not None else TOOL_RESULTS_DIRNAME
    return frozenset({storage_subdir})


async def _produced_output_paths(
    before: WorkspaceSnapshot | None,
    *,
    thread_id: str,
    user_id: str | None,
    extra_excluded_dir_names: frozenset[str] | None = None,
) -> list[str]:
    """이 run이 생성하거나 수정한 일반 output 파일을 찾아낸다."""
    if before is None:
        return []
    try:
        after = await capture_workspace_snapshot(thread_id, user_id=user_id, include_text=False, extra_excluded_dir_names=extra_excluded_dir_names)
        return get_changed_output_paths(before, after)
    except Exception:
        logger.warning("Could not detect produced output artifacts for run thread %s", thread_id, exc_info=True)
        return []


# 이 streaming 정책은 middleware의 write 권한 집합과 분리해서 유지한다.
_LARGE_FILE_TOOL_NAMES = frozenset({"str_replace", "write_file"})
_LARGE_FILE_TOOL_BATCH_SIZE = 32


@dataclass
class _LargeFileToolChunkBatcher:
    """브라우저의 2차 파싱 비용을 피하려고 파일 본문 인자 delta를 배치로 묶는다.

    일반 assistant 텍스트와 파일이 아닌 tool call은 그대로 토큰 단위로 stream된다.
    큰 파일 인자도 여전히 점진적으로 갱신되지만, 모델 토큰마다 커지는 JSON을 브라우저가
    다시 파싱하게 만드는 대신 크기가 제한된 배치로 내보낸다.
    """

    batch_size: int = _LARGE_FILE_TOOL_BATCH_SIZE
    tool_names: dict[tuple[str, str, str], str] = field(default_factory=dict)
    pending_identity: tuple[str, str, str] | None = None
    pending_message: Any | None = None
    pending_metadata: dict[str, Any] = field(default_factory=dict)
    pending_count: int = 0

    def push(self, chunk: Any) -> list[Any]:
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            return [*self.flush(), chunk]

        message, metadata = chunk
        message_id = getattr(message, "id", None)
        tool_call_chunks = getattr(message, "tool_call_chunks", None)
        if not isinstance(message_id, str) or not message_id or not isinstance(tool_call_chunks, list) or len(tool_call_chunks) != 1:
            return [*self.flush(), chunk]

        tool_chunk = tool_call_chunks[0]
        if not isinstance(tool_chunk, dict):
            return [*self.flush(), chunk]
        index = tool_chunk.get("index")
        tool_call_id = tool_chunk.get("id")
        if isinstance(index, int):
            discriminator = f"index:{index}"
        elif isinstance(tool_call_id, str) and tool_call_id:
            discriminator = f"id:{tool_call_id}"
        else:
            discriminator = "single"
        raw_namespace = None
        if isinstance(metadata, dict):
            raw_namespace = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns")
        namespace = raw_namespace if isinstance(raw_namespace, str) else ""
        identity = (namespace, message_id, discriminator)
        name_fragment = tool_chunk.get("name")
        tool_name = self.tool_names.get(identity, "")
        if tool_name not in _LARGE_FILE_TOOL_NAMES and isinstance(name_fragment, str) and name_fragment:
            tool_name += name_fragment
            if any(candidate.startswith(tool_name) for candidate in _LARGE_FILE_TOOL_NAMES):
                self.tool_names[identity] = tool_name
            else:
                self.tool_names.pop(identity, None)
        # 누적된 이름이 일치한 뒤에야 배칭이 시작된다. 그 전까지 쪼개졌거나 불완전한
        # 이름 조각은 chunk 단위로 stream된다.
        if tool_name not in _LARGE_FILE_TOOL_NAMES:
            return [*self.flush(), chunk]

        model_copy = getattr(message, "model_copy", None)
        if not callable(model_copy):
            return [*self.flush(), chunk]
        additional_kwargs = getattr(message, "additional_kwargs", None)
        sanitized_additional_kwargs = additional_kwargs
        if isinstance(additional_kwargs, dict) and ("function_call" in additional_kwargs or "tool_calls" in additional_kwargs):
            sanitized_additional_kwargs = {key: value for key, value in additional_kwargs.items() if key not in {"function_call", "tool_calls"}}
        has_non_tool_payload = bool(getattr(message, "content", None) or sanitized_additional_kwargs or getattr(message, "usage_metadata", None) or getattr(message, "response_metadata", None))
        outputs: list[Any] = []
        if self.pending_identity is not None and self.pending_identity != identity:
            outputs.extend(self.flush())
        if has_non_tool_payload:
            visible_message = model_copy(
                update={
                    "additional_kwargs": sanitized_additional_kwargs,
                    "invalid_tool_calls": [],
                    "tool_call_chunks": [],
                    "tool_calls": [],
                }
            )
            outputs.append((visible_message, metadata))

        tool_only_message = model_copy(
            update={
                "additional_kwargs": {},
                "content": "",
                "invalid_tool_calls": [],
                "response_metadata": {},
                "tool_calls": [],
                "usage_metadata": None,
            }
        )
        self.pending_identity = identity
        self.pending_message = tool_only_message if self.pending_message is None else self.pending_message + tool_only_message
        if isinstance(metadata, dict):
            self.pending_metadata.update(metadata)
        self.pending_count += 1
        if self.pending_count >= self.batch_size:
            outputs.extend(self.flush())
        return outputs

    def flush(self) -> list[Any]:
        if self.pending_message is None:
            return []
        chunk = (self.pending_message, self.pending_metadata)
        self.pending_identity = None
        self.pending_message = None
        self.pending_metadata = {}
        self.pending_count = 0
        return [chunk]

    def finish(self) -> list[Any]:
        """values 경계 또는 stream 종료 경계에서 flush하고 identity를 해제한다.

        일반적인 batch-size flush나 모드 교차 flush는 identity를 유지해야 한다. 이어지는
        chunk는 대개 tool 이름을 생략하기 때문이다.
        """
        chunks = self.flush()
        self.tool_names.clear()
        return chunks


def _build_runtime_context(
    thread_id: str,
    run_id: str,
    caller_context: Any | None,
    app_config: AppConfig | None = None,
    task_store: Any | None = None,
    extensions: Any | None = None,
) -> dict[str, Any]:
    """이 run의 ``ToolRuntime.context``가 될 dict를 만든다.

    항상 ``thread_id``와 ``run_id``를 포함한다. 호출자의 ``config['context']``에 있던 추가
    키(예: bootstrap 흐름의 ``agent_name`` — issue #2677)는 병합되지만 절대
    ``thread_id``/``run_id``를 덮어쓰지 않는다. 해석된 ``AppConfig``는 worker가 넣어주므로
    tool이 전역 조회 없이 사용할 수 있다.

    langgraph 1.1+는 ``config['configurable']['__pregel_runtime']``에 저장된 부모 runtime을
    통해 이것을 ``runtime.context``로 노출한다. ``langgraph.pregel.main``에서
    ``parent_runtime.merge(...)``가 호출되는 부분을 참고한다.
    """
    runtime_ctx: dict[str, Any] = {"thread_id": thread_id, "run_id": run_id}
    if isinstance(caller_context, dict):
        for key, value in caller_context.items():
            if key == CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY:
                continue
            runtime_ctx.setdefault(key, value)
    if app_config is not None:
        runtime_ctx["app_config"] = app_config
    if task_store is not None:
        from deerflow_extension_api import EXTENSION_TASK_STORE_KEY

        runtime_ctx[EXTENSION_TASK_STORE_KEY] = task_store
    # run의 extension snapshot을 게시해서, graph 실행 중 dispatch되는 작업(task 위임)이
    # run 도중 교체되었을 수도 있는 singleton을 다시 읽는 대신 lead agent가 빌드될 때
    # 쓰인 것과 같은 generation에 바인딩되게 한다. caller 병합 이후에 쓰고 값이 없으면
    # pop한다. 이 host 내부 키에 대해 caller가 준 값은 절대 권위를 갖지 않기 때문이다.
    from deerflow.extensions import EXTENSION_SNAPSHOT_CONTEXT_KEY

    if extensions is not None:
        runtime_ctx[EXTENSION_SNAPSHOT_CONTEXT_KEY] = extensions
    else:
        runtime_ctx.pop(EXTENSION_SNAPSHOT_CONTEXT_KEY, None)
    return runtime_ctx


@dataclass(frozen=True)
class RunContext:
    """agent run 하나에 필요한 인프라 의존성.

    checkpointer, store, persistence 관련 singleton을 묶어서 ``run_agent``(그리고 앞으로의
    호출자)가 계속 늘어나는 키워드 인자 목록 대신 객체 하나를 받도록 한다.
    """

    checkpointer: Any
    store: Any | None = field(default=None)
    event_store: Any | None = field(default=None)
    run_events_config: Any | None = field(default=None)
    thread_store: Any | None = field(default=None)
    app_config: AppConfig | None = field(default=None)
    extensions: Any | None = field(default=None)
    checkpoint_channel_mode: CheckpointChannelMode = "full"
    # 시작 시 고정된 delta snapshot 주기. ``None``은 "이 프로세스에서 고정되지
    # 않음"(embedded/테스트)을 뜻하며 config 기본값으로 해석된다.
    checkpoint_snapshot_frequency: int | None = None
    on_run_completed: Any | None = field(default=None)


def _install_runtime_context(config: dict, runtime_context: dict[str, Any]) -> None:
    existing_context = config.get("context")
    if isinstance(existing_context, dict):
        existing_context.setdefault("thread_id", runtime_context["thread_id"])
        existing_context.setdefault("run_id", runtime_context["run_id"])
        if DEERFLOW_TRACE_METADATA_KEY in runtime_context:
            existing_context.setdefault(DEERFLOW_TRACE_METADATA_KEY, runtime_context[DEERFLOW_TRACE_METADATA_KEY])
        if "app_config" in runtime_context:
            existing_context["app_config"] = runtime_context["app_config"]
        if CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY in runtime_context:
            existing_context[CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY] = runtime_context[CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY]
        return

    config["context"] = dict(runtime_context)


def _compute_agent_factory_supports_app_config(agent_factory: Any) -> bool:
    try:
        return "app_config" in inspect.signature(agent_factory).parameters
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=128)
def _cached_agent_factory_supports_app_config(agent_factory: Any) -> bool:
    return _compute_agent_factory_supports_app_config(agent_factory)


def _agent_factory_supports_app_config(agent_factory: Any) -> bool:
    try:
        return _cached_agent_factory_supports_app_config(agent_factory)
    except TypeError:
        # 일부 callable 인스턴스는 hashable하지 않으므로 직접 검사로 fallback한다.
        return _compute_agent_factory_supports_app_config(agent_factory)


class _SubagentEventBuffer:
    """subagent ``task_*`` step 이벤트를 버퍼링해 lock 한 번으로 배치 flush한다 (#3779).

    실시간 표시용으로는 live SSE bridge가 이미 이 이벤트를 전달한다. 여기서는 추가로
    저장해서 subtask 카드의 step 이력이 새로고침 후에도 남게 한다.

    ``RunEventStore.put``은 저빈도 경로로 문서화되어 있다. Postgres에서는 호출마다 자체
    트랜잭션을 열고 thread별 advisory lock을 잡는다. 깊은 subagent(``general-purpose``는
    ``max_turns=150``까지 돈다)는 hot stream loop에서 수백 개의 ``task_running`` step을
    내보내므로, 각각을 ``put()``으로 저장하면 run 자신의 message-batch writer와 직렬화된다.
    그래서 인식된 subagent 이벤트를 모아 ``put_batch``로 쓰고, batch당 lock을 한 번만
    잡아 store의 계약을 지킨다.

    best-effort로 동작한다. store가 없거나(run_events 미설정) 인식되지 않는 chunk는 no-op
    이고, flush 실패는 로그만 남기고 stream loop로 전파하지 않는다. terminal
    ``subagent.end`` 이벤트는 즉시 flush해서 완료된 subagent의 step 이력이 run 종료 시점이
    아니라 곧바로 durable해지도록 한다.
    """

    #: 이만큼 이벤트가 쌓이면 flush한다. step마다 lock을 잡지 않으면서도 깊은 subagent
    #: 하나가 쓰는 메모리와 새로고침 지연을 제한한다.
    FLUSH_THRESHOLD = 25

    def __init__(self, event_store: Any | None, thread_id: str, run_id: str) -> None:
        self._event_store = event_store
        self._thread_id = thread_id
        self._run_id = run_id
        self._pending: list[dict[str, Any]] = []

    async def add(self, chunk: Any) -> None:
        """custom stream chunk 하나를 버퍼링한다. terminal 이벤트나 임계치에서 flush한다."""
        if self._event_store is None:
            return
        # lazy import: 모듈 로드 시점에 deerflow.subagents를 import하면 패키지
        # __init__(executor → agents → tools → task_tool)이 실행되는데, 이것이 다시
        # deerflow.subagents에서 import해 gateway 시작 시 deadlock이 난다. 호출 시점(모든
        # 모듈이 로드된 뒤)으로 미루면 그 순환이 끊긴다.
        from deerflow.subagents.step_events import subagent_run_event

        record = subagent_run_event(chunk)
        if record is None:
            return
        self._pending.append({"thread_id": self._thread_id, "run_id": self._run_id, **record})
        if record["event_type"] == "subagent.end" or len(self._pending) >= self.FLUSH_THRESHOLD:
            await self.flush()

    async def flush(self) -> None:
        """버퍼링된 이벤트를 ``put_batch`` 한 번으로 저장한다. store 오류는 삼킨다."""
        if self._event_store is None or not self._pending:
            return
        batch = self._pending
        self._pending = []
        try:
            await self._event_store.put_batch(batch)
        except Exception:
            # 실패한 batch를 (그 사이 쌓인 이벤트보다 앞에) 다시 버퍼에 넣어, 일시적인
            # store 오류가 subagent step 이벤트를 조용히 버리지 않게 한다.
            self._pending = batch + self._pending
            logger.warning("Run %s: failed to persist %d subagent step event(s)", self._run_id, len(batch), exc_info=True)


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: Any,
    graph_input: dict,
    config: dict,
    stream_modes: list[str] | None = None,
    stream_subgraphs: bool = False,
    interrupt_before: list[str] | Literal["*"] | None = None,
    interrupt_after: list[str] | Literal["*"] | None = None,
) -> None:
    """agent를 백그라운드에서 실행하며 이벤트를 *bridge*에 publish한다."""

    # RunContext에서 인프라 의존성을 꺼낸다.
    checkpointer = ctx.checkpointer
    store = ctx.store
    event_store = ctx.event_store
    run_events_config = ctx.run_events_config
    thread_store = ctx.thread_store
    terminal_status_kwargs = {"persist": False} if event_store is not None else {}

    run_id = record.run_id
    thread_id = record.thread_id

    from deerflow_extension_api import ExtensionData

    from deerflow.extensions import get_loaded_extensions

    extensions = ctx.extensions if ctx.extensions is not None else get_loaded_extensions()
    task_store: ExtensionData | None = None
    pre_run_checkpoint_id: str | None = None
    pre_run_workspace_snapshot: WorkspaceSnapshot | None = None
    workspace_changes_user_id: str | None = None
    workspace_excluded_dir_names: frozenset[str] | None = None
    snapshot_capture_failed = False
    llm_error_fallback_message: str | None = None
    checkpoint_rollback_completed = False
    # 이 run이 시작되기 *전에* checkpoint된 message id들. stream loop는 같은 thread의
    # 이전 run에 속한 ``deerflow_error_fallback`` 마커를 이 집합으로 걸러낸다. 이것이
    # 없으면 이력에 남은 오래된 fallback 하나가 이 thread의 모든 후속 run을 ``error``로
    # 표시하게 된다.
    pre_existing_message_ids: set[str] = set()

    # 바인딩된 agent graph accessor와 캡처된 run 이전 rollback 지점. finally의 rollback
    # 경로가 run 이전 checkpoint 계보를 fork할 수 있도록 try 블록 안에서 할당한다(아래 참고).
    accessor: CheckpointStateAccessor | None = None
    rollback_point: RollbackPoint | None = None
    journal = None
    delivery_content: dict[str, Any] | None = None
    produced_output_paths: list[str] | None = None
    # journal 생성을 preflight보다 앞으로 옮겨서 모든 terminal run이 receipt를 낼 수 있게
    # 했다. completion 저장은 이전 경계를 유지한다. #4272 이전에는 preflight가 성공하기
    # 전까지 journal이 없었으므로, 이른 checkpoint 실패나 대기 중 취소는 빈 completion
    # snapshot을 RunStore에 쓰지 않았다.
    persist_completion = False
    # subagent step 이벤트를 모아 저장하기 위한 버퍼 (#3779). streaming이 시작될 때
    # 할당되고 finally 블록에서 flush된다. streaming 전에 예외가 나도 finally가 안전하도록
    # None으로 미리 바인딩한다.
    subagent_events: _SubagentEventBuffer | None = None
    started = False

    async def _finish_cancellation(
        action: str,
        *,
        restore_checkpoint: bool = True,
    ) -> None:
        nonlocal checkpoint_rollback_completed
        await run_manager.set_finalizing(run_id, True)
        if action == "rollback":
            await run_manager.set_status(
                run_id,
                RunStatus.error,
                error="Rolled back by user",
                **terminal_status_kwargs,
            )
            if not restore_checkpoint:
                return
            try:
                checkpoint_rollback_completed = await _rollback_to_pre_run_checkpoint(
                    accessor=accessor,
                    checkpointer=checkpointer,
                    thread_id=thread_id,
                    run_id=run_id,
                    rollback_point=rollback_point,
                    snapshot_capture_failed=snapshot_capture_failed,
                )
                logger.info(
                    "Run %s rolled back to pre-run checkpoint %s",
                    run_id,
                    pre_run_checkpoint_id,
                )
            except Exception:
                logger.warning(
                    "Run %s cancellation rollback failed",
                    run_id,
                    exc_info=True,
                )
        else:
            await run_manager.set_status(
                run_id,
                RunStatus.interrupted,
                **terminal_status_kwargs,
            )
            logger.info("Run %s was cancelled", run_id)

    try:
        normalized_stream_modes = normalize_stream_modes(stream_modes)
        requested_modes: set[str] = set(normalized_stream_modes)
        lg_modes = to_langgraph_stream_modes(normalized_stream_modes)
        # 실패하거나 취소될 수 있는 preflight 작업 전에 run 범위 journal을 초기화한다.
        # event store가 있는 모든 terminal run은 run.delivery receipt에 쓸 journal을 가진
        # 채로 공통 finally 블록에 도달해야 한다. checkpoint 검증 실패나 앞선 run의
        # finalize 대기 중 취소도 마찬가지다.
        if event_store is not None:
            from deerflow.runtime.journal import RunJournal

            journal = RunJournal(
                run_id=run_id,
                thread_id=thread_id,
                event_store=event_store,
                track_token_usage=getattr(run_events_config, "track_token_usage", True),
                progress_reporter=lambda snapshot: run_manager.update_run_progress(run_id, **snapshot),
            )

        await run_manager.wait_for_prior_finalizing(
            thread_id,
            run_id,
            abort_event=record.abort_event,
        )

        start_outcome = await run_manager.try_start(run_id)
        if start_outcome is not RunStartOutcome.started:
            if record.abort_event.is_set():
                await _finish_cancellation(
                    record.abort_action,
                    restore_checkpoint=False,
                )
            return
        started = True

        if extensions.needs_task_store:
            task_store = ExtensionData(run_id)

        if not record.ownership_lost and thread_store is not None:
            try:
                await thread_store.update_status(thread_id, "running")
            except Exception:
                logger.debug("Failed to update thread_meta status for %s (non-fatal)", thread_id)
        mode = ctx.checkpoint_channel_mode
        inject_checkpoint_mode(config, mode)
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        if checkpointer is not None:
            await aensure_checkpoint_mode_compatible(
                checkpointer,
                checkpoint_config,
                mode,
            )
            configurable = config["configurable"]
            selected_configurable = {
                "thread_id": thread_id,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            }
            for selector_key in ("checkpoint_id", "checkpoint_map"):
                if selector_key in configurable:
                    selected_configurable[selector_key] = configurable[selector_key]
            selected_checkpoint_config = {
                "configurable": selected_configurable,
            }
            if selected_checkpoint_config != checkpoint_config:
                await aensure_checkpoint_mode_compatible(
                    checkpointer,
                    selected_checkpoint_config,
                    mode,
                )

        persist_completion = True

        if event_store is not None:
            workspace_changes_user_id = get_effective_user_id()
            # run마다 한 번만 해석해서 run 이전 snapshot, run 이후 delivery 스캔,
            # workspace-changes 스캔이 모두 같은 제외 집합을 쓰게 한다.
            workspace_excluded_dir_names = _workspace_excluded_dir_names(ctx.app_config)
            try:
                pre_run_workspace_snapshot = await capture_workspace_snapshot(
                    thread_id,
                    user_id=workspace_changes_user_id,
                    extra_excluded_dir_names=workspace_excluded_dir_names,
                )
            except Exception:
                logger.warning("Could not capture pre-run workspace snapshot for run %s", run_id, exc_info=True)

        # 2. metadata publish — useStream은 run_id와 thread_id 둘 다 필요하다
        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )

        # 3. agent 빌드
        from langchain_core.runnables import RunnableConfig
        from langgraph.runtime import Runtime

        # middleware와 tool이 (ToolRuntime.context를 통해) thread 수준 데이터에 접근할 수
        # 있도록 runtime context를 주입한다. langgraph-cli는 이것을 자동으로 하지만,
        # 여기서는 공식 ``context=`` 파라미터 없이 ``agent.astream(config=...)``으로 graph를
        # 구동하므로 직접 해야 한다.
        runtime_ctx = _build_runtime_context(
            thread_id,
            run_id,
            config.get("context"),
            ctx.app_config,
            task_store,
            extensions,
        )
        incoming_metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
        deerflow_trace_id = resolve_deerflow_trace_id(incoming_metadata.get(DEERFLOW_TRACE_METADATA_KEY))
        if deerflow_trace_id:
            runtime_ctx[DEERFLOW_TRACE_METADATA_KEY] = deerflow_trace_id
            if is_trace_id_from_request_header():
                merged_metadata = dict(incoming_metadata)
                merged_metadata[DEERFLOW_TRACE_METADATA_KEY] = deerflow_trace_id
                config["metadata"] = merged_metadata
        # middleware가 audit 이벤트를 쓸 수 있도록(예: SafetyFinishReasonMiddleware가
        # 억제된 tool call을 기록) run 범위 journal을 sentinel 키로 노출한다. 이중 밑줄
        # 접두사는 runtime 내부 채널임을 뜻한다. 사용자 코드는 이 키 이름에 의존하면 안 된다.
        if journal is not None:
            runtime_ctx["__run_journal"] = journal
        _install_runtime_context(config, runtime_ctx)
        runtime = Runtime(context=cast(Any, runtime_ctx), store=store)
        config.setdefault("configurable", {})["__pregel_runtime"] = runtime

        # RunJournal을 LangChain callback handler로 주입한다.
        # on_llm_end는 token usage를, on_chain_start/end는 lifecycle을 수집한다.
        if journal is not None:
            config.setdefault("callbacks", []).append(journal)

        # langchain CallbackHandler가 session_id / user_id / trace_name / tags를 root
        # trace로 끌어올릴 수 있도록 Langfuse trace-attribute metadata를 주입한다.
        # ``DeerFlowClient.stream``과 헬퍼를 공유해 두 진입점이 어긋나지 않게 하고,
        # 헬퍼 내부의 setdefault 덕분에 호출자가 준 metadata가 우선한다.
        inject_langfuse_metadata(
            config,
            thread_id=thread_id,
            user_id=resolve_runtime_user_id(runtime),
            assistant_id=record.assistant_id,
            model_name=record.model_name,
            environment=os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT"),
            deerflow_trace_id=deerflow_trace_id,
        )

        # runtime context 설치 이후에 해석해서 context/configurable이 이 run이 실제로
        # 실행할 agent 이름을 반영하게 한다.
        config.setdefault("run_name", resolve_root_run_name(config, record.assistant_id))
        initial_runnable_config = RunnableConfig(**config)

        def _continuation_runnable_config() -> RunnableConfig:
            continuation_config = dict(config)
            configurable = dict(continuation_config.get("configurable", {}) or {})
            configurable["checkpoint_ns"] = ""
            configurable.pop("checkpoint_id", None)
            configurable.pop("checkpoint_map", None)
            continuation_config["configurable"] = configurable
            return RunnableConfig(**continuation_config)

        agent_factory_kwargs: dict[str, Any] = {"config": initial_runnable_config}
        if ctx.app_config is not None and _agent_factory_supports_app_config(agent_factory):
            agent_factory_kwargs["app_config"] = ctx.app_config
        from deerflow.extensions import bind_agent_build_extensions

        with bind_agent_build_extensions(extensions):
            agent = agent_factory(**agent_factory_kwargs)

        accessor = CheckpointStateAccessor.bind(
            agent,
            checkpointer,
            store=store,
            mode=mode,
        )

        # 이 run이 thread를 변경하기 전에 run 이전 rollback 지점(materialize된 state와 raw
        # pending writes)을 캡처한다. raw checkpoint blob으로는 Delta 채널 메시지를 복원할
        # 수 없으므로(그 checkpoint에는 channel_values가 없다) rollback은 graph를 통해 run
        # 이전 계보를 fork하며, materialize된 메시지가 미리 필요하다. 캡처가 실패하면
        # rollback을 비활성화한다. 비어 있거나 일부만 있는 메시지 이력을 복원하면 thread가
        # 조용히 잘려나가기 때문이다.
        if checkpointer is not None:
            # 앞서 성공한 run이 active admission slot을 반납한 뒤에도 duration metadata를
            # 계속 저장 중일 수 있다. 그 checkpoint lock을 공유해서 rollback snapshot과
            # 재개 재작성이 head에 대해 중단 없는 하나의 읽기/쓰기 시퀀스가 되게 한다.
            async with _checkpoint_thread_lock(thread_id):
                try:
                    rollback_point = await _capture_rollback_point(accessor, checkpointer, checkpoint_config)
                except Exception:
                    snapshot_capture_failed = True
                    logger.warning("Could not capture pre-run checkpoint snapshot for run %s", run_id, exc_info=True)
                if rollback_point is not None:
                    pre_run_checkpoint_id = rollback_point.config.get("configurable", {}).get("checkpoint_id")
                    pre_existing_message_ids = _collect_pre_existing_message_ids({"messages": list(rollback_point.messages)})

                # 오래된 checkpoint에서 재개하는 것은 fork이고, delta fork는 버려진 형제의
                # write를 state로 다시 materialize한다 (#4458). rollback 지점을 캡처한
                # *뒤에* 선형 head write로 재작성해서, rollback을 동반한 취소가 되돌려진
                # head가 아니라 진짜 run 이전 head를 복원하게 한다.
                resumed_messages = await _linearize_delta_checkpoint_resume(
                    accessor=accessor,
                    checkpointer=checkpointer,
                    config=config,
                    thread_id=thread_id,
                    run_id=run_id,
                )
            if resumed_messages is not None:
                # 이제 graph는 선택된 state에서 시작하므로, 현재 run의 메시지 경계는
                # rollback용으로 캡처한 head가 아니라 그 state다.
                pre_existing_message_ids = _collect_pre_existing_message_ids({"messages": list(resumed_messages)})
                initial_runnable_config = RunnableConfig(**config)

        runtime_ctx[CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY] = frozenset(pre_existing_message_ids)
        _install_runtime_context(config, runtime_ctx)

        # agent metadata에서 실제로 적용된(해석된) model 이름을 가져온다. agent.py의
        # _resolve_model_name은 요청한 이름이 allowlist에 없으면 기본 model을 반환할 수
        # 있으므로, 이 갱신으로 저장되는 model_name이 실제 사용된 model을 반영하게 한다.
        if record.model_name is not None:
            resolved = getattr(agent, "metadata", {}) or {}
            if isinstance(resolved, dict):
                effective = resolved.get("model_name")
                if effective and effective != record.model_name:
                    await run_manager.update_model_name(record.run_id, effective)

        # 4. checkpointer와 store 연결
        if checkpointer is not None:
            agent.checkpointer = checkpointer
        if store is not None:
            agent.store = store

        # 5. interrupt node 설정
        if interrupt_before:
            agent.interrupt_before_nodes = interrupt_before
        if interrupt_after:
            agent.interrupt_after_nodes = interrupt_after

        logger.info("Run %s: streaming with modes %s (requested: %s)", run_id, lg_modes, requested_modes)

        # hot stream loop에서 step마다 저빈도 put()을 호출하는 대신, subagent step
        # 이벤트를 버퍼링해 배치로 저장한다 (#3779). finally 블록에서 flush하므로 abort나
        # 예외 경로에서도 버퍼링된 step이 살아남는다.
        subagent_events = _SubagentEventBuffer(event_store, thread_id, run_id)

        goal_evaluator_model: Any | None = None

        def _get_goal_evaluator_model() -> Any:
            nonlocal goal_evaluator_model
            if goal_evaluator_model is None:
                goal_evaluator_model = create_goal_evaluator_model(
                    model_name=record.model_name,
                    app_config=ctx.app_config,
                )
            return goal_evaluator_model

        async def _stream_once(input_payload: Any, stream_config: RunnableConfig) -> None:
            nonlocal llm_error_fallback_message
            file_tool_chunk_batcher = _LargeFileToolChunkBatcher() if "values" in requested_modes else None
            try:
                async with _checkpoint_thread_lock(thread_id):
                    if len(lg_modes) == 1 and not stream_subgraphs:
                        # 단일 모드, subgraph 없음: astream은 raw chunk를 그대로 준다
                        single_mode = lg_modes[0]
                        async for chunk in agent.astream(input_payload, config=stream_config, stream_mode=single_mode):
                            if record.abort_event.is_set():
                                logger.info("Run %s abort requested — stopping", run_id)
                                break
                            llm_error_fallback_message = llm_error_fallback_message or _extract_llm_error_fallback_message(chunk, pre_existing_message_ids)
                            sse_event = _lg_mode_to_sse_event(single_mode)
                            await bridge.publish(run_id, sse_event, serialize(chunk, mode=single_mode))
                            if single_mode == "custom":
                                await subagent_events.add(chunk)
                        return
                    # 여러 모드나 subgraph: astream은 튜플을 준다
                    async for item in agent.astream(
                        input_payload,
                        config=stream_config,
                        stream_mode=lg_modes,
                        subgraphs=stream_subgraphs,
                    ):
                        if record.abort_event.is_set():
                            logger.info("Run %s abort requested — stopping", run_id)
                            break

                        mode, chunk, namespace = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                        if mode is None:
                            continue

                        if not namespace:
                            # 부모 run의 error fallback을 결정할 수 있는 것은 root graph
                            # frame뿐이다. 위임된 subagent의 fallback 마커는 이 run이
                            # 아니라 executor가 (task_failed로) 매핑할 몫이다.
                            llm_error_fallback_message = llm_error_fallback_message or _extract_llm_error_fallback_message(chunk, pre_existing_message_ids)
                        await _publish_stream_item(
                            bridge=bridge,
                            run_id=run_id,
                            mode=mode,
                            chunk=chunk,
                            namespace=namespace,
                            file_tool_chunk_batcher=file_tool_chunk_batcher,
                            subagent_events=subagent_events,
                        )
            finally:
                stream_error = sys.exception()
                if file_tool_chunk_batcher is not None:
                    try:
                        for publish_chunk in file_tool_chunk_batcher.finish():
                            await bridge.publish(run_id, "messages", serialize(publish_chunk, mode="messages"))
                    except Exception:
                        if stream_error is None:
                            raise
                        logger.debug("Could not flush pending file-tool chunks for run %s", run_id, exc_info=True)

        # 7. 요청된 turn을 stream하고, 필요하면 숨겨진 goal turn을 이어간다.
        # 오래된 stop_reason은 첫 (사용자에게 보이는) turn 이전에만 지운다. 이어지는 turn은
        # 사용자 turn에서 나온 cap 사유를 보존한다. 사용자 turn 중에 cap에 걸린 run은 이후
        # 숨겨진 goal-evaluator turn이 깨끗하게 끝나더라도 cap된 것이다 (#4176 리뷰).
        if isinstance(runtime.context, dict):
            runtime.context.pop("stop_reason", None)
        await _stream_once(graph_input, initial_runnable_config)
        while not record.abort_event.is_set() and not llm_error_fallback_message and (journal is None or not journal.had_llm_error_fallback):
            continuation_input = await _prepare_goal_continuation_input(
                bridge=bridge,
                accessor=accessor,
                checkpointer=checkpointer,
                thread_id=thread_id,
                run_id=run_id,
                model_name=record.model_name,
                app_config=ctx.app_config,
                evaluator_model_factory=_get_goal_evaluator_model,
                abort_event=record.abort_event,
                user_id=resolve_runtime_user_id(runtime),
                deerflow_trace_id=deerflow_trace_id,
            )
            if continuation_input is None or record.abort_event.is_set():
                break
            await _stream_once(continuation_input, _continuation_runnable_config())

        # 8. 최종 status
        if record.abort_event.is_set():
            await _finish_cancellation(record.abort_action)
        elif llm_error_fallback_message or (journal is not None and journal.had_llm_error_fallback):
            error_msg = llm_error_fallback_message
            if error_msg is None and journal is not None:
                error_msg = journal.llm_error_fallback_message
            error_msg = error_msg or "LLM provider failed after retries"
            await _ensure_finalizing_before_edit_failure(run_manager, record)
            cancel_action = await run_manager.set_status_if_not_cancelled(
                run_id,
                RunStatus.error,
                error=error_msg,
                **terminal_status_kwargs,
            )
            if cancel_action is not None:
                await _finish_cancellation(cancel_action)
        else:
            runtime_context = runtime.context if isinstance(runtime.context, dict) else None
            # tool_calls를 제거해 run을 강제 중단하는 guard middleware들은 worker가 run
            # record에 노출할 수 있도록 runtime.context에 stop_reason을 새긴다:
            #   loop_detection      -> "loop_capped"
            #   token_budget        -> "token_capped"
            #   safety_finish_reason -> "safety_capped"
            #   subagent_limit       -> "subagent_limit_capped"
            #   model_length_finish_reason -> "model_length_capped"
            #
            # stop_reason 의미를 갖는 guard가 더 늘어난다면, 각 guard가 같은 키에 직접
            # 쓰는 대신 publish/collect 패턴(예: 각 guard middleware가 전용
            # runtime.context 채널에 cap 사유를 publish하고 worker가 가장 심각한 것 /
            # 첫 번째 / 전체를 모으는 방식)을 고려한다.
            stop_reason = runtime_context.get("stop_reason") if runtime_context is not None else None
            produced_output_paths = await _produced_output_paths(
                pre_run_workspace_snapshot,
                thread_id=thread_id,
                user_id=workspace_changes_user_id,
                extra_excluded_dir_names=workspace_excluded_dir_names,
            )
            delivery_content = _delivery_content_with_outputs(
                journal.get_delivery_content() if journal is not None else _empty_delivery_content(),
                produced_output_paths,
            )
            delivery_error = _delivery_error(delivery_content)
            cancel_action = await run_manager.set_status_if_not_cancelled(
                run_id,
                RunStatus.error if delivery_error else RunStatus.success,
                error=delivery_error,
                stop_reason=stop_reason,
                **terminal_status_kwargs,
            )
            if cancel_action is not None:
                await _finish_cancellation(cancel_action)

    except asyncio.CancelledError:
        await _finish_cancellation(record.abort_action)

    except Exception as exc:
        error_msg = f"{exc}"
        logger.exception("Run %s failed: %s", run_id, error_msg)
        await _ensure_finalizing_before_edit_failure(run_manager, record)
        cancel_action = await run_manager.set_status_if_not_cancelled(
            run_id,
            RunStatus.error,
            error=error_msg,
            **terminal_status_kwargs,
        )
        if cancel_action is not None:
            await _finish_cancellation(cancel_action)
        else:
            await bridge.publish(
                run_id,
                "error",
                {
                    "message": error_msg,
                    "name": type(exc).__name__,
                },
            )

    finally:
        if record.ownership_lost:
            logger.warning(
                "Skipping durable finalization for run %s because this worker no longer owns its lease",
                run_id,
            )

        if not record.ownership_lost and _is_edit_replay_run(record) and record.status != RunStatus.success:
            if not record.finalizing:
                await run_manager.set_finalizing(run_id, True)
            try:
                if not checkpoint_rollback_completed:
                    checkpoint_rollback_completed = await _rollback_to_pre_run_checkpoint(
                        accessor=accessor,
                        checkpointer=checkpointer,
                        thread_id=thread_id,
                        run_id=run_id,
                        rollback_point=rollback_point,
                        snapshot_capture_failed=snapshot_capture_failed,
                    )
                if checkpoint_rollback_completed:
                    await _publish_restored_checkpoint_values(
                        bridge=bridge,
                        run_id=run_id,
                        accessor=accessor,
                        thread_id=thread_id,
                    )
                    logger.info("Run %s edit replay restored pre-run checkpoint %s", run_id, pre_run_checkpoint_id)
            except Exception:
                logger.warning("Run %s edit replay rollback failed", run_id, exc_info=True)

        # 아직 버퍼에 남은 subagent step 이벤트를 저장한다 (#3779). stream loop가 자체
        # flush 전에 빠져나간 abort/예외 경로도 포함한다.
        if not record.ownership_lost and subagent_events is not None:
            await subagent_events.flush()

        if not record.ownership_lost and event_store is not None and pre_run_workspace_snapshot is not None:
            try:
                await record_workspace_changes(
                    event_store,
                    thread_id,
                    run_id,
                    pre_run_workspace_snapshot,
                    user_id=workspace_changes_user_id,
                    extra_excluded_dir_names=workspace_excluded_dir_names,
                )
            except Exception:
                logger.warning("Failed to record workspace changes for run %s", run_id, exc_info=True)

        # terminal receipt보다 먼저 버퍼링된 journal 이벤트를 flush한다. receipt는
        # recovery와 공유하는 run 범위 멱등 write를 쓰고, 그 다음에 준비된 terminal
        # status를 저장한다. 이 순서가 terminal run이 receipt보다 오래 살아남을 수 있는
        # 크래시 구간을 막는다. fence된 worker는 receipt 복구를 그 run을 가져간 peer에
        # 맡긴다.
        if not record.ownership_lost and journal is not None:
            try:
                await journal.flush()
            except Exception:
                logger.warning("Failed to flush journal for run %s", run_id, exc_info=True)

            if delivery_content is None:
                if produced_output_paths is None:
                    produced_output_paths = await _produced_output_paths(
                        pre_run_workspace_snapshot,
                        thread_id=thread_id,
                        user_id=workspace_changes_user_id,
                        extra_excluded_dir_names=workspace_excluded_dir_names,
                    )
                delivery_content = _delivery_content_with_outputs(journal.get_delivery_content(), produced_output_paths)
            receipt_persisted = await _persist_delivery_receipt(
                event_store,
                thread_id=thread_id,
                run_id=run_id,
                content=delivery_content,
            )
            if produced_output_paths and record.status == RunStatus.success and not receipt_persisted:
                await run_manager.set_status(
                    run_id,
                    RunStatus.error,
                    error=_DELIVERY_RECEIPT_FAILED_ERROR,
                    persist=False,
                )

        if not record.ownership_lost and event_store is not None:
            try:
                # 제한된 receipt retry를 모두 소진한 뒤에도 실제 worker 결과는 저장한다.
                # 성공한 행을 inflight로 남겨두면 lease recovery가 그것을 합성된 zero
                # receipt와 함께 error로 다시 쓸 수 있다.
                if record.abort_event.is_set():
                    await run_manager.persist_current_status(run_id)
                else:
                    cancel_action = await run_manager.set_status_if_not_cancelled(
                        run_id,
                        record.status,
                        error=record.error,
                        stop_reason=record.stop_reason,
                    )
                    if cancel_action is not None:
                        await _finish_cancellation(cancel_action)
                        await run_manager.persist_current_status(run_id)
            except Exception:
                logger.warning("Failed to persist terminal status for run %s after delivery receipt attempts", run_id, exc_info=True)

        if not record.ownership_lost and journal is not None and persist_completion:
            try:
                # token usage와 편의 필드를 RunStore에 저장한다
                completion = journal.get_completion_data()
                await run_manager.update_run_completion(run_id, status=record.status.value, **completion)
            except Exception:
                logger.warning("Failed to persist run completion for %s (non-fatal)", run_id, exc_info=True)

        if started and not record.ownership_lost and checkpointer is not None and record.status == RunStatus.interrupted and not _is_edit_replay_run(record):
            try:
                await run_manager.wait_for_prior_finalizing(thread_id, run_id)
                if not await run_manager.has_later_started_run(thread_id, run_id):
                    await _ensure_interrupted_title(checkpointer=checkpointer, thread_id=thread_id, app_config=ctx.app_config, graph_input=graph_input)
            except Exception:
                logger.debug("Failed to generate interrupted title for thread %s (non-fatal)", thread_id)

        # checkpoint의 title을 threads_meta.display_name으로 동기화한다
        if started and not record.ownership_lost and checkpointer is not None and thread_store is not None:
            try:
                ckpt_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                ckpt_tuple = await checkpointer.aget_tuple(ckpt_config)
                if ckpt_tuple is not None:
                    ckpt = getattr(ckpt_tuple, "checkpoint", {}) or {}
                    title = ckpt.get("channel_values", {}).get("title")
                    if title:
                        await thread_store.update_display_name(thread_id, title)
            except Exception:
                logger.debug("Failed to sync title for thread %s (non-fatal)", thread_id)

        # run duration을 checkpoint metadata에 저장해서, 이력 조회가 run과 event를
        # 대조할 필요가 없게 한다.
        if started and not record.ownership_lost and checkpointer is not None and record.status == RunStatus.success:
            try:
                created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
                updated = datetime.fromisoformat(record.updated_at.replace("Z", "+00:00"))
                # 기존 이력 의미를 따른다. turn_duration은 admission 지연을 포함한
                # RunRecord 전체 수명을 정수 초로 나타낸다. 1초 미만으로 성공한 turn은
                # 0으로 저장한다.
                duration = max(0, int((updated - created).total_seconds()))
                await _persist_run_duration(
                    checkpointer=checkpointer,
                    thread_id=thread_id,
                    run_id=run_id,
                    duration_seconds=duration,
                )
            except Exception:
                logger.debug("Failed to persist run duration for thread %s run %s (non-fatal)", thread_id, run_id)

        # run 결과에 따라 threads_meta status를 갱신한다
        if started and not record.ownership_lost and thread_store is not None:
            try:
                final_status = "idle" if record.status == RunStatus.success else record.status.value
                await thread_store.update_status(thread_id, final_status)
            except Exception:
                logger.debug("Failed to update thread_meta status for %s (non-fatal)", thread_id)

        if not record.ownership_lost and ctx.on_run_completed is not None:
            try:
                await ctx.on_run_completed(record)
            except Exception:
                logger.warning("Run completion hook failed for %s (non-fatal)", run_id, exc_info=True)
        if record.finalizing:
            await run_manager.set_finalizing(run_id, False)

        await bridge.publish_end(run_id)
        asyncio.create_task(bridge.cleanup(run_id, delay=60))


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _checkpoint_id(checkpoint_tuple: Any) -> str | None:
    config = getattr(checkpoint_tuple, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    checkpoint_id = configurable.get("checkpoint_id") if isinstance(configurable, dict) else None
    if isinstance(checkpoint_id, str):
        return checkpoint_id
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("id"), str):
        return checkpoint["id"]
    return None


def _goal_instance_matches(left: GoalState | None, right: GoalState | None) -> bool:
    if not left or not right:
        return False
    same_status = left.get("status") == right.get("status") == "active"
    same_objective = left.get("objective") == right.get("objective")
    same_created_at = left.get("created_at") == right.get("created_at")
    return same_status and same_objective and same_created_at


async def _materialized_checkpoint_messages(accessor: CheckpointStateAccessor, thread_id: str) -> list[Any]:
    """모드에 맞는 accessor로 ``messages``를 읽는다.

    delta 모드에서 raw ``channel_values``를 읽으면 sentinel만 보인다. materialize된
    읽기만 리스트를 복원한다. raw checkpoint 튜플은 튜플 수준 metadata(checkpoint id,
    ``pending_writes``)에는 여전히 유효하다.
    """
    snapshot = await accessor.aget({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages") if isinstance(values, dict) else None
    return list(messages) if isinstance(messages, list) else []


def _read_checkpoint_goal(checkpoint_tuple: Any) -> GoalState | None:
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
    raw_goal = channel_values.get("goal") if isinstance(channel_values, dict) else None
    return copy.deepcopy(raw_goal) if isinstance(raw_goal, dict) else None


def _has_durable_goal_turn_receipt(checkpoint_tuple: Any, messages: list[Any]) -> bool:
    """완료된, 사용자에게 보이는 assistant turn이 안전하게 checkpoint되었으면 true를 반환한다.

    durability 신호는 ``pending_writes``다. ``CheckpointTuple``에는 ``tasks`` 필드가 없으므로
    (그것은 ``StateSnapshot``에 있다) 대기 중인 write가 있는지가 그 turn이 아직 진행 중임을
    알려주는 유일한 단서다.
    """
    if _checkpoint_id(checkpoint_tuple) is None:
        return False
    if getattr(checkpoint_tuple, "pending_writes", None):
        return False
    visible_messages = []
    for message in messages:
        if _is_visible_message(message) and message_to_text(message).strip():
            visible_messages.append(message)
    if not visible_messages:
        return False
    return _message_type(visible_messages[-1]) == "ai"


def _stand_down_reason(goal: GoalState, evaluation: GoalEvaluation, no_progress_count: int) -> str | None:
    if evaluation["satisfied"]:
        return None
    if evaluation["blocker"] != "goal_not_met_yet":
        return f"blocked:{evaluation['blocker']}"
    # 기본 상한은 should_continue_goal과 동일하게 맞춰서, 이 필드들이 빠진 goal dict에
    # 대해 두 gate 함수가 같은 판단을 하게 한다.
    if int(goal.get("continuation_count", 0)) >= int(goal.get("max_continuations", DEFAULT_MAX_GOAL_CONTINUATIONS)):
        return "max_continuations_reached"
    if no_progress_count >= int(goal.get("max_no_progress_continuations", DEFAULT_MAX_NO_PROGRESS_CONTINUATIONS)):
        return "no_progress_detected"
    return None


async def _persist_goal_evaluation(
    *,
    bridge: StreamBridge,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    goal: GoalState,
    evaluation: GoalEvaluation,
    no_progress_count: int,
    continuation_count: int | None = None,
    stand_down_reason: str | None = None,
    evidence_signature: str = "",
) -> GoalState | None:
    try:
        async with goal_thread_lock(thread_id):
            checkpoint_tuple = await _call_checkpointer_method(
                checkpointer,
                "aget_tuple",
                "get_tuple",
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            )
            if checkpoint_tuple is None:
                return None
            current_goal = _read_checkpoint_goal(checkpoint_tuple)
            if current_goal is None or not _goal_instance_matches(goal, current_goal):
                return None
            # 방어적으로, lock 안에서 최신 current_goal로 continuation_count를 다시
            # 계산한다. 호출자는 오래되었을 수도 있는 goal snapshot으로 계산했고, 경쟁하는
            # continuation이 이미 카운트를 올렸을 수 있다.
            if continuation_count is not None:
                current_count = int(current_goal.get("continuation_count", 0))
                continuation_count = max(continuation_count, current_count + 1)
            expected_checkpoint_id = _checkpoint_id(checkpoint_tuple)
            updated_goal = attach_goal_evaluation(
                current_goal,
                evaluation,
                run_id=run_id,
                continuation_count=continuation_count,
                no_progress_count=no_progress_count,
                stand_down_reason=stand_down_reason,
                evidence_signature=evidence_signature,
            )
            values = await write_thread_goal(
                checkpointer,
                thread_id,
                updated_goal,
                as_node="goal_evaluator",
                expected_checkpoint_id=expected_checkpoint_id,
            )
        await bridge.publish(run_id, "values", serialize(values, mode="values"))
        return updated_goal
    except GoalWriteConflict:
        return None
    except Exception:
        logger.warning("Could not persist goal evaluation for thread %s", thread_id, exc_info=True)
        return None


async def _reread_goal_and_checkpoint(checkpointer: Any, thread_id: str) -> tuple[GoalState | None, Any]:
    """동시성 재확인을 위해 goal과 최신 checkpoint를 함께 다시 읽는다."""
    goal = await read_thread_goal(checkpointer, thread_id)
    checkpoint_tuple = await _call_checkpointer_method(
        checkpointer,
        "aget_tuple",
        "get_tuple",
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
    )
    return goal, checkpoint_tuple


async def _prepare_goal_continuation_input(
    *,
    bridge: StreamBridge,
    accessor: CheckpointStateAccessor,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    model_name: str | None,
    app_config: AppConfig | None,
    evaluator_model_factory: Any | None = None,
    abort_event: asyncio.Event | None = None,
    user_id: str | None = None,
    deerflow_trace_id: str | None = None,
) -> dict[str, Any] | None:
    """활성 goal을 평가하고, 필요하면 숨겨진 continuation 입력을 반환한다.

    참고: 아래의 재조회는 continuation을 큐에 넣기 전에 경쟁하는 사용자 메시지나
    ``/goal clear``를 잡아낸다. goal write는 thread별로 직렬화되고 자신이 읽은 checkpoint
    id를 함께 넘기므로, 오래된 evaluator write는 더 새로운 goal 변경을 덮어쓰지 않고
    물러난다.
    """
    if checkpointer is None:
        return None
    if abort_event is not None and abort_event.is_set():
        return None

    try:
        goal = await read_thread_goal(checkpointer, thread_id)
    except Exception:
        logger.warning("Could not read goal for thread %s after run %s", thread_id, run_id, exc_info=True)
        return None
    if not goal or goal.get("status") != "active":
        return None

    async def _persist(
        goal: GoalState,
        evaluation: GoalEvaluation,
        no_progress_count: int,
        *,
        stand_down_reason: str | None = None,
        continuation_count: int | None = None,
    ) -> GoalState | None:
        """아직 유효한 goal 인스턴스에 대해 평가 결과를 기록한다."""
        return await _persist_goal_evaluation(
            bridge=bridge,
            checkpointer=checkpointer,
            thread_id=thread_id,
            run_id=run_id,
            goal=goal,
            evaluation=evaluation,
            no_progress_count=no_progress_count,
            continuation_count=continuation_count,
            stand_down_reason=stand_down_reason,
            evidence_signature=evidence_signature,
        )

    try:
        checkpoint_tuple = await _call_checkpointer_method(
            checkpointer,
            "aget_tuple",
            "get_tuple",
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )
        if checkpoint_tuple is None:
            return None
        checkpoint_id_before = _checkpoint_id(checkpoint_tuple)
        messages = await _materialized_checkpoint_messages(accessor, thread_id)
        conversation_signature_before = visible_conversation_signature(messages)
        evidence_signature = latest_visible_assistant_signature(messages)

        if not _has_durable_goal_turn_receipt(checkpoint_tuple, messages):
            evaluation = GoalEvaluation(
                satisfied=False,
                blocker="run_failed",
                reason="No durable assistant end-of-turn receipt was available.",
                evidence_summary="",
            )
            no_progress_count = compute_no_progress_count(goal, evaluation, evidence_signature=evidence_signature)
            await _persist(goal, evaluation, no_progress_count, stand_down_reason="no_durable_end_of_turn")
            return None

        if abort_event is not None and abort_event.is_set():
            return None
        evaluator_model = evaluator_model_factory() if evaluator_model_factory is not None else None
        evaluation = await evaluate_goal_completion(
            goal,
            messages,
            model=evaluator_model,
            model_name=model_name,
            app_config=app_config,
            thread_id=thread_id,
            user_id=user_id,
            deerflow_trace_id=deerflow_trace_id,
        )
        if abort_event is not None and abort_event.is_set():
            return None
    except Exception:
        logger.warning("Goal evaluator failed for thread %s after run %s", thread_id, run_id, exc_info=True)
        return None

    no_progress_count = compute_no_progress_count(goal, evaluation, evidence_signature=evidence_signature)

    # evaluator가 도는 동안 goal도, 보이는 대화도 바뀌지 않았는지 다시 확인한다. 평가와
    # 경쟁한 사용자 메시지나 /goal clear가 우선해야 한다.
    try:
        current_goal, current_checkpoint_tuple = await _reread_goal_and_checkpoint(checkpointer, thread_id)
    except Exception:
        logger.warning("Could not re-check goal state for thread %s after evaluation", thread_id, exc_info=True)
        return None

    if not _goal_instance_matches(goal, current_goal) or current_checkpoint_tuple is None:
        return None

    checkpoint_changed = _checkpoint_id(current_checkpoint_tuple) != checkpoint_id_before
    messages_changed = visible_conversation_signature(await _materialized_checkpoint_messages(accessor, thread_id)) != conversation_signature_before
    if checkpoint_changed or messages_changed:
        await _persist(current_goal, evaluation, no_progress_count, stand_down_reason="thread_changed_after_evaluation")
        return None

    if evaluation["satisfied"]:
        try:
            async with goal_thread_lock(thread_id):
                latest_checkpoint_tuple = await _call_checkpointer_method(
                    checkpointer,
                    "aget_tuple",
                    "get_tuple",
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                )
                if latest_checkpoint_tuple is None:
                    return None
                latest_goal = _read_checkpoint_goal(latest_checkpoint_tuple)
                if latest_goal is None or not _goal_instance_matches(goal, latest_goal):
                    return None
                values = await write_thread_goal(
                    checkpointer,
                    thread_id,
                    None,
                    as_node="goal_evaluator",
                    expected_checkpoint_id=_checkpoint_id(latest_checkpoint_tuple),
                )
            await bridge.publish(run_id, "values", serialize(values, mode="values"))
        except GoalWriteConflict:
            return None
        except Exception:
            logger.warning("Could not clear satisfied goal for thread %s", thread_id, exc_info=True)
        return None

    stand_down_reason = _stand_down_reason(goal, evaluation, no_progress_count)
    if stand_down_reason is not None or not should_continue_goal(goal, evaluation, no_progress_count=no_progress_count):
        await _persist(goal, evaluation, no_progress_count, stand_down_reason=stand_down_reason)
        return None

    next_count = int(goal.get("continuation_count", 0)) + 1
    updated_goal = await _persist(goal, evaluation, no_progress_count, continuation_count=next_count)
    if updated_goal is None:
        return None

    # 마지막 guard. 위의 저장이 checkpoint id를 올렸으므로, 여기서 경쟁하는 사용자 turn을
    # 감지하는 데 의미가 있는 것은 보이는 대화의 signature뿐이다.
    try:
        latest_goal, latest_checkpoint_tuple = await _reread_goal_and_checkpoint(checkpointer, thread_id)
    except Exception:
        logger.warning("Could not verify queued goal continuation for thread %s", thread_id, exc_info=True)
        return None
    if not _goal_instance_matches(updated_goal, latest_goal) or latest_checkpoint_tuple is None:
        return None
    if visible_conversation_signature(await _materialized_checkpoint_messages(accessor, thread_id)) != conversation_signature_before:
        # 여기서는 continuation_count를 넘기지 않는다. 위의 저장이 이미 그 값을
        # (next_count로) 커밋했다. next_count를 다시 넘기면 _persist_goal_evaluation의
        # race guard(#4088)가 그 write를 "current_count" 증가로 보고 +1을 더 얹어서,
        # 실제로는 전달되지 않고 물러나는 이 continuation 시도 하나를 continuation
        # budget에서 조용히 두 번 세게 된다. 생략하면 이미 커밋된 카운트가 그대로
        # 유지되며, 이 함수의 다른 모든 stand-down 호출 지점과 동일해진다.
        await _persist(
            latest_goal,
            evaluation,
            no_progress_count,
            stand_down_reason="thread_changed_before_continuation",
        )
        return None

    logger.info(
        "Run %s continuing thread %s for active goal (%d/%d)",
        run_id,
        thread_id,
        updated_goal.get("continuation_count", next_count),
        updated_goal.get("max_continuations", 0),
    )
    return {"messages": [make_goal_continuation_message(updated_goal, evaluation)]}


def _is_edit_replay_run(record: RunRecord) -> bool:
    metadata = record.metadata or {}
    return metadata.get("replay_kind") == "edit"


async def _ensure_finalizing_before_edit_failure(run_manager: RunManager, record: RunRecord) -> None:
    if _is_edit_replay_run(record) and not record.finalizing:
        await run_manager.set_finalizing(record.run_id, True)


async def _publish_restored_checkpoint_values(
    *,
    bridge: StreamBridge,
    run_id: str,
    accessor: CheckpointStateAccessor | None,
    thread_id: str,
) -> None:
    if accessor is None:
        return
    snapshot = await accessor.aget({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        await bridge.publish(run_id, "values", serialize(values, mode="values"))


@dataclass(frozen=True)
class RollbackPoint:
    """취소 이후 thread를 복원하는 데 쓰는, materialize된 run 이전 state.

    raw checkpoint blob으로는 Delta 채널 메시지를 복원할 수 없으므로(그 checkpoint에는
    materialize된 값이 없다), rollback은 raw pending writes에 더해 그 메시지와 delta
    모드에서 materialize된 비메시지 state까지 보존한다.
    """

    config: dict[str, Any]
    state_values: dict[str, Any]
    messages: tuple[Any, ...]
    metadata: dict[str, Any]
    pending_writes: tuple[tuple[str, str, Any], ...]


async def _capture_rollback_point(
    accessor: CheckpointStateAccessor,
    checkpointer: Any,
    read_config: dict[str, Any],
) -> RollbackPoint | None:
    """run 이전 checkpoint state와 그 raw pending writes를 materialize한다.

    thread에 아직 checkpoint가 없으면 ``None``을 반환한다. 그 경우 호출자는 기존
    삭제/초기화 rollback 계약을 그대로 따른다.
    """
    snapshot = await accessor.aget(read_config)
    snapshot_config = getattr(snapshot, "config", None) or {}
    configurable = snapshot_config.get("configurable") or {}
    if not configurable.get("checkpoint_id"):
        return None
    checkpoint_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", snapshot_config)
    raw_values = getattr(snapshot, "values", None) or {}
    messages = raw_values.get("messages") if isinstance(raw_values, dict) else None
    state_values = copy.deepcopy({key: value for key, value in raw_values.items() if key != "messages"}) if accessor.mode == "delta" and isinstance(raw_values, dict) else {}
    return RollbackPoint(
        config={
            "configurable": {
                "thread_id": configurable.get("thread_id"),
                "checkpoint_ns": configurable.get("checkpoint_ns") or "",
                "checkpoint_id": configurable.get("checkpoint_id"),
            }
        },
        state_values=state_values,
        messages=tuple(messages or ()),
        metadata=dict(getattr(snapshot, "metadata", None) or {}),
        pending_writes=tuple(getattr(checkpoint_tuple, "pending_writes", ()) or ()),
    )


def _complete_state_replacement_values(
    *,
    mutation_graph: Any,
    selected_values: dict[str, Any],
    current_values: dict[str, Any],
    run_id: str,
    operation: str,
) -> dict[str, Any]:
    """graph의 실제 schema를 통해 전체 state 교체 값을 만든다."""
    writable_fields = graph_writable_channels(mutation_graph)
    reducer_fields = graph_reducer_channels(mutation_graph)
    if writable_fields is None or reducer_fields is None:
        raise RuntimeError(f"Run {run_id} could not inspect the state schema for {operation}")

    replacement_values: dict[str, Any] = {}
    for field_name in writable_fields:
        if field_name in selected_values:
            replacement = copy.deepcopy(selected_values[field_name])
        elif field_name in current_values:
            # LangGraph에는 공개된 "채널 해제" 업데이트가 없다. 새 채널은 schema 기본값이
            # 있으면 그것을 노출하고(예: [] / {}), optional이거나 달리 생성할 수 없는
            # 채널은 None으로 초기화된다.
            channel = mutation_graph.channels.get(field_name)
            replacement = copy.deepcopy(channel.get()) if channel is not None and channel.is_available() else None
        else:
            continue
        replacement_values[field_name] = Overwrite(replacement) if field_name in reducer_fields else replacement
    return replacement_values


async def _linearize_delta_checkpoint_resume(
    *,
    accessor: CheckpointStateAccessor,
    checkpointer: Any,
    config: dict[str, Any],
    thread_id: str,
    run_id: str,
) -> list[Any] | None:
    """delta 모드의 checkpoint fork를 동등한 선형 write로 대체한다.

    오래된 checkpoint에서 재개하면 계보가 fork되는데, ``delta`` 모드에서는 그 fork의
    state를 올바르게 materialize할 수 없다. delta 이력 순회는 경로상의 각 조상에 저장된
    ``pending_writes`` 항목을 **전부** 모으는데, 공유된 부모는 버려진 형제 자식의 write도
    함께 갖고 있기 때문이다. 그 write들이 fork로 replay되므로, run은 자신이 대체하려던
    답변이 여전히 들어 있는 메시지 목록에서 시작한다. 분기된 thread에서 재생성할 때
    새로고침 후 예전 assistant 메시지가 새 것 옆에 다시 나타나는 형태로 드러났다 (#4458).
    postgres, sqlite, in-memory saver에서 모두 재현되었다. ``full`` 모드는 checkpoint가
    완전한 ``channel_values``를 담고 있어 replay가 필요 없으므로 영향받지 않는다.

    write-to-child 소유권은 upstream 계약(`BaseCheckpointSaver.get_delta_channel_history`와
    이를 override하는 saver들)의 몫이므로 여기서 다시 구현하지 않는다. 대신 fork를 그
    의미대로 표현한다. 요청된 checkpoint의 state를 materialize한 다음, 다른 자식이 없는
    **현재 head**에 교체 의미로 쓰고 선형으로 진행한다. materialize된 모든 채널이
    복원되며, 더 새로운 head에만 존재하는 채널은 schema 기본값(생성 가능한 기본값이 없으면
    ``None``)으로 초기화된다. 버려진 turn은 재작성된 head의 조상으로 checkpoint 이력에
    남는다.

    재개가 선형화되었으면 materialize된 메시지를, 할 일이 없었으면(full 모드, checkpoint
    선택자 없음, 비 root namespace, 또는 이미 head를 가리키는 선택자) ``None``을 반환한다.
    실패는 그대로 전파한다. 조용히 fork로 되돌아가면 이 함수가 막으려는 손상된 이력이
    저장되기 때문이다. worker 호출 지점은 rollback 캡처와 이 재작성 전체에 걸쳐
    ``_checkpoint_thread_lock``을 잡고 있다. 재진입 불가능한 그 lock을 이 헬퍼 안에서 다시
    잡으면 안 된다.
    """
    if checkpointer is None or accessor.mode != "delta":
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    checkpoint_id = configurable.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        return None
    if configurable.get("checkpoint_ns"):
        # subgraph namespace는 자체 계보를 갖는다. Gateway는 root checkpoint만 선택하므로
        # 그 외에는 건드리지 않는다.
        return None

    head_config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    head = await accessor.aget(head_config)
    if _checkpoint_id(head) == checkpoint_id:
        # head를 선택하는 것은 이미 선형이다. 아직 형제가 존재할 수 없다.
        return None

    source_config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": checkpoint_id}}
    snapshot = await accessor.aget(source_config)
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages") if isinstance(values, dict) else None
    if not isinstance(messages, list):
        raise RuntimeError(f"Run {run_id} could not materialize resume checkpoint {checkpoint_id}")

    # thread의 실제 schema를 통해 써서 모든 애플리케이션·middleware 채널이 복원될 수 있게
    # 한다. reducer 채널은 이미 집계된 값을 다시 병합하지 않고 교체하려면 Overwrite가
    # 필요하다.
    mutation_graph = build_state_mutation_graph("checkpoint_resume", accessor.mode, graph_state_schema(getattr(accessor, "graph", None)))
    selected_values = dict(values)
    head_values = getattr(head, "values", None) or {}
    head_values = dict(head_values) if isinstance(head_values, dict) else {}
    replacement_values = _complete_state_replacement_values(
        mutation_graph=mutation_graph,
        selected_values=selected_values,
        current_values=head_values,
        run_id=run_id,
        operation="checkpoint resume",
    )

    mutation_accessor = CheckpointStateAccessor.bind(mutation_graph, checkpointer, mode=accessor.mode)
    await mutation_accessor.aupdate(head_config, replacement_values, as_node="checkpoint_resume")
    configurable.pop("checkpoint_id", None)
    configurable.pop("checkpoint_map", None)
    logger.info("Run %s linearized a delta-mode resume of checkpoint %s onto thread %s", run_id, checkpoint_id, thread_id)
    return list(messages)


async def _rollback_to_pre_run_checkpoint(
    *,
    accessor: CheckpointStateAccessor | None,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    rollback_point: RollbackPoint | None,
    snapshot_capture_failed: bool,
) -> bool:
    """run 이전 state 전체를 복원하고 완료되었는지 보고한다.

    full 모드는 캡처된 run 이전 checkpoint를 fork하고 messages를 덮어쓴다. 나머지 채널은
    그 부모에서 상속된다. delta 모드는 취소된 경로가 같은 부모에 write를 붙인 뒤에는
    안전하게 fork할 수 없으므로, 대신 캡처된 모든 채널을 현재 head에 교체해 쓴다. 두 write
    모두 state 전용 mutation graph를 쓰며, 그 합성 ``rollback_restore`` node는 즉시 끝나고
    agent 작업을 예약하지 않는다.
    """
    if checkpointer is None:
        logger.info("Run %s rollback requested but no checkpointer is configured", run_id)
        return False

    if snapshot_capture_failed:
        logger.warning("Run %s rollback skipped: pre-run checkpoint capture failed", run_id)
        return False

    if rollback_point is None:
        await _call_checkpointer_method(checkpointer, "adelete_thread", "delete_thread", thread_id)
        logger.info("Run %s rollback reset thread %s to empty state", run_id, thread_id)
        return True

    configurable = rollback_point.config.get("configurable", {})
    if not configurable.get("checkpoint_id"):
        logger.warning("Run %s rollback skipped: pre-run checkpoint has no checkpoint id", run_id)
        return False

    if accessor is None:
        # 실제로는 도달하지 않는다. rollback 지점은 바인딩된 accessor를 통해서만 캡처될
        # 수 있다. fail-closed를 유지한다.
        logger.warning("Run %s rollback skipped: agent accessor unavailable", run_id)
        return False

    # thread의 실제 schema로 컴파일해서 middleware가 기여한 채널이 살아남게 한다(기본
    # ThreadState fallback은 그것들을 조용히 버린다).
    mutation_graph = build_state_mutation_graph("rollback_restore", accessor.mode, graph_state_schema(getattr(accessor, "graph", None)))
    mutation_accessor = CheckpointStateAccessor.bind(mutation_graph, checkpointer, mode=accessor.mode)
    if accessor.mode == "delta":
        # delta rollback fork는 checkpoint 재개와 같은 write 소유권 문제를 갖는다. 캡처된
        # 부모가 이제 취소된 형제의 write를 갖고 있기 때문이다. 대신 현재 head에 선형으로
        # 복원한다.
        restore_config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        current = await accessor.aget(restore_config)
        raw_current_values = getattr(current, "values", None) or {}
        current_values = dict(raw_current_values) if isinstance(raw_current_values, dict) else {}
        selected_values = copy.deepcopy(rollback_point.state_values)
        selected_values["messages"] = list(rollback_point.messages)
        replacement_values = _complete_state_replacement_values(
            mutation_graph=mutation_graph,
            selected_values=selected_values,
            current_values=current_values,
            run_id=run_id,
            operation="rollback",
        )
    else:
        restore_config = rollback_point.config
        replacement_values = {"messages": Overwrite(list(rollback_point.messages))}

    restored_config = await mutation_accessor.aupdate(
        restore_config,
        replacement_values,
        as_node="rollback_restore",
    )
    if not isinstance(restored_config, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config: expected dict")
    restored_configurable = restored_config.get("configurable", {})
    if not isinstance(restored_configurable, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config payload")
    restored_checkpoint_id = restored_configurable.get("checkpoint_id")
    if not restored_checkpoint_id:
        raise RuntimeError(f"Run {run_id} rollback restore did not return checkpoint_id")

    pending_writes = rollback_point.pending_writes
    if not pending_writes:
        return True

    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for item in pending_writes:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write is not a 3-tuple: {item!r}")
        task_id, channel, value = item
        if not isinstance(channel, str):
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write has non-string channel: task_id={task_id!r}, channel={channel!r}")
        writes_by_task.setdefault(str(task_id), []).append((channel, value))

    for task_id, writes in writes_by_task.items():
        await _call_checkpointer_method(
            checkpointer,
            "aput_writes",
            "put_writes",
            restored_config,
            writes,
            task_id=task_id,
        )
    return True


def _new_checkpoint_marker() -> dict[str, str]:
    marker = empty_checkpoint()
    return {"id": marker["id"], "ts": marker["ts"]}


def _bump_channel_version(checkpointer: Any, current_version: Any) -> Any:
    """checkpoint 채널의 다음 버전을 반드시 다른 값으로 반환한다.

    DB 기반 LangGraph saver(PostgresSaver / v4 SqliteSaver blob 레이아웃)는 채널 blob을
    ``channel_versions[<channel>]``를 키로 저장하므로, 새 값은 이전 값과 **반드시** 달라야
    한다. 가능하면 checkpointer의 ``get_next_version``에 위임한다. 그것이 각 saver가 고른
    표준 버전 체계(int, 단조 증가 float, UUID 형태 문자열)다. checkpointer가 그것을
    노출하지 않거나 ``None``/변하지 않은 값을 반환하면, 그래도 불일치를 보장하는 방어적
    증가로 fallback한다.
    """
    get_next_version = getattr(checkpointer, "get_next_version", None)
    if callable(get_next_version):
        try:
            next_version = get_next_version(current_version, None)
        except Exception:
            next_version = None
        if next_version is not None and next_version != current_version:
            return next_version
        # 방어적 증가로 넘어간다

    if isinstance(current_version, bool):
        # ``bool``은 ``int``의 하위 클래스다. boolean 자체에 더하면 결과는 어차피 int지만
        # 읽는 사람을 놀라게 하는 경로이므로, True/False를 1/0으로 취급한다.
        return int(current_version) + 1
    if isinstance(current_version, int):
        return current_version + 1
    if isinstance(current_version, float):
        # LangGraph의 기본 float 버전 체계(단조 증가)에 맞춘다.
        return current_version + 1.0
    if isinstance(current_version, str):
        try:
            return str(int(current_version) + 1)
        except ValueError:
            return f"{current_version}.1"
    return 1


def _checkpoint_identity(ckpt_tuple: Any | None, checkpoint: dict[str, Any]) -> str | None:
    tuple_config = getattr(ckpt_tuple, "config", {}) or {}
    tuple_configurable = tuple_config.get("configurable", {}) if isinstance(tuple_config, dict) else {}
    if isinstance(tuple_configurable, dict):
        checkpoint_id = tuple_configurable.get("checkpoint_id")
        if isinstance(checkpoint_id, str) and checkpoint_id:
            return checkpoint_id
    checkpoint_id = checkpoint.get("id")
    return checkpoint_id if isinstance(checkpoint_id, str) and checkpoint_id else None


def _checkpoint_namespace(ckpt_tuple: Any | None) -> str:
    tuple_config = getattr(ckpt_tuple, "config", {}) or {}
    tuple_configurable = tuple_config.get("configurable", {}) if isinstance(tuple_config, dict) else {}
    checkpoint_ns = tuple_configurable.get("checkpoint_ns", "") if isinstance(tuple_configurable, dict) else ""
    return checkpoint_ns if isinstance(checkpoint_ns, str) else ""


def _graph_input_messages(graph_input: Any | None) -> list[Any]:
    if not isinstance(graph_input, dict):
        return []
    messages = graph_input.get("messages")
    if isinstance(messages, list):
        return messages
    if isinstance(messages, tuple):
        return list(messages)
    return []


def _title_generation_state(channel_values: dict[str, Any], graph_input: Any | None) -> dict[str, Any]:
    state = dict(channel_values)
    messages = state.get("messages")
    if not messages:
        fallback_messages = _graph_input_messages(graph_input)
        if fallback_messages:
            state["messages"] = fallback_messages
    return state


def valid_duration_entry(run_id: Any, duration_seconds: Any) -> bool:
    """(run_id, duration_seconds)가 올바른 형태의 duration 항목인지 확인한다."""
    return isinstance(run_id, str) and bool(run_id) and isinstance(duration_seconds, int) and not isinstance(duration_seconds, bool)


async def persist_run_durations(
    *,
    checkpointer: Any,
    thread_id: str,
    durations: dict[str, int],
) -> bool:
    """검증된 run duration을 metadata 전용 checkpoint에 병합한다.

    duration은 누적되므로 이력 fast path가 최신 checkpoint 하나로 알려진 모든 turn을
    제공할 수 있다. 항목당 오버헤드(run_id당 약 50바이트)는 graph checkpoint마다 쓰이는
    messages 채널 blob에 비하면 무시할 수준이라 pruning이 필요 없다.
    """
    updates = {run_id: max(0, duration_seconds) for run_id, duration_seconds in durations.items() if valid_duration_entry(run_id, duration_seconds)}
    if not updates:
        return False

    ckpt_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with _checkpoint_thread_lock(thread_id):
        for _attempt in range(3):
            ckpt_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", ckpt_config)
            if ckpt_tuple is None:
                return False

            checkpoint = dict(getattr(ckpt_tuple, "checkpoint", {}) or {})
            metadata = dict(getattr(ckpt_tuple, "metadata", {}) or {})
            raw_run_durations = metadata.get("run_durations")
            run_durations = {key: value for key, value in raw_run_durations.items() if valid_duration_entry(key, value)} if isinstance(raw_run_durations, dict) else {}
            changed_durations = {run_id: duration for run_id, duration in updates.items() if run_durations.get(run_id) != duration}
            if not changed_durations:
                return False

            run_durations.update(changed_durations)
            parent_checkpoint_id = _checkpoint_identity(ckpt_tuple, checkpoint)
            latest_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", ckpt_config)
            latest_checkpoint = dict(getattr(latest_tuple, "checkpoint", {}) or {}) if latest_tuple is not None else {}
            if _checkpoint_identity(latest_tuple, latest_checkpoint) != parent_checkpoint_id:
                continue

            checkpoint.update(_new_checkpoint_marker())
            metadata["source"] = "update"
            prev_step = metadata.get("step")
            metadata["step"] = (prev_step + 1) if isinstance(prev_step, int) else 1
            metadata["run_durations"] = run_durations
            metadata["writes"] = {"runtime_run_duration": {"run_ids": sorted(changed_durations)}}

            checkpoint_ns = _checkpoint_namespace(ckpt_tuple)
            write_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            await _call_checkpointer_method(
                checkpointer,
                "aput",
                "put",
                write_config,
                checkpoint,
                metadata,
                {},
            )
            return True
    return False


async def _persist_run_duration(
    *,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    duration_seconds: int,
) -> None:
    """완료된 run 하나의 duration을 thread checkpoint metadata에 저장한다."""
    await persist_run_durations(
        checkpointer=checkpointer,
        thread_id=thread_id,
        durations={run_id: duration_seconds},
    )


async def _ensure_interrupted_title(*, checkpointer: Any, thread_id: str, app_config: AppConfig | None, graph_input: Any | None = None) -> str | None:
    """첫 turn이 중단된 run에 대해 로컬 fallback title을 저장한다.

    현재 저장된 title(기존 값이거나 새로 쓴 값)을 반환하고, 사용할 checkpoint가 없거나
    title 텍스트를 유도할 수 없으면 ``None``을 반환한다. 멱등하다. 이미 title이 있는
    checkpoint에 대해 다시 호출하면 새 checkpoint를 쓰지 않고 곧바로 빠져나온다.
    """
    from deerflow.agents.middlewares.title_middleware import TitleMiddleware

    middleware = TitleMiddleware(app_config=app_config) if app_config is not None else TitleMiddleware()
    ckpt_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    for _attempt in range(3):
        ckpt_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", ckpt_config)
        checkpoint = copy.deepcopy(getattr(ckpt_tuple, "checkpoint", {}) or {}) if ckpt_tuple is not None else empty_checkpoint()
        channel_values = dict(checkpoint.get("channel_values", {}) or {})
        existing_title = channel_values.get("title")
        if existing_title:
            return existing_title

        result = middleware._generate_title_result(_title_generation_state(channel_values, graph_input), allow_partial_exchange=True)
        title = result.get("title") if isinstance(result, dict) else None
        if not title:
            return None

        # ``empty_checkpoint()``는 매번 새 id를 만든다. 오래된 snapshot 비교에 쓸 만큼
        # 안정적인 identity는 실제 튜플만 갖고 있다.
        base_identity = _checkpoint_identity(ckpt_tuple, checkpoint) if ckpt_tuple is not None else None
        latest_tuple = await _call_checkpointer_method(checkpointer, "aget_tuple", "get_tuple", ckpt_config)
        latest_checkpoint = copy.deepcopy(getattr(latest_tuple, "checkpoint", {}) or {}) if latest_tuple is not None else empty_checkpoint()
        latest_identity = _checkpoint_identity(latest_tuple, latest_checkpoint) if latest_tuple is not None else None
        if base_identity is None:
            if latest_identity is not None:
                continue
        elif latest_identity != base_identity:
            continue

        checkpoint = latest_checkpoint
        channel_values = dict(checkpoint.get("channel_values", {}) or {})
        existing_title = channel_values.get("title")
        if existing_title:
            return existing_title

        channel_values["title"] = title
        marker = _new_checkpoint_marker()
        checkpoint.update({"id": marker["id"], "ts": marker["ts"], "channel_values": channel_values})

        # ``channel_versions["title"]``을 올리고 그 증가를 ``new_versions``에 선언해서
        # DB 기반 saver(SqliteSaver v4 / PostgresSaver)가 실제로 새 blob을 저장하게 한다.
        # 그 saver들은 ``put``에서 인라인 ``channel_values``를 떼어내고 ``new_versions``에
        # 나열된 채널의 blob만 쓴다. 구형 단일 테이블 sqlite saver는 ``new_versions``를
        # 무시하고 snapshot을 인라인하므로, 이 경로는 두 레이아웃 모두에 맞다. 같은 파일의
        # ``_rollback_to_pre_run_checkpoint``와 같은 방식이다.
        channel_versions = dict(checkpoint.get("channel_versions", {}) or {})
        next_title_version = _bump_channel_version(checkpointer, channel_versions.get("title"))
        channel_versions["title"] = next_title_version
        checkpoint["channel_versions"] = channel_versions

        metadata = dict(getattr(latest_tuple, "metadata", {}) or {})
        metadata["source"] = "update"
        prev_step = metadata.get("step")
        metadata["step"] = (prev_step + 1) if isinstance(prev_step, int) else 1
        metadata["writes"] = {"runtime_interrupt_title": {"title": title}}

        checkpoint_ns = _checkpoint_namespace(latest_tuple)
        # 이 write가 파생된 checkpoint를 부모로 삼는다. 부모 없는 raw write는 Delta 채널
        # replay 계보를 끊고 full 모드 이력 순회도 잘라먹는다.
        write_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": latest_identity}}
        await _call_checkpointer_method(
            checkpointer,
            "aput",
            "put",
            write_config,
            checkpoint,
            metadata,
            {"title": next_title_version},
        )
        return title

    return None


def _lg_mode_to_sse_event(mode: str) -> str:
    """LangGraph 내부 stream_mode 이름을 SSE 이벤트 이름으로 매핑한다.

    LangGraph의 ``astream(stream_mode="messages")``는 message 튜플을 만든다. SSE 프로토콜에서는
    클라이언트가 명시적으로 요청할 때 이것을 ``messages-tuple``이라 부르지만, LangGraph
    Platform이 쓰는 기본 SSE 이벤트 이름은 그냥 ``"messages"``다.
    """
    # 모든 LG 모드는 SSE 이벤트 이름과 1:1로 대응한다. "messages"는 "messages" 그대로다
    return mode


def _error_fallback_message_from_metadata(metadata: dict[str, Any], content: Any) -> str:
    detail = metadata.get("error_detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    reason = metadata.get("error_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    if isinstance(content, str) and content.strip():
        return content.strip()[:2000]
    return "LLM provider failed after retries"


def _message_id(obj: Any) -> str | None:
    """message 형태 객체에서 안정적인 message id를 best-effort로 추출한다."""
    msg_id = getattr(obj, "id", None)
    if isinstance(msg_id, str) and msg_id:
        return msg_id
    if isinstance(obj, dict):
        raw = obj.get("id")
        if isinstance(raw, str) and raw:
            return raw
    return None


def _try_extract_from_message(obj: Any, pre_existing_ids: set[str] | None = None) -> str | None:
    """message 객체나 dict 하나에서 fallback 마커 추출을 시도한다.

    id가 ``pre_existing_ids``에 있는 메시지는 건너뛴다. 그것들은 이 thread의 *이전* run이
    checkpoint한 이력이고, 거기 붙은 fallback 마커는 그 run이 끝날 때 이미 처리되었기
    때문이다. 이 필터가 없으면 fallback 마커로 끝난 과거 run 하나가 같은 thread의 모든
    후속 run을 ``error``로 표시하게 된다. LangGraph가 ``stream_mode="values"``로 전체
    메시지 이력을 replay하기 때문이다.
    """
    if pre_existing_ids:
        msg_id = _message_id(obj)
        if msg_id is not None and msg_id in pre_existing_ids:
            return None

    additional_kwargs = getattr(obj, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs.get("deerflow_error_fallback"):
        return _error_fallback_message_from_metadata(additional_kwargs, getattr(obj, "content", None))

    if isinstance(obj, dict):
        nested_kwargs = obj.get("additional_kwargs")
        if isinstance(nested_kwargs, dict) and nested_kwargs.get("deerflow_error_fallback"):
            return _error_fallback_message_from_metadata(nested_kwargs, obj.get("content"))
    return None


def _extract_llm_error_fallback_message(value: Any, pre_existing_ids: set[str] | None = None) -> str | None:
    """stream된 LangGraph chunk에서 LLM fallback 마커를 찾는다.

    model-call middleware가 반환하는 error fallback 메시지는 LLM end callback을 반드시
    거친다는 보장이 없지만, graph state chunk에는 나타난다.

    id가 ``pre_existing_ids``에 있는 메시지는 무시한다. 그것들은 같은 thread의 이전 run에서
    온 이력이고(LangGraph는 ``stream_mode="values"`` chunk로 messages 채널 전체를 replay한다),
    그 이력의 error fallback은 해당 run이 끝날 때 이미 처리되었다.
    """
    # fast path: stream_mode="values"가 만드는 큰 state chunk에는 최상위 "messages"
    # 리스트가 있다. 그 리스트만 훑으면 큰 state dict를 깊게 재귀 탐색하는 비용을 피한다.
    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, (list, tuple)):
            for msg in messages:
                result = _try_extract_from_message(msg, pre_existing_ids)
                if result is not None:
                    return result
            # fallback 마커는 messages 채널의 AI 메시지에 붙는다. values chunk의 다른
            # 곳에는 절대 나타나지 않는다.
            return None
        # 최상위 "messages"가 없다면 "updates" chunk(노드 이름을 키로 하는 작은 dict)일
        # 가능성이 높다. 아래 깊은 순회로 넘어간다. 이런 payload에는 저렴하다.

    # updates / messages / tuple / list 모드를 위한 깊은 순회. payload가 작으므로 여기서는
    # 전체 재귀가 허용된다.
    seen: set[int] = set()

    def walk(obj: Any) -> str | None:
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        result = _try_extract_from_message(obj, pre_existing_ids)
        if result is not None:
            return result

        if isinstance(obj, dict):
            for item in obj.values():
                result = walk(item)
                if result is not None:
                    return result
            return None

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                result = walk(item)
                if result is not None:
                    return result
        return None

    return walk(value)


def _collect_pre_existing_message_ids(values: Any) -> set[str]:
    """graph가 materialize한 channel value에서 안정적인 message ID를 모은다."""
    if not isinstance(values, dict):
        return set()
    messages = values.get("messages")
    if not isinstance(messages, (list, tuple)):
        return set()
    return {message_id for message in messages if (message_id := _message_id(message)) is not None}


def _unpack_stream_item(
    item: Any,
    lg_modes: list[str],
    stream_subgraphs: bool,
) -> tuple[str | None, Any, tuple[str, ...]]:
    """다중 모드 또는 subgraph stream 항목을 (mode, chunk, namespace)로 분해한다.

    ``namespace``는 ``subgraphs=True``일 때 LangGraph가 각 frame 앞에 붙이는 subgraph
    namespace 튜플이며, root graph frame에서는 비어 있다. 위임된 subagent graph는 부모의
    checkpoint namespace를 상속하므로(``subagents/executor.py`` 참고) 그 frame은 비어 있지
    않은 namespace로 도착하며 root frame으로 오인하면 안 된다.

    항목을 파싱할 수 없으면 ``(None, None, ())``을 반환한다.
    """
    if stream_subgraphs:
        if isinstance(item, tuple) and len(item) == 3:
            ns, mode, chunk = item
            namespace = tuple(str(part) for part in ns) if isinstance(ns, (list, tuple)) else (str(ns),)
            return str(mode), chunk, namespace
        if isinstance(item, tuple) and len(item) == 2:
            mode, chunk = item
            return str(mode), chunk, ()
        return None, None, ()

    if isinstance(item, tuple) and len(item) == 2:
        mode, chunk = item
        return str(mode), chunk, ()

    # fallback: 첫 번째 모드의 단일 요소 출력
    return lg_modes[0] if lg_modes else None, item, ()


def _compose_sse_event(sse_event: str, namespace: tuple[str, ...]) -> str:
    """LangGraph Platform 방식의, namespace가 붙은 SSE 이벤트 이름.

    root frame은 이벤트 이름을 그대로 두고, subgraph frame은 ``mode|ns1|ns2``가 되어
    클라이언트가 구분할 수 있게 한다. LangGraph SDK는 정확히 이 형태를
    (``event.split("|").slice(1)``로) 파싱해 subagent namespace가 붙은 값을 thread 뷰에서
    빼낸다.
    """
    if not namespace:
        return sse_event
    return "|".join((sse_event, *namespace))


async def _publish_stream_item(
    *,
    bridge: Any,
    run_id: str,
    mode: str,
    chunk: Any,
    namespace: tuple[str, ...],
    file_tool_chunk_batcher: Any,
    subagent_events: Any,
) -> None:
    """subgraph namespace를 유지한 채 stream frame 하나를 publish한다.

    subgraph frame을 이름 그대로의 이벤트로 publish하면 root graph를 사칭하게 된다. 그러면
    위임된 subagent의 ``values`` snapshot이 SDK 클라이언트의 thread 뷰 전체를 대체하고, 그
    토큰 chunk가 부모 message stream을 뒤덮는다 (#4399). 따라서 subgraph frame은 이벤트
    이름에 namespace를 유지하고 root 전용 소비자(file-tool chunk batcher, subagent 이벤트
    저장 — task_* lifecycle 이벤트는 이미 root frame이다)를 건너뛴다.
    """
    sse_event = _compose_sse_event(_lg_mode_to_sse_event(mode), namespace)
    if namespace:
        await bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))
        return
    if file_tool_chunk_batcher is not None and mode != "messages":
        pending_chunks = file_tool_chunk_batcher.finish() if mode == "values" else file_tool_chunk_batcher.flush()
        for publish_chunk in pending_chunks:
            await bridge.publish(run_id, "messages", serialize(publish_chunk, mode="messages"))
    chunks_to_publish = file_tool_chunk_batcher.push(chunk) if mode == "messages" and file_tool_chunk_batcher is not None else [chunk]
    for publish_chunk in chunks_to_publish:
        await bridge.publish(run_id, sse_event, serialize(publish_chunk, mode=mode))
    if mode == "custom":
        await subagent_events.add(chunk)
