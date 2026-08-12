"""수동 thread 컨텍스트 compaction 헬퍼."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace

from langgraph.types import Overwrite

from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware, SummaryGenerationError, create_summarization_middleware
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.runtime.checkpoint_state import CheckpointStateAccessor

logger = logging.getLogger(__name__)


class ContextCompactionDisabled(RuntimeError):
    """summarization이 비활성화된 상태에서 수동 compaction을 요청하면 발생한다."""


class ContextCompactionFailed(RuntimeError):
    """압축 가능한 thread를 요약하지 못하면 발생한다."""


@dataclass(frozen=True)
class ThreadCompactionResult:
    """수동 컨텍스트 compaction 시도 후 반환되는 결과."""

    thread_id: str
    compacted: bool
    reason: str | None = None
    removed_message_count: int = 0
    preserved_message_count: int = 0
    summary_updated: bool = False
    checkpoint_id: str | None = None
    total_tokens: int = 0


def _create_compaction_middleware(
    *,
    app_config: AppConfig,
    keep: tuple[str, int | float] | None,
    run_model_name: str | None = None,
) -> DeerFlowSummarizationMiddleware:
    middleware = create_summarization_middleware(app_config=app_config, keep=keep, run_model_name=run_model_name)
    if middleware is None:
        raise ContextCompactionDisabled("Context compaction is disabled.")
    return middleware


def _safe_load_agent_config(agent_name: str, user_id: str | None):
    """custom agent의 config를 로드하고, 실패하면 ``None``을 반환한다.

    agent config가 없거나 파싱되지 않는다고 compaction이 실패해서는 안 된다. run model은
    best-effort 최적화이고 기본값이 안전한 fallback이다. 호출자가 ``asyncio.to_thread``로
    event loop 밖에서 실행하므로, 엄격한 blocking-IO detector가 파일 읽기를 잡지 않고
    여기의 넓은 ``except``가 loop에서 발생한 ``BlockingError``를 가릴 수도 없다.
    """
    from deerflow.config.agents_config import load_agent_config

    try:
        return load_agent_config(agent_name, user_id=user_id)
    except Exception:
        logger.warning("Could not load agent config for %r; using the default model for summarization", agent_name, exc_info=True)
        return None


async def _aresolve_thread_model_name(
    model_name: str | None,
    agent_name: str | None,
    user_id: str | None,
    app_config: AppConfig,
) -> str | None:
    """thread를 요약할 때 쓸 모델을 lead 해석 방식과 동일하게 결정한다.

    우선순위는 ``lead_agent._resolve_model_name``과 같다: 명시적 request 모델 override
    (설정된 모델 목록으로 검증)가 우선, 그다음 thread의 custom-agent 설정 모델,
    마지막으로 ``config.models[0]``. 수동 ``/compact``는 agent를 실행하지 않아서 선택된
    모델을 들고 있는 live runtime이 없다. 그래서 호출자(route / client)가 일반 run이
    ``context.model_name``을 넘기듯 ``model_name``으로 넘긴다. custom-agent config 읽기는
    request 모델이 없을 때만 수행하고, event loop 밖에서 실행하며, 사용자별 agent 디렉터리를
    찾을 수 있도록 소유자 ``user_id``를 넘긴다.
    """
    default = app_config.models[0].name if getattr(app_config, "models", None) else None
    candidate = model_name
    if not candidate and agent_name:
        agent_config = await asyncio.to_thread(_safe_load_agent_config, agent_name, user_id)
        if agent_config and agent_config.model:
            candidate = agent_config.model
    if candidate and app_config.get_model_config(candidate):
        return candidate
    return default


async def compact_thread_context(
    accessor: CheckpointStateAccessor,
    thread_id: str,
    *,
    keep: tuple[str, int | float] | None = None,
    force: bool = True,
    user_id: str | None = None,
    agent_name: str | None = None,
    model_name: str | None = None,
    app_config: AppConfig | None = None,
) -> ThreadCompactionResult:
    """thread의 오래된 메시지를 요약하고 compaction된 checkpoint를 기록한다."""
    resolved_app_config = app_config or get_app_config()
    run_model_name = await _aresolve_thread_model_name(model_name, agent_name, user_id, resolved_app_config)
    middleware = _create_compaction_middleware(app_config=resolved_app_config, keep=keep, run_model_name=run_model_name)

    read_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    snapshot = await accessor.aget(read_config)
    snapshot_config = snapshot.config or {}
    checkpoint_id = snapshot_config.get("configurable", {}).get("checkpoint_id")
    if not checkpoint_id:
        raise LookupError(f"Thread {thread_id} checkpoint not found")

    channel_values = snapshot.values or {}
    messages = channel_values.get("messages")
    if not isinstance(messages, list) or not messages:
        return ThreadCompactionResult(thread_id=thread_id, compacted=False, reason="not_enough_messages")

    state = {
        "messages": list(messages),
        "summary_text": channel_values.get("summary_text"),
    }

    runtime_context = {"thread_id": thread_id, "user_id": user_id}
    if agent_name:
        runtime_context["agent_name"] = agent_name
    runtime = SimpleNamespace(context=runtime_context)
    try:
        # ``raise_on_failure``는 ``force``와 무관하다. 수동 호출자는 생성 실패를 항상 드러내길
        # 원하므로(임계값을 넘긴 force=False 호출이라도), 아래 force=False의 "compaction할 것
        # 없음" 분기로 뭉개져서는 안 된다.
        result = await middleware.acompact_state(state, runtime, force=force, raise_on_failure=True)  # type: ignore[arg-type]
    except SummaryGenerationError as exc:
        # 압축 가능한 thread인데 (run-model fallback 이후에도) 요약 LLM이 실패한 것은
        # "compaction할 것 없음"과는 다른 실제 실패다. "compaction이 필요 없다"로 읽히는
        # compacted=False 결과 대신, 이미 소비되고 있는 ContextCompactionFailed 경로
        # (HTTP 500 -> frontend 에러 toast)로 보낸다.
        raise ContextCompactionFailed("summary generation failed") from exc
    if result is None:
        return ThreadCompactionResult(thread_id=thread_id, compacted=False, reason="not_enough_messages")

    updated_config = await accessor.aupdate(
        snapshot.config,
        {
            "messages": Overwrite(list(result.preserved_messages)),
            "summary_text": result.summary_text,
        },
        as_node="manual_compaction",
    )
    new_checkpoint_id = updated_config.get("configurable", {}).get("checkpoint_id")

    return ThreadCompactionResult(
        thread_id=thread_id,
        compacted=True,
        removed_message_count=len(result.messages_to_summarize),
        preserved_message_count=len(result.preserved_messages),
        summary_updated=True,
        checkpoint_id=new_checkpoint_id,
        total_tokens=result.total_tokens,
    )
