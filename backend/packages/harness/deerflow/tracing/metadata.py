"""Langfuse trace 속성 metadata builder.

Langfuse v4의 ``langchain.CallbackHandler``는 ``RunnableConfig.metadata``에서 정해진 예약 key
집합을 root trace로 끌어올린다:

- ``langfuse_session_id`` → trace를 묶는다(LangGraph thread → Langfuse Session)
- ``langfuse_user_id``    → trace user_id(Users 페이지의 기반)
- ``langfuse_trace_name`` → 사람이 읽을 수 있는 trace 이름
- ``langfuse_tags``       → trace tag

계약은 ``langfuse/langchain/CallbackHandler.py::_parse_langfuse_trace_attributes``와
https://langfuse.com/docs/observability/features/sessions 를 참고한다. 여기의 builder는
gateway/run worker가 호출 지점에 Langfuse 내부 사정을 흘리지 않고 올바른 metadata를 주입할 수
있게 하려고 존재한다.
"""

from __future__ import annotations

from typing import Any

from deerflow.config import get_enabled_tracing_providers
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY, get_current_trace_id, normalize_trace_id

# 순환 import를 피하려고 아래에서 lazy import한다. ``deerflow.runtime``이 run worker를 즉시
# import하고, 그 worker가 다시 ``deerflow.tracing``을 필요로 하기 때문이다.
_DEFAULT_TRACE_NAME = "lead-agent"


def build_langfuse_trace_metadata(
    *,
    thread_id: str | None,
    user_id: str | None = None,
    assistant_id: str | None = None,
    model_name: str | None = None,
    environment: str | None = None,
    deerflow_trace_id: str | None = None,
) -> dict[str, Any]:
    """``RunnableConfig.metadata``에 넣을 Langfuse trace 속성 metadata를 반환한다.

    Langfuse가 활성 tracing provider에 없으면 ``{}``를 반환하므로, 호출자는 LangSmith나 다른
    tracer에 영향을 주지 않고 결과를 무조건 병합할 수 있다.

    Args:
        thread_id: LangGraph thread id. ``langfuse_session_id``로 매핑된다.
        user_id: 유효 user id. ``None``이면 ``DEFAULT_USER_ID``로 대체되어, 인증 없는 모드에서도
            Langfuse Users 페이지가 동작한다.
        assistant_id: agent 식별자(선택). 기본값은 ``"lead-agent"``.
        model_name: model 이름. ``langfuse_tags``에 ``model:<name>``으로 들어간다.
        environment: 배포 env(예: ``"production"``). ``langfuse_tags``에 ``env:<value>``로
            들어간다.
        deerflow_trace_id: DeerFlow request trace id(선택). 생략하면 현재 request trace
            context 값을 쓴다.
    """
    if "langfuse" not in get_enabled_tracing_providers():
        return {}

    from deerflow.runtime.user_context import DEFAULT_USER_ID

    metadata: dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "langfuse_user_id": user_id or DEFAULT_USER_ID,
        "langfuse_trace_name": assistant_id or _DEFAULT_TRACE_NAME,
    }
    request_trace_id = normalize_trace_id(deerflow_trace_id) or get_current_trace_id()
    if request_trace_id:
        metadata[DEERFLOW_TRACE_METADATA_KEY] = request_trace_id

    tags: list[str] = []
    if environment:
        tags.append(f"env:{environment}")
    if model_name:
        tags.append(f"model:{model_name}")
    if tags:
        metadata["langfuse_tags"] = tags

    return metadata


def inject_langfuse_metadata(
    config: dict,
    *,
    thread_id: str | None,
    user_id: str | None = None,
    assistant_id: str | None = None,
    model_name: str | None = None,
    environment: str | None = None,
    deerflow_trace_id: str | None = None,
) -> None:
    """Langfuse trace 속성 metadata를 ``config["metadata"]``에 병합한다.

    gateway worker(``runtime/runs/worker.py``)와 embedded client(``client.py``)가 공유하므로
    두 경로가 서로 어긋날 수 없다.

    ``setdefault``를 쓰므로 호출자가 준 metadata가 이긴다. 예를 들어 frontend가 설정한
    ``langfuse_session_id`` 값은 그대로 유지된다. ``config`` dict는 제자리에서 변경된다.
    Langfuse가 활성 tracing provider에 없으면 아무 일도 하지 않는다.
    """
    langfuse_metadata = build_langfuse_trace_metadata(
        thread_id=thread_id,
        user_id=user_id,
        assistant_id=assistant_id,
        model_name=model_name,
        environment=environment,
        deerflow_trace_id=deerflow_trace_id,
    )
    if not langfuse_metadata:
        return

    merged_metadata = dict(config.get("metadata") or {})
    for key, value in langfuse_metadata.items():
        merged_metadata.setdefault(key, value)
    config["metadata"] = merged_metadata
