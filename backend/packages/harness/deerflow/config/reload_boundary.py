"""설정 hot-reload 경계의 단일 기준점.

Bytedance/deer-flow issue #3144: gateway 요청 의존성은 매 요청마다
``get_app_config()``로 ``AppConfig``를 해석하므로, 실행 단위 필드는 gateway 재시작 없이
다음 메시지부터 반영된다. 이 모듈에 등록된 필드는 gateway가 기동 시 한 번만 붙잡는
**인프라** 영역(engine, singleton, IM client, logging handler)이라 런타임에 바꾸려면
프로세스 재시작이 필요하다.

레지스트리는 두 종류를 담는다.

- ``AppConfig`` 최상위 필드(``database``, ``checkpointer``, ``run_events``,
  ``stream_bridge``, ``sandbox``, ``log_level``). 이들은
  :func:`format_field_description`이 만드는 표준 ``"startup-only: ..."`` prefix를 각
  Pydantic ``Field(description=...)``이 그대로 갖고 있어, IDE hover에서 필드 옆에 경계가
  바로 드러난다.
- ``AppConfig`` 스키마에 없는 ``config.yaml`` 최상위 섹션(``channels``). 스키마 수준에서
  표준화할 수 없으므로 이 레지스트리가 유일한 기준이다.

앞으로 만들 "재시작 필요" 스캐너(운영 도구, lint hook, 문서 생성기)는 산문을 다시 파싱하지
말고 이 레지스트리를 기준으로 삼아야 한다.
"""

from __future__ import annotations

from collections.abc import Iterator

#: 재시작이 필요한 필드 설명이 모두 시작해야 하는 표준 prefix. ``test_reload_boundary``가
#: 양방향을 강제한다. 등록된 필드는 스키마에서 이 prefix를 써야 하고, 이 prefix를 쓰는
#: 스키마 필드는 반드시 레지스트리에 있어야 한다.
STARTUP_ONLY_PREFIX = "startup-only:"


#: 재시작이 필요한 필드 경로와 사람이 읽을 수 있는 이유의 매핑.
#:
#: 이유 텍스트는 ``Field(description=...)``에 그대로 노출되므로, 단순히 "재시작 필요"가
#: 아니라 *어느 코드*가 스냅샷을 붙잡는지 설명해야 한다. 그래야 값을 바꾸는 운영자가 어떤
#: 하위 시스템을 재시작해야 하는지 안다.
STARTUP_ONLY_FIELDS: dict[str, str] = {
    "plugins": ("load_extensions() runs once during create_app() and the process-wide middleware registry is not rebuilt on config.yaml edits; adding, removing or reconfiguring a plugin requires a restart."),
    "database": ("init_engine_from_config() runs once during langgraph_runtime() startup; the SQLAlchemy engine holds the connection pool and is not rebuilt on config.yaml edits."),
    "checkpointer": ("make_checkpointer() binds the persistent checkpointer once at startup, including SQLite WAL / busy_timeout settings."),
    "run_events": ("make_run_event_store() picks the memory- vs SQL-backed implementation at startup and is frozen onto app.state.run_events_config to stay paired with the underlying event store."),
    "agent_storage": ("langgraph_runtime() validates agent_storage.backend against database.backend once at startup, and the db backend's synchronous SQLAlchemy engine is process-cached on first use; switching backend needs a restart."),
    "stream_bridge": ("make_stream_bridge() constructs the stream-bridge singleton once during startup."),
    "sandbox": ("get_sandbox_provider() caches the provider singleton (``_default_sandbox_provider``); a different ``sandbox.use`` class path only takes effect on next process start."),
    "log_level": (
        "apply_logging_level() runs only during app.py startup; it sets the deerflow/app logger levels and may lower root handler thresholds so configured messages can propagate. A freshly reloaded AppConfig does not retrigger it."
    ),
    "logging": (
        "configure_logging() runs only during app.py startup; it installs/removes the trace-context filter and the enhanced formatter on root handlers, "
        "and TraceMiddleware captures logging.enhance.enabled once at startup so response X-Trace-Id headers, log trace_id fields, and Langfuse "
        "deerflow_trace_id stay coherent. A freshly reloaded AppConfig does not retrigger any of this."
    ),
    # AppConfig Pydantic 스키마에 없다. channel 자격 증명은 lifespan 기동 시
    # ``start_channel_service()``가 한 번 직접 소비하며, 살아 있는 channel client는
    # config.yaml 수정으로 다시 만들어지지 않는다.
    "channels": ("start_channel_service() is invoked once during startup; the live IM channel clients (Feishu, Slack, Telegram, DingTalk) are not rebuilt when channels.* changes."),
    "channel_connections": (
        "start_channel_service() wires the connection repository and channel workers once at startup, and the channel-connections router caches the merged provider config on app.state; channel_connections.* edits need a restart."
    ),
    "scheduler": (
        "ScheduledTaskService is constructed and started once during Gateway lifespan startup; enabled, poll_interval_seconds, lease_seconds, "
        "and max_concurrent_runs are captured into the service instance and the background poller task is not rebuilt on config.yaml edits."
    ),
    "mcp_tasks": (
        "McpTaskService is constructed and started once during Gateway lifespan startup; enabled, poll_interval_seconds, lease_seconds, "
        "and max_concurrent_polls are captured into the service instance and the background poller task is not rebuilt on config.yaml edits."
    ),
    "run_ownership": (
        "RunOwnershipConfig is captured once into RunManager at langgraph_runtime() startup; the lease heartbeat background task is created and "
        "started there, and heartbeat_enabled / lease_seconds / grace_seconds are not re-read on config.yaml edits."
    ),
    "dedupe_storage": (
        "make_inbound_dedupe_store() resolves the inbound dedupe store once when ChannelService is constructed at startup; the store "
        "(in-process memory or shared Postgres) is captured onto ChannelManager and is not rebuilt on config.yaml edits."
    ),
}


def iter_startup_only_field_paths() -> Iterator[str]:
    """등록된 재시작 필요 필드 경로를 모두 순회한다."""
    return iter(STARTUP_ONLY_FIELDS)


def is_startup_only_field(field_path: str) -> bool:
    """*field_path*가 재시작 필요로 등록돼 있으면 ``True``를 반환한다.

    최상위 경로(``"database"``, ``"sandbox"`` 등)만 받는다. 경계가 섹션 단위이지 말단 키
    단위가 아니므로 ``"database.url"`` 같은 중첩 키는 다루지 않는다.
    """
    return field_path in STARTUP_ONLY_FIELDS


def format_field_description(field_path: str, *, field_doc: str | None = None) -> str:
    """등록된 필드의 표준 설명 문자열을 만든다.

    ``AppConfig``의 ``Field(description=...)``에서 사용한다. 그래야 IDE hover 텍스트가
    레지스트리와 일치하고 drift 테스트가 양쪽을 서로 고정할 수 있다.

    Args:
        field_path: 등록된 최상위 필드 경로(예: ``"log_level"``).
        field_doc: 필드 자체에 대한 사람 대상 설명(허용 값, 의미 등, 선택).
            주어지면 ``startup-only:`` 마커 블록 뒤에 빈 줄로 구분해 덧붙인다. 그러면 IDE
            hover에 재시작 필요 이유와 일반 문서가 함께 보인다. 이렇게 조합하면 기계가
            읽는 기준인 선두 마커는 유지하면서, 레지스트리 도입 전에
            ``Field(description=)``이 담던 설명도 되살릴 수 있다.

    Raises:
        KeyError: *field_path*가 등록돼 있지 않은 경우. 의도적이다. 조용히 placeholder를
            반환하면 오타가 drift 검사를 빠져나간다.
    """
    reason = STARTUP_ONLY_FIELDS[field_path]
    header = f"{STARTUP_ONLY_PREFIX} {reason}"
    if field_doc is None:
        return header
    return f"{header}\n\n{field_doc.strip()}"
