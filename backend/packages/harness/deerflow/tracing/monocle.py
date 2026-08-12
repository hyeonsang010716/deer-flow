"""Monocle telemetry. ``MONOCLE_TRACING``이 설정되면 Gateway lifespan에서 한 번만 초기화한다."""

from __future__ import annotations

import logging

from deerflow.config import (
    get_enabled_tracing_providers,
    get_tracing_config,
    is_monocle_tracing_enabled,
)

logger = logging.getLogger(__name__)

# build_tracing_callbacks()가 읽는다. MONOCLE_TRACING은 켰지만 Gateway lifespan setup을 실행한
# 적 없는 embedded/TUI 프로세스에 힌트를 주기 위함이다.
_setup_completed = False


def is_monocle_setup_completed() -> bool:
    """이 프로세스에서 :func:`setup_monocle_tracing_if_enabled`가 실행됐는지 여부."""
    return _setup_completed


def setup_monocle_tracing_if_enabled() -> bool:
    """``MONOCLE_TRACING``이 켜져 있으면 Monocle telemetry를 초기화하고, 아니면 no-op이다.

    ``monocle_apptrace.setup_monocle_telemetry()``는 멱등하므로 이 함수는 config로만 걸러내는
    얇은 wrapper로 남는다. 활성화된 경우 ``True``를 반환한다.
    """
    if not is_monocle_tracing_enabled():
        return False

    monocle = get_tracing_config().monocle
    # 계측 전에 알 수 없는 MONOCLE_EXPORTERS 값이나 누락된 OKAHU_API_KEY에 대해 명확한 메시지와
    # 함께 즉시 실패한다. run별 callback 경로가 아니라 여기서 검증하므로 config 오타가 agent run을
    # 깨뜨리지 않는다.
    monocle.validate()

    # Langfuse(v4, 역시 OTel 기반)와의 공존은 검증되어 있다. 나중에 초기화되는 라이브러리가 기존
    # 전역 TracerProvider를 재사용하고 자신의 span processor를 붙이므로 어느 쪽도 span을 잃지
    # 않는다(test_coexists_with_langfuse 참고). 두 processor가 모든 span을 보므로, 둘 다 켜져
    # 있으면 Monocle exporter가 Langfuse의 span도 함께 내보낸다.
    exporters = monocle.exporters

    # `console`은 로컬 stdout에 머무르므로 원격 exporter만 경고 대상으로 잡는다.
    off_box = [e for e in monocle.exporter_list if e not in ("file", "console")]
    if off_box:
        # Monocle exporter는 공유 전역 provider의 모든 span을 보므로, 함께 켜진 OTel provider의
        # span도 외부로 나간다.
        langfuse_note = " Langfuse is also enabled and shares the global provider, so its spans are exported there as well." if "langfuse" in get_enabled_tracing_providers() else ""
        logger.warning(
            "Monocle is exporting trace data (prompts, tool inputs/outputs, completions) beyond the local .monocle/ file via: %s. Make sure that destination is trusted.%s",
            ", ".join(off_box),
            langfuse_note,
        )

    try:
        from monocle_apptrace import setup_monocle_telemetry
    except ImportError as exc:
        raise RuntimeError("MONOCLE_TRACING is enabled but monocle_apptrace is not installed. Install the 'monocle' extra: `uv sync --extra monocle` in backend/, or `pip install 'deerflow-harness[monocle]'`.") from exc

    # monocle_exporters_list는 콤마로 구분된 문자열을 그대로 받는다(monocle_apptrace의 API).
    setup_monocle_telemetry(workflow_name="deer-flow", monocle_exporters_list=exporters)
    global _setup_completed
    _setup_completed = True
    logger.info("Monocle telemetry enabled (exporters=%s)", exporters)
    return True
