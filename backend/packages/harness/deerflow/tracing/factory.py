from __future__ import annotations

import logging
from typing import Any

from deerflow.config import (
    get_enabled_tracing_providers,
    get_tracing_config,
    is_monocle_tracing_enabled,
    validate_enabled_tracing_providers,
)
from deerflow.tracing.monocle import is_monocle_setup_completed

logger = logging.getLogger(__name__)


def _create_langsmith_tracer(config) -> Any:
    from langchain_core.tracers.langchain import LangChainTracer

    return LangChainTracer(project_name=config.project)


def _create_langfuse_handler(config) -> Any:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    # langfuse>=4는 client singleton을 통해 프로젝트별 자격 증명을 초기화한다. LangChain
    # callback은 그렇게 설정된 client에 붙는다.
    Langfuse(
        secret_key=config.secret_key,
        public_key=config.public_key,
        host=config.host,
    )
    return LangfuseCallbackHandler(public_key=config.public_key)


def build_tracing_callbacks() -> list[Any]:
    """명시적으로 활성화된 모든 tracing provider의 callback을 만든다."""
    validate_enabled_tracing_providers()
    # Monocle은 callback provider가 아니다. 이 run 단위 경로는 Gateway lifespan setup을
    # 건너뛴 embedded 프로세스에 그 사실을 알려 주는 자리일 뿐이다.
    if is_monocle_tracing_enabled() and not is_monocle_setup_completed():
        logger.debug(
            "MONOCLE_TRACING is set but Monocle is not initialized in this process — only the Gateway lifespan runs setup automatically; embedded/TUI callers must call deerflow.tracing.setup_monocle_tracing_if_enabled() themselves."
        )
    enabled_providers = get_enabled_tracing_providers()
    if not enabled_providers:
        return []

    tracing_config = get_tracing_config()
    callbacks: list[Any] = []

    for provider in enabled_providers:
        if provider == "langsmith":
            try:
                callbacks.append(_create_langsmith_tracer(tracing_config.langsmith))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"LangSmith tracing initialization failed: {exc}") from exc
        elif provider == "langfuse":
            try:
                callbacks.append(_create_langfuse_handler(tracing_config.langfuse))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"Langfuse tracing initialization failed: {exc}") from exc

    return callbacks
