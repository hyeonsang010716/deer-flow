"""DeerFlow의 extension 메커니즘(host 쪽).

공개 contract는 별도의 `deerflow-extension-api` 패키지에 있고, 이 모듈은 로딩, 등록,
middleware 주입, hook 지점 연결을 구현한다.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from deerflow.extensions.loader import (
    Diagnostic,
    ExtensionLoadError,
    ExtensionSpec,
    load_extensions,
)
from deerflow.extensions.registry import EMPTY_EXTENSIONS, ExtensionRegistry, LoadedExtensions

#: run의 불변 extension snapshot을 담는 runtime context 키.
#:
#: 아래의 graph-build 바인딩은 동기 agent 구성 범위로 한정된 ContextVar이므로, tool이 작업을
#: 위임할 시점에는 이미 사라진 뒤다. runtime context가 run이 그 이후 코드에 도달하는 통로다.
#: 이중 밑줄 prefix는 host 내부용임을 나타낸다. Gateway는 호출자가 넘긴 ``__`` 키를 제거하고,
#: 이 snapshot은 공개 extension contract의 일부가 아니다.
EXTENSION_SNAPSHOT_CONTEXT_KEY = "__deerflow_extension_snapshot"

_loaded: LoadedExtensions = EMPTY_EXTENSIONS
_agent_build_extensions: ContextVar[LoadedExtensions | None] = ContextVar(
    "deerflow_agent_build_extensions",
    default=None,
)


def get_loaded_extensions() -> LoadedExtensions:
    """프로세스 전역에 로드된 extension을 반환한다.

    기존 `get_app_config()` 관례를 따르므로, 호출 지점이 명시적 override 파라미터를 받고
    없을 때 이 함수로 fallback할 수 있다.
    """
    return _loaded


def get_agent_build_extensions() -> LoadedExtensions:
    """agent graph를 구성하는 동안 run에 바인딩된 snapshot을 반환한다."""
    return _agent_build_extensions.get() or get_loaded_extensions()


@contextmanager
def bind_agent_build_extensions(loaded: LoadedExtensions) -> Iterator[None]:
    """불변 extension snapshot 하나를 동기 graph 조립에 바인딩한다."""
    token = _agent_build_extensions.set(loaded)
    try:
        yield
    finally:
        _agent_build_extensions.reset(token)


def resolve_run_extensions(context: Any | None) -> LoadedExtensions | None:
    """*context*에서 run의 extension snapshot을 반환하고, 없으면 ``None``을 반환한다.

    runtime context는 호출자가 병합할 수 있으므로 값을 신뢰하지 않고 타입을 검사한다.
    ``None``은 "이 호출자는 snapshot을 설치하지 않았다"는 뜻이며(embedded client, 독립
    LangGraph Server), 소비자는 기존 ``get_loaded_extensions()`` fallback을 그대로 쓴다.
    """
    if not isinstance(context, Mapping):
        return None
    snapshot = context.get(EXTENSION_SNAPSHOT_CONTEXT_KEY)
    return snapshot if isinstance(snapshot, LoadedExtensions) else None


def set_loaded_extensions(loaded: LoadedExtensions) -> None:
    global _loaded
    _loaded = loaded


def reset_loaded_extensions() -> None:
    """완전히 새로운 빈 집합으로 초기화한다. singleton 누수를 막기 위해 테스트에서 쓴다.

    EMPTY_EXTENSIONS를 재사용하지 않고 새 인스턴스를 만든다. 그 singleton은 변경 가능한
    ExtensionData app_store를 갖고 있어서, 그것으로 초기화하면 "빈" 상태에서 이뤄진 쓰기가
    이후의 모든 reset과 프로세스 전체에 계속 따라다니기 때문이다.
    """
    global _loaded
    _loaded = ExtensionRegistry().build()


_runtime_diagnostics: list[Diagnostic] = []
_runtime_diagnostics_lock = threading.RLock()
_MAX_RUNTIME_DIAGNOSTICS = 1000


def _trim_runtime_diagnostics() -> None:
    overflow = len(_runtime_diagnostics) - _MAX_RUNTIME_DIAGNOSTICS
    if overflow > 0:
        del _runtime_diagnostics[:overflow]


def initialize_runtime_diagnostics(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    """현재 host의 live diagnostic list를 설치하고 반환한다."""
    with _runtime_diagnostics_lock:
        _runtime_diagnostics.clear()
        _runtime_diagnostics.extend(diagnostics)
        _trim_runtime_diagnostics()
        return _runtime_diagnostics


def record_runtime_diagnostic(diagnostic: Diagnostic) -> None:
    """diagnostic 하나를 프로세스의 표준 sink에 모은다."""
    with _runtime_diagnostics_lock:
        _runtime_diagnostics.append(diagnostic)
        _trim_runtime_diagnostics()


def record_runtime_diagnostics(diagnostics: list[Diagnostic]) -> None:
    """diagnostic 묶음을 프로세스의 표준 sink에 모은다."""
    with _runtime_diagnostics_lock:
        _runtime_diagnostics.extend(diagnostics)
        _trim_runtime_diagnostics()


def get_runtime_diagnostics() -> list[Diagnostic]:
    with _runtime_diagnostics_lock:
        return list(_runtime_diagnostics)


def reset_runtime_diagnostics() -> None:
    with _runtime_diagnostics_lock:
        _runtime_diagnostics.clear()


__all__ = [
    "EMPTY_EXTENSIONS",
    "EXTENSION_SNAPSHOT_CONTEXT_KEY",
    "Diagnostic",
    "ExtensionLoadError",
    "ExtensionRegistry",
    "ExtensionSpec",
    "LoadedExtensions",
    "bind_agent_build_extensions",
    "get_agent_build_extensions",
    "get_loaded_extensions",
    "get_runtime_diagnostics",
    "initialize_runtime_diagnostics",
    "load_extensions",
    "record_runtime_diagnostic",
    "record_runtime_diagnostics",
    "reset_loaded_extensions",
    "reset_runtime_diagnostics",
    "resolve_run_extensions",
    "set_loaded_extensions",
]
