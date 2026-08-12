"""config 기반 extension 로딩.

entry point는 `module.path:install` 형식으로 지정하며, guardrails provider가 이미 쓰는 것과
같은 `resolve_variable` 헬퍼로 해석한다. 로드 순서는 config list의 순서 그대로다. middleware
stack은 위치에 민감하므로 명시적이고 재현 가능한 순서가 중요하다.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deerflow_extension_api import API_VERSION
from pydantic import BaseModel, ConfigDict, Field

from deerflow.extensions.registry import ExtensionRegistry, LoadedExtensions
from deerflow.reflection import resolve_variable

logger = logging.getLogger(__name__)

DiagnosticLevel = Literal["debug", "info", "warning", "error"]


class ExtensionSpec(BaseModel):
    """config.yaml의 `plugins:` 목록 항목 하나."""

    model_config = ConfigDict(extra="forbid")

    use: str = Field(description="Entry point path, e.g. 'my_extension:install'")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Extension-private configuration, passed to install() verbatim",
    )
    required: bool = Field(
        default=False,
        description="When true, a load failure aborts startup instead of being skipped",
    )


@dataclass(frozen=True)
class Diagnostic:
    """특정 extension에 귀속된 로드/실행 시점 문제.

    현재 저장소에는 구조화된 diagnostics 채널이 없다. 이것은 실패의 출처를 추적 가능하게
    유지하는 것만을 목적으로 하는 최소한의 채널이다.
    """

    level: DiagnosticLevel
    source: str
    message: str

    @classmethod
    def error(cls, source: str, message: str) -> Diagnostic:
        return cls("error", source, message)

    @classmethod
    def warning(cls, source: str, message: str) -> Diagnostic:
        return cls("warning", source, message)

    @classmethod
    def info(cls, source: str, message: str) -> Diagnostic:
        return cls("info", source, message)

    @classmethod
    def debug(cls, source: str, message: str) -> Diagnostic:
        return cls("debug", source, message)


class ExtensionLoadError(RuntimeError):
    """`required: true`로 표시된 extension의 로드가 실패했을 때 발생한다."""


def _parse_version(version: object) -> tuple[int, ...] | None:
    if not isinstance(version, str):
        return None
    try:
        return tuple(int(part) for part in str.split(version, "."))
    except ValueError:
        return None


def _compatible(declared: str, current: str) -> bool:
    """단방향 호환성 검사이며, 계약의 생애 단계에 맞는 semver 창을 쓴다.

    1.0 이전에는 계약 표면이 관찰용뿐이고 minor에서 깨질 수 있으므로, 창은 같은 major.minor에
    patch만 추가되는 범위다(host >= declared). 1.0부터는 계약이 major 안에서 늘어나기만 하므로,
    더 새로운 host는 옛 extension과 계속 호환되지만 더 새로운 minor를 대상으로 작성된
    extension은 거부된다. host가 구현하지 않은 계약 추가분을 요구할 것이기 때문이다. 파싱할 수
    없는 버전은 통과시키지 않고 거부한다."""
    declared_parts = _parse_version(declared)
    current_parts = _parse_version(current)
    if not declared_parts or not current_parts:
        return False
    width = max(len(declared_parts), len(current_parts), 2)
    declared_padded = declared_parts + (0,) * (width - len(declared_parts))
    current_padded = current_parts + (0,) * (width - len(current_parts))
    if declared_padded[0] != current_padded[0]:
        return False
    if declared_padded[0] == 0 and declared_padded[1] != current_padded[1]:
        return False
    return current_padded >= declared_padded


def _range_for(declared: str) -> str:
    """``_compatible``의 규칙에 대응하는 pip 버전 범위. 조치 가능한 거부 메시지에 쓴다.
    선언된 버전을 파싱할 수 없으면 정확한 버전 지정으로 폴백한다. 메시지는 그 원인이 된 버전
    때문에 깨지면 안 되기 때문이다."""
    parts = _parse_version(declared)
    if not parts:
        return f"=={declared}"
    if parts[0] == 0:
        minor = parts[1] if len(parts) > 1 else 0
        return f">={declared},<0.{minor + 1}"
    return f">={declared},<{parts[0] + 1}.0"


def load_extensions(specs: Sequence[ExtensionSpec]) -> tuple[LoadedExtensions, list[Diagnostic]]:
    """설정된 extension을 모두 해석하고 install 한다.

    기본은 fail-open이다. 망가진 extension은 diagnostic만 남기고 건너뛰므로 Gateway는 계속
    기동한다. `required: true`는 이를 fail-closed로 바꾼다. 없을 때 관찰 가능성이 아니라 동작
    자체가 달라지는 extension을 위한 것이다.
    """
    registry = ExtensionRegistry()
    diagnostics: list[Diagnostic] = []
    loaded_sources: list[str] = []

    for spec in specs:
        try:
            install = resolve_variable(spec.use)
        except Exception as exc:
            message = f"could not resolve extension entry point: {exc}"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.error("Extension %s: %s", spec.use, message)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} failed to load") from exc
            continue

        if not callable(install):
            message = f"extension entry point is not callable: {type(install).__name__}"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.error("Extension %s: %s", spec.use, message)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} is not callable")
            continue

        try:
            declared = getattr(install, "__deerflow_api__", None)
        except Exception as exc:
            message = f"could not inspect extension-api version marker: {type(exc).__name__}"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.error("Extension %s: %s", spec.use, message)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} could not inspect api marker") from exc
            continue
        if declared is not None and _parse_version(declared) is None:
            message = f"extension declares invalid extension-api version marker of type {type(declared).__name__}; expected a dotted numeric string such as '0.1'"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.error("Extension %s: %s", spec.use, message)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} declares invalid api marker")
            continue
        if declared is not None:
            # ``isinstance(..., str)``는 subclass도 통과시키는데, 그 subclass의
            # ``__str__``/``__format__``이 비호환 diagnostic을 만드는 도중에 plugin 코드를
            # 실행할 수 있다. 호환성 검사와 렌더링 전에 base 구현으로 정규화한다.
            declared = str.__str__(declared)
        if declared is not None and not _compatible(declared, API_VERSION):
            message = f"extension requires extension-api {declared}, host provides {API_VERSION}. Install a matching version: pip install 'deerflow-extension-api{_range_for(declared)}'"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.error("Extension %s: %s", spec.use, message)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} declares incompatible api {declared}")
            continue

        # registry.discard(spec.use)가 아니라 위치 기반 rollback을 쓴다. 서로 다른 config로
        # 같은 `use`를 공유하는 spec이 정당하게 존재할 수 있는데, source 기준으로 지우면 이
        # spec과 `use`가 같을 뿐인, 앞서 성공적으로 install된 인스턴스까지 지워진다.
        mark = registry.mark()
        try:
            with registry.attributed_to(spec.use):
                install(registry, _frozen_config(spec.config))
        except Exception as exc:
            registry.rollback_to(mark)
            message = f"install() failed: {exc}"
            diagnostics.append(Diagnostic.error(spec.use, message))
            logger.exception("Extension %s: install() failed", spec.use)
            if spec.required:
                raise ExtensionLoadError(f"required extension {spec.use} failed to install") from exc
            continue

        loaded_sources.append(spec.use)

    # 서드파티 코드 로딩이야말로 운영자가 성공을 확실히 확인해야 하는 사건인데, 여기의 다른
    # 분기는 전부 실패 전용이다. 이 줄이 없으면 완전히 성공한 로드와 host가 아예 읽지 않은
    # `plugins:` 블록을 구분할 수 없다. x/y 카운트는 위에서 이미 남긴 실패별 오류를 반복하지
    # 않으면서 "전부 로드"와 "일부 건너뜀"을 구분해준다.
    if specs:
        logger.info("Extensions loaded: %d/%d (%s)", len(loaded_sources), len(specs), ", ".join(loaded_sources) or "none")
    else:
        # info가 아니라 debug다. plugin 미설정은 거의 모든 배포의 기본 상태이고, 무조건
        # 찍으면 순수한 부팅 노이즈가 된다.
        logger.debug("No extensions configured")

    return registry.build(), diagnostics


def _frozen_config(config: dict[str, Any]) -> Mapping[str, Any]:
    """extension에 자기 config 블록의 얕은 복사본을 넘긴다.

    얕은 복사이므로, extension이 다른 extension이나 호출자의 config dict에서 최상위 키를 다시
    할당하는 것은 막지만, 중첩 구조(list, dict)는 여전히 참조로 공유되어 제자리에서 변경될 수
    있다. 이 보장이 중요하다면 평평한 최상위 config 값을 쓴다.
    """
    return dict(config)
