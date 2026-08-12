"""extension contract와 그 데이터 타입.

이 모듈 전체에서 지키는 호환성 규칙은 두 가지다.

  * 모든 Protocol 메서드는 기본 구현을 가진다. 나중에 메서드를 추가해도 이미 릴리스된
    extension에는 additive로 남는다.
  * 모든 선택적 dataclass 필드는 기본값을 가진다. 필드를 추가해도 additive로 남는다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from deerflow_extension_api.state import ExtensionData

if TYPE_CHECKING:  # pragma: no cover - 타입 체크 전용
    from deerflow_extension_api.placement import AgentBuildContext, MiddlewarePlacement

F = TypeVar("F", bound=Callable[..., Any])


# --- Host 투영 --------------------------------------------------------------


@dataclass(frozen=True)
class HostPolicySnapshot:
    """host가 실제로 강제하는 제한을 extension용으로 투영한 값.

    host의 AppConfig 대신 좁게 투영한다. AppConfig를 그대로 노출하면 모든 extension이
    harness 릴리스 주기에 묶인다. 모든 필드에 기본값이 있어 확장해도 additive로 남는다.
    """

    token_budget_enabled: bool = False
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    budget_warn_fraction: float | None = None
    budget_hard_fraction: float | None = None
    max_subagents_per_run: int | None = None


# --- Middleware -------------------------------------------------------------


class MiddlewareContributor(Protocol):
    def contribute_middlewares(
        self,
        app_store: ExtensionData,
        ctx: AgentBuildContext,
    ) -> Sequence[MiddlewarePlacement]:
        return ()


# --- 등록 표면 ---------------------------------------------------------------


@runtime_checkable
class ExtensionRegistry(Protocol):
    """``install()``에 넘겨지는 쓰기 전용 등록 표면.

    의도적으로 구조적이고 최소한이다. 이 첫 capability 슬라이스는 middleware 기여만
    노출하며, 이후 슬라이스는 기본 구현이 있는 등록 메서드를 추가해도 기존 구현을 깨지
    않는다. host의 실제 registry는 host 전용 기능(attribution, 위치 기반 rollback, build)을
    추가로 갖지만 여기에는 일부러 두지 않는다.
    """

    def middlewares(self, contributor: MiddlewareContributor) -> None:
        return None


#: 모든 extension이 노출하는 install() 진입점 시그니처.
ExtensionInstall = Callable[[ExtensionRegistry, Mapping[str, Any]], None]


# --- 선언 decorator ----------------------------------------------------------


def extension(*, api: str, name: str | None = None) -> Callable[[F], F]:
    """install 함수에 어떤 API 버전을 기준으로 작성됐는지 표시한다.

    선택 사항이다. 호환성은 기본적으로 pip의 의존성 해석이 담당한다. 이 decorator는
    버전이 어긋날 수 있는 `--no-deps` 설치와 editable monorepo checkout을 보완하며,
    깊은 곳에서 터지는 AttributeError를 조치 가능한 startup 진단으로 바꾼다.
    """

    def _decorate(func: F) -> F:
        func.__deerflow_api__ = api  # type: ignore[attr-defined]
        func.__deerflow_name__ = name  # type: ignore[attr-defined]
        return func

    return _decorate
