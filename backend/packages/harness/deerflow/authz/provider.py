"""세분화된 리소스 authorization을 위한 AuthorizationProvider protocol과 데이터 구조.

리소스 수준 authorization(RBAC 및 그 이상)의 정책 두뇌이며, :mod:`deerflow.guardrails`에
흡수시키지 않고 형제 모듈로 일부러 분리했다. ``GuardrailRequest``에 ``user_role``/``user_id``를
추가한 PR #3665는 guardrail의 범위를 *실행 시점* 검사로만 명시적으로 한정했다.
— *"保持 Guardrail 的职责边界不变：不新增 policy engine、RBAC 系统、
governance 子系统"*. 이 모듈이 #3665가 미뤄둔 RBAC 두뇌다.

하나의 정책을 **두 계층**에서 강제한다.

1. **Assembly-time capability filter** — role이 절대 쓸 수 없는 도구를 agent에 바인딩하기
   *전에* 제거한다. 모델이 아예 보지 못하므로 ``tool_search``가 다시 승격시킬 수도 없다(fail-closed).
2. **Run-time execution deny** — 얇은 adapter(:mod:`deerflow.authz.adapter` 참고)를 통해
   :class:`~deerflow.guardrails.middleware.GuardrailMiddleware`를 재사용하며,
   동적 리소스와 인자 기반 제약을 잡아낸다.

전체 설계 배경은 ``docs/plans/2026-07-10-pluggable-authorization-rfc.md``(issue #4063)를 참고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Principal:
    """신뢰된 runtime identity context에서 해석한 행위 주체.

    identity 필드는 ``inject_authenticated_user_context``(``app/gateway/services.py``)가
    이미 run context에 찍어둔 값과 동일한 형태라서, provider는 일관된 identity 하나만 본다.
    Layer 1과 실행 시점 guardrail adapter 모두 ``build_principal_from_context``를 쓴다.
    adapter는 요청마다 값을 다시 만들기 때문에 오래된 runtime identity를 캐싱하지 않는다.
    """

    user_id: str | None = None
    role: str | None = None
    oauth_provider: str | None = None
    oauth_id: str | None = None
    channel_user_id: str | None = None
    is_internal: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthzRequest:
    """authorization 검사마다 provider에 전달되는 context."""

    principal: Principal
    resource: str
    """리소스 종류. 예: ``"tool"``, ``"model"``, ``"skill"``, ``"sandbox"``, ``"mcp_server"``, ``"route"``."""

    action: str
    """리소스에 대한 동작. 예: ``"call"``, ``"list"``, ``"use"``, ``"activate"``, ``"execute"``, ``"read"``, ``"write"``."""

    target: str
    """리소스 식별자: tool 이름, model 이름, skill 이름, ``"route:threads:read"`` 등."""

    context: dict[str, Any] = field(default_factory=dict)
    """추가 context: ``thread_id``, ``run_id``, ``tool_call_id``, ``tool_input``, ``is_subagent`` 등."""


@dataclass
class AuthzReason:
    """allow/deny 판정의 구조화된 사유."""

    code: str
    message: str = ""


@dataclass
class AuthzDecision:
    """provider의 allow/deny 판정 결과."""

    allow: bool
    reasons: list[AuthzReason] = field(default_factory=list)
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuthorizationProvider(Protocol):
    """플러그인 가능한 세분화 authorization의 계약.

    이 메서드들만 있으면 어떤 클래스든 동작한다. 베이스 클래스는 필요 없다.
    provider는 ``resolve_variable()``로 class path를 통해 로드되며, DeerFlow가 model, tool,
    sandbox, guardrail에 쓰는 것과 같은 방식이다.

    ``resource``, ``action``, ``target``은 enum이 아니라 자유 형식 문자열이라서
    새 리소스 종류나 provider 고유 리소스를 추가해도 스키마를 바꿀 필요가 없다.
    내장 RBAC provider가 이 값들을 해석하며, 커스텀 provider는 각자 정의한다.
    """

    name: str

    def authorize(self, request: AuthzRequest) -> AuthzDecision:
        """호출 단위 판정. Layer 2(실행)와 route 검사에 쓰인다."""
        ...

    async def aauthorize(self, request: AuthzRequest) -> AuthzDecision:
        """비동기 버전."""
        ...

    def filter_resources(
        self,
        principal: Principal,
        resource_type: str,
        candidates: list[str],
    ) -> list[str]:
        """Layer 1: assembly 시점의 일괄 가시성 필터.

        *candidates* 중 principal이 볼 수 있는 부분집합을 반환한다.
        필수 메서드다. 정적 role→resource 맵이 없는 provider는 항목마다
        :meth:`authorize`에 위임해 허용된 부분집합만 반환하면 된다.
        정적 맵이 있는 provider는 O(1) 필터링과 fail-closed 가시성을 위해 이 메서드를 재정의할 수 있다.
        """
        ...
