"""Layer 1 도구 authorization 필터링용 편의 wrapper.

provider 해석, Principal 생성, 도구 필터링을 한 번의 호출로 묶어서
세 조립 경로(lead agent, subagent, embedded client)가 한 줄로 유지되게 한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool

from deerflow.authz.enforcement import filter_tools_by_authorization
from deerflow.authz.principal import build_principal_from_context
from deerflow.authz.provider import AuthorizationProvider
from deerflow.authz.runtime import resolve_authorization_provider
from deerflow.config.app_config import AppConfig


def apply_tool_authorization(
    tools: list[BaseTool],
    *,
    context: Mapping[str, Any],
    app_config: AppConfig,
    authorization_provider: AuthorizationProvider | None = None,
) -> tuple[list[BaseTool], AuthorizationProvider | None]:
    """Layer 1 도구 authorization 필터링을 적용한다.

    provider를 해석하고(또는 Layer 1과 Layer 2가 같은 인스턴스를 공유하도록 호출자가 넘긴 것을
    재사용하고), *context*로 Principal을 만든 뒤, provider 정책으로 *tools*를 필터링한다.

    ``authorization.enabled``가 false면 아무것도 하지 않고 원래 도구와 ``None``을 반환한다.

    Args:
        tools: 후보 도구 목록(skill 필터링 등은 이미 끝난 상태).
        context: runtime context mapping(병합된 ``cfg`` dict 또는 ``self.*`` 필드로
            조립한 동등한 dict).
        app_config: 해석된 AppConfig. authorization 설정에 쓴다.
        authorization_provider: 이미 해석된 provider. ``None``이면 여기서
            ``app_config.authorization``으로부터 해석한다.

    Returns:
        ``(filtered_tools, provider)`` — 필터링된 도구 목록과 provider 인스턴스
        (Layer 2 middleware 연결에 넘기기 위한 값. authorization이 비활성이면 ``None``).
    """
    authz_config = app_config.authorization
    # 테스트의 Mock 객체 방어: MagicMock은 ``enabled`` 속성 접근에 truthy한 자식 mock을 돌려주므로
    # 문자열이 아닌 ``provider.use``로 provider 해석이 시작될 수 있다. 실제 AuthorizationConfig는
    # ``enabled: bool``이므로 진짜 ``True``가 아니면 건너뛴다.
    if authz_config.enabled is not True:
        return tools, None

    if authorization_provider is None:
        authorization_provider = resolve_authorization_provider(authz_config)

    if authorization_provider is None:
        return tools, None

    principal = build_principal_from_context(context, default_role=authz_config.default_role)
    filtered = filter_tools_by_authorization(
        tools,
        provider=authorization_provider,
        principal=principal,
        fail_closed=authz_config.fail_closed,
    )
    return filtered, authorization_provider
