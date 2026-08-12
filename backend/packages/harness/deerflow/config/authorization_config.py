"""세분화된 resource authorization 설정.

활성화하면 교체 가능한 :class:`~deerflow.authz.provider.AuthorizationProvider`가 resource 수준
authorization의 정책 두뇌가 되며, 두 계층에서 강제된다. 조립 시점의 capability 필터링(agent가
아예 볼 수 없는 tool)과 실행 시점의 거부(adapter를 통해
:class:`~deerflow.guardrails.middleware.GuardrailMiddleware`를 재사용)다. 기본값
``enabled: false``는 인증된 모든 user가 모든 tool, model, skill, sandbox에 접근하는 현재 동작을
유지한다.
"""

from pydantic import BaseModel, Field


class AuthorizationProviderConfig(BaseModel):
    """authorization provider 설정."""

    use: str = Field(description="Class path (e.g. 'deerflow.authz.rbac:RbacAuthorizationProvider')")
    config: dict = Field(default_factory=dict, description="Provider-specific settings passed as kwargs")


class AuthorizationConfig(BaseModel):
    """세분화된 resource authorization 설정.

    :class:`~deerflow.config.guardrails_config.GuardrailsConfig`와 같은 형태다. class 경로로
    로드하는 provider, fail-closed 기본값, 실시간 재로드가 가능한 singleton으로 구성된다.
    """

    enabled: bool = Field(default=False, description="Enable fine-grained authorization")
    fail_closed: bool = Field(default=True, description="Block access if the provider errors or identity is unresolved")
    default_role: str = Field(default="user", description="Role applied when user_role is None (e.g. unbound IM channels)")
    provider: AuthorizationProviderConfig | None = Field(default=None, description="Authorization provider configuration")


_authorization_config: AuthorizationConfig | None = None


def get_authorization_config() -> AuthorizationConfig:
    """authorization config를 반환한다. 로드되지 않았으면 기본값을 반환한다."""
    global _authorization_config
    if _authorization_config is None:
        _authorization_config = AuthorizationConfig()
    return _authorization_config


def load_authorization_config_from_dict(data: dict) -> AuthorizationConfig:
    """dict로부터 authorization config를 로드한다(AppConfig 로딩 중에 호출된다)."""
    global _authorization_config
    _authorization_config = AuthorizationConfig.model_validate(data)
    return _authorization_config


def reset_authorization_config() -> None:
    """캐싱된 config 인스턴스를 초기화한다. 테스트에서 singleton 누수를 막기 위해 쓴다."""
    global _authorization_config
    _authorization_config = None
