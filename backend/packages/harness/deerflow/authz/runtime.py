"""Provider factory — 설정된 AuthorizationProvider를 해석하고 생성한다.

``AuthorizationConfig``로부터 authorization provider를 만드는 단일 진입점이다.
인스턴스를 캐싱하지 않는다. Phase 1B는 agent 빌드마다 한 번 해석해서 같은 인스턴스를
Layer 1과 Layer 2에 넘긴다.
"""

from __future__ import annotations

from deerflow.authz.provider import AuthorizationProvider
from deerflow.config.authorization_config import AuthorizationConfig
from deerflow.reflection import resolve_variable


def resolve_authorization_provider(
    config: AuthorizationConfig,
) -> AuthorizationProvider | None:
    """config에서 authorization provider를 해석한다.

    Returns:
        생성된 ``AuthorizationProvider`` 인스턴스. authorization이 비활성이면 ``None``.

    Raises:
        ValueError: ``enabled``가 True인데 provider가 설정되지 않았거나,
            class path가 잘못됐거나, 생성에 실패했거나, 인스턴스가
            ``AuthorizationProvider`` Protocol을 만족하지 않을 때.
    """
    if not config.enabled:
        return None

    if config.provider is None:
        raise ValueError("authorization.enabled is true but no provider is configured; set authorization.provider.use to a class path")

    class_path = config.provider.use
    try:
        provider_cls = resolve_variable(class_path, expected_type=type)
    except (ImportError, ValueError) as err:
        raise ValueError(f"Failed to resolve authorization provider class '{class_path}': {err}") from err

    kwargs = dict(config.provider.config) if config.provider.config else {}
    try:
        instance = provider_cls(**kwargs)
    except Exception as err:
        raise ValueError(f"Failed to construct authorization provider '{class_path}': {err}") from err

    if not isinstance(instance, AuthorizationProvider):
        raise ValueError(f"Authorization provider '{class_path}' does not satisfy the AuthorizationProvider Protocol")

    from deerflow.authz.rbac import RbacAuthorizationProvider

    if isinstance(instance, RbacAuthorizationProvider):
        try:
            instance.validate_role(config.default_role, field="authorization.default_role")
        except ValueError as err:
            raise ValueError(f"Invalid authorization default_role for provider '{class_path}': {err}") from err

    return instance
