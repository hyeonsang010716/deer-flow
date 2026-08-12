"""교체 가능한 세분화 authorization (리소스 수준 RBAC 및 그 이상)."""

from deerflow.authz.adapter import GuardrailAuthorizationAdapter
from deerflow.authz.enforcement import filter_tools_by_authorization
from deerflow.authz.principal import build_principal_from_context, normalize_authz_attributes
from deerflow.authz.provider import AuthorizationProvider, AuthzDecision, AuthzReason, AuthzRequest, Principal
from deerflow.authz.rbac import RbacAuthorizationProvider
from deerflow.authz.runtime import resolve_authorization_provider
from deerflow.authz.tool_filter import apply_tool_authorization

__all__ = [
    "AuthzDecision",
    "AuthzReason",
    "AuthzRequest",
    "AuthorizationProvider",
    "GuardrailAuthorizationAdapter",
    "Principal",
    "RbacAuthorizationProvider",
    "apply_tool_authorization",
    "build_principal_from_context",
    "filter_tools_by_authorization",
    "normalize_authz_attributes",
    "resolve_authorization_provider",
]
