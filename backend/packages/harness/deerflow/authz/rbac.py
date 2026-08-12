"""내장 RBAC authorization provider.

config에서 role→resource 정책을 읽어 생성 시점에 불변 구조로 컴파일한다.
deny는 언제나 allow를 이긴다. 알 수 없거나 없는 role은 조용히 허용하지 않고
``ValueError``를 던져서, 실행 계층의 ``fail_closed``가 최종 판단을 내리게 한다.

전체 의미 표는 ``docs/plans/2026-07-15-authz-phase1a-implementation-plan.md`` §3.3을 참고한다.
"""

from __future__ import annotations

from typing import Any

from deerflow.authz.provider import (
    AuthzDecision,
    AuthzReason,
    AuthzRequest,
    Principal,
)

# resource-type → config-key 명시적 매핑. ``AuthzRequest.resource``(단수, 예: "tool")가
# config 키(복수, 예: "tools")와 다를 때 조용히 잘못 조회되는 것을 막는다.
_RESOURCE_POLICY_KEYS: dict[str, str] = {
    "tool": "tools",
    "model": "models",
    "skill": "skills",
    "sandbox": "sandbox",
    "mcp_server": "mcp_servers",
    "route": "routes",
}

_ALL = object()  # "모든 후보 허용"을 뜻하는 sentinel
_ABSENT = object()  # "dict에 키가 없음"을 뜻하는 sentinel


# resource policy dict에서 지원하는 유일한 키들. 그 외 키(오타, 알 수 없는 필드)는
# 조용한 잘못된 권한 부여를 막기 위해 생성 시점에 거부한다.
_SUPPORTED_POLICY_KEYS: frozenset[str] = frozenset({"allow", "deny"})


def _require_non_empty_string(value: object, *, field: str) -> str:
    """검증된 요청 식별자를 반환하거나 일관된 경계 오류를 던진다."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    return value


class _CompiledPolicy:
    """(role, resource_type) 한 쌍에 대한 불변·사전 검증된 정책."""

    __slots__ = ("allowed", "denied")

    def __init__(self, *, allowed: frozenset[str] | object, denied: frozenset[str]):
        self.allowed = allowed
        self.denied = denied

    def is_allowed(self, target: str) -> bool:
        # deny가 언제나 우선한다.
        if target in self.denied:
            return False
        if self.allowed is _ALL:
            return True
        return target in self.allowed


class RbacAuthorizationProvider:
    """내장 role 기반 authorization provider.

    ``roles`` mapping으로 설정하며, 각 role은 resource-type 키를
    ``{allow: ..., deny: [...]}`` 정책에 매핑한다. 정책 설정은 생성 시점에 전부 검증하고,
    요청 경로에서는 멤버십 검사 전에 식별자를 검증한다.

    정책의 범위는 role, resource, target이다. ``AuthzRequest.action``은 protocol 호환을 위해
    받아들이지만 이 내장 provider에서는 규칙 차원으로 쓰지 않는다.

    설정 예시::

        roles:
          admin:
            tools: {allow: "*"}
          user:
            tools: {allow: "*", deny: ["update_agent"]}
          guest:
            tools: {allow: ["web_search", "read_file"]}
    """

    name = "rbac"

    def __init__(self, *, roles: dict[str, Any] | object = _ABSENT, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError(f"unknown provider config keys {sorted(kwargs, key=repr)}; supported: ['roles']")
        if roles is _ABSENT:
            raise ValueError("missing required provider config key 'roles'")
        if not isinstance(roles, dict):
            raise ValueError(f"roles must be a dict, got {type(roles).__name__}")

        # 모든 정책을 미리 컴파일한다.
        self._policies: dict[tuple[str, str], _CompiledPolicy] = {}
        self._known_roles: frozenset[str] = frozenset(roles.keys())

        for role_name, role_config in roles.items():
            if not isinstance(role_name, str) or not role_name:
                raise ValueError(f"role name must be a non-empty string, got {role_name!r}")
            if not isinstance(role_config, dict):
                raise ValueError(f"role '{role_name}' config must be a dict, got {type(role_config).__name__}")

            for resource_key, resource_policy in role_config.items():
                if not isinstance(resource_key, str) or not resource_key:
                    raise ValueError(f"role '{role_name}' has invalid resource key {resource_key!r}")
                mapped_resource_key = _RESOURCE_POLICY_KEYS.get(resource_key)
                if mapped_resource_key is not None and mapped_resource_key != resource_key:
                    raise ValueError(f"role '{role_name}' resource key '{resource_key}' is a reserved request alias; use '{mapped_resource_key}' in RBAC config")
                if not isinstance(resource_policy, dict):
                    raise ValueError(f"role '{role_name}' resource '{resource_key}' must be a dict, got {type(resource_policy).__name__}")

                compiled = self._compile_resource_policy(role_name, resource_key, resource_policy)
                self._policies[(role_name, resource_key)] = compiled

    def validate_role(self, role: str, *, field: str = "role") -> None:
        """operator가 설정한 role이 정의되어 있지 않으면 즉시 실패한다."""
        role = _require_non_empty_string(role, field=field)
        if role not in self._known_roles:
            if field == "role":
                raise ValueError(f"Unknown role '{role}'; known roles: {sorted(self._known_roles)}")
            raise ValueError(f"{field} '{role}' is not defined; known roles: {sorted(self._known_roles)}")

    @staticmethod
    def _compile_resource_policy(
        role_name: str,
        resource_key: str,
        policy: dict[str, Any],
    ) -> _CompiledPolicy:
        """resource policy 하나를 검증해 불변 구조로 컴파일한다.

        "키 없음"(기본값 사용)과 "키는 있으나 null"(잘못됨 — 거부)을 구분한다.
        알 수 없는 키(오타)는 조용한 잘못된 권한 부여를 막기 위해 거부한다.
        """
        # --- 알 수 없는 키 거부("alow" 같은 오타를 잡는다) ---
        unknown_keys = set(policy.keys()) - _SUPPORTED_POLICY_KEYS
        if unknown_keys:
            raise ValueError(f"role '{role_name}' resource '{resource_key}': unknown policy keys {sorted(unknown_keys, key=repr)}; supported: {sorted(_SUPPORTED_POLICY_KEYS)}")

        # --- allow ---
        raw_allow = policy.get("allow", _ABSENT)
        if raw_allow is _ABSENT:
            allowed: frozenset[str] | object = _ALL  # allow 없음 = 전부 허용(deny는 그대로 적용)
        elif raw_allow is None:
            raise ValueError(f"role '{role_name}' resource '{resource_key}': allow must not be null; omit the key, or use '*' / bool / list")
        elif raw_allow is True:
            allowed = _ALL
        elif raw_allow is False:
            allowed = frozenset()  # allow: false = 전부 거부
        elif isinstance(raw_allow, str):
            if raw_allow == "*":
                allowed = _ALL
            else:
                raise ValueError(f"role '{role_name}' resource '{resource_key}': allow string must be '*', got {raw_allow!r}")
        elif isinstance(raw_allow, (list, tuple)):
            for item in raw_allow:
                if not isinstance(item, str) or not item:
                    raise ValueError(f"role '{role_name}' resource '{resource_key}': allow list contains non-string or empty item {item!r}")
            allowed = frozenset(raw_allow)
        else:
            raise ValueError(f"role '{role_name}' resource '{resource_key}': allow must be '*', bool, or list of strings, got {type(raw_allow).__name__}")

        # --- deny ---
        raw_deny = policy.get("deny", _ABSENT)
        if raw_deny is _ABSENT or raw_deny is None:
            if raw_deny is None:
                raise ValueError(f"role '{role_name}' resource '{resource_key}': deny must not be null; omit the key for no deny list")
            denied: frozenset[str] = frozenset()
        elif isinstance(raw_deny, (list, tuple)):
            for item in raw_deny:
                if not isinstance(item, str) or not item:
                    raise ValueError(f"role '{role_name}' resource '{resource_key}': deny list contains non-string or empty item {item!r}")
            denied = frozenset(raw_deny)
        else:
            raise ValueError(f"role '{role_name}' resource '{resource_key}': deny must be a list of strings, got {type(raw_deny).__name__}")

        return _CompiledPolicy(allowed=allowed, denied=denied)

    def _resolve_policy(
        self,
        principal: Principal,
        resource: str,
        *,
        resource_field: str = "resource",
    ) -> _CompiledPolicy | None:
        """(role, resource_type)에 해당하는 컴파일된 정책을 조회한다.

        이 role+resource에 설정된 정책이 없으면 ``None``을 반환한다(제한 없음이라는 뜻).
        잘못된 resource 식별자, 알 수 없거나 없는 role에는 ``ValueError``를 던진다.
        """
        role = principal.role
        if role is None or role == "":
            raise ValueError("Principal has no role; cannot evaluate RBAC policy")

        self.validate_role(role)

        resource = _require_non_empty_string(resource, field=resource_field)
        resource_key = _RESOURCE_POLICY_KEYS.get(resource, resource)
        return self._policies.get((role, resource_key))

    def authorize(self, request: AuthzRequest) -> AuthzDecision:
        """authorization 요청 하나를 평가한다."""
        policy = self._resolve_policy(request.principal, request.resource)
        target = _require_non_empty_string(request.target, field="target")
        if policy is None:
            # 이 role+resource에 정책 없음 → 제한 없음.
            return AuthzDecision(
                allow=True,
                reasons=[AuthzReason(code="authz.no_policy", message="no policy configured")],
                policy_id="rbac:unrestricted",
            )

        if policy.is_allowed(target):
            return AuthzDecision(
                allow=True,
                reasons=[AuthzReason(code="authz.allowed")],
                policy_id="rbac:allow",
            )
        return AuthzDecision(
            allow=False,
            reasons=[
                AuthzReason(
                    code="authz.denied",
                    message=f"role '{request.principal.role}' is denied '{target}' on resource '{request.resource}'",
                )
            ],
            policy_id="rbac:deny",
        )

    async def aauthorize(self, request: AuthzRequest) -> AuthzDecision:
        return self.authorize(request)

    def filter_resources(
        self,
        principal: Principal,
        resource_type: str,
        candidates: list[str],
    ) -> list[str]:
        """일괄 가시성 필터.

        후보의 순서와 중복을 그대로 유지하고 항목을 추가하지 않으며,
        :meth:`authorize`와 동일한 role/resource 오류를 던진다.
        """
        policy = self._resolve_policy(principal, resource_type, resource_field="resource_type")
        if not isinstance(candidates, list):
            raise ValueError(f"candidates must be a list, got {type(candidates).__name__}")
        validated_candidates = [_require_non_empty_string(candidate, field=f"candidates[{index}]") for index, candidate in enumerate(candidates)]
        if policy is None:
            return validated_candidates

        return [candidate for candidate in validated_candidates if policy.is_allowed(candidate)]
