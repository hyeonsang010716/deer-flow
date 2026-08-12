"""AuthorizationProvider를 GuardrailProvider처럼 보이게 하는 adapter.

기존 :class:`~deerflow.guardrails.middleware.GuardrailMiddleware`가 tool call 시점에
:class:`~deerflow.authz.provider.AuthorizationProvider`의 결정을 강제할 수 있게 해준다 — 새
middleware 클래스가 필요 없다(RFC §6.1 참고).

adapter는 :class:`~deerflow.guardrails.provider.GuardrailRequest` 필드를
:class:`~deerflow.authz.provider.AuthzRequest` 필드로 매핑하고, authorization provider를 호출한
뒤 :class:`~deerflow.authz.provider.AuthzDecision`을
:class:`~deerflow.guardrails.provider.GuardrailDecision`으로 되돌린다.

Principal 생성은 :func:`~deerflow.authz.principal.build_principal_from_context`에 위임한다.
Layer 1(tool 조립)과 Layer 2(이 adapter)가 하나의 identity builder를 공유해 ``default_role``과
``attributes`` 의미가 일관되게 유지된다.
"""

from __future__ import annotations

from collections.abc import Iterable

from deerflow.authz.principal import build_principal_from_context
from deerflow.authz.provider import AuthorizationProvider, AuthzDecision, AuthzRequest
from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest


class GuardrailAuthorizationAdapter:
    """:class:`AuthorizationProvider`를 ``GuardrailProvider`` Protocol에 맞춘다.

    ``resource_type``과 ``action``의 기본값은 ``"tool"`` / ``"call"``이며 tool 실행 경로에
    적합하다. adapter를 tool 경로 밖에서 재사용한다면 다른 resource/action 쌍을 주입할 수 있다.

    Args:
        provider: 결정을 위임할 authorization provider.
        default_role: runtime context에 ``user_role``이 없거나 비었을 때 쓰는 role. Phase 1B
            배선이 ``AuthorizationConfig.default_role``에서 전달해야 한다.
        resource_type: 모든 ``AuthzRequest``에 쓸 resource 타입.
        action: 모든 ``AuthzRequest``에 쓸 action.
        infrastructure_tool_names: 이미 인가된 capability 집합으로 만들어진 framework 도구.
            provider 결정을 한 번 더 거치지 않고 실행될 수 있다. 호출자는 이 이름들을 정적
            설정이 아니라 현재 build의 구체적인 deferred setup에서 유도해야 한다.
    """

    name = "authorization"

    def __init__(
        self,
        provider: AuthorizationProvider,
        *,
        default_role: str = "user",
        resource_type: str = "tool",
        action: str = "call",
        infrastructure_tool_names: Iterable[str] = (),
    ) -> None:
        self._provider = provider
        self._default_role = default_role
        self._resource_type = resource_type
        self._action = action
        self._infrastructure_tool_names = frozenset(infrastructure_tool_names)

    def _infrastructure_decision(self, request: GuardrailRequest) -> GuardrailDecision | None:
        """이미 필터링된 capability 집합으로 만들어진 framework 도구를 허용한다."""
        if request.tool_name not in self._infrastructure_tool_names:
            return None
        return GuardrailDecision(
            allow=True,
            reasons=[GuardrailReason(code="authz.infrastructure_tool")],
            policy_id="authz:infrastructure",
        )

    def _to_authz(self, gr: GuardrailRequest) -> AuthzRequest:
        """guardrail request를 authorization request로 매핑한다."""
        principal = build_principal_from_context(
            {
                "user_id": gr.user_id,
                "user_role": gr.user_role,
                "oauth_provider": gr.oauth_provider,
                "oauth_id": gr.oauth_id,
                "channel_user_id": gr.channel_user_id,
                "is_internal": gr.is_internal,
                "authz_attributes": gr.authz_attributes,
            },
            default_role=self._default_role,
        )
        return AuthzRequest(
            principal=principal,
            resource=self._resource_type,
            action=self._action,
            target=gr.tool_name,
            context={
                "thread_id": gr.thread_id,
                "run_id": gr.run_id,
                "tool_call_id": gr.tool_call_id,
                "tool_input": gr.tool_input,
                "is_subagent": gr.is_subagent,
                "agent_id": gr.agent_id,
                "timestamp": gr.timestamp,
            },
        )

    @staticmethod
    def _to_guardrail(d: AuthzDecision) -> GuardrailDecision:
        """authorization 결정을 guardrail 결정으로 변환한다."""
        return GuardrailDecision(
            allow=d.allow,
            reasons=[GuardrailReason(code=r.code, message=r.message) for r in d.reasons],
            policy_id=d.policy_id,
            metadata=d.metadata,
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """동기 평가: ``provider.authorize``에 위임한다.

        provider 예외는 의도적으로 전파되게 둔다. 이 adapter를 소비하는
        :class:`~deerflow.guardrails.middleware.GuardrailMiddleware`의
        ``wrap_tool_call`` / ``awrap_tool_call``이 이미 ``fail_closed`` 파라미터
        (``AuthorizationConfig.fail_closed`` 기반)에 따라 fail-closed 의미를 적용한다. 여기서
        예외를 잡으면 그 로직이 중복되고 두 계층의 동작이 어긋날 위험이 있다.
        """
        if infrastructure_decision := self._infrastructure_decision(request):
            return infrastructure_decision
        decision = self._provider.authorize(self._to_authz(request))
        return self._to_guardrail(decision)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """비동기 평가: ``provider.aauthorize``에 위임한다.

        예외 전파 근거는 :meth:`evaluate`를 참고한다.
        """
        if infrastructure_decision := self._infrastructure_decision(request):
            return infrastructure_decision
        decision = await self._provider.aauthorize(self._to_authz(request))
        return self._to_guardrail(decision)
