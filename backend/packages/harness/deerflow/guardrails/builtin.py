"""DeerFlow에 기본 탑재된 guardrail provider."""

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest


class AllowlistProvider:
    """단순 allowlist/denylist provider. 외부 의존성이 없다."""

    name = "allowlist"

    def __init__(self, *, allowed_tools: list[str] | None = None, denied_tools: list[str] | None = None):
        # "allowlist 미설정"(None -> 전부 허용)과 "명시적 빈 allowlist"([] -> 전부 거부)를
        # 구분한다. 진리값 검사를 쓰면 []가 None으로 뭉개져 fail open이 되고, 운영자가
        # 아무것도 허용하지 않으려 했는데 모든 도구가 통과한다.
        self._allowed = set(allowed_tools) if allowed_tools is not None else None
        self._denied = set(denied_tools) if denied_tools else set()

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if self._allowed is not None and request.tool_name not in self._allowed:
            return GuardrailDecision(allow=False, reasons=[GuardrailReason(code="oap.tool_not_allowed", message=f"tool '{request.tool_name}' not in allowlist")])
        if request.tool_name in self._denied:
            return GuardrailDecision(allow=False, reasons=[GuardrailReason(code="oap.tool_not_allowed", message=f"tool '{request.tool_name}' is denied")])
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)
