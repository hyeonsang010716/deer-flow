"""도구 호출 직전 인가를 위한 GuardrailProvider protocol과 데이터 구조."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GuardrailRequest:
    """도구 호출마다 provider에게 전달되는 context."""

    tool_name: str
    tool_input: dict[str, Any]
    agent_id: str | None = None
    thread_id: str | None = None
    is_subagent: bool = False
    timestamp: str = ""
    user_id: str | None = None
    user_role: str | None = None
    oauth_provider: str | None = None
    oauth_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    # 인가 신원 필드. GuardrailMiddleware가 runtime context에서 채운다.
    # 기본값이 있어 이 필드를 읽지 않는 provider와도 호환된다.
    channel_user_id: str | None = None
    is_internal: bool = False
    authz_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailReason:
    """허용/거부 결정의 구조화된 사유(OAP reason object)."""

    code: str
    message: str = ""


@dataclass
class GuardrailDecision:
    """provider의 허용/거부 판정(OAP Decision object에 맞춘다)."""

    allow: bool
    reasons: list[GuardrailReason] = field(default_factory=list)
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GuardrailProvider(Protocol):
    """교체 가능한 도구 호출 인가 계약.

    이 메서드들만 있으면 어떤 클래스든 동작한다. base class는 필요 없다.
    provider는 resolve_variable()로 클래스 경로를 통해 로드되며, DeerFlow가 model,
    tool, sandbox에 쓰는 것과 같은 방식이다.
    """

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """도구 호출을 진행해도 되는지 판단한다."""
        ...

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """비동기 버전."""
        ...
