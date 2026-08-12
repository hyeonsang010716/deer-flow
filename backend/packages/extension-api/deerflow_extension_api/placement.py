"""extension의 middleware가 host middleware stack에서 어디에 놓이는지 정의한다.

placement는 구조적 위치("3번 레이어에 넣어라")가 아니라 *의미적 보장*("raw tool 반환값을
관찰해야 한다")으로 선언한다. middleware는 리스트에서 하나의 index를 차지하지만, 그 index는
실제로 구현한 hook chain 위에서만 의미가 있다. 그래서 "가장 바깥"은 model 축과 tool 축에서
서로 다른 뜻이 된다. 축과 끝단으로 선언하면 이 모호함이 사라지고 host는 stack을 자유롭게
재구성할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, StrEnum, auto
from typing import Any

from deerflow_extension_api.contracts import HostPolicySnapshot


class Placement(StrEnum):
    MODEL_LOGICAL = "model_logical"
    """model 축의 바깥 끝. 보장: retry와 error handling보다 바깥.
    아래에서 host가 몇 번 retry하든 논리적 결정 한 번당 한 번만 실행된다."""

    MODEL_PHYSICAL = "model_physical"
    """model 축의 안쪽 끝. 보장: request를 변형하는 모든 middleware보다 안쪽.
    물리적인 provider 호출 한 번당 한 번 실행되며, retry 때마다 다시 진입한다."""

    TOOL_VISIBLE = "tool_visible"
    """tool 축의 바깥 끝. 보장: truncation, sanitization, error wrapping보다 바깥.
    model이 최종적으로 보는 내용을 관찰한다."""

    TOOL_RAW = "tool_raw"
    """tool 축의 안쪽 끝. 보장: 실제 callable 경계에 인접한다.
    가공 전 tool의 raw 반환값을 관찰한다."""

    STANDARD = "standard"
    """전/후처리 요구가 없다. 다른 STANDARD contributor와의 상대 순서는 보장되지 않는다."""


class AgentScope(Flag):
    LEAD = auto()
    SUBAGENT = auto()
    BOTH = LEAD | SUBAGENT


@dataclass(frozen=True)
class AgentBuildContext:
    """extension이 무엇을 기여할지 결정할 때 알 수 있는 정보."""

    scope: AgentScope
    agent_name: str | None = None
    model_name: str | None = None
    policy: HostPolicySnapshot = field(default_factory=HostPolicySnapshot)


@dataclass(frozen=True)
class MiddlewarePlacement:
    """middleware 하나와 그것이 놓여야 할 위치.

    ``middleware``는 이 모듈을 import 가볍게 유지하려고 ``AgentMiddleware`` 대신 ``Any``로
    타이핑한다. 타입 검증은 주입 시점에 host가 한다.
    """

    middleware: Any
    placement: Placement
    scope: AgentScope = AgentScope.BOTH
    order: int = 0
