"""의미적 placement를 실제 stack index로 변환한다.

DeerFlow middleware stack의 형태를 아는 유일한 module이다. stack을 재구성하면 여기 anchor
테이블을 갱신하면 되고, 무엇을 관찰해야 하는지만 선언하는 extension은 손댈 필요가 없다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_Side = Literal[
    "outer",
    "inner",
    "outer_last",
    "inner_last",
    "inner_last_after",
    "start",
    "end",
]


@dataclass(frozen=True)
class AnchorRule:
    """삽입 index를 찾는 한 번의 시도.

    ``side``가 "outer"/"inner"이면 타입이 ``types``에 속하는 첫 middleware를 기준으로 위치를
    잡고, "outer_last"/"inner_last"는 마지막으로 일치하는 middleware를 쓴다.
    "inner_last_after"는 추가로 그 일치 지점이 ``after_types``의 마지막 middleware보다 뒤에
    있어야 한다. "start"/"end"는 stack의 절대 양 끝이며 ``types``를 무시한다.
    """

    side: _Side
    types: tuple[type, ...] = ()
    after_types: tuple[type, ...] = ()

    def resolve(self, middlewares: Sequence[object]) -> int | None:
        if self.side == "start":
            return 0
        if self.side == "end":
            return len(middlewares)
        if self.side in {"outer_last", "inner_last"}:
            for index in range(len(middlewares) - 1, -1, -1):
                if isinstance(middlewares[index], self.types):
                    return index if self.side == "outer_last" else index + 1
            return None
        if self.side == "inner_last_after":
            boundary = next(
                (index for index in range(len(middlewares) - 1, -1, -1) if isinstance(middlewares[index], self.after_types)),
                None,
            )
            if boundary is None:
                return None
            for index in range(len(middlewares) - 1, boundary, -1):
                if isinstance(middlewares[index], self.types):
                    return index + 1
            return None
        for index, middleware in enumerate(middlewares):
            if isinstance(middleware, self.types):
                return index if self.side == "outer" else index + 1
        return None


@dataclass(frozen=True)
class PlacementAnchor:
    """anchor rule의 순서 있는 fallback chain."""

    chain: tuple[AnchorRule, ...]

    @classmethod
    def of(cls, *anchors: PlacementAnchor) -> PlacementAnchor:
        """여러 anchor를 하나의 fallback chain으로 이어붙인다."""
        rules: list[AnchorRule] = []
        for anchor in anchors:
            rules.extend(anchor.chain)
        return cls(tuple(rules))

    def resolve(self, middlewares: Sequence[object]) -> tuple[int, bool]:
        """(index, used_primary_rule)을 반환한다.

        첫 rule이 일치하지 않으면 ``used_primary_rule``이 False가 되고, 호출자는 이를 diagnostic으로
        보고한다. 조용히 낮아진 placement는 아무 신호 없이 extension이 관찰하는 대상을 바꾸기
        때문이다.
        """
        for position, rule in enumerate(self.chain):
            index = rule.resolve(middlewares)
            if index is not None:
                return index, position == 0
        return len(middlewares), False


def outer_of(*types: type) -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("outer", types),))


def inner_of(*types: type) -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("inner", types),))


def inner_of_last(*types: type) -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("inner_last", types),))


def inner_of_last_after(*types: type, after: tuple[type, ...]) -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("inner_last_after", types, after),))


def outer_of_last(*types: type) -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("outer_last", types),))


def outermost() -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("start"),))


def innermost() -> PlacementAnchor:
    return PlacementAnchor((AnchorRule("end"),))
