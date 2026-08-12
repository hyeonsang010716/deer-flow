"""middleware 스택의 순서 불변식을 선언적으로 정의한다.

손으로 쓰던 인덱스 비교를 대체한다. extension이 기여한 middleware도 검증 전에
합쳐지므로 어떤 기여도 불변식을 우회할 수 없고, 실패 메시지가 원인 extension을 지목한다.

깨진 불변식은 이 시스템에서 유일하게 치명적인 실패다. 관측이 누락되는 경우와 달리
에러 없이 잘못된 동작을 만들어낸다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from deerflow.extensions.isolation import IsolatedMiddleware


@dataclass(frozen=True)
class OrderingConstraint:
    outer: type
    inner: type
    reason: str


def _indices_of(middlewares: Sequence[object], target: type) -> list[int]:
    indices: list[int] = []
    for index, middleware in enumerate(middlewares):
        candidate = middleware.inner if isinstance(middleware, IsolatedMiddleware) else middleware
        if isinstance(candidate, target):
            indices.append(index)
    return indices


def assert_ordering(
    middlewares: Sequence[object],
    provenance: Mapping[int, str],
    constraints: Sequence[OrderingConstraint] | None = None,
) -> None:
    """제약이 깨지면 예외를 던진다. 양쪽 모두 없으면 아무것도 하지 않는다."""
    for constraint in constraints if constraints is not None else core_ordering_constraints():
        outer_indices = _indices_of(middlewares, constraint.outer)
        inner_indices = _indices_of(middlewares, constraint.inner)
        if not outer_indices or not inner_indices:
            continue
        if max(outer_indices) < min(inner_indices):
            continue
        violating_indices = [index for index in outer_indices if index >= min(inner_indices)] + [index for index in inner_indices if index <= max(outer_indices)]
        culprits = sorted({source for index in violating_indices if (source := provenance.get(index)) is not None})
        blame = ", ".join(culprits) if culprits else "core middleware order"
        raise RuntimeError(
            f"Middleware ordering constraint violated: {constraint.outer.__name__} must be outer "
            f"(lower index) of every {constraint.inner.__name__}, but found outer indices "
            f"{outer_indices} vs inner indices {inner_indices}. Reason: {constraint.reason}. "
            f"Contributed by: {blame}."
        )


@cache
def core_ordering_constraints() -> tuple[OrderingConstraint, ...]:
    """host의 순서 불변식을 첫 사용 시점에 해석한다.

    지연 해석은 의도된 것이며, 단순한 순환 참조가 아니라 의존 *방향*의 문제다.
    ``extensions/``는 middleware 계층이 호출해 들어오는 계층이므로 여기서
    ``agents.middlewares``를 모듈 스코프에서 import하면 의존 방향이 거꾸로 향하고,
    어떤 middleware든 ``extensions/`` 아래를 모듈 레벨에서 import하는 순간 순환이 닫힌다.
    대신 ``assert_ordering`` 시점에 해석하는데, 이미 middleware 빌더 안에서 실행되므로
    한 계층 내부의 전방 참조가 된다.

    반환값은 평범한 tuple이다. 이전 구현은 ``__iter__``만 오버라이드한 ``tuple``
    서브클래스로 지연을 흉내 냈다. tuple은 생성 후 자기 저장소를 채울 수 없으므로
    그 저장소를 읽는 모든 연산(``len``, ``bool``, ``in``, 인덱싱, 슬라이싱,
    ``reversed``, ``==``)이 빈 시퀀스를 보고했고 순회만 실제 제약을 내놓았다.
    값을 위조하는 대신 호출 자체를 지연하면 답이 하나로 유지된다.
    """
    from deerflow.agents.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
    from deerflow.agents.middlewares.tool_progress_middleware import ToolProgressMiddleware

    return (
        OrderingConstraint(
            outer=ToolProgressMiddleware,
            inner=ToolErrorHandlingMiddleware,
            reason=("ToolProgressMiddleware reads deerflow_tool_meta in _update_state_from_result, so its wrap_tool_call chain must enclose the ToolErrorHandlingMiddleware step that stamps it"),
        ),
    )
