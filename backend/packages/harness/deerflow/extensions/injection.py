"""extension이 기여한 middleware를 host stack에 병합한다."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from deerflow_extension_api import AgentBuildContext, AgentScope, MiddlewarePlacement, Placement
from langchain.agents.middleware import AgentMiddleware

from deerflow.extensions.anchors import PlacementAnchor
from deerflow.extensions.isolation import IsolatedMiddleware, graph_safe_middleware_name
from deerflow.extensions.loader import Diagnostic
from deerflow.extensions.registry import LoadedExtensions

logger = logging.getLogger(__name__)


def inject_middlewares(
    middlewares: Sequence[object],
    anchors: Mapping[Placement, PlacementAnchor],
    scope: AgentScope,
    ctx: AgentBuildContext,
    extensions: LoadedExtensions,
    *,
    isolation_diagnostic_sink: Callable[[Diagnostic], None] | None = None,
) -> tuple[list[object], dict[int, str], list[Diagnostic]]:
    """기여받은 middleware를 각자의 의미적 위치에 삽입한다.

    병합된 stack, 최종 index에서 extension source로 가는 provenance map(core middleware는
    포함되지 않는다), 그리고 생성 단계 diagnostic을 반환한다. 이후 isolation 실패는
    ``isolation_diagnostic_sink``로 보내며, 생략되면 독립 호출자를 위해 반환되는 diagnostic
    리스트에 추가한다.
    """
    result = list(middlewares)
    diagnostics: list[Diagnostic] = []

    if not extensions.has_middleware_contributors:
        return result, {}, diagnostics

    collected: list[tuple[str, MiddlewarePlacement]] = []
    for source, contributor in extensions.middleware_contributors:
        try:
            contributions = tuple(contributor.contribute_middlewares(extensions.app_store, ctx) or ())
        except Exception as exc:
            message = f"contribute_middlewares() failed: {exc}"
            diagnostics.append(Diagnostic.error(source, message))
            logger.exception("Extension %s: contribute_middlewares() failed", source)
            continue
        for index, placement in enumerate(contributions):
            if not isinstance(placement, MiddlewarePlacement):
                message = f"contribution {index} must be a MiddlewarePlacement, got {type(placement).__name__}"
                diagnostics.append(Diagnostic.error(source, message))
                logger.error("Extension %s: %s", source, message)
                continue
            if not isinstance(placement.scope, AgentScope):
                message = f"contribution {index} has invalid scope {placement.scope!r}"
                diagnostics.append(Diagnostic.error(source, message))
                logger.error("Extension %s: %s", source, message)
                continue
            if not isinstance(placement.placement, Placement):
                message = f"contribution {index} has invalid placement {placement.placement!r}"
                diagnostics.append(Diagnostic.error(source, message))
                logger.error("Extension %s: %s", source, message)
                continue
            if not isinstance(placement.order, int) or isinstance(placement.order, bool):
                message = f"contribution {index} has invalid order {placement.order!r}; expected int"
                diagnostics.append(Diagnostic.error(source, message))
                logger.error("Extension %s: %s", source, message)
                continue
            if not isinstance(placement.middleware, AgentMiddleware):
                message = f"contribution {index} middleware must be an AgentMiddleware, got {type(placement.middleware).__name__}"
                diagnostics.append(Diagnostic.error(source, message))
                logger.error("Extension %s: %s", source, message)
                continue
            if not (placement.scope & scope):
                continue
            collected.append((source, placement))

    if not collected:
        return result, {}, diagnostics

    # 선언된 order로 정렬한 뒤 등록 순서로 정렬한다. dict 순회 방식과 무관하게 결과가 재현되도록
    # 하기 위함이다.
    ordered = sorted(enumerate(collected), key=lambda item: (item[1][1].order, item[0]))

    # 가장 안쪽 위치부터 삽입한다. 삽입할 때마다 그 뒤의 index가 밀리므로 뒤에서부터 작업해야 앞선
    # anchor가 유효하게 남는다.
    #
    # `priority`는 각 contribution이 `ordered`에서 갖는 위치다(이미 선언 order, 등록 순서로
    # 정렬되어 있다). 두 contribution이 *같은* 대상 index로 해석될 때 순서를 가른다. 삽입은 항상
    # 그 index의 기존 점유자를 바깥으로 밀어내므로, 우선순위가 높은(= `ordered`에서 앞선)
    # contribution이 가장 바깥에 오려면 그 index에 *마지막*으로 삽입되어야 한다. (index,
    # priority) 내림차순 정렬이 바로 그것을 만든다. 우선순위가 낮은 항목이 먼저 처리되고, 먼저
    # 삽입되며, 따라서 먼저 바깥으로 밀려난다.
    resolved: list[tuple[int, int, str, object]] = []
    for priority, (_, (source, placement)) in enumerate(ordered):
        anchor = anchors.get(placement.placement)
        if anchor is None:
            diagnostics.append(Diagnostic.error(source, f"no anchor configured for placement {placement.placement.name}"))
            continue
        index, used_primary = anchor.resolve(result)
        if not used_primary:
            message = f"placement {placement.placement.name} fell back to a secondary anchor (primary anchor middleware is absent from this stack); the observation semantics of this placement may differ from its documented guarantee"
            diagnostics.append(Diagnostic.warning(source, message))
            logger.warning("Extension %s: %s", source, message)
        resolved.append((index, priority, source, placement.middleware))

    # LangChain은 전체 stack에서 이름이 유일할 것을 요구하며, 이 이름을 trace identity로,
    # before/after hook에서는 LangGraph node ID로 사용한다.
    used_names = {getattr(middleware, "name", type(middleware).__name__) for middleware in result}
    runtime_diagnostic_sink = isolation_diagnostic_sink if isolation_diagnostic_sink is not None else diagnostics.append
    for index, priority, source, middleware in sorted(resolved, key=lambda item: (item[0], item[1]), reverse=True):
        try:
            inner_name = getattr(middleware, "name", type(middleware).__name__)
            base_name = graph_safe_middleware_name(f"extension:{source}:{inner_name}:{priority}")
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            wrapped = IsolatedMiddleware(
                middleware,
                source,
                runtime_diagnostic_sink,
                name=name,
            )
        except Exception as exc:
            message = f"middleware construction failed: {exc}"
            diagnostics.append(Diagnostic.error(source, message))
            logger.exception("Extension %s: %s", source, message)
            continue
        used_names.add(name)
        result.insert(index, wrapped)

    provenance = {index: middleware.source for index, middleware in enumerate(result) if isinstance(middleware, IsolatedMiddleware)}
    return result, provenance, diagnostics
