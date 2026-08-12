"""anchor 테이블과 단일 composition 진입점.

DeerFlow 스택의 형태가 여기에 인코딩되어 있다. 두 가지 구조적 사실이 이를 결정한다.

* 스택은 중첩된 두 지점에서 만들어진다. `build_lead_runtime_middlewares()`가 base를
  만들고, 그다음 `build_middlewares()`가 lead 전용 middleware 약 18개를 덧붙이는데
  이들은 모두 base보다 *안쪽*이다. MODEL_PHYSICAL은 두 번째 그룹에 속하므로 extension
  주입은 최종 리스트가 완성된 뒤에 해야 하며 base 빌더 안에서 해서는 안 된다.
* 리스트의 첫 항목이 가장 바깥 wrapper다(LangChain composition 규칙).
"""

from __future__ import annotations

from collections.abc import Sequence

from deerflow_extension_api import AgentBuildContext, AgentScope, Placement

from deerflow.extensions.anchors import (
    PlacementAnchor,
    inner_of_last,
    inner_of_last_after,
    innermost,
    outer_of,
    outer_of_last,
    outermost,
)
from deerflow.extensions.injection import inject_middlewares
from deerflow.extensions.ordering import assert_ordering
from deerflow.extensions.registry import LoadedExtensions


def _anchors() -> dict[Placement, PlacementAnchor]:
    from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
    from deerflow.agents.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
    from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware
    from deerflow.agents.middlewares.terminal_response_middleware import TerminalResponseMiddleware

    return {
        # retry 루프 바깥. 아래에서 LLMErrorHandlingMiddleware가 재시도해도 논리적 결정
        # 하나가 이벤트 하나로 유지된다.
        Placement.MODEL_LOGICAL: outer_of(LLMErrorHandlingMiddleware),
        # lead agent의 모든 요청 변환보다 안쪽. innermost()가 아닌 것은 의도적이다.
        # 현재 ClarificationMiddleware가 이 지점보다 안쪽에 있고, anchor를 그 너머로
        # 옮기면 "최종 요청"의 의미 자체가 바뀐다.
        Placement.MODEL_PHYSICAL: PlacementAnchor.of(
            inner_of_last_after(
                SafetyFinishReasonMiddleware,
                after=(TerminalResponseMiddleware,),
            ),
            inner_of_last(TerminalResponseMiddleware),
            outer_of_last(ClarificationMiddleware),
            innermost(),
        ),
        Placement.TOOL_VISIBLE: outermost(),
        # 체인이 허용하는 한 tool 호출부에 가장 가깝게 둔다.
        # inner_of(ToolErrorHandlingMiddleware)가 아닌 것은 의도적이다.
        # SkillToolPolicyMiddleware와 ClarificationMiddleware가 나중에 추가되며 tool 호출도
        # 감싸기 때문에, 거기에 anchor를 두면 "raw"보다 안쪽에 wrapper가 둘 남아
        # placement가 말하는 의미를 조용히 잃는다.
        #
        # ClarificationMiddleware만 예외이며 위 MODEL_PHYSICAL과 같은 형태다. 이 middleware는
        # 마지막에 있어야 하고(Command(goto=END)로 tool 루프를 끊는다), ask_clarification만
        # 가로챈다. 실제로 실행되는 tool의 결과는 변형하지 않으므로 TOOL_RAW는 여전히
        # 원본 결과를 본다.
        Placement.TOOL_RAW: PlacementAnchor.of(
            outer_of_last(ClarificationMiddleware),
            innermost(),
        ),
        Placement.STANDARD: PlacementAnchor.of(
            outer_of(LLMErrorHandlingMiddleware),
            innermost(),
        ),
    }


class _AnchorTable(dict):
    """테이블을 지연 해석해 이 모듈 import를 가볍게 유지하고
    middleware import 순환을 피한다."""

    _loaded = False

    def _ensure(self) -> None:
        if not _AnchorTable._loaded:
            self.update(_anchors())
            _AnchorTable._loaded = True

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._ensure()
        return dict.get(self, key, default)

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def __len__(self):
        self._ensure()
        return dict.__len__(self)

    def snapshot(self) -> dict[Placement, PlacementAnchor]:
        """해석이 끝난 평범한 dict 복사본을 반환한다.

        CPython의 ``dict(subclass)`` 빠른 경로는 이 클래스의 지연 ``__iter__``나
        ``__len__`` hook을 거치지 않고 내부 저장소를 그대로 복사할 수 있다. 따라서 복사본이
        필요한 호출자는 해석을 명시적으로 강제해야 한다.
        """
        self._ensure()
        return dict(self)


PLACEMENT_ANCHORS = _AnchorTable()


def _placement_anchors_for_scope(scope: AgentScope) -> dict[Placement, PlacementAnchor]:
    if scope != AgentScope.SUBAGENT:
        return PLACEMENT_ANCHORS

    from deerflow.agents.middlewares.system_message_coalescing_middleware import SystemMessageCoalescingMiddleware

    anchors = PLACEMENT_ANCHORS.snapshot()
    anchors[Placement.MODEL_PHYSICAL] = PlacementAnchor.of(
        inner_of_last(SystemMessageCoalescingMiddleware),
        PLACEMENT_ANCHORS[Placement.MODEL_PHYSICAL],
    )
    return anchors


def compose_with_extensions(
    middlewares: Sequence[object],
    scope: AgentScope,
    ctx: AgentBuildContext | None,
    extensions: LoadedExtensions | None = None,
) -> list[object]:
    """완성된 스택에 extension 기여를 병합하고 검증한다.

    가장 바깥 빌더의 마지막에서 한 번만 호출한다. base 빌더 안에서 호출하면
    MODEL_PHYSICAL 기여가 이후 추가되는 lead 전용 middleware 약 18개보다 위에 놓인다.
    """
    from deerflow.extensions import get_agent_build_extensions, record_runtime_diagnostic

    resolved = extensions if extensions is not None else get_agent_build_extensions()

    if not resolved.has_middleware_contributors:
        assert_ordering(middlewares, {})
        return middlewares if isinstance(middlewares, list) else list(middlewares)

    if ctx is None:
        raise ValueError("AgentBuildContext is required when middleware extensions are loaded")

    result = list(middlewares)

    result, provenance, diagnostics = inject_middlewares(
        result,
        _placement_anchors_for_scope(scope),
        scope,
        ctx,
        resolved,
        isolation_diagnostic_sink=record_runtime_diagnostic,
    )
    _record_diagnostics(diagnostics)
    assert_ordering(result, provenance)
    return result


def _record_diagnostics(diagnostics) -> None:
    """스택 구성 중 발생한 진단은 생산자 쪽에서 이미 로깅된다.
    이 hook은 Gateway가 app.state에도 노출할 수 있게 하려고 존재한다."""
    from deerflow.extensions import record_runtime_diagnostics

    record_runtime_diagnostics(diagnostics)


def middleware_implements(middleware: object, hook_name: str) -> bool:
    """``middleware``가 ``hook_name``을 실제로 오버라이드하는지 판별한다.

    placement 보장은 리스트 인덱스가 아니라 hook 체인 단위다. middleware의 위치는 그것이
    참여하는 체인에서만 의미를 갖는다. 보장 테스트는 이 함수로 단순한 존재와 실제 참여를
    구분한다.
    """
    from langchain.agents.middleware import AgentMiddleware

    own = getattr(type(middleware), hook_name, None)
    base = getattr(AgentMiddleware, hook_name, None)
    return own is not None and own is not base
