"""DeerFlow TUI의 순수 view-state reducer.

이 모듈은 Textual이나 렌더링에 **전혀** 의존하지 않는다. 보이는 대화를 타입이 정해진 row의
불변 리스트와 소수의 action으로 모델링하고, 순수 함수 ``reduce(state, action) -> state``
하나만 노출한다.

이 계층을 순수하게 유지하면 중요한 동작(streaming delta, tool 카드, 에러 row)을 터미널과
무관하게 평범한 ``pytest``와 소수의 합성 action만으로 테스트할 수 있다.

runtime bridge(``deerflow.tui.runtime``)가 ``DeerFlowClient``의 ``StreamEvent`` 객체를 이
action들로 변환하고, Textual app이 ``ViewState``를 widget으로 렌더링한다. 양쪽 모두 서로가
아니라 이 모듈에 의존한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from .message_format import format_tool_detail, format_tool_result, summarize_tool_title

# --------------------------------------------------------------------------- #
# Row — transcript를 구성하는 불변 단위.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UserRow:
    text: str
    kind: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantRow:
    text: str
    id: str | None = None
    error: bool = False
    kind: Literal["assistant"] = "assistant"


@dataclass(frozen=True)
class ToolRow:
    tool_call_id: str
    tool_name: str
    title: str
    detail: str = ""
    result: str = ""
    status: Literal["running", "ok", "error"] = "running"
    kind: Literal["tool"] = "tool"


@dataclass(frozen=True)
class SystemRow:
    text: str
    tone: Literal["info", "error"] = "info"
    kind: Literal["system"] = "system"


Row = UserRow | AssistantRow | ToolRow | SystemRow


# --------------------------------------------------------------------------- #
# Action — state를 바꿀 수 있는 유일한 수단.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UserSubmitted:
    text: str


@dataclass(frozen=True)
class RunStarted:
    pass


@dataclass(frozen=True)
class RunEnded:
    usage: dict | None = None


@dataclass(frozen=True)
class AssistantDelta:
    id: str
    text: str


@dataclass(frozen=True)
class AssistantError:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    tool_call_id: str
    tool_name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    tool_name: str = ""


@dataclass(frozen=True)
class SystemMessage:
    text: str
    tone: Literal["info", "error"] = "info"


@dataclass(frozen=True)
class ThreadTitle:
    title: str


@dataclass(frozen=True)
class ClearRows:
    pass


Action = UserSubmitted | RunStarted | RunEnded | AssistantDelta | AssistantError | ToolStarted | ToolResult | SystemMessage | ThreadTitle | ClearRows


# --------------------------------------------------------------------------- #
# State.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ViewState:
    rows: tuple[Row, ...] = ()
    streaming: bool = False
    usage: dict | None = None
    title: str | None = None
    # 이번 turn에 생성 중인 메시지의 id. streaming 동안 이 row만 평문으로 렌더링하고, 나머지
    # (history)는 Markdown으로 유지한다.
    streaming_id: str | None = None
    # 이번 turn에 delta를 받는 *익명*(빈 id) assistant row의 인덱스. 없으면 None. 진짜 id는
    # chunk를 가로질러 신뢰할 수 있는 key지만(`_apply_assistant_delta`의 transcript 전체 id
    # 스캔 참고), 빈 id("" — `runtime._as_str` 참고)는 모든 turn의 id 없는 chunk가 공유하므로
    # 같은 방식으로 매칭할 수 없다. `row.id == ""`로 스캔하면 새 turn의 텍스트가 마침 id가 없던
    # 이전 turn의 row로 흡수된다. 이 인덱스는 대신 "이번 turn의" 익명 row를 위치로 고정하며,
    # 매 turn 시작/종료 시 `streaming_id`와 함께 초기화된다.
    streaming_anonymous_row_index: int | None = None


def initial_state(rows: tuple[Row, ...] = ()) -> ViewState:
    return ViewState(rows=tuple(rows))


# --------------------------------------------------------------------------- #
# Reducer.
# --------------------------------------------------------------------------- #


def _append(state: ViewState, row: Row) -> ViewState:
    return replace(state, rows=state.rows + (row,))


def reduce(state: ViewState, action: Action) -> ViewState:
    """``action``을 적용한 새 ``ViewState``를 반환한다. 순수 함수다."""

    if isinstance(action, UserSubmitted):
        return _append(state, UserRow(text=action.text))

    if isinstance(action, RunStarted):
        # 새 turn: 아직 streaming 중인 메시지가 없다(client가 이전 메시지들을 먼저 다시
        # 내보내는데, 그것들을 활성 메시지로 취급하면 안 된다).
        return replace(state, streaming=True, streaming_id=None, streaming_anonymous_row_index=None)

    if isinstance(action, RunEnded):
        return replace(
            state,
            streaming=False,
            streaming_id=None,
            streaming_anonymous_row_index=None,
            usage=action.usage if action.usage is not None else state.usage,
        )

    if isinstance(action, AssistantDelta):
        return _apply_assistant_delta(state, action)

    if isinstance(action, AssistantError):
        return _append(state, AssistantRow(text=action.text, error=True))

    if isinstance(action, ToolStarted):
        return _apply_tool_started(state, action)

    if isinstance(action, ToolResult):
        return _apply_tool_result(state, action)

    if isinstance(action, SystemMessage):
        return _append(state, SystemRow(text=action.text, tone=action.tone))

    if isinstance(action, ThreadTitle):
        return replace(state, title=action.title)

    if isinstance(action, ClearRows):
        return replace(state, rows=(), streaming_id=None, streaming_anonymous_row_index=None)

    return state


def _apply_assistant_delta(state: ViewState, action: AssistantDelta) -> ViewState:
    """이 delta에 해당하는 assistant row를 갱신하거나 새로 시작한다.

    진짜(비어 있지 않은) id는 마지막 assistant row뿐 아니라 transcript 어디에서든 매칭한다.
    history가 있는 thread에서 client는 매 turn마다 이전 메시지를 전부 다시 내보내며(dedup은
    turn 단위다), 다시 보낸 *더 오래된* 메시지가 새 메시지가 시작된 뒤에 도착할 수 있다. 따라서
    마지막 row만 매칭하면 이전 답변이 중복된다.

    빈 id("" — 일부 provider/경로는 chunk별 id를 아예 붙이지 않는다, ``runtime._as_str`` 참고)는
    같은 스캔의 key로 신뢰할 수 없다. 진짜 id와 달리 *모든* turn의 id 없는 chunk가 공유하므로,
    transcript 전체에서 `row.id == ""`를 매칭하면 새 turn의 텍스트가 마침 id가 없던 이전 turn의
    row로 흡수되어 새 turn의 답변이 오래된 row 속으로 조용히 사라진다. 그래서 빈 id delta는
    `_apply_assistant_delta_anonymous`로 보내고, 거기서는 "이번 turn의" row를 id가 아니라
    위치로 추적한다.

    갱신은 무작정 이어붙이지 않고 내용 기준으로 병합해, 전체 재전송 / 누적 snapshot과 진짜
    증분 delta를 모두 흡수한다.

    * 새 텍스트 == 누적본이거나 누적본으로 시작 -> 누적/재전송: 교체
    * 누적본이 새 텍스트로 시작                 -> 오래되거나 짧은 재전송: 유지
    * 그 외                                     -> 진짜 delta: 이어붙임
    """
    if not action.id:
        return _apply_assistant_delta_anonymous(state, action)

    rows = list(state.rows)
    for i, row in enumerate(rows):
        # ``not row.error``: 에러 row는 id 없이 추가되므로 어차피 여기 매칭되지 않는다. 이
        # 가드는 나중에 에러 row에 id가 생기더라도 병합 대상이 되지 않게 하는 이중 안전장치다.
        if isinstance(row, AssistantRow) and row.id == action.id and not row.error:
            # 동일한 전체 텍스트의 정확한 재전송(예: 재접속 후 history를 다시 내보내는 values
            # snapshot)은 no-op이다. 버퍼와 우연히 같아진 한 글자 delta(CJK 첩어)가 no-op으로
            # 오인되지 않도록, 여러 글자일 때만 재전송으로 취급한다.
            if row.text == action.text and len(action.text) > 1:
                return state
            merged = _merge_stream_text(row.text, action.text)
            rows[i] = replace(row, text=merged)
            return _mark_streaming(replace(state, rows=tuple(rows)), action.id)
    return _mark_streaming(_append(state, AssistantRow(text=action.text, id=action.id)), action.id)


def _apply_assistant_delta_anonymous(state: ViewState, action: AssistantDelta) -> ViewState:
    """id가 빈 ``AssistantDelta``를 처리한다(`_apply_assistant_delta` 참고).

    한 turn에서 id 없는 chunk가 여러 개 오는 것은 정상이다. chunk별 id를 붙이지 않는 provider도
    ``"Hel"`` 다음 ``"lo"``처럼 토큰 단위로 스트리밍한다. 그래서 turn의 첫 빈 id delta가 새 row를
    시작하고, 이후 빈 id delta들은 그 row에 계속 이어붙인다(``RunStarted``/``RunEnded``/
    ``ClearRows``마다 초기화되는 ``state.streaming_anonymous_row_index``로 추적한다). id로
    추적하지 않으므로 다음 turn은 이전 turn에 남은 id 없는 row에 매칭되지 않고 항상 자기 row를
    새로 시작한다. 이 분리는 바로 그 버그를 피하려고 존재한다.

    추적 중인 row는 transcript의 마지막 row일 때만 재사용한다. 진짜 id는 tool 왕복을 거치면
    자연히 바뀌므로(LangGraph가 tool 이후 이어지는 응답에 새 AIMessage id를 준다), id 기반
    경로에서는 중간에 낀 ``ToolStarted``/``ToolResult``가 이미 새 row를 시작한다
    (`test_assistant_delta_with_new_id_after_tool_creates_separate_row` 참고). 빈 id에는 그런
    자연스러운 신호가 없고 tool call 전후 모두 항상 ``""``이므로, 이 함수는 row의 *위치*를 그
    대체 수단으로 쓴다. 다른 무언가(실제로는 tool 카드)가 추가되는 순간 익명 row는 더 이상
    마지막이 아니고, 다음 빈 id delta는 tool 카드를 넘어 오래된 텍스트로 거슬러 가지 않고 새
    row를 시작한다.
    """
    index = state.streaming_anonymous_row_index
    if index is not None and index == len(state.rows) - 1:
        row = state.rows[index]
        if isinstance(row, AssistantRow) and not row.error:
            # 위의 id 기반 경로와 동일한 no-op / 병합 규칙을 쓴다.
            if row.text == action.text and len(action.text) > 1:
                return state
            rows = list(state.rows)
            merged = _merge_stream_text(row.text, action.text)
            rows[index] = replace(row, text=merged)
            return _mark_streaming_anonymous(replace(state, rows=tuple(rows)), index)

    new_state = _append(state, AssistantRow(text=action.text, id=action.id))
    return _mark_streaming_anonymous(new_state, len(new_state.rows) - 1)


def _mark_streaming(state: ViewState, message_id: str) -> ViewState:
    """streaming 중인 메시지 id를 기록한다(run이 활성일 때만)."""
    if state.streaming:
        return replace(state, streaming_id=message_id)
    return state


def _mark_streaming_anonymous(state: ViewState, index: int) -> ViewState:
    """활성 turn의 익명 row 인덱스를 기록한다(run이 활성일 때만).

    ``streaming_id``는 의도적으로 ``""``가 아니라 ``None``으로 둔다. 진짜 id와 달리 ``""``는
    모든 turn의 모든 익명 row가 공유하므로, 이를 렌더 계층의 "이 row가 streaming 중인가" key
    (``render.render_transcript``)로 쓰면 새 row가 시작되는 순간 과거의 익명 row까지 전부
    streaming 중으로 표시된다. 대가는 순전히 표시상의 문제뿐이다. 익명 row는 다른 row가 받는
    streaming 중 평문 처리 혜택을 받지 못하지만 영향은 id 없는 fallback 경로에 한정되며, 대신 이
    수정이 ``rows``에서 제거한 turn 간 모호성을 렌더 계층에 다시 들이지 않는다.
    """
    if state.streaming:
        return replace(state, streaming_id=None, streaming_anonymous_row_index=index)
    return state


def _merge_stream_text(existing: str, incoming: str) -> str:
    if not existing:
        return incoming
    # 누적 재전달: incoming이 existing을 엄격히 확장한다.
    if len(incoming) > len(existing) and incoming.startswith(existing):
        return incoming
    # 오래되거나 짧은 재전송: existing이 이미 incoming을 prefix로 포함한다(예: delta로 이미
    # 누적된 history를 다시 내보내는 values snapshot). 엄격히 짧을 때만 오래된 것으로 본다.
    if len(existing) > len(incoming) and existing.startswith(incoming):
        return existing
    return existing + incoming  # 진짜 증분 delta


def _apply_tool_started(state: ViewState, action: ToolStarted) -> ViewState:
    """``tool_call_id``로 중복을 제거하면서 tool 카드를 만들거나 갱신한다.

    streaming tool call은 하나의 call id에 대해 여러 chunk로 도착하고(이름이 먼저, 그다음
    인자가 점점 늘어난다), client가 values snapshot으로 그 호출을 다시 내보낼 수도 있다. id가
    없는 chunk는 인자 조각 노이즈이므로 버린다.
    """
    if not action.tool_call_id:
        return state

    rows = list(state.rows)
    for i, row in enumerate(rows):
        if isinstance(row, ToolRow) and row.tool_call_id == action.tool_call_id:
            name = action.tool_name or row.tool_name
            detail = format_tool_detail(name, action.args) or row.detail
            rows[i] = replace(row, tool_name=name, title=summarize_tool_title(name), detail=detail)
            return replace(state, rows=tuple(rows))

    return _append(
        state,
        ToolRow(
            tool_call_id=action.tool_call_id,
            tool_name=action.tool_name,
            title=summarize_tool_title(action.tool_name),
            detail=format_tool_detail(action.tool_name, action.args),
            status="running",
        ),
    )


def _apply_tool_result(state: ViewState, action: ToolResult) -> ViewState:
    if not action.tool_call_id:
        return state

    rows = list(state.rows)
    for i, row in enumerate(rows):
        if isinstance(row, ToolRow) and row.tool_call_id == action.tool_call_id:
            rows[i] = replace(
                row,
                status="error" if action.is_error else "ok",
                result=format_tool_result(action.content),
            )
            return replace(state, rows=tuple(rows))

    # 매칭되는 tool 카드가 없으면(started chunk를 놓친 경우) 그래도 결과를 노출한다.
    return _append(
        state,
        ToolRow(
            tool_call_id=action.tool_call_id,
            tool_name=action.tool_name,
            title=summarize_tool_title(action.tool_name),
            status="error" if action.is_error else "ok",
            result=format_tool_result(action.content),
        ),
    )
