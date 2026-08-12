"""request trace context 헬퍼.

여기 저장되는 값은 DeerFlow의 request 단위 상관관계 id다. Langfuse 자체 trace id나
DeerFlow run id와는 별개다.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Final

TRACE_ID_HEADER: Final[str] = "X-Trace-Id"
DEERFLOW_TRACE_METADATA_KEY: Final[str] = "deerflow_trace_id"
_MAX_TRACE_ID_LENGTH: Final[int] = 512

_current_trace_id: Final[ContextVar[str | None]] = ContextVar("deerflow_current_trace_id", default=None)
_trace_id_from_request_header: Final[ContextVar[bool]] = ContextVar(
    "deerflow_trace_id_from_request_header",
    default=False,
)


def generate_trace_id() -> str:
    """header에 안전한 새 trace id를 반환한다."""
    return uuid.uuid4().hex


def normalize_trace_id(value: object) -> str | None:
    """안전한 trace id 문자열을 반환하고, *value*를 쓸 수 없으면 ``None``을 반환한다.

    출력 가능한 ASCII(0x20-0x7E)만 허용한다. trace id는 HTTP 응답 header를 왕복하고
    Starlette은 이를 latin-1로 인코딩하기 때문에 0x7E 초과 코드포인트는 거부한다.
    0xFF 초과 코드포인트는 ``MutableHeaders.__setitem__`` 안에서 ``UnicodeEncodeError``를
    일으켜 응답 본문을 보내기도 전에 500을 만들고, C1 제어문자(0x80-0x9F)는 인코딩 자체는
    되지만 보안이 강화된 중계 서버(nginx / envoy / cloudfront)가 제거하거나 거부해서 응답을
    조용히 망가뜨린다. C0 제어문자(< 0x20)와 DEL(0x7F)도 같은 header 안전성 이유에 더해
    log injection 방어를 위해 거부한다.
    """
    if not isinstance(value, str):
        return None
    trace_id = value.strip()
    if not trace_id or len(trace_id) > _MAX_TRACE_ID_LENGTH:
        return None
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in trace_id):
        return None
    return trace_id


def set_current_trace_id(trace_id: str) -> Token[str | None]:
    """*trace_id*를 현재 실행 컨텍스트에 바인딩한다."""
    normalized = normalize_trace_id(trace_id)
    if normalized is None:
        normalized = generate_trace_id()
    return _current_trace_id.set(normalized)


def reset_current_trace_id(token: Token[str | None]) -> None:
    """*token*이 캡처한 trace context를 복원한다."""
    _current_trace_id.reset(token)


def get_current_trace_id() -> str | None:
    """바인딩되어 있다면 현재 request trace id를 반환한다."""
    return _current_trace_id.get()


def mark_trace_id_from_request_header(*, from_header: bool) -> Token[bool]:
    """현재 trace id가 유효한 인바운드 header에서 왔는지 기록한다."""
    return _trace_id_from_request_header.set(from_header)


def reset_trace_id_from_request_header(token: Token[bool]) -> None:
    """*token*이 캡처한 인바운드 header 플래그를 복원한다."""
    _trace_id_from_request_header.reset(token)


def is_trace_id_from_request_header() -> bool:
    """유효한 ``X-Trace-Id`` header가 request를 바인딩했으면 ``True``를 반환한다."""
    return _trace_id_from_request_header.get()


def resolve_deerflow_trace_id(metadata_trace_id: object) -> str | None:
    """run에 적용될 실제 ``deerflow_trace_id``를 결정한다.

    Gateway ``TraceMiddleware``가 유효한 인바운드 ``X-Trace-Id``를 바인딩했다면 그 값이
    ``config.metadata.deerflow_trace_id``보다 우선해서, 로그·응답 header·Langfuse·runtime
    context가 서로 일치한다. 그렇지 않으면 호출자 metadata가 우선하고, 그다음이 주변
    request trace context다.
    """
    if is_trace_id_from_request_header():
        return get_current_trace_id()
    return normalize_trace_id(metadata_trace_id) or get_current_trace_id()


@contextmanager
def request_trace_context(trace_id: str | None = None) -> Iterator[str]:
    """request 또는 진입점이 지속되는 동안 request trace id를 바인딩한다."""
    normalized = normalize_trace_id(trace_id) or generate_trace_id()
    token = _current_trace_id.set(normalized)
    try:
        yield normalized
    finally:
        _current_trace_id.reset(token)


@contextmanager
def ensure_trace_context(trace_id: str | None = None) -> Iterator[str]:
    """*trace_id*를 바인딩하거나, 현재 trace를 물려받거나, 새로 만든다."""
    normalized = normalize_trace_id(trace_id) or get_current_trace_id() or generate_trace_id()
    token = _current_trace_id.set(normalized)
    try:
        yield normalized
    finally:
        _current_trace_id.reset(token)
