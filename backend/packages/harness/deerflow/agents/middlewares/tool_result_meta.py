"""구조화된 신호 생산을 위한 통합 도구 결과 의미 체계.

ToolErrorHandlingMiddleware를 거치는 모든 도구 결과는 additional_kwargs에
``deerflow_tool_meta`` 항목을 얻는다. 하위 소비자(ToolProgressMiddleware 등)는 텍스트를
파싱하는 대신 이 키를 읽는다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.types import Command

TOOL_META_KEY = "deerflow_tool_meta"

_ERROR_PREFIX = "Error:"
_PARTIAL_MARKERS = (
    "partial results",
    "limited results",
    "truncated",
    "results may be incomplete",
    # status="error" 대신 status="success"에 결과 없음 본문을 담아 돌려주는 도구도 정체 감지에
    # 걸려야 모델이 다른 쿼리를 시도하도록 유도된다.
    "no results found",
    "no content found",
    "no images found",
)


@dataclass(frozen=True, slots=True)
class ToolResultMeta:
    status: Literal["success", "error", "partial_success"]
    error_type: str | None
    recoverable_by_model: bool
    recommended_next_action: Literal["continue", "rewrite_query", "try_alternative", "summarize", "stop"]
    source: Literal["exception", "tool_return", "content_analysis", "progress_middleware"]


_ERROR_RULES: list[tuple[list[str], dict[str, object]]] = [
    (
        ["401", "403", "unauthorized", "authentication", "invalid api key"],
        {"error_type": "auth", "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
    (
        ["rate limit", "rate limited", "rate_limit"],
        {"error_type": "rate_limited", "recoverable_by_model": False, "recommended_next_action": "summarize"},
    ),
    (
        ["timeout", "timed out", "connection", "network error", "temporarily unavailable"],
        {"error_type": "transient", "recoverable_by_model": False, "recommended_next_action": "try_alternative"},
    ),
    (
        ["not configured", "not installed", "missing required", "disabled", "no api key"],
        {"error_type": "config", "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
    (
        ["permission denied", "access denied", "path traversal", "forbidden"],
        {"error_type": "permission", "recoverable_by_model": True, "recommended_next_action": "try_alternative"},
    ),
    (
        ["no results found", "no content found", "no images found", "no results"],
        {"error_type": "no_results", "recoverable_by_model": True, "recommended_next_action": "rewrite_query"},
    ),
    (
        ["not found", "no such file", "does not exist", "404"],
        {"error_type": "not_found", "recoverable_by_model": True, "recommended_next_action": "rewrite_query"},
    ),
    (
        ["unexpected error", "internal error", "500"],
        {"error_type": "internal", "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
]

_UNKNOWN_ERROR: dict[str, object] = {
    "error_type": "unknown",
    "recoverable_by_model": True,
    "recommended_next_action": "try_alternative",
}

# 결과 내용이 도구 자신의 메시지가 아니라 *렌더링된 원격 페이지*인 도구 이름들.
# ToolResultSanitizationMiddleware의 _REMOTE_CONTENT_TOOL_NAMES와 마찬가지로 이름 기반이다.
# 자체 fetch provider는 모두 ``web_fetch``로 정규화되므로(community/*/tools.py 참고) 이
# gate는 provider와 무관하게 동작한다. normalize_tool_message()가 *모든* 도구에 대해 돌기
# 때문에 gate가 필요하다. 짧은 "not found" 줄은 많은 도구에서 정상 출력이다.
# ``web_capture``는 의도적으로 빠져 있다. 그 결과는 렌더링된 페이지가 아니라 artifact에 대한
# 도구 메시지("Captured screenshot: <path> (warning: ...)")라 title 규칙을 적용할 수 없다.
# 죽은 대상을 캡처해도 artifact와 모델이 볼 수 있는 경고는 나오며, 그 표시는 provider 경계의
# 몫이다(#4239).
_PAGE_CONTENT_TOOL_NAMES: frozenset[str] = frozenset({"web_fetch"})

# error-shell 경로가 재사용하는 카테고리 속성. _ERROR_RULES가 이미 선언한 error_type으로
# 색인한다. 복제가 아니라 파생이라, shell이 자기 카테고리의 recoverable/next-action 계약과
# 어긋날 수 없다.
_ATTRS_BY_ERROR_TYPE: dict[str, dict[str, object]] = {str(attrs["error_type"]): attrs for _keywords, attrs in _ERROR_RULES}

# reason 문구(RFC 9110 §15 및 실제 서버가 쓰는 표현)를 _ERROR_RULES가 이미 가진 error_type에
# 매핑한다. fetch가 렌더링된 페이지로 실제 마주치는 status로 한정한다. 5xx 분할은
# _ERROR_RULES와 같다. 500/501은 "500"/"internal error" 키워드 쪽(internal → stop),
# 502/503/504는 "timeout"/"temporarily unavailable" 키워드 쪽(transient → try_alternative)이다.
# gateway 오류는 다른 출처를 시도해야 하는 경우이고, 같은 단어가 _classify_error_text를 거칠
# 때와 여기서 다르게 분류되면 안 된다.
_ERROR_SHELL_PHRASES: dict[str, str] = {
    "unauthorized": "auth",
    "proxy authentication required": "auth",
    "forbidden": "permission",
    "access denied": "permission",
    "permission denied": "permission",
    "not found": "not_found",
    "too many requests": "rate_limited",
    "internal server error": "internal",
    "not implemented": "internal",
    "bad gateway": "transient",
    "service unavailable": "transient",
    "service temporarily unavailable": "transient",
    "gateway timeout": "transient",
}

# 서버가 reason 문구 앞에 붙일 수 있는 일반 명사들("Page not found", IIS의
# "404 - File or directory not found."). *앞쪽*에서만 제거하므로 문구 뒤에 남는 단어가
# 하나라도 있으면 그 title은 거른다.
_STATUS_TITLE_FILLER: frozenset[str] = frozenset({"http", "error", "page", "the", "file", "or", "directory", "url", "resource"})


# 모듈 로드 시 _ERROR_RULES에서 미리 컴파일한다. 숫자 코드(401, 403, 404, 500)를 단어 경계에
# 고정해 "took 500ms" 같은 무관한 숫자에 부분 일치하지 않게 한다. _ERROR_RULES 뒤에서 계산해
# 이 집합이 유일한 기준이자 thread-safe가 되게 하고, 분류 hot path에서 지연 쓰기를 없앤다.
_NUMERIC_KW_RE: dict[str, re.Pattern[str]] = {kw: re.compile(rf"\b{kw}\b") for rule_keywords, _ in _ERROR_RULES for kw in rule_keywords if kw.isdigit()}

_SEMANTIC_ZERO_ERROR_STRINGS: frozenset[str] = frozenset({"none", "null", "false", "no", "ok", "success", "n/a", ""})


def _extract_json_error_text(content: str) -> str | None:
    """{"error": "...", "query": "..."} 같은 JSON 포장 오류에서 error 문자열을 반환한다.

    ``error`` 필드가 falsy(JSON null / 0 / false / 빈 문자열)이거나 관례상 "오류 없음"을
    뜻하는 sentinel 문자열(예: ``"none"``, ``"null"``, ``"false"``)이면 None을 반환한다.
    성공 시 ``{"error": "none", "results": [...]}``를 돌려주는 도구가 오류로 잘못 분류되는
    것을 막는다.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    error = data.get("error") if isinstance(data, dict) else None
    if not error:
        return None
    if isinstance(error, str) and error.lower().strip() in _SEMANTIC_ZERO_ERROR_STRINGS:
        return None
    # 문자열이 아닌 값은 JSON으로 직렬화해 _classify_error_text가 예측 가능한 형태를 보게 한다
    # (예: {"error": 404} → "404", {"error": [...]} → "[...]"). Python repr을 쓰면
    # "missing required" 같은 키워드 규칙에 잘못 걸릴 수 있다.
    return error if isinstance(error, str) else json.dumps(error)


def _match_keyword(kw: str, lower: str) -> bool:
    """소문자 텍스트에 키워드를 매칭한다. 숫자 코드는 단어 경계를 적용한다."""
    if kw.isdigit():
        return bool(_NUMERIC_KW_RE[kw].search(lower))
    return kw in lower


def _classify_error_text(text: str) -> dict[str, object]:
    lower = text.lower()
    for keywords, attrs in _ERROR_RULES:
        if any(_match_keyword(kw, lower) for kw in keywords):
            return {**attrs}
    return {**_UNKNOWN_ERROR}


def _classify_error_shell(msg: ToolMessage, content: str) -> dict[str, object] | None:
    """가져온 페이지가 HTTP 오류 페이지일 때 카테고리 속성을 반환한다.

    없는 URL을 fetch해도 transport 계층에서는 성공하므로 위의 어떤 분기에도 걸리지 않고,
    서버의 오류 페이지가 ``status="success"``로 표시된 채 모델에 도달한다. 담고 있지도 않은
    근거로 집계되는 셈이다(issue #4273).

    신호는 *추출된 title*이다. nginx / Apache / IIS / Cloudflare의 오류 페이지는 모두 status
    줄을 제목으로 렌더링하고 본문에는 서버 상용구만 담는다("# 404 Not Found" 아래
    "nginx/1.24.0"). 매칭은 정규화 후 부분 일치가 아니라 완전 일치이므로, 단지 어떤 status에
    *관한* 문서는 title의 다른 단어가 남아 걸러진다. "404 Ways to Cook Rice"와
    "Not Found: a short history of the 404"는 모두 success로 살아남는다.

    콘텐츠 길이는 의도적으로 쓰지 않는다. 실제 오류 페이지로 측정해 보면 구분되지 않는다
    (IIS 404는 193자, 진짜 기사는 202자로 렌더링됐다).

    이것은 provider와 무관한 fallback일 뿐이다. 기준이 되는 신호는 provider 자신의 status
    code이고 그것은 web_fetch 경계에 남는다(Browserless는 #4239에 따라 ``X-Response-Code``를
    노출한다). title이 status 줄이 아닌 페이지는 그 계층의 몫이다.
    """
    if msg.name not in _PAGE_CONTENT_TOOL_NAMES:
        return None
    title = next((line for line in content.splitlines() if line.strip()), "")
    phrase = _as_status_line(title.lstrip("#").strip())
    error_type = _ERROR_SHELL_PHRASES.get(phrase) if phrase else None
    return {**_ATTRS_BY_ERROR_TYPE[error_type]} if error_type else None


def _as_status_line(title: str) -> str | None:
    """페이지 title을 순수 reason 문구로 줄인다. 내용이 남아 있으면 None을 반환한다.

    "404 Not Found" -> "not found"; "404 - File or directory not found." -> "not found";
    "404 Ways to Cook Rice" -> "ways to cook rice"(단어가 남으므로 문서다).
    """
    words = re.sub(r"[^0-9a-z]+", " ", title.lower()).split()
    # 앞의 status code와 일반 명사를 순서와 무관하게 제거한다. 서버는
    # "404 - File or directory not found"와 "HTTP Error 404 - Not Found"를 모두 쓴다.
    while words and (words[0] in _STATUS_TITLE_FILLER or (len(words[0]) == 3 and words[0].isdigit() and 400 <= int(words[0]) <= 599)):
        words = words[1:]
    return " ".join(words) or None


def _make_meta(*, status: str, source: str, error_type: str | None = None, recoverable_by_model: bool = True, recommended_next_action: str = "continue") -> dict[str, object]:
    return {
        "status": status,
        "error_type": error_type,
        "recoverable_by_model": recoverable_by_model,
        "recommended_next_action": recommended_next_action,
        "source": source,
    }


def stamp_exception_meta(msg: ToolMessage, exc_info: str) -> ToolMessage:
    """예외에서 파생된 ToolMessage에 source='exception'인 deerflow_tool_meta를 찍는다.

    기존 표시를 보존하는 normalize_tool_message와 달리, 이 함수는 이미 있는 TOOL_META_KEY
    항목을 항상 덮어쓴다. 예외 기반 분류가 도구 자신의 반환 시점 표시보다 우선한다.
    """
    attrs = _classify_error_text(exc_info)
    updated_kwargs = dict(msg.additional_kwargs or {})
    updated_kwargs[TOOL_META_KEY] = _make_meta(status="error", source="exception", **attrs)
    msg.additional_kwargs = updated_kwargs
    return msg


def normalize_tool_message(msg: ToolMessage) -> ToolMessage:
    """아직 없으면 ToolMessage에 deerflow_tool_meta를 붙인다."""
    existing = (msg.additional_kwargs or {}).get(TOOL_META_KEY)
    if existing is not None:
        return msg

    content = msg.content if isinstance(msg.content, str) else ""
    # 한 번만 계산해 아래 partial-success 마커 검사에서 재사용한다. 그러지 않으면 generator
    # 안에서 _PARTIAL_MARKERS 항목마다 content.lower()를 호출하게 된다.
    content_lower = content.lower()

    # 비표준 오류: 도구가 "Error:" 접두사 관례 없이 status="error"를 반환한 경우다.
    # (ToolErrorHandlingMiddleware의 실제 예외는 stamp_exception_meta가 미리 표시하고 위에서
    # 조기 반환하므로 이 분기에 오지 않는다.)
    # 먼저 JSON 추출을 시도해 다른 JSON 필드(예: "query")에 우연히 들어간 키워드가 아니라
    # "error" 필드 값만으로 분류하게 한다.
    if msg.status == "error" and not content.startswith(_ERROR_PREFIX):
        json_error = _extract_json_error_text(content)
        if json_error is not None:
            attrs = _classify_error_text(json_error)
        else:
            # content가 단지 'error' 키가 없는 JSON 객체인지 판단한다. 그렇다면 raw JSON
            # 문자열로 분류하면 안 된다. 부수적인 필드 값(예: {"user_id": 401})이 키워드
            # 규칙에 잘못 걸려 도구를 차단할 수 있다. 유효한 JSON이 아닐 때만 원문 텍스트로
            # 분류한다.
            try:
                is_json_dict = isinstance(json.loads(content), dict)
            except (json.JSONDecodeError, ValueError):
                is_json_dict = False
            attrs = {**_UNKNOWN_ERROR} if is_json_dict else _classify_error_text(content)
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif content.startswith(_ERROR_PREFIX):
        attrs = _classify_error_text(content[len(_ERROR_PREFIX) :])
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif (json_error := _extract_json_error_text(content)) is not None:
        attrs = _classify_error_text(json_error)
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif (shell_attrs := _classify_error_shell(msg, content)) is not None:
        meta = _make_meta(status="error", source="content_analysis", **shell_attrs)
    elif any(m in content_lower for m in _PARTIAL_MARKERS):
        meta = _make_meta(
            status="partial_success",
            source="content_analysis",
            recommended_next_action="rewrite_query",
        )
    else:
        meta = _make_meta(status="success", source="content_analysis")

    updated_kwargs = dict(msg.additional_kwargs or {})
    updated_kwargs[TOOL_META_KEY] = meta
    msg.additional_kwargs = updated_kwargs
    return msg


def normalize_tool_result(result: ToolMessage | Command) -> ToolMessage | Command:
    """도구 결과를 정규화한다. Command wrapper도 그대로 처리한다."""
    if isinstance(result, ToolMessage):
        return normalize_tool_message(result)
    return result
