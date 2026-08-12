"""Tool search — 런타임의 deferred tool 탐색.

구성:
- DeferredToolCatalog: deferred tool의 불변·검색 가능 catalog.
- build_tool_search_tool: catalog를 감싼 closure로 `tool_search` tool을 만든다. promotion을
  ``Command``로 graph state에 기록한다.
- build_deferred_tool_setup: 이번 agent 빌드에 설정된 tool들로 catalog와 tool을 조립한다.
- build_mcp_routing_middleware: 호출자가 쓸 수 있는 deferred tool의 직렬화된 routing
  metadata로 PR2 auto-promote middleware를 만든다.

agent는 <available-deferred-tools>에서 deferred tool 이름을 보지만, tool_search tool로 전체
schema를 가져오기 전까지는 호출할 수 없다. deferred 집합은 빌드 시점 closure에 실리고 promotion은
thread별 graph state에 있다. ContextVar는 쓰지 않는다. source에 무관하게, ``deerflow_mcp``
metadata 태그를 달고 있으면 "deferred" tool이다.
"""

import hashlib
import html
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.types import Command

from deerflow.tools.mcp_metadata import get_mcp_routing, is_mcp_tool

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)

MAX_RESULTS = 5  # 검색 한 번에 반환하는 최대 tool 수


def _compile_catalog_regex(pattern: str) -> re.Pattern[str]:
    """``pattern``을 대소문자 무시로 컴파일하고, 실패하면 리터럴 매칭으로 대체한다.

    검색 query는 모델이 만들므로, 잘못된 regex(예: 괄호 불일치)는 예외를 던지지 않고 리터럴
    substring 매칭으로 낮춰야 한다.
    """
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


# ── Catalog ──


# NOTE: slots=True 없이 frozen=True를 쓰면 __dict__가 유지되고, 그래야 아래 @cached_property
# 필드가 캐싱된다(frozen __setattr__를 우회해 instance.__dict__에 기록한다). slots=True를
# 추가하지 말 것. 런타임에 hash/names가 깨진다.
@dataclass(frozen=True)
class DeferredToolCatalog:
    """deferred tool의 불변 catalog. 검색만 하고 변경은 없다."""

    tools: tuple[BaseTool, ...]

    @cached_property
    def names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tools)

    @cached_property
    def hash(self) -> str:
        canon = [{"name": t.name, "schema": convert_to_openai_function(t)} for t in sorted(self.tools, key=lambda t: t.name)]
        blob = json.dumps(canon, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def search(self, query: str) -> list[BaseTool]:
        query = query.strip()
        if not query:
            return []

        if query.startswith("select:"):
            # 개수 제한 없음: ``select:``는 tool을 명시적으로 지목하므로, 일부만 반환하면 모델이
            # 이름으로 요청한 schema가 조용히 누락된다. ``SkillCatalog.search``
            # (``skills/catalog.py``)와 같은 방식이며, 아래의 순위 기반 모드는 ``MAX_RESULTS``로
            # 제한을 유지한다.
            wanted = {n.strip() for n in query[7:].split(",")}
            return [t for t in self.tools if t.name in wanted]

        if query.startswith("+"):
            parts = query[1:].split(None, 1)
            if not parts:
                return []  # 필수 토큰 없는 "+" 하나뿐 — 요구할 것이 없다
            required = parts[0].lower()
            candidates = [t for t in self.tools if required in t.name.lower()]
            if len(parts) > 1:
                candidates.sort(key=lambda t: _catalog_regex_score(parts[1], t), reverse=True)
            return candidates[:MAX_RESULTS]

        regex = _compile_catalog_regex(query)
        scored: list[tuple[int, BaseTool]] = []
        for t in self.tools:
            searchable = f"{t.name} {t.description or ''}"
            if regex.search(searchable):
                scored.append((2 if regex.search(t.name) else 1, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored][:MAX_RESULTS]


def _catalog_regex_score(pattern: str, t: BaseTool) -> int:
    regex = _compile_catalog_regex(pattern)
    return len(regex.findall(f"{t.name} {t.description or ''}"))


# ── Setup / tool ──


@dataclass(frozen=True)
class DeferredToolSetup:
    """agent 빌드 하나에 대한 deferred-tool 지원 조립 결과.

    세 필드는 한 덩어리로 움직이므로, 호출자는 ``tool_search_tool``로 분기한다.

    - **비어 있음** ``(None, frozenset(), None)``: deferral이 꺼져 있거나 후보 목록에 MCP tool이
      없다. 아무것도 deferred되지 않으므로 tool을 그대로 바인딩한다.
    - **채워짐**: ``tool_search_tool``이 agent tool 목록에 추가되고, ``deferred_names``는
      promotion 전까지 모델에게 감춰지며, ``catalog_hash``가 graph state에서 그 promotion의
      범위를 정한다.

    불변식: ``tool_search_tool is None`` ⟺ ``deferred_names``가 비어 있음 ⟺
    ``catalog_hash is None``.
    """

    tool_search_tool: BaseTool | None
    deferred_names: frozenset[str]
    catalog_hash: str | None


def build_tool_search_tool(catalog: DeferredToolCatalog) -> BaseTool:
    catalog_hash = catalog.hash

    @tool
    def tool_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """deferred tool의 전체 schema 정의를 가져와 호출할 수 있게 만든다.

        deferred tool은 system prompt의 <available-deferred-tools>에 이름만 나타난다.
        가져오기 전에는 이름만 알 수 있다. 이 tool은 query를 deferred tool과 매칭해
        매칭된 tool들의 전체 schema를 반환한다. 반환된 tool은 그때부터 호출할 수 있다.

        query 형식:
          - "select:Read,Edit" -- 이 tool들을 이름으로 정확히 가져온다
          - "notebook jupyter" -- keyword 검색, 최대 max_results개의 최적 매칭
          - "+slack send" -- 이름에 "slack"을 요구하고, 나머지 단어로 순위를 매긴다
        """
        matched = catalog.search(query)
        if not matched:
            content, names = f"No tools found matching: {query}", []
        else:
            content = json.dumps([convert_to_openai_function(t) for t in matched], indent=2, ensure_ascii=False)
            names = [t.name for t in matched]
        return Command(
            update={
                "promoted": {"catalog_hash": catalog_hash, "names": names},
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id, name="tool_search")],
            }
        )

    return tool_search


def build_deferred_tool_setup(candidate_tools: list[BaseTool], *, enabled: bool) -> DeferredToolSetup:
    """agent 빌드 하나의 후보 tool로부터 deferred-tool setup을 만든다.

    lead agent는 설정된 전체 tool 목록을 넘긴다. 이후 ``SkillToolPolicyMiddleware``가 활성
    skill에 맞춰 모델이 보는 schema, 실행, ``tool_search`` 결과를 필터링하되 탐색 tool 자체는
    남겨둔다. subagent는 설정된 skill이 시작 시점에 로드되므로 정적으로 정책 필터링된 목록을
    넘길 수 있다. 어느 쪽이든 하위 deferred-schema middleware가 promotion되지 않은 MCP schema를
    계속 숨긴다.

    두 가지 서로 다른 경우에 빈 setup(:class:`DeferredToolSetup` 참고)을 반환한다. deferral이
    꺼져 있거나, 켜져 있지만 호출자의 빌드 시점 선별에서 살아남은 MCP tool이 없는 경우다.
    """
    if not enabled:
        # deferral 비활성: 아무것도 미루지 않고 모델이 예전처럼 모든 tool을 바인딩한다.
        return DeferredToolSetup(None, frozenset(), None)
    deferred = [t for t in candidate_tools if is_mcp_tool(t)]
    if not deferred:
        # 활성이지만 미룰 MCP tool이 없음: 같은 빈 결과, 다른 이유.
        return DeferredToolSetup(None, frozenset(), None)
    catalog = DeferredToolCatalog(tuple(deferred))
    return DeferredToolSetup(build_tool_search_tool(catalog), catalog.names, catalog.hash)


def assemble_deferred_tools(candidate_tools: list[BaseTool], *, enabled: bool) -> tuple[list[BaseTool], DeferredToolSetup]:
    """후보 tool로부터 최종 tool 목록과 deferred setup을 만든다.

    deferral 조립 자체는 fail-closed다. tool_search가 켜져 있고 MCP 후보가 있는데 deferred
    집합을 복원하지 못했다면, 전체 schema를 조용히 모델에 바인딩하는 대신 예외를 던진다.
    lead agent의 authorization은 런타임에 ``SkillToolPolicyMiddleware``가 따로 강제하고,
    subagent는 ``candidate_tools``에 이미 정적 skill 정책을 적용했을 수 있다.

    모든 agent 빌드 경로(lead, embedded client, subagent)가 이를 공유하므로 fail-closed 보장이
    한곳에서 동일하게 적용된다.
    """
    deferred_setup = build_deferred_tool_setup(candidate_tools, enabled=enabled)
    if enabled and not deferred_setup.deferred_names and any(is_mcp_tool(t) for t in candidate_tools):
        raise RuntimeError("tool_search enabled and MCP candidates exist, but no deferred set was recovered - refusing to bind MCP schemas (fail-closed).")
    final_tools = list(candidate_tools)
    if deferred_setup.tool_search_tool:
        final_tools.append(deferred_setup.tool_search_tool)
    return final_tools, deferred_setup


def _routing_priority(value: Any) -> int:
    # routing index에 저장되는 타입 지정 priority를 만든다. McpRoutingMiddleware
    # ._normalize_index가 이를 방어적으로 다시 파싱하므로(임의의 직렬화 데이터를 받도록
    # 만들어졌다), 둘 중 하나가 바뀌면 두 변환 규칙을 함께 맞춘다.
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _routing_keywords(value: Any) -> list[str]:
    # _routing_priority 참고: McpRoutingMiddleware._normalize_index가 keyword를 방어적으로
    # 다시 정규화하므로 두 변환 규칙을 함께 맞춘다.
    if not isinstance(value, list):
        return []
    return [keyword for keyword in (str(item).strip() for item in value) if keyword]


def build_mcp_routing_middleware(
    tools: Iterable[BaseTool],
    deferred_setup: DeferredToolSetup,
    *,
    top_k: int,
) -> "AgentMiddleware | None":
    """호출자의 deferred tool로부터 PR2 auto-promotion middleware를 만든다.

    builder는 생성 시점에 ``BaseTool.metadata``를 들여다볼 수 있지만, 반환되는 middleware는
    평탄하고 직렬화 가능한 routing index만 받는다.
    """
    if deferred_setup.catalog_hash is None or not deferred_setup.deferred_names:
        return None

    routing_index: dict[str, dict[str, Any]] = {}
    for candidate in tools:
        tool_name = getattr(candidate, "name", "")
        if tool_name not in deferred_setup.deferred_names:
            continue
        routing = get_mcp_routing(candidate)
        if routing is None or routing.get("mode") != "prefer":
            continue
        keywords = _routing_keywords(routing.get("keywords"))
        if not keywords:
            continue
        if routing.get("auto_promote_top_k") is not None:
            logger.debug("Ignoring per-tool MCP routing auto_promote_top_k for %s in PR2", tool_name)
        routing_index[str(tool_name)] = {
            "priority": _routing_priority(routing.get("priority", 0)),
            "keywords": keywords,
        }

    if not routing_index:
        return None

    from deerflow.agents.middlewares.mcp_routing_middleware import McpRoutingMiddleware

    return McpRoutingMiddleware(routing_index, deferred_setup.catalog_hash, top_k)


# Prompt 렌더링


def get_deferred_tools_prompt_section(*, deferred_names: frozenset[str] = frozenset()) -> str:
    """명시적으로 주어진 deferred 이름 집합으로 <available-deferred-tools>를 생성한다.

    이름만 나열해 agent가 무엇이 있는지 알고 tool_search로 불러올 수 있게 한다. deferred tool이
    없으면 빈 문자열을 반환한다. 이 집합은 agent 빌드 시점에 계산되어 전달된다. lead agent의
    집합은 활성 skill 정책이 런타임에 적용되므로 설정된 MCP catalog 전체를 담고, subagent의
    집합은 시작 시점 skill 정책으로 이미 필터링되어 있을 수 있다.

    ``deferred_names``를 만드는 조립부 바로 옆에 두어, 모든 agent 빌드 경로(lead, embedded
    client, subagent)가 ``lead_agent.prompt``에 다시 결합되지 않고 같은 방식으로 이 섹션을
    렌더링한다.
    """
    if not deferred_names:
        return ""
    # 이름은 외부 MCP 서버에서 그대로 온다. 조작된 tool 이름이 이 블록을 닫고 framework 태그를
    # 위조하지 못하도록 escape한다. get_skill_index_prompt_section과 동일한 방식이다.
    names = "\n".join(html.escape(name, quote=False) for name in sorted(deferred_names))
    return f"<available-deferred-tools>\n{names}\n</available-deferred-tools>"


def _format_keyword_list(keywords: list[str]) -> str:
    if len(keywords) == 1:
        return keywords[0]
    return f"{', '.join(keywords[:-1])}, or {keywords[-1]}"


def get_mcp_routing_hints_prompt_section(tools: Iterable[BaseTool], *, deferred_names: frozenset[str] = frozenset()) -> str:
    """routing metadata를 가진 MCP tool로 <mcp_routing_hints>를 렌더링한다.

    tool_search가 MCP tool을 deferred 처리한 경우, hint는 모델을 먼저 promotion으로 안내해야
    한다. 그러지 않으면 바인딩된 모델 요청에서 숨겨진 schema를 호출하려 할 수 있다.
    """
    hints: list[tuple[int, str, list[str]]] = []
    for candidate in tools:
        routing = get_mcp_routing(candidate)
        if routing is None or routing.get("mode") != "prefer":
            continue
        keywords = routing.get("keywords") or []
        if not keywords:
            continue
        hints.append((int(routing.get("priority", 0)), candidate.name, [html.escape(str(keyword), quote=False) for keyword in keywords]))

    if not hints:
        return ""

    lines = ["<mcp_routing_hints>"]
    for priority, tool_name, keywords in sorted(hints, key=lambda item: (-item[0], item[1])):
        # tool_name은 외부 MCP 서버에서 그대로 온다. 렌더링 시점에 escape한다(위의
        # deferred_names 포함 검사에는 원본 이름을 그대로 쓴다).
        esc_name = html.escape(tool_name, quote=False)
        lines.append(f"When the user's request involves {_format_keyword_list(keywords)}:")
        if tool_name in deferred_names:
            lines.append(f"  use `tool_search` to fetch `{esc_name}`, then prefer that MCP tool.")
        else:
            lines.append(f"  prefer the `{esc_name}` tool.")
    lines.append("</mcp_routing_hints>")
    return "\n".join(lines)
