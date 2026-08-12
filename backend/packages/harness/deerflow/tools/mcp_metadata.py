"""MCP tool metadata 태그의 단일 기준점.

``deerflow_mcp`` metadata 플래그를 달고 있으면 그 tool은 "MCP에서 온 것"이다. 태그는 MCP tool을
로드하는 곳(``tools.py``)에서 *쓰이고*, deferred tool 조립(``tool_search.py``)과 agent 빌드
지점(``agent.py``)에서 *읽힌다*. 키와 tagger, predicate를 여기 모아 두면 매직 문자열이 정확히
한 곳에만 존재하고, 읽는 쪽은 모듈 간 private 헬퍼 대신 공개 predicate를 import하게 된다.

의도적으로 leaf 모듈이다. ``BaseTool``에만 의존하므로 어떤 모듈(tool loader 포함)이든 import
cycle 없이 가져올 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.tools import BaseTool

MCP_TOOL_METADATA_KEY = "deerflow_mcp"
MCP_TOOL_ROUTING_METADATA_KEY = "deerflow_mcp_routing"


def tag_mcp_tool(tool: BaseTool) -> BaseTool:
    """``tool``을 MCP 출처로 표시한다. 제자리에서 변경하고 체이닝을 위해 그대로 반환한다."""
    tool.metadata = {**(tool.metadata or {}), MCP_TOOL_METADATA_KEY: True}
    return tool


def is_mcp_tool(tool: BaseTool) -> bool:
    """``tool``이 :func:`tag_mcp_tool`이 쓴 MCP 출처 태그를 갖고 있으면 True."""
    return (getattr(tool, "metadata", None) or {}).get(MCP_TOOL_METADATA_KEY) is True


def tag_mcp_routing(tool: BaseTool, routing: Mapping[str, Any]) -> BaseTool:
    """직렬화된 MCP routing metadata를 ``tool``에 붙인다."""
    tool.metadata = {
        **(tool.metadata or {}),
        MCP_TOOL_ROUTING_METADATA_KEY: dict(routing),
    }
    return tool


def get_mcp_routing(tool: BaseTool) -> dict[str, Any] | None:
    """routing mode가 활성인 MCP tool에 대해서만 routing metadata를 반환한다."""
    if not is_mcp_tool(tool):
        return None
    routing = (getattr(tool, "metadata", None) or {}).get(MCP_TOOL_ROUTING_METADATA_KEY)
    if not isinstance(routing, dict) or routing.get("mode") == "off":
        return None
    return routing
