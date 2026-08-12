"""TUI에서 tool 활동을 간결하고 읽기 좋게 포맷하는 모듈.

순수 헬퍼만 있다. tool 이름 + args(또는 tool 결과)를 받아 raw JSON을 쏟아내는 대신 transcript용
짧고 읽기 좋은 문자열을 만든다. Textual 의존성은 없다.
"""

from __future__ import annotations

import json
from typing import Any

# 내장 tool의 친숙한 제목. 목록에 없으면 원본 이름을 사람이 읽기 좋게 변환해 쓴다.
_TOOL_TITLES: dict[str, str] = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "str_replace": "Edit",
    "bash": "Bash",
    "shell": "Shell",
    "command": "Run",
    "web_search": "Search",
    "web_fetch": "Fetch",
    "todo_write": "Todo",
    "task": "Subagent",
    "ls": "List",
    "glob": "Find",
    "grep": "Search",
}

# tool별로 inline에 보여 줄 가장 핵심적인 값이 어느 arg에 있는지 지정한다.
_DETAIL_KEYS: dict[str, tuple[str, ...]] = {
    "read_file": ("path", "file_path", "filename"),
    "write_file": ("path", "file_path", "filename"),
    "edit_file": ("path", "file_path", "filename"),
    "bash": ("command", "cmd"),
    "shell": ("command", "cmd"),
    "command": ("command", "cmd"),
    "web_search": ("query", "q"),
    "grep": ("pattern", "query"),
    "glob": ("pattern",),
    "web_fetch": ("url",),
}

# tool이 _DETAIL_KEYS에 없을 때 시도할 일반 arg 키.
_GENERIC_DETAIL_KEYS = ("path", "file_path", "command", "query", "url", "pattern", "name")

DEFAULT_DETAIL_LIMIT = 80
DEFAULT_RESULT_LIMIT = 160


def truncate(text: str, limit: int) -> str:
    """``text``를 ``limit``자로 자르고 말줄임 표시를 붙인다."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def summarize_tool_title(tool_name: str) -> str:
    if not tool_name or not tool_name.strip():
        return "Tool"
    if tool_name in _TOOL_TITLES:
        return _TOOL_TITLES[tool_name]
    return _humanize(tool_name)


def format_tool_detail(tool_name: str, args: Any, limit: int = DEFAULT_DETAIL_LIMIT) -> str:
    """tool 호출의 짧은 inline 상세 정보(예: 경로나 명령어)를 반환한다."""
    if not isinstance(args, dict) or not args:
        return ""

    keys = _DETAIL_KEYS.get(tool_name, ()) + _GENERIC_DETAIL_KEYS
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return truncate(_one_line(value), limit)

    # fallback: args를 압축 JSON으로 만든다.
    try:
        compact = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        compact = str(args)
    return truncate(compact, limit)


def format_tool_result(result: Any, limit: int = DEFAULT_RESULT_LIMIT) -> str:
    """tool 결과를 한 줄로 자른 미리보기를 반환한다."""
    if result is None:
        return ""
    if not isinstance(result, str):
        try:
            result = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            result = str(result)
    return truncate(_one_line(result), limit)


def _one_line(text: str) -> str:
    """연속된 공백(줄바꿈 포함)을 모두 공백 하나로 합친다."""
    return " ".join(text.split())


def _humanize(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return name
    return " ".join(word[:1].upper() + word[1:] for word in cleaned.split())
