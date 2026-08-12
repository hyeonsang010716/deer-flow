"""DeerFlow TUI의 slash command registry(순수 모듈).

두 가지 command source를 검색 가능한 하나의 목록으로 정규화한다:

* **Built-in** — TUI가 소유한 기능(``/help``, ``/model``, ``/threads`` …).
* **Skill** — enabled된 skill마다 ``/<skill-name>`` 하나씩. DeerFlow의 기존 slash-skill 활성화
  의미를 그대로 유지한다.

picker는 이 목록을 필터링하고, :func:`resolve`는 제출된 입력 줄을 built-in command, skill 활성화,
알 수 없는 command, 일반 메시지 중 하나로 분류한다. Textual 의존성은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Command:
    name: str  # 앞의 슬래시는 제외
    description: str
    category: Literal["builtin", "skill"] = "builtin"


@dataclass(frozen=True)
class Resolution:
    kind: Literal["builtin", "skill", "unknown", "message"]
    name: str = ""
    args: str = ""
    text: str = ""


# built-in command. /help과 picker에서 표시할 순서대로 정렬되어 있다.
BUILTIN_COMMANDS: tuple[Command, ...] = (
    Command("help", "Show commands and keybindings"),
    Command("new", "Start a fresh thread"),
    Command("clear", "Clear the transcript display"),
    Command("threads", "Open the thread switcher"),
    Command("switch", "Open the thread switcher"),
    Command("resume", "Resume a thread by id or title"),
    Command("goal", "Set, show or clear the active goal"),
    Command("model", "Open the model picker"),
    Command("skills", "Browse enabled and available skills"),
    Command("tools", "Show built-in, MCP and sandbox tools"),
    Command("mcp", "Show MCP server status"),
    Command("memory", "Show memory status and injected facts"),
    Command("uploads", "Show uploaded files for this thread"),
    Command("artifacts", "Show generated artifacts"),
    Command("details", "Toggle verbose activity rendering"),
    Command("usage", "Show token usage and context"),
    Command("config", "Show resolved config paths and overrides"),
    Command("quit", "Exit the TUI"),
)

_BUILTIN_NAMES = frozenset(c.name for c in BUILTIN_COMMANDS)


def format_command_help() -> str:
    """``/help``에 쓰는, 모든 built-in slash command의 한 줄 요약.

    :data:`BUILTIN_COMMANDS`에서 파생하므로 help 텍스트가 registry(따라서 picker)와 어긋날 수
    없다. built-in을 추가하면 ``/help``에 자동으로 나타난다.
    """
    names = "  ".join(f"/{command.name}" for command in BUILTIN_COMMANDS)
    return f"Commands:  {names}"


def build_registry(skills: list[dict]) -> list[Command]:
    """built-in에 enabled된 skill별 command 하나씩을 합친다."""
    commands = list(BUILTIN_COMMANDS)
    for skill in skills:
        if not skill.get("enabled", False):
            continue
        name = skill.get("name")
        if not name or name in _BUILTIN_NAMES:
            continue
        commands.append(Command(name=name, description=skill.get("description", "") or "", category="skill"))
    return commands


def filter_commands(commands: list[Command], query: str) -> list[Command]:
    """picker용으로 command를 필터링하고 순위를 매긴다.

    순위는 이름 prefix 일치가 먼저, 그다음 이름 substring, 마지막이 description substring이다.
    같은 등급 안에서는 원래 순서를 유지한다.
    """
    q = query.strip().lower()
    if not q:
        return commands

    prefix: list[Command] = []
    substring: list[Command] = []
    description: list[Command] = []
    for command in commands:
        name = command.name.lower()
        if name.startswith(q):
            prefix.append(command)
        elif q in name:
            substring.append(command)
        elif q in command.description.lower():
            description.append(command)
    return prefix + substring + description


def resolve(text: str, skills: list[str] | None = None) -> Resolution:
    """제출된 입력 줄을 분류한다."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return Resolution(kind="message", text=text)

    body = stripped[1:]
    name, _, args = body.partition(" ")
    name = name.strip()
    args = args.strip()

    if not name:
        return Resolution(kind="unknown", name="")

    if name in _BUILTIN_NAMES:
        return Resolution(kind="builtin", name=name, args=args)

    if skills and name in skills:
        return Resolution(kind="skill", name=name, args=args)

    return Resolution(kind="unknown", name=name, args=args)
