import logging
from typing import Protocol

from deerflow.skills.types import Skill

logger = logging.getLogger(__name__)


class NamedTool(Protocol):
    name: str


# 활성 skill이 allowed-tools를 선언해도 계속 쓸 수 있는 프레임워크 내장 tool. 리뷰/활성화된
# skill 자신의 business tool 권한을 넓히는 것이 아니라, 통제된 파일/리뷰/탐색 워크플로를
# 뒷받침한다. 특히 tool_search를 통한 promotion은 SkillToolPolicyMiddleware가 제거한 tool을
# 되살리지 않으며, describe_skill은 catalog metadata만 반환한다.
ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES = frozenset(
    {
        "describe_skill",
        "read_file",
        "review_skill_package",
        "tool_search",
    }
)


def allowed_tool_names_for_skills(skills: list[Skill]) -> set[str] | None:
    """명시적으로 선언된 skill allowed-tools의 합집합을 반환한다.

    None은 legacy의 전체 허용 동작을 뜻하며, 로드된 어떤 skill도 allowed-tools를 선언하지 않은
    경우에만 반환된다. 어느 한 skill이라도 그 필드를 선언하면, 필드가 없는 legacy skill은 다른
    skill의 명시적 제한을 무력화하는 대신 아무 tool도 기여하지 않는다.
    """
    if not skills:
        return None

    allowed: set[str] = set()
    has_explicit_declaration = False
    for skill in skills:
        if skill.allowed_tools is None:
            continue
        has_explicit_declaration = True
        if not skill.allowed_tools:
            logger.info("Skill %s declared empty allowed-tools", skill.name)
        allowed.update(skill.allowed_tools)

    if not has_explicit_declaration:
        return None
    return allowed


def filter_tools_by_skill_allowed_tools[ToolT: NamedTool](
    tools: list[ToolT],
    skills: list[Skill],
    *,
    always_allowed_tool_names: set[str] | frozenset[str] = frozenset(),
) -> list[ToolT]:
    allowed = allowed_tool_names_for_skills(skills)
    if allowed is None:
        return tools

    allowed_with_framework_tools = allowed | set(always_allowed_tool_names)
    return [tool for tool in tools if tool.name in allowed_with_framework_tools]
