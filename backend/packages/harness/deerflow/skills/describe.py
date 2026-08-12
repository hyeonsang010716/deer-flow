"""describe_skill — runtime에 skill metadata를 지연 조회한다.

``describe_skill`` tool을 :class:`SkillCatalog`에 대한 closure로 만든다. 이 tool은 구조화된
metadata(description, allowed tools, 파일 위치)를 반환하므로, LLM이 전체 SKILL.md를
``read_file``할지 판단할 수 있다.

``tool_search.py``의 ``build_tool_search_tool``과 동일한 구조다. query 문법, ``Command`` +
``ToolMessage`` 반환 형태, fail-safe 축소 동작이 모두 같다.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.tools import BaseTool

from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.skills.catalog import SkillCatalog
from deerflow.skills.types import SkillCategory

logger = logging.getLogger(__name__)


# ── 설정 ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillSearchSetup:
    """agent 빌드 한 번에 대해 skill search를 조립한 결과.

    ``tool_search.py``의 ``DeferredToolSetup``과 동일한 구조다.

    - **비어 있음** ``(None, frozenset())``: 사용 가능한 skill이 없거나 skill search가
      비활성이다. agent는 기존 전체 metadata prompt로 fallback한다.
    - **채워짐**: ``describe_skill_tool``이 agent tool 목록에 추가되고, 전체 metadata 대신
      ``skill_names``가 ``<skill_index>``에 렌더링된다.
    """

    describe_skill_tool: BaseTool | None
    skill_names: frozenset[str]


def build_describe_skill_tool(
    catalog: SkillCatalog,
    *,
    container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
) -> BaseTool:
    """``describe_skill`` tool을 *catalog*에 대한 closure로 만든다.

    반환되는 tool은 catalog를 검색해 ``ToolMessage``를 감싼 ``Command``를 돌려주는 평범한
    ``@tool`` 함수다. deferred tool을 승격시키는 ``tool_search``와 달리 graph state 변경은
    필요 없다.
    """

    @tool
    def describe_skill(
        name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """설치된 skill의 사용 metadata를 가져와 로드할지 판단한다.

        skill은 system prompt의 <skill_index>에 이름만 나타난다.  가져오기 전에는
        이름만 알 수 있다.  이 tool은 query를 설치된 skill과 매칭해 전체 metadata —
        description, allowed tools, 파일 위치 — 를 반환하므로, read_file로 SKILL.md를
        로드할지 판단할 수 있다.

        query 형식:
          - "select:data-analysis,deep-research" -- 이 skill들을 정확히 가져온다 (개수 제한 없음)
          - "chart visualization" -- keyword 검색, 가장 잘 맞는 것들 (최대 5개)
          - "+podcast gen" -- 이름에 "podcast"를 요구하고, 나머지 단어로 순위를 매긴다 (최대 5개)
        """
        matched = catalog.search(name)
        if not matched:
            content = f"No skills matched: {name}"
        else:
            content = _render_skill_metadata(matched, container_base_path)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name="describe_skill",
                    )
                ],
            }
        )

    return describe_skill


def build_skill_search_setup(
    skills: list,
    *,
    enabled: bool,
    container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
) -> SkillSearchSetup:
    """필터링된 skill 목록으로 skill search setup을 만든다.

    ``tool_search.py``의 ``build_deferred_tool_setup``과 동일한 구조다.

    *enabled*가 ``False``이거나 *skills*가 비어 있으면 빈 setup을 반환한다.
    """
    if not enabled or not skills:
        return SkillSearchSetup(None, frozenset())

    catalog = SkillCatalog(tuple(skills))
    return SkillSearchSetup(
        describe_skill_tool=build_describe_skill_tool(
            catalog,
            container_base_path=container_base_path,
        ),
        skill_names=catalog.names,
    )


# ── 렌더링 ──────────────────────────────────────────────────────────────────────


def _render_skill_metadata(skills: list, container_base_path: str) -> str:
    """매칭된 skill 목록의 구조화된 metadata를 렌더링한다."""
    blocks: list[str] = []
    for s in skills:
        mutability = "[custom, editable]" if s.category == SkillCategory.CUSTOM else "[built-in]"
        tools_line = ", ".join(s.allowed_tools) if s.allowed_tools else "(all)"
        location = s.get_container_file_path(container_base_path)
        # name/description/allowed-tools는 신뢰할 수 없는 ``.skill`` frontmatter에서 온다.
        # 값이 describe_skill 출력에서 framework tag를 위조하지 못하도록 escape한다.
        name = html.escape(s.name, quote=False)
        description = html.escape(s.description, quote=False)
        tools = html.escape(tools_line, quote=False)
        loc = html.escape(location, quote=False)
        blocks.append(f"## Skill: {name}\n- Description: {description} {mutability}\n- Allowed tools: {tools}\n- Location: {loc}")
    return "\n\n".join(blocks)


# ── Prompt 렌더링 ───────────────────────────────────────────────────────────────


def get_skill_index_prompt_section(
    *,
    skill_names: frozenset[str] = frozenset(),
    container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
    skill_evolution_section: str = "",
) -> str:
    """이름만 담은 ``<skill_index>``와 함께 ``<skill_system>``을 생성한다.

    ``tool_search.py``의 ``get_deferred_tools_prompt_section``과 동일한 구조다. agent는 무엇이
    존재하는지 알고, ``describe_skill``로 metadata를 불러올 수 있다.

    skill이 없으면 빈 문자열을 반환한다.
    """
    if not skill_names:
        return ""

    names = ", ".join(html.escape(name, quote=False) for name in sorted(skill_names))
    evolution = f"\n{skill_evolution_section}" if skill_evolution_section else ""

    return f"""<skill_system>
You have access to skills that provide optimized workflows for specific tasks.

**Skill Discovery:**
1. Check <skill_index> for a skill name that matches your task
2. Call describe_skill(name) to fetch its description and capabilities
3. If the skill matches, call read_file on the returned location to load full instructions
4. Follow the skill's instructions precisely

**Explicit Slash Skill Activation:**
- If the user starts a request with `/<skill-name>`, that skill was explicitly requested.
- The runtime injects the activated skill content; do not call `read_file` for that SKILL.md again unless the injected skill references supporting resources you need.
{evolution}
<skill_index>
{names}
</skill_index>

Skills are located at: {container_base_path}
</skill_system>"""
