from __future__ import annotations

import re
from dataclasses import dataclass

from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.skills.types import Skill

#: 앞의 슬래시를 차지하는 composer 제어 명령. 절대 ``/skill`` 활성화로 취급하면 안 된다.
#: 이 값들과 :data:`_SLASH_SKILL_RE`는 ``frontend/src/core/skills/slash.ts``의 프론트엔드 표시
#: 파서와 짝을 이룬다. 양쪽 모두 contract 테스트(여기서는
#: ``tests/test_slash_skill_contract.py``, 프론트엔드에서는 ``slash-contract.test.ts``)로
#: 공유 fixture ``contracts/slash_skill_contract.json``에 고정되어 있어, 한쪽 언어에서만
#: 예약 명령이나 문법을 바꾸면 CI가 실패한다.
RESERVED_SLASH_SKILL_NAMES = frozenset({"bootstrap", "goal", "help", "memory", "models", "new", "status"})
_SLASH_SKILL_RE = re.compile(r"^/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+|$)")


@dataclass(frozen=True, slots=True)
class SlashSkillReference:
    """skill 이름과 남은 작업 텍스트를 담은, 파싱된 slash-skill 명령."""

    name: str
    remaining_text: str


@dataclass(frozen=True, slots=True)
class ResolvedSlashSkill:
    """runtime에서 보이는 활성 skill을 기준으로 해석된 slash-skill 활성화."""

    skill: Skill
    remaining_text: str
    container_file_path: str


def parse_slash_skill_reference(text: str) -> SlashSkillReference | None:
    """엄격한 `/skill-name task` 문법을 파싱하며, 예약된 제어 명령은 무시한다."""
    match = _SLASH_SKILL_RE.match(text)
    if not match:
        return None
    name = match.group(1)
    if name in RESERVED_SLASH_SKILL_NAMES:
        return None
    return SlashSkillReference(
        name=name,
        remaining_text=text[match.end() :].lstrip(),
    )


def resolve_slash_skill(
    text: str,
    skills: list[Skill],
    *,
    available_skills: set[str] | None = None,
    container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
) -> ResolvedSlashSkill | None:
    """가능하면 텍스트를 활성화되고 허용 목록에 있는 skill 활성화로 해석한다."""
    reference = parse_slash_skill_reference(text)
    if reference is None:
        return None
    if available_skills is not None and reference.name not in available_skills:
        return None

    skill = next((candidate for candidate in skills if candidate.name == reference.name and candidate.enabled), None)
    if skill is None:
        return None

    return ResolvedSlashSkill(
        skill=skill,
        remaining_text=reference.remaining_text,
        container_file_path=skill.get_container_file_path(container_base_path),
    )
