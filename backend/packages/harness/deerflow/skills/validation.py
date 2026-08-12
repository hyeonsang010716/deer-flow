"""skill frontmatter 검증 유틸리티.

SKILL.md frontmatter를 순수 로직으로 검증한다. FastAPI나 HTTP 의존성은 없다.
"""

import re
from pathlib import Path

from deerflow.skills.frontmatter import ALLOWED_FRONTMATTER_PROPERTIES, split_skill_markdown
from deerflow.skills.parser import parse_allowed_tools
from deerflow.skills.types import SKILL_MD_FILE


def _validate_skill_frontmatter(skill_dir: Path) -> tuple[bool, str, str | None]:
    """skill 디렉터리의 SKILL.md frontmatter를 검증한다.

    Args:
        skill_dir: SKILL.md를 담고 있는 skill 디렉터리 경로.

    Returns:
        (is_valid, message, skill_name) 튜플.
    """
    skill_md = skill_dir / SKILL_MD_FILE
    if not skill_md.exists():
        return False, f"{SKILL_MD_FILE} not found", None

    content = skill_md.read_text(encoding="utf-8")
    parts, error = split_skill_markdown(content)
    if error:
        return False, error, None
    if parts is None:
        return False, "Invalid frontmatter format", None
    frontmatter = parts.metadata

    # 예상치 못한 속성이 있는지 확인한다
    unexpected_keys = set(frontmatter.keys()) - ALLOWED_FRONTMATTER_PROPERTIES
    if unexpected_keys:
        return False, f"Unexpected key(s) in SKILL.md frontmatter: {', '.join(sorted(unexpected_keys))}", None

    # 필수 필드를 확인한다
    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter", None
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter", None

    # name을 검증한다
    name = frontmatter.get("name", "")
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}", None
    name = name.strip()
    if not name:
        return False, "Name cannot be empty", None

    # 이름 규칙을 확인한다(hyphen-case: 소문자와 하이픈)
    if not re.match(r"^[a-z0-9-]+$", name):
        return False, f"Name '{name}' should be hyphen-case (lowercase letters, digits, and hyphens only)", None
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, f"Name '{name}' cannot start/end with hyphen or contain consecutive hyphens", None
    if len(name) > 64:
        return False, f"Name is too long ({len(name)} characters). Maximum is 64 characters.", None

    # description을 검증한다
    description = frontmatter.get("description", "")
    if not isinstance(description, str):
        return False, f"Description must be a string, got {type(description).__name__}", None
    description = description.strip()
    if description:
        if "<" in description or ">" in description:
            return False, "Description cannot contain angle brackets (< or >)", None
        if len(description) > 1024:
            return False, f"Description is too long ({len(description)} characters). Maximum is 1024 characters.", None

    try:
        parse_allowed_tools(frontmatter.get("allowed-tools"), skill_md)
    except ValueError as e:
        return False, str(e).replace(str(skill_md), SKILL_MD_FILE), None

    required_secrets = frontmatter.get("required-secrets")
    if required_secrets is not None and not isinstance(required_secrets, list):
        return False, f"required-secrets in {SKILL_MD_FILE} must be a list", None

    secrets_autonomous = frontmatter.get("secrets-autonomous")
    if secrets_autonomous is not None and not isinstance(secrets_autonomous, bool):
        return False, f"secrets-autonomous in {SKILL_MD_FILE} must be a boolean", None

    return True, "Skill is valid!", name
