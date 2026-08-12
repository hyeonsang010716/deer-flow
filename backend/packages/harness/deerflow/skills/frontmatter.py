"""공유 SKILL.md frontmatter 파싱 헬퍼.

runtime 파서, 설치 시점 validator, review core가 모두 이 모듈을 DeerFlow SKILL.md metadata의
schema 기준으로 사용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

ALLOWED_FRONTMATTER_PROPERTIES = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "required-secrets",
    "secrets-autonomous",
    "metadata",
    "compatibility",
    "version",
    "author",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class SkillMarkdownParts:
    """SKILL.md 문서를 파싱한 조각들."""

    metadata: dict[str, Any]
    frontmatter_text: str
    body: str


def split_skill_markdown(content: str) -> tuple[SkillMarkdownParts | None, str | None]:
    """SKILL.md 문서를 frontmatter와 body로 나눈다.

    성공하면 ``(parts, None)``을, 실패하면 ``(None, message)``를 반환한다. message는 host 경로를
    의도적으로 담지 않으므로, 호출자가 결정적인 review 출력에 그대로 재사용할 수 있다.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None, "No YAML frontmatter found"

    frontmatter_text = match.group(1)
    try:
        metadata = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML in frontmatter: {exc}"

    if not isinstance(metadata, dict):
        return None, "Frontmatter must be a YAML dictionary"

    # YAML은 문자열이 아닌 키도 허용하지만, 이후 검증은 필드 이름이 문자열이라고 가정한다.
    metadata = {str(key): value for key, value in metadata.items()}

    return (
        SkillMarkdownParts(
            metadata=metadata,
            frontmatter_text=frontmatter_text,
            body=content[match.end() :],
        ),
        None,
    )
