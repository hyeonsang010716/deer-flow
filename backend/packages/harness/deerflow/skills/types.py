from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH

SKILL_MD_FILE = "SKILL.md"


class SkillCategory(StrEnum):
    """skill의 출처 카테고리.

    - ``PUBLIC``: 플랫폼에 기본 포함된 built-in skill. 읽기 전용.
    - ``CUSTOM``: 사용자가 작성한 skill. 수정·삭제할 수 있다.
    - ``INTEGRATION``: 관리형 서드파티 integration skill. 읽기 전용.
    - ``LEGACY``: 사용자 격리 migration 이전의 전역 custom skill. 읽기 전용으로
      노출된다(보이지만 수정·삭제 불가). sandbox에서 ``/mnt/skills/legacy/<name>/``에
      mount된다.
    """

    PUBLIC = "public"
    CUSTOM = "custom"
    INTEGRATION = "integrations"
    LEGACY = "legacy"


@dataclass(frozen=True)
class SecretRequirement:
    """skill이 필요하다고 선언한 request 스코프 secret(이슈 #3861).

    ``name``은 request의 ``context.secrets``에서 조회하는 키이자, skill이 활성화될 때
    해당 skill의 sandbox subprocess에 주입되는 환경 변수 이름이다.
    """

    name: str
    optional: bool = False


@dataclass(frozen=True)
class Skill:
    """skill과 그 metadata, 파일 경로를 나타낸다"""

    name: str
    description: str
    license: str | None
    skill_dir: Path
    skill_file: Path
    relative_path: Path  # 카테고리 루트에서 skill 디렉터리까지의 상대 경로
    category: SkillCategory  # 'public' 또는 'custom'
    allowed_tools: tuple[str, ...] | None = None
    enabled: bool = False  # 이 skill이 활성화되어 있는지 여부
    required_secrets: tuple[SecretRequirement, ...] = field(default_factory=tuple)
    # 선언된 secret이 모델의 자율적 로드(skill_context)로 skill이 context에 들어왔을 때도
    # 바인딩될 수 있는지, 아니면 명시적 /slash 활성화 때만 가능한지. frontmatter:
    # ``secrets-autonomous``(기본값 true).
    secrets_autonomous: bool = True

    @property
    def skill_path(self) -> str:
        """카테고리 루트(skills/{category})에서 이 skill의 디렉터리까지의 상대 경로를 반환한다"""
        path = self.relative_path.as_posix()
        return "" if path == "." else path

    def get_container_path(self, container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH) -> str:
        """
        container 안에서 이 skill의 전체 경로를 얻는다.

        Args:
            container_base_path: container에서 skill이 mount되는 기준 경로

        Returns:
            skill 디렉터리의 전체 container 경로
        """
        category_base = f"{container_base_path}/{self.category}"
        skill_path = self.skill_path
        if skill_path:
            return f"{category_base}/{skill_path}"
        return category_base

    def get_container_file_path(self, container_base_path: str = DEFAULT_SKILLS_CONTAINER_PATH) -> str:
        """
        container 안에서 이 skill의 메인 파일(SKILL.md) 전체 경로를 얻는다.

        Args:
            container_base_path: container에서 skill이 mount되는 기준 경로

        Returns:
            skill의 SKILL.md 파일의 전체 container 경로
        """
        return f"{self.get_container_path(container_base_path)}/SKILL.md"

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, description={self.description!r}, category={self.category!r})"
