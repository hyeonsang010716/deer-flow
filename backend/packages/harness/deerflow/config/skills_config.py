import os
from pathlib import Path

from pydantic import BaseModel, Field

from deerflow.config.runtime_paths import project_root, resolve_path
from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH


def _legacy_skills_candidates() -> tuple[Path, ...]:
    """monorepo 호환을 위해 소스 트리의 skills 위치를 반환한다."""
    backend_dir = Path(__file__).resolve().parents[4]
    repo_root = backend_dir.parent
    return (repo_root / "skills",)


class SkillsConfig(BaseModel):
    """skills 시스템 설정."""

    use: str = Field(
        default="deerflow.skills.storage.local_skill_storage:LocalSkillStorage",
        description="Class path of the SkillStorage implementation.",
    )
    path: str | None = Field(
        default=None,
        description=("Path to skills directory. If not specified, defaults to `skills` under the caller project root, falling back to the legacy repo-root location for monorepo compatibility."),
    )
    container_path: str = Field(
        default=DEFAULT_SKILLS_CONTAINER_PATH,
        description="Path where skills are mounted in the sandbox container",
    )
    deferred_discovery: bool = Field(
        default=False,
        description=("When enabled, skill metadata is not injected into the system prompt. Instead, only skill names appear in <skill_index> and the LLM discovers details on demand via the describe_skill tool."),
    )

    def get_skills_path(self) -> Path:
        """
        확정된 skills 디렉터리 경로를 반환한다.

        결정 순서:
            1. 명시적인 ``path`` 필드
            2. ``DEER_FLOW_SKILLS_PATH`` 환경변수
            3. 호출자 프로젝트 루트(``project_root()``) 아래의 ``skills``
            4. monorepo 호환용 legacy 저장소 루트 후보(``_legacy_skills_candidates``)

        (3)과 (4) 모두 디스크에 없으면 프로젝트 루트 기본값을 반환한다. 예외를 던지지 않고도
        호출자가 "skill 없음"을 일관된 위치로 표현할 수 있게 하기 위함이다.
        """
        if self.path:
            # 설정된 경로를 쓴다(절대 경로이거나 프로젝트 루트 기준 상대 경로).
            return resolve_path(self.path)
        if env_path := os.getenv("DEER_FLOW_SKILLS_PATH"):
            return resolve_path(env_path)

        project_default = project_root() / "skills"
        if project_default.is_dir():
            return project_default

        for candidate in _legacy_skills_candidates():
            if candidate.is_dir():
                return candidate

        return project_default

    def get_skill_container_path(self, skill_name: str, category: str = "public") -> str:
        """
        특정 skill의 컨테이너 전체 경로를 반환한다.

        Args:
            skill_name: skill 이름(디렉터리 이름).
            category: skill 카테고리(public 또는 custom).

        Returns:
            컨테이너 안에서의 skill 전체 경로.
        """
        return f"{self.container_path}/{category}/{skill_name}"
