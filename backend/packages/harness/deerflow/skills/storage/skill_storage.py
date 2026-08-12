"""template-method 흐름을 갖춘 추상 SkillStorage base class."""

from __future__ import annotations

import dataclasses
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.skills.types import SKILL_MD_FILE, Skill, SkillCategory  # noqa: F401

logger = logging.getLogger(__name__)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillStorage(ABC):
    """skill storage backend의 추상 base.

    subclass는 저장 매체별 atomic 연산 몇 개만 구현한다. 이 base class는 그것들을 프로토콜
    수준 헬퍼와 조합한 최종 template-method 흐름(load_skills, history 직렬화, 경로 헬퍼,
    검증)을 제공한다.
    """

    def __init__(self, container_path: str = DEFAULT_SKILLS_CONTAINER_PATH) -> None:
        self._container_root = container_path

    # ------------------------------------------------------------------
    # 정적 프로토콜 헬퍼 (storage에 종속되지 않음)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_skill_name(name: str) -> str:
        """skill 이름을 검증·정규화하고 정규화된 형태를 반환한다."""
        normalized = name.strip()
        if not _SKILL_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Skill name must be hyphen-case using lowercase letters, digits, and hyphens only.")
        if len(normalized) > 64:
            raise ValueError("Skill name must be 64 characters or fewer.")
        return normalized

    @staticmethod
    def validate_relative_path(relative_path: str, base_dir: Path) -> Path:
        """*relative_path*를 *base_dir* 기준으로 검증하고 resolve된 대상을 반환한다.

        *relative_path*가 비어 있지 않은지 확인한 뒤 *base_dir*와 결합해 결과를 resolve한다
        (symlink를 따라간다). resolve된 대상이 *base_dir* 안에 있지 않으면 ``ValueError``를
        던진다.
        """
        if not relative_path:
            raise ValueError("relative_path must not be empty.")
        resolved_base = base_dir.resolve()
        target = (resolved_base / relative_path).resolve()
        try:
            target.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError("relative_path must resolve within the skill directory.") from exc
        return target

    @staticmethod
    def validate_skill_markdown_content(name: str, content: str) -> None:
        """SKILL.md 내용을 검증한다. frontmatter를 파싱하고 이름이 일치하는지 확인한다."""
        import tempfile

        from deerflow.skills.validation import _validate_skill_frontmatter

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_skill_dir = Path(tmp_dir) / SkillStorage.validate_skill_name(name)
            temp_skill_dir.mkdir(parents=True, exist_ok=True)
            (temp_skill_dir / SKILL_MD_FILE).write_text(content, encoding="utf-8")
            is_valid, message, parsed_name = _validate_skill_frontmatter(temp_skill_dir)
            if not is_valid:
                raise ValueError(message)
            if parsed_name != name:
                raise ValueError(f"Frontmatter name '{parsed_name}' must match requested skill name '{name}'.")

    def ensure_safe_support_path(self, name: str, relative_path: str) -> Path:
        """support 파일의 resolve된 절대 경로를 검증해 반환한다."""
        _ALLOWED_SUPPORT_SUBDIRS = {"references", "templates", "scripts", "assets"}
        skill_dir = self.get_custom_skill_dir(self.validate_skill_name(name)).resolve()
        if not relative_path or relative_path.endswith("/"):
            raise ValueError("Supporting file path must include a filename.")
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("Supporting file path must be relative.")
        if any(part in {"..", ""} for part in relative.parts):
            raise ValueError("Supporting file path must not contain parent-directory traversal.")
        top_level = relative.parts[0] if relative.parts else ""
        if top_level not in _ALLOWED_SUPPORT_SUBDIRS:
            raise ValueError(f"Supporting files must live under one of: {', '.join(sorted(_ALLOWED_SUPPORT_SUBDIRS))}.")
        target = (skill_dir / relative).resolve()
        allowed_root = (skill_dir / top_level).resolve()
        try:
            target.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("Supporting file path must stay within the selected support directory.") from exc
        return target

    # ------------------------------------------------------------------
    # 추상 atomic 연산 (저장 매체별 구현)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_skills_root_path(self) -> Path:
        """sandbox 마운트에 쓰는 skills 루트의 host 절대 경로.

        출처: ``deerflow.skills.loader.get_skills_root_path``.
        """

    def validate_skill_file_path(self, skill_file: Path) -> Path:
        """*skill_file*이 허용된 루트 안에 있는지 검증하고 resolve된 경로를 반환한다.

        기본 구현은 ``skill_file``이 ``get_skills_root_path()`` 아래에 있는지 확인한다. public과
        custom skill이 같은 루트 아래에 있는 :class:`LocalSkillStorage`에는 이것으로 충분하다.

        :class:`UserScopedSkillStorage`는 이를 override해서 사용자별 custom 루트 아래 파일도
        받아들인다. custom skill은 전역 루트의 하위 경로가 아닌 별도 디렉터리 트리에
        저장되기 때문이다.

        Raises:
            ValueError: resolve된 경로가 허용된 모든 루트를 벗어날 때.
        """
        resolved_file = skill_file.resolve()
        resolved_root = self.get_skills_root_path().resolve()
        try:
            resolved_file.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Resolved skill file must stay within the configured skills root.") from exc
        return resolved_file

    @abstractmethod
    def _iter_skill_files(self) -> Iterable[tuple[SkillCategory, Path, Path]]:
        """모든 SKILL.md에 대해 ``(category, category_root, skill_md_path)``를 yield한다.

        출처: ``deerflow.skills.loader.load_skills`` 안의 디렉터리 순회 로직에서 추출.
        """

    @abstractmethod
    def read_custom_skill(self, name: str) -> str:
        """custom skill의 SKILL.md 내용을 읽는다.

        출처: ``deerflow.skills.manager.read_custom_skill_content``.
        """

    @abstractmethod
    def write_custom_skill(self, name: str, relative_path: str, content: str) -> None:
        """``custom/<name>/<relative_path>`` 아래에 텍스트 파일을 atomic하게 쓴다.

        출처: ``deerflow.skills.manager.atomic_write``.
        """

    def remove_custom_skill_file(self, name: str, relative_path: str) -> str:
        """support 파일을 제거하고 이전 텍스트 내용을 반환한다."""
        target = self.ensure_safe_support_path(name, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Supporting file '{relative_path}' not found for skill '{name}'.")
        previous_content = target.read_text(encoding="utf-8")
        target.unlink()
        return previous_content

    @abstractmethod
    async def ainstall_skill_from_archive(self, archive_path: str | Path) -> dict:
        """``.skill`` ZIP archive에서 skill을 비동기로 설치한다.

        출처: ``deerflow.skills.installer.ainstall_skill_from_archive``.
        """

    def install_skill_from_archive(self, archive_path: str | Path) -> dict:
        """동기 wrapper. :meth:`ainstall_skill_from_archive`에 위임한다."""
        from deerflow.skills.installer import _run_async_install

        return _run_async_install(self.ainstall_skill_from_archive(archive_path))

    @abstractmethod
    def delete_custom_skill(self, name: str, *, history_meta: dict | None = None) -> None:
        """custom skill을 삭제한다(검증 + 선택적 history 기록 + 디렉터리 제거).

        출처: ``app.gateway.routers.skills.delete_custom_skill`` + ``skill_manage_tool``.
        """

    @abstractmethod
    def custom_skill_exists(self, name: str) -> bool:
        """출처: ``deerflow.skills.manager.custom_skill_exists``."""

    @abstractmethod
    def public_skill_exists(self, name: str) -> bool:
        """출처: ``deerflow.skills.manager.public_skill_exists``."""

    @abstractmethod
    def append_history(self, name: str, record: dict) -> None:
        """``name``에 대한 JSONL history 항목을 덧붙인다.

        출처: ``deerflow.skills.manager.append_history``.
        """

    @abstractmethod
    def read_history(self, name: str) -> list[dict]:
        """``name``의 모든 history 레코드를 오래된 순으로 반환한다.

        출처: ``deerflow.skills.manager.read_history``.
        """

    # ------------------------------------------------------------------
    # 구체 경로 헬퍼 (레이아웃은 SKILL.md 프로토콜의 일부다)
    # ------------------------------------------------------------------

    def get_container_root(self) -> str:
        """출처: ``deerflow.config.skills_config.SkillsConfig.container_path`` accessor."""
        return self._container_root

    def get_custom_skill_dir(self, name: str) -> Path:
        """``custom/<name>`` 경로. 디렉터리를 만들지는 않는다.

        출처: ``deerflow.skills.manager.get_custom_skill_dir``.
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_skills_root_path() / SkillCategory.CUSTOM.value / normalized_name

    def get_custom_skill_file(self, name: str) -> Path:
        """``custom/<name>/SKILL.md`` 경로.

        출처: ``deerflow.skills.manager.get_custom_skill_file``.
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_custom_skill_dir(normalized_name) / SKILL_MD_FILE

    def get_skill_history_file(self, name: str) -> Path:
        """``custom/.history/<name>.jsonl`` 경로. 상위 디렉터리를 만들지는 않는다.

        **주의:** 이 기본 구현은 전역 skills 루트 아래 경로를 반환하며, 이는
        :class:`LocalSkillStorage`에는 맞지만 :class:`UserScopedSkillStorage`에는 **틀리다**.
        custom skill 경로를 다른 곳으로 돌리는 subclass는 이 메서드를 override해야 한다
        (``UserScopedSkillStorage``는 이미 그렇게 한다).

        출처: ``deerflow.skills.manager.get_skill_history_file``.
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_skills_root_path() / SkillCategory.CUSTOM.value / ".history" / f"{normalized_name}.jsonl"

    # ------------------------------------------------------------------
    # 최종 template-method 흐름
    # ------------------------------------------------------------------

    def load_skills(self, *, enabled_only: bool = False) -> list[Skill]:
        """모든 skill을 찾아 enabled 상태를 병합하고 정렬하며, 선택적으로 필터링한다.

        출처: ``deerflow.skills.loader.load_skills``.
        """
        from deerflow.skills.parser import parse_skill_file

        skills_by_name: dict[str, Skill] = {}
        for category, category_root, md_path in self._iter_skill_files():
            skill = parse_skill_file(
                md_path,
                category=category,
                relative_path=md_path.parent.relative_to(category_root),
            )
            if skill:
                skills_by_name[skill.name] = skill

        skills = list(skills_by_name.values())

        # extensions config에서 enabled 상태를 병합한다(다른 프로세스의 변경이 즉시 반영되도록
        # 호출마다 다시 읽는다). 모든 skill category(PUBLIC, LEGACY, CUSTOM)가
        # extensions_config의 enabled/disabled 상태를 따른다. CUSTOM skill은 명시적 config
        # 항목이 없으면 기본으로 enabled다(새로 설치한 skill이 수동 토글 없이 바로 활성으로
        # 보이게 하기 위함).
        try:
            from deerflow.config.extensions_config import ExtensionsConfig

            extensions_config = ExtensionsConfig.from_file()
            skills = [dataclasses.replace(s, enabled=extensions_config.is_skill_enabled(s.name, s.category)) for s in skills]
        except Exception as e:
            logger.warning("Failed to load extensions config: %s", e)

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        skills.sort(key=lambda s: s.name)
        return skills

    def ensure_custom_skill_is_editable(self, name: str) -> None:
        """출처: ``deerflow.skills.manager.ensure_custom_skill_is_editable``.

        CUSTOM category skill만 편집할 수 있다. PUBLIC(내장)과 LEGACY(마이그레이션 이전 공유)
        skill은 읽기 전용이며, 편집을 시도하면 도움말이 담긴 ``ValueError``를 던진다.
        """
        if self.custom_skill_exists(name):
            return
        if self.public_skill_exists(name):
            raise ValueError(f"'{name}' is a built-in skill. To customise it, create a new skill with the same name under skills/custom/.")
        raise FileNotFoundError(f"Custom skill '{name}' not found.")
