"""custom skill을 사용자별로 격리하는 user-scoped SkillStorage.

custom skill은 전역 ``{base_dir}/skills/custom/`` 대신
``{base_dir}/users/{user_id}/skills/custom/`` 아래에 저장한다. public skill은 여전히 전역
``{base_dir}/skills/public/``에서 읽는다(read-only).

레이아웃::

    <host_root>/public/<name>/SKILL.md                   ← 전역, read-only
    <user_custom_root>/<name>/SKILL.md                   ← 사용자별, read-write
    <integrations_root>/<provider>/<name>/SKILL.md       ← 전역, read-only
    <user_custom_root>/.history/<name>.jsonl             ← 사용자별 history
    <user_skills_root>/_skill_states.json                ← 사용자별 enabled 상태
    <global_custom_root>/<name>/SKILL.md                 ← legacy fallback, read-only

Fallback: 사용자에게 아직 custom skill이 없으면 전역 ``skills/custom/`` skill을
``SkillCategory.LEGACY``(read-only)로 내보내서, 보이기는 하되 사용자가 편집/삭제할 수는
없게 한다. 이렇게 하면 legacy skill에 변경 권한을 흘리지 않으면서 마이그레이션 동안 하위
호환을 유지한다. legacy skill은 sandbox의 ``/mnt/skills/legacy/<name>/``에 마운트되므로
관련 파일(references, templates, scripts, assets)에 agent가 접근할 수 있다.

CUSTOM과 LEGACY skill의 enabled/disabled 상태는 사용자별 ``_skill_states.json``에
skill 이름을 키로 저장한다. PUBLIC skill 상태는 ``extensions_config.json``에 전역으로
남는다. 두 사용자가 같은 이름의 custom skill을 가질 때 상태가 서로 섞이는 것을 막는다.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.skills.permissions import make_skill_written_path_sandbox_readable
from deerflow.skills.storage.local_skill_storage import LocalSkillStorage
from deerflow.skills.storage.skill_storage import SKILL_MD_FILE
from deerflow.skills.types import SkillCategory

logger = logging.getLogger(__name__)


class UserScopedSkillStorage(LocalSkillStorage):
    """custom skill을 사용자별로 격리하는 skill storage.

    public skill 동작은 :class:`LocalSkillStorage`에서 그대로 물려받는다
    (``_host_root/public/``에서 읽는다). custom skill 경로만 ``_user_custom_root``로
    돌려서 각 사용자의 custom skill이 자기 디렉터리 트리에 놓이게 한다.

    Fallback: 사용자의 custom 디렉터리가 비어 있고 전역 ``skills/custom/``에 내용이 있으면
    그 legacy skill들을 ``SkillCategory.LEGACY``로 로드한다 — 목록에는 보이지만 read-only로
    취급되어 편집/삭제할 수 없다. 다른 사용자의 legacy skill에 변경 권한을 주지 않으면서
    마이그레이션 동안 하위 호환을 유지한다.

    **설계 노트**: 사용자가 첫 custom skill을 만드는 순간 사용자별 디렉터리가 생기므로 전역
    custom fallback은 더 이상 적용되지 않고, LEGACY skill이 그 사용자의 목록에서 사라진다.
    이는 의도된 동작이다(shadow-mount 의미론: 사용자 자신의 디렉터리가 전역 디렉터리를
    가린다).
    """

    def __init__(
        self,
        user_id: str,
        host_path: str | None = None,
        container_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
        app_config=None,
    ) -> None:
        super().__init__(host_path=host_path, container_path=container_path, app_config=app_config)

        from deerflow.config.paths import _validate_user_id, get_paths

        self._user_id = _validate_user_id(user_id)
        paths = get_paths()
        self._paths = paths
        self._user_custom_root: Path = paths.user_custom_skills_dir(self._user_id)
        self._integrations_root: Path = paths.integration_skills_dir()
        self._user_skills_root: Path = paths.user_skills_dir(self._user_id)
        self._global_custom_root: Path = self._host_root / SkillCategory.CUSTOM.value
        self._skill_states_file: Path = self._user_skills_root / "_skill_states.json"

    # ------------------------------------------------------------------
    # 사용자별 skill enabled 상태 (CUSTOM / LEGACY 전용)
    # ------------------------------------------------------------------

    def _read_skill_states(self) -> dict[str, dict[str, bool]]:
        """``_skill_states.json``에서 사용자별 skill enabled 상태를 읽는다.

        skill 이름을 키로 하고 값이 ``{"enabled": True/False}``인 dict를 반환한다.
        파일이 없거나 읽을 수 없으면 빈 dict를 반환한다.
        """
        if not self._skill_states_file.exists():
            return {}
        try:
            with open(self._skill_states_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read skill states file %s", self._skill_states_file)
        return {}

    def _write_skill_states(self, states: dict[str, dict[str, bool]]) -> None:
        """사용자별 skill enabled 상태를 ``_skill_states.json``에 저장한다.

        같은 디렉터리의 temp 파일에 쓴 뒤 ``Path.replace``로 atomic write를 한다(동일
        파일시스템에서 POSIX atomic). 이렇게 하지 않으면 쓰기 도중 crash/SIGTERM/디스크
        가득참이 발생했을 때 파일이 잘리거나 비게 되고, ``_read_skill_states``가 ``{}``를
        반환해 ``get_skill_enabled_state``가 사용자가 비활성화했던 모든 skill을 조용히 다시
        활성화하게 된다. 같은 모듈의 ``LocalSkillStorage.write_custom_skill``과 동일한 패턴이다.
        """
        self._user_skills_root.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self._user_skills_root),
            prefix=".skill_states_",
            suffix=".json.tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(states, f, indent=2)
            tmp_path.replace(self._skill_states_file)
        except Exception:
            # 실패 시 temp 파일을 best-effort로 정리한다.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def get_skill_enabled_state(self, skill_name: str) -> bool:
        """custom/legacy skill의 enabled 상태를 반환한다.

        기본값은 ``True``다(새로 만든 skill은 기본적으로 활성화된다).
        """
        states = self._read_skill_states()
        entry = states.get(skill_name)
        if entry is None:
            return True
        return entry.get("enabled", True)

    def set_skill_enabled_state(self, skill_name: str, enabled: bool) -> None:
        """custom/legacy skill의 enabled 상태를 설정하고 저장한다."""
        removal_names = (skill_name,) if not enabled else ()
        with self._skill_projection_mutation(remove_names=removal_names):
            states = self._read_skill_states()
            states[skill_name] = {"enabled": enabled}
            self._write_skill_states(states)

    # ------------------------------------------------------------------
    # 경로 헬퍼 — custom skill 경로를 사용자 디렉터리로 돌린다
    # ------------------------------------------------------------------

    def get_custom_skill_dir(self, name: str) -> Path:
        """사용자별 custom skill 디렉터리: ``<user_custom_root>/<name>/``."""
        normalized_name = self.validate_skill_name(name)
        return self._user_custom_root / normalized_name

    def get_custom_skill_file(self, name: str) -> Path:
        """사용자별 custom SKILL.md 경로."""
        return self.get_custom_skill_dir(name) / SKILL_MD_FILE

    def get_skill_history_file(self, name: str) -> Path:
        """사용자별 custom skill history: ``<user_custom_root>/.history/<name>.jsonl``."""
        normalized_name = self.validate_skill_name(name)
        return self._user_custom_root / ".history" / f"{normalized_name}.jsonl"

    # ------------------------------------------------------------------
    # enabled 상태 — custom/legacy는 사용자별 상태를 쓰도록 override한다
    # ------------------------------------------------------------------

    def load_skills(self, *, enabled_only: bool = False) -> list:
        """모든 skill을 탐색하고 격리 범위에 맞게 enabled 상태를 병합한다.

        skill 탐색과 PUBLIC enabled 상태는 :meth:`LocalSkillStorage.load_skills`에 위임한다
        (override된 ``_iter_skill_files``에서 읽는다). 그다음 CUSTOM/LEGACY의 enabled 상태를
        사용자별 ``_skill_states.json``으로 덮어써서, 같은 이름의 custom skill을 가진 두
        사용자가 독립적으로 토글할 수 있게 한다.

        ``super().load_skills()``를 호출하면 template method 흐름 전체(탐색 → 전역 enabled
        상태 병합 → 필터 → 정렬)가 유지되므로, 테스트에서 ``LocalSkillStorage.load_skills``를
        패치하면 여전히 이 호출을 가로챌 수 있다.
        """
        # 전체 탐색 + 전역 enabled 상태 병합은 부모에게 맡긴다.
        # override된 _iter_skill_files()가 custom 읽기는 _user_custom_root로,
        # legacy 읽기는 _global_custom_root로 보낸다.
        skills = super().load_skills(enabled_only=False)

        # CUSTOM / LEGACY의 enabled 상태를 사용자별 상태로 덮어쓰되, 전역 extensions_config
        # 기본값과 AND 연산한다. 이렇게 하면 업그레이드 전에 전역에서 비활성화한 공용
        # custom/legacy skill이 사용자별 항목이 없다는 이유로 조용히 다시 활성화되지 않으면서,
        # 둘 다 존재할 때는 사용자별 상태가 전역 기본값을 덮어쓸 수 있다. PUBLIC skill 상태는
        # 오로지 extensions_config가 관장한다(위의 ``super().load_skills``가 처리한다).
        # 사용자 projection을 재구성하는 동안 다른 worker의 갱신이 이 프로세스의 singleton
        # 캐시에 가려지지 않도록 여기서도 디스크에서 다시 읽는다.
        from deerflow.config.extensions_config import ExtensionsConfig

        extensions_config = ExtensionsConfig.from_file()
        skills = [
            dataclasses.replace(s, enabled=self.get_skill_enabled_state(s.name) and extensions_config.is_skill_enabled(s.name, s.category.value if hasattr(s.category, "value") else s.category))
            if dataclasses.is_dataclass(s) and not isinstance(s, type) and (s.category.value if hasattr(s.category, "value") else s.category) != SkillCategory.PUBLIC.value
            else s
            for s in skills
        ]

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        return skills

    # ------------------------------------------------------------------
    # skill 순회 — public은 전역에서, custom은 사용자 디렉터리 + fallback에서
    # ------------------------------------------------------------------

    def public_skill_exists(self, name: str) -> bool:
        """skill이 public **또는** 전역 custom fallback으로 존재하는지 확인한다.

        전역 ``skills/custom/`` 디렉터리에는 사용자별 custom skill이 아직 없는 사용자에게
        ``SkillCategory.LEGACY``로 노출되는 legacy skill이 들어 있다. 이 override 덕분에 그런
        skill이 "read-only"로 인식되어, ``ensure_custom_skill_is_editable``이
        ``FileNotFoundError`` 대신 도움이 되는 에러 메시지를 낼 수 있다.
        """
        normalized_name = self.validate_skill_name(name)
        # 표준 public 검사
        if (self._host_root / SkillCategory.PUBLIC.value / normalized_name / SKILL_MD_FILE).exists():
            return True
        # 전역 custom fallback 검사(모든 사용자에게 보이는 legacy skill)
        if (self._global_custom_root / normalized_name / SKILL_MD_FILE).exists():
            return True
        return False

    def ensure_custom_skill_is_editable(self, name: str) -> None:
        """전역 custom fallback skill을 매끄럽게 처리하기 위한 override.

        사용자가 legacy 전역 custom skill(fallback 때문에 ``SkillCategory.LEGACY``로 보이는
        skill)을 편집/삭제하려 하면, 혼란스러운 ``FileNotFoundError`` 대신 자기만의 버전을
        만들라고 안내한다.
        """
        if self.custom_skill_exists(name):
            return
        # public과 전역 custom fallback을 모두 확인한다
        normalized_name = self.validate_skill_name(name)
        is_global_public = (self._host_root / SkillCategory.PUBLIC.value / normalized_name / SKILL_MD_FILE).exists()
        is_global_custom_fallback = (self._global_custom_root / normalized_name / SKILL_MD_FILE).exists()
        if is_global_public:
            raise ValueError(f"'{name}' is a built-in skill. Use the skill_manage tool to create your own version — it will shadow the built-in one.")
        is_integration = any((candidate / SKILL_MD_FILE).exists() for candidate in self._integrations_root.glob(f"*/{normalized_name}") if candidate.is_dir())
        if is_global_custom_fallback:
            raise ValueError(f"'{name}' is a legacy shared skill (not editable). To customise it, create your own version with the same name — it will shadow the shared one.")
        if is_integration:
            raise ValueError(f"'{name}' is a managed integration skill and cannot be edited. Create a custom skill with another name if you need a modified workflow.")
        raise FileNotFoundError(f"Custom skill '{name}' not found.")

    def _iter_skill_files(self) -> Iterable[tuple[SkillCategory, Path, Path]]:
        # 1. public skill: 항상 전역 root에서 읽는다
        public_path = self._host_root / SkillCategory.PUBLIC.value
        if public_path.exists() and public_path.is_dir():
            for current_root, dir_names, file_names in os.walk(public_path, followlinks=True):
                dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
                if SKILL_MD_FILE not in file_names:
                    continue
                dir_names.clear()
                yield SkillCategory.PUBLIC, public_path, Path(current_root) / SKILL_MD_FILE

        # 2. 관리형 integration skill: 전역 설치, read-only. enabled 상태는 여전히 이
        # 사용자의 _skill_states.json에서 병합한다.
        integration_path = self._integrations_root
        if integration_path.exists() and integration_path.is_dir():
            for current_root, dir_names, file_names in os.walk(integration_path, followlinks=True):
                dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
                if SKILL_MD_FILE not in file_names:
                    continue
                yield SkillCategory.INTEGRATION, integration_path, Path(current_root) / SKILL_MD_FILE

        # 3. custom skill: 사용자 수준 디렉터리를 우선한다
        user_custom_exists = False
        user_custom_path = self._user_custom_root
        if user_custom_path.exists() and user_custom_path.is_dir():
            for current_root, dir_names, file_names in os.walk(user_custom_path, followlinks=True):
                dir_names[:] = sorted(name for name in dir_names if not name.startswith(".") and name != ".history")
                if SKILL_MD_FILE not in file_names:
                    continue
                dir_names.clear()
                user_custom_exists = True
                yield SkillCategory.CUSTOM, user_custom_path, Path(current_root) / SKILL_MD_FILE

        # 4. fallback: 사용자에게 custom skill이 없으면 전역 custom을 LEGACY(read-only)로
        #    로드해서, legacy skill이 보이되 사용자가 편집/삭제할 수는 없게 한다. LEGACY
        #    skill은 sandbox의 /mnt/skills/legacy/<name>/에 마운트되므로 관련 파일
        #    (references, templates, scripts, assets)에 접근할 수 있다.
        if not user_custom_exists:
            global_custom_path = self._global_custom_root
            if global_custom_path.exists() and global_custom_path.is_dir():
                for current_root, dir_names, file_names in os.walk(global_custom_path, followlinks=True):
                    dir_names[:] = sorted(name for name in dir_names if not name.startswith(".") and name != ".history")
                    if SKILL_MD_FILE not in file_names:
                        continue
                    dir_names.clear()
                    yield SkillCategory.LEGACY, global_custom_path, Path(current_root) / SKILL_MD_FILE

    # ------------------------------------------------------------------
    # 설치 — custom_dir을 사용자 디렉터리로 돌린다
    # ------------------------------------------------------------------

    async def ainstall_skill_from_archive(self, archive_path: str | Path) -> dict:
        from deerflow.skills.installer import _scan_skill_archive_contents_or_raise

        logger.info("Installing skill from %s for user %s", archive_path, self._user_id)
        path = Path(archive_path)
        custom_dir = self._user_custom_root

        # 사용자 custom 디렉터리가 있는지 확인한다
        custom_dir.mkdir(parents=True, exist_ok=True)

        # 파일별 보안 스캔은 async LLM 호출이라 event loop에 남아야 한다. 그 주변의 모든
        # 파일시스템 단계는 worker thread에서 실행한다.
        tmp = await asyncio.to_thread(tempfile.mkdtemp)
        try:
            skill_dir, skill_name, target = await asyncio.to_thread(self._prepare_skill_archive, path, Path(tmp), custom_dir, archive_path)

            await _scan_skill_archive_contents_or_raise(skill_dir, skill_name)

            await asyncio.to_thread(self._commit_skill_install, skill_dir, skill_name, custom_dir, target)
            logger.info("Skill %r installed to %s for user %s", skill_name, target, self._user_id)
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._cleanup_install_tmp, tmp),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning("Timed out cleaning up skill install temp dir %s", tmp)

        return {
            "success": True,
            "skill_name": skill_name,
            "message": f"Skill '{skill_name}' installed successfully for user '{self._user_id}'",
        }

    # ------------------------------------------------------------------
    # 쓰기 — 쓰기 전에 사용자 custom 디렉터리를 보장한다
    # ------------------------------------------------------------------

    def write_custom_skill(self, name: str, relative_path: str, content: str) -> None:
        # 사용자 custom skill 디렉터리가 있는지 확인한다
        self._user_custom_root.mkdir(parents=True, exist_ok=True)
        target = self.validate_relative_path(relative_path, self.get_custom_skill_dir(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(target.parent),
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        try:
            with self._skill_projection_mutation():
                tmp_path.replace(target)
                make_skill_written_path_sandbox_readable(self.get_custom_skill_dir(name), target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # 공개 헬퍼
    # ------------------------------------------------------------------

    @property
    def user_id(self) -> str:
        """이 storage가 대상으로 하는 user ID."""
        return self._user_id

    def get_user_custom_root(self) -> Path:
        """이 사용자의 custom skill root 디렉터리 host 경로."""
        return self._user_custom_root

    def get_integrations_root(self) -> Path:
        """전역 관리형 integration skill root 디렉터리 host 경로."""
        return self._integrations_root

    def get_user_integrations_root(self) -> Path:
        """:meth:`get_integrations_root`의 호환용 별칭."""
        return self.get_integrations_root()

    # ------------------------------------------------------------------
    # 경로 검증 — public, 사용자별 custom, integration root를 허용한다
    # ------------------------------------------------------------------

    def validate_skill_file_path(self, skill_file: Path) -> Path:
        """public, 사용자별 custom, integration root 아래의 파일을 허용한다.

        custom과 관리형 integration skill은 ``_host_root`` 밖에 있으므로, 기본 구현의
        단일 root 검사로는 거부된다.
        """
        resolved_file = skill_file.resolve()
        allowed_roots = (
            self._host_root.resolve(),
            self._user_custom_root.resolve(),
            self._integrations_root.resolve(),
        )
        for allowed_root in allowed_roots:
            try:
                resolved_file.relative_to(allowed_root)
                return resolved_file
            except ValueError:
                continue
        raise ValueError(
            f"Resolved skill file {resolved_file} must stay within the global skills root "
            f"({self._host_root.resolve()}), the per-user custom root "
            f"({self._user_custom_root.resolve()}), or the managed integration skills root "
            f"({self._integrations_root.resolve()})."
        )
