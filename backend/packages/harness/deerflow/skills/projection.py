"""sandbox 파일시스템에 노출할, 활성 skill만 담은 tree를 materialize한다."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deerflow.skills.parser import parse_skill_file
from deerflow.skills.types import SKILL_MD_FILE, Skill, SkillCategory

if TYPE_CHECKING:
    from deerflow.skills.storage.skill_storage import SkillStorage

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt

_locks_guard = threading.Lock()
_process_locks: dict[Path, threading.RLock] = {}
_MANIFEST_VERSION = 1
_MAX_REBUILD_ATTEMPTS = 2


@dataclass(frozen=True)
class SkillProjectionPaths:
    """sandbox provider가 mount하거나 업로드하는 안정적인 category root들."""

    public: Path
    custom: Path
    legacy: Path
    integrations: Path


def get_skill_projection_paths(storage: SkillStorage) -> SkillProjectionPaths:
    from deerflow.config.paths import get_paths

    paths = getattr(storage, "_paths", None) or get_paths()
    user_id = getattr(storage, "user_id", None)
    if user_id is None:
        return SkillProjectionPaths(
            public=paths.public_skills_view_dir,
            custom=paths.skills_view_dir / "custom",
            legacy=paths.skills_view_dir / "legacy",
            integrations=paths.skills_view_dir / "integrations",
        )
    return SkillProjectionPaths(
        public=paths.public_skills_view_dir,
        custom=paths.user_custom_skills_view_dir(user_id),
        legacy=paths.user_legacy_skills_view_dir(user_id),
        integrations=paths.user_integration_skills_view_dir(user_id),
    )


def _lock_for(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _locks_guard:
        return _process_locks.setdefault(resolved, threading.RLock())


@contextmanager
def _projection_lock(root: Path) -> Iterator[None]:
    """projection 교체를 프로세스 내부와 POSIX worker 간에 직렬화한다."""
    lock_path = root.parent / f".{root.name}.projection.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _lock_for(lock_path)
    with process_lock, lock_path.open("a", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _link_or_copy(source: str, target: str, *, follow_symlinks: bool = True) -> str:
    # hardlink는 원본 inode를 공유하므로 write 격리를 제공하지 않는다. 읽기 전용 보장은
    # 이를 소비하는 sandbox나 mount에서 와야 한다.
    try:
        os.link(source, target, follow_symlinks=follow_symlinks)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.copy2(source, target, follow_symlinks=follow_symlinks)
    return target


def _stage_skill(source: Path, target: Path, nested_skill_roots: set[Path]) -> None:
    def _exclude_nested_skills(current: str, names: list[str]) -> list[str]:
        relative_root = Path(current).relative_to(source)
        return [name for name in names if relative_root / name in nested_skill_roots]

    shutil.copytree(
        source,
        target,
        copy_function=_link_or_copy,
        symlinks=True,
        ignore=_exclude_nested_skills,
        dirs_exist_ok=True,
    )


def _path_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    return "file"


def _tree_entries(root: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in dir_names:
            path = current / name
            entries[path.relative_to(root)] = _path_kind(path)
        dir_names[:] = [name for name in dir_names if not (current / name).is_symlink()]
        for name in file_names:
            path = current / name
            entries[path.relative_to(root)] = _path_kind(path)
    return entries


def _remove_projection_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _validate_projection_relative_path(relative_path: Path) -> None:
    if relative_path.is_absolute() or not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("Projection removal path must identify a package within its category root")


def _remove_projection_relative(root: Path, relative_path: Path) -> None:
    """어긋난 namespace symlink를 따라가지 않고 projection된 package를 제거한다."""
    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            current.unlink()
            return
    _remove_projection_entry(current)


def _sync_staged_category(root: Path, staging: Path) -> None:
    desired = _tree_entries(staging)
    live = _tree_entries(root)

    for relative_path, live_kind in sorted(live.items(), key=lambda item: len(item[0].parts), reverse=True):
        if desired.get(relative_path) != live_kind:
            _remove_projection_entry(root / relative_path)

    for relative_path, kind in sorted(desired.items(), key=lambda item: len(item[0].parts)):
        if kind == "directory":
            (root / relative_path).mkdir(parents=True, exist_ok=True)

    for relative_path, kind in desired.items():
        if kind == "directory":
            continue
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / relative_path).replace(target)


def _replace_category(root: Path, desired: dict[Path, Skill], skill_boundaries: set[Path]) -> None:
    """안정적인 category root를 비우지 않고 그 아래 항목들을 정합화한다."""
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.projection-", dir=root.parent) as staging_dir:
        staging = Path(staging_dir)
        for relative_path, skill in desired.items():
            nested_roots = {boundary.relative_to(relative_path) for boundary in skill_boundaries if boundary != relative_path and boundary.is_relative_to(relative_path)}
            _stage_skill(skill.skill_dir, staging / relative_path, nested_roots)
        _sync_staged_category(root, staging)


def _clear_category(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _clear_projection_scope(scope_root: Path, *category_roots: Path) -> None:
    for category_root in category_roots:
        _clear_category(category_root)
    _manifest_path(scope_root).unlink(missing_ok=True)


def _update_tree_digest(digest, root: Path, label: str) -> None:
    """파일 내용이 아니라 디렉터리 metadata(inode/mode/size/mtime)를 해시한다.

    trade-off: sandbox acquire마다 돌려도 될 만큼 빠르지만(O(파일 수), 읽기 없음), inode+size+
    mtime을 그대로 유지하는 외부 수정은 — 드물지만 확률이 0은 아니다 — 이 signature에 잡히지
    않아 다음 명시적 rebuild 전까지 projection이 낡은 상태로 남는다. 이 코드베이스를 통한
    runtime write는 어느 경우든 커버된다. mutation 경로는 lock을 잡고 rebuild하고, atomic
    rename은 항상 inode를 바꾸기 때문이다.
    """
    digest.update(f"root:{label}\0".encode())
    if not root.exists():
        digest.update(b"absent\0")
        return

    stack = [(root, Path("."))]
    while stack:
        current, relative_root = stack.pop()
        with os.scandir(current) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        child_dirs: list[tuple[Path, Path]] = []
        for entry in ordered:
            relative = relative_root / entry.name
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink():
                kind = "link"
            elif entry.is_dir(follow_symlinks=False):
                kind = "dir"
                child_dirs.append((Path(entry.path), relative))
            else:
                kind = "file"
            digest.update((f"{label}:{relative.as_posix()}:{kind}:{metadata.st_ino}:{metadata.st_mode}:{metadata.st_size}:{metadata.st_mtime_ns}\0").encode())
        stack.extend(reversed(child_dirs))


def _extensions_state() -> dict:
    from deerflow.config.extensions_config import ExtensionsConfig

    config = ExtensionsConfig.from_file()
    return {name: state.model_dump(mode="json") for name, state in config.skills.items()}


def _source_signature(storage: SkillStorage, scope: str) -> str:
    digest = hashlib.sha256()
    host_root = storage.get_skills_root_path()
    if scope == "public":
        _update_tree_digest(digest, host_root / SkillCategory.PUBLIC.value, "public")
        state = {"extensions": _extensions_state()}
    elif scope == "user":
        user_custom_root = storage.get_user_custom_root()
        integration_root = storage.get_user_integrations_root()
        _update_tree_digest(digest, user_custom_root, "custom")
        _update_tree_digest(digest, host_root / SkillCategory.CUSTOM.value, "legacy")
        _update_tree_digest(digest, integration_root, "integrations")
        # CUSTOM/LEGACY/INTEGRATION의 가시성은 사용자별 state와 전역 extensions 기본값의
        # 교집합이므로, 둘 다 이 signature에 들어가야 한다.
        state = {
            "extensions": _extensions_state(),
            "user": storage._read_skill_states(),
        }
    else:  # pragma: no cover - 내부 불변식
        raise ValueError(f"Unknown skill projection scope: {scope}")
    digest.update(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _manifest_path(scope_root: Path) -> Path:
    return scope_root / ".projection-manifest.json"


def _read_manifest(scope_root: Path) -> dict | None:
    try:
        value = json.loads(_manifest_path(scope_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(scope_root: Path, source_signature: str) -> None:
    scope_root.mkdir(parents=True, exist_ok=True)
    target = _manifest_path(scope_root)
    fd, temporary_name = tempfile.mkstemp(prefix=".projection-manifest-", suffix=".tmp", dir=scope_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"version": _MANIFEST_VERSION, "source_signature": source_signature}, stream, sort_keys=True)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_public_skills(storage: SkillStorage, *, enabled_only: bool) -> list[Skill]:
    from deerflow.config.extensions_config import ExtensionsConfig

    public_root = storage.get_skills_root_path() / SkillCategory.PUBLIC.value
    if not public_root.is_dir():
        return []
    extensions = ExtensionsConfig.from_file()
    skills: list[Skill] = []
    for current_root, dir_names, file_names in os.walk(public_root, followlinks=True):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
        if SKILL_MD_FILE not in file_names:
            continue
        # runtime loader와 동작을 맞춘다. package 안에 중첩된 SKILL.md는 독립적으로 설정
        # 가능한 skill이 아니라 보조 데이터다.
        dir_names.clear()
        skill_file = Path(current_root) / SKILL_MD_FILE
        skill = parse_skill_file(
            skill_file,
            category=SkillCategory.PUBLIC,
            relative_path=skill_file.parent.relative_to(public_root),
        )
        if skill is None:
            continue
        enabled = extensions.is_skill_enabled(skill.name, SkillCategory.PUBLIC.value)
        if not enabled_only or enabled:
            skills.append(skill)
    return skills


def _by_relative_path(skills: list[Skill], category: SkillCategory) -> dict[Path, Skill]:
    return {skill.relative_path: skill for skill in skills if skill.category == category}


def _category_boundaries(skills: list[Skill], category: SkillCategory) -> set[Path]:
    return {skill.relative_path for skill in skills if skill.category == category}


def _rebuild_public_locked(storage: SkillStorage, paths: SkillProjectionPaths) -> None:
    scope_root = paths.public.parent
    try:
        for _attempt in range(_MAX_REBUILD_ATTEMPTS):
            before = _source_signature(storage, "public")
            all_public_skills = _load_public_skills(storage, enabled_only=False)
            enabled_public_skills = _load_public_skills(storage, enabled_only=True)
            _replace_category(
                paths.public,
                _by_relative_path(enabled_public_skills, SkillCategory.PUBLIC),
                _category_boundaries(all_public_skills, SkillCategory.PUBLIC),
            )
            after = _source_signature(storage, "public")
            if before == after:
                _write_manifest(scope_root, after)
                return
        raise RuntimeError("Public skills changed repeatedly while rebuilding the sandbox projection")
    except Exception:
        _clear_projection_scope(scope_root, paths.public)
        raise


def _rebuild_user_locked(storage: SkillStorage, paths: SkillProjectionPaths) -> None:
    scope_root = paths.custom.parent
    try:
        for _attempt in range(_MAX_REBUILD_ATTEMPTS):
            before = _source_signature(storage, "user")
            all_user_skills = storage.load_skills(enabled_only=False)
            enabled_user_skills = [skill for skill in all_user_skills if skill.enabled]
            _replace_category(
                paths.custom,
                _by_relative_path(enabled_user_skills, SkillCategory.CUSTOM),
                _category_boundaries(all_user_skills, SkillCategory.CUSTOM),
            )
            _replace_category(
                paths.legacy,
                _by_relative_path(enabled_user_skills, SkillCategory.LEGACY),
                _category_boundaries(all_user_skills, SkillCategory.LEGACY),
            )
            _replace_category(
                paths.integrations,
                _by_relative_path(enabled_user_skills, SkillCategory.INTEGRATION),
                _category_boundaries(all_user_skills, SkillCategory.INTEGRATION),
            )
            after = _source_signature(storage, "user")
            if before == after:
                _write_manifest(scope_root, after)
                return
        raise RuntimeError("User skills changed repeatedly while rebuilding the sandbox projection")
    except Exception:
        _clear_projection_scope(scope_root, paths.custom, paths.legacy, paths.integrations)
        raise


def rebuild_skill_projections(
    storage: SkillStorage,
    *,
    include_public: bool = True,
    include_user: bool = True,
) -> SkillProjectionPaths:
    """``storage``를 통해 보이는, 활성 skill만 담은 projection scope를 rebuild한다."""
    paths = get_skill_projection_paths(storage)
    user_id = getattr(storage, "user_id", None)
    if include_public:
        with _projection_lock(paths.public.parent):
            _rebuild_public_locked(storage, paths)
    if include_user and user_id is not None:
        with _projection_lock(paths.custom.parent):
            _rebuild_user_locked(storage, paths)

    return paths


def _public_projection_is_fresh(storage: SkillStorage, paths: SkillProjectionPaths) -> bool:
    if not paths.public.is_dir():
        return False
    manifest_before = _read_manifest(paths.public.parent)
    if manifest_before is None or manifest_before.get("version") != _MANIFEST_VERSION:
        return False
    signature = _source_signature(storage, "public")
    manifest_after = _read_manifest(paths.public.parent)
    return manifest_before == manifest_after and manifest_before.get("source_signature") == signature


def ensure_skill_projections(storage: SkillStorage) -> SkillProjectionPaths:
    """낡은 projection scope만 복구하고, 그 외에는 inode를 건드리지 않는다."""
    paths = get_skill_projection_paths(storage)

    try:
        public_is_fresh = _public_projection_is_fresh(storage, paths)
    except Exception:
        # fail-closed하기 전에 mutation lock을 잡고 다시 확인한다. 동시 writer가 일시적인
        # source/manifest 상태를 노출했을 수 있다.
        public_is_fresh = False
    if not public_is_fresh:
        with _projection_lock(paths.public.parent):
            try:
                if not _public_projection_is_fresh(storage, paths):
                    _rebuild_public_locked(storage, paths)
            except Exception:
                _clear_projection_scope(paths.public.parent, paths.public)
                raise

    if getattr(storage, "user_id", None) is not None:
        with _projection_lock(paths.custom.parent):
            try:
                manifest = _read_manifest(paths.custom.parent)
                signature = _source_signature(storage, "user")
                if not paths.custom.is_dir() or not paths.legacy.is_dir() or not paths.integrations.is_dir() or manifest is None or manifest.get("version") != _MANIFEST_VERSION or manifest.get("source_signature") != signature:
                    _rebuild_user_locked(storage, paths)
            except Exception:
                _clear_projection_scope(paths.custom.parent, paths.custom, paths.legacy, paths.integrations)
                raise
    return paths


@contextmanager
def skill_projection_mutation(
    storage: SkillStorage,
    scope: str,
    *,
    remove: tuple[tuple[SkillCategory, Path], ...] = (),
    remove_names: tuple[str, ...] = (),
) -> Iterator[None]:
    """source/state 변경 동안 projection scope lock을 유지한다."""
    if not isinstance(storage.get_skills_root_path(), Path):
        # 가벼운 단위 테스트 double이 여기서 MagicMock을 반환하기도 한다. SkillStorage 계약은
        # Path를 요구하므로, 실제 storage 구현은 이 호환용 분기를 타지 않는다.
        yield
        return
    paths = get_skill_projection_paths(storage)
    if scope == "public":
        scope_root = paths.public.parent
        category_roots = {SkillCategory.PUBLIC: paths.public}

        def rebuild() -> None:
            _rebuild_public_locked(storage, paths)

    elif scope == "user":
        scope_root = paths.custom.parent
        category_roots = {
            SkillCategory.CUSTOM: paths.custom,
            SkillCategory.LEGACY: paths.legacy,
            SkillCategory.INTEGRATION: paths.integrations,
        }

        def rebuild() -> None:
            _rebuild_user_locked(storage, paths)

    else:
        raise ValueError(f"Unknown skill projection scope: {scope}")

    removals: set[tuple[Path, Path]] = set()
    for category, relative_path in remove:
        root = category_roots.get(category)
        if root is None:
            raise ValueError(f"Skill category {category.value!r} does not belong to projection scope {scope!r}")
        _validate_projection_relative_path(relative_path)
        removals.add((root, relative_path))

    with _projection_lock(scope_root):
        if remove_names:
            names = set(remove_names)
            skills = _load_public_skills(storage, enabled_only=False) if scope == "public" else storage.load_skills(enabled_only=False)
            for skill in skills:
                root = category_roots.get(skill.category)
                if skill.name not in names or root is None:
                    continue
                _validate_projection_relative_path(skill.relative_path)
                removals.add((root, skill.relative_path))

        try:
            _manifest_path(scope_root).unlink(missing_ok=True)
            for root, relative_path in removals:
                _remove_projection_relative(root, relative_path)
            yield
            rebuild()
        except Exception:
            _clear_projection_scope(scope_root, *category_roots.values())
            raise


def ensure_public_skill_projection(*, app_config=None) -> bool:
    """사용자 데이터를 스캔하지 않고 boot 시점에 전역 public view를 보장한다.

    사용자 projection은 sandbox acquire가 지연 복구한다. 과거 사용자를 전부 미리 rebuild하면
    gateway readiness가 테넌트 수에 비례해 느려지는 데다, 해당 사용자의 다음 acquire 전까지
    추가 안전성도 없다.
    """
    from deerflow.config import get_app_config
    from deerflow.config.paths import get_paths
    from deerflow.skills.storage import get_or_new_skill_storage

    try:
        config = app_config or get_app_config()
        public_storage = get_or_new_skill_storage(app_config=config)
        ensure_skill_projections(public_storage)
    except Exception:
        logger.warning("Failed to ensure the public skill projection during boot; clearing it until a sandbox acquire self-heals it", exc_info=True)
        try:
            paths = get_paths()
            with _projection_lock(paths.public_skills_view_dir.parent):
                _clear_projection_scope(paths.public_skills_view_dir.parent, paths.public_skills_view_dir)
        except Exception:
            logger.error("Failed to clear the public skill projection after a boot-time error", exc_info=True)
        return False
    return True
