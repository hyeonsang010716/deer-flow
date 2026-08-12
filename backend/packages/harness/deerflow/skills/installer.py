"""공용 skill 아카이브 설치 로직.

FastAPI/HTTP 의존성이 없는 순수 비즈니스 로직이다.
Gateway와 Client 모두 이 함수들에 위임한다.
"""

import asyncio
import concurrent.futures
import logging
import posixpath
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from deerflow.skills.permissions import make_skill_tree_sandbox_readable
from deerflow.skills.security_scanner import scan_skill_content
from deerflow.skills.security_static_scanner import (
    StaticFinding,
    StaticScanBlockedError,
    StaticScannerError,
    enforce_static_scan,
    scan_archive_preflight,
    skill_scan_enabled,
)

logger = logging.getLogger(__name__)

_PROMPT_INPUT_DIRS = {"references", "templates"}
_PROMPT_INPUT_SUFFIXES = frozenset({".json", ".markdown", ".md", ".rst", ".txt", ".yaml", ".yml"})
_CODE_SUFFIXES = frozenset({".bash", ".cjs", ".js", ".mjs", ".php", ".pl", ".ps1", ".py", ".rb", ".sh", ".ts", ".zsh"})
# 변종마다 전체 magic을 적는다. 더 짧은 공통 접두사를 쓰면 실행 파일이 아닌 데이터 파일까지
# 매칭된다.
_EXECUTABLE_MAGIC_PREFIXES = (
    b"\x7fELF",  # ELF
    b"MZ",  # PE/DOS
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit big-endian
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit big-endian
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit little-endian
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit little-endian
    b"\xca\xfe\xba\xbe",  # Mach-O fat binary big-endian
    b"\xbe\xba\xfe\xca",  # Mach-O fat binary little-endian
    b"\xca\xfe\xba\xbf",  # Mach-O fat64 binary big-endian
    b"\xbf\xba\xfe\xca",  # Mach-O fat64 binary little-endian
)


class SkillAlreadyExistsError(ValueError):
    """같은 이름의 skill이 이미 설치돼 있을 때 raise된다."""


class SkillSecurityScanError(ValueError):
    """skill 아카이브가 보안 스캔을 통과하지 못했을 때 raise된다."""

    findings: list[StaticFinding]
    skill_name: str | None

    def __init__(self, message: str, *, findings: list[StaticFinding] | None = None, skill_name: str | None = None) -> None:
        super().__init__(message)
        self.findings = [dict(finding) for finding in (findings or [])]
        self.skill_name = skill_name


def is_unsafe_zip_member(info: zipfile.ZipInfo) -> bool:
    """zip member 경로가 절대 경로거나, directory traversal을 시도하거나, 콜론을 포함하면
    True를 반환한다.

    상대 아카이브 member 경로에서 콜론은 정당한 쓰임이 없다. zip 엔트리는 항상 ``/``
    구분자를 쓰고, 실제 Windows 드라이브 접두사(``C:\\...``)는 위에서 이미 절대 경로로
    거부된다. 그런데 Windows/NTFS에서는 경로 다른 위치의 콜론(예:
    ``scripts/run.sh:hidden.txt``)이 새 파일을 만드는 대신 앞선 경로 구성 요소의 Alternate
    Data Stream을 가리킨다. 즉 형제 파일을 만드는 게 아니라 ``scripts/run.sh``에 조용히
    내용을 덧붙인다. 이 stream은 ``Path.rglob()`` / ``os.walk()`` 기반 목록에 보이지 않으므로,
    아카이브가 디렉터리 기반 보안 스캔을 우회해 내용을 디스크에 심을 수 있다. "안전한" 콜론
    위치를 allow-list하려 하지 말고 곧바로 거부한다.
    """
    name = info.filename
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return True
    if PureWindowsPath(name).is_absolute():
        return True
    if ".." in path.parts:
        return True
    if ":" in name:
        return True
    return False


def is_symlink_member(info: zipfile.ZipInfo) -> bool:
    """ZipInfo에 저장된 external attribute로 symlink를 감지한다."""
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def is_executable_binary_prefix(prefix: bytes) -> bool:
    """magic byte로 ELF, PE, Mach-O 실행 파일을 감지한다."""
    return prefix.startswith(_EXECUTABLE_MAGIC_PREFIXES)


def should_ignore_archive_entry(path: Path) -> bool:
    """macOS 메타데이터 디렉터리와 dotfile이면 True를 반환한다."""
    return path.name.startswith(".") or path.name == "__MACOSX"


def resolve_skill_dir_from_archive(temp_path: Path) -> Path:
    """추출된 아카이브 내용에서 skill 루트 디렉터리를 찾는다.

    macOS 메타데이터(__MACOSX)와 dotfile(.DS_Store)은 걸러낸다.

    Returns:
        skill 디렉터리 경로.

    Raises:
        ValueError: 필터링 후 아카이브가 비어 있을 때.
    """
    items = [p for p in temp_path.iterdir() if not should_ignore_archive_entry(p)]
    if not items:
        raise ValueError("Skill archive is empty")
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return temp_path


def safe_extract_skill_archive(
    zip_ref: zipfile.ZipFile,
    dest_path: Path,
    max_total_size: int = 512 * 1024 * 1024,
    max_entries: int = 4096,
) -> None:
    """보안 보호 장치를 적용해 skill 아카이브를 안전하게 추출한다.

    보호 장치:
    - 절대 경로와 directory traversal(..)을 거부한다.
    - symlink 엔트리는 실제로 만들지 않고 건너뛴다.
    - 압축 해제 총 크기에 하드 리밋을 강제한다(zip bomb 방어).
    - member 개수에 하드 리밋을 강제한다(엔트리 수 기준 zip bomb 방어. 아주 작거나 빈
      member가 엄청나게 많으면 저장 비용은 싸지만 총 크기와 무관하게 추출이 느려진다).
    - magic byte로 실행 바이너리(ELF/PE/Mach-O)를 거부한다.

    Raises:
        ValueError: 안전하지 않은 member, 실행 바이너리, 엔트리 수 또는 크기 제한 초과 시.
    """
    dest_root = dest_path.resolve()
    total_written = 0

    infos = zip_ref.infolist()
    if len(infos) > max_entries:
        # 아래 member별 작업을 시작하기 전에 조기 중단한다.
        # skillscan/orchestrator.py::scan_archive_preflight의 조기 중단과 같은 방식이다
        # (그쪽 주석: "총 크기가 작아도 엄청난 member 개수는 유한한 DoS 벡터다"). 그 스캔은
        # 선택적이지만(skill_scan.enabled), 이 검사는 모든 설치가 거치는 추출 경로에 있으므로
        # 무조건 적용돼야 한다.
        raise ValueError(f"Skill archive contains too many entries ({len(infos)} > {max_entries}).")

    for info in infos:
        if is_unsafe_zip_member(info):
            raise ValueError(f"Archive contains unsafe member path: {info.filename!r}")

        if is_symlink_member(info):
            logger.warning("Skipping symlink entry in skill archive: %s", info.filename)
            continue

        normalized_name = posixpath.normpath(info.filename.replace("\\", "/"))
        member_path = dest_root.joinpath(*PurePosixPath(normalized_name).parts)
        if not member_path.resolve().is_relative_to(dest_root):
            raise ValueError(f"Zip entry escapes destination: {info.filename!r}")
        member_path.parent.mkdir(parents=True, exist_ok=True)

        if info.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue

        with zip_ref.open(info) as src, member_path.open("wb") as dst:
            first_chunk = True
            while chunk := src.read(65536):
                if first_chunk and is_executable_binary_prefix(chunk):
                    raise ValueError(f"Archive contains executable binary member: {info.filename!r}")
                first_chunk = False
                total_written += len(chunk)
                if total_written > max_total_size:
                    raise ValueError("Skill archive is too large or appears highly compressed.")
                dst.write(chunk)


def _is_script_support_file(rel_path: Path) -> bool:
    return bool(rel_path.parts) and rel_path.parts[0] == "scripts"


def _should_scan_support_file(rel_path: Path) -> bool:
    if _is_script_support_file(rel_path):
        return True
    return bool(rel_path.parts) and rel_path.parts[0] in _PROMPT_INPUT_DIRS and rel_path.suffix.lower() in _PROMPT_INPUT_SUFFIXES


def _has_shebang(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return False


def _is_code_file_by_name(rel_path: Path) -> bool:
    """이름만으로 코드 파일을 분류한다. scripts/ 하위 member와 코드 확장자가 대상이다."""
    if _is_script_support_file(rel_path):
        return True
    return rel_path.suffix.lower() in _CODE_SUFFIXES


async def _is_code_file(path: Path, rel_path: Path) -> bool:
    """executable 스캔 정책을 위해 트리 어디에 있든 코드 파일을 분류한다.

    이름 검사는 순수하므로 event loop에 남겨 두고, 확장자 없는 파일의 shebang 확인만 파일을
    읽으므로 offload한다.
    """
    if _is_code_file_by_name(rel_path):
        return True
    return not rel_path.suffix and await asyncio.to_thread(_has_shebang, path)


def _move_staged_skill_into_reserved_target(staging_target: Path, target: Path) -> None:
    installed = False
    reserved = False
    try:
        target.mkdir(mode=0o700)
        reserved = True
        for child in staging_target.iterdir():
            shutil.move(str(child), target / child.name)
        make_skill_tree_sandbox_readable(target)
        installed = True
    except FileExistsError as e:
        raise SkillAlreadyExistsError(f"Skill '{target.name}' already exists") from e
    finally:
        if reserved and not installed and target.exists():
            shutil.rmtree(target)


def _findings_for_file(findings: list[StaticFinding], rel_path: str) -> list[StaticFinding]:
    return [finding for finding in findings if finding.get("file") in {rel_path, None}]


async def _scan_skill_file_or_raise(skill_dir: Path, path: Path, skill_name: str, *, executable: bool, static_findings: list[StaticFinding] | None = None) -> None:
    rel_path = path.relative_to(skill_dir).as_posix()
    location = f"{skill_name}/{rel_path}"
    try:
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SkillSecurityScanError(f"Security scan failed for skill '{skill_name}': {location} must be valid UTF-8") from e

    try:
        result = await scan_skill_content(content, executable=executable, location=location, static_findings=static_findings or [])
    except Exception as e:
        raise SkillSecurityScanError(f"Security scan failed for {location}: {e}") from e

    decision = getattr(result, "decision", None)
    reason = str(getattr(result, "reason", "") or "No reason provided.")
    if decision == "block":
        if rel_path == "SKILL.md":
            raise SkillSecurityScanError(f"Security scan blocked skill '{skill_name}': {reason}")
        raise SkillSecurityScanError(f"Security scan blocked {location}: {reason}")
    if executable and decision != "allow":
        raise SkillSecurityScanError(f"Security scan rejected executable {location}: {reason}")
    if decision not in {"allow", "warn"}:
        raise SkillSecurityScanError(f"Security scan failed for {location}: invalid scanner decision {decision!r}")


def scan_archive_preflight_or_raise(archive_path: Path, *, app_config=None) -> None:
    if not skill_scan_enabled(app_config):
        return
    result = scan_archive_preflight(archive_path)
    if result["blocked"]:
        critical = [finding for finding in result["findings"] if finding["severity"] == "CRITICAL"]
        raise SkillSecurityScanError(
            f"Static security scan blocked unsafe skill archive: {format_static_archive_findings(critical)}",
            findings=critical,
            skill_name=None,
        )


def format_static_archive_findings(findings: list[StaticFinding]) -> str:
    return "; ".join(f"{finding['rule_id']} ({finding['severity']}) at {finding.get('file') or '<archive>'}: {finding['message']}" for finding in findings)


async def _scan_static_skill_archive_or_raise(skill_dir: Path, skill_name: str, *, app_config=None) -> list[StaticFinding]:
    try:
        return await asyncio.to_thread(enforce_static_scan, skill_dir, skill_name=skill_name, app_config=app_config)
    except StaticScanBlockedError as e:
        raise SkillSecurityScanError(str(e), findings=e.findings, skill_name=e.skill_name) from e
    except StaticScannerError as e:
        raise SkillSecurityScanError(f"Static security scan failed for skill '{skill_name}': {e}", skill_name=skill_name) from e


def _collect_scannable_files(skill_dir: Path) -> list[Path]:
    """스캔 대상 아카이브 파일을 열거한다(blocking이므로 event loop 밖에서 실행한다)."""
    return [candidate for candidate in sorted(skill_dir.rglob("*")) if candidate.is_file()]


async def _scan_skill_archive_contents_or_raise(skill_dir: Path, skill_name: str, *, app_config=None) -> list[StaticFinding]:
    """설치 대상 텍스트·스크립트 파일 전부에 skill 보안 스캐너를 실행한다."""
    static_findings = await _scan_static_skill_archive_or_raise(skill_dir, skill_name, app_config=app_config)

    skill_md = skill_dir / "SKILL.md"
    await _scan_skill_file_or_raise(skill_dir, skill_md, skill_name, executable=False, static_findings=_findings_for_file(static_findings, "SKILL.md"))

    for path in await asyncio.to_thread(_collect_scannable_files, skill_dir):
        rel_path = path.relative_to(skill_dir)
        if rel_path == Path("SKILL.md"):
            continue
        if path.name == "SKILL.md":
            raise SkillSecurityScanError(f"Security scan failed for skill '{skill_name}': nested SKILL.md is not allowed at {skill_name}/{rel_path.as_posix()}")
        rel_path_posix = rel_path.as_posix()
        if await _is_code_file(path, rel_path):
            await _scan_skill_file_or_raise(
                skill_dir,
                path,
                skill_name,
                executable=True,
                static_findings=_findings_for_file(static_findings, rel_path_posix),
            )
        elif _should_scan_support_file(rel_path):
            await _scan_skill_file_or_raise(
                skill_dir,
                path,
                skill_name,
                executable=False,
                static_findings=_findings_for_file(static_findings, rel_path_posix),
            )
    return static_findings


def _run_async_install(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
