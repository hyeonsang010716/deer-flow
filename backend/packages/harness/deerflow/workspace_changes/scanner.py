from __future__ import annotations

import fnmatch
import hashlib
import os
from codecs import BOM_UTF16_BE, BOM_UTF16_LE, getincrementaldecoder
from pathlib import Path

from deerflow.constants import BROWSER_FRAMES_DIRNAME, TOOL_RESULTS_DIRNAME

from .types import (
    DiffUnavailableReason,
    FileSnapshot,
    WorkspaceChangeLimits,
    WorkspaceRoot,
    WorkspaceSnapshot,
)

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".venv",
    # 단계별로 생기는 일시적인 browser 스크린샷. workspace 산출물이 아니라 browser 패널과
    # 인라인 썸네일로 보여주는 진행 상황 피드백이다. browser 도구와 상수를 공유하므로
    # 이름이 서로 어긋날 수 없다.
    BROWSER_FRAMES_DIRNAME,
    # 외부화된 대용량 tool 출력(tool-output budget middleware의 기본 storage_subdir).
    # workspace 산출물이 아니라 모델이 read_file로 다시 읽는 process feedback이며, 위의
    # browser frames 제외와 같은 취지다. 이게 없으면 tool 출력을 외부화한 run이 run delivery
    # 검증(생산된 출력이 제시되지 않음)에 걸려 에러로 실패한다. 사용자 지정 storage_subdir
    # 값은 대신 ``extra_excluded_dir_names``로 전달된다.
    TOOL_RESULTS_DIRNAME,
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

BINARY_EXTENSIONS = {
    ".7z",
    ".avif",
    ".bmp",
    ".class",
    ".db",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}

SENSITIVE_PATH_PATTERNS = (
    ".env",
    ".env.*",
    "*api_key*",
    "*apikey*",
    "*.key",
    "*.pem",
    "*credential*",
    "*password*",
    "*private_key*",
    "*secret*",
    "*token*",
)

SAMPLE_BYTES = 4096
_UTF16_BOMS = (BOM_UTF16_LE, BOM_UTF16_BE)


def is_sensitive_workspace_path(path: str) -> bool:
    normalized = path.lower()
    parts = [part.lower() for part in Path(path).parts]
    basename = parts[-1] if parts else normalized
    for pattern in SENSITIVE_PATH_PATTERNS:
        if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(normalized, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def scan_workspace_roots(
    roots: list[WorkspaceRoot],
    *,
    limits: WorkspaceChangeLimits | None = None,
    include_text: bool = True,
    text_paths: set[str] | None = None,
    text_cache_dir: Path | None = None,
    extra_excluded_dir_names: frozenset[str] | None = None,
) -> WorkspaceSnapshot:
    resolved_limits = limits or WorkspaceChangeLimits()
    cache_dir = Path(text_cache_dir) if text_cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    # 운영자가 지정한 tool_output.storage_subdir 값이 여기로 온다. 기본 이름은 이미
    # EXCLUDED_DIR_NAMES에 있으므로 병합해도 안전하다. 의미가 있는 것은 한 segment짜리
    # 디렉터리 이름뿐이다. os.walk는 한 segment짜리 dirname을 내주므로 "cache/tool-results"
    # 같은 중첩 값은 절대 매칭되지 않는다. ToolOutputConfig가 단일 segment 계약을 강제하므로,
    # 여러 segment 값은 조용한 no-op이 아니라 호출자의 오류다.
    excluded_dir_names = EXCLUDED_DIR_NAMES | extra_excluded_dir_names if extra_excluded_dir_names else EXCLUDED_DIR_NAMES
    files: dict[str, FileSnapshot] = {}
    scanned = 0
    truncated = False

    for root in roots:
        if not root.host_path.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root.host_path, followlinks=False):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in excluded_dir_names and not (Path(dirpath) / dirname).is_symlink()]
            for filename in sorted(filenames):
                if scanned >= resolved_limits.max_scanned_files:
                    truncated = True
                    return WorkspaceSnapshot(
                        files=files,
                        truncated=truncated,
                        text_cache_dir=str(cache_dir) if cache_dir is not None else None,
                    )

                host_file = Path(dirpath) / filename
                if host_file.is_symlink():
                    # stat이나 내용 확인을 위해 symlink를 따라가서는 절대 안 된다. 대상은
                    # 스캔 대상 root 바깥을 포함해 host의 어디든 가리킬 수 있다. 그래서
                    # snapshot에서 조용히 빼는 대신, 아래에서 binary/대용량/민감해 보이는
                    # 파일을 다루는 방식과 동일하게 metadata만 담은 stub으로 기록한다.
                    symlink_snapshot = _snapshot_symlink(root, host_file)
                    if symlink_snapshot is not None:
                        files[symlink_snapshot.path] = symlink_snapshot
                        scanned += 1
                    continue
                if not host_file.is_file():
                    continue

                snapshot = _snapshot_file(
                    root,
                    host_file,
                    limits=resolved_limits,
                    include_text=include_text,
                    text_paths=text_paths,
                    text_cache_dir=cache_dir,
                )
                if snapshot is not None:
                    files[snapshot.path] = snapshot
                    scanned += 1

    return WorkspaceSnapshot(
        files=files,
        truncated=truncated,
        text_cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def _snapshot_file(
    root: WorkspaceRoot,
    host_file: Path,
    *,
    limits: WorkspaceChangeLimits,
    include_text: bool,
    text_paths: set[str] | None,
    text_cache_dir: Path | None,
) -> FileSnapshot | None:
    try:
        stat = host_file.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
        relative = host_file.relative_to(root.host_path).as_posix()
        virtual_path = f"{root.virtual_prefix}/{relative}"
        sensitive = is_sensitive_workspace_path(virtual_path)
    except OSError:
        return None

    if sensitive:
        return FileSnapshot(
            path=virtual_path,
            root=root.name,
            size=size,
            mtime_ns=mtime_ns,
            sha256=None,
            binary=False,
            sensitive=True,
            text=None,
            content_unavailable_reason="sensitive",
        )

    try:
        sample = host_file.read_bytes()[:SAMPLE_BYTES] if size <= SAMPLE_BYTES else _read_sample(host_file)
    except OSError:
        return None

    binary = host_file.suffix.lower() in BINARY_EXTENSIONS or _looks_binary(sample)
    sha256 = _sha256_file(host_file) if size <= limits.max_file_bytes_for_diff else None
    text: str | None = None
    text_path: str | None = None
    reason: DiffUnavailableReason | None = None

    should_include_text = include_text and (text_paths is None or virtual_path in text_paths)

    if binary:
        reason = "binary"
    elif size > limits.max_file_bytes_for_diff:
        reason = "large"
    elif not should_include_text:
        text = None
    else:
        try:
            raw = host_file.read_bytes()
        except OSError:
            return None
        decoded = _decode_text_bytes(raw)
        if decoded is None:
            binary = True
            reason = "binary"
        elif text_cache_dir is not None:
            text_path = str(_cache_text_file(decoded, virtual_path, text_cache_dir))
        else:
            text = decoded

    return FileSnapshot(
        path=virtual_path,
        root=root.name,
        size=size,
        mtime_ns=mtime_ns,
        sha256=sha256,
        binary=binary,
        sensitive=sensitive,
        text=text,
        text_path=text_path,
        content_unavailable_reason=reason,
    )


def _snapshot_symlink(root: WorkspaceRoot, host_file: Path) -> FileSnapshot | None:
    # 의도적으로 링크를 따라가지 않는다(대상에 read_bytes()/open()을 하지 않는다).
    # 대상은 스캔 대상 root 바깥을 포함해 host의 어디든 가리킬 수 있으므로, 여기서 그걸
    # stat하거나 읽으면 임의의 host 파일 내용/metadata가 workspace의 것인 양 노출될 수 있다.
    try:
        stat = host_file.lstat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
        relative = host_file.relative_to(root.host_path).as_posix()
        virtual_path = f"{root.virtual_prefix}/{relative}"
        sensitive = is_sensitive_workspace_path(virtual_path)
    except OSError:
        return None

    try:
        target = os.readlink(host_file)
    except OSError:
        target = None

    return FileSnapshot(
        path=virtual_path,
        root=root.name,
        size=size,
        mtime_ns=mtime_ns,
        sha256=None,
        binary=False,
        sensitive=sensitive,
        text=None,
        content_unavailable_reason="symlink",
        symlink=True,
        symlink_target=target,
    )


def _cache_text_file(text: str, virtual_path: str, cache_dir: Path) -> Path:
    cache_name = hashlib.sha256(virtual_path.encode("utf-8")).hexdigest()
    target = cache_dir / cache_name
    target.write_text(text, encoding="utf-8")
    return target


def _read_sample(path: Path) -> bytes:
    with path.open("rb") as file:
        return file.read(SAMPLE_BYTES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_text_bytes(data: bytes) -> str | None:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    if data.startswith(_UTF16_BOMS):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None

    return None


def _sample_decodes_as_text(sample: bytes, encoding: str) -> bool:
    try:
        decoder = getincrementaldecoder(encoding)()
        decoder.decode(sample, final=False)
    except UnicodeDecodeError:
        return False
    return True


def _looks_binary(sample: bytes) -> bool:
    if sample.startswith(_UTF16_BOMS) and _sample_decodes_as_text(sample, "utf-16"):
        return False
    if b"\x00" in sample:
        return True
    if _sample_decodes_as_text(sample, "utf-8"):
        return False
    return True
