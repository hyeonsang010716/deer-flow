"""공용 upload 관리 로직.

FastAPI/HTTP 의존성이 없는 순수 비즈니스 로직이다.
Gateway와 Client 모두 이 함수들에 위임한다.
"""

import errno
import logging
import os
import stat
from pathlib import Path
from urllib.parse import quote

from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.utils.thread_id import validate_thread_id


class PathTraversalError(ValueError):
    """경로가 허용된 base directory를 벗어날 때 raise된다."""


class UnsafeUploadPathError(ValueError):
    """upload 목적지가 안전한 일반 파일 경로가 아닐 때 raise된다."""


logger = logging.getLogger(__name__)

UPLOAD_STAGING_PREFIX = ".upload-"
UPLOAD_STAGING_SUFFIX = ".part"


def get_uploads_dir(thread_id: str, *, user_id: str | None = None) -> Path:
    """thread의 uploads 디렉터리 경로를 반환한다(부수 효과 없음)."""
    validate_thread_id(thread_id)
    return get_paths().sandbox_uploads_dir(thread_id, user_id=user_id or get_effective_user_id())


def ensure_uploads_dir(thread_id: str, *, user_id: str | None = None) -> Path:
    """thread의 uploads 디렉터리를 반환하며, 없으면 생성한다."""
    base = get_uploads_dir(thread_id, user_id=user_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


def normalize_filename(filename: str) -> str:
    """basename만 뽑아 파일명을 정제한다.

    디렉터리 구성 요소를 제거하고 traversal 패턴을 거부한다.

    Args:
        filename: 사용자 입력으로 받은 원본 파일명(경로 구성 요소가 있을 수 있다).

    Returns:
        안전한 파일명(basename만).

    Raises:
        ValueError: 파일명이 비었거나 traversal 패턴으로 해석될 때.
    """
    if not filename:
        raise ValueError("Filename is empty")
    safe = Path(filename).name
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Filename is unsafe: {filename!r}")
    # 역슬래시는 거부한다. Linux에서는 Path.name이 이를 리터럴 문자로 남기지만,
    # 제거하거나 거부해야 할 Windows 스타일 경로임을 뜻한다.
    if "\\" in safe:
        raise ValueError(f"Filename contains backslash: {filename!r}")
    if len(safe.encode("utf-8")) > 255:
        raise ValueError(f"Filename too long: {len(safe)} chars")
    return safe


def claim_unique_filename(name: str, seen: set[str]) -> str:
    """이름이 충돌하면 ``_N`` 접미사를 붙여 유일한 파일명을 만든다.

    반환한 이름을 *seen*에 자동으로 추가하므로 caller가 직접 넣을 필요가 없다.

    Args:
        name: 후보 파일명.
        seen: 이미 확보된 파일명 집합(제자리에서 변경된다).

    Returns:
        *seen*에 없던 파일명(이미 *seen*에 추가된 상태).
    """
    if name not in seen:
        seen.add(name)
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 1
    candidate = f"{stem}_{counter}{suffix}"
    while candidate in seen:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    seen.add(candidate)
    return candidate


def is_upload_staging_file(filename: str) -> bool:
    """*filename*이 임시 Gateway upload staging 파일인지 반환한다."""
    return filename.startswith(UPLOAD_STAGING_PREFIX) and filename.endswith(UPLOAD_STAGING_SUFFIX)


def validate_path_traversal(path: Path, base: Path) -> None:
    """*path*가 *base* 안에 있는지 확인한다.

    Raises:
        PathTraversalError: path traversal이 감지될 때.
    """
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        raise PathTraversalError("Path traversal detected") from None


def validate_upload_destination(base_dir: Path, filename: str) -> Path:
    """기존 파일을 변경하지 않고 upload 목적지를 검증한다."""
    safe_name = normalize_filename(filename)
    dest = base_dir / safe_name

    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        st = None

    if st is not None and not stat.S_ISREG(st.st_mode):
        raise UnsafeUploadPathError(f"Upload destination is not a regular file: {safe_name}")
    if st is not None and st.st_nlink > 1:
        raise UnsafeUploadPathError(f"Upload destination has multiple links: {safe_name}")

    validate_path_traversal(dest, base_dir)
    return dest


def _iter_upload_dirs(base_dir: Path):
    yield from base_dir.glob("threads/*/user-data/uploads")
    yield from base_dir.glob("users/*/threads/*/user-data/uploads")


def cleanup_stale_upload_staging_files(base_dir: Path | str | None = None) -> int:
    """하드 크래시로 남겨진 고아 Gateway upload staging 파일을 제거한다."""
    root = Path(base_dir) if base_dir is not None else get_paths().base_dir
    removed = 0
    for uploads_dir in _iter_upload_dirs(root):
        if not uploads_dir.is_dir():
            continue
        try:
            with os.scandir(uploads_dir) as entries:
                for entry in entries:
                    if not is_upload_staging_file(entry.name) or not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        os.unlink(entry.path)
                        removed += 1
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.warning("Failed to remove stale upload staging file: %s", entry.path, exc_info=True)
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to scan uploads directory for stale staging files: %s", uploads_dir, exc_info=True)
    return removed


def open_upload_file_no_symlink(base_dir: Path, filename: str) -> tuple[Path, object]:
    """안전한 streaming write를 위해 upload 목적지를 연다.

    upload 디렉터리는 local sandbox에 mount될 수 있어서, sandbox 프로세스가 앞으로 쓰일
    upload 파일명 자리에 symlink를 남길 수 있다. 평범한 ``Path.write_bytes``는 그 link를
    따라가 gateway 권한으로 uploads 디렉터리 밖의 파일을 덮어쓸 수 있다. 이 헬퍼는 POSIX에서
    ``O_NOFOLLOW``로 symlink 목적지를 거부한다. ``O_NOFOLLOW``가 없는 Windows에서는
    ``lstat``을 두 번 검사하고 ``open()`` 이후 ``fstat``으로 검증해 TOCTOU 창을 줄인다.
    모든 race를 없애지는 못하지만 악용을 훨씬 어렵게 만든다. path-traversal 검증은 두 경우
    모두 *base_dir* 밖으로의 이탈을 막는다.
    """
    safe_name = normalize_filename(filename)
    dest = validate_upload_destination(base_dir, safe_name)
    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        st = None

    has_nofollow = hasattr(os, "O_NOFOLLOW")

    if has_nofollow:
        # POSIX: O_NOFOLLOW를 쓰면 dest가 symlink일 때 open()이 ELOOP로 실패한다.
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK

        try:
            fd = os.open(dest, flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EISDIR, errno.ENOTDIR, errno.ENXIO, errno.EAGAIN}:
                raise UnsafeUploadPathError(f"Unsafe upload destination: {safe_name}") from exc
            raise

        try:
            opened_stat = os.fstat(fd)
            if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                raise UnsafeUploadPathError(f"Upload destination is not an exclusive regular file: {safe_name}")
            os.ftruncate(fd, 0)
            fh = os.fdopen(fd, "wb")
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)
        return dest, fh

    # Windows: O_NOFOLLOW가 없다. open() 직전에 lstat을 한 번 더 해서 TOCTOU 창을 좁히고,
    # open() 이후 fstat으로 한 겹 더 방어한다.
    # 참고: pre-open lstat과 open() 사이에 좁은 race 창이 남는다. path-traversal 검사가
    # base_dir 밖으로의 이탈은 완화하지만, 검사 후 dest를 symlink로 atomic하게 바꿔치기할
    # 수 있는 공격자는 막지 못한다.
    if st is not None and st.st_nlink > 1:
        raise UnsafeUploadPathError(f"Upload destination has multiple links: {safe_name}")

    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        pre_open_st = os.lstat(dest)
    except FileNotFoundError:
        pre_open_st = None

    if pre_open_st is not None and not stat.S_ISREG(pre_open_st.st_mode):
        raise UnsafeUploadPathError(f"Upload destination is not a regular file: {safe_name}")
    if pre_open_st is not None and pre_open_st.st_nlink > 1:
        raise UnsafeUploadPathError(f"Upload destination has multiple links: {safe_name}")

    try:
        fd = os.open(dest, flags, 0o600)
    except OSError as exc:
        if exc.errno in {errno.EISDIR, errno.ENOTDIR, errno.ENXIO, errno.EAGAIN}:
            raise UnsafeUploadPathError(f"Unsafe upload destination: {safe_name}") from exc
        raise

    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink > 1:
            raise UnsafeUploadPathError(f"Upload destination is not an exclusive regular file: {safe_name}")
        os.ftruncate(fd, 0)
        fh = os.fdopen(fd, "wb")
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)
    return dest, fh


def write_upload_file_no_symlink(base_dir: Path, filename: str, data: bytes) -> Path:
    """이미 존재하는 목적지 symlink를 따라가지 않고 upload 바이트를 쓴다."""
    dest, fh = open_upload_file_no_symlink(base_dir, filename)
    with fh:
        fh.write(data)
    return dest


def list_files_in_dir(directory: Path) -> dict:
    """*directory* 안의 파일(디렉터리 제외)을 나열한다.

    Args:
        directory: 스캔할 디렉터리.

    Returns:
        이름순으로 정렬된 "files" 리스트와 "count"를 담은 dict.
        각 파일 항목의 ``size``는 바이트 단위 *int*다. virtual / artifact URL을 추가하려면
        :func:`enrich_file_listing`을 호출한다.
    """
    if not directory.is_dir():
        return {"files": [], "count": 0}

    files = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if is_upload_staging_file(entry.name):
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            st = entry.stat(follow_symlinks=False)
            files.append(
                {
                    "filename": entry.name,
                    "size": st.st_size,
                    "path": entry.path,
                    "extension": Path(entry.name).suffix,
                    "modified": st.st_mtime,
                }
            )
    return {"files": files, "count": len(files)}


def delete_file_safe(base_dir: Path, filename: str, *, convertible_extensions: set[str] | None = None) -> dict:
    """path-traversal 검증 후 *base_dir* 안의 파일을 삭제한다.

    *convertible_extensions*가 주어지고 파일 확장자가 거기 해당하면, 동반 ``.md`` 파일도
    (존재할 경우) 함께 제거한다.

    Args:
        base_dir: 파일이 들어 있는 디렉터리.
        filename: 삭제할 파일 이름.
        convertible_extensions: 동반 markdown을 정리해야 하는 소문자 확장자
            (예: ``{".pdf", ".docx"}``).

    Returns:
        success와 message를 담은 dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        PathTraversalError: path traversal이 감지될 때.
    """
    file_path = (base_dir / filename).resolve()
    validate_path_traversal(file_path, base_dir)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    file_path.unlink()

    # upload 변환 중 생성된 동반 markdown을 정리한다.
    if convertible_extensions and file_path.suffix.lower() in convertible_extensions:
        file_path.with_suffix(".md").unlink(missing_ok=True)

    return {"success": True, "message": f"Deleted {filename}"}


def upload_artifact_url(thread_id: str, filename: str) -> str:
    """thread의 uploads 디렉터리에 있는 파일의 artifact URL을 만든다.

    공백, ``#``, ``?`` 등이 안전하도록 *filename*을 percent-encoding한다.
    """
    return f"/api/threads/{thread_id}/artifacts{VIRTUAL_PATH_PREFIX}/uploads/{quote(filename, safe='')}"


def upload_virtual_path(filename: str) -> str:
    """uploads 디렉터리에 있는 파일의 virtual path를 만든다."""
    return f"{VIRTUAL_PATH_PREFIX}/uploads/{filename}"


def enrich_file_listing(result: dict, thread_id: str) -> dict:
    """목록 결과에 virtual path와 artifact URL을 추가한다.

    *result*를 제자리에서 변경하고, 편의를 위해 그대로 반환한다.
    """
    for f in result["files"]:
        filename = f["filename"]
        f["virtual_path"] = upload_virtual_path(filename)
        f["artifact_url"] = upload_artifact_url(thread_id, filename)
    return result
