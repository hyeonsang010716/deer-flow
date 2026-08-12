"""관리형 Lark/Feishu CLI integration 지원.

이 integration은 공식 ``lark-*`` AI 에이전트 skill을 전역 읽기 전용 관리형 integration
skill 디렉터리에 설치한다. 일반 custom skill 아카이브 경로를 쓰지 않는 것은 의도적이다.
사용자가 작성해 변경 가능한 콘텐츠가 아니라 신뢰된 버전 관리 1st-party 패키지이기 때문이다.

버전 결정과 무결성
-------------------------------
설치되는 skill pack 버전은 Gateway 런타임 ``lark-cli`` 바이너리 버전
(``lark-cli --version``)을 따른다. 관리형 skill을 실제로 실행할 서버 측 CLI와 맞추기
위해서다. ``FALLBACK_LARK_CLI_VERSION``은 Dockerfile/npm 고정 버전과 일치하며, 런타임
바이너리가 없거나 파싱 가능한 버전을 알려주지 않을 때만 쓴다.

무결성은 버전별 아카이브 바이트 해시를 고정하지 않고 확보한다(GitHub는 내부 git 업그레이드를
거쳐도 소스 아카이브 바이트가 동일하다고 보장하지 않으며, 해시 고정은 최신 버전 추적과 충돌한다).
대신 다음을 지킨다.

* 다운로드 원본은 HTTPS 공식 GitHub 호스트로 고정하고, 버전은 Gateway 런타임 CLI 버전이나
  고정된 fallback에서만 온다(외부 URL 주입 불가).
* 모든 아카이브 멤버가 구조적 검사(zip-slip, symlink, 실행 바이너리, 크기, 필수 skill 완비,
  ``SKILL.md`` 파싱)를 통과한다.
* DeerFlow 공용 가이드를 주입한 뒤 추출된 skill 트리에 대한 **콘텐츠** SHA-256을 manifest에
  기록한다. 그래서 GitHub가 같은 콘텐츠를 다른 바이트로 다시 패킹하더라도, 실효 skill 콘텐츠가
  바뀐 재설치는 감지하고 감사할 수 있다.

런타임 결합: npm으로 설치되는 ``lark-cli`` 바이너리 버전은 ``backend/Dockerfile``
(``ARG LARK_CLI_NPM_VERSION``)과 ``docker/docker-compose*.yaml``에 부트스트랩 fallback으로
고정되어 있다. 관리자 설치 경로는 ``.deer-flow/integrations/lark-cli/gateway-cli`` 아래에
쓰기 가능한 DeerFlow 소유 Gateway CLI도 관리하며 시스템 PATH보다 우선한다. 덕분에 사용자가
터미널 설치 명령을 실행할 필요가 없다. integration을 재설치하면 네트워크가 되는 한 관리형
Gateway CLI와 skill pack이 같은 버전으로 갱신된다. ``get_lark_integration_status``는 운영자를
위해 ``latest_available_version``과 ``runtime_version_mismatch``를 노출하고,
``test_python_and_docker_lark_cli_versions_match``가 fallback 상수를 Dockerfile ARG에 고정해
패키지 배포본이 조용히 어긋나지 않게 한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback 경로
    fcntl = None  # type: ignore[assignment]
    import msvcrt

from deerflow.config.app_config import AppConfig
from deerflow.config.paths import Paths, get_paths
from deerflow.integrations.lark_broker import LARK_BROKER_URL_ENV
from deerflow.skills.installer import is_executable_binary_prefix, is_symlink_member, is_unsafe_zip_member
from deerflow.skills.parser import parse_skill_file
from deerflow.skills.permissions import make_skill_tree_sandbox_readable
from deerflow.skills.types import SKILL_MD_FILE, SkillCategory

logger = logging.getLogger(__name__)

INTEGRATION_ID = "lark-cli"
# Gateway 이미지/npm 고정 버전과 일치한다. 런타임 바이너리가 없거나 파싱 불가능한 버전을
# 보고할 때 쓴다.
FALLBACK_LARK_CLI_VERSION = "v1.0.65"
LARK_CLI_NPM_VERSION = FALLBACK_LARK_CLI_VERSION.removeprefix("v")
LARK_CLI_NPM_PACKAGE = "@larksuite/cli"
LARK_CLI_GITHUB_REPO = "larksuite/cli"
LARK_CLI_LATEST_RELEASE_API = f"https://api.github.com/repos/{LARK_CLI_GITHUB_REPO}/releases/latest"
LARK_CLI_SOURCE_ARCHIVE_ENV = "DEER_FLOW_LARK_CLI_SKILLS_ARCHIVE"
LARK_CLI_SANDBOX_RUNTIME_SOURCE_ENV = "DEER_FLOW_LARK_CLI_SANDBOX_RUNTIME_DIR"
LARK_CLI_DOWNLOAD_TIMEOUT_SECONDS = 60
LARK_CLI_NPM_INSTALL_TIMEOUT_SECONDS = 180
LARK_HTTP_TIMEOUT_SECONDS = 20
LARK_CONFIG_POLL_TIMEOUT_SECONDS = 45
LARK_AUTH_COMPLETE_DEFAULT_WAIT_SECONDS = 45
LARK_AUTH_COMPLETE_MIN_WAIT_SECONDS = 5
LARK_AUTH_COMPLETE_MAX_WAIT_SECONDS = 45
LARK_CLI_LATEST_VERSION_TTL_SECONDS = 3600
LARK_CLI_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
LARK_CLI_MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
LARK_CLI_MAX_RUNTIME_ASSET_BYTES = 128 * 1024 * 1024
LARK_CLI_MANIFEST_FILE = ".deerflow-lark-cli-manifest.json"
LARK_CLI_SANDBOX_CONFIG_DIR = "/mnt/integrations/lark-cli/config"
LARK_CLI_SANDBOX_LOCKS_DIR = f"{LARK_CLI_SANDBOX_CONFIG_DIR}/locks"
LARK_CLI_SANDBOX_DATA_DIR = "/mnt/integrations/lark-cli/data"
LARK_CLI_SANDBOX_RUNTIME_DIR = "/mnt/integrations/lark-cli/runtime"
LARK_CLI_LINUX_ARCHES = ("amd64", "arm64")
LARK_CLI_RUNTIME_MANIFEST_FILE = ".deerflow-lark-cli-runtime.json"
LARK_CLI_FLOW_STATE_FILE = ".deerflow-lark-cli-flow.json"

# Pattern B(issue #4338): sandbox shim이 broker sidecar에 닿는 loopback URL.
# LARK_BROKER_URL_ENV는 broker 모듈에서 import해 shim, 서버, Gateway overlay가
# 하나의 진실 원천을 공유하게 한다.
LARK_BROKER_SANDBOX_URL = "http://127.0.0.1:8788"

# sandbox 런타임 레이아웃용 아키텍처 분기 launcher. Gateway 쪽 writer
# (`_write_lark_cli_sandbox_launcher`)와 `docker/lark-cli-init` init 이미지가 공유하므로
# `bin/lark-cli`를 만드는 두 생산자가 어긋나지 않는다.
LARK_CLI_SANDBOX_LAUNCHER_SCRIPT = """#!/bin/sh
set -eu
case "$(uname -m)" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "Unsupported sandbox architecture: $(uname -m)" >&2; exit 126 ;;
esac
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$script_dir/../linux-$arch/lark-cli" "$@"
"""
_VERSION_TAG_RE = re.compile(r"v?\d+\.\d+\.\d+")
_DEERFLOW_LARK_SHARED_GUIDANCE_MARKER = "<!-- deerflow-lark-cli-auth-guidance-v2 -->"
_DEERFLOW_LARK_SHARED_GUIDANCE_LEGACY_MARKERS = ("<!-- deerflow-lark-cli-auth-guidance-v1 -->",)
_LARK_APP_REGISTRATION_PATH = "/oauth/v1/app/registration"

LARK_SKILL_NAMES: tuple[str, ...] = (
    "lark-approval",
    "lark-apps",
    "lark-attendance",
    "lark-base",
    "lark-calendar",
    "lark-contact",
    "lark-doc",
    "lark-drive",
    "lark-event",
    "lark-im",
    "lark-mail",
    "lark-markdown",
    "lark-minutes",
    "lark-note",
    "lark-okr",
    "lark-openapi-explorer",
    "lark-shared",
    "lark-sheets",
    "lark-skill-maker",
    "lark-slides",
    "lark-task",
    "lark-vc",
    "lark-vc-agent",
    "lark-whiteboard",
    "lark-wiki",
    "lark-workflow-meeting-summary",
    "lark-workflow-standup-report",
)
LARK_SKILL_NAME_SET = frozenset(LARK_SKILL_NAMES)
_LARK_INSTALL_THREAD_LOCK = threading.Lock()
_LARK_RUNTIME_INSTALL_THREAD_LOCK = threading.Lock()
_LARK_CREDENTIAL_LOCKS_GUARD = threading.Lock()
_LARK_CREDENTIAL_LOCKS: WeakValueDictionary[str, threading.Lock] = WeakValueDictionary()


@dataclass(frozen=True)
class LarkCliProbe:
    available: bool
    path: str | None = None
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LarkAuthProbe:
    status: str
    message: str | None = None
    user: str | None = None
    verified: bool = False


@dataclass(frozen=True)
class LarkIntegrationStatus:
    installed: bool
    version: str
    manifest_version: str | None
    latest_available_version: str | None
    runtime_version_mismatch: bool
    app_configured: bool
    app_id: str | None
    app_brand: str | None
    skills_expected: int
    skills_installed: int
    installed_skills: tuple[str, ...]
    enabled_skills: tuple[str, ...]
    install_path: str
    cli: LarkCliProbe
    auth: LarkAuthProbe
    sandbox_runtime_mode: str = "none"
    sandbox_runtime_ready: bool = False
    sandbox_runtime_detail: str | None = None


@dataclass(frozen=True)
class LarkInstallResult:
    success: bool
    installed_skills: tuple[str, ...]
    status: LarkIntegrationStatus
    message: str


@dataclass(frozen=True)
class LarkConfigStartResult:
    verification_url: str
    device_code: str
    generation: str
    expires_in: int | None = None
    interval: int | None = None
    user_code: str | None = None
    brand: str = "feishu"


@dataclass(frozen=True)
class LarkConfigCompleteResult:
    success: bool
    status: LarkIntegrationStatus
    message: str
    generation: str


@dataclass(frozen=True)
class LarkAuthStartResult:
    verification_url: str
    device_code: str
    generation: str
    expires_in: int | None = None
    user_code: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class LarkAuthCompleteResult:
    success: bool
    status: LarkIntegrationStatus
    message: str


class LarkFlowSupersededError(ValueError):
    """지연된 integration flow가 더 이상 최신이 아닐 때 발생한다."""


def lark_integration_root(_user_id: str | None = None) -> Path:
    """전역 설치된 관리형 Lark skill의 공유 루트를 반환한다.

    ``_user_id``는 전역 설치 이전 API와의 소스 호환을 위해 한시적으로 받는다.
    공유 패키지 경로에는 영향을 주지 않는다.
    """
    return get_paths().integration_skills_dir() / INTEGRATION_ID


def lark_manifest_path(user_id: str) -> Path:
    return lark_integration_root(user_id) / LARK_CLI_MANIFEST_FILE


def lark_skills_installed(user_id: str | None = None) -> bool:
    """관리형 Lark skill pack이 설치되어 있는지 반환한다.

    CLI나 auth를 조회하지 않고 :func:`get_lark_integration_status`의 ``installed`` 필드와
    같은 판단(manifest 존재 + ``lark-shared`` 추출됨)을 한다. sandbox가 lark-cli 런타임을
    요청할지 결정하는 데 쓴다.
    """
    root = lark_integration_root(user_id)
    manifest = _read_manifest(root)
    if not manifest:
        return False
    return "lark-shared" in _installed_lark_skill_names(root)


def lark_cli_config_dir(user_id: str) -> Path:
    return get_paths().user_dir(user_id) / "integrations" / INTEGRATION_ID / "config"


def lark_cli_data_dir(user_id: str) -> Path:
    return get_paths().user_dir(user_id) / "integrations" / INTEGRATION_ID / "data"


def _lark_cli_credential_root(user_id: str) -> Path:
    return get_paths().user_dir(user_id) / "integrations" / INTEGRATION_ID


def ensure_lark_cli_credential_tree(user_id: str, *, paths: Paths | None = None) -> None:
    """비밀을 담은 사용자 Lark CLI 트리를 소유자 전용 권한으로 만든다.

    CLI가 이 트리 아래에 평문 app secret과 OAuth 토큰을 쓴다. 권한을 바꾸기 전에 링크를
    거부해, 침해된 트리가 chmod나 이후 CLI 쓰기를 사용자 integration 디렉터리 밖으로
    돌리지 못하게 한다.
    """
    paths = paths or get_paths()
    root = paths.user_dir(user_id) / "integrations" / INTEGRATION_ID
    if root.is_symlink():
        raise ValueError(f"Lark CLI credential path must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    for required in (root / "config", root / "config" / "locks", root / "data"):
        if required.is_symlink():
            raise ValueError(f"Lark CLI credential path must not be a symlink: {required}")
        required.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Lark CLI credential path must not be a symlink: {path}")
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)
        else:
            raise ValueError(f"Unsupported file type in Lark CLI credential tree: {path}")


def lark_cli_managed_gateway_dir() -> Path:
    """Gateway 범위의 DeerFlow 관리형 lark-cli 설치 루트."""
    return get_paths().base_dir / "integrations" / INTEGRATION_ID / "gateway-cli"


def lark_cli_managed_sandbox_dir() -> Path:
    """Linux AIO sandbox에 마운트되는 Gateway 측 원본 디렉터리."""
    return get_paths().base_dir / "integrations" / INTEGRATION_ID / "sandbox-cli"


def _lark_cli_release_asset_name(version: str, arch: str) -> str:
    tag = _normalize_lark_cli_version_tag(version)
    if tag is None:
        raise ValueError(f"Invalid Lark CLI version tag: {version!r}")
    if arch not in LARK_CLI_LINUX_ARCHES:
        raise ValueError(f"Unsupported Lark CLI Linux architecture: {arch!r}")
    return f"lark-cli-{tag.removeprefix('v')}-linux-{arch}.tar.gz"


def _lark_cli_release_asset_url(version: str, asset_name: str) -> str:
    tag = _normalize_lark_cli_version_tag(version)
    if tag is None:
        raise ValueError(f"Invalid Lark CLI version tag: {version!r}")
    quoted_asset = urllib.parse.quote(asset_name, safe="")
    return f"https://github.com/{LARK_CLI_GITHUB_REPO}/releases/download/{tag}/{quoted_asset}"


def _download_lark_release_asset(version: str, asset_name: str, *, max_bytes: int = LARK_CLI_MAX_RUNTIME_ASSET_BYTES) -> bytes:
    """공식 release asset 하나를 엄격한 크기 제한과 함께 다운로드한다."""
    request = urllib.request.Request(
        _lark_cli_release_asset_url(version, asset_name),
        headers={"Accept": "application/octet-stream", "User-Agent": "deer-flow"},
    )
    try:
        with urllib.request.urlopen(request, timeout=LARK_CLI_DOWNLOAD_TIMEOUT_SECONDS) as response:
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Lark CLI release asset {asset_name!r} is too large.")
                chunks.append(chunk)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - 네트워크 경계
        raise ValueError(f"Could not download official Lark CLI release asset {asset_name!r} for {version}.") from exc
    return b"".join(chunks)


def _release_checksums(raw: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Lark CLI release checksums are not valid UTF-8.") from exc
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            continue
        checksums[parts[-1].lstrip("*")] = parts[0].lower()
    return checksums


def _extract_lark_cli_runtime_binary(archive: bytes, destination: Path) -> None:
    """공식 tar 아카이브에서 CLI 실행 파일 하나만 안전하게 추출한다."""
    candidate: bytes | None = None
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tf:
            for member in tf.getmembers():
                normalized = posixpath.normpath(member.name.replace("\\", "/"))
                parts = PurePosixPath(normalized).parts
                if normalized.startswith("/") or ".." in parts or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unsafe Lark CLI runtime archive member: {member.name}")
                if member.isfile():
                    total += member.size
                    if total > LARK_CLI_MAX_RUNTIME_ASSET_BYTES:
                        raise ValueError("Lark CLI runtime archive expands beyond the allowed size.")
                    if PurePosixPath(normalized).name == "lark-cli":
                        extracted = tf.extractfile(member)
                        if extracted is None or candidate is not None:
                            raise ValueError("Lark CLI runtime archive must contain exactly one lark-cli executable.")
                        candidate = extracted.read()
    except tarfile.TarError as exc:
        raise ValueError("Lark CLI runtime archive is not a valid tar archive.") from exc
    if not candidate:
        raise ValueError("Lark CLI runtime archive does not contain a lark-cli executable.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(candidate)
    destination.chmod(0o755)


def _write_lark_cli_sandbox_launcher(staging: Path) -> None:
    launcher = staging / "bin" / "lark-cli"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(LARK_CLI_SANDBOX_LAUNCHER_SCRIPT, encoding="utf-8")
    launcher.chmod(0o755)


def _validate_lark_cli_sandbox_runtime(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Managed Lark CLI sandbox runtime root must be a regular directory, not a symlink.")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Managed Lark CLI sandbox runtime must not contain a symlink: {path}")
        if not (path.is_dir() or path.is_file()):
            raise ValueError(f"Managed Lark CLI sandbox runtime contains an unsupported file type: {path}")
    for relative in (Path("bin/lark-cli"), *(Path(f"linux-{arch}/lark-cli") for arch in LARK_CLI_LINUX_ARCHES)):
        candidate = root / relative
        if not candidate.is_file():
            raise ValueError(f"Managed Lark CLI sandbox runtime is missing a regular file: {relative}")
        if candidate.stat().st_mode & 0o111 == 0:
            raise ValueError(f"Managed Lark CLI sandbox runtime file is not executable: {relative}")


def _read_json_object_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


@contextmanager
def _exclusive_install_lock(lock_path: Path, thread_lock):
    """advisory 파일 lock과 프로세스 내 lock을 함께 잡는다."""
    with thread_lock, lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows fallback 경로
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            lock_file.seek(0)
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows fallback 경로
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _lark_credential_thread_lock(user_id: str) -> threading.Lock:
    with _LARK_CREDENTIAL_LOCKS_GUARD:
        return _LARK_CREDENTIAL_LOCKS.setdefault(user_id, threading.Lock())


@contextmanager
def _lark_credential_lock(user_id: str):
    """한 사용자의 자격증명 교체를 thread/프로세스 간에 직렬화한다."""
    root = _lark_cli_credential_root(user_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{INTEGRATION_ID}.credentials.lock"
    with _exclusive_install_lock(lock_path, _lark_credential_thread_lock(user_id)):
        yield


def _lark_flow_state_path(user_id: str) -> Path:
    return _lark_cli_credential_root(user_id) / LARK_CLI_FLOW_STATE_FILE


def _write_lark_flow_generation_locked(user_id: str, generation: str) -> None:
    ensure_lark_cli_credential_tree(user_id)
    path = _lark_flow_state_path(user_id)
    fd, temp_name = tempfile.mkstemp(prefix=f".{LARK_CLI_FLOW_STATE_FILE}.", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps({"generation": generation}) + "\n", encoding="utf-8")
        temp_path.chmod(0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _advance_lark_flow_generation_locked(user_id: str) -> str:
    generation = uuid4().hex
    _write_lark_flow_generation_locked(user_id, generation)
    return generation


def _require_lark_flow_generation_locked(user_id: str, generation: str) -> str:
    expected = generation.strip()
    state = _read_json_object_file(_lark_flow_state_path(user_id))
    if not expected or state is None or state.get("generation") != expected:
        raise LarkFlowSupersededError("This Lark integration flow was superseded by a newer action.")
    return expected


def _ensure_managed_sandbox_lark_cli(version: str) -> Path:
    """AIO sandbox 실행용으로 검증된 공식 Linux 바이너리를 설치한다."""
    tag = _normalize_lark_cli_version_tag(version)
    if tag is None:
        raise ValueError(f"Invalid Lark CLI version tag: {version!r}")
    target = lark_cli_managed_sandbox_dir()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_install_lock(parent / ".sandbox-cli.install.lock", _LARK_RUNTIME_INSTALL_THREAD_LOCK):
        return _ensure_managed_sandbox_lark_cli_locked(tag, target, parent)


def _ensure_managed_sandbox_lark_cli_locked(tag: str, target: Path, parent: Path) -> Path:
    manifest = _read_json_object_file(target / LARK_CLI_RUNTIME_MANIFEST_FILE)
    if manifest and manifest.get("version") == tag:
        _validate_lark_cli_sandbox_runtime(target)
        return target

    staging = Path(tempfile.mkdtemp(prefix=".installing-sandbox-cli-", dir=str(parent)))
    backup: Path | None = None
    try:
        source_override = os.getenv(LARK_CLI_SANDBOX_RUNTIME_SOURCE_ENV)
        if source_override:
            source = Path(source_override)
            _validate_lark_cli_sandbox_runtime(source)
            shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=False)
        else:
            checksums = _release_checksums(_download_lark_release_asset(tag, "checksums.txt", max_bytes=1024 * 1024))
            for arch in LARK_CLI_LINUX_ARCHES:
                asset_name = _lark_cli_release_asset_name(tag, arch)
                archive = _download_lark_release_asset(tag, asset_name)
                expected = checksums.get(asset_name)
                actual = hashlib.sha256(archive).hexdigest()
                if expected is None or actual != expected:
                    raise ValueError(f"Lark CLI release asset checksum mismatch: {asset_name}")
                _extract_lark_cli_runtime_binary(archive, staging / f"linux-{arch}" / "lark-cli")
            _write_lark_cli_sandbox_launcher(staging)

        _validate_lark_cli_sandbox_runtime(staging)
        (staging / LARK_CLI_RUNTIME_MANIFEST_FILE).write_text(
            json.dumps({"version": tag}, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            backup = parent / f".replacing-sandbox-cli-{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            target.rename(backup)
        staging.rename(target)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        return target
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _lark_cli_managed_bin_dir() -> Path:
    return lark_cli_managed_gateway_dir() / "node_modules" / ".bin"


def _lark_cli_managed_path() -> str | None:
    for name in ("lark-cli", "lark-cli.cmd"):
        candidate = _lark_cli_managed_bin_dir() / name
        if candidate.exists():
            return str(candidate)
    return None


def lark_cli_env_overlay(user_id: str, *, sandbox_paths: bool = False, broker: bool = False) -> dict[str, str]:
    """DeerFlow 관리형 자격증명을 쓰는 lark-cli용 환경 overlay.

    디렉터리는 사용자별이라 로컬 trusted 모드 로그인이 다른 계정으로 새지 않는다.

    ``broker``가 설정되면(Pattern B, issue #4338) sandbox는 자격증명을 소유한 broker
    sidecar와 통신하므로 overlay에는 broker URL과 런타임 PATH만 담기고
    ``LARKSUITE_CLI_CONFIG_DIR``/``DATA_DIR``은 절대 들어가지 않는다. 덕분에 평문 app
    secret과 OAuth 토큰이 sandbox 파일시스템에 아예 존재하지 않는다. ``broker``는
    ``sandbox_paths``를 함의한다.
    """
    if broker:
        return {
            "PATH": f"{LARK_CLI_SANDBOX_RUNTIME_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            LARK_BROKER_URL_ENV: LARK_BROKER_SANDBOX_URL,
        }
    if sandbox_paths:
        config_dir: Path | str = LARK_CLI_SANDBOX_CONFIG_DIR
        data_dir: Path | str = LARK_CLI_SANDBOX_DATA_DIR
    else:
        config_dir = lark_cli_config_dir(user_id)
        data_dir = lark_cli_data_dir(user_id)
        ensure_lark_cli_credential_tree(user_id)
    overlay = {
        "LARKSUITE_CLI_CONFIG_DIR": str(config_dir),
        "LARKSUITE_CLI_DATA_DIR": str(data_dir),
        "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
        "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
    }
    if not sandbox_paths and _lark_cli_managed_path() is not None:
        overlay["PATH"] = f"{_lark_cli_managed_bin_dir()}{os.pathsep}{os.environ.get('PATH', '')}"
    elif sandbox_paths:
        overlay["PATH"] = f"{LARK_CLI_SANDBOX_RUNTIME_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return overlay


def lark_cli_env(user_id: str) -> dict[str, str]:
    """Gateway 쪽 lark-cli probe에 쓰는 전체 환경."""
    return {**os.environ, **lark_cli_env_overlay(user_id)}


def probe_lark_cli() -> LarkCliProbe:
    path = _resolve_lark_cli_path()
    if path is None:
        return LarkCliProbe(available=False, error="lark-cli is not installed on the Gateway")
    return _probe_lark_cli_at_path(path)


def _probe_lark_cli_at_path(path: str) -> LarkCliProbe:
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001 - probe 경계
        return LarkCliProbe(available=False, path=path, error=str(exc))

    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return LarkCliProbe(available=False, path=path, error=output or f"exit code {result.returncode}")
    return LarkCliProbe(available=True, path=path, version=output or None)


def probe_lark_auth(user_id: str, *, verify: bool = False) -> LarkAuthProbe:
    """사용자의 Lark 인가 상태를 조회한다.

    기본적으로는 로컬 토큰 존재 여부만 확인한다(``auth status --json``). 저렴하고 오프라인이라
    자주 폴링되는 상태 엔드포인트에 적합하다. ``verify=True``를 주면 ``--verify``를 붙여 Lark에
    실제 토큰 검증을 한다. 호출마다 네트워크 왕복 비용이 들므로 명시적인 "인가 완료" 단계에만
    쓴다.
    """
    path = _resolve_lark_cli_path()
    if path is None:
        return LarkAuthProbe(status="unavailable", message="lark-cli is not installed on the Gateway")
    app_config = read_lark_app_config(user_id)
    if not app_config["configured"]:
        return LarkAuthProbe(status="not_configured", message="Lark app is not configured")
    args = [path, "auth", "status", "--json"]
    if verify:
        args.append("--verify")
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            env=lark_cli_env(user_id),
        )
    except subprocess.TimeoutExpired:
        return LarkAuthProbe(status="error", message="lark-cli auth status timed out")
    except Exception as exc:  # noqa: BLE001 - probe 경계
        return LarkAuthProbe(status="error", message=str(exc))

    raw = (result.stdout or result.stderr or "").strip()
    data: dict[str, Any] | None = None
    if raw:
        try:
            parsed = json.loads(raw)
            data = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            data = None

    if result.returncode != 0:
        message = _auth_error_message(data) if data else raw
        return LarkAuthProbe(status="not_authorized", message=message or "Lark user authorization is not configured")

    user = None
    if data:
        identities = data.get("identities")
        if isinstance(identities, dict):
            user_info = identities.get("user")
            if isinstance(user_info, dict):
                user = str(user_info.get("userName") or user_info.get("openId") or "") or None
        if user is None and data.get("userName"):
            user = str(data["userName"])
    if verify:
        return LarkAuthProbe(
            status="authenticated",
            user=user,
            message="Lark/Feishu authorization is live-verified.",
            verified=True,
        )
    return LarkAuthProbe(
        status="authenticated",
        user=user,
        message="Lark/Feishu credentials are configured locally but not live-verified.",
        verified=False,
    )


def _resolve_sandbox_runtime_readiness(
    config: AppConfig,
    *,
    probe: bool,
) -> tuple[str, bool, str | None]:
    """sandbox lark-cli 런타임 모드와 준비 상태를 판정한다.

    모드:
    - ``none``: sandbox가 lark-cli를 실행하지 않는다(AIO가 아닌 provider).
    - ``gateway-download``: 로컬 AIO. Gateway가 ``sandbox-cli`` 디렉터리를 준비해
      bind-mount한다. 그 디렉터리가 검증을 통과하면 준비 완료다.
    - ``init-container``: 원격 provisioner. lark-cli init 이미지가 런타임 바이너리와 평문
      자격증명 마운트를 제공한다(Pattern A). provisioner가 init 이미지 설정을 보고하면
      준비 완료다.
    - ``broker``: 원격 provisioner. lark-cli broker sidecar가 자격증명을 갖고 sandbox에는
      shim만 준다(Pattern B, issue #4338). provisioner가 broker 이미지 설정을 보고하면
      준비 완료다. 둘 다 가능하면 broker가 ``init-container``보다 우선한다.

    ``probe``는 provisioner capability 호출(best-effort, 짧은 timeout) 수행 여부를 결정한다.
    """
    if not _uses_aio_sandbox(config):
        return "none", False, "Sandbox does not run lark-cli in this configuration."

    if _uses_remote_provisioner(config):
        if not probe:
            return "init-container", False, None
        caps = _probe_provisioner_capabilities(config)
        if caps is None:
            return "init-container", False, "Could not reach the provisioner to confirm the lark-cli runtime image."
        # provisioner에 broker 이미지가 설정되어 있으면 Pattern B(broker)가
        # Pattern A(init container 바이너리)보다 우선한다.
        if caps["lark_cli_broker_image"]:
            return "broker", True, None
        if caps["lark_cli_init_image"]:
            return "init-container", True, None
        return "init-container", False, "The provisioner has no lark-cli runtime image configured (LARK_CLI_INIT_IMAGE / LARK_CLI_BROKER_IMAGE)."

    # 로컬 AIO: Gateway가 내려받는 런타임 디렉터리.
    runtime_dir = lark_cli_managed_sandbox_dir()
    try:
        _validate_lark_cli_sandbox_runtime(runtime_dir)
    except (ValueError, OSError):
        return "gateway-download", False, "The managed sandbox lark-cli runtime is not installed."
    return "gateway-download", True, None


LARK_BROKER_MODE_TTL_SECONDS = 60
# 부정 결과(broker 비활성)를 긍정 결과보다 오래 캐싱한다. broker가 아닌 원격 provisioner
# 배포는 도중에 broker로 바뀌기보다 프로세스 수명 내내 그대로일 확률이 훨씬 높으므로,
# bash hot path가 1분마다 재조회하지 않게 한다. 긍정 결과는 여전히 짧은 TTL로 갱신된다.
LARK_BROKER_MODE_NEGATIVE_TTL_SECONDS = 300
# bash 호출마다 도는 hot path이므로 probe 예산을 빡빡하게 잡는다. Settings 상태 probe(5초,
# 사용자가 페이지에서 기다린다)와 달리 이건 sandbox lark-cli 명령 직전에 인라인으로 돌기
# 때문에, 느리거나 닿지 않는 provisioner가 broker 아닌 배포의 TTL마다 첫 호출에 수 초의
# 지연을 더해서는 안 된다.
LARK_BROKER_MODE_PROBE_TIMEOUT_SECONDS = 1.5
# sandbox_lark_broker_active의 캐시 속성을 동시 bash 호출로부터 보호한다. 정확성이
# 멱등한 race에 기대지 않게 하기 위함이다.
_LARK_BROKER_MODE_CACHE_LOCK = threading.Lock()


def sandbox_lark_broker_active(config: AppConfig | None = None) -> bool:
    """sandbox ``lark-cli``가 broker 모드(Pattern B)로 도는지 반환한다.

    모든 ``lark-cli`` bash 호출에서 참조되고 provisioner capability를 HTTP로 읽으므로 짧은
    TTL로 캐싱한다. broker 모드는 broker 이미지 설정을 보고하는 원격 provisioner를 요구한다.
    그 밖의 구성(로컬 AIO, init container 바이너리 모드, 닿지 않는 provisioner)은 False이며,
    호출자는 자격증명 마운트 overlay로 되돌아간다.

    probe는 빡빡한 timeout을 쓰고 부정 결과를 긍정보다 오래 캐싱하므로, broker가 아닌 원격
    provisioner 배포가 bash hot path에서 지연 비용을 치르지 않는다.
    """
    if config is None:
        try:
            from deerflow.config.app_config import get_app_config

            config = get_app_config()
        except Exception:  # noqa: BLE001 - broker 아닌 overlay로 degrade
            return False

    now = time.monotonic()
    with _LARK_BROKER_MODE_CACHE_LOCK:
        cached = getattr(sandbox_lark_broker_active, "_cache", None)
        if cached is not None:
            ts, value = cached
            ttl = LARK_BROKER_MODE_TTL_SECONDS if value else LARK_BROKER_MODE_NEGATIVE_TTL_SECONDS
            if now - ts < ttl:
                return value

    active = False
    if _uses_aio_sandbox(config) and _uses_remote_provisioner(config):
        caps = _probe_provisioner_capabilities(config, timeout=LARK_BROKER_MODE_PROBE_TIMEOUT_SECONDS)
        active = bool(caps and caps["lark_cli_broker_image"])
    with _LARK_BROKER_MODE_CACHE_LOCK:
        sandbox_lark_broker_active._cache = (now, active)  # type: ignore[attr-defined]
    return active


def get_lark_integration_status(
    user_id: str,
    config: AppConfig,
    *,
    verify_auth: bool = False,
    check_latest: bool = False,
    check_runtime: bool = False,
) -> LarkIntegrationStatus:
    root = lark_integration_root(user_id)
    manifest = _read_manifest(root)
    app_config = read_lark_app_config(user_id)
    installed_skills = tuple(sorted(_installed_lark_skill_names(root)))
    enabled_skills = tuple(sorted(_enabled_lark_skill_names(user_id, config)))
    manifest_version = str(manifest.get("version")) if manifest else None
    cli = probe_lark_cli()
    latest_available = _cached_latest_lark_cli_version() if check_latest else None
    runtime_mode, runtime_ready, runtime_detail = _resolve_sandbox_runtime_readiness(config, probe=check_runtime)
    return LarkIntegrationStatus(
        installed=bool(manifest) and "lark-shared" in installed_skills,
        version=manifest_version or FALLBACK_LARK_CLI_VERSION,
        manifest_version=manifest_version,
        latest_available_version=latest_available,
        runtime_version_mismatch=_versions_drifted(manifest_version, cli.version),
        app_configured=bool(app_config["configured"]),
        app_id=app_config["app_id"],
        app_brand=app_config["brand"],
        skills_expected=len(LARK_SKILL_NAMES),
        skills_installed=len(installed_skills),
        installed_skills=installed_skills,
        enabled_skills=enabled_skills,
        install_path=str(root),
        cli=cli,
        auth=probe_lark_auth(user_id, verify=verify_auth),
        sandbox_runtime_mode=runtime_mode,
        sandbox_runtime_ready=runtime_ready,
        sandbox_runtime_detail=runtime_detail,
    )


def _normalize_version(value: str | None) -> str | None:
    """버전처럼 생긴 문자열에서 비교 가능한 ``major.minor.patch``를 뽑는다."""
    if not value:
        return None
    match = re.search(r"\d+\.\d+\.\d+", value)
    return match.group(0) if match else None


def _versions_drifted(manifest_version: str | None, cli_version: str | None) -> bool:
    """두 버전을 모두 알고 있고 숫자 부분이 다르면 True를 반환한다.

    manifest는 설치된 skill pack 버전을, ``cli_version``은 Gateway 런타임 ``lark-cli``
    바이너리 버전을 담는다. 한쪽이라도 알 수 없으면 불일치를 주장할 수 없으므로 조용히 넘어간다.
    """
    left = _normalize_version(manifest_version)
    right = _normalize_version(cli_version)
    if left is None or right is None:
        return False
    return left != right


def _resolve_runtime_lark_cli_version() -> str:
    """Gateway 런타임 CLI에 맞는 skill pack 버전을 결정한다.

    관리형 Lark skill은 서버 측 ``lark-cli`` 바이너리가 실행하므로, integration 설치는
    GitHub 최신 릴리스를 무작정 따르지 말고 그 바이너리에 맞춰야 한다. 패키지 배포본은
    Gateway 이미지에 고정된 fallback을 설치한다. 로컬/개발 배포는 더 새로운 ``lark-cli``를
    Gateway PATH에 놓고 백엔드를 재시작해 이를 덮어쓸 수 있다.
    """
    cli = probe_lark_cli()
    version = _normalize_version(cli.version)
    return f"v{version}" if version is not None else FALLBACK_LARK_CLI_VERSION


def read_lark_app_config(user_id: str) -> dict[str, str | bool | None]:
    ensure_lark_cli_credential_tree(user_id)
    config_path = lark_cli_config_dir(user_id) / "config.json"
    if not config_path.is_file():
        return {"configured": False, "app_id": None, "brand": None}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"configured": False, "app_id": None, "brand": None}
    if not isinstance(data, dict):
        return {"configured": False, "app_id": None, "brand": None}
    apps = data.get("apps")
    if not isinstance(apps, list) or not apps:
        return {"configured": False, "app_id": None, "brand": None}
    current = data.get("currentApp")
    app = None
    if isinstance(current, str) and current:
        app = next((candidate for candidate in apps if isinstance(candidate, dict) and (candidate.get("name") == current or candidate.get("appId") == current)), None)
    if app is None:
        app = apps[0] if isinstance(apps[0], dict) else None
    if not isinstance(app, dict):
        return {"configured": False, "app_id": None, "brand": None}
    app_id = str(app.get("appId") or "").strip()
    app_secret = app.get("appSecret")
    brand = str(app.get("brand") or "feishu").strip() or "feishu"
    return {"configured": bool(app_id and app_secret), "app_id": app_id or None, "brand": brand}


def install_lark_integration(
    user_id: str,
    config: AppConfig,
    *,
    source_archive: str | Path | None = None,
) -> LarkInstallResult:
    env_archive = os.getenv(LARK_CLI_SOURCE_ARCHIVE_ENV)
    if source_archive is not None:
        archive_path = Path(source_archive)
        resolved_version = None
        created_temp_archive = False
    elif env_archive:
        archive_path = Path(env_archive)
        resolved_version = None
        created_temp_archive = False
    else:
        cli = _ensure_managed_gateway_lark_cli()
        runtime_version = _normalize_version(cli.version)
        resolved_version = f"v{runtime_version}" if runtime_version is not None else FALLBACK_LARK_CLI_VERSION
        archive_path = _download_lark_archive(resolved_version)
        created_temp_archive = True

    previous = _read_manifest(lark_integration_root(user_id))
    previous_content_sha = str(previous.get("content_sha256")) if previous else None
    try:
        installed_skills, content_sha = _install_lark_skills_from_archive(user_id, archive_path, version=resolved_version)
    finally:
        if created_temp_archive:
            try:
                archive_path.unlink(missing_ok=True)
            except OSError:
                pass

    if _uses_aio_sandbox(config) and not _uses_remote_provisioner(config):
        installed_manifest = _read_manifest(lark_integration_root()) or {}
        sandbox_version = str(installed_manifest.get("version") or resolved_version or FALLBACK_LARK_CLI_VERSION)
        _ensure_managed_sandbox_lark_cli(sandbox_version)

    status = get_lark_integration_status(user_id, config)
    content_changed = previous_content_sha is not None and previous_content_sha != content_sha
    message = f"Installed {len(installed_skills)} Lark/Feishu skills."
    if content_changed:
        message += " Skill content changed since the previous install."
    return LarkInstallResult(
        success=True,
        installed_skills=installed_skills,
        status=status,
        message=message,
    )


def _uses_aio_sandbox(config: AppConfig) -> bool:
    sandbox = getattr(config, "sandbox", None)
    use = getattr(sandbox, "use", None)
    if use is None and isinstance(sandbox, dict):
        use = sandbox.get("use")
    return isinstance(use, str) and "aio_sandbox" in use.lower()


def _sandbox_config_value(config: AppConfig, key: str) -> str:
    """sandbox config 값을 문자열로 읽는다. dict/속성 형태의 config를 모두 허용한다."""
    sandbox = getattr(config, "sandbox", None)
    value = getattr(sandbox, key, None)
    if value is None and isinstance(sandbox, dict):
        value = sandbox.get(key)
    return str(value).strip() if value else ""


def _uses_remote_provisioner(config: AppConfig) -> bool:
    """sandbox를 원격 provisioner가 프로비저닝하는 경우(K8s 모드) True."""
    return bool(_sandbox_config_value(config, "provisioner_url"))


def _probe_provisioner_capabilities(config: AppConfig, *, timeout: float = 5.0) -> dict[str, bool] | None:
    """provisioner의 lark-cli capability를 best-effort로 읽는다.

    provisioner가 응답하면 capability dict를, 접근할 수 없으면 None을 반환한다. status의
    준비 여부 신호와 bash hot path에서 broker/binary 모드를 고르는 데 모두 쓴다. 실패는
    예외를 던지지 않고 "준비 안 됨"/"broker 아님"으로 낮춘다. ``timeout``은 호출자가 조절할
    수 있어서, bash 호출마다 도는 probe는 Settings의 status probe보다 더 빡빡한 예산을 쓸
    수 있다.
    """
    base = _sandbox_config_value(config, "provisioner_url")
    if not base:
        return None
    api_key = _sandbox_config_value(config, "provisioner_api_key")
    headers = {"X-API-Key": api_key} if api_key else {}
    url = f"{base.rstrip('/')}/api/capabilities"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "deer-flow", **headers})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        return {
            "lark_cli_init_image": bool(payload.get("lark_cli_init_image")),
            "lark_cli_broker_image": bool(payload.get("lark_cli_broker_image")),
        }
    except Exception:
        return None


def start_lark_config(user_id: str, *, brand: str = "feishu") -> LarkConfigStartResult:
    """이 사용자를 위한 Lark OAuth 앱을 생성/바인딩하는 브라우저 flow를 시작한다."""
    parsed_brand = _normalize_lark_brand(brand)
    with _lark_credential_lock(user_id):
        generation = _advance_lark_flow_generation_locked(user_id)
    begin_data = _request_lark_app_registration_begin(parsed_brand)
    user_code = str(begin_data.get("user_code") or "").strip()
    device_code = str(begin_data.get("device_code") or "").strip()
    if not user_code or not device_code:
        raise ValueError("Lark app registration did not return a user_code and device_code.")
    verification_url = _build_lark_config_verification_url(parsed_brand, user_code)
    return LarkConfigStartResult(
        verification_url=verification_url,
        device_code=device_code,
        generation=generation,
        expires_in=_int_or_none(begin_data.get("expires_in")),
        interval=_int_or_none(begin_data.get("interval")),
        user_code=user_code,
        brand=parsed_brand,
    )


def complete_lark_config(
    user_id: str,
    config: AppConfig,
    *,
    device_code: str,
    generation: str,
    brand: str = "feishu",
    interval: int | None = None,
    expires_in: int | None = None,
) -> LarkConfigCompleteResult:
    """앱 등록을 완료하고 lark-cli를 통해 앱 credential을 저장한다."""
    device_code = device_code.strip()
    if not device_code:
        raise ValueError("device_code is required.")
    parsed_brand = _normalize_lark_brand(brand)
    with _lark_credential_lock(user_id):
        generation = _require_lark_flow_generation_locked(user_id, generation)
    result = _poll_lark_app_registration(
        device_code=device_code,
        brand=parsed_brand,
        interval=interval or 5,
        expires_in=expires_in or 300,
    )
    if not result.get("client_secret") and _tenant_brand(result) == "lark":
        # Lark CLI는 두 brand 모두 Feishu accounts host에서 polling을 시작한다.
        # Lark tenant의 경우 그 응답에 user_info.tenant_brand와 client_id는 있어도
        # client_secret이 빠질 수 있다. 같은 device_code로 Lark accounts host를 polling하면
        # 완전한 앱 credential을 받는다.
        result = _poll_lark_app_registration(
            device_code=device_code,
            brand="lark",
            interval=interval or 5,
            expires_in=expires_in or 300,
        )

    app_id = str(result.get("client_id") or "").strip()
    app_secret = str(result.get("client_secret") or "").strip()
    final_brand = _tenant_brand(result) or parsed_brand
    if not app_id or not app_secret:
        raise ValueError("Lark app registration succeeded but did not return app credentials.")

    with _lark_credential_lock(user_id):
        generation = _require_lark_flow_generation_locked(user_id, generation)
        _replace_lark_app_credentials_locked(
            user_id,
            app_id=app_id,
            app_secret=app_secret,
            brand=final_brand,
        )
        status = get_lark_integration_status(user_id, config)
    return LarkConfigCompleteResult(
        success=True,
        status=status,
        message="Lark/Feishu connection setup completed.",
        generation=generation,
    )


def set_lark_app_credentials(
    user_id: str,
    config: AppConfig,
    *,
    app_id: str,
    app_secret: str,
    brand: str = "feishu",
) -> LarkConfigCompleteResult:
    """이 사용자의 앱을 원자적으로 교체하고 이전 OAuth token을 폐기한다."""
    app_id = app_id.strip()
    app_secret = app_secret.strip()
    if not app_id:
        raise ValueError("app_id is required.")
    if not app_secret:
        raise ValueError("app_secret is required.")
    parsed_brand = brand.strip().lower()
    if parsed_brand not in {"feishu", "lark"}:
        raise ValueError("brand must be feishu or lark.")

    with _lark_credential_lock(user_id):
        _validate_lark_app_credentials_with_cli(app_id=app_id, app_secret=app_secret, brand=parsed_brand)
        generation = _advance_lark_flow_generation_locked(user_id)
        _replace_lark_app_credentials_locked(
            user_id,
            app_id=app_id,
            app_secret=app_secret,
            brand=parsed_brand,
        )
        status = get_lark_integration_status(user_id, config)
    return LarkConfigCompleteResult(
        success=True,
        status=status,
        message="Lark/Feishu app switched. Reconnect to authorize the new app.",
        generation=generation,
    )


def start_lark_auth(
    user_id: str,
    *,
    domains: tuple[str, ...] = (),
    scope: str | None = None,
    recommend: bool = False,
    generation: str | None = None,
) -> LarkAuthStartResult:
    """블로킹하지 않는 Lark device authorization flow를 시작한다.

    반환된 URL은 브라우저 UI나 채팅 메시지에 그대로 보여줘도 안전하다. 사용자가
    Lark/Feishu에서 인증을 마치면 ``device_code``를 :func:`complete_lark_auth`로 돌려줘야
    한다.
    """
    path = _require_lark_cli_path()
    args = [path, "auth", "login", "--no-wait", "--json"]
    if recommend:
        args.append("--recommend")
    if scope:
        args.extend(["--scope", scope])
    for domain in domains:
        if domain:
            args.extend(["--domain", domain])

    with _lark_credential_lock(user_id):
        if generation is None:
            generation = _advance_lark_flow_generation_locked(user_id)
        else:
            generation = _require_lark_flow_generation_locked(user_id, generation)
        data = _run_lark_cli_json(args, user_id=user_id, timeout=20)
        verification_url = str(data.get("verification_url") or data.get("verification_uri_complete") or "").strip()
        device_code = str(data.get("device_code") or "").strip()
        if not verification_url or not device_code:
            raise ValueError("lark-cli did not return a verification_url and device_code.")

    return LarkAuthStartResult(
        verification_url=verification_url,
        device_code=device_code,
        generation=generation,
        expires_in=_int_or_none(data.get("expires_in")),
        user_code=str(data.get("user_code") or "") or None,
        hint=str(data.get("hint") or "") or None,
    )


def complete_lark_auth(
    user_id: str,
    config: AppConfig,
    *,
    device_code: str,
    generation: str,
    wait_timeout_seconds: int = LARK_AUTH_COMPLETE_DEFAULT_WAIT_SECONDS,
) -> LarkAuthCompleteResult:
    """사용자가 승인한 뒤 Lark device authorization flow를 완료한다."""
    device_code = device_code.strip()
    if not device_code:
        raise ValueError("device_code is required.")
    if not LARK_AUTH_COMPLETE_MIN_WAIT_SECONDS <= wait_timeout_seconds <= LARK_AUTH_COMPLETE_MAX_WAIT_SECONDS:
        raise ValueError(f"wait_timeout_seconds must be between {LARK_AUTH_COMPLETE_MIN_WAIT_SECONDS} and {LARK_AUTH_COMPLETE_MAX_WAIT_SECONDS}.")

    with _lark_credential_lock(user_id):
        _require_lark_flow_generation_locked(user_id, generation)
        path = _require_lark_cli_path()
        _run_lark_cli_json(
            [path, "auth", "login", "--device-code", device_code, "--json"],
            user_id=user_id,
            timeout=wait_timeout_seconds,
            allow_empty_success=True,
        )
        status = get_lark_integration_status(user_id, config, verify_auth=True)
    return LarkAuthCompleteResult(
        success=status.auth.status == "authenticated",
        status=status,
        message="Lark/Feishu authorization completed." if status.auth.status == "authenticated" else (status.auth.message or "Lark/Feishu authorization status is still pending."),
    )


def _resolve_lark_cli_path() -> str | None:
    return _lark_cli_managed_path() or shutil.which("lark-cli")


def _ensure_managed_gateway_lark_cli() -> LarkCliProbe:
    """DeerFlow가 관리하는 Gateway lark-cli를 설치/갱신한다.

    비기술 사용자가 터미널에서 ``@larksuite/cli``를 설치할 필요가 없도록 관리자 설치
    엔드포인트에서 호출한다. npm/GitHub에 접근할 수 없어도 이미 사용 가능한 CLI가 있으면
    (관리 대상이든 PATH에 있든) 그것을 계속 쓰고, skill pack 설치가 그 runtime 버전에
    맞추게 한다.
    """
    target_version = _resolve_latest_lark_cli_version()
    current = probe_lark_cli()
    current_version = _normalize_version(current.version)
    if current.available and current_version == _normalize_version(target_version):
        return current

    try:
        return _install_managed_gateway_lark_cli(target_version)
    except Exception:
        fallback = probe_lark_cli()
        if fallback.available:
            logger.warning("Could not update managed lark-cli; using existing Gateway lark-cli", exc_info=True)
            return fallback
        raise


def _install_managed_gateway_lark_cli(version: str) -> LarkCliProbe:
    normalized = _normalize_lark_cli_version_tag(version)
    if normalized is None:
        raise ValueError(f"Invalid Lark CLI npm version: {version!r}")
    npm_version = normalized.removeprefix("v")
    npm = shutil.which("npm")
    if npm is None:
        raise FileNotFoundError("npm is not available on the Gateway; cannot install managed @larksuite/cli.")

    install_root = lark_cli_managed_gateway_dir()
    install_root.mkdir(parents=True, exist_ok=True)
    # NOTE: 여기서는 @larksuite/cli의 install script가 실행된다(postinstall이 플랫폼용
    # lark-cli 바이너리를 받는다). 따라서 `--ignore-scripts`는 쓸 수 없다 — 그것 없이는
    # CLI가 동작하지 않는다. 트레이드오프: 관리자가 트리거한 설치가 Gateway 권한으로 공식
    # 패키지(및 그 의존성)의 install script를 실행하므로, 해당 패키지의 supply-chain
    # 침해가 곧 피해 범위다. Gateway Dockerfile의 버전 고정된 `npm install -g`와 같은
    # 구조이며, 둘 다 관리자 설치 액션 뒤에 있다.
    result = subprocess.run(
        [
            npm,
            "install",
            "--prefix",
            str(install_root),
            "--no-audit",
            "--no-fund",
            f"{LARK_CLI_NPM_PACKAGE}@{npm_version}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=LARK_CLI_NPM_INSTALL_TIMEOUT_SECONDS,
        env={**os.environ, "npm_config_update_notifier": "false"},
    )
    if result.returncode != 0:
        raw = (result.stderr or result.stdout or "").strip()
        raise ValueError(raw or f"npm install {LARK_CLI_NPM_PACKAGE}@{npm_version} exited with code {result.returncode}")

    path = _lark_cli_managed_path()
    if path is None:
        raise FileNotFoundError("Managed lark-cli install completed, but no lark-cli binary was found.")
    probe = _probe_lark_cli_at_path(path)
    if not probe.available:
        raise ValueError(probe.error or "Managed lark-cli install did not produce a runnable CLI.")
    return probe


def _require_lark_cli_path() -> str:
    path = _resolve_lark_cli_path()
    if path is None:
        raise FileNotFoundError("lark-cli is not installed on the Gateway. Install the managed Lark integration as an admin, or rebuild the Gateway image with @larksuite/cli installed.")
    return path


def _normalize_lark_brand(brand: str) -> str:
    return "lark" if brand.strip().lower() == "lark" else "feishu"


def _lark_endpoints(brand: str) -> dict[str, str]:
    if _normalize_lark_brand(brand) == "lark":
        return {
            "open": "https://open.larksuite.com",
            "accounts": "https://accounts.larksuite.com",
        }
    return {
        "open": "https://open.feishu.cn",
        "accounts": "https://accounts.feishu.cn",
    }


def _request_lark_app_registration_begin(brand: str) -> dict[str, Any]:
    # lark-cli는 begin 단계에서 Feishu accounts 엔드포인트를 쓰고, poll 응답이 Lark임을
    # 알릴 때만 tenant brand로 전환한다.
    accounts_url = _lark_endpoints("feishu")["accounts"] + _LARK_APP_REGISTRATION_PATH
    body = urllib.parse.urlencode(
        {
            "action": "begin",
            "archetype": "PersonalAgent",
            "auth_method": "client_secret",
            "request_user_info": "open_id tenant_brand",
        }
    ).encode("utf-8")
    data = _post_lark_form(accounts_url, body)
    if "error" in data:
        raise ValueError(str(data.get("error_description") or data.get("error") or "Lark app registration failed."))
    return data


def _build_lark_config_verification_url(brand: str, user_code: str) -> str:
    base = f"{_lark_endpoints(brand)['open']}/page/cli"
    # lpv/ocv는 인증을 수행하는 *runtime* lark-cli 클라이언트 버전, 즉 서버 쪽 Gateway
    # 바이너리를 반영한다 — 최신 skill pack 버전이 아니다.
    runtime_version = _resolve_runtime_lark_cli_version()
    query = urllib.parse.urlencode(
        {
            "user_code": user_code,
            "lpv": runtime_version,
            "ocv": runtime_version,
            "from": "cli",
        }
    )
    return f"{base}?{query}"


def _poll_lark_app_registration(
    *,
    device_code: str,
    brand: str,
    interval: int,
    expires_in: int,
) -> dict[str, Any]:
    accounts_url = _lark_endpoints(brand)["accounts"] + _LARK_APP_REGISTRATION_PATH
    deadline = time.monotonic() + min(max(expires_in, 1), LARK_CONFIG_POLL_TIMEOUT_SECONDS)
    poll_interval = max(min(interval, 10), 1)
    last_error = "authorization_pending"
    while time.monotonic() < deadline:
        body = urllib.parse.urlencode({"action": "poll", "device_code": device_code}).encode("utf-8")
        data = _post_lark_form(accounts_url, body)
        if not data.get("error") and data.get("client_id"):
            return data
        error = str(data.get("error") or "")
        last_error = str(data.get("error_description") or error or "Lark app registration is still pending.")
        if error == "authorization_pending":
            time.sleep(poll_interval)
            continue
        if error == "slow_down":
            poll_interval = min(poll_interval + 5, 30)
            time.sleep(poll_interval)
            continue
        raise ValueError(last_error)
    raise TimeoutError(f"Lark app registration is still pending: {last_error}")


def _post_lark_form(url: str, body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LARK_HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - 네트워크 경계
        raise ValueError(f"Lark app registration request failed: {exc}") from exc
    parsed = _parse_json_object(raw)
    if parsed is None:
        raise ValueError("Lark app registration returned non-JSON response.")
    return parsed


def _tenant_brand(result: dict[str, Any]) -> str | None:
    user_info = result.get("user_info")
    if not isinstance(user_info, dict):
        return None
    brand = str(user_info.get("tenant_brand") or "").strip().lower()
    return brand if brand in {"feishu", "lark"} else None


def _lark_cli_env_for_directories(*, config_dir: Path, data_dir: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "LARKSUITE_CLI_CONFIG_DIR": str(config_dir),
        "LARKSUITE_CLI_DATA_DIR": str(data_dir),
        "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
        "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
    }
    managed_bin = _lark_cli_managed_bin_dir()
    if _lark_cli_managed_path() is not None:
        env["PATH"] = f"{managed_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    return env


def _run_lark_config_init(*, app_id: str, app_secret: str, brand: str, env: dict[str, str]) -> None:
    path = _require_lark_cli_path()
    try:
        result = subprocess.run(
            [path, "config", "init", "--app-id", app_id, "--app-secret-stdin", "--brand", brand],
            input=app_secret + "\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Timed out while saving Lark connection setup.") from exc
    if result.returncode != 0:
        raw = (result.stderr or result.stdout or "").strip()
        parsed = _parse_json_object(raw)
        message = _auth_error_message(parsed) if parsed else raw
        raise ValueError(message or f"lark-cli config init exited with code {result.returncode}")


def _save_lark_app_config_with_cli(user_id: str, *, app_id: str, app_secret: str, brand: str) -> None:
    try:
        _run_lark_config_init(
            app_id=app_id,
            app_secret=app_secret,
            brand=brand,
            env=lark_cli_env(user_id),
        )
    finally:
        ensure_lark_cli_credential_tree(user_id)


def _validate_lark_app_credentials_with_cli(*, app_id: str, app_secret: str, brand: str) -> None:
    """config init의 실시간 tenant-token probe로 credential을 검증한다."""
    with tempfile.TemporaryDirectory(prefix=".validating-lark-app-") as temp_dir:
        root = Path(temp_dir)
        config_dir = root / "config"
        data_dir = root / "data"
        config_dir.mkdir(mode=0o700)
        data_dir.mkdir(mode=0o700)
        _run_lark_config_init(
            app_id=app_id,
            app_secret=app_secret,
            brand=brand,
            env=_lark_cli_env_for_directories(config_dir=config_dir, data_dir=data_dir),
        )


def _replace_lark_app_credentials_locked(user_id: str, *, app_id: str, app_secret: str, brand: str) -> None:
    ensure_lark_cli_credential_tree(user_id)
    root = _lark_cli_credential_root(user_id)
    with _lark_credential_transaction(user_id, root) as snapshot:
        _save_lark_app_config_with_cli(user_id, app_id=app_id, app_secret=app_secret, brand=brand)
        _clear_directory_contents(lark_cli_data_dir(user_id))
        _revoke_lark_auth_from_snapshot(snapshot)


def _clear_directory_contents(directory: Path) -> None:
    if directory.is_symlink():
        raise ValueError(f"Lark CLI credential path must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


@contextmanager
def _lark_credential_transaction(user_id: str, root: Path):
    """전환 단계가 실패하면 활성 credential 트리를 복원한다."""
    with tempfile.TemporaryDirectory(prefix=".switching-lark-app-", dir=str(root.parent)) as temp_dir:
        snapshot = Path(temp_dir) / "credentials"
        shutil.copytree(root, snapshot, symlinks=False)
        try:
            yield snapshot
        except Exception:
            _restore_lark_credential_tree(root, snapshot)
            ensure_lark_cli_credential_tree(user_id)
            raise


def _restore_lark_credential_tree(root: Path, snapshot: Path) -> None:
    for name in ("config", "data"):
        target = root / name
        _clear_directory_contents(target)
        shutil.copytree(snapshot / name, target, dirs_exist_ok=True, symlinks=False)


def _revoke_lark_auth_from_snapshot(snapshot: Path) -> None:
    data_dir = snapshot / "data"
    if not any(path.is_file() for path in data_dir.rglob("*")):
        return
    path = _require_lark_cli_path()
    try:
        result = subprocess.run(
            [path, "auth", "logout", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=_lark_cli_env_for_directories(config_dir=snapshot / "config", data_dir=data_dir),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Timed out while revoking the previous Lark authorization.") from exc
    if result.returncode != 0:
        raw = (result.stderr or result.stdout or "").strip()
        parsed = _parse_json_object(raw)
        message = _auth_error_message(parsed) if parsed else raw
        raise ValueError(message or f"lark-cli auth logout exited with code {result.returncode}")


def _run_lark_cli_json(
    args: list[str],
    *,
    user_id: str,
    timeout: int,
    allow_empty_success: bool = False,
) -> dict[str, Any]:
    try:
        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=lark_cli_env(user_id),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Timed out waiting for Lark/Feishu authorization. Complete authorization in the browser, then try again.") from exc
    finally:
        # OAuth 명령은 명령 실행 전 환경 guard가 돈 뒤에 새 평문 token 파일을 만들 수
        # 있다. timeout이나 CLI 실패 시에도 Gateway로 제어를 넘기기 전에 모든 파일의 권한을
        # 다시 강화한다.
        ensure_lark_cli_credential_tree(user_id)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    raw = stdout or stderr
    parsed = _parse_json_object(raw)

    if result.returncode != 0:
        message = _auth_error_message(parsed) if parsed else raw
        raise ValueError(message or f"lark-cli exited with code {result.returncode}")

    if not raw and allow_empty_success:
        return {}
    if parsed is None:
        if allow_empty_success:
            return {}
        raise ValueError(raw or "lark-cli did not return JSON output.")
    return parsed


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _auth_error_message(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    error = data.get("error")
    if isinstance(error, dict):
        for key in ("message", "hint", "type"):
            value = error.get(key)
            if value:
                return str(value)
    for key in ("message", "msg", "hint"):
        value = data.get(key)
        if value:
            return str(value)
    return None


def _read_manifest(root: Path) -> dict[str, Any] | None:
    path = root / LARK_CLI_MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _installed_lark_skill_names(root: Path) -> set[str]:
    names: set[str] = set()
    if not root.is_dir():
        return names
    for skill_name in LARK_SKILL_NAMES:
        if (root / skill_name / SKILL_MD_FILE).is_file():
            names.add(skill_name)
    return names


def _enabled_lark_skill_names(user_id: str, config: AppConfig) -> set[str]:
    from deerflow.skills.storage import get_or_new_user_skill_storage

    try:
        storage = get_or_new_user_skill_storage(user_id, app_config=config)
        return {skill.name for skill in storage.load_skills(enabled_only=True) if skill.name in LARK_SKILL_NAME_SET}
    except Exception:
        return set()


def _resolve_latest_lark_cli_version() -> str:
    """가장 최근에 배포된 ``larksuite/cli`` release 태그를 해석한다.

    공식 ``releases/latest`` API를 조회한다. 실패(rate limit, 오프라인, 폐쇄망, 잘못된
    payload)하면 ``FALLBACK_LARK_CLI_VERSION``으로 fallback해서, 설치를 중단하는 대신
    알려진 정상 버전으로 진행할 수 있게 한다.
    """
    try:
        request = urllib.request.Request(
            LARK_CLI_LATEST_RELEASE_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "deer-flow"},
        )
        with urllib.request.urlopen(request, timeout=LARK_HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        tag = str(data.get("tag_name") or "").strip() if isinstance(data, dict) else ""
        version = _normalize_lark_cli_version_tag(tag)
        if version is not None:
            return version
    except Exception:  # noqa: BLE001 - 버전 탐색은 best-effort
        pass
    return FALLBACK_LARK_CLI_VERSION


def _cached_latest_lark_cli_version() -> str | None:
    """상태 표시용 최신 버전을 best-effort로 가져오고 짧은 TTL로 캐시한다.

    실패하면 ``None``을 반환해서 GitHub 장애 때 status 엔드포인트가 UI를 막지 않게 한다.
    설치 경로는 자체 fallback이 있는 :func:`_resolve_latest_lark_cli_version`을 쓴다.
    """
    now = time.monotonic()
    cached = getattr(_cached_latest_lark_cli_version, "_cache", None)
    if cached is not None and now - cached[0] < LARK_CLI_LATEST_VERSION_TTL_SECONDS:
        return cached[1]
    try:
        request = urllib.request.Request(
            LARK_CLI_LATEST_RELEASE_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "deer-flow"},
        )
        with urllib.request.urlopen(request, timeout=LARK_HTTP_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        tag = str(data.get("tag_name") or "").strip() if isinstance(data, dict) else ""
        version = _normalize_lark_cli_version_tag(tag)
    except Exception:  # noqa: BLE001 - 상태 probe는 best-effort
        version = None
    _cached_latest_lark_cli_version._cache = (now, version)  # type: ignore[attr-defined]
    return version


def _lark_archive_url(version: str) -> str:
    tag = _normalize_lark_cli_version_tag(version)
    if tag is None:
        raise ValueError(f"Invalid Lark CLI version tag: {version!r}")
    return f"https://codeload.github.com/{LARK_CLI_GITHUB_REPO}/zip/refs/tags/{tag}"


def _normalize_lark_cli_version_tag(value: str | None) -> str | None:
    tag = (value or "").strip()
    if not _VERSION_TAG_RE.fullmatch(tag):
        return None
    return tag if tag.startswith("v") else f"v{tag}"


def _download_lark_archive(version: str) -> Path:
    fd, archive_name = tempfile.mkstemp(prefix="lark-cli-skills-", suffix=".zip")
    os.close(fd)
    archive_path = Path(archive_name)
    url = _lark_archive_url(version)
    try:
        with urllib.request.urlopen(url, timeout=LARK_CLI_DOWNLOAD_TIMEOUT_SECONDS) as response:
            total = 0
            with archive_path.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > LARK_CLI_MAX_ARCHIVE_BYTES:
                        raise ValueError("Lark CLI source archive is too large.")
                    out.write(chunk)
    except ValueError:
        archive_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - 네트워크 경계
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"Could not download the Lark skill pack ({version}) from GitHub. Check the Gateway's internet access, or pre-stage the archive via {LARK_CLI_SOURCE_ARCHIVE_ENV}.") from exc
    return archive_path


def _content_sha256(root: Path, skill_names: set[str]) -> str:
    """archive 바이트가 아니라 실제로 설치된 skill 내용에 대한 SHA-256.

    호출자는 DeerFlow의 공용 guidance를 주입한 뒤 이 값을 계산하므로, digest는 공식
    추출 파일과 사용자/agent가 실제로 읽는 guidance를 모두 포함한다. 내용이 같으면 GitHub가
    archive를 다시 패킹해도 값이 유지된다. 결정성을 위해 경로와 바이트를 정렬된 순서로
    해싱한다.
    """
    digest = hashlib.sha256()
    for skill_name in sorted(skill_names):
        skill_dir = root / skill_name
        if not skill_dir.is_dir():
            continue
        for file_path in sorted(p for p in skill_dir.rglob("*") if p.is_file()):
            rel = file_path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _infer_lark_archive_version(zf: zipfile.ZipFile) -> str | None:
    """``cli-1.0.65/`` 같은 GitHub 소스 archive 루트에서 버전을 추론한다.

    archive 자체가 release를 명확히 밝히고 있을 때, 폐쇄망/미리 준비한 archive가 fallback
    버전으로 잘못 표시되는 것을 막는다.
    """
    for info in zf.infolist():
        normalized = posixpath.normpath(info.filename.replace("\\", "/"))
        parts = PurePosixPath(normalized).parts
        if not parts:
            continue
        match = re.fullmatch(r"cli-(\d+\.\d+\.\d+)", parts[0])
        if match:
            return f"v{match.group(1)}"
    return None


def _install_lark_skills_from_archive(user_id: str, archive_path: Path, *, version: str | None = None) -> tuple[tuple[str, ...], str]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Lark CLI skills archive not found: {archive_path}")

    parent = get_paths().integration_skills_dir()
    parent.mkdir(parents=True, exist_ok=True)
    with _lark_install_lock(parent):
        return _install_lark_skills_from_archive_locked(archive_path, parent, version=version)


@contextmanager
def _lark_install_lock(parent: Path):
    """전역 pack의 프로세스 간 원자적 교체를 직렬화한다."""
    with _exclusive_install_lock(parent / ".lark-cli.install.lock", _LARK_INSTALL_THREAD_LOCK):
        yield


def _install_lark_skills_from_archive_locked(
    archive_path: Path,
    parent: Path,
    *,
    version: str | None = None,
) -> tuple[tuple[str, ...], str]:
    target = parent / INTEGRATION_ID
    staging_parent = Path(tempfile.mkdtemp(prefix=".installing-lark-cli-", dir=str(parent)))
    staging_target = staging_parent / INTEGRATION_ID
    staging_target.mkdir(parents=True, exist_ok=True)

    backup: Path | None = None
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            archive_version = version or _infer_lark_archive_version(zf)
            extracted = _extract_lark_skills(zf, staging_target)
        _validate_extracted_lark_skills(staging_target, extracted)
        _append_deerflow_lark_shared_guidance(staging_target)
        content_sha = _content_sha256(staging_target, extracted)
        _write_manifest(staging_target, extracted, version=archive_version, content_sha256=content_sha)
        make_skill_tree_sandbox_readable(staging_target)

        if target.exists():
            backup = parent / f".replacing-{INTEGRATION_ID}-{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
        staging_target.rename(target)
        if backup is not None:
            # best-effort: rename 이후 새 skill은 이미 활성 상태이므로, 예전 백업을 지우다
            # 난 일시적 오류가 성공한 설치를 실패로 뒤집어서는 안 된다(``target``이 이미 새
            # 내용으로 존재하므로 except 분기의 복원 guard도 동작하지 않는다).
            shutil.rmtree(backup, ignore_errors=True)
        return tuple(sorted(extracted)), content_sha
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _extract_lark_skills(zf: zipfile.ZipFile, destination: Path) -> set[str]:
    extracted: set[str] = set()
    total_written = 0
    dest_root = destination.resolve()

    for info in zf.infolist():
        if info.is_dir():
            continue
        if is_unsafe_zip_member(info) or is_symlink_member(info):
            raise ValueError(f"Unsafe Lark CLI archive member: {info.filename!r}")

        skill_name, relative = _resolve_lark_skill_member(info.filename)
        if skill_name is None or relative is None:
            continue

        target = dest_root / skill_name / relative
        if not target.resolve().is_relative_to(dest_root):
            raise ValueError(f"Archive member escapes destination: {info.filename!r}")
        target.parent.mkdir(parents=True, exist_ok=True)

        with zf.open(info) as src, target.open("wb") as out:
            first_chunk = True
            while chunk := src.read(65536):
                if first_chunk and is_executable_binary_prefix(chunk):
                    raise ValueError(f"Archive contains executable binary member: {info.filename!r}")
                first_chunk = False
                total_written += len(chunk)
                if total_written > LARK_CLI_MAX_EXTRACTED_BYTES:
                    raise ValueError("Lark CLI skills archive expands to too much data.")
                out.write(chunk)
        extracted.add(skill_name)

    return extracted


def _resolve_lark_skill_member(raw_name: str) -> tuple[str | None, Path | None]:
    normalized = posixpath.normpath(raw_name.replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("../"):
        return None, None
    parts = PurePosixPath(normalized).parts

    if "skills" in parts:
        idx = parts.index("skills")
        if len(parts) <= idx + 2:
            return None, None
        skill_name = parts[idx + 1]
        rel_parts = parts[idx + 2 :]
    elif parts and parts[0] in LARK_SKILL_NAME_SET:
        skill_name = parts[0]
        rel_parts = parts[1:]
    else:
        return None, None

    if skill_name not in LARK_SKILL_NAME_SET or not rel_parts:
        return None, None
    if any(part in {"", ".", ".."} for part in rel_parts):
        raise ValueError(f"Unsafe Lark skill archive member: {raw_name!r}")
    return skill_name, Path(*rel_parts)


def _validate_extracted_lark_skills(root: Path, extracted: set[str]) -> None:
    missing = sorted(set(LARK_SKILL_NAMES) - extracted)
    if missing:
        raise ValueError(f"Lark CLI archive is missing required skills: {', '.join(missing)}")

    for skill_name in LARK_SKILL_NAMES:
        skill_file = root / skill_name / SKILL_MD_FILE
        parsed = parse_skill_file(skill_file, SkillCategory.INTEGRATION, relative_path=Path(INTEGRATION_ID) / skill_name)
        if parsed is None:
            raise ValueError(f"Invalid Lark skill metadata: {skill_name}/{SKILL_MD_FILE}")
        if parsed.name != skill_name:
            raise ValueError(f"Lark skill directory {skill_name!r} declares name {parsed.name!r}")


def _append_deerflow_lark_shared_guidance(root: Path) -> None:
    skill_file = root / "lark-shared" / SKILL_MD_FILE
    content = skill_file.read_text(encoding="utf-8")
    if _DEERFLOW_LARK_SHARED_GUIDANCE_MARKER in content:
        return
    for legacy_marker in _DEERFLOW_LARK_SHARED_GUIDANCE_LEGACY_MARKERS:
        if legacy_marker in content:
            content = content.split(legacy_marker, maxsplit=1)[0].rstrip()
            break
    guidance = f"""

{_DEERFLOW_LARK_SHARED_GUIDANCE_MARKER}

## DeerFlow 授权入口

在 DeerFlow 中，如果 `lark-cli auth status` 或业务命令提示未配置、未登录、token 过期或缺少用户授权：

1. 不要要求用户在终端执行 `lark-cli config init`、`lark-cli auth login` 或 `lark-cli auth login --device-code`。
2. 回复用户这个可点击链接：[打开飞书授权设置](?settings=integrations)。
3. 告诉用户在 **Settings → Integrations → Lark / Feishu CLI** 点击“连接飞书”，在浏览器里完成授权后再回来继续当前任务。
4. 如果错误中包含缺失的 `scope`、`permission_violations` 或建议的 `--domain`，告诉用户在该设置页选择对应权限域（例如日历选择 Calendar），或把具体 scope 填入“Exact OAuth scope / 具体 OAuth scope”后重新授权。

只有在用户明确说明已经完成授权后，才继续调用具体的 `lark-cli` 业务命令。
"""
    skill_file.write_text(content.rstrip() + guidance + "\n", encoding="utf-8")


def _write_manifest(root: Path, installed_skills: set[str], *, version: str | None, content_sha256: str) -> None:
    resolved_version = version or FALLBACK_LARK_CLI_VERSION
    manifest = {
        "provider": INTEGRATION_ID,
        "version": resolved_version,
        "source": _lark_archive_url(resolved_version),
        "content_sha256": content_sha256,
        "installed_at": datetime.now(UTC).isoformat(),
        "skills": sorted(installed_skills),
    }
    (root / LARK_CLI_MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
