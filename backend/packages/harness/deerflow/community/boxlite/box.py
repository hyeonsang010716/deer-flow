"""``BoxliteBox`` — BoxLite micro-VM 기반 DeerFlow :class:`Sandbox`.

DeerFlow의 ``Sandbox`` 계약은 동기지만 BoxLite SDK는 async 네이티브이고 box handle은
event loop에 종속된다. provider(:mod:`.provider`)가 daemon thread 위에 전용 asyncio loop를
하나 두고, 각 coroutine을 ``run_coroutine_threadsafe``로 그 loop에 넘기는 ``run`` callable을
주입한다. 덕분에 모든 연산이 box를 시작한 loop에서 실행되며, DeerFlow가 어떤
``asyncio.to_thread`` worker에서 호출하든 안전하다.

모든 연산은 box 안에서 실행되는 shell 명령(``cat`` / ``find`` / ``grep`` / 분할 ``base64``)이며,
공용 ``deerflow.sandbox.search`` 헬퍼로 파싱한다. ``community/e2b_sandbox``와 같은 exec 기반
방식이다. 어떤 OCI 이미지에서도 동작하도록 busybox 호환 플래그만 쓴다.
"""

from __future__ import annotations

import base64
import errno
import logging
import posixpath
import re
import shlex
import threading
from typing import TYPE_CHECKING, TypeVar

from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.search import GrepMatch, path_matches, should_ignore_path, truncate_line

if TYPE_CHECKING:
    from collections.abc import Callable

    from boxlite import SimpleBox

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
# base64 chunk 하나는 Linux MAX_ARG_STRLEN(argv 항목당 128 KiB)보다 충분히 작아야 한다.
# 60000은 4의 배수라 각 chunk가 자체 완결된 base64 단위가 되고, 디코딩된 바이트를
# 이어 붙여도 손실이 없다.
_B64_CHUNK = 60000


class BoxliteBox(Sandbox):
    """실행 중인 BoxLite ``SimpleBox``에 위임하는 adapter.

    Args:
        id: DeerFlow 쪽 sandbox id(BoxLite box id와 동일).
        box: 이미 시작된 async ``SimpleBox``. lifecycle은 provider가 소유하고, 이
            adapter는 :meth:`close`에서 중지시킨다.
        run: provider의 전용 loop에서 coroutine을 실행하고 결과를 반환한다
            (호출 thread를 block한다).
        default_env: 모든 명령에 병합되는 정적 환경 변수. 호출별 ``env``
            (request 범위 secret)가 우선한다.
    """

    TERMINAL_ERROR_MARKERS = (
        "vsock",
        "disconnected",
        "broken pipe",
        "connection reset",
        "connection refused",
        "no such box",
        "box has been stopped",
        "engine reported an error",
    )
    RETRYABLE_ERROR_MARKERS = (
        "transport not ready",
        "retry later",
        "temporarily unavailable",
        "resource busy",
    )

    def __init__(
        self,
        id: str,
        box: SimpleBox,
        run: Callable[..., T],
        *,
        default_env: dict[str, str] | None = None,
        on_terminal_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__(id)
        self._box = box
        self._run = run
        self._default_env = dict(default_env or {})
        self._on_terminal_failure = on_terminal_failure
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def _is_terminal_box_failure(cls, error: Exception) -> bool:
        if isinstance(error, (BrokenPipeError, ConnectionError, EOFError)):
            return True
        if not isinstance(error, RuntimeError | OSError):
            return False
        msg = str(error).lower()
        if any(marker in msg for marker in cls.RETRYABLE_ERROR_MARKERS):
            return False
        return any(marker in msg for marker in cls.TERMINAL_ERROR_MARKERS)

    # ── bridge 헬퍼 ─────────────────────────────────────────────────────

    def _exec(
        self,
        *argv: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("sandbox has been closed")
                box = self._box
            return self._run(box.exec(*argv, env=env, timeout=timeout), timeout=timeout)
        except Exception as e:
            if self._on_terminal_failure is not None and self._is_terminal_box_failure(e):
                try:
                    self._on_terminal_failure(self.id, str(e))
                except Exception:
                    logger.exception("Terminal BoxLite failure callback errored for %s", self.id)
            raise

    def _sh(
        self,
        script: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        return self._exec("sh", "-lc", script, env=env, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._run(self._box.stop())
        except Exception as e:
            logger.warning("Error stopping BoxLite box %s: %s", self.id, e)

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    # ── 경로 안전성 검사 (community/e2b_sandbox와 동일) ──────────────────

    @staticmethod
    def _guard_traversal(path: str) -> str:
        if not path:
            raise ValueError("path must be a non-empty string")
        normalized = path.replace("\\", "/")
        for segment in normalized.split("/"):
            if segment == "..":
                raise PermissionError(f"Access denied: path traversal detected in '{path}'")
        return normalized

    def _resolve_path(self, path: str) -> str:
        # provider가 box rootfs에 /mnt/user-data prefix를 미리 만들어 두므로
        # DeerFlow의 virtual path를 그대로 쓴다. 여기서는 traversal만 막는다.
        return self._guard_traversal(path)

    # ── 명령 실행 ───────────────────────────────────────────────────────

    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """box 안의 shell로 ``command``를 실행하고 출력을 반환한다.

        DeerFlow는 bash 명령을 *문자열*로 넘기지만 BoxLite ``exec``은 argv를 받으므로
        ``sh -lc``를 거쳐 실행한다. 호출별 ``env``는 정적 config 환경 위에 덮이며 이
        명령에만 적용된다.

        *timeout*은 두 계층 모두를 제한한다. BoxLite SDK의 ``exec(timeout=...)``이 VM 안의
        명령 timeout을 처리하고, event loop bridge도 같은 값을 받아
        ``run_coroutine_threadsafe(...).result(timeout)``이 SDK future가 끝내 완료되지 않아도
        호출자를 영원히 block하지 않게 한다.
        """
        _validate_extra_env(env)  # POSIX 환경 변수 키 규칙. 잘못된 키면 ValueError를 던진다.
        if self.is_closed:
            return "Error: sandbox has been closed"
        merged_env = {**self._default_env, **(env or {})} or None
        try:
            result = self._exec("sh", "-lc", command, env=merged_env, timeout=timeout)
        except Exception as e:
            logger.error("Failed to execute command in BoxLite box %s: %s", self.id, e)
            return f"Error: {e}"

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stdout and stderr:
            output = f"{stdout}\n{stderr}"
        else:
            output = stdout or stderr
        if result.exit_code not in (0, None) and not output:
            output = f"Command exited with code {result.exit_code}"
        return output if output else "(no output)"

    # ── 파일 연산 ───────────────────────────────────────────────────────

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        resolved = self._resolve_path(path)
        try:
            r = self._exec("cat", "--", resolved)
        except Exception as e:
            logger.error("read_file %s failed: %s", resolved, e)
            return f"Error: {e}"
        if r.exit_code not in (0, None):
            return f"Error: {(r.stderr or '').strip() or 'cannot read file'}"
        content = r.stdout or ""
        if start_line is None and end_line is None:
            return content
        lines = content.splitlines()
        start = start_line or 1
        end = end_line if end_line is not None else len(lines)
        return "\n".join(lines[start - 1 : end])

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        self._write_bytes(self._resolve_path(path), content.encode("utf-8"), append=append)

    def update_file(self, path: str, content: bytes) -> None:
        self._write_bytes(self._resolve_path(path), content, append=False)

    def _write_bytes(self, resolved: str, data: bytes, *, append: bool) -> None:
        parent = posixpath.dirname(resolved)
        if parent:
            mk = self._sh(f"mkdir -p {shlex.quote(parent)}")
            if mk.exit_code not in (0, None):
                raise OSError(f"cannot create parent of '{resolved}': {(mk.stderr or '').strip()}")

        b64 = base64.b64encode(data).decode("ascii")
        if not b64:  # 빈 파일이면 파이프 없이 생성/비우기만 한다
            r = self._sh(f": {'>>' if append else '>'} {shlex.quote(resolved)}")
            if r.exit_code not in (0, None):
                raise OSError(f"write '{resolved}' failed: {(r.stderr or '').strip()}")
            return

        first = True
        for i in range(0, len(b64), _B64_CHUNK):
            chunk = b64[i : i + _B64_CHUNK]
            redir = ">>" if (append or not first) else ">"
            r = self._sh(f"printf %s {shlex.quote(chunk)} | base64 -d {redir} {shlex.quote(resolved)}")
            if r.exit_code not in (0, None):
                raise OSError(f"write '{resolved}' failed: {(r.stderr or '').strip()}")
            first = False

    def download_file(self, path: str) -> bytes:
        normalized = self._guard_traversal(path)
        stripped = normalized.lstrip("/")
        allowed = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped != allowed and not stripped.startswith(f"{allowed}/"):
            raise PermissionError(f"Access denied: path must be under '{VIRTUAL_PATH_PREFIX}': '{path}'")

        # 전체 payload를 버퍼링하기 전에 크기 상한을 먼저 확인한다.
        size_r = self._sh(f"wc -c < {shlex.quote(normalized)}")
        if size_r.exit_code not in (0, None):
            raise OSError(f"cannot read '{path}' from box: {(size_r.stderr or '').strip() or 'not found'}")
        try:
            size = int((size_r.stdout or "0").strip() or "0")
        except ValueError:
            size = 0
        if size > _MAX_DOWNLOAD_SIZE:
            raise OSError(errno.EFBIG, f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes", path)

        r = self._sh(f"base64 {shlex.quote(normalized)}")
        if r.exit_code not in (0, None):
            raise OSError(f"cannot read '{path}' from box: {(r.stderr or '').strip()}")
        try:
            return base64.b64decode("".join((r.stdout or "").split()))
        except Exception as e:
            raise OSError(f"failed to decode '{path}' from box: {e}") from e

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        resolved = self._resolve_path(path)
        r = self._sh(f"find {shlex.quote(resolved)} -maxdepth {int(max_depth)} \\( -type f -o -type d \\) 2>/dev/null | head -500")
        return [line.strip() for line in (r.stdout or "").splitlines() if line.strip()]

    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        resolved = self._resolve_path(path)
        types = ("f", "d") if include_dirs else ("f",)
        type_expr = " -o ".join(f"-type {t}" for t in types)
        hard_limit = max(max_results * 4, max_results + 50)
        r = self._sh(f"find {shlex.quote(resolved)} \\( {type_expr} \\) -print 2>/dev/null | head -{hard_limit}")

        matches: list[str] = []
        root = resolved.rstrip("/") or "/"
        root_prefix = root if root == "/" else f"{root}/"
        for entry in (r.stdout or "").splitlines():
            entry = entry.strip()
            if not entry or (entry != root and not entry.startswith(root_prefix)):
                continue
            if should_ignore_path(entry):
                continue
            rel_path = entry[len(root) :].lstrip("/")
            if not rel_path:
                continue
            if path_matches(pattern, rel_path):
                matches.append(entry)
                if len(matches) >= max_results:
                    return matches, True
        return matches, False

    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        # 경계에서 regex pattern을 Python regex로 한 번 검사한다(grep은 POSIX ERE를 쓰지만
        # 명백한 오류는 이걸로 걸린다). literal은 검증할 필요가 없다.
        # grep에는 원본 pattern을 그대로 넘긴다. -F는 literal로, -E는 regex로 매칭한다.
        if not literal:
            re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)

        resolved = self._resolve_path(path)
        # busybox+GNU 양쪽에서 통하는 플래그만 쓴다. -r 재귀, -H 항상 파일명 출력
        # (path가 단일 파일일 때도), -n 줄 번호, -I 바이너리 건너뛰기, -E/-F regex 대 고정 문자열.
        # --include와 -m은 busybox 호환성 때문에 빼고, glob 범위 제한과 결과 상한은
        # Python 쪽에서 적용한다.
        flags = ["-r", "-H", "-n", "-I"]
        if not case_sensitive:
            flags.append("-i")
        flags.append("-F" if literal else "-E")
        total_cap = max(max_results * 4, max_results + 50)
        cmd = "grep " + " ".join(flags) + f" -e {shlex.quote(pattern)} {shlex.quote(resolved)} 2>/dev/null | head -{total_cap}"
        r = self._sh(cmd)

        include = glob.split("/")[-1] if glob else None
        matches: list[GrepMatch] = []
        truncated = False
        for raw in (r.stdout or "").splitlines():
            try:
                file_path, line_no_str, line_text = raw.split(":", 2)
            except ValueError:
                continue
            try:
                line_number = int(line_no_str)
            except ValueError:
                continue
            if should_ignore_path(file_path):
                continue
            if include and not path_matches(include, posixpath.basename(file_path)):
                continue
            matches.append(GrepMatch(path=file_path, line_number=line_number, line=truncate_line(line_text)))
            if len(matches) >= max_results:
                truncated = True
                break
        return matches, truncated
