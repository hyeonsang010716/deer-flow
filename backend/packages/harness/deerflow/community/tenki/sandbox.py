"""``TenkiSandbox`` — Tenki 클라우드 sandbox를 백엔드로 쓰는 DeerFlow :class:`Sandbox`.

Tenki의 Python SDK(``tenki-sandbox``)는 동기식이라 ``community/boxlite`` 와 달리 이
adapter는 event-loop bridge 없이 SDK를 직접 호출한다. 파일 전송은 Tenki의 네이티브
``sandbox.fs`` API(``read_text`` / ``read_stream`` / ``write_stream`` / ``mkdir`` /
``stat``)를 쓴다. 바이너리에 안전하고 스트리밍이므로 base64/shell 인코딩이 필요 없다.
반면 디렉터리와 내용 *검색*(``list_dir`` / ``glob`` / ``grep``)은 여전히 ``find`` / ``grep``
을 shell로 호출한다. fs API는 한 단계만 보고 내용 검색 기능이 없기 때문이다. 결과는
``community/e2b_sandbox`` 와 같은 방식으로 공용 ``deerflow.sandbox.search`` 헬퍼로 파싱한다.
이 명령들은 busybox에서도 통하는 플래그만 써서 어떤 Tenki base image에서도 동작한다.

Tenki SDK는 모듈 로드 시점에 import하지 않는다(예외 *클래스 이름*만 문자열로 비교한다).
따라서 이 패키지를 import하는 것만으로는 ``tenki-sandbox`` 설치가 필요하지 않고,
provider를 선택해 sandbox를 실제로 만들 때만 필요하다.
"""

from __future__ import annotations

import errno
import logging
import posixpath
import re
import shlex
import threading
from typing import TYPE_CHECKING, Any, TypeVar

from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.search import GrepMatch, path_matches, should_ignore_path, truncate_line

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tenki_sandbox import Sandbox as TenkiClientSandbox
    from tenki_sandbox.fs import SandboxFS

T = TypeVar("T")

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
# Tenki sandbox는 권한 없는 ``tenki`` 사용자(HOME=/home/tenki)로 돌고 ``/mnt`` 는 root 소유라서
# DeerFlow의 ``/mnt/user-data`` virtual prefix에 직접 쓸 수 없다. ``community/e2b_sandbox`` 처럼
# 파일 연산을 이 home dir 아래로 remap한다. (provider가 /mnt/user-data → 여기로 symlink도
# best-effort로 걸어서 문자 그대로의 경로를 쓰는 agent shell 명령도 동작한다.)
DEFAULT_TENKI_HOME_DIR = "/home/tenki"
# fs.write_stream 업로드의 frame 크기.
_STREAM_CHUNK = 1024 * 1024

# 원격 session이 영구히 사라졌음을 뜻하는 Tenki SDK 예외 *클래스 이름*. 이 모듈이
# ``tenki-sandbox`` 없이도 import되도록 문자열로 비교한다. 종료/미존재/닫힘 session은 복구
# 불가이므로 provider가 버리고 다음 호출에서 새로 만든다. 이건 규칙의 이름 기반 절반일 뿐이다.
# _is_terminal_failure는 builtin ConnectionError / BrokenPipeError / EOFError도 isinstance로
# terminal 취급하므로, transport reset이 나면 sandbox를 evict하고 다음 acquire를 cold start한다.
# 의도적인 fail-safe다(reset은 대개 microVM이 사라졌다는 뜻). 대신 일시적 네트워크 장애 한 번에
# warm sandbox를 버리는 비용을 감수한다.
_TERMINAL_ERROR_NAMES = frozenset(
    {
        "SessionTerminatedError",
        "SessionNotFoundError",
        "InvalidStateError",
        "StreamClosedError",
    }
)


class TenkiSandbox(Sandbox):
    """실행 중인 Tenki 클라우드 sandbox로 위임하는 DeerFlow Sandbox adapter.

    Args:
        id: DeerFlow 쪽 sandbox id(provider의 캐시 키).
        sandbox: 이미 시작된 ``tenki_sandbox.Sandbox``. lifecycle은 provider가 소유하며
            이 adapter는 :meth:`close` 에서 종료시킨다.
        default_env: 모든 명령에 병합되는 정적 environment. :meth:`execute_command` 에
            넘긴 호출별 ``env``(request-scoped secrets)가 이를 덮어쓴다.
        home_dir: sandbox 안에서 ``VIRTUAL_PATH_PREFIX``(``/mnt/user-data``)를 뒷받침하는
            쓰기 가능 디렉터리. 기본값은 :data:`DEFAULT_TENKI_HOME_DIR`.
        on_terminal_failure: 선택적 콜백 ``(sandbox_id, reason)``. 연산이 복구 불가한 Tenki
            에러로 실패했을 때 호출되어 provider가 죽은 sandbox를 evict할 수 있게 한다.
    """

    def __init__(
        self,
        id: str,
        sandbox: TenkiClientSandbox,
        *,
        default_env: dict[str, str] | None = None,
        home_dir: str = DEFAULT_TENKI_HOME_DIR,
        on_terminal_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__(id)
        self._sandbox = sandbox
        self._default_env = dict(default_env or {})
        self._home_dir = home_dir.rstrip("/") or "/"
        self._on_terminal_failure = on_terminal_failure
        self._lock = threading.Lock()
        # append의 read-modify-write를 세 fs 연산에 걸쳐 직렬화한다. _lock과 별도의 lock이라
        # 전체 시퀀스를 감싸면서도, provider를 다시 호출하는 연산별 eviction 콜백이
        # 이 lock 아래에서 실행되지 않게 한다.
        self._write_lock = threading.Lock()
        self._closed = False

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @staticmethod
    def _is_terminal_failure(error: Exception) -> bool:
        if isinstance(error, (BrokenPipeError, ConnectionError, EOFError)):
            return True
        return type(error).__name__ in _TERMINAL_ERROR_NAMES

    def close(self) -> None:
        """하위 Tenki session을 종료한다(멱등).

        microVM을 *먼저* 종료하고, session이 실제로 사라진 뒤에야 adapter를 closed로 표시한다.
        그래야 종료 실패가 재시도 가능한 상태로 남고, 실행 중인(과금되는) sandbox가 조용히
        새어 나가지 않는다. terminal session 에러는 이미 사라졌다는 뜻이므로 closed로 친다.
        그 밖의 예외는 호출자가 재시도하거나 알릴 수 있도록 그대로 올린다.
        """
        with self._lock:
            if self._closed:
                return
            sandbox = self._sandbox
        try:
            sandbox.close()
        except Exception as e:
            if not self._is_terminal_failure(e):
                logger.error("Error terminating Tenki sandbox %s: %s", self.id, e)
                raise
            logger.info("Tenki sandbox %s was already gone at close: %s", self.id, e)
        with self._lock:
            self._closed = True

    # ── bridge 헬퍼 ─────────────────────────────────────────────────────

    def _note_failure(self, error: Exception) -> None:
        """연산이 복구 불가한 에러로 실패했으면 이 sandbox를 evict한다."""
        if self._on_terminal_failure is None or not self._is_terminal_failure(error):
            return
        try:
            self._on_terminal_failure(self.id, str(error))
        except Exception:
            logger.exception("Terminal Tenki failure callback errored for %s", self.id)

    def _fs_op(self, op: Callable[[SandboxFS], T]) -> T:
        """네이티브 ``sandbox.fs`` 호출을 실행하고, terminal 에러면 sandbox를 evict한다.

        lock은 fs 조회만이 아니라 ``op`` 전체에 걸쳐 유지해서 같은 sandbox의 동시 호출을
        직렬화한다. Tenki SDK는 community/e2b_sandbox처럼 인스턴스당 connection 하나를 공유한다.
        ``_note_failure`` 는 lock을 푼 *뒤에* 실행한다. 이 콜백은 provider를 다시 호출하고
        provider는 반대 순서(provider → sandbox)로 lock을 잡으므로, 둘을 동시에 쥐면 deadlock이 난다.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("sandbox has been closed")
            fs = self._sandbox.fs
            try:
                return op(fs)
            except Exception as e:
                failure = e
        self._note_failure(failure)
        raise failure

    def _exec(self, *argv: str, env: dict[str, str] | None = None, timeout: float | None = None) -> Any:
        # cwd를 강제하지 않는다. 명령은 community/e2b_sandbox, community/boxlite처럼 sandbox의
        # 기본 작업 디렉터리에서 돈다. 파일 연산은 home으로 remap된 절대 경로를 쓰므로 cwd와 무관하다.
        #
        # 자동 재시도도 하지 않는다. exec는 멱등이 아니다(transport ack가 유실되기 전에 서버에서
        # 이미 실행됐을 수 있다). 다시 돌리면 부작용이 두 번 날 위험이 있다. boxlite처럼 일시적
        # 에러는 호출자에게 그대로 노출하고(execute_command가 텍스트로 반환), terminal session
        # 에러는 추가로 sandbox를 evict해서 다음 acquire가 새로 만들게 한다.
        with self._lock:
            if self._closed:
                raise RuntimeError("sandbox has been closed")
            sandbox = self._sandbox
        try:
            return sandbox.exec(*argv, env=env, timeout=timeout)
        except Exception as e:
            self._note_failure(e)
            raise

    def _sh(self, script: str, env: dict[str, str] | None = None, timeout: float | None = None) -> Any:
        return self._exec("sh", "-lc", script, env=env, timeout=timeout)

    # ── 경로 안전성 (community/e2b_sandbox와 동일) ────────────────────────

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
        """DeerFlow virtual path를 쓰기 가능한 sandbox home dir로 매핑한다.

        ``VIRTUAL_PATH_PREFIX``(``/mnt/user-data``)는 :attr:`_home_dir` 아래로 재작성한다.
        그 외 절대 경로는 그대로 통과시켜 필요할 때 시스템 디렉터리에 접근할 수 있게 한다.
        경로 traversal은 항상 거부한다.
        """
        normalized = self._guard_traversal(path)
        if normalized == VIRTUAL_PATH_PREFIX or normalized.startswith(f"{VIRTUAL_PATH_PREFIX}/"):
            tail = normalized[len(VIRTUAL_PATH_PREFIX) :].lstrip("/")
            return f"{self._home_dir}/{tail}".rstrip("/") if tail else self._home_dir
        return normalized

    def _virtual_path(self, resolved: str) -> str:
        """:meth:`_resolve_path` 의 역변환. 호출자가 넘겨준 형태로 되돌린다.

        경로를 *반환하는* 모든 API(``list_dir``/``glob``/``grep``)는 sandbox 내부 home dir이
        아니라 ``VIRTUAL_PATH_PREFIX`` 기준으로 보고한다. 그래야 결과를 다른 파일 API에
        그대로 다시 넣을 수 있다.
        """
        if resolved == self._home_dir:
            return VIRTUAL_PATH_PREFIX
        if resolved.startswith(f"{self._home_dir}/"):
            return f"{VIRTUAL_PATH_PREFIX}/{resolved[len(self._home_dir) :].lstrip('/')}"
        return resolved

    # ── 명령 실행 ───────────────────────────────────────────────────────

    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Tenki sandbox의 shell에서 ``command`` 를 실행하고 출력을 반환한다.

        DeerFlow는 bash 명령 *문자열*을 넘기며, 이는 ``sh -lc`` 로 실행된다.
        호출별 ``env`` 는 정적 설정 environment 위에 덧씌워지고 이 명령에만 적용된다
        (request-scoped secrets, issue #3861).
        """
        _validate_extra_env(env)  # POSIX env-var 키 규칙. 잘못된 키면 ValueError
        if self.is_closed:
            return "Error: sandbox has been closed"
        merged_env = {**self._default_env, **(env or {})} or None
        try:
            result = self._sh(command, env=merged_env, timeout=timeout)
        except Exception as e:
            logger.error("Failed to execute command in Tenki sandbox %s: %s", self.id, e)
            return f"Error: {e}"

        stdout = result.stdout_text or ""
        stderr = result.stderr_text or ""
        if stdout and stderr:
            output = f"{stdout}\n{stderr}"
        else:
            output = stdout or stderr
        if result.exit_code not in (0, None) and not output:
            output = f"Command exited with code {result.exit_code}"
        return output if output else "(no output)"

    # ── 파일 연산 ───────────────────────────────────────────────────────

    def read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        try:
            return self._fs_op(lambda fs: fs.read_text(resolved))
        except Exception as e:
            logger.error("read_file %s failed: %s", resolved, e)
            return f"Error: {e}"

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        self._write_bytes(self._resolve_path(path), content.encode("utf-8"), append=append)

    def update_file(self, path: str, content: bytes) -> None:
        self._write_bytes(self._resolve_path(path), content, append=False)

    def _write_bytes(self, resolved: str, data: bytes, *, append: bool) -> None:
        parent = posixpath.dirname(resolved)
        if not append:
            if parent:
                self._fs_op(lambda fs: fs.mkdir(parent))
            self._fs_op(lambda fs: fs.write_stream(resolved, _frames(data)))
            return

        # Tenki의 write stream에는 append 모드가 없어(항상 offset 0부터 시작)
        # community/e2b_sandbox처럼 read-modify-write를 한다. read와 write가 별도의 fs 연산이라
        # 동시 append 두 개가 같은 이전 내용을 읽고 뒤쪽이 앞쪽을 덮어쓸 수 있다.
        # _write_lock이 전체 시퀀스를 원자적으로 만든다.
        with self._write_lock:
            if parent:
                self._fs_op(lambda fs: fs.mkdir(parent))
            try:
                data = self._fs_op(lambda fs: fs.read_bytes(resolved)) + data
            except Exception as e:
                if type(e).__name__ != "FileNotFoundError":
                    raise
            self._fs_op(lambda fs: fs.write_stream(resolved, _frames(data)))

    def download_file(self, path: str) -> bytes:
        normalized = self._guard_traversal(path)
        stripped = normalized.lstrip("/")
        allowed = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped != allowed and not stripped.startswith(f"{allowed}/"):
            raise PermissionError(f"Access denied: path must be under '{VIRTUAL_PATH_PREFIX}': '{path}'")
        resolved = self._resolve_path(path)

        with self._lock:
            if self._closed:
                raise RuntimeError("sandbox has been closed")
            fs = self._sandbox.fs

        # 의도적으로, 연산 내내 lock을 쥐는 _fs_op와 달리 스트리밍 전에 lock을 푼다.
        # _fs_op의 직렬화는 짧고 유한한 호출을 보호하는 용도다. 다운로드는 최대
        # _MAX_DOWNLOAD_SIZE(100 MB)까지 갈 수 있어, 전송 내내 인스턴스 lock을 쥐면 이 sandbox의
        # 다른 모든 도구가 막힌다. Tenki read stream은 다른 연산과 함께 돌아도 안전하므로
        # (SDK가 connection을 multiplexing한다) 지연 시간을 위해 인터리브를 허용하고,
        # terminal transport 에러는 아래 _note_failure로 여전히 evict한다.
        #
        # 상한은 실제로 수신한 바이트로 검사한다. 그래야 전송 도중 커지는 파일도 상한을
        # 넘길 수 없다(stat 후 read 방식이면 넘길 수 있다).
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in fs.read_stream(resolved):
                total += len(chunk)
                if total > _MAX_DOWNLOAD_SIZE:
                    raise OSError(errno.EFBIG, f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes", path)
                chunks.append(chunk)
        except OSError as e:
            # 우리가 직접 던진 EFBIG 크기 상한은 session이 죽은 게 아니므로 evict 없이 통과시킨다.
            # 나머지 OSError는 실제 transport 실패다. ConnectionError / BrokenPipeError /
            # EOFError는 OSError 하위 클래스이고 _is_terminal_failure가 terminal로 보므로
            # _fs_op/_exec처럼 _note_failure를 거쳐야 한다. 그러지 않으면 다운로드 도중 죽은
            # session이 영영 evict되지 않고, 다른 연산이 우연히 회수할 때까지 agent가 계속
            # OSError를 맞는다.
            if e.errno == errno.EFBIG:
                raise
            self._note_failure(e)
            raise
        except Exception as e:
            self._note_failure(e)
            raise OSError(f"cannot read '{path}' from sandbox: {e}") from e
        return b"".join(chunks)

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        resolved = self._resolve_path(path)
        r = self._sh(f"find {shlex.quote(resolved)} -maxdepth {int(max_depth)} \\( -type f -o -type d \\) 2>/dev/null | head -500")
        return [self._virtual_path(line.strip()) for line in (r.stdout_text or "").splitlines() if line.strip()]

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
        for entry in (r.stdout_text or "").splitlines():
            entry = entry.strip()
            if not entry or (entry != root and not entry.startswith(root_prefix)):
                continue
            if should_ignore_path(entry):
                continue
            rel_path = entry[len(root) :].lstrip("/")
            if not rel_path:
                continue
            if path_matches(pattern, rel_path):
                matches.append(self._virtual_path(entry))
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
        # 경계에서 regex 패턴을 검증한다(grep은 POSIX ERE를 쓰지만 이 검사로 큰 오류는 잡힌다).
        # literal이면 검증이 필요 없다. grep에는 원본 패턴을 그대로 넘기며,
        # -F는 문자 그대로, -E는 regex로 매칭한다.
        if not literal:
            re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)

        resolved = self._resolve_path(path)
        # busybox와 GNU 양쪽에서 통하는 플래그만 쓴다. -r 재귀, -H 항상 파일명 출력
        # (없으면 단일 파일로 해석되는 경로에 grep -r을 걸었을 때 "line:text"만 나와서
        # 아래 file:line:text 언패킹이 모든 매치를 버린다), -n 줄 번호, -I 바이너리 건너뛰기,
        # -E/-F regex vs 고정 문자열. --include와 -m은 busybox 호환을 위해 쓰지 않고,
        # glob 범위 제한과 결과 상한은 아래 Python에서 적용한다.
        flags = ["-r", "-H", "-n", "-I"]
        if not case_sensitive:
            flags.append("-i")
        flags.append("-F" if literal else "-E")
        total_cap = max(max_results * 4, max_results + 50)
        cmd = "grep " + " ".join(flags) + f" -e {shlex.quote(pattern)} {shlex.quote(resolved)} 2>/dev/null | head -{total_cap}"
        r = self._sh(cmd)

        root = resolved.rstrip("/") or "/"
        root_prefix = root if root == "/" else f"{root}/"
        matches: list[GrepMatch] = []
        truncated = False
        for raw in (r.stdout_text or "").splitlines():
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
            if glob is not None:
                # 호출자가 지정한 실제 디렉터리 범위에 맞춘다. "src/*.js" 같은 패턴이
                # 트리 전체의 *.js로 넓어지면 안 된다. 위 glob()과 같은 헬퍼, 같은
                # root 상대 경로 의미를 쓴다.
                if file_path != root and not file_path.startswith(root_prefix):
                    continue
                rel_path = posixpath.basename(file_path) if file_path == root else file_path[len(root) :].lstrip("/")
                if not path_matches(glob, rel_path):
                    continue
            matches.append(GrepMatch(path=self._virtual_path(file_path), line_number=line_number, line=truncate_line(line_text)))
            if len(matches) >= max_results:
                truncated = True
                break
        return matches, truncated


def _frames(data: bytes) -> Iterator[bytes]:
    """``data`` 를 ``fs.write_stream`` 용 업로드 frame으로 자른다."""
    for i in range(0, len(data), _STREAM_CHUNK):
        yield data[i : i + _STREAM_CHUNK]
