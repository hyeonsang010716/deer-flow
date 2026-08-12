from __future__ import annotations

import errno
import logging
import re
import shlex
import threading

from e2b_code_interpreter import Sandbox as E2BClientSandbox

from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.search import GrepMatch, path_matches, should_ignore_path, truncate_line

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

# DeerFlow의 ``/mnt/user-data`` 가상 prefix가 e2b sandbox 안에서 실제로 위치하는 곳.
# e2b code-interpreter 템플릿의 기본 작업 디렉터리는 ``/home/user``다.
DEFAULT_E2B_HOME_DIR = "/home/user"

_E2B_NOT_FOUND_SIGNATURES = (
    "sandbox was not found",
    "sandbox not found",
    "paused sandbox",
)


def _is_sandbox_gone_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _E2B_NOT_FOUND_SIGNATURES)


class E2BSandbox(Sandbox):
    """e2b 클라우드 sandbox에 위임하는 DeerFlow Sandbox 어댑터.

    Args:
        id: DeerFlow 쪽 sandbox id. provider에서 캐시 키로 쓴다.
        client: 살아 있는 ``e2b_code_interpreter.Sandbox``(sync) 인스턴스.
            연결의 소유권과 ``kill()`` 책임은 호출자에게 있다. 이 wrapper는
            release 시 호스트 쪽 HTTP client에 ``close()``만 호출한다.
        home_dir: sandbox 안에서 ``VIRTUAL_PATH_PREFIX``(``/mnt/user-data``)를
            받쳐 주는 디렉터리. 기본값은 :data:`DEFAULT_E2B_HOME_DIR`.
    """

    def __init__(
        self,
        id: str,
        client: E2BClientSandbox,
        *,
        home_dir: str = DEFAULT_E2B_HOME_DIR,
    ) -> None:
        super().__init__(id)
        self._client = client
        self._home_dir = home_dir.rstrip("/") or "/"
        self._lock = threading.Lock()
        self._closed = False
        self._dead = False

    # ── 프로퍼티 / lifecycle ─────────────────────────────────────────────

    @property
    def client(self) -> E2BClientSandbox:
        return self._client

    @property
    def home_dir(self) -> str:
        return self._home_dir

    @property
    def sandbox_id(self) -> str:
        """e2b 쪽 sandbox id. DeerFlow의 ``self.id`` 캐시 키와는 다르다."""
        return getattr(self._client, "sandbox_id", None) or self.id

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None

        if client is None:
            return

        for closer in (
            getattr(client, "close", None),
            getattr(getattr(client, "_transport", None), "close", None),
        ):
            if callable(closer):
                try:
                    closer()
                except Exception as e:
                    logger.warning("Error closing E2BSandbox %s: %s", self.id, e)
                return

    def _resolve_path(self, path: str) -> str:
        """DeerFlow 가상 경로를 e2b sandbox 파일시스템 경로로 변환한다.

        ``VIRTUAL_PATH_PREFIX``(``/mnt/user-data``)는 :attr:`home_dir` 아래로
        재작성한다. ``LocalContainerBackend``가 호스트 workspace를 AIO 컨테이너의
        ``/mnt/user-data``에 bind-mount하는 것과 같은 구조다.
        그 밖의 절대 경로는 그대로 돌려주어 sandbox가 필요할 때 시스템 디렉터리
        (``/tmp``, ``/etc`` 등)에 접근할 수 있게 한다.
        """
        if not path:
            raise ValueError("path must be a non-empty string")
        normalised = path.replace("\\", "/")
        for segment in normalised.split("/"):
            if segment == "..":
                raise PermissionError(f"Access denied: path traversal detected in '{path}'")
        if normalised == VIRTUAL_PATH_PREFIX or normalised.startswith(f"{VIRTUAL_PATH_PREFIX}/"):
            tail = normalised[len(VIRTUAL_PATH_PREFIX) :].lstrip("/")
            return f"{self._home_dir}/{tail}".rstrip("/") if tail else self._home_dir
        return normalised

    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """``sandbox.commands.run``으로 셸 명령을 실행한다.

        stdout과 stderr를 합쳐 반환한다. e2b SDK는 sandbox마다 HTTP/2 연결
        하나를 공유하므로, lock으로 같은 인스턴스의 동시 호출을 직렬화한다.

        Args:
            command: 실행할 명령.
            env: 호출 단위 환경 변수(request-scoped secret, issue #3861). local/AIO
                sandbox와 공유하는 POSIX 환경 변수 이름 규칙으로 검증한 뒤 e2b에
                ``envs``로 전달한다. 이 명령에만 적용되며 명령 문자열에는 절대
                들어가지 않는다.
            timeout: 호출 단위 명령 타임아웃(초). ``None``이면 e2b SDK 기본값(60초)을 쓴다.
        """
        _validate_extra_env(env)
        with self._lock:
            client = self._client
            if client is None:
                return "Error: sandbox client has been closed"
            if self._dead:
                return "Error: e2b sandbox has been reaped by the control plane (idle timeout or explicit pause). The provider will rebuild a fresh sandbox on the next tool call."
            try:
                kwargs: dict[str, object] = {}
                if env is not None:
                    kwargs["envs"] = env
                if timeout is not None:
                    kwargs["timeout"] = timeout
                result = client.commands.run(command, **kwargs)
                stdout = getattr(result, "stdout", "") or ""
                stderr = getattr(result, "stderr", "") or ""
                exit_code = getattr(result, "exit_code", 0)
                if stdout and stderr:
                    output = f"{stdout}\n{stderr}"
                else:
                    output = stdout or stderr
                if exit_code not in (0, None) and not output:
                    output = f"Command exited with code {exit_code}"
                return output if output else "(no output)"
            except Exception as e:
                if _is_sandbox_gone_error(e):
                    self._dead = True
                logger.error("Failed to execute command in e2b sandbox: %s", e)
                return f"Error: {e}"

    @property
    def is_dead(self) -> bool:
        """하위 e2b VM이 회수된 것으로 확인되었는지 여부.

        ``execute_command``와 provider의 ``ping``/bootstrap 호출이 lazy하게
        갱신한다. 선제적인 heartbeat는 없다. 이 값을 읽어도 API를 왕복하지 *않는다*.
        """
        with self._lock:
            return self._dead

    def ping(self) -> bool:
        """가벼운 health check. e2b VM이 회수되었으면 False를 반환한다.

        ``commands.run("true")``로 실행하므로, 성공하면 전체 HTTP 경로
        (auth + control plane + envd)가 살아 있다는 뜻이다.
        :func:`_is_sandbox_gone_error`가 인식하는 "sandbox not found" 시그니처가
        나오면 ``_dead = True``로 표시해 이후 호출을 곧바로 끊는다.
        """
        with self._lock:
            if self._dead or self._client is None:
                return False
            client = self._client
        try:
            client.commands.run("true")
            return True
        except Exception as e:
            if _is_sandbox_gone_error(e):
                with self._lock:
                    self._dead = True
                return False
            logger.warning("e2b sandbox ping raised non-fatal error: %s", e)
            return True

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        resolved = self._resolve_path(path)
        try:
            content = self._client.files.read(resolved)
            if start_line is None and end_line is None:
                return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content or ""
            text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content or ""
            lines = text.splitlines()
            start = start_line or 1
            end = end_line if end_line is not None else len(lines)
            content = "\n".join(lines[start - 1 : end])
            return content
        except Exception as e:
            logger.error("Failed to read file %s in e2b sandbox: %s", resolved, e)
            return f"Error: {e}"

    def download_file(self, path: str) -> bytes:
        normalised = path.replace("\\", "/")
        for segment in normalised.split("/"):
            if segment == "..":
                logger.error("Refused download due to path traversal: %s", path)
                raise PermissionError(f"Access denied: path traversal detected in '{path}'")

        stripped_path = normalised.lstrip("/")
        allowed_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped_path != allowed_prefix and not stripped_path.startswith(f"{allowed_prefix}/"):
            logger.error(
                "Refused download outside allowed directory: path=%s, allowed_prefix=%s",
                path,
                VIRTUAL_PATH_PREFIX,
            )
            raise PermissionError(f"Access denied: path must be under '{VIRTUAL_PATH_PREFIX}': '{path}'")

        resolved = self._resolve_path(path)
        # 100MB 상한을 gateway 프로세스가 전체 payload를 버퍼링하기 *전에* 적용하기
        # 위해 스트리밍 API를 우선 쓴다. e2b SDK의 ``format="bytes"``는
        # ``bytearray(r.content)``로 구현되어 있어 파일 전체를 메모리에 올린 뒤
        # 반환한다. 수 GB짜리 artifact 하나로 호스팅된 공용 gateway가 OOM에 빠질 수 있다.
        # ``format="stream"``은 자체 HTTP 응답을 소유하고 소진/close/에러 시 pool 연결을
        # 반납하는 ``FileStreamReader``(``Iterator[bytes]``)를 반환한다.
        with self._lock:
            client = self._client
            if client is None:
                raise RuntimeError("sandbox client has been closed")
            try:
                data = client.files.read(resolved, format="stream")
            except TypeError:
                try:
                    data = client.files.read(resolved, format="bytes")
                except Exception as e:
                    logger.error("Failed to download file %s from e2b sandbox: %s", resolved, e)
                    raise OSError(f"Failed to download file '{path}' from sandbox: {e}") from e
            except Exception as e:
                logger.error("Failed to download file %s from e2b sandbox: %s", resolved, e)
                raise OSError(f"Failed to download file '{path}' from sandbox: {e}") from e

        if data is None:
            return b""

        # 버퍼링 fallback(bytes/bytearray/str): 이 경로에서도 초과 payload를 거부하도록
        # 상한을 먼저 적용한다.
        if isinstance(data, (bytes, bytearray)):
            if len(data) > _MAX_DOWNLOAD_SIZE:
                raise OSError(
                    errno.EFBIG,
                    f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes",
                    path,
                )
            return bytes(data)
        if isinstance(data, str):
            encoded = data.encode("utf-8")
            if len(encoded) > _MAX_DOWNLOAD_SIZE:
                raise OSError(
                    errno.EFBIG,
                    f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes",
                    path,
                )
            return encoded

        chunks: list[bytes] = []
        total = 0
        close = getattr(data, "close", None)
        try:
            try:
                for chunk in data:
                    if not chunk:
                        continue
                    chunk_bytes = chunk if isinstance(chunk, bytes) else bytes(chunk)
                    total += len(chunk_bytes)
                    if total > _MAX_DOWNLOAD_SIZE:
                        raise OSError(
                            errno.EFBIG,
                            f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes",
                            path,
                        )
                    chunks.append(chunk_bytes)
            except OSError:
                raise
            except Exception as e:
                logger.error("Failed to stream file %s from e2b sandbox: %s", resolved, e)
                raise OSError(f"Failed to download file '{path}' from sandbox: {e}") from e
        finally:
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return b"".join(chunks)

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        resolved = self._resolve_path(path)
        with self._lock:
            client = self._client
            if client is None:
                return []
            try:
                result = client.commands.run(f"find {shlex.quote(resolved)} -maxdepth {int(max_depth)} \\( -type f -o -type d \\) 2>/dev/null | head -500")
                output = getattr(result, "stdout", "") or ""
                return [line.strip() for line in output.splitlines() if line.strip()]
            except Exception as e:
                logger.error("Failed to list_dir %s in e2b sandbox: %s", resolved, e)
                return []

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        resolved = self._resolve_path(path)
        with self._lock:
            client = self._client
            if client is None:
                raise RuntimeError("sandbox client has been closed")
            try:
                if append:
                    existing = ""
                    try:
                        existing = client.files.read(resolved) or ""
                        if isinstance(existing, bytes):
                            existing = existing.decode("utf-8", errors="replace")
                    except Exception:
                        existing = ""
                    content = (existing or "") + content
                client.files.write(resolved, content)
            except Exception as e:
                logger.error("Failed to write file %s in e2b sandbox: %s", resolved, e)
                raise

    def update_file(self, path: str, content: bytes) -> None:
        resolved = self._resolve_path(path)
        with self._lock:
            client = self._client
            if client is None:
                raise RuntimeError("sandbox client has been closed")
            try:
                # e2b의 ``files.write``는 ``str``과 ``bytes``를 모두 받는다.
                # bytes로 넘겨야 바이너리 내용이 손실 없이 보존된다.
                client.files.write(resolved, content)
            except Exception as e:
                logger.error("Failed to update file %s in e2b sandbox: %s", resolved, e)
                raise

    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        resolved = self._resolve_path(path)
        types = "f,d" if include_dirs else "f"
        with self._lock:
            client = self._client
            if client is None:
                return [], False
            try:
                hard_limit = max(max_results * 4, max_results + 50)
                cmd = f"find {shlex.quote(resolved)} \\( " + " -o ".join(f"-type {t}" for t in types.split(",")) + f" \\) -print 2>/dev/null | head -{hard_limit}"
                result = client.commands.run(cmd)
                output = getattr(result, "stdout", "") or ""
            except Exception as e:
                logger.error("Failed to glob in e2b sandbox: %s", e)
                return [], False

        matches: list[str] = []
        root = resolved.rstrip("/") or "/"
        root_prefix = root if root == "/" else f"{root}/"
        for entry in output.splitlines():
            entry = entry.strip()
            if not entry:
                continue
            if entry != root and not entry.startswith(root_prefix):
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
        regex_source = re.escape(pattern) if literal else pattern
        re.compile(regex_source, 0 if case_sensitive else re.IGNORECASE)

        resolved = self._resolve_path(path)
        # 이식성 있는 ``grep`` 호출을 구성한다.
        # -r 재귀, -n 줄 번호, -H 파일명 항상 출력, -I 바이너리 파일 건너뛰기,
        # -E 확장 정규식(literal이면 -F로 고정 문자열).
        flags = ["-r", "-n", "-H", "-I"]
        if not case_sensitive:
            flags.append("-i")
        if literal:
            flags.append("-F")
        else:
            flags.append("-E")
        if glob is not None:
            # ``grep --include``는 깊이에 상관없이 basename으로만 매칭하므로
            # ``src/*.js``의 ``src/`` 같은 디렉터리 범위 prefix를 표현하지 못한다.
            # basename 부분만 거친 pre-filter로 넘기고(참 매칭 집합의 superset이다.
            # ``path_matches``가 받아들이는 파일은 모두 이 basename 패턴도 만족한다)
            # 실제 디렉터리 범위는 아래에서 ``glob()``과 같은 헬퍼인
            # ``path_matches``로 적용한다.
            include_pattern = glob.split("/")[-1] or glob
            flags.append(f"--include={include_pattern}")

        per_file_cap = max(max_results, 50)
        total_cap = max(max_results * 4, max_results + 50)
        flags.append(f"-m{per_file_cap}")

        cmd = "grep " + " ".join(flags) + f" -- {shlex.quote(regex_source)} {shlex.quote(resolved)} 2>/dev/null" + f" | head -{total_cap}"

        with self._lock:
            client = self._client
            if client is None:
                return [], False
            try:
                result = client.commands.run(cmd)
                output = getattr(result, "stdout", "") or ""
            except Exception as e:
                logger.error("Failed to grep in e2b sandbox: %s", e)
                return [], False

        root = resolved.rstrip("/") or "/"
        root_prefix = root if root == "/" else f"{root}/"

        matches: list[GrepMatch] = []
        truncated = False
        for raw in output.splitlines():
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
                # 위의 ``--include`` 플래그는 basename으로만 걸렀으므로,
                # 여기서 호출자가 요청한 실제 디렉터리 범위로 좁힌다.
                if file_path != root and not file_path.startswith(root_prefix):
                    continue
                rel_path = file_path.rsplit("/", 1)[-1] if file_path == root else file_path[len(root) :].lstrip("/")
                if not path_matches(glob, rel_path):
                    continue
            matches.append(
                GrepMatch(
                    path=file_path,
                    line_number=line_number,
                    line=truncate_line(line_text),
                )
            )
            if len(matches) >= max_results:
                truncated = True
                break
        return matches, truncated
