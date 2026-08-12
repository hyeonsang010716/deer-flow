import base64
import errno
import logging
import shlex
import threading
import uuid

import httpx
from agent_sandbox import Sandbox as AioSandboxClient
from agent_sandbox.core.api_error import ApiError

from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.search import GrepMatch, path_matches, should_ignore_path, truncate_line

from .backend import sandbox_http_trust_env

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

_ERROR_OBSERVATION_SIGNATURE = "'ErrorObservation' object has no attribute 'exit_code'"

# env를 실어 보내는 명령은 bash.exec API(POST /v1/bash/exec)가 필요한데,
# all-in-one-sandbox 이미지는 1.9.x부터만 이를 제공한다. 더 오래된 이미지(1.0.0.x 계열에
# 고정된 ``latest`` 태그 포함)는 /v1/bash/* 네임스페이스 전체에 404를 준다.
# 날것의 404는 모델에게 쓸모없고 재시도만 유발하므로, sandbox는 대신 이 operator용 메시지로
# 즉시 실패한다(#3921).
_BASH_EXEC_UNSUPPORTED_ERROR = (
    "Error: this sandbox image does not support per-command environment injection "
    "(POST /v1/bash/exec returned 404), which is required to run skills that declare "
    "required-secrets. This is a deployment issue that retrying cannot fix: upgrade the "
    "sandbox image to all-in-one-sandbox >= 1.9.3 (set `sandbox.image` in config.yaml, "
    "e.g. pin the tag `1.11.0`) and recreate the sandbox container, then try again."
)


class AioSandbox(Sandbox):
    """agent-infra/sandbox Docker 컨테이너를 사용하는 Sandbox 구현.

    실행 중인 AIO sandbox 컨테이너에 HTTP API로 연결한다.
    threading lock으로 shell 명령을 직렬화해서, 동시 요청이 컨테이너의 단일 persistent session을
    망가뜨리지 않게 한다(#1433 참고).
    """

    def __init__(self, id: str, base_url: str, home_dir: str | None = None):
        """AIO sandbox를 초기화한다.

        Args:
            id: 이 sandbox 인스턴스의 고유 식별자.
            base_url: sandbox API의 URL(예: http://localhost:8080).
            home_dir: sandbox 내부의 홈 디렉터리. None이면 sandbox에서 가져온다.
        """
        super().__init__(id)
        self._base_url = base_url
        if sandbox_http_trust_env(base_url):
            self._client = AioSandboxClient(base_url=base_url, timeout=600)
        else:
            direct_client = httpx.Client(timeout=600, follow_redirects=True, trust_env=False)
            self._client = AioSandboxClient(
                base_url=base_url,
                timeout=600,
                httpx_client=direct_client,
            )
        self._home_dir = home_dir
        self._lock = threading.Lock()
        self._closed = False
        # bash.exec가 404를 준 뒤 True가 된다(이미지가 /v1/bash/* 이전 버전).
        # 이후 env를 실은 호출이 HTTP를 다시 때리지 않고 즉시 실패한다(#3921).
        self._bash_exec_unsupported = False

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        """이 sandbox가 소유한 host 측 HTTP client를 best-effort로 닫는다.

        agent_sandbox SDK는 Fern으로 생성되어 ``close()`` / ``__exit__``를 노출하지 않는다.
        그래서 socket을 실제로 소유한 ``httpx.Client``에 속성 체인을 따라 직접 접근한다::

            Sandbox._client_wrapper        -> SyncClientWrapper
                .httpx_client              -> Fern HttpClient (a wrapper, NOT httpx.Client)
                    .httpx_client          -> httpx.Client     <- the real socket owner

        이를 닫으면 pool에 있던 socket이 반환되므로, 오래 사는 provider lifecycle이
        회수되지 않은 host 측 자원을 쌓지 않는다(#2872).

        해석 순서는 가장 구체적인 것부터이며 단계적으로 물러난다. 미래의 SDK가 최상위
        ``Sandbox.close()``를 추가하면 이 코드를 고치지 않아도 자동으로 그것을 쓴다.
        멱등하고 thread-safe하며 치명적이지 않다. teardown 중 실패는 로그만 남기고 삼켜서
        provider/backend 정리를 막지 않는다.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            # use-after-close 안전을 위해 lock 안에서 참조를 버린다. 이후 이 인스턴스에
            # 들어오는 명령은 반쯤 닫힌 client를 재사용하지 않고 크게 실패한다.
            self._client = None

        if client is None:
            return

        # 실제 httpx.Client에서 최상위 client까지 거슬러 올라가며 close()를 실제로
        # 노출하는 첫 객체를 고른다.
        wrapper = getattr(client, "_client_wrapper", None)
        fern_http = getattr(wrapper, "httpx_client", None)
        real_httpx = getattr(fern_http, "httpx_client", None)
        target = next(
            (c for c in (real_httpx, fern_http, client) if c is not None and hasattr(c, "close")),
            None,
        )
        if target is None:
            logger.debug("AioSandbox %s: no closable client found, nothing to release", self.id)
            return

        try:
            target.close()
        except Exception as e:
            logger.warning(f"Error closing AioSandbox client for {self.id}: {e}")

    @property
    def home_dir(self) -> str:
        """sandbox 내부의 홈 디렉터리를 반환한다."""
        if self._home_dir is None:
            context = self._client.sandbox.get_context()
            self._home_dir = context.home_dir
        return self._home_dir

    # exec_command의 기본 no_change_timeout(초). client 수준 timeout과 맞춰서,
    # 출력이 없는 장시간 명령이 sandbox 내장 기본값 120초에 조기 종료되지 않게 한다.
    _DEFAULT_NO_CHANGE_TIMEOUT = 600

    # bash.exec로 보내는 env 포함 명령의 wall-clock hard timeout.
    # bash.exec API는 idle/no-change timeout을 노출하지 않는다(legacy 경로의
    # shell.exec_command ``no_change_timeout``과 다르다). 따라서 env 포함 명령은
    # 마지막 출력 이후 시간이 아니라 총 경과 wall-clock 시간으로 제한된다.
    # 두 경로가 명령 하나의 허용 실행 시간에 대해 대체로 일치하도록 legacy idle 예산과
    # 같은 숫자로 유지한다. 미래 SDK가 bash.exec에 idle timeout을 노출하면 이 호출부를
    # 그쪽으로 바꿔야 한다.
    _DEFAULT_HARD_TIMEOUT = 600.0

    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """sandbox에서 shell 명령을 실행한다.

        lock으로 동시 요청을 직렬화한다. AIO sandbox 컨테이너는 단일 persistent shell session을
        유지하는데, 동시 exec_command 호출을 받으면 망가져서 실제 출력 대신
        ``ErrorObservation``을 반환한다. lock에도 불구하고 손상이 감지되면
        (예: 여러 프로세스가 sandbox를 공유) 새 session에서 명령을 재시도한다.

        Args:
            command: 실행할 명령.
            env: 호출 단위 환경 변수(request-scoped secrets, issue #3861). 주어지면 명령은
                per-command env를 지원하는 ``bash.exec`` API를 통해 자동 생성된 새 session에서
                실행된다. 덕분에 secret은 이 명령 하나로만 범위가 한정되고 남지 않는다.
                secret 값은 명령 문자열이 아니라 구조화된 ``env`` 필드로 전달된다.
                ``None``이면 기존 persistent-shell 경로가 그대로 동작한다.
            timeout: 호출 단위 timeout. 현재 sandbox SDK는 client/request timeout과 구분되는
                명령 수준 timeout을 노출하지 않으므로 DeerFlow는 여기서 backend 기본값을 쓴다.

        Returns:
            명령의 출력.
        """
        del timeout
        # ``env`` 키를 ``bash.exec`` API로 넘기기 전에 검증한다. 공개 ``Sandbox.execute_command``
        # 계약은 임의의 dict 키를 받지만, POSIX 환경 변수 이름 규칙을 강제하면 local·e2b sandbox와
        # 계약이 일관되게 유지되고 위험한 키를 일찍 잡아낸다.
        # ``env``가 None이거나 비어 있으면 ``_validate_extra_env``는 아무것도 하지 않는다.
        _validate_extra_env(env)
        if env:
            return self._execute_with_env(command, env)
        with self._lock:
            try:
                result = self._client.shell.exec_command(command=command, no_change_timeout=self._DEFAULT_NO_CHANGE_TIMEOUT)
                output = result.data.output if result.data else ""

                if output and _ERROR_OBSERVATION_SIGNATURE in output:
                    logger.warning("ErrorObservation detected in sandbox output, retrying on a fresh session")
                    # exec_command는 id 없이 호출될 때만 session을 자동 생성하므로,
                    # 재시도에서 지정할 recovery session은 미리 명시적으로 만들어야 한다.
                    fresh_id = str(uuid.uuid4())
                    self._client.shell.create_session(id=fresh_id)
                    try:
                        result = self._client.shell.exec_command(command=command, id=fresh_id, no_change_timeout=self._DEFAULT_NO_CHANGE_TIMEOUT)
                        output = result.data.output if result.data else ""
                    finally:
                        # 일회용 recovery session을 best-effort로 반환해서,
                        # 손상이 반복돼도 session이 쌓이지 않게 한다.
                        try:
                            self._client.shell.cleanup_session(fresh_id)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to release recovery session {fresh_id}: {cleanup_error}")

                return output if output else "(no output)"
            except Exception as e:
                logger.error(f"Failed to execute command in sandbox: {e}")
                return f"Error: {e}"

    def _execute_with_env(self, command: str, env: dict[str, str]) -> str:
        """호출 단위 환경 변수를 주입해 명령을 실행한다.

        persistent-shell ``shell.exec_command`` API에는 env 파라미터가 없으므로, 주입이 필요한
        명령은 per-command env를 받는 ``bash.exec`` API를 쓴다. 호출마다 ``session_id`` 없이
        sandbox가 새 session을 자동 생성하게 해서, 주입된 request-scoped secret이 이 명령으로만
        범위가 한정되고 호출 간에 남지 않게 한다. secret 값은 명령 문자열이 아니라 구조화된
        ``env`` 필드로 전달된다.

        새 session을 쓰는 선택의 트레이드오프: 같은 skill 안에서 연속으로 실행되는 env 포함 bash
        호출은 session 상태(cwd, source한 venv, export한 변수)를 공유하지 않는다. 이는 LocalSandbox
        모델(호출마다 새 subprocess)과 같으며 의도된 동작이다. session_id를 공유하면 request-scoped
        secret이 session env를 타고 이후 명령까지 흘러갈 수 있는데, SDK가 이를 계약으로 금지하지
        않는다. 사전 준비가 필요한 skill은 하나의 명령으로 합쳐야 한다
        (예: ``cd /mnt/user-data/workspace && source .venv/bin/activate && python run.py``).

        ``_ERROR_OBSERVATION_SIGNATURE`` 복구 계약은 legacy persistent-shell 경로와 공유한다.
        (호출마다 새 session이라 가능성은 낮지만) 손상 표식이 나타나면 그대로 반환하지 않고
        또 다른 새 session에서 재시도한다.

        all-in-one-sandbox 1.9.x 이전 이미지에는 ``/v1/bash/*`` 라우트가 없다. secret 값을 명령
        문자열 밖에 유지해 주는 legacy shell 경로의 fallback은 존재하지 않으므로, 안전한 동작은
        실행 가능한 안내를 담은 오류로 즉시 실패하는 것뿐이다(#3921).
        """
        if self._bash_exec_unsupported:
            return _BASH_EXEC_UNSUPPORTED_ERROR
        output = self._run_bash_exec(command, env)
        if output and _ERROR_OBSERVATION_SIGNATURE in output:
            logger.warning("ErrorObservation detected in bash.exec output, retrying on a fresh session")
            retried = self._run_bash_exec(command, env)
            if retried and _ERROR_OBSERVATION_SIGNATURE not in retried:
                return retried
        return output

    def _run_bash_exec(self, command: str, env: dict[str, str]) -> str:
        """env를 주입한 bash.exec 단일 호출(새 session 하나)."""
        with self._lock:
            try:
                result = self._client.bash.exec(
                    command=command,
                    env=env,
                    hard_timeout=self._DEFAULT_HARD_TIMEOUT,
                )
                data = result.data if result else None
                stdout = (data.stdout or "") if data else ""
                stderr = (data.stderr or "") if data else ""
                output = stdout
                if stderr:
                    output += f"\nStd Error:\n{stderr}" if output else stderr
                return output if output else "(no output)"
            except ApiError as e:
                if e.status_code == 404:
                    self._bash_exec_unsupported = True
                    logger.error("Sandbox %s does not support bash.exec (/v1/bash/exec returned 404); env-bearing commands are unavailable until the sandbox image is upgraded to all-in-one-sandbox >= 1.9.3", self.id)
                    return _BASH_EXEC_UNSUPPORTED_ERROR
                logger.error(f"Failed to execute command with injected env in sandbox: {e}")
                return f"Error: {e}"
            except Exception as e:
                logger.error(f"Failed to execute command with injected env in sandbox: {e}")
                return f"Error: {e}"

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """sandbox에 있는 파일의 내용을 읽는다.

        Args:
            path: 읽을 파일의 절대 경로.

        Returns:
            파일의 내용.
        """
        try:
            kwargs = {}
            if start_line is not None:
                kwargs["start_line"] = max(start_line - 1, 0)
            if end_line is not None:
                kwargs["end_line"] = max(end_line, 0)
            result = self._client.file.read_file(file=path, **kwargs)
            return result.data.content if result.data else ""
        except Exception as e:
            logger.error(f"Failed to read file in sandbox: {e}")
            return f"Error: {e}"

    def download_file(self, path: str) -> bytes:
        """sandbox에서 파일 bytes를 내려받는다.

        Raises:
            PermissionError: 경로에 '..' traversal 구간이 있거나
                ``VIRTUAL_PATH_PREFIX`` 밖일 때.
            OSError: sandbox에서 파일을 가져오지 못했을 때.
        """
        # 컨테이너 API로 보내기 전에 path traversal을 거부한다.
        # LocalSandbox는 _resolve_path를 통해 이를 암묵적으로 처리하지만,
        # 여기서는 경로를 그대로 전달하므로 명시적으로 검사해야 한다.
        normalised = path.replace("\\", "/")
        for segment in normalised.split("/"):
            if segment == "..":
                logger.error(f"Refused download due to path traversal: {path}")
                raise PermissionError(f"Access denied: path traversal detected in '{path}'")

        stripped_path = normalised.lstrip("/")
        allowed_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped_path != allowed_prefix and not stripped_path.startswith(f"{allowed_prefix}/"):
            logger.error("Refused download outside allowed directory: path=%s, allowed_prefix=%s", path, VIRTUAL_PATH_PREFIX)
            raise PermissionError(f"Access denied: path must be under '{VIRTUAL_PATH_PREFIX}': '{path}'")

        with self._lock:
            try:
                chunks: list[bytes] = []
                total = 0
                for chunk in self._client.file.download_file(path=path):
                    total += len(chunk)
                    if total > _MAX_DOWNLOAD_SIZE:
                        raise OSError(
                            errno.EFBIG,
                            f"File exceeds maximum download size of {_MAX_DOWNLOAD_SIZE} bytes",
                            path,
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
            except OSError:
                raise
            except Exception as e:
                logger.error(f"Failed to download file in sandbox: {e}")
                raise OSError(f"Failed to download file '{path}' from sandbox: {e}") from e

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        """sandbox에 있는 디렉터리의 내용을 나열한다.

        Args:
            path: 나열할 디렉터리의 절대 경로.
            max_depth: 순회할 최대 깊이. 기본값은 2.

        Returns:
            디렉터리의 내용.
        """
        with self._lock:
            try:
                result = self._client.shell.exec_command(command=f"find {shlex.quote(path)} -maxdepth {max_depth} -type f -o -type d 2>/dev/null | head -500", no_change_timeout=self._DEFAULT_NO_CHANGE_TIMEOUT)
                output = result.data.output if result.data else ""
                if output:
                    return [line.strip() for line in output.strip().split("\n") if line.strip()]
                return []
            except Exception as e:
                logger.error(f"Failed to list directory in sandbox: {e}")
                return []

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """sandbox에 있는 파일에 내용을 쓴다.

        Args:
            path: 쓸 파일의 절대 경로.
            content: 파일에 쓸 텍스트 내용.
            append: 내용을 파일 끝에 덧붙일지 여부.
        """
        with self._lock:
            try:
                if append:
                    existing = self.read_file(path)
                    if not existing.startswith("Error:"):
                        content = existing + content
                self._client.file.write_file(file=path, content=content)
            except Exception as e:
                logger.error(f"Failed to write file in sandbox: {e}")
                raise

    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        if not include_dirs:
            result = self._client.file.find_files(path=path, glob=pattern)
            files = result.data.files if result.data and result.data.files else []
            filtered = [file_path for file_path in files if not should_ignore_path(file_path)]
            truncated = len(filtered) > max_results
            return filtered[:max_results], truncated

        result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
        entries = result.data.files if result.data and result.data.files else []
        matches: list[str] = []
        root_path = path.rstrip("/") or "/"
        root_prefix = root_path if root_path == "/" else f"{root_path}/"
        for entry in entries:
            if entry.path != root_path and not entry.path.startswith(root_prefix):
                continue
            if should_ignore_path(entry.path):
                continue
            rel_path = entry.path[len(root_path) :].lstrip("/")
            if path_matches(pattern, rel_path):
                matches.append(entry.path)
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
        import re as _re

        regex_source = _re.escape(pattern) if literal else pattern
        # 패턴을 로컬에서 검증해서, 잘못된 regex가 범용 원격 API 오류가 아니라
        # re.error를 던지게 한다(grep_tool의 except re.error 핸들러가 잡는다).
        _re.compile(regex_source, 0 if case_sensitive else _re.IGNORECASE)
        total_cap = max(max_results * 4, max_results + 50)
        result = self._client.file.grep_files(
            path=path,
            pattern=pattern,
            case_insensitive=not case_sensitive,
            fixed_strings=literal,
            max_results=total_cap,
            max_file_size="1M",
            recursive=True,
        )
        data = result.data
        provider_matches = data.matches if data and data.matches else []
        root = path.rstrip("/") or "/"
        root_prefix = root if root == "/" else f"{root}/"

        matches: list[GrepMatch] = []
        truncated = bool(data and data.truncated)
        for match in provider_matches:
            file_path = match.file
            if should_ignore_path(file_path):
                continue
            if file_path == root:
                rel_path = file_path.rsplit("/", 1)[-1]
            elif file_path.startswith(root_prefix):
                rel_path = file_path[len(root_prefix) :]
            else:
                continue
            if glob is not None and not path_matches(glob, rel_path):
                continue
            matches.append(
                GrepMatch(
                    path=file_path,
                    line_number=match.line_number,
                    line=truncate_line(match.line_content),
                )
            )
            if len(matches) >= max_results:
                truncated = True
                break

        return matches, truncated

    def update_file(self, path: str, content: bytes) -> None:
        """sandbox에 있는 파일을 바이너리 내용으로 갱신한다.

        Args:
            path: 갱신할 파일의 절대 경로.
            content: 파일에 쓸 바이너리 내용.
        """
        with self._lock:
            try:
                base64_content = base64.b64encode(content).decode("utf-8")
                self._client.file.write_file(file=path, content=base64_content, encoding="base64")
            except Exception as e:
                logger.error(f"Failed to update file in sandbox: {e}")
                raise
