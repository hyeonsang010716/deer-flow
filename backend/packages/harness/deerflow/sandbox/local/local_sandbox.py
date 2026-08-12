import errno
import logging
import ntpath
import os
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.env_policy import build_sandbox_env
from deerflow.sandbox.local.list_dir import list_dir
from deerflow.sandbox.path_patterns import build_output_mask_pattern
from deerflow.sandbox.sandbox import Sandbox, _validate_extra_env
from deerflow.sandbox.search import GrepMatch, find_glob_matches, find_grep_matches

logger = logging.getLogger(__name__)

# 단일 host bash 명령의 기본 wall-clock timeout(초). 블로킹되는 foreground 명령(예: 백그라운드로
# 돌리지 않고 시작한 서버)은 이 시간이 지나면 종료되므로 agent의 turn이 무한정 멈추지 않는다.
# 호출별로는 ``execute_command(timeout=...)``로, bash tool에서는 config.yaml의
# ``sandbox.bash_command_timeout``으로 재정의할 수 있다.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 600
_COMMAND_CAPTURE_LIMIT_BYTES = 10 * 1024 * 1024
_PIPE_DRAIN_JOIN_TIMEOUT_SECONDS = 0.2


class _BoundedPipeCapture:
    """subprocess pipe를 계속 비워 내되, 메모리에는 제한된 크기의 출력만 유지한다."""

    def __init__(self, *, limit_bytes: int = _COMMAND_CAPTURE_LIMIT_BYTES) -> None:
        self._limit_bytes = limit_bytes
        self._chunks: list[bytes] = []
        self._kept_bytes = 0
        self._total_bytes = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._total_bytes += len(chunk)
            if self._kept_bytes >= self._limit_bytes:
                return
            remaining = self._limit_bytes - self._kept_bytes
            kept = chunk[:remaining]
            self._chunks.append(kept)
            self._kept_bytes += len(kept)

    def read(self) -> str:
        with self._lock:
            data = b"".join(self._chunks)
            truncated = self._total_bytes > self._kept_bytes
            total_bytes = self._total_bytes
            kept_bytes = self._kept_bytes

        output = data.decode("utf-8", errors="replace")
        if truncated:
            notice = f"\n... [output truncated after {kept_bytes} of {total_bytes} bytes; remaining output discarded] ..."
            output += notice
        return output


@dataclass(frozen=True)
class PathMapping:
    """container 경로에서 로컬 경로로의 매핑. read-only 플래그를 선택적으로 가진다."""

    container_path: str
    local_path: str
    read_only: bool = False


class ResolvedPath(NamedTuple):
    path: str
    mapping: PathMapping | None


class LocalSandbox(Sandbox):
    @staticmethod
    def _shell_name(shell: str) -> str:
        """shell 경로나 명령에서 실행 파일 이름을 반환한다."""
        return shell.replace("\\", "/").rsplit("/", 1)[-1].lower()

    @staticmethod
    def _is_powershell(shell: str) -> bool:
        """선택된 shell이 PowerShell 실행 파일인지 반환한다."""
        return LocalSandbox._shell_name(shell) in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}

    @staticmethod
    def _is_cmd_shell(shell: str) -> bool:
        """선택된 shell이 cmd.exe인지 반환한다."""
        return LocalSandbox._shell_name(shell) in {"cmd", "cmd.exe"}

    @staticmethod
    def _is_msys_shell(shell: str) -> bool:
        """선택된 shell이 Git Bash/MSYS shell인지 반환한다."""
        normalized = shell.replace("\\", "/").lower()
        shell_name = LocalSandbox._shell_name(shell)
        return shell_name in {"sh.exe", "bash.exe"} and any(part in normalized for part in ("/git/", "/mingw", "/msys"))

    @staticmethod
    def _find_first_available_shell(candidates: tuple[str, ...]) -> str | None:
        """후보 중에서 처음으로 발견된 실행 가능한 shell 경로나 명령을 반환한다."""
        for shell in candidates:
            if os.path.isabs(shell):
                if os.path.isfile(shell) and os.access(shell, os.X_OK):
                    return shell
                continue

            shell_from_path = shutil.which(shell)
            if shell_from_path is not None:
                return shell_from_path

        return None

    @staticmethod
    def _format_timeout_duration(timeout: float) -> str:
        seconds = float(timeout)
        if seconds.is_integer():
            amount = str(int(seconds))
        else:
            amount = f"{seconds:g}"
        unit = "second" if seconds == 1 else "seconds"
        return f"{amount} {unit}"

    @staticmethod
    def _format_timeout_notice(timeout: float) -> str:
        return (
            f"Command timed out after {LocalSandbox._format_timeout_duration(timeout)} and was terminated. "
            "To run a long-lived process such as a web server, start it in the background "
            "and redirect its output, e.g. `your-command > /mnt/user-data/workspace/server.log 2>&1 &`."
        )

    @staticmethod
    def _coerce_process_output(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _drain_pipe(fd: int, capture: _BoundedPipeCapture) -> None:
        try:
            while chunk := os.read(fd, 8192):
                capture.append(chunk)
        except OSError:
            logger.debug("Subprocess output pipe closed while draining", exc_info=True)
        finally:
            try:
                os.close(fd)
            except OSError:
                # pipe teardown 중에 fd가 이미 닫혔을 수 있다. 정리는 best-effort다.
                pass

    @staticmethod
    def _start_pipe_drain(fd: int, name: str) -> tuple[_BoundedPipeCapture, threading.Thread]:
        capture = _BoundedPipeCapture()
        thread = threading.Thread(target=LocalSandbox._drain_pipe, args=(fd, capture), name=name, daemon=True)
        thread.start()
        return capture, thread

    @staticmethod
    def _process_group_exists(pgid: int | None) -> bool:
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def __init__(self, id: str, path_mappings: list[PathMapping] | None = None):
        """
        선택적인 path mapping과 함께 local sandbox를 초기화한다.

        Args:
            id: sandbox 식별자
            path_mappings: read-only 플래그를 선택적으로 가지는 path mapping 목록.
                          skills 디렉터리는 기본적으로 read-only다.
        """
        super().__init__(id)
        self.path_mappings = path_mappings or []
        # write_file로 쓴 파일을 추적해서, read_file이 agent가 작성한 content에서만 경로를
        # 역해석하도록 한다.
        self._agent_written_paths: set[str] = set()

    # ``path_mappings``는 ``__init__``에서 한 번 설정되고 이후 변경되지 않으므로, 아래의 정렬된
    # 뷰와 컴파일된 경로 재작성 pattern은 sandbox 수명 동안 안정적이다. 이를 캐싱하면 매
    # bash/read_file/write_file 호출(agent의 hot path)마다 다시 정렬하고 regex를 다시 컴파일하는
    # 비용을 없앨 수 있다.

    @cached_property
    def _command_pattern(self) -> re.Pattern[str] | None:
        """shell 명령 안의 container 경로를 매칭하는 컴파일된 pattern(shell 경계 인식)."""
        mappings = sorted(self.path_mappings, key=lambda m: len(m.container_path), reverse=True)
        if not mappings:
            return None
        # lookahead (?=/|$|...)는 경로 세그먼트 경계에서만 매칭되도록 보장해, /mnt/skills가
        # /mnt/skills-extra 안에서 매칭되는 것을 막는다.
        patterns = [re.escape(m.container_path) + r"(?=/|$|[\s\"';&|<>()])(?:/[^\s\"';&|<>()]*)?" for m in mappings]
        return re.compile("|".join(f"({p})" for p in patterns))

    @cached_property
    def _content_pattern(self) -> re.Pattern[str] | None:
        """평문 파일 content 안의 container 경로를 매칭하는 컴파일된 pattern(텍스트 경계)."""
        mappings = sorted(self.path_mappings, key=lambda m: len(m.container_path), reverse=True)
        if not mappings:
            return None
        patterns = [re.escape(m.container_path) + r"(?=/|$|[^\w./-])(?:/[^\s\"';&|<>()]*)?" for m in mappings]
        return re.compile("|".join(f"({p})" for p in patterns))

    @cached_property
    def _reverse_output_patterns(self) -> list[re.Pattern[str]]:
        """명령 출력 안의 로컬 경로를 매칭하는 컴파일된 pattern들(가장 긴 로컬 경로 우선)."""
        # 이 규칙 — 세그먼트 경계 + 경로 꼬리 — 은 ``deerflow.sandbox.path_patterns``가 소유하며,
        # host 경로를 virtual 경로로 되돌리는 다른 지점인 ``sandbox.tools._compiled_mask_patterns``와
        # 공유한다. 그 근거(왜 경계 문자 집합이 ``_command_pattern``처럼 shell 지향이 아니라 텍스트
        # 지향인지, 왜 ``$``가 필수인지)는 여기에 두 번째 사본을 두지 않고 소유자 쪽에 있다. 사본을
        # 두었던 것이 예전에 둘을 어긋나게 한 원인이다(#4035는 여기에 경계를 추가하면서 그 지점을
        # 놓쳤고, #4053이 그쪽에 추가했다).
        #
        # 이 지점에만 해당하는 사항: 경계가 없으면 regex가 맨 root를 내놓고, 그 값은 mount root와
        # *같아지므로* ``_reverse_resolve_path``의 ``+ "/"`` 가드를 통과한다 — 그러면 형제 경로가
        # forward resolution이 다시 매핑하기를 거부하는 container 경로로 재작성된다. 그리고 base는
        # 구분자에 *민감한* 상태로 둔다. ``Path.resolve()``에서 오기 때문에 이미 플랫폼 구분자를
        # 담고 있고, 이를 느슨하게 하면 마스킹 범위가 넓어진다.
        return [build_output_mask_pattern(self._resolved_local_paths[m]) for m in self._mappings_by_local_specificity]

    @cached_property
    def _resolved_local_paths(self) -> dict[PathMapping, str]:
        """mapping별로 파일시스템에서 해석한 로컬 root. ``Path.resolve()``는 디스크를 건드리고
        mount된 디렉터리는 움직이지 않으므로, 한 번만 해석해 재사용한다."""
        return {m: str(Path(m.local_path).resolve()) for m in self.path_mappings}

    @cached_property
    def _mappings_by_container_specificity(self) -> list[PathMapping]:
        """container 경로가 가장 구체적인 것부터 정렬한 mapping 목록(forward resolution용)."""
        return sorted(self.path_mappings, key=lambda m: len(m.container_path.rstrip("/") or "/"), reverse=True)

    @cached_property
    def _mappings_by_local_specificity(self) -> list[PathMapping]:
        """로컬 경로가 긴 것부터 정렬한 mapping 목록(reverse resolution용)."""
        return sorted(self.path_mappings, key=lambda m: len(m.local_path), reverse=True)

    def _is_read_only_path(self, resolved_path: str) -> bool:
        """해석된 경로가 read-only mount 아래에 있는지 확인한다.

        여러 mapping이 매칭되면(중첩 mount) 가장 구체적인 mapping, 즉 local_path가 해석된 경로의
        가장 긴 prefix인 것을 택한다. ``_resolve_path``가 container 경로를 다루는 방식과 같다.
        """
        resolved = str(Path(resolved_path).resolve())

        best_mapping: PathMapping | None = None
        best_prefix_len = -1

        for mapping in self.path_mappings:
            local_resolved = self._resolved_local_paths[mapping]
            if resolved == local_resolved or resolved.startswith(local_resolved + os.sep):
                prefix_len = len(local_resolved)
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_mapping = mapping

        if best_mapping is None:
            return False

        return best_mapping.read_only

    def _find_path_mapping(self, path: str) -> tuple[PathMapping, str] | None:
        path_str = str(path)

        for mapping in self._mappings_by_container_specificity:
            container_path = mapping.container_path.rstrip("/") or "/"
            if container_path == "/":
                if path_str.startswith("/"):
                    return mapping, path_str.lstrip("/")
                continue

            if path_str == container_path or path_str.startswith(container_path + "/"):
                relative = path_str[len(container_path) :].lstrip("/")
                return mapping, relative

        return None

    def _resolve_path_with_mapping(self, path: str) -> ResolvedPath:
        """
        mapping을 사용해 container 경로를 실제 로컬 경로로 해석한다.

        Args:
            path: container 경로일 수 있는 경로

        Returns:
            해석된 로컬 경로와, 매칭된 mapping이 있으면 그 mapping
        """
        path_str = str(path)

        mapping_match = self._find_path_mapping(path_str)
        if mapping_match is None:
            return ResolvedPath(path_str, None)

        mapping, relative = mapping_match
        local_root = Path(self._resolved_local_paths[mapping])
        resolved_path = (local_root / relative).resolve() if relative else local_root

        try:
            resolved_path.relative_to(local_root)
        except ValueError as exc:
            raise PermissionError(errno.EACCES, "Access denied: path escapes mounted directory", path_str) from exc

        return ResolvedPath(str(resolved_path), mapping)

    def _resolve_path(self, path: str) -> str:
        return self._resolve_path_with_mapping(path).path

    def _is_resolved_path_read_only(self, resolved: ResolvedPath) -> bool:
        return bool(resolved.mapping and resolved.mapping.read_only) or self._is_read_only_path(resolved.path)

    def _reverse_resolve_path(self, path: str) -> str:
        """
        mapping을 사용해 로컬 경로를 container 경로로 역해석한다.

        Args:
            path: container 경로로 매핑해야 할 수 있는 로컬 경로

        Returns:
            mapping이 있으면 container 경로, 없으면 원래 경로
        """
        normalized_path = path.replace("\\", "/")
        path_str = str(Path(normalized_path).resolve())

        # 각 mapping을 시도한다(더 구체적인 매칭을 위해 로컬 경로가 긴 것부터)
        for mapping in self._mappings_by_local_specificity:
            local_path_resolved = self._resolved_local_paths[mapping]
            # ``Path.resolve()``는 위의 슬래시 정규화와 무관하게 항상 네이티브 구분자(Windows에서는
            # 백슬래시)로 결과를 만든다. 그래서 포함 검사도 하드코딩된 "/"가 아니라 ``os.sep``으로
            # 비교해야 한다 — ``_is_read_only_path``와 같은 방식이다. 하드코딩된 "/"는 Windows에서
            # 백슬래시로 이어진 중첩 경로와 절대 매칭되지 않으므로, 모든 중첩 경로가 조용히 아래
            # "mapping 없음" 분기로 빠져 raw host 경로(실제 사용자명, 전체 디렉터리 트리)를
            # 노출했다.
            if path_str == local_path_resolved or path_str.startswith(local_path_resolved + os.sep):
                # 로컬 경로 prefix를 container 경로로 치환한다. container 경로는 항상 POSIX
                # 스타일이므로, 추출한 상대 경로 부분(Windows에서는 네이티브 구분자)을 이어 붙이기
                # 전에 슬래시로 정규화한다.
                relative = path_str[len(local_path_resolved) :].lstrip(os.sep).replace(os.sep, "/")
                resolved = f"{mapping.container_path}/{relative}" if relative else mapping.container_path
                return resolved

        # mapping을 찾지 못했으므로 원래 경로를 반환한다
        return path_str

    def _reverse_resolve_paths_in_output(self, output: str) -> str:
        """
        출력 문자열 안의 로컬 경로를 container 경로로 역해석한다.

        Args:
            output: 로컬 경로를 담고 있을 수 있는 출력 문자열

        Returns:
            로컬 경로가 container 경로로 해석된 출력
        """
        # pattern은 sandbox마다 한 번만 컴파일되고(올바른 prefix 매칭을 위해 로컬 경로가 긴 것부터)
        # 호출 간에 재사용된다.
        result = output
        for pattern in self._reverse_output_patterns:

            def replace_match(match: re.Match) -> str:
                matched_path = match.group(0)
                return self._reverse_resolve_path(matched_path)

            result = pattern.sub(replace_match, result)

        return result

    def _resolve_paths_in_command(self, command: str) -> str:
        """
        명령 문자열 안의 container 경로를 로컬 경로로 해석한다.

        Args:
            command: container 경로를 담고 있을 수 있는 명령 문자열

        Returns:
            container 경로가 로컬 경로로 해석된 명령
        """
        pattern = self._command_pattern
        if pattern is None:
            return command

        def replace_match(match: re.Match) -> str:
            matched_path = match.group(0)
            # bash가 Windows 백슬래시 시퀀스(\\U, \\a, \\d, \\s, \\n, \\t)를 escape로 해석하지
            # 않도록 슬래시로 정규화한다.
            return self._resolve_path(matched_path).replace("\\", "/")

        return pattern.sub(replace_match, command)

    def _resolve_paths_in_content(self, content: str) -> str:
        """임의의 파일 content 안의 container 경로를 로컬 경로로 해석한다.

        shell 경계 문자를 쓰는 ``_resolve_paths_in_command``와 달리, 이 메서드는 content를 평문
        텍스트로 보고 container 경로 prefix가 나오는 모든 곳을 해석한다. 해석된 경로는 Windows
        host에서의 백슬래시 escape 문제(예: ``C:\\Users\\..``가 Python 문자열 리터럴을 깨뜨리는
        경우)를 피하려고 슬래시로 정규화한다.

        Args:
            content: container 경로를 담고 있을 수 있는 파일 content.

        Returns:
            container 경로가 로컬 경로(슬래시)로 해석된 content.
        """
        pattern = self._content_pattern
        if pattern is None:
            return content

        def replace_match(match: re.Match) -> str:
            matched_path = match.group(0)
            resolved = self._resolve_path(matched_path)
            # Windows 백슬래시 경로가 소스 파일에서 잘못된 escape 시퀀스를 만들지 않도록
            # 슬래시로 정규화한다.
            return resolved.replace("\\", "/")

        return pattern.sub(replace_match, content)

    @staticmethod
    def _get_shell() -> str:
        """사용 가능한 shell 실행 파일을 fallback과 함께 탐지한다."""
        shell = LocalSandbox._find_first_available_shell(("/bin/zsh", "/bin/bash", "/bin/sh", "sh"))
        if shell is not None:
            return shell

        if os.name == "nt":
            system_root = os.environ.get("SystemRoot", r"C:\Windows")
            shell = LocalSandbox._find_first_available_shell(
                (
                    "pwsh",
                    "pwsh.exe",
                    "powershell",
                    "powershell.exe",
                    ntpath.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
                    "cmd.exe",
                )
            )
            if shell is not None:
                return shell

            raise RuntimeError("No suitable shell executable found. Tried /bin/zsh, /bin/bash, /bin/sh, `sh` on PATH, then PowerShell and cmd.exe fallbacks for Windows.")

        raise RuntimeError("No suitable shell executable found. Tried /bin/zsh, /bin/bash, /bin/sh, and `sh` on PATH.")

    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        # ``env`` 키를 POSIX 환경 변수 규칙으로 검증한다. defense in depth 차원이다.
        # ``subprocess.run(env=...)``은 shell을 거치지 않으므로 여기 키에 metachar가 있어도 실제로
        # 주입되지는 않는다 — 하지만 공개 ``Sandbox.execute_command`` 계약은 키를
        # ``export <k>=<v>``에 이어 붙이는 AIO sandbox와 공유된다. 두 구현에 같은 규칙을 강제하면
        # 계약이 일관되게 유지되고, 새 호출자도 안전한 키 이름을 쓰게 된다.
        _validate_extra_env(env)
        # 실행 전에 명령 안의 container 경로를 해석한다
        resolved_command = self._resolve_paths_in_command(command)
        shell = self._get_shell()
        if timeout is None:
            timeout = DEFAULT_COMMAND_TIMEOUT_SECONDS

        # os.environ에서 플랫폼 secret을 뺀 값을 상속한 뒤, 주입된 request-scoped secret을 그 위에
        # 얹는다(#3861). 항상 명시적인 env를 넘기므로 플랫폼 credential이 skill subprocess로
        # 새어 나가지 않는다.
        sandbox_env = build_sandbox_env(env)
        timed_out = False
        if os.name == "nt":
            if self._is_powershell(shell):
                args = [shell, "-NoProfile", "-Command", resolved_command]
            elif self._is_cmd_shell(shell):
                args = [shell, "/c", resolved_command]
            else:
                args = [shell, "-c", resolved_command]
                if self._is_msys_shell(shell):
                    sandbox_env = {
                        **sandbox_env,
                        "MSYS_NO_PATHCONV": "1",
                        "MSYS2_ARG_CONV_EXCL": "*",
                    }

            try:
                result = subprocess.run(
                    args,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=sandbox_env,
                )
                stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = self._coerce_process_output(exc.stdout if exc.stdout is not None else exc.output)
                stderr = self._coerce_process_output(exc.stderr)
                returncode = 0
        else:
            args = [shell, "-c", resolved_command]
            stdout, stderr, returncode, timed_out = self._run_posix_command(args, timeout, sandbox_env)

        output = stdout
        if stderr:
            output += f"\nStd Error:\n{stderr}" if output else stderr
        if timed_out:
            notice = self._format_timeout_notice(timeout)
            output += f"\n{notice}" if output else notice
        elif returncode != 0:
            output += f"\nExit Code: {returncode}"

        final_output = output if output else "(no output)"
        # 출력 안의 로컬 경로를 container 경로로 역해석한다
        return self._reverse_resolve_paths_in_output(final_output)

    @staticmethod
    def _run_posix_command(
        args: list[str],
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> tuple[str, str, int, bool]:
        """POSIX에서 명령을 실행하며 pipe 캡처 크기를 제한한다.

        여기서는 ``subprocess.communicate()``를 쓸 수 없다. 백그라운드로 도는 장수 프로세스
        (``server &``)가 stdout/stderr를 상속해 pipe를 열어 두므로, foreground shell이 이미
        반환했는데도 ``communicate()``는 timeout까지 블로킹된다. 대신 daemon drain thread가 pipe를
        계속 비우면서 메모리에는 제한된 크기의 출력만 남긴다. 덕분에 백그라운드 프로세스에 눈에
        띄지 않게 커지는 익명 임시 파일을 넘기지 않고도, foreground shell이 종료되는 즉시 호출이
        반환된다. ``stdin``은 ``/dev/null``에서 받아 stdin을 읽는 명령이 즉시 EOF를 얻게 하고,
        ``start_new_session``은 명령을 자기 프로세스 그룹에 두어 실제로 블로킹되는 foreground
        명령이 timeout됐을 때 자식까지 통째로 kill할 수 있게 한다.

        ``env``는 :class:`subprocess.Popen`으로 전달된다. ``None``은 현재 프로세스 환경을
        상속한다는 뜻이다(일반적인 경우).

        ``(stdout, stderr, returncode, timed_out)``을 반환한다.
        """
        timed_out = False
        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        try:
            process = subprocess.Popen(
                args,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_write_fd,
                stderr=stderr_write_fd,
                start_new_session=True,
                env=env,
            )
        except Exception:
            for fd in (stdout_read_fd, stdout_write_fd, stderr_read_fd, stderr_write_fd):
                try:
                    os.close(fd)
                except OSError:
                    # 원래의 Popen 실패를 보존한다. fd 정리는 best-effort다.
                    pass
            raise
        finally:
            for fd in (stdout_write_fd, stderr_write_fd):
                try:
                    os.close(fd)
                except OSError:
                    # 위의 예외 정리에서 write fd가 이미 닫혔을 수 있다.
                    pass

        stdout_capture, stdout_thread = LocalSandbox._start_pipe_drain(stdout_read_fd, "deerflow-bash-stdout-drain")
        stderr_capture, stderr_thread = LocalSandbox._start_pipe_drain(stderr_read_fd, "deerflow-bash-stderr-drain")
        try:
            process_group_id = os.getpgid(process.pid)
        except OSError:
            process_group_id = None

        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                LocalSandbox._terminate_process_group(process)
            returncode = process.returncode if process.returncode is not None else 0
        finally:
            join_timeout = 10 if timed_out or not LocalSandbox._process_group_exists(process_group_id) else _PIPE_DRAIN_JOIN_TIMEOUT_SECONDS
            for thread in (stdout_thread, stderr_thread):
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.debug("Subprocess output drain thread still active after command returned")

        stdout = stdout_capture.read()
        stderr = stderr_capture.read()
        return stdout, stderr, returncode, timed_out

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        """명령의 프로세스 그룹 전체를 kill한 뒤 회수한다.

        그룹이 이미 사라졌으면(예: timeout과 이 호출 사이에 명령이 종료된 경우) 직계 자식만
        kill하는 것으로 되돌아간다.
        """
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # 프로세스 그룹이 이미 사라졌다(timeout과 이 호출 사이의 경쟁에서 명령이 종료됨).
            # 직계 자식만 kill하는 것으로 되돌아간다.
            try:
                process.kill()
            except OSError:
                # 직계 자식도 이미 회수됐다 — kill할 대상이 남아 있지 않다.
                logger.debug("Process %s already exited before fallback kill", process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process group for pid %s did not exit after SIGKILL", process.pid)

    def list_dir(self, path: str, max_depth=2) -> list[str]:
        resolved_path = self._resolve_path(path)
        entries = list_dir(resolved_path, max_depth)
        # 로컬 경로를 container 경로로 역해석하고, 디렉터리에 대한 list_dir의 후행 "/" 표시를
        # 유지한다.
        result: list[str] = []
        for entry in entries:
            is_dir = entry.endswith(("/", "\\"))
            reversed_entry = self._reverse_resolve_path(entry.rstrip("/\\")) if is_dir else self._reverse_resolve_path(entry)
            result.append(f"{reversed_entry}/" if is_dir and not reversed_entry.endswith("/") else reversed_entry)

        # virtual 하위 디렉터리 overlay: /mnt/skills 같은 container 경로에 자식 mapping(public,
        # custom, legacy)이 있고 그 local_path 대상이 해석된 host 디렉터리 바깥에 있으면(symlink나
        # bind-mount 방식), ``list_dir`` 유틸리티는 보안상 이를 건너뛴다. 여기서 빠진 virtual
        # 자식들을 다시 채워 넣어, agent가 ``ls /mnt/skills``로 발견할 수 있게 한다.
        container_path = path.rstrip("/")
        existing_dirs = {e.rstrip("/") for e in result if e.endswith("/")}
        for mapping in self.path_mappings:
            # mapping이 virtual 자식인 조건:
            # 1. container_path가 요청된 경로의 직계 자식이다
            # 2. 결과에 아직 없다(list_dir이 건너뛰었다)
            if mapping.container_path.startswith(container_path + "/"):
                child_rel = mapping.container_path[len(container_path) + 1 :]
                # 직계 자식만 대상으로 한다(슬래시가 더 없는 경우). 예: "public", "custom".
                # existing_dirs는 전체 경로(예: "/mnt/user-data/workspace")를 담고 있으므로, 맨
                # 자식 이름이 아니라 mapping의 전체 container 경로와 비교한다. 여기서 맨 이름을
                # 비교하면 절대 매칭되지 않아, 이미 나열된 mount(흔한 경우: /mnt/user-data 아래의
                # 실제 중첩 workspace/uploads/outputs 하위 디렉터리)가 두 번 추가된다.
                if "/" not in child_rel and mapping.container_path.rstrip("/") not in existing_dirs:
                    # 유령 항목을 추가하지 않도록 host 경로가 존재하는지 확인한다
                    try:
                        if Path(mapping.local_path).resolve().is_dir():
                            result.append(f"{mapping.container_path}/")
                    except OSError:
                        pass

        return sorted(result)

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        resolved_path = self._resolve_path(path)
        should_slice = start_line is not None or end_line is not None
        try:
            with open(resolved_path, encoding="utf-8") as f:
                if not should_slice:
                    content = f.read()

                start = max(start_line or 1, 1)
                if should_slice:
                    selected: list[str] = []
                    for line_number, line in enumerate(f, start=1):
                        if line_number < start:
                            continue
                        if end_line is not None and line_number > end_line:
                            break
                        selected.append(line.rstrip("\r\n"))
                    content = "\n".join(selected)
            # 앞서 write_file로 쓴 파일(agent가 작성한 content)에서만 경로를 역해석한다. 사용자가
            # 업로드한 파일, 외부 도구 출력, 그 밖의 agent 산출물이 아닌 content는 조용히
            # 재작성되면 안 된다 — PR #1935의 논의 참고.
            if resolved_path in self._agent_written_paths:
                content = self._reverse_resolve_paths_in_output(content)
            return content
        except OSError as e:
            # 내부 해석 경로를 감추고 더 명확한 오류 메시지를 주도록 원래 경로로 다시 raise한다
            raise type(e)(e.errno, e.strerror, path) from None

    def download_file(self, path: str) -> bytes:
        normalised = path.replace("\\", "/")
        stripped_path = normalised.lstrip("/")
        allowed_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")
        if stripped_path != allowed_prefix and not stripped_path.startswith(f"{allowed_prefix}/"):
            logger.error("Refused download outside allowed directory: path=%s, allowed_prefix=%s", path, VIRTUAL_PATH_PREFIX)
            raise PermissionError(errno.EACCES, f"Access denied: path must be under '{VIRTUAL_PATH_PREFIX}'", path)

        resolved_path = self._resolve_path(path)
        max_download_size = 100 * 1024 * 1024
        try:
            file_size = os.path.getsize(resolved_path)
            if file_size > max_download_size:
                raise OSError(errno.EFBIG, f"File exceeds maximum download size of {max_download_size} bytes", path)
            # TOCTOU 주의: getsize()와 read() 사이에 파일이 커질 수 있다. 통제된 sandbox
            # 환경이므로 감수하는 tradeoff다.
            with open(resolved_path, "rb") as f:
                return f.read()
        except OSError as e:
            # 내부 해석 경로를 감추고 더 명확한 오류 메시지를 주도록 원래 경로로 다시 raise한다
            raise type(e)(e.errno, e.strerror, path) from None

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        resolved = self._resolve_path_with_mapping(path)
        resolved_path = resolved.path
        if self._is_resolved_path_read_only(resolved):
            raise OSError(errno.EROFS, "Read-only file system", path)
        try:
            dir_path = os.path.dirname(resolved_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            # content 전용 resolver(슬래시 안전)를 써서 content 안의 container 경로를
            # 로컬 경로로 해석한다
            resolved_content = self._resolve_paths_in_content(content)
            mode = "a" if append else "w"
            with open(resolved_path, mode, encoding="utf-8") as f:
                f.write(resolved_content)
            # read_file이 읽을 때 역해석해야 함을 알 수 있도록 이 경로를 기록한다.
            # agent가 쓴 파일만 역해석되고, 사용자 업로드와 외부 도구 출력은 그대로 둔다.
            self._agent_written_paths.add(resolved_path)
        except OSError as e:
            # 내부 해석 경로를 감추고 더 명확한 오류 메시지를 주도록 원래 경로로 다시 raise한다
            raise type(e)(e.errno, e.strerror, path) from None

    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        resolved_path = Path(self._resolve_path(path))
        matches, truncated = find_glob_matches(resolved_path, pattern, include_dirs=include_dirs, max_results=max_results)
        return [self._reverse_resolve_path(match) for match in matches], truncated

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
        resolved_path = Path(self._resolve_path(path))
        matches, truncated = find_grep_matches(
            resolved_path,
            pattern,
            glob_pattern=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
        return [
            GrepMatch(
                path=self._reverse_resolve_path(match.path),
                line_number=match.line_number,
                line=match.line,
            )
            for match in matches
        ], truncated

    def update_file(self, path: str, content: bytes) -> None:
        resolved = self._resolve_path_with_mapping(path)
        resolved_path = resolved.path
        if self._is_resolved_path_read_only(resolved):
            raise OSError(errno.EROFS, "Read-only file system", path)
        try:
            dir_path = os.path.dirname(resolved_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(resolved_path, "wb") as f:
                f.write(content)
        except OSError as e:
            # 내부 해석 경로를 감추고 더 명확한 오류 메시지를 주도록 원래 경로로 다시 raise한다
            raise type(e)(e.errno, e.strerror, path) from None
