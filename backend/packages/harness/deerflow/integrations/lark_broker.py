"""Lark CLI sandbox 자격증명 broker(Pattern B, issue #4338).

Pattern A(PR #3971)는 ``lark-cli`` *바이너리*를 sandbox에 넣어주지만, 사용자별 자격증명
디렉터리(장수명 ``appSecret``이 든 ``config``, OAuth 토큰이 든 ``data``)를 여전히
sandbox 컨테이너에 마운트하므로 에이전트의 ``bash`` 도구가 읽을 수 있다.

이 모듈은 Pattern B의 broker 쪽을 구현한다. ``lark-cli``와 자격증명을 소유하는 장수명
프로세스가 loopback으로 *명령 표면*만 노출한다. sandbox에는 argv/stdin을 broker로
전달하는 작은 ``lark-cli`` shim만 ``PATH``에 놓이므로, 원본 자격증명 파일은 sandbox
파일시스템에 아예 존재하지 않는다.

여기 있는 코드는 Python 3 표준 라이브러리만 쓴다. 최소 구성의 broker sidecar 이미지에서
추가 의존성 없이 같은 모듈을 실행할 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

# ── Loopback 통신 규약 ────────────────────────────────────────────────

# sandbox와 broker sidecar는 Pod 네트워크 네임스페이스를 공유하므로 shim은 loopback으로
# broker에 닿는다. 포트는 고정이며 DEERFLOW_LARK_BROKER_URL로 sandbox에 주입된다.
LARK_BROKER_DEFAULT_HOST = "127.0.0.1"
LARK_BROKER_DEFAULT_PORT = 8788
LARK_BROKER_URL_ENV = "DEERFLOW_LARK_BROKER_URL"
LARK_BROKER_EXEC_PATH = "/v1/exec"
LARK_BROKER_HEALTH_PATH = "/v1/health"

# 방어용 상한. 침해된 sandbox가 broker 자원을 고갈시키지 못하게 한다.
LARK_BROKER_MAX_REQUEST_BYTES = 1 * 1024 * 1024
LARK_BROKER_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
LARK_BROKER_DEFAULT_TIMEOUT_SECONDS = 120
LARK_BROKER_MAX_CONCURRENCY = 8
# 연결당 socket timeout. ThreadingHTTPServer는 연결마다 thread를 띄우므로, 이게 없으면
# sandbox가 큰 Content-Length만 선언하고 본문을 보내지 않아 thread를 영원히 붙잡을 수 있다.
# 읽기에 상한을 둬서 느리거나 멈춘 클라이언트가 thread를 놓아주게 한다.
# (loopback 전용이라 결국 sandbox가 자기 broker를 묶는 것을 막는 용도다.)
LARK_BROKER_SOCKET_TIMEOUT_SECONDS = 30

# 아래 subcommand denylist를 지정하는 선택적 env(쉼표 구분).
LARK_BROKER_DENY_SUBCOMMANDS_ENV = "DEERFLOW_LARK_BROKER_DENY_SUBCOMMANDS"

# sandbox가 보는 런타임 레이아웃은 ``bin/`` 아래 파일 두 개다.
#
#   bin/lark-cli          POSIX sh *launcher*(아래) — PATH에 놓이는 실행 파일
#   bin/lark-cli-shim.py  launcher가 exec하는 Python *shim* 본체(아래)
#
# Pattern A의 launcher는 순수 ``#!/bin/sh``라(아키텍처별 바이너리를 exec할 뿐) 런타임
# 의존성이 없다. broker shim은 HTTP를 말해야 해서 Python이다. 그런데 이를 그냥
# ``#!/usr/bin/env python3`` 스크립트로 배포하면 ``python3``가 PATH에 없는 sandbox
# 이미지에서 모든 ``lark-cli`` 호출이 ENOEXEC/exit 127로 죽는다. broker 모드는 opt-in이라
# 이 문제가 CI를 통과해 운영자에게서만 드러날 수 있다. 그래서 ``/bin/sh`` launcher가 직접
# Python 3 인터프리터를 찾고(shell 내장만 쓰므로 PATH가 비어 있고 인터프리터가 고정된
# 상황에서도 동작한다), 없으면 불투명한 ENOEXEC 대신 조치 가능한 메시지와 함께 *요란하게*
# 실패한다. ``DEERFLOW_LARK_BROKER_PYTHON``은 Python을 비표준 이름으로 제공하는 이미지를
# 위해 특정 인터프리터를 고정한다.
#
# launcher가 참조하는 shim 본체 경로는 ``$0``에서 유도하지 않고 설치 시점에 박아 넣는다.
# sandbox가 PATH로 ``lark-cli``를 실행하면 ``$0``는 디렉터리 없는 명령 이름뿐이라 형제 파일
# 탐색이 실패하기 때문이다. 설치 디렉터리는 안정적인 공유 마운트라 절대 경로가 이를 쓰는
# init container와 읽는 sandbox 양쪽에서 모두 유효하다.
#
# 두 스크립트 모두 여기를 단일 진실 원천으로 두고 broker 이미지 빌드가 ``install_shim``으로
# 복제한다. Pattern A의 ``LARK_CLI_SANDBOX_LAUNCHER_SCRIPT``와 같은 방식이며, 이미지 사본이
# Gateway 쪽과 어긋나지 않게 한다.
LARK_BROKER_PYTHON_ENV = "DEERFLOW_LARK_BROKER_PYTHON"
LARK_CLI_BROKER_SHIM_FILENAME = "lark-cli-shim.py"
_LARK_CLI_BROKER_SHIM_PATH_PLACEHOLDER = "@@LARK_CLI_BROKER_SHIM_PATH@@"

LARK_CLI_BROKER_LAUNCHER_TEMPLATE = (
    "#!/bin/sh\n"
    "# DeerFlow lark-cli broker launcher (Pattern B). Resolves a Python 3\n"
    "# interpreter and execs the forwarding shim. Fails loudly (not with an opaque\n"
    "# ENOEXEC) when the sandbox image ships no python3. Uses only shell built-ins\n"
    "# so it still works when PATH is empty and DEERFLOW_LARK_BROKER_PYTHON pins\n"
    "# the interpreter.\n"
    "set -eu\n"
    'shim="' + _LARK_CLI_BROKER_SHIM_PATH_PLACEHOLDER + '"\n'
    'if [ -n "${DEERFLOW_LARK_BROKER_PYTHON:-}" ]; then\n'
    '  exec "$DEERFLOW_LARK_BROKER_PYTHON" "$shim" "$@"\n'
    "fi\n"
    "for _py in python3 python; do\n"
    '  if command -v "$_py" >/dev/null 2>&1; then\n'
    '    exec "$_py" "$shim" "$@"\n'
    "  fi\n"
    "done\n"
    'echo "lark-cli: broker mode needs a Python 3 interpreter but none was found;" >&2\n'
    'echo "          set DEERFLOW_LARK_BROKER_PYTHON to a python3 path in the sandbox image." >&2\n'
    "exit 127\n"
)


def render_launcher_script(shim_path: str) -> str:
    """shim 본체의 절대 경로를 박아 넣은 ``/bin/sh`` launcher를 렌더링한다."""
    return LARK_CLI_BROKER_LAUNCHER_TEMPLATE.replace(_LARK_CLI_BROKER_SHIM_PATH_PLACEHOLDER, shim_path)


# shim은 argv/stdin을 읽어 broker에 POST하고 broker의 stdout/stderr/종료 코드를 그대로
# 재현한다. 전송이 실패하면 요란하게 0이 아닌 코드로 죽으므로 broker 장애가 성공한
# lark-cli 실행처럼 보이는 일은 없다. 위 launcher가 ``<python> lark-cli-shim.py <args...>``
# 형태로 호출하므로 자체 shebang 해석에 의존하지 않으며, stdin/argv는 그대로 통과한다.
LARK_CLI_BROKER_SHIM_SCRIPT = r'''#!/usr/bin/env python3
"""DeerFlow lark-cli broker shim (Pattern B). Forwards argv/stdin to the broker.

Note: the broker runs lark-cli in the *sidecar's* working directory and cannot
see the sandbox filesystem, so cwd is intentionally not forwarded. Subcommands
that read/write files relative to the sandbox cwd are unsupported in broker mode.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

BROKER_URL = os.environ.get("DEERFLOW_LARK_BROKER_URL", "http://127.0.0.1:8788")


def _fail(message, code=127):
    sys.stderr.write("lark-cli: " + message + "\n")
    sys.exit(code)


def main():
    try:
        stdin_bytes = b"" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.buffer.read()
    except Exception:
        stdin_bytes = b""
    payload = json.dumps(
        {
            "args": sys.argv[1:],
            "stdin_b64": base64.b64encode(stdin_bytes).decode("ascii"),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        BROKER_URL.rstrip("/") + "/v1/exec",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        _fail("broker rejected request (HTTP %d%s)" % (exc.code, ": " + detail if detail else ""))
    except (urllib.error.URLError, OSError) as exc:
        _fail("broker unreachable at %s (%s)" % (BROKER_URL, exc))
    except Exception as exc:  # noqa: BLE001
        _fail("broker call failed (%s)" % exc)
    sys.stdout.buffer.write(base64.b64decode(body.get("stdout_b64", "")))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(base64.b64decode(body.get("stderr_b64", "")))
    sys.stderr.buffer.flush()
    sys.exit(int(body.get("exit_code", 1)))


if __name__ == "__main__":
    main()
'''


# ── Broker 서버 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BrokerConfig:
    """broker sidecar의 런타임 설정."""

    lark_cli_path: str
    config_dir: str
    data_dir: str
    host: str = LARK_BROKER_DEFAULT_HOST
    port: int = LARK_BROKER_DEFAULT_PORT
    timeout_seconds: int = LARK_BROKER_DEFAULT_TIMEOUT_SECONDS
    # broker가 실행을 거부할 ``lark-cli`` subcommand 경로의 opt-in denylist(issue #4338 강화).
    # 각 항목은 공백으로 이어진 명령 prefix이며(예: "config show", "auth token") 요청의
    # 선두 non-flag 토큰과 대조한다. prompt injection된 에이전트가 닿을 수 있는 명령 표면을
    # 좁힌다. broker는 자격증명 *파일*은 이미 제거했지만, 비밀을 덤프하는 subcommand를
    # 여기서 막지 않으면 명령 표면은 그대로 열려 있다. 기본값은 비어 있다(동작 변화 없음).
    deny_subcommands: tuple[tuple[str, ...], ...] = ()

    def credential_env(self) -> dict[str, str]:
        """broker가 모든 lark-cli 호출에 주입하는 env.

        클라이언트는 이 값을 제공하지 않는다. 자격증명 경로는 broker가 소유하므로
        sandbox 프로세스가 lark-cli를 다른 profile로 돌릴 수 없다.
        """
        return {
            "LARKSUITE_CLI_CONFIG_DIR": self.config_dir,
            "LARKSUITE_CLI_DATA_DIR": self.data_dir,
            "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
            "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
        }


def parse_deny_subcommands(raw: str | None) -> tuple[tuple[str, ...], ...]:
    """쉼표로 구분된 denylist env를 명령 prefix tuple로 파싱한다.

    ``"config show, auth token"`` → ``(("config", "show"), ("auth", "token"))``.
    비어 있거나 공백뿐인 항목은 버린다.
    """
    if not raw:
        return ()
    prefixes: list[tuple[str, ...]] = []
    for entry in raw.split(","):
        tokens = tuple(entry.split())
        if tokens:
            prefixes.append(tokens)
    return tuple(prefixes)


def _denied_subcommand(deny: tuple[tuple[str, ...], ...], args: list[str]) -> tuple[str, ...] | None:
    """``args``가 금지된 subcommand면 매칭된 denylist prefix를 반환한다.

    선두 non-flag 토큰과 대조하므로(옵션과 그 값은 건너뛴다) ``config --json show``도
    ``config show`` 규칙에 걸린다.
    """
    if not deny:
        return None
    positional = [token for token in args if not token.startswith("-")]
    for prefix in deny:
        if positional[: len(prefix)] == list(prefix):
            return prefix
    return None


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    truncated: bool


def run_lark_cli(config: BrokerConfig, args: list[str], stdin: bytes) -> ExecResult:
    """broker가 소유한 자격증명으로 ``lark-cli``를 한 번 실행한다.

    ``args``는 ``shell=False``와 함께 argv 리스트로 넘기므로 sandbox가 준 인자가 shell을
    거쳐 두 번째 명령으로 해석될 수 없다. ``deny_subcommands``에 걸리면 바이너리를 띄우기
    전에 거부한다.
    """
    denied = _denied_subcommand(config.deny_subcommands, args)
    if denied is not None:
        message = f"lark-cli: subcommand '{' '.join(denied)}' is disabled in broker mode\n"
        return ExecResult(126, b"", message.encode("utf-8"), False)
    env = {**os.environ, **config.credential_env()}
    try:
        completed = subprocess.run(  # noqa: S603 - argv 리스트, shell=False, 고정 바이너리
            [config.lark_cli_path, *args],
            input=stdin,
            capture_output=True,
            timeout=config.timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(124, b"", b"lark-cli: broker timed out\n", False)
    except FileNotFoundError:
        return ExecResult(127, b"", b"lark-cli: binary not found in broker\n", False)

    stdout, out_trunc = _cap(completed.stdout or b"")
    stderr, err_trunc = _cap(completed.stderr or b"")
    return ExecResult(completed.returncode, stdout, stderr, out_trunc or err_trunc)


def _cap(data: bytes) -> tuple[bytes, bool]:
    if len(data) <= LARK_BROKER_MAX_OUTPUT_BYTES:
        return data, False
    return data[:LARK_BROKER_MAX_OUTPUT_BYTES], True


def make_handler(config: BrokerConfig) -> type[BaseHTTPRequestHandler]:
    """``config``에 바인딩된 request handler를 만든다.

    bounded semaphore로 동시 실행 수를 제한해, sandbox 호출이 몰려도 ``lark-cli``
    subprocess가 무제한으로 생기지 않게 한다.
    """
    semaphore = threading.BoundedSemaphore(LARK_BROKER_MAX_CONCURRENCY)

    class Handler(BaseHTTPRequestHandler):
        # 연결당 읽기에 상한을 둬서, 큰 Content-Length만 선언하고 본문을 보내지 않는
        # 클라이언트가 thread를 영원히 붙잡지 못하게 한다(ThreadingHTTPServer는 연결당
        # thread 하나다). 표준 라이브러리가 이 값을 읽어 socket timeout을 건다.
        timeout = LARK_BROKER_SOCKET_TIMEOUT_SECONDS

        # 조용히: 기본 BaseHTTPRequestHandler는 요청마다 stderr에 로그를 남긴다.
        def log_message(self, *_args: Any) -> None:  # noqa: D401
            return

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - 표준 라이브러리 API
            if self.path.rstrip("/") == LARK_BROKER_HEALTH_PATH:
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - 표준 라이브러리 API
            if self.path.rstrip("/") != LARK_BROKER_EXEC_PATH:
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "bad content-length"})
                return
            if length <= 0 or length > LARK_BROKER_MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "request too large"})
                return
            try:
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                args = request["args"]
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    raise ValueError("args must be a list of strings")
                stdin = base64.b64decode(request.get("stdin_b64", "") or "")
            except Exception:  # noqa: BLE001 - 신뢰할 수 없는 클라이언트 입력
                self._send_json(400, {"error": "invalid request"})
                return

            if not semaphore.acquire(blocking=False):
                self._send_json(503, {"error": "broker busy"})
                return
            try:
                result = run_lark_cli(config, args, stdin)
            except Exception:  # noqa: BLE001 - 통신 규약을 일관되게 유지한다
                # run_lark_cli은 예상 가능한 실패(timeout, 바이너리 없음)를 이미
                # ExecResult로 변환한다. 그 밖의 예외(OSError, PermissionError 등)는
                # 본문 없이 연결이 닫히면서 shim에서 불투명한 전송 오류로 보인다.
                # 구조화된 500을 반환해 shim이 의미 있는 에러를 알리게 한다.
                logger.exception("lark-cli exec failed unexpectedly")
                self._send_json(500, {"error": "broker exec failed"})
                return
            finally:
                semaphore.release()

            self._send_json(
                200,
                {
                    "exit_code": result.exit_code,
                    "stdout_b64": base64.b64encode(result.stdout).decode("ascii"),
                    "stderr_b64": base64.b64encode(result.stderr).decode("ascii"),
                    "truncated": result.truncated,
                },
            )

    return Handler


def serve(config: BrokerConfig) -> ThreadingHTTPServer:
    """loopback에 바인딩한 broker HTTP 서버를 시작하고 반환한다."""
    if not shutil.which(config.lark_cli_path) and not os.path.isfile(config.lark_cli_path):
        logger.warning("lark-cli not found at %s; broker will report 127 for exec", config.lark_cli_path)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    logger.info("lark-cli broker listening on %s:%d", config.host, config.port)
    return server


def install_shim(dest_dir: str, *, version: str | None = None) -> str:
    """launcher, shim, 런타임 마커를 sandbox 런타임 디렉터리에 기록한다.

    broker 이미지의 ``install-shim`` init container 모드에서 호출한다. Pattern A가 만드는
    것과 같은 ``bin/lark-cli`` + ``.deerflow-lark-cli-runtime.json`` 레이아웃을 만들되
    ``kind="shim"``으로 표시해, 런타임 검증기가 ``linux-*`` 바이너리 부재가 의도된 것임을
    알게 한다(실제 바이너리는 sidecar가 갖고 있다).

    ``bin/``에는 파일 두 개를 쓴다. PATH에 놓이는 실행 파일 ``lark-cli``는 Python 3
    인터프리터를 찾아 옆의 ``lark-cli-shim.py`` *본체*를 exec하는 ``/bin/sh`` *launcher*다.
    둘을 나눠야 ``python3``가 없는 sandbox 이미지에서 broker 모드가 ENOEXEC로 조용히
    실패하지 않는다(대신 launcher가 조치 가능한 메시지로 요란하게 실패한다). 둘 다 이
    프로세스의 상수에서 나오므로 이미지 사본이 Gateway 쪽과 어긋날 수 없다.
    """
    dest = os.path.abspath(dest_dir)
    bin_dir = os.path.join(dest, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    shim_body = os.path.join(bin_dir, LARK_CLI_BROKER_SHIM_FILENAME)
    with open(shim_body, "w", encoding="utf-8") as handle:
        handle.write(LARK_CLI_BROKER_SHIM_SCRIPT)
    os.chmod(shim_body, 0o755)
    launcher = os.path.join(bin_dir, "lark-cli")
    with open(launcher, "w", encoding="utf-8") as handle:
        handle.write(render_launcher_script(shim_body))
    os.chmod(launcher, 0o755)
    marker = os.path.join(dest, ".deerflow-lark-cli-runtime.json")
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"version": version or "unknown", "kind": "shim"}, handle)
    return launcher


def _config_from_env() -> BrokerConfig:
    return BrokerConfig(
        lark_cli_path=os.environ.get("DEERFLOW_LARK_BROKER_CLI", "lark-cli"),
        config_dir=os.environ.get("LARKSUITE_CLI_CONFIG_DIR", "/var/lark/config"),
        data_dir=os.environ.get("LARKSUITE_CLI_DATA_DIR", "/var/lark/data"),
        host=os.environ.get("DEERFLOW_LARK_BROKER_HOST", LARK_BROKER_DEFAULT_HOST),
        port=int(os.environ.get("DEERFLOW_LARK_BROKER_PORT", str(LARK_BROKER_DEFAULT_PORT))),
        timeout_seconds=int(os.environ.get("DEERFLOW_LARK_BROKER_TIMEOUT", str(LARK_BROKER_DEFAULT_TIMEOUT_SECONDS))),
        deny_subcommands=parse_deny_subcommands(os.environ.get(LARK_BROKER_DENY_SUBCOMMANDS_ENV)),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    argv = sys.argv[1:]
    if argv and argv[0] == "install-shim":
        dest = argv[1] if len(argv) > 1 else os.environ.get("LARK_CLI_RUNTIME_DEST", "/mnt/integrations/lark-cli/runtime")
        launcher = install_shim(dest, version=os.environ.get("LARK_CLI_VERSION"))
        logger.info("Installed lark-cli broker shim at %s", launcher)
        return
    server = serve(_config_from_env())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
