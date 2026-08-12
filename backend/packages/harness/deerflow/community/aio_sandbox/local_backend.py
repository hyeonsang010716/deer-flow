"""sandbox provisioning용 로컬 컨테이너 backend.

로컬 머신에서 Docker 또는 Apple Container로 sandbox 컨테이너를 관리한다. 컨테이너
lifecycle, port 할당, cross-process 컨테이너 탐색을 담당한다.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from datetime import datetime

from deerflow.utils.network import get_free_port, release_port

from .backend import SandboxBackend, wait_for_sandbox_ready
from .sandbox_info import SandboxInfo

logger = logging.getLogger(__name__)


def _parse_docker_timestamp(raw: str) -> float:
    """Docker의 ISO 8601 timestamp를 Unix epoch float로 파싱한다.

    Docker는 나노초 정밀도에 끝에 ``Z``가 붙은 timestamp를 준다
    (예: ``2026-04-08T01:22:50.123456789Z``). Python의 ``fromisoformat``은 마이크로초까지만
    받고 3.11 이전에는 ``Z``도 못 받으므로, 파싱 전에 문자열을 정규화한다. 입력이 비었거나
    파싱에 실패하면 ``0.0``을 반환해 호출자가 "나이 불명" sentinel로 쓸 수 있게 한다.
    """
    if not raw:
        return 0.0
    try:
        s = raw.strip()
        if "." in s:
            dot_pos = s.index(".")
            tz_start = dot_pos + 1
            while tz_start < len(s) and s[tz_start].isdigit():
                tz_start += 1
            frac = s[dot_pos + 1 : tz_start][:6]  # 마이크로초까지만 남긴다
            tz_suffix = s[tz_start:]
            s = s[: dot_pos + 1] + frac + tz_suffix
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not parse docker timestamp {raw!r}: {e}")
        return 0.0


def _extract_host_port(inspect_entry: dict, container_port: int) -> int | None:
    """docker inspect 항목에서 ``container_port/tcp``에 매핑된 host port를 뽑아낸다.

    해당 port에 대한 매핑이 없으면 None을 반환한다.
    """
    try:
        ports = (inspect_entry.get("NetworkSettings") or {}).get("Ports") or {}
        bindings = ports.get(f"{container_port}/tcp") or []
        if bindings:
            host_port = bindings[0].get("HostPort")
            if host_port:
                return int(host_port)
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _format_container_mount(runtime: str, host_path: str, container_path: str, read_only: bool) -> list[str]:
    """선택된 runtime에 맞는 bind-mount 인자를 만든다.

    Docker의 ``-v host:container`` 문법은 ``D:/...`` 같은 Windows 드라이브 문자 경로에서
    모호하다. ``:``가 드라이브 구분자이자 volume 구분자이기 때문이다. Docker에서는 이
    파싱 모호성을 피하려고 ``--mount type=bind,...``를 쓴다. Apple Container는 계속 ``-v``를 쓴다.
    """
    if runtime == "docker":
        mount_spec = f"type=bind,src={host_path},dst={container_path}"
        if read_only:
            mount_spec += ",readonly"
        return ["--mount", mount_spec]

    mount_spec = f"{host_path}:{container_path}"
    if read_only:
        mount_spec += ":ro"
    return ["-v", mount_spec]


def _redact_container_command_for_log(cmd: list[str]) -> list[str]:
    """환경 변수 값을 가린 Docker/Container 명령을 반환한다."""
    redacted: list[str] = []
    redact_next_env = False

    for arg in cmd:
        if redact_next_env:
            if "=" in arg:
                key = arg.split("=", 1)[0]
                redacted.append(f"{key}=<redacted>" if key else "<redacted>")
            else:
                redacted.append(arg)
            redact_next_env = False
            continue

        if arg in {"-e", "--env"}:
            redacted.append(arg)
            redact_next_env = True
            continue

        if arg.startswith("--env="):
            value = arg.removeprefix("--env=")
            if "=" in value:
                key = value.split("=", 1)[0]
                redacted.append(f"--env={key}=<redacted>" if key else "--env=<redacted>")
            else:
                redacted.append(arg)
            continue

        redacted.append(arg)

    return redacted


def _format_container_command_for_log(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _normalize_sandbox_host(host: str) -> str:
    return host.strip().lower()


def _is_ipv6_loopback_sandbox_host(host: str) -> bool:
    return _normalize_sandbox_host(host) in {"::1", "[::1]"}


def _is_loopback_sandbox_host(host: str) -> bool:
    return _normalize_sandbox_host(host) in {"", "localhost", "127.0.0.1", "::1", "[::1]"}


def _resolve_docker_bind_host(sandbox_host: str | None = None, bind_host: str | None = None) -> str:
    """레거시 Docker ``-p`` sandbox 공개에 쓸 host 인터페이스를 고른다.

    베어메탈/로컬 실행은 localhost로 sandbox와 통신하므로 sandbox HTTP API를 모든 host
    인터페이스에 노출하면 안 된다. Docker-outside-of-Docker 배포는 보통 다른 컨테이너에서
    ``host.docker.internal``을 쓰므로, 운영자가 ``DEER_FLOW_SANDBOX_BIND_HOST``로 더 좁은
    bind를 선택하지 않는 한 기존의 넓은 bind를 유지한다. 운영자가 IPv6 loopback sandbox
    host를 고르면 Docker도 IPv6 loopback에 bind해서, 광고되는 sandbox URL과 공개된 소켓이
    같은 address family를 쓰게 한다.
    """
    explicit_bind = bind_host if bind_host is not None else os.environ.get("DEER_FLOW_SANDBOX_BIND_HOST")
    if explicit_bind is not None:
        explicit_bind = explicit_bind.strip()
        if explicit_bind:
            logger.debug("Docker sandbox bind: %s (explicit bind host override)", explicit_bind)
            return explicit_bind

    host = sandbox_host if sandbox_host is not None else os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")
    if _is_ipv6_loopback_sandbox_host(host):
        logger.debug("Docker sandbox bind: [::1] (IPv6 loopback sandbox host)")
        return "[::1]"
    if _is_loopback_sandbox_host(host):
        logger.debug("Docker sandbox bind: 127.0.0.1 (loopback default)")
        return "127.0.0.1"

    logger.debug("Docker sandbox bind: 0.0.0.0 (non-loopback sandbox host compatibility)")
    return "0.0.0.0"


def _is_no_such_container_error(stderr: str, container_name: str) -> bool:
    """stderr가 컨테이너 부재를 확정적으로 말할 때만 True를 반환한다.

    Docker는 "No such object" / "No such container"를 낸다. Apple Container는 뭉뚱그린
    "not found"를 내므로, 메시지가 검사 대상 컨테이너 이름을 함께 언급하거나
    container/object를 가리킬 때만 그 문구를 신뢰한다. 텍스트에 우연히 "not found"가 들어간
    일시적 실패(예: "command not found", "context not found")는 죽은 컨테이너로 오독하지
    말고 raise 경로에 남겨야 한다.
    """
    message = stderr.lower()
    if "no such object" in message or "no such container" in message:
        return True
    if "not found" not in message:
        return False
    return container_name.lower() in message or "container" in message or "object" in message


class LocalContainerBackend(SandboxBackend):
    """Docker 또는 Apple Container로 sandbox 컨테이너를 로컬에서 관리하는 backend.

    macOS에서는 Apple Container가 있으면 자동으로 우선 쓰고, 없으면 Docker로 폴백한다.
    다른 플랫폼에서는 Docker를 쓴다.

    기능:

    - cross-process 탐색을 위한 결정적 컨테이너 이름
    - thread-safe 유틸리티를 통한 port 할당
    - 컨테이너 lifecycle 관리(--rm으로 start/stop)
    - volume mount와 환경 변수 지원
    """

    # `stop` 한 번에 대한 실제 시간 상한. runtime 자체의 기본 SIGKILL 승격(docker/podman은
    # 10초)보다 충분히 크므로, 느리지만 진행 중인 stop을 자르지 않고 daemon 자체가 먹통일
    # 때만 발동한다.
    _STOP_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        *,
        image: str,
        base_port: int,
        container_prefix: str,
        config_mounts: list,
        environment: dict[str, str],
    ):
        """로컬 컨테이너 backend를 초기화한다.

        Args:
            image: 사용할 컨테이너 이미지.
            base_port: 빈 port 탐색을 시작할 기준 port 번호.
            container_prefix: 컨테이너 이름 prefix (예: "deer-flow-sandbox").
            config_mounts: config에서 온 volume mount 설정(VolumeMountConfig 목록).
            environment: 컨테이너에 주입할 환경 변수.
        """
        self._image = image
        self._base_port = base_port
        self._container_prefix = container_prefix
        self._config_mounts = config_mounts
        self._environment = environment
        self._runtime = self._detect_runtime()

    @property
    def runtime(self) -> str:
        """감지된 컨테이너 runtime("docker" 또는 "container")."""
        return self._runtime

    def _detect_runtime(self) -> str:
        """어떤 컨테이너 runtime을 쓸지 감지한다.

        macOS에서는 Apple Container가 있으면 우선하고, 없으면 Docker로 폴백한다.
        다른 플랫폼에서는 Docker를 쓴다.

        Returns:
            Apple Container면 "container", Docker면 "docker".
        """
        import platform

        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["container", "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                logger.info(f"Detected Apple Container: {result.stdout.strip()}")
                return "container"
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.info("Apple Container not available, falling back to Docker")

        return "docker"

    # ── SandboxBackend 인터페이스 ──────────────────────────────────────────

    def create(
        self,
        thread_id: str | None,
        sandbox_id: str,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
        *,
        user_id: str | None = None,
        provision_lark_cli_runtime: bool = False,
        provision_lark_cli_broker: bool = False,
    ) -> SandboxInfo:
        """새 컨테이너를 띄우고 연결 정보를 반환한다.

        Args:
            thread_id: sandbox를 생성할 대상 thread ID. sandbox를 thread 단위로 정리하려는 backend에 유용하다.
            sandbox_id: 결정적 sandbox 식별자(컨테이너 이름에 쓰인다).
            extra_mounts: 추가 volume mount. (host_path, container_path, read_only) 튜플 목록이다.
            user_id: extra_mounts에 이미 반영된 사용자 bucket. remote backend와 인터페이스를
                맞추기 위해 받기만 한다.
            provision_lark_cli_runtime: 무시한다. 로컬 backend는 extra_mounts의
                Gateway-download bind mount로 lark-cli runtime을 공급한다.
            provision_lark_cli_broker: 무시한다. 로컬 backend에는 보호할 sandbox 경계가 없으므로
                credential-mount overlay를 그대로 유지한다.

        Returns:
            컨테이너 정보가 담긴 SandboxInfo.

        Raises:
            RuntimeError: 컨테이너 기동에 실패한 경우.
        """
        del user_id, provision_lark_cli_runtime, provision_lark_cli_broker
        container_name = f"{self._container_prefix}-{sandbox_id}"

        # 재시도 루프: Docker가 port를 거부하면(예: 프로세스 재시작 후 낡은 컨테이너가 아직
        # binding을 쥐고 있는 경우) 그 port를 건너뛰고 다음 port로 시도한다. get_free_port의
        # 소켓 bind 검사는 Docker의 0.0.0.0 bind를 흉내 내지만 Docker의 port 해제는 약간
        # 비동기일 수 있으므로, 여기 반응형 폴백이 항상 진행을 보장한다.
        _next_start = self._base_port
        container_id: str | None = None
        port: int = 0
        for _attempt in range(10):
            port = get_free_port(start_port=_next_start)
            try:
                container_id = self._start_container(container_name, port, extra_mounts)
                break
            except RuntimeError as exc:
                release_port(port)
                err = str(exc)
                err_lower = err.lower()
                # port가 이미 점유됨: 이 port를 건너뛰고 다음 port로 재시도한다.
                if "port is already allocated" in err or "address already in use" in err_lower:
                    logger.warning(f"Port {port} rejected by Docker (already allocated), retrying with next port")
                    _next_start = port + 1
                    continue
                # 컨테이너 이름 충돌: 다른 프로세스가 이미 이 sandbox_id의 결정적 sandbox
                # 컨테이너를 띄웠을 수 있다. 실패하는 대신 기존 컨테이너를 찾아 흡수한다.
                if "is already in use by container" in err_lower or "conflict. the container name" in err_lower:
                    logger.warning(f"Container name {container_name} already in use, attempting to discover existing sandbox instance")
                    existing = self.discover(sandbox_id)
                    if existing is not None:
                        return existing
                raise
        else:
            raise RuntimeError("Could not start sandbox container: all candidate ports are already allocated by Docker")

        # Docker 안에서 실행할 때(DooD) sandbox 컨테이너는 host daemon에서 돌기 때문에
        # localhost가 아니라 host.docker.internal로 접근한다.
        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")
        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_url=f"http://{sandbox_host}:{port}",
            container_name=container_name,
            container_id=container_id,
        )

    def destroy(self, info: SandboxInfo) -> None:
        """컨테이너를 멈추고 port를 반환한다."""
        # container_id를 우선하고 없으면 container_name으로 폴백한다(docker stop은 둘 다 받는다).
        # 그래야 이름만 가진 list_running()으로 찾은 컨테이너도 멈출 수 있다.
        stop_target = info.container_id or info.container_name
        if stop_target:
            self._stop_container(stop_target)
        # 반환할 port를 sandbox_url에서 뽑아낸다
        try:
            from urllib.parse import urlparse

            port = urlparse(info.sandbox_url).port
            if port:
                release_port(port)
        except Exception:
            pass

    def is_alive(self, info: SandboxInfo) -> bool:
        """컨테이너가 아직 실행 중인지 확인한다(가볍고 HTTP를 쓰지 않는다)."""
        if info.container_name:
            return self._is_container_running(info.container_name)
        return False

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """결정적 이름으로 기존 컨테이너를 찾는다.

        기대하는 이름의 컨테이너가 실행 중인지 확인하고, port를 얻고, health check에
        응답하는지 검증한다.

        Args:
            sandbox_id: 결정적 sandbox ID(컨테이너 이름을 결정한다).

        Returns:
            컨테이너를 찾았고 정상이면 SandboxInfo, 아니면 None. runtime 확인이 실패한
            경우(예: 일시적 daemon 오류)도 None을 반환한다. 검증하지 못한 컨테이너를 흡수하면
            안 되고, create로 흘려보내면 일시적 오류에서 하드 실패 대신 acquire가 복구된다.
        """
        container_name = f"{self._container_prefix}-{sandbox_id}"

        try:
            running = self._is_container_running(container_name)
        except RuntimeError as e:
            logger.warning(f"Could not verify container {container_name} during discovery; not adopting it: {e}")
            return None

        if not running:
            return None

        port = self._get_container_port(container_name)
        if port is None:
            return None

        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")
        sandbox_url = f"http://{sandbox_host}:{port}"
        if not wait_for_sandbox_ready(sandbox_url, timeout=5):
            return None

        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_url=sandbox_url,
            container_name=container_name,
        )

    def list_running(self) -> list[SandboxInfo]:
        """설정된 prefix에 맞는 실행 중 컨테이너를 모두 열거한다.

        ``docker ps`` 한 번으로 컨테이너 이름을 받고, 배치 ``docker inspect`` 한 번으로 모든
        컨테이너의 생성 timestamp와 port 매핑을 한꺼번에 가져온다. 총 subprocess 호출은
        2번이다(컨테이너마다 따로 부르는 순진한 방식의 2N+1에서 줄였다).

        주의: Docker의 ``--filter name=``은 *부분 문자열* 매칭이므로, prefix가 정확히 일치하는
        컨테이너만 포함되도록 ``startswith`` 검사를 한 번 더 한다.

        port 매핑이 없는 컨테이너도 (빈 sandbox_url로) 포함한다. 그래야 startup reconciliation이
        port 상태와 무관하게 orphan을 흡수할 수 있다.
        """
        # 1단계: docker ps로 컨테이너 이름을 열거한다
        try:
            result = subprocess.run(
                [
                    self._runtime,
                    "ps",
                    "--filter",
                    f"name={self._container_prefix}-",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                logger.warning(
                    "Failed to list running containers with %s ps (returncode=%s, stderr=%s)",
                    self._runtime,
                    result.returncode,
                    stderr or "<empty>",
                )
                return []
            if not result.stdout.strip():
                return []
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to list running containers: {e}")
            return []

        # prefix가 정확히 일치하는 이름만 남긴다(docker filter는 부분 문자열 기반이다)
        container_names = [name.strip() for name in result.stdout.strip().splitlines() if name.strip().startswith(self._container_prefix + "-")]
        if not container_names:
            return []

        # 2단계: 배치 docker inspect — 모든 컨테이너를 subprocess 한 번으로 처리한다
        inspections = self._batch_inspect(container_names)

        infos: list[SandboxInfo] = []
        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")
        for container_name in container_names:
            data = inspections.get(container_name)
            if data is None:
                # ps와 inspect 사이에 컨테이너가 사라졌거나 inspect가 실패했다
                continue
            created_at, host_port = data
            sandbox_id = container_name[len(self._container_prefix) + 1 :]
            sandbox_url = f"http://{sandbox_host}:{host_port}" if host_port else ""

            infos.append(
                SandboxInfo(
                    sandbox_id=sandbox_id,
                    sandbox_url=sandbox_url,
                    container_name=container_name,
                    created_at=created_at,
                )
            )

        logger.info(f"Found {len(infos)} running sandbox container(s)")
        return infos

    def _batch_inspect(self, container_names: list[str]) -> dict[str, tuple[float, int | None]]:
        """subprocess 한 번으로 여러 컨테이너를 배치 inspect 한다.

        ``container_name -> (created_at, host_port)`` 매핑을 반환한다. 없는 컨테이너나 파싱
        실패는 결과에서 조용히 빠진다.
        """
        if not container_names:
            return {}
        try:
            result = subprocess.run(
                [self._runtime, "inspect", *container_names],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to batch-inspect containers: {e}")
            return {}

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.warning(
                "Failed to batch-inspect containers with %s inspect (returncode=%s, stderr=%s)",
                self._runtime,
                result.returncode,
                stderr or "<empty>",
            )
            return {}

        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse docker inspect output as JSON: {e}")
            return {}

        out: dict[str, tuple[float, int | None]] = {}
        for entry in payload:
            # docker inspect 응답에서 ``Name``은 앞에 ``/``가 붙어 온다
            name = (entry.get("Name") or "").lstrip("/")
            if not name:
                continue
            created_at = _parse_docker_timestamp(entry.get("Created", ""))
            host_port = _extract_host_port(entry, 8080)
            out[name] = (created_at, host_port)
        return out

    # ── 컨테이너 연산 ─────────────────────────────────────────────────────

    def _start_container(
        self,
        container_name: str,
        port: int,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
    ) -> str:
        """새 컨테이너를 띄운다.

        Args:
            container_name: 컨테이너 이름.
            port: 컨테이너 port 8080에 매핑할 host port.
            extra_mounts: 추가 volume mount.

        Returns:
            컨테이너 ID.

        Raises:
            RuntimeError: 컨테이너 기동에 실패한 경우.
        """
        cmd = [self._runtime, "run"]

        # Docker 전용 보안 옵션
        if self._runtime == "docker":
            cmd.extend(["--security-opt", "seccomp=unconfined"])

        if self._runtime == "docker":
            port_mapping = f"{_resolve_docker_bind_host()}:{port}:8080"
        else:
            port_mapping = f"{port}:8080"

        cmd.extend(
            [
                "--rm",
                "-d",
                "-p",
                port_mapping,
                "--name",
                container_name,
            ]
        )

        # 환경 변수
        for key, value in self._environment.items():
            cmd.extend(["-e", f"{key}={value}"])

        # config 레벨 volume mount
        for mount in self._config_mounts:
            cmd.extend(
                _format_container_mount(
                    self._runtime,
                    mount.host_path,
                    mount.container_path,
                    mount.read_only,
                )
            )

        # 추가 mount(thread 전용, skills 등)
        if extra_mounts:
            for host_path, container_path, read_only in extra_mounts:
                cmd.extend(
                    _format_container_mount(
                        self._runtime,
                        host_path,
                        container_path,
                        read_only,
                    )
                )

        cmd.append(self._image)

        log_cmd = _format_container_command_for_log(_redact_container_command_for_log(cmd))
        logger.info(f"Starting container using {self._runtime}: {log_cmd}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            logger.info(f"Started container {container_name} (ID: {container_id}) using {self._runtime}")
            return container_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container using {self._runtime}: {e.stderr}")
            raise RuntimeError(f"Failed to start sandbox container: {e.stderr}")

    def _stop_container(self, container_id: str) -> None:
        """컨테이너를 멈춘다(--rm 덕분에 자동 제거된다).

        timeout은 ownership 레이어와 무관하게 최악의 경우를 제한한다. teardown lease가 이 작업
        중에 peer의 재획득을 막지만, 그건 lease이므로 만료될 수 있다(TTL보다 긴 store 장애).
        그러면 먹통 daemon에 대한 무제한 ``docker stop``이 lease보다 오래 살아남아 peer가 쓰는
        컨테이너를 멈출 수 있다 — #4206. stop을 제한하면 store가 완전히 정상일 때조차 그 노출
        시간이 얼마나 길어질 수 있는지 상한이 생긴다.
        """
        try:
            subprocess.run(
                [self._runtime, "stop", container_id],
                capture_output=True,
                text=True,
                check=True,
                timeout=self._STOP_TIMEOUT_SECONDS,
            )
            logger.info(f"Stopped container {container_id} using {self._runtime}")
        except subprocess.TimeoutExpired:
            # CalledProcessError처럼 삼키지 않는다. 컨테이너가 아직 돌고 있을 수 있으므로
            # 호출자가 정상 종료로 보고하면 안 된다.
            logger.error(f"Timed out after {self._STOP_TIMEOUT_SECONDS}s stopping container {container_id} using {self._runtime}")
            raise
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to stop container {container_id}: {e.stderr}")

    def _is_container_running(self, container_name: str) -> bool:
        """이름으로 지정한 컨테이너가 지금 실행 중인지 확인한다.

        결정적 컨테이너 이름 덕분에 어떤 프로세스든 다른 프로세스가 띄운 컨테이너를 찾을 수 있어
        cross-process 탐색이 가능해진다.

        Raises:
            RuntimeError: 컨테이너 runtime이 inspect 질의에 답하지 못한 경우. 확인 실패는
                "컨테이너가 없다"는 확정적 결과와 의도적으로 구분한다. 그래야 일시적인
                Docker/Container daemon 오류 중에 호출자가 멀쩡한 컨테이너를 부수지 않는다.
        """
        try:
            result = subprocess.run(
                [self._runtime, "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Timed out checking container {container_name}") from exc

        if result.returncode == 0:
            return result.stdout.strip().lower() == "true"
        if _is_no_such_container_error(result.stderr, container_name):
            return False
        raise RuntimeError(f"Failed to inspect container {container_name}: {result.stderr.strip()}")

    def _get_container_port(self, container_name: str) -> int | None:
        """실행 중인 컨테이너의 host port를 얻는다.

        Args:
            container_name: inspect할 컨테이너 이름.

        Returns:
            컨테이너 port 8080에 매핑된 host port. 찾지 못하면 None.
        """
        try:
            result = subprocess.run(
                [self._runtime, "port", container_name, "8080"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # 출력 형식: "0.0.0.0:PORT" 또는 ":::PORT"
                port_str = result.stdout.strip().split(":")[-1]
                return int(port_str)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            pass
        return None
