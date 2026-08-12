"""sandbox provisioning backend의 추상 베이스 클래스."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import httpx
import requests

from .sandbox_info import SandboxInfo

logger = logging.getLogger(__name__)


def sandbox_http_trust_env(sandbox_url: str) -> bool:
    """*sandbox_url*용 HTTP 클라이언트가 proxy 설정을 상속할지 여부를 판단한다.

    로컬 Docker, DooD, Kubernetes sandbox endpoint는 인터넷 트래픽이 아니라
    control-plane 연결이다. 이들을 ``HTTP_PROXY``로 보내면 sandbox 컨테이너가
    정상인데도 proxy가 만든 502가 나올 수 있다(#3441). 외부 FQDN 호스트는
    기존 환경 proxy 동작을 그대로 유지한다.
    """
    try:
        hostname = (urlparse(sandbox_url).hostname or "").rstrip(".").lower()
    except ValueError:
        return True
    if not hostname:
        return True
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".docker.internal") or hostname.endswith(".containers.internal"):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return "." in hostname
    return not (address.is_loopback or address.is_private or address.is_link_local)


def wait_for_sandbox_ready(sandbox_url: str, timeout: int = 30) -> bool:
    """sandbox health endpoint를 준비될 때까지, 또는 timeout까지 polling한다.

    Args:
        sandbox_url: sandbox URL (예: http://k3s:30001).
        timeout: 최대 대기 시간(초).

    Returns:
        sandbox가 준비되면 True, 아니면 False.
    """
    start_time = time.time()
    with requests.Session() as session:
        session.trust_env = sandbox_http_trust_env(sandbox_url)
        while time.time() - start_time < timeout:
            try:
                response = session.get(f"{sandbox_url}/v1/sandbox", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
    return False


async def wait_for_sandbox_ready_async(sandbox_url: str, timeout: int = 30, poll_interval: float = 1.0) -> bool:
    """sandbox 준비 상태 polling의 async 버전.

    async runtime 경로에서는 이 함수를 써야 sandbox 기동 대기가 event loop를
    막지 않는다. 동기 ``wait_for_sandbox_ready``는 기존 동기 backend/provider
    호출부를 위해 남겨 둔다.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async with httpx.AsyncClient(timeout=5, trust_env=sandbox_http_trust_env(sandbox_url)) as client:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                response = await client.get(f"{sandbox_url}/v1/sandbox", timeout=min(5.0, remaining))
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))
    return False


class SandboxBackend(ABC):
    """sandbox provisioning backend의 추상 베이스.

    구현체는 두 가지다.

    - LocalContainerBackend: 로컬에서 Docker/Apple Container를 띄우고 port를 관리한다.
    - RemoteSandboxBackend: 이미 존재하는 URL(K8s service, 외부)에 접속한다.
    """

    @abstractmethod
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
        """새 sandbox를 생성/provisioning한다.

        Args:
            thread_id: sandbox를 생성할 대상 thread ID. sandbox를 thread 단위로 정리하려는 backend에 유용하다.
            sandbox_id: 결정적으로 계산된 sandbox 식별자.
            extra_mounts: 추가 volume mount. (host_path, container_path, read_only) 튜플 목록이다.
                컨테이너를 관리하지 않는 backend(예: remote)는 무시한다.
            user_id: sandbox가 mount 또는 provisioning할 사용자 bucket.
            provision_lark_cli_runtime: backend의 고유 메커니즘(예: provisioner의
                init container + emptyDir)으로 sandbox lark-cli runtime을 provisioning하도록
                요청한다. 지원하지 않는 backend는 무시한다.
            provision_lark_cli_broker: 자격 증명이 sandbox 밖에 남도록 lark-cli broker
                sidecar(Pattern B, issue #4338) provisioning을 요청한다. backend가 지원하면
                ``provision_lark_cli_runtime``보다 우선하고, 지원하지 않으면 무시한다.

        Returns:
            연결 정보가 담긴 SandboxInfo.
        """
        ...

    @abstractmethod
    def destroy(self, info: SandboxInfo) -> None:
        """sandbox를 파괴/정리하고 자원을 반환한다.

        Args:
            info: 파괴할 sandbox 메타데이터.
        """
        ...

    @abstractmethod
    def is_alive(self, info: SandboxInfo) -> bool:
        """sandbox가 아직 살아 있는지 빠르게 확인한다.

        전체 health check가 아니라 가벼운 확인(예: container inspect)이어야 한다.

        Args:
            info: 확인할 sandbox 메타데이터.

        Returns:
            sandbox가 살아 있는 것으로 보이면 True.
        """
        ...

    @abstractmethod
    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """결정적 ID로 기존 sandbox를 찾아본다.

        cross-process 복구에 쓴다. 다른 프로세스가 띄운 sandbox를 이 프로세스가
        결정적 컨테이너 이름이나 URL로 찾아낼 수 있다.

        Args:
            sandbox_id: 찾을 결정적 sandbox ID.

        Returns:
            찾았고 정상이면 SandboxInfo, 아니면 None.
        """
        ...

    def list_running(self) -> list[SandboxInfo]:
        """이 backend가 관리하는 실행 중인 sandbox를 모두 열거한다.

        startup reconciliation에 쓴다. 프로세스가 재시작되면 이전 프로세스가 띄운
        컨테이너를 찾아 warm pool로 흡수하거나, 너무 오래 idle이면 파괴해야 한다.

        기본 구현은 빈 리스트를 반환한다. 로컬 컨테이너를 관리하지 않는 backend에는
        이것이 맞다(예: RemoteSandboxBackend는 lifecycle을 provisioner에 위임하고,
        provisioner가 자체적으로 정리한다).

        Returns:
            현재 실행 중인 모든 sandbox의 SandboxInfo 목록.
        """
        return []
