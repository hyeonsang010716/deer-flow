"""Remote sandbox backend — Pod lifecycle을 provisioner 서비스에 위임한다.

provisioner는 k3s에 sandbox id별 Pod와 NodePort Service를 동적으로 만든다. backend는
``k3s:{NodePort}``로 sandbox pod에 직접 접근한다.

구조:
    ┌────────────┐  HTTP   ┌─────────────┐  K8s API  ┌──────────┐
    │ this file  │ ──────▸ │ provisioner │ ────────▸ │   k3s    │
    │ (backend)  │         │ :8002       │           │ :6443    │
    └────────────┘         └─────────────┘           └─────┬────┘
                                                           │ creates
                           ┌─────────────┐           ┌─────▼──────┐
                           │   backend   │ ────────▸ │  sandbox   │
                           │             │  direct   │  Pod(s)    │
                           └─────────────┘ k3s:NPort └────────────┘
"""

from __future__ import annotations

import logging

import requests

from deerflow.runtime.user_context import get_effective_user_id
from deerflow.skills.storage import user_should_see_legacy_skills

from .backend import SandboxBackend
from .sandbox_info import SandboxInfo

logger = logging.getLogger(__name__)

_PROVISIONER_EXTRA_MOUNT_PATHS = {
    "/mnt/acp-workspace",
    "/mnt/skills/custom",
    "/mnt/skills/integrations",
    "/mnt/integrations/lark-cli/config",
    "/mnt/integrations/lark-cli/config/locks",
    "/mnt/integrations/lark-cli/data",
    "/mnt/integrations/lark-cli/runtime",
}

_LARK_CLI_RUNTIME_CONTAINER_PATH = "/mnt/integrations/lark-cli/runtime"
_LARK_CLI_CONFIG_CONTAINER_PATH = "/mnt/integrations/lark-cli/config"
_LARK_CLI_DATA_CONTAINER_PATH = "/mnt/integrations/lark-cli/data"


def _provisioner_extra_mounts_payload(
    extra_mounts: list[tuple[str, str, bool]] | None,
    *,
    provision_lark_cli_runtime: bool = False,
    provision_lark_cli_broker: bool = False,
) -> list[dict[str, object]]:
    """provisioner가 안전하게 재생성할 수 있는 extra mount만 반환한다.

    ``provision_lark_cli_runtime``이 설정되면 provisioner가 init container + emptyDir로
    lark-cli runtime을 공급하므로, 같은 경로에 hostPath/PVC mount가 충돌하지 않도록 여기서
    runtime extra mount를 뺀다. 사용자별 config/locks/data mount는 그대로 전달한다
    (Pattern A에서는 sandbox에 mount된다). config 루트는 read-only로 두고, 중첩된 locks
    mount만 lark-cli 조정 파일을 위해 쓰기 가능하게 둔다.

    ``provision_lark_cli_broker``가 설정되면(Pattern B, issue #4338) provisioner가 자격
    증명을 보유하는 broker sidecar를 띄우므로, config/locks/data mount는 **전달**하고
    (provisioner가 sandbox가 아니라 sidecar에 연결한다) runtime mount는 뺀다. 이 payload에서
    달라지는 건 provisioner가 배치할 수 있도록 해당 자격 증명 mount를 남겨 두는 것뿐이며,
    runtime 항목은 두 모드 모두에서 제외된다.
    """
    if not extra_mounts:
        return []

    drop_runtime = provision_lark_cli_runtime or provision_lark_cli_broker

    payload: list[dict[str, object]] = []
    for host_path, container_path, read_only in extra_mounts:
        if container_path not in _PROVISIONER_EXTRA_MOUNT_PATHS:
            continue
        if drop_runtime and container_path == _LARK_CLI_RUNTIME_CONTAINER_PATH:
            continue
        payload.append(
            {
                "host_path": host_path,
                "container_path": container_path,
                "read_only": read_only,
            }
        )
    return payload


class RemoteSandboxBackend(SandboxBackend):
    """sandbox lifecycle을 provisioner 서비스에 위임하는 backend.

    Pod 생성, 파괴, 탐색은 모두 provisioner가 처리한다. 이 backend는 얇은 HTTP 클라이언트다.

    전형적인 config.yaml::

        sandbox:
          use: deerflow.community.aio_sandbox:AioSandboxProvider
          provisioner_url: http://provisioner:8002
          provisioner_api_key: $PROVISIONER_API_KEY
    """

    def __init__(self, provisioner_url: str, api_key: str = ""):
        """provisioner 서비스 URL과 선택적 API key로 초기화한다.

        Args:
            provisioner_url: provisioner 서비스 URL
                             (예: ``http://provisioner:8002``).
            api_key: 매 요청에 ``X-API-Key`` 헤더로 보낼 값.
                     비워 두면 인증 헤더를 보내지 않는다.
        """
        self._provisioner_url = provisioner_url.rstrip("/")
        self._api_key = api_key

    @property
    def provisioner_url(self) -> str:
        return self._provisioner_url

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key} if self._api_key else {}

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
        """provisioner를 통해 sandbox Pod와 Service를 생성한다.

        ``POST /api/sandboxes``를 호출하며, 이 API가 k3s에 전용 Pod와 NodePort Service를
        만든다.
        """
        return self._provisioner_create(
            thread_id,
            sandbox_id,
            extra_mounts,
            user_id=user_id,
            provision_lark_cli_runtime=provision_lark_cli_runtime,
            provision_lark_cli_broker=provision_lark_cli_broker,
        )

    def destroy(self, info: SandboxInfo) -> None:
        """provisioner를 통해 sandbox Pod와 Service를 파괴한다."""
        self._provisioner_destroy(info.sandbox_id)

    def is_alive(self, info: SandboxInfo) -> bool:
        """sandbox Pod가 실행 중인지 확인한다."""
        return self._provisioner_is_alive(info.sandbox_id)

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """provisioner를 통해 기존 sandbox를 찾는다.

        ``GET /api/sandboxes/{sandbox_id}``를 호출하고, Pod가 있으면 정보를 반환한다.
        """
        return self._provisioner_discover(sandbox_id)

    def list_running(self) -> list[SandboxInfo]:
        """provisioner가 현재 관리하는 모든 sandbox를 반환한다.

        ``GET /api/sandboxes``를 호출해 ``AioSandboxProvider._reconcile_orphans()``가
        이전 프로세스가 만들고 명시적으로 파괴되지 않은 pod를 흡수할 수 있게 한다.
        이게 없으면 프로세스 재시작 시 기존 k8s Pod가 전부 조용히 orphan이 된다. idle
        checker는 in-process 상태만 추적하므로 그 pod들은 영원히 남는다.
        """
        return self._provisioner_list()

    # ── Provisioner API 호출 ─────────────────────────────────────────────

    def _provisioner_list(self) -> list[SandboxInfo]:
        """GET /api/sandboxes → 실행 중인 sandbox를 모두 나열한다."""
        try:
            resp = requests.get(f"{self._provisioner_url}/api/sandboxes", headers=self._auth_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("Provisioner list_running returned non-dict payload: %r", type(data))
                return []

            sandboxes = data.get("sandboxes", [])
            if not isinstance(sandboxes, list):
                logger.warning("Provisioner list_running returned non-list sandboxes: %r", type(sandboxes))
                return []

            infos: list[SandboxInfo] = []
            for sandbox in sandboxes:
                if not isinstance(sandbox, dict):
                    logger.warning("Provisioner list_running entry is not a dict: %r", type(sandbox))
                    continue

                sandbox_id = sandbox.get("sandbox_id")
                sandbox_url = sandbox.get("sandbox_url")
                if isinstance(sandbox_id, str) and sandbox_id and isinstance(sandbox_url, str) and sandbox_url:
                    infos.append(SandboxInfo(sandbox_id=sandbox_id, sandbox_url=sandbox_url))

            logger.info("Provisioner list_running: %d sandbox(es) found", len(infos))
            return infos
        except requests.RequestException as exc:
            logger.warning("Provisioner list_running failed: %s", exc)
            return []

    def _provisioner_create(
        self,
        thread_id: str | None,
        sandbox_id: str,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
        *,
        user_id: str | None = None,
        provision_lark_cli_runtime: bool = False,
        provision_lark_cli_broker: bool = False,
    ) -> SandboxInfo:
        """POST /api/sandboxes → Pod와 Service를 생성한다."""
        effective_user_id = user_id or get_effective_user_id()
        include_legacy_skills = user_should_see_legacy_skills(effective_user_id)
        payload = {
            "sandbox_id": sandbox_id,
            "thread_id": thread_id,
            "user_id": effective_user_id,
            "include_legacy_skills": include_legacy_skills,
            "provision_lark_cli_runtime": provision_lark_cli_runtime,
            "provision_lark_cli_broker": provision_lark_cli_broker,
        }
        provisioner_extra_mounts = _provisioner_extra_mounts_payload(
            extra_mounts,
            provision_lark_cli_runtime=provision_lark_cli_runtime,
            provision_lark_cli_broker=provision_lark_cli_broker,
        )
        if provisioner_extra_mounts:
            payload["extra_mounts"] = provisioner_extra_mounts
        try:
            resp = requests.post(
                f"{self._provisioner_url}/api/sandboxes",
                json=payload,
                headers=self._auth_headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Provisioner created sandbox {sandbox_id}: sandbox_url={data['sandbox_url']}")
            return SandboxInfo(
                sandbox_id=sandbox_id,
                sandbox_url=data["sandbox_url"],
            )
        except requests.RequestException as exc:
            logger.error(f"Provisioner create failed for {sandbox_id}: {exc}")
            raise RuntimeError(f"Provisioner create failed: {exc}") from exc

    def _provisioner_destroy(self, sandbox_id: str) -> None:
        """DELETE /api/sandboxes/{sandbox_id} → Pod와 Service를 파괴한다."""
        try:
            resp = requests.delete(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                headers=self._auth_headers(),
                timeout=15,
            )
            if resp.ok:
                logger.info(f"Provisioner destroyed sandbox {sandbox_id}")
            else:
                logger.warning(f"Provisioner destroy returned {resp.status_code}: {resp.text}")
        except requests.RequestException as exc:
            logger.warning(f"Provisioner destroy failed for {sandbox_id}: {exc}")

    def _provisioner_is_alive(self, sandbox_id: str) -> bool:
        """GET /api/sandboxes/{sandbox_id} → Pod phase를 확인한다."""
        try:
            resp = requests.get(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                headers=self._auth_headers(),
                timeout=10,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Provisioner health check failed for {sandbox_id}: {exc}") from exc

        if resp.status_code == 404:
            return False
        if not resp.ok:
            raise RuntimeError(f"Provisioner health check failed for {sandbox_id}: HTTP {resp.status_code} {resp.text}")

        data = resp.json()
        return data.get("status") == "Running"

    def _provisioner_discover(self, sandbox_id: str) -> SandboxInfo | None:
        """GET /api/sandboxes/{sandbox_id} → 기존 sandbox를 찾는다."""
        try:
            resp = requests.get(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                headers=self._auth_headers(),
                timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return SandboxInfo(
                sandbox_id=sandbox_id,
                sandbox_url=data["sandbox_url"],
            )
        except requests.RequestException as exc:
            logger.debug(f"Provisioner discover failed for {sandbox_id}: {exc}")
            return None
