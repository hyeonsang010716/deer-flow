from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SandboxOwnershipType = Literal["memory", "redis"]
SandboxOverflowPolicy = Literal["wait", "reject", "burst"]


class SandboxOwnershipConfig(BaseModel):
    """인스턴스 간 sandbox 컨테이너 ownership 설정(#4206).

    Gateway 인스턴스들은 sandbox 컨테이너를 공유하지만 warm pool은 각자 메모리에 따로 갖는다.
    공유 ownership 상태가 없으면 한 인스턴스의 reconciliation이 다른 인스턴스가 쓰고 있는
    컨테이너를 입양했다가 나중에 idle로 판단해 파괴한다. 이 설정은 그 ownership 상태를 어디에
    둘지 고른다.
    """

    type: SandboxOwnershipType = Field(
        default="memory",
        description=(
            "Sandbox ownership store backend. 'memory' keeps ownership in-process (single-instance deployments only, where cross-instance adoption cannot occur). "
            "'redis' shares ownership across gateway instances and is required for load-balanced / multi-worker deployments that share a container backend."
        ),
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for the redis ownership type. If omitted, DEER_FLOW_SANDBOX_OWNERSHIP_REDIS_URL, DEER_FLOW_STREAM_BRIDGE_REDIS_URL, REDIS_URL, or redis://localhost:6379/0 is used.",
    )
    renewal_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "How often an owning instance refreshes its leases. The lease TTL is derived from this (interval x ttl_multiplier), so ownership liveness is independent of sandbox.idle_timeout: "
            "renewal keeps running even when idle cleanup is disabled (idle_timeout: 0)."
        ),
    )
    ttl_multiplier: float = Field(
        default=4.0,
        ge=2,
        description="Lease TTL as a multiple of renewal_interval_seconds. At least 2, so a single missed renewal (slow host, brief Redis blip) cannot expire a live owner's lease. Default 4 tolerates three consecutive misses.",
    )
    key_prefix: str = Field(
        default="deerflow:sandbox:owner",
        description="Redis key prefix for ownership leases. Only applies to the redis ownership type.",
    )


class VolumeMountConfig(BaseModel):
    """volume mount 하나에 대한 설정."""

    host_path: str = Field(
        ...,
        description=(
            "Source path for the mount. Resolution depends on the active provider: "
            "``LocalSandboxProvider`` checks this path from the gateway process — in "
            "``make dev`` that is the host machine, but in Docker deployments "
            "(``make up`` / docker-compose) it is the path *inside* the "
            "``deer-flow-gateway`` container, so the host directory must also be "
            "bind-mounted into the gateway service for the mount to take effect. "
            "``AioSandboxProvider`` (DooD) passes this value straight to ``docker -v`` "
            "for the sandbox container, where it is resolved by the host Docker daemon "
            "from the host machine's perspective."
        ),
    )
    container_path: str = Field(..., description="Path inside the container")
    read_only: bool = Field(default=False, description="Whether the mount is read-only")


class SandboxConfig(BaseModel):
    """sandbox 설정 섹션.

    공통 옵션:
        use: sandbox provider의 클래스 경로(필수).
        allow_host_bash: LocalSandboxProvider에서 host 쪽 bash 실행을 허용한다.
            위험하므로 완전히 신뢰할 수 있는 로컬 workflow에서만 쓴다.

    AioSandboxProvider, BoxliteProvider, E2BSandboxProvider 공통 옵션:
        image: 사용할 sandbox 이미지(Docker/AIO 이미지 또는 BoxLite OCI 이미지).
        replicas: provider 용량(양수). E2B는 ownership이 Redis일 때 Gateway worker 간에
            공유하고, 그 밖의 모드/provider는 프로세스 로컬로 집계한다.
        idle_timeout: 반납된 warm sandbox/VM을 중지하기까지의 유휴 시간(초, 기본값 600 = 10분).
            0이면 비활성화한다.
        environment: sandbox에 주입할 환경변수($로 시작하는 값은 host 환경변수에서 해석).

    BoxliteProvider 전용 옵션:
        health_check_skip_seconds: 최근 반납된 warm VM에 대해 reclaim 시 health check를
            건너뛰는 구간(초). 기본값 0.0은 재사용 전 항상 검증한다는 뜻이다.

    AioSandboxProvider 전용 옵션:
        port: sandbox 컨테이너 기본 포트(기본값 8080).
        container_prefix: 컨테이너 이름 prefix(기본값 deer-flow-sandbox).
        mounts: 컨테이너와 디렉터리를 공유할 volume mount 목록.
        thread_data_mounts: thread 데이터가 공유 mount로 이미 sandbox에 보이는지를 강제로
            지정한다. 생략하면 backend에서 자동 감지한다.

    AioSandboxProvider와 E2BSandboxProvider 공통 옵션:
        ownership: 인스턴스 간 sandbox ownership 저장소(memory | redis). sandbox backend를
            공유하는 다중 인스턴스 배포에는 redis가 필요하다. SandboxOwnershipConfig 참고.
    """

    use: str = Field(
        ...,
        description="Class path of the sandbox provider (e.g. deerflow.sandbox.local:LocalSandboxProvider)",
    )
    allow_host_bash: bool = Field(
        default=False,
        description="Allow the bash tool to execute directly on the host when using LocalSandboxProvider. Dangerous; intended only for fully trusted local environments.",
    )
    image: str | None = Field(
        default=None,
        description="Sandbox image to use (Docker/AIO image or BoxLite OCI image)",
    )
    port: int | None = Field(
        default=None,
        description="Base port for sandbox containers",
    )
    replicas: int | None = Field(
        default=None,
        gt=0,
        description=("Positive provider capacity. E2B enforces it deployment-wide when sandbox ownership uses Redis; otherwise accounting is per Gateway process. Each provider defines which lifecycle states count."),
    )
    overflow_policy: SandboxOverflowPolicy = Field(
        default="wait",
        description="E2B capacity policy. Use wait, reject, or burst.",
    )
    acquire_timeout: int = Field(
        default=30,
        gt=0,
        description="Seconds that E2B wait policy waits for capacity.",
    )
    burst_limit: int = Field(
        default=0,
        ge=0,
        description="Extra E2B capacity slots when overflow_policy is burst.",
    )
    container_prefix: str | None = Field(
        default=None,
        description="Prefix for container names",
    )
    idle_timeout: int | None = Field(
        default=None,
        description="Idle timeout in seconds before released warm sandboxes/VMs are stopped (default: 600 = 10 minutes). Set to 0 to disable.",
    )
    health_check_skip_seconds: float | None = Field(
        default=None,
        ge=0,
        description="BoxLite-only reclaim skip window in seconds for boxes recently released by this provider instance. Set to 0 to always validate before warm reuse.",
    )
    ownership: SandboxOwnershipConfig | None = Field(
        default=None,
        description=(
            "AioSandboxProvider/E2BSandboxProvider: where cross-instance sandbox ownership is tracked (#4206, #4341). Omitted = memory (single-instance). "
            "Multi-worker / load-balanced gateways sharing one sandbox backend must set type: redis, or peers can adopt and destroy each other's live sandboxes."
        ),
    )
    mounts: list[VolumeMountConfig] = Field(
        default_factory=list,
        description="List of volume mounts to share directories between host and container",
    )
    thread_data_mounts: bool | None = Field(
        default=None,
        description=("AioSandboxProvider: override whether /mnt/user-data is already visible through shared mounts. Omitted uses backend auto-detection; true skips explicit upload synchronization; false forces it."),
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to inject into the sandbox container. Values starting with $ will be resolved from host environment variables.",
    )

    bash_output_max_chars: int = Field(
        default=20000,
        ge=0,
        description="Maximum characters to keep from bash tool output. Output exceeding this limit is middle-truncated (head + tail), preserving the first and last half. Set to 0 to disable truncation.",
    )
    read_file_output_max_chars: int = Field(
        default=50000,
        ge=0,
        description="Maximum characters to keep from read_file tool output. Output exceeding this limit is head-truncated. Set to 0 to disable truncation.",
    )
    ls_output_max_chars: int = Field(
        default=20000,
        ge=0,
        description="Maximum characters to keep from ls tool output. Output exceeding this limit is head-truncated. Set to 0 to disable truncation.",
    )
    bash_command_timeout: int = Field(
        default=600,
        gt=0,
        description=(
            "Maximum wall-clock seconds a host bash command may run before it is terminated, process group and all (LocalSandboxProvider). "
            "Keeps a blocking foreground command (e.g. an un-backgrounded server) from hanging the turn; background `&` processes return immediately."
        ),
    )

    provisioner_api_key: str | None = Field(
        default=None,
        description=(
            "API key sent as X-API-Key header to the provisioner service. "
            "Must match PROVISIONER_API_KEY on the provisioner container. "
            "Both sides must be set to the same value; "
            "the provisioner rejects all /api/* requests when the key is unset or mismatched."
        ),
    )

    model_config = ConfigDict(extra="allow")
