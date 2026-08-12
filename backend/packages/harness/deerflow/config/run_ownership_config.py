"""multi-worker 배포용 run ownership 설정."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunOwnershipConfig(BaseModel):
    """run 단위 ownership과 lease 설정.

    ``heartbeat_enabled``가 True면 각 worker가 자신이 실행 중인 run의 lease를 주기적으로
    갱신한다. multi-worker 배포에서 죽은 worker의 고아 run을 감지하려면 필요하다.

    시계 동기화 가정
    ---------------------
    reconciliation은 다른 worker의 UTC ``lease_expires_at``을 이 worker의
    ``datetime.now(UTC)``와 비교한다. 두 worker 시계 사이에 허용되는 오차는
    ``grace_seconds``뿐이다(여기에 현재 주기에 남은 heartbeat 여유, 최대
    ``lease_seconds / 3``이 더해진다). 최악의 경우 소유 worker의 heartbeat가 막 발동하려는
    시점에, 시계가 ``grace_seconds``보다 앞선 peer가 살아 있는 run을 고아로 잘못 회수할 수 있다.

    운영자는 worker 시계를 몇 초 이내로 동기화해야 한다(K8s 노드의 NTP/chrony/
    systemd-timesyncd). 환경상 보장할 수 없으면 ``grace_seconds``를 올린다. 대가는 실제로
    죽은 worker의 복구 지연이 길어지는 것이다(마지막 heartbeat부터 회수까지
    ``lease_seconds + grace_seconds``).
    """

    lease_seconds: int = Field(
        default=30,
        ge=5,
        description="Seconds before a run lease expires if not renewed. Heartbeat renews every lease_seconds / 3.",
    )
    grace_seconds: int = Field(
        default=10,
        ge=0,
        description=(
            "Extra seconds past lease expiry before an orphaned run is reclaimed. Also the clock-skew budget between workers — raise it if worker clocks are not tightly synced; cost is slower recovery of genuinely dead-worker runs."
        ),
    )
    heartbeat_enabled: bool = Field(
        default=False,
        description="When True, the worker periodically renews leases on its active runs. Enable for multi-worker deployments (GATEWAY_WORKERS > 1).",
    )
