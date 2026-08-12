"""cross-process 탐색과 상태 영속화를 위한 sandbox 메타데이터."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SandboxInfo:
    """cross-process 탐색을 가능하게 하는 영속 sandbox 메타데이터.

    다른 프로세스(gateway vs langgraph, 여러 worker, 저장소를 공유하는 K8s pod 등)에서
    기존 sandbox에 다시 연결하는 데 필요한 정보를 모두 담는다.
    """

    sandbox_id: str
    sandbox_url: str  # 예: http://localhost:8080 또는 http://k3s:30001
    container_name: str | None = None  # local container backend 전용
    container_id: str | None = None  # local container backend 전용
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_url": self.sandbox_url,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SandboxInfo:
        return cls(
            sandbox_id=data["sandbox_id"],
            sandbox_url=data.get("sandbox_url", data.get("base_url", "")),
            container_name=data.get("container_name"),
            container_id=data.get("container_id"),
            created_at=data.get("created_at", time.time()),
        )
