"""커스텀 agent 정의 저장소 설정.

커스텀 agent의 *정의*(``config.yaml`` + ``SOUL.md``)를 어디에 보관할지 정한다.
run/thread/event 영속 계층을 담당하는 :class:`DatabaseConfig` 나 deermem memory store와는 별개다.

백엔드:
- file: ``{base_dir}/users/{user_id}/agents/{name}/`` 아래의 사용자별 파일
  (현재 레이아웃 그대로). 구조상 단일 노드 전용이라, 공유 마운트가 없으면 한 노드에서 만든
  agent가 다른 노드에는 보이지 않는다. 단일 노드와 무설정 개발 환경에 영향을 주지 않도록 기본값이다.
- db: 기존 SQL 영속 계층의 ``agents`` 테이블 row로 저장하며 모든 노드가 공유한다.
  ``database.backend`` 가 ``sqlite`` 또는 ``postgres`` 여야 한다(시작 시 검증한다. gateway
  ``deps`` 모듈 참고).

agent의 *memory*(``memory.json``)는 deermem 저장 계층이 담당하는 별개 관심사이며
이 스위치의 영향을 받지 않는다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentStorageConfig(BaseModel):
    backend: Literal["file", "db"] = Field(
        default="file",
        description=(
            "Storage backend for custom agent definitions (config.yaml + SOUL.md). "
            "'file' (default) keeps today's per-user on-disk layout — single-node only. "
            "'db' stores each agent as a row in the shared SQL persistence layer so a "
            "multi-instance deployment sees the same agents on every node; it requires "
            "database.backend to be 'sqlite' or 'postgres'."
        ),
    )
