"""inbound webhook dedupe 저장소 설정.

ChannelManager의 inbound dedupe state가 어디에 사는지를 결정한다. issue #4120(cross-pod
webhook dedupe) 참고. 기본값 ``auto``는 database.backend='postgres'일 때 Postgres
애플리케이션 DB를 재사용하고, 그 외에는 in-process memory store를 쓴다. ``memory``는 pod
단위라 replica 간에 공유되지 않고, ``postgres``는 pod 간에 state를 공유한다.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from deerflow.config.reload_boundary import format_field_description


class DedupeStorageBackend(StrEnum):
    AUTO = "auto"
    MEMORY = "memory"
    POSTGRES = "postgres"


class DedupeStorageConfig(BaseModel):
    """inbound webhook dedupe state가 저장되는 위치."""

    backend: DedupeStorageBackend = Field(
        default=DedupeStorageBackend.AUTO,
        description=format_field_description(
            "dedupe_storage",
            field_doc=(
                "Storage backend for inbound webhook dedupe state. "
                "'auto' uses the Postgres application database whenever database.backend='postgres', "
                "otherwise an in-process memory store (single-pod). "
                "'memory' forces the in-process store (per-pod; not shared across replicas). "
                "'postgres' shares dedupe state across pods via the application database. "
                "See issue #4120."
            ),
        ),
    )
