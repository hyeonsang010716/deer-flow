"""run event 저장 설정.

run event(메시지 + 실행 trace)를 어디에 보존할지 결정한다.

Backend:
- memory: 메모리 저장. 재시작하면 사라진다. 개발과 테스트용.
- db: SQLAlchemy ORM 기반 SQL 데이터베이스. 완전한 조회가 가능하며 운영 배포용.
- jsonl: append-only JSONL 파일. 데이터베이스 없이 보존이 필요한 단일 노드 배포용 경량 대안.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunEventsConfig(BaseModel):
    backend: Literal["memory", "db", "jsonl"] = Field(
        default="memory",
        description="Storage backend for run events. 'memory' for development (no persistence), 'db' for production (SQL queries), 'jsonl' for lightweight single-node persistence.",
    )
    max_trace_content: int = Field(
        default=10240,
        description="Maximum trace content size in bytes before truncation (db backend only).",
    )
    track_token_usage: bool = Field(
        default=True,
        description="Whether RunJournal should accumulate token counts to RunRow.",
    )
