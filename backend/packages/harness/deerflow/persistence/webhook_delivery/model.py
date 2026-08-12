"""공유 inbound webhook 중복 제거 테이블의 ORM model(issue #4120).

각 행은 특정 inbound webhook(``_inbound_dedupe_key`` 4튜플로 식별)이 이미 dispatch되었음을
기록한다. 그래서 다른 gateway pod로 라우팅된 재전송도 중복으로 버려진다. 행은 지연 정리로
만료되며(``PostgresInboundDedupeStore`` 참고), ``first_seen`` 컬럼을 기준으로
``INBOUND_DEDUPE_TTL_SECONDS``보다 오래된 행을 삭제한다.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, PrimaryKeyConstraint, String, func
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class WebhookDeliveryRow(Base):
    __tablename__ = "webhook_deliveries"

    # 복합 primary key는 ChannelManager._inbound_dedupe_key와 정확히 같다:
    # (channel, workspace_id, chat_id, message_id). 네 컬럼을 그대로 쓰면 문자열을 이어 붙인
    # 대리 키가 필요 없다. 구성 요소에 Postgres TEXT 컬럼 하나에 담을 수 없는 문자(예: NUL)가
    # 들어갈 수 있어 중요하며, ON CONFLICT 대상도 자연스럽게 유지된다.
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(512), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(512), nullable=False)
    message_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("channel", "workspace_id", "chat_id", "message_id", name="pk_webhook_deliveries"),
        Index("ix_webhook_deliveries_first_seen", "first_seen"),
    )
