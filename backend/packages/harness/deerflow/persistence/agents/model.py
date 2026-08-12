"""custom agent 정의를 위한 ORM model.

custom agent ``(user_id, name)``마다 한 행이다. ``config``는
:class:`~deerflow.config.agents_config.AgentConfig` 문서 전체에서 ``name``만 뺀 것을 담는다
(``name``은 자연 키이며 ``name`` 컬럼이 들고 있다). 필드마다 컬럼을 두지 않고 config를 JSON
문서 하나로 저장하는 것은 의도적이다. 이 코드베이스는 이미 ``preserve_non_managed_fields``로,
앞으로 ``AgentConfig``에 추가되는 필드는 그것을 모르는 writer를 거쳐도 그대로 왕복해야 한다고
선언하고 있다. 문서 컬럼은 schema 변경 없이 그 불변식을 지킨다(``AgentConfig``에 새 필드가
생겨도 여기서는 migration이 필요 없다). 쿼리는 ``(user_id, name)`` 조회와 user별 목록뿐이며,
정확히 index가 걸린 컬럼들이다.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class AgentRow(Base):
    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_agents_user_name"),)

    # 대리 primary key(uuid4 hex). 자연 키는 (user_id, name)이며 위의 UNIQUE 제약이 강제한다.
    # 대리 PK를 두면 나중에 agent 이름이 바뀌어도 행의 identity가 그대로 유지된다.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    # 소문자로 저장한다. 디스크 레이아웃(Paths.user_agent_dir이 소문자로 변환)과 일치시킨다.
    name: Mapped[str] = mapped_column(String(128))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    soul: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
