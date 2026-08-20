"""사용되지 않던 durable long-running MCP task 테이블을 제거한다.

Revision ID: 0012_drop_mcp_tasks
Revises: 0011_mcp_tasks
Create Date: 2026-08-20

``mcp_tasks`` 런타임(driver protocol, repository, poller)은 구체적인 ``McpTaskDriver``
구현이 한 번도 등록되지 않았고 ``McpTaskService.submit``을 호출하는 프로덕션 경로도 없어서,
행이 생길 수 없는 상태로 유지되고 있었다. 코드와 함께 테이블도 제거한다.

``0011_mcp_tasks``는 체인에 그대로 남겨둔다. 그 사이 ``dev``를 추적한 DB는 이미
``0011_mcp_tasks``로 stamp되어 있을 수 있고, revision 파일을 지우면 alembic이 그 stamp를
해석하지 못해 ``upgrade head``가 죽는다. 새 DB는 0011이 만들고 0012가 바로 지우는 형태가
되지만, 그 편이 기존 배포를 깨뜨리는 것보다 낫다.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_drop_mcp_tasks"
down_revision: str | Sequence[str] | None = "0011_mcp_tasks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("mcp_tasks"):
        return
    op.drop_table("mcp_tasks")


def downgrade() -> None:
    # 0011의 upgrade가 테이블 정의를 소유한다. 되돌리려면 0011까지 내려간 뒤 다시 올린다.
    raise NotImplementedError("Downgrade past 0012 is not supported; restore from a backup instead.")
