"""``runs.token_usage_by_model`` 컬럼을 추가한다.

Revision ID: 0002_runs_token_usage
Revises: 0001_baseline
Create Date: 2026-06-22

GitHub 이슈 #3682을 고친다. PR #3658의 커밋 e7a03e52 이전에 만들어진 DB에는 ``runs``에
``token_usage_by_model`` JSON 컬럼이 없다. 이 migration이 없으면 ``runs``를 ``SELECT``하는
모든 endpoint가 ``no such column: runs.token_usage_by_model``을 낸다.

``Base.metadata``와의 schema 일치
---------------------------------

ORM 모델은 이 컬럼을 ``Mapped[dict] = mapped_column(JSON, default=dict,
server_default=text("'{}'"))``로 선언한다. Optional이 아니므로 SQLAlchemy가
``nullable=False``로 추론한다. 따라서 ``Base.metadata.create_all``(빈 DB bootstrap 경로)은
새 DB에 ``token_usage_by_model JSON NOT NULL DEFAULT '{}'``를 만든다.

legacy DB를 업그레이드한 결과가 새 DB와 동일한 schema가 되도록, 이 migration도 같은
``nullable=False``와 ``server_default='{}'``로 컬럼을 추가한다. 이 server default 덕분에
데이터가 있는 테이블에서도 ``ALTER TABLE runs ADD COLUMN ... NOT NULL``이 성공한다.
기존 row는 ALTER 시점에 빈 객체 기본값을 받아 ``NOT NULL`` 위반이 나지 않는다.

멱등성
------

``safe_add_column``을 쓰므로 컬럼이 이미 있는 DB에 이 revision을 다시 실행해도 아무 일도
일어나지 않는다. 실제로 두 가지 경우를 커버한다:

1. 이슈에 있는 우회책을 수동으로 적용한 사용자
   (``ALTER TABLE runs ADD COLUMN token_usage_by_model JSON``).
2. 프로세스 간 lock이 어떤 이유로 우회되었을 때 여러 Gateway 인스턴스가 동시에 bootstrap
   하는 경우. ``bootstrap_schema``의 advisory lock / sentinel row mutex 위에 얹는
   defence-in-depth다.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from deerflow.persistence.migrations._helpers import safe_add_column, safe_drop_column

# Alembic이 사용하는 revision 식별자.
revision: str = "0002_runs_token_usage"
down_revision: str | Sequence[str] | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    safe_add_column(
        "runs",
        sa.Column(
            "token_usage_by_model",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    safe_drop_column("runs", "token_usage_by_model")
