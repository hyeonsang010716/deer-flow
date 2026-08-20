"""ORM model 등록 진입점.

이 모듈을 import하면 모든 ORM model이 ``Base.metadata``에 등록되어 Alembic autogenerate가 모든
테이블을 감지한다.

실제 ORM 클래스들은 엔티티별 하위 패키지로 옮겨졌다.
- ``deerflow.persistence.thread_meta``
- ``deerflow.persistence.run``
- ``deerflow.persistence.feedback``
- ``deerflow.persistence.user``

``RunEventRow``는 ``deerflow.persistence.models.run_event``에 남아 있다. 저장 구현이
``deerflow.runtime.events.store.db``에 있고 대응하는 엔티티 디렉터리가 없기 때문이다.
"""

from deerflow.persistence.agents.model import AgentRow
from deerflow.persistence.channel_connections.model import (
    ChannelConnectionRow,
    ChannelConversationRow,
    ChannelCredentialRow,
    ChannelOAuthStateRow,
)
from deerflow.persistence.feedback.model import FeedbackRow
from deerflow.persistence.models.run_event import RunEventRow
from deerflow.persistence.run.model import RunRow
from deerflow.persistence.scheduled_task_runs.model import ScheduledTaskRunRow
from deerflow.persistence.scheduled_tasks.model import ScheduledTaskRow
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.persistence.user.model import UserRow
from deerflow.persistence.webhook_delivery.model import WebhookDeliveryRow

__all__ = [
    "AgentRow",
    "ChannelConnectionRow",
    "ChannelConversationRow",
    "ChannelCredentialRow",
    "ChannelOAuthStateRow",
    "FeedbackRow",
    "RunEventRow",
    "RunRow",
    "ScheduledTaskRow",
    "ScheduledTaskRunRow",
    "ThreadMetaRow",
    "UserRow",
    "WebhookDeliveryRow",
]
