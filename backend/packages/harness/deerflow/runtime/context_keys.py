"""DeerFlow runtime 컴포넌트들이 공유하는 내부 runtime context key."""

from typing import Final

CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY: Final[str] = "__deerflow_pre_run_message_ids"
