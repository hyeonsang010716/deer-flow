"""LangGraph 호환 runtime — run, streaming, lifecycle 관리.

:mod:`~deerflow.runtime.runs`와 :mod:`~deerflow.runtime.stream_bridge`의 공개 API를 다시
export해, 소비자가 ``deerflow.runtime``에서 바로 import할 수 있게 한다.
"""

from .checkpoint_state import CheckpointStateAccessor, build_state_mutation_graph
from .checkpointer import checkpointer_context, get_checkpointer, make_checkpointer, reset_checkpointer
from .runs import ORPHAN_RECOVERY_STOP_REASON, STARTUP_ORPHAN_RECOVERY_ERROR, CancelOutcome, ConflictError, DisconnectMode, RunContext, RunManager, RunRecord, RunStatus, ThreadOperationKind, UnsupportedStrategyError, run_agent
from .serialization import serialize, serialize_channel_values, serialize_channel_values_for_api, serialize_lc_object, serialize_messages_tuple, strip_data_url_image_blocks
from .store import get_store, make_store, reset_store, store_context

# NOTE: ``RedisStreamBridge``는 의도적으로 다시 export하지 않는다. ``redis``는 선택적 extra이며
# 여기서 import하면 모든 프로세스가 ``redis.asyncio``를 로드하게 된다. 필요하면
# ``deerflow.runtime.stream_bridge.redis``에서 import한다.
from .stream_bridge import END_SENTINEL, HEARTBEAT_SENTINEL, MemoryStreamBridge, StreamBridge, StreamEvent, StreamGap, StreamItem, make_stream_bridge

__all__ = [
    # checkpoint state
    "CheckpointStateAccessor",
    "build_state_mutation_graph",
    # checkpointer
    "checkpointer_context",
    "get_checkpointer",
    "make_checkpointer",
    "reset_checkpointer",
    # runs
    "CancelOutcome",
    "ConflictError",
    "DisconnectMode",
    "ORPHAN_RECOVERY_STOP_REASON",
    "RunContext",
    "RunManager",
    "RunRecord",
    "RunStatus",
    "ThreadOperationKind",
    "STARTUP_ORPHAN_RECOVERY_ERROR",
    "UnsupportedStrategyError",
    "run_agent",
    # serialization
    "serialize",
    "serialize_channel_values",
    "serialize_channel_values_for_api",
    "serialize_lc_object",
    "serialize_messages_tuple",
    "strip_data_url_image_blocks",
    # store
    "get_store",
    "make_store",
    "reset_store",
    "store_context",
    # stream_bridge
    "END_SENTINEL",
    "HEARTBEAT_SENTINEL",
    "MemoryStreamBridge",
    "StreamBridge",
    "StreamEvent",
    "StreamGap",
    "StreamItem",
    "make_stream_bridge",
]
