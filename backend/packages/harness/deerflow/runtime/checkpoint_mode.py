"""이중 mode checkpoint channel 안전장치: mode 고정, metadata marker, fail-closed gate.

Checkpointer 저장은 ``full`` mode(전체 snapshot channel 값) 또는 ``delta`` mode(LangGraph
``DeltaChannel``: sentinel blob + step별 write)로 동작한다. mode는 agent 빌드 시점에 프로세스
단위로 고정되고, write마다 각 checkpoint의 metadata에 찍히며, 모든 state 접근 전에 강제된다.
full mode 프로세스가 delta thread를 열면 조용히 빈 state를 materialize하지 않고
:class:`CheckpointModeMismatchError`를 던진다. delta mode 프로세스는 기존 full checkpoint를
그대로 읽을 수 있으므로, full -> delta가 지원되는 migration 경로다.
"""

from __future__ import annotations

from typing import Any

from deerflow.config.database_config import DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY, CheckpointChannelMode

INTERNAL_CHECKPOINT_MODE_KEY = "__deerflow_checkpoint_channel_mode"
CHECKPOINT_MODE_METADATA_KEY = "deerflow_checkpoint_channel_mode"


class CheckpointModeMismatchError(RuntimeError):
    """full mode graph가 Delta checkpoint를 읽기 직전에 발생한다."""


class CheckpointModeReconfigurationError(RuntimeError):
    """프로세스가 persistence mode를 실행 중에 바꾸려 할 때 발생한다."""


_frozen_checkpoint_channel_mode: CheckpointChannelMode | None = None
_frozen_checkpoint_snapshot_frequency: int | None = None


def frozen_checkpoint_channel_mode() -> CheckpointChannelMode | None:
    """이미 고정되어 있다면 프로세스에 고정된 checkpoint channel mode를 반환한다."""
    return _frozen_checkpoint_channel_mode


def freeze_checkpoint_channel_mode(mode: CheckpointChannelMode) -> CheckpointChannelMode:
    global _frozen_checkpoint_channel_mode
    if _frozen_checkpoint_channel_mode is None:
        _frozen_checkpoint_channel_mode = mode
    elif _frozen_checkpoint_channel_mode != mode:
        raise CheckpointModeReconfigurationError("checkpoint_channel_mode is restart-required and cannot change in a running process")
    return _frozen_checkpoint_channel_mode


def frozen_checkpoint_snapshot_frequency() -> int | None:
    """이미 고정되어 있다면 프로세스에 고정된 delta snapshot 주기를 반환한다."""
    return _frozen_checkpoint_snapshot_frequency


def freeze_checkpoint_snapshot_frequency(snapshot_frequency: int) -> int:
    """channel mode와 함께 delta snapshot 주기를 고정한다.

    이 주기는 각 graph의 channel table에 컴파일되어 들어가므로, mode와 마찬가지로 재시작이
    필요하며 하나의 checkpoint database를 공유하는 모든 프로세스에서 일치해야 한다. checkpoint
    metadata에는 의도적으로 찍지 않는다. mode marker 계약(없으면 full)과 full -> delta
    migration 의미는 주기 값에 영향받지 않기 때문이다.
    """
    global _frozen_checkpoint_snapshot_frequency
    if snapshot_frequency <= 0:
        raise ValueError("snapshot frequency must be positive")
    if _frozen_checkpoint_snapshot_frequency is None:
        _frozen_checkpoint_snapshot_frequency = snapshot_frequency
    elif _frozen_checkpoint_snapshot_frequency != snapshot_frequency:
        raise CheckpointModeReconfigurationError("checkpoint_delta.snapshot_frequency is restart-required and cannot change in a running process")
    return _frozen_checkpoint_snapshot_frequency


def resolve_checkpoint_snapshot_frequency(snapshot_frequency: int | None = None) -> int:
    """유효 snapshot 주기를 결정한다. 명시 값, 없으면 프로세스에 고정된 값,
    그것도 없으면 config 기본값 순이다."""
    if snapshot_frequency is not None:
        return snapshot_frequency
    frozen = _frozen_checkpoint_snapshot_frequency
    return frozen if frozen is not None else DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY


def inject_checkpoint_mode(config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    configurable = config.setdefault("configurable", {})
    configurable[INTERNAL_CHECKPOINT_MODE_KEY] = mode
    metadata = config.setdefault("metadata", {})
    if mode == "delta":
        metadata[CHECKPOINT_MODE_METADATA_KEY] = "delta"
    else:
        metadata.pop(CHECKPOINT_MODE_METADATA_KEY, None)


def checkpoint_metadata_uses_delta(metadata: Any) -> bool:
    """checkpoint metadata가 delta mode marker를 담고 있는지 여부."""
    if not metadata:
        return False
    if metadata.get(CHECKPOINT_MODE_METADATA_KEY) == "delta":
        return True
    counters = metadata.get("counters_since_delta_snapshot")
    return isinstance(counters, dict) and "messages" in counters


def checkpoint_tuple_uses_delta(checkpoint_tuple: Any) -> bool:
    if checkpoint_tuple is None:
        return False
    return checkpoint_metadata_uses_delta(getattr(checkpoint_tuple, "metadata", {}) or {})


def state_snapshot_uses_delta(snapshot: Any) -> bool:
    """materialize된 ``StateSnapshot``이 delta checkpoint에서 온 것인지 여부."""
    if snapshot is None:
        return False
    return checkpoint_metadata_uses_delta(getattr(snapshot, "metadata", {}) or {})


def raise_if_snapshot_incompatible(snapshot: Any, mode: CheckpointChannelMode) -> None:
    """full mode 프로세스가 delta checkpoint를 materialize했으면 fail-closed한다.

    ``get_state``/``get_state_history``가 반환한 ``StateSnapshot``에 대해 실행되므로, 읽기
    비용은 checkpoint fetch 한 번이다. marker는 ``snapshot.metadata``에 있다. blob을 읽는 것
    자체는 무해하다. 위험한 것은 비었거나 부분적인 state를 조용히 *사용하는* 것이며, 호출자는
    그 state를 절대 받지 않는다.
    """
    if mode == "full" and state_snapshot_uses_delta(snapshot):
        raise CheckpointModeMismatchError("Thread requires delta mode; materialize and convert its checkpoints before using full mode.")


def ensure_checkpoint_mode_compatible(checkpointer: Any, config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    """write 전 gate. write는 되돌릴 수 없으므로 미리 검사한다.

    읽기는 대신 반환된 snapshot에 :func:`raise_if_snapshot_incompatible`를 적용해 추가 fetch를
    피한다.
    """
    if mode == "delta":
        return
    if checkpoint_tuple_uses_delta(checkpointer.get_tuple(config)):
        raise CheckpointModeMismatchError("Thread requires delta mode; materialize and convert its checkpoints before using full mode.")


async def aensure_checkpoint_mode_compatible(checkpointer: Any, config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    if mode == "delta":
        return
    if checkpoint_tuple_uses_delta(await checkpointer.aget_tuple(config)):
        raise CheckpointModeMismatchError("Thread requires delta mode; materialize and convert its checkpoints before using full mode.")
