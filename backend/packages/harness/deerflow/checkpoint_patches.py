"""서드파티 checkpoint 기계장치에 대한 호환성 패치.

``deerflow.runtime``이 아니라 최상위 패키지에 둔다. 그래야 ``deerflow.agents.thread_state``에서
import할 때 runs 기계장치를 즉시 import하는 무거운 ``deerflow.runtime`` 패키지 __init__을
끌어오지 않는다. ``deerflow.agents.thread_state``에서 앵커링하므로 DeerFlow 그래프를 만드는
모든 프로세스(gateway, worker, in-process LangGraph runtime, 테스트)가 이 수정을 적용받는다.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Sequence
from typing import Any

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import ErrorCode, InvalidUpdateError, create_error_message
from langgraph.types import Overwrite
from packaging.version import Version

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_deerflow_delta_history_patched"
# 이 패치는 langgraph 1.2.9
# (langgraph/checkpoint/memory/__init__.py::InMemorySaver.get_delta_channel_history)를
# 기준으로 작성·검증했다. 더 최신 LangGraph에서는 패치를 유지하기 전에 override를 다시 확인해야 한다.
# upstream이 고쳤거나 제거했다면 이 모듈은 물러나야 한다.
_PATCH_VALIDATED_LANGGRAPH_VERSION = Version("1.2.9")


def _get_delta_channel_history_via_base(self: Any, *, config: Any, channels: Any) -> Any:
    return BaseCheckpointSaver.get_delta_channel_history(self, config=config, channels=channels)


async def _aget_delta_channel_history_via_base(self: Any, *, config: Any, channels: Any) -> Any:
    return await BaseCheckpointSaver.aget_delta_channel_history(self, config=config, channels=channels)


def _upstream_override_present() -> bool:
    """InMemorySaver가 여전히 자체 (버그 있는) override를 제공하는 동안 True."""
    return (
        getattr(InMemorySaver, "get_delta_channel_history", None) is not None
        and InMemorySaver.get_delta_channel_history is not BaseCheckpointSaver.get_delta_channel_history
        and InMemorySaver.aget_delta_channel_history is not BaseCheckpointSaver.aget_delta_channel_history
    )


def ensure_inmemory_delta_history_patch() -> None:
    """full -> delta로 migrate된 thread에서 InMemorySaver가 write를 잃는 문제를 고친다.

    ``InMemorySaver.get_delta_channel_history``는 base walk를 single-pass 버전으로 override한다.
    이 버전은 어떤 채널에 비어 있지 않은 plain-value blob을 가진 첫 checkpoint에 도달하면
    그 checkpoint *자신의* pending write를 blob에 "포함된" 것으로 보고 건너뛴다. 이는 blob을
    바로 그 checkpoint가 썼을 때만 참이다. 버전이 더 오래된 조상에서 그대로 이어져 온 경우 —
    정확히 full -> delta migration 직후의 첫 superstep, 즉 입력 write가 아직 pre-delta blob
    버전을 참조하는 checkpoint에 떨어지는 상황 — 그 pending write들은 blob보다 나중이므로
    조용히 버려진다. 그 결과 migration 후 처음 추가된 메시지가 materialize된 state에서 사라진다.

    base 구현(SQLite saver들이 사용)과 Postgres override는 둘 다 종단 checkpoint의 write를
    blob을 seed로 삼기 *전에* 수집하는데, 이것이 올바른 순서다. 이 패치는 InMemorySaver를
    base 구현에 위임한다. 융합된 단일 walk 대신 조상마다 ``get_tuple`` 한 번을 쓰지만
    dict 기반 저장소에서는 문제없다.

    멱등하다. upstream override가 사라지거나 대입이 실패하면 물러나도록 방어했고,
    LangGraph가 검증된 버전을 넘어가면 upstream 수정을 조용히 덮어쓰지 않도록 경고를 남겨
    패치를 재검토하게 한다. LangGraph가 upstream에서 override를 고치면 제거한다
    (아직 upstream 이슈는 없다. langgraph를 올릴 때마다
    ``InMemorySaver.get_delta_channel_history``를 다시 확인한다).
    """
    if getattr(InMemorySaver, _PATCH_FLAG, False):
        return
    try:
        langgraph_version = Version(importlib.metadata.version("langgraph"))
    except Exception:
        langgraph_version = _PATCH_VALIDATED_LANGGRAPH_VERSION
    if langgraph_version > _PATCH_VALIDATED_LANGGRAPH_VERSION:
        logger.warning(
            "langgraph %s is newer than the version (%s) the InMemorySaver delta-history patch was validated against; re-inspect the upstream override before relying on the patch.",
            langgraph_version,
            _PATCH_VALIDATED_LANGGRAPH_VERSION,
        )
    try:
        if not _upstream_override_present():
            # upstream이 override를 제거했다(수정 또는 리팩터링).
            # 이미 base 구현이 쓰이고 있으므로 패치할 것이 없다.
            return
        InMemorySaver.get_delta_channel_history = _get_delta_channel_history_via_base  # type: ignore[method-assign]
        InMemorySaver.aget_delta_channel_history = _aget_delta_channel_history_via_base  # type: ignore[method-assign]
        setattr(InMemorySaver, _PATCH_FLAG, True)
    except (AttributeError, TypeError):
        logger.warning("Failed to apply the InMemorySaver delta-history patch; leaving the upstream implementation untouched.", exc_info=True)


_BINOP_PATCH_FLAG = "_deerflow_overwrite_first_write_patched"
_unpatched_binop_update = BinaryOperatorAggregate.update


def _as_overwrite(value: Any) -> tuple[bool, Any]:
    """langgraph의 private ``_get_overwrite``를 대신하는 로컬 구현.

    공개 ``Overwrite`` *클래스* 형태만 매칭한다. DeerFlow의 write 경로가 이 Union 채널들에
    만들어내는 유일한 형태다(branch와 ``/state`` 라우트가 replace 방식 write를
    ``Overwrite(...)``로 감싼다). 밑줄로 시작하는 ``_get_overwrite``를 import하지 않는 이유는,
    upstream이 그것을 제거하는 리팩터링(하필 버그를 고치는 바로 그 릴리스일 수 있다)을 했을 때
    이 모듈의 import가 실패해 probe가 패치를 물리기도 전에 startup이 죽는 것을 막기 위해서다.
    upstream이 함께 받아들이는 dict sentinel 형태는 내부 직렬화 세부사항이며 DeerFlow는
    이 채널들에 그것을 내보내지 않는다.
    """
    if isinstance(value, Overwrite):
        return True, value.value
    return False, None


def _binop_first_write_stores_overwrite_wrapper() -> bool:
    """upstream이 아직도 Overwrite 첫 write를 그대로 저장하는지 탐지한다.

    Union 타입 채널(생성 가능한 기본값이 없어 MISSING에서 시작한다)을 쓴다.
    ``ThreadState``의 ``sandbox`` / ``goal`` / ``todos`` / ``promoted`` 채널과 같은 형태다.
    """
    channel = BinaryOperatorAggregate(dict | None, lambda existing, new: new)
    channel.key = "deerflow-overwrite-probe"
    channel.update([Overwrite({"probe": True})])
    return isinstance(channel.get(), Overwrite)


def _binop_update_unwrapping_empty_channel(self: Any, values: Sequence[Any]) -> bool:
    """Overwrite 첫 write를 풀어주는 ``BinaryOperatorAggregate.update``.

    빈 채널 + 선두 Overwrite 조합만 가로채고 나머지는 upstream 구현에 위임한다.
    가로챈 경우의 동작은 upstream의 post-Overwrite 배치 의미를 그대로 따른다.
    뒤따르는 plain 값은 건너뛰고, 두 번째 Overwrite는 ``InvalidUpdateError``를 던진다.
    """
    if not self.is_available() and values:
        is_overwrite, overwrite_value = _as_overwrite(values[0])
        if is_overwrite:
            self.value = overwrite_value
            for value in values[1:]:
                if _as_overwrite(value)[0]:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
            return True
    return _unpatched_binop_update(self, values)


def ensure_binop_overwrite_first_write_patch() -> None:
    """빈 채널에 ``Overwrite`` 첫 write가 그대로 저장되는 문제를 고친다.

    upstream ``BinaryOperatorAggregate.update``는 빈 채널(``self.value is MISSING``)을
    ``values[0]``으로 그대로 seed한다. 메서드의 나머지 부분이 적용하는 Overwrite unwrapping을
    거치지 않는다. 타입이 Union인 채널(``SandboxState | None``, ``GoalState | None``, ...)은
    생성 가능한 기본값이 없어 MISSING에서 시작하므로, 새 thread(thread branching)나 한 번도
    쓰이지 않은 채널(state update)에 replace 방식 write가 들어오면 ``Overwrite`` wrapper 자체가
    checkpoint에 저장되고, 다음 소비자가 ``TypeError: 'Overwrite' object is not subscriptable``로
    죽는다(#4380). ``DeltaChannel.update``는 같은 상황에서 이미 unwrap하므로, 이 패치는
    두 reducer 채널 타입 간의 동작 불일치도 함께 없앤다.

    멱등하다. 버전 고정이 아니라 동작 probe로 방어한다. 미래의 LangGraph가 첫 write를 직접
    unwrap하면 probe가 버그 없음을 보고하고 패치는 물러난다.
    """
    if getattr(BinaryOperatorAggregate, _BINOP_PATCH_FLAG, False):
        return
    try:
        if not _binop_first_write_stores_overwrite_wrapper():
            # upstream이 첫 write를 직접 unwrap한다. 패치할 것이 없다.
            return
        BinaryOperatorAggregate.update = _binop_update_unwrapping_empty_channel  # type: ignore[method-assign]
        setattr(BinaryOperatorAggregate, _BINOP_PATCH_FLAG, True)
    except Exception:
        logger.warning("Failed to apply the BinaryOperatorAggregate Overwrite first-write patch; leaving the upstream implementation untouched.", exc_info=True)


ensure_inmemory_delta_history_patch()
ensure_binop_overwrite_first_write_patch()
