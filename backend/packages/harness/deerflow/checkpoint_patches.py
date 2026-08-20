"""서드파티 checkpoint 기계장치에 대한 호환성 패치.

``deerflow.runtime``이 아니라 최상위 패키지에 둔다. 그래야 ``deerflow.agents.thread_state``에서
import할 때 runs 기계장치를 즉시 import하는 무거운 ``deerflow.runtime`` 패키지 __init__을
끌어오지 않는다. ``deerflow.agents.thread_state``에서 앵커링하므로 DeerFlow 그래프를 만드는
모든 프로세스(gateway, worker, in-process LangGraph runtime, 테스트)가 이 수정을 적용받는다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.errors import ErrorCode, InvalidUpdateError, create_error_message
from langgraph.types import Overwrite

logger = logging.getLogger(__name__)

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
    죽는다(#4380).

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


ensure_binop_overwrite_first_write_patch()
