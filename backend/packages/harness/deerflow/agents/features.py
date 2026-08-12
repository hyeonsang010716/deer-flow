"""create_deerflow_agent용 선언적 feature 플래그와 middleware 위치 지정.

순수 데이터 클래스와 decorator만 있다. I/O도 부수 효과도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from deerflow.config.memory_config import MemoryConfig


@dataclass
class RuntimeFeatures:
    """``create_deerflow_agent``용 선언적 feature 플래그.

    대부분의 feature는 다음을 받는다.

    - ``True``: 내장 기본 middleware를 쓴다.
    - ``False``: 비활성화한다.
    - ``AgentMiddleware`` 인스턴스: 이 커스텀 구현으로 대체한다.

    ``summarization``과 ``guardrail``은 내장 기본값이 없어 ``False``(비활성화) 또는
    ``AgentMiddleware`` 인스턴스(커스텀)만 받는다.
    """

    sandbox: bool | AgentMiddleware = True
    memory: bool | AgentMiddleware = False
    # create_deerflow_agent(features=...)를 직접 호출하는 쪽이 명시하는 memory config.
    # lead-agent의 AppConfig 경로는 resolved_app_config.memory를 그대로 넘긴다.
    memory_config: MemoryConfig | None = None
    summarization: Literal[False] | AgentMiddleware = False
    subagent: bool | AgentMiddleware = False
    vision: bool | AgentMiddleware = False
    auto_title: bool | AgentMiddleware = False
    guardrail: Literal[False] | AgentMiddleware = False
    loop_detection: bool | AgentMiddleware = True
    token_budget: bool | AgentMiddleware = False


# ---------------------------------------------------------------------------
# middleware 위치 지정 decorator
# ---------------------------------------------------------------------------


def Next(anchor: type[AgentMiddleware]):
    """이 middleware를 chain에서 *anchor* 뒤에 놓도록 선언한다."""
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Next expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._next_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator


def Prev(anchor: type[AgentMiddleware]):
    """이 middleware를 chain에서 *anchor* 앞에 놓도록 선언한다."""
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Prev expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._prev_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator
