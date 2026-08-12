"""DeerFlow extension의 공개 contract.

이 패키지는 `deerflow`를 import하면 안 된다. extension이 통합에 필요한 것은 전부 여기에
있으므로, extension은 이 패키지에만 의존하며 host와 별개로 릴리스할 수 있다.
"""

from __future__ import annotations

from deerflow_extension_api.contracts import (
    ExtensionInstall,
    ExtensionRegistry,
    HostPolicySnapshot,
    MiddlewareContributor,
    extension,
)
from deerflow_extension_api.placement import (
    AgentBuildContext,
    AgentScope,
    MiddlewarePlacement,
    Placement,
)
from deerflow_extension_api.runtime_bridge import (
    EXTENSION_TASK_STORE_KEY,
    task_store_from_runtime,
)
from deerflow_extension_api.state import ExtensionData

#: Contract 버전. 1.0 이전에는 contract 표면이 관찰용(contributor와 observer)뿐이라
#: minor 버전은 깨질 수 있고 patch만 additive를 보장한다. 1.0부터는 breaking change가
#: 있으면 major를 올린다. 무엇이 additive인지는 spec의 evolution 규칙을 따른다.
API_VERSION = "0.1.0"

__all__ = [
    "API_VERSION",
    "EXTENSION_TASK_STORE_KEY",
    "AgentBuildContext",
    "AgentScope",
    "ExtensionData",
    "ExtensionInstall",
    "ExtensionRegistry",
    "HostPolicySnapshot",
    "MiddlewareContributor",
    "MiddlewarePlacement",
    "Placement",
    "extension",
    "task_store_from_runtime",
]
