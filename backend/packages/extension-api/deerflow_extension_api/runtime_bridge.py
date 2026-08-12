"""LangGraph runtime context와 task 범위 store 사이의 bridge.

middleware는 agent graph 안에서 실행되며 host 상태에는 ``request.runtime``으로만 닿을 수
있다. host는 host 소유 key로 task store를 설치하고, extension은 이 helper로 그것을 읽어
자기 객체를 store *안에* 보관한다. 그래야 두 extension이 runtime context key에서 충돌하지
않는다.
"""

from __future__ import annotations

from collections.abc import Mapping

from deerflow_extension_api.state import ExtensionData

#: host 소유 key. extension은 runtime context에 직접 쓰면 안 된다.
EXTENSION_TASK_STORE_KEY = "__deerflow_extension_task_store"


def task_store_from_runtime(runtime: object) -> ExtensionData | None:
    """task 범위 store를 반환한다. 살아 있는 task가 없으면 None을 반환한다."""
    context = getattr(runtime, "context", None)
    if not isinstance(context, Mapping):
        return None
    store = context.get(EXTENSION_TASK_STORE_KEY)
    return store if isinstance(store, ExtensionData) else None
