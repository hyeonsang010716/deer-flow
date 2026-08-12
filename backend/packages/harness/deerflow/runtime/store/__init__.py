"""DeerFlow runtime의 Store provider.

async provider(장시간 실행되는 서버용)와 sync provider(CLI 도구 및 embedded client용)의 공개
API를 다시 export한다.

async 사용법(FastAPI lifespan)::

    from deerflow.runtime.store import make_store

    async with make_store() as store:
        app.state.store = store

sync 사용법(CLI / DeerFlowClient)::

    from deerflow.runtime.store import get_store, store_context

    store = get_store()                   # 싱글턴
    with store_context() as store: ...    # 일회성
"""

from .async_provider import make_store
from .provider import get_store, reset_store, store_context

__all__ = [
    # async
    "make_store",
    # sync
    "get_store",
    "reset_store",
    "store_context",
]
