"""터미널 session이 Web UI에 보이도록 공유 persistence를 연결한다.

Web UI는 checkpointer가 아니라 ``threads_meta`` SQL 테이블(``user_id``로 필터링)에서 대화를
나열한다. embedded run은 checkpointer만 쓰므로 TUI thread는 사이드바에서 보이지 않는다. 이
모듈이 그 간극을 메운다. Gateway 프로세스가 떠 있지 않아도, Gateway가 읽는 **같은** 데이터베이스에
로컬 기본 user 소유의 ``threads_meta`` row를 쓴다.

여기 있는 모든 동작은 best-effort다. 데이터베이스가 memory 기반이거나 사용할 수 없으면 writer는
no-op으로 낮아지고 TUI는 계속 동작한다.

SQLAlchemy async engine은 자신을 만든 event loop에 묶이므로, 모든 DB 작업은 호출마다 새로
``asyncio.run``을 돌리는 대신(그러면 connection이 일회용 loop에 묶인다) 오래 사는 단일 background
loop(``_LoopThread``)에서 실행한다.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import Any

from deerflow.runtime.user_context import DEFAULT_USER_ID


class _LoopThread:
    """DB 작업용 asyncio event loop 하나를 돌리는 daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="deerflow-tui-db", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Awaitable[Any], *, timeout: float = 15.0) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class ThreadMetaWriter:
    """로컬 기본 user의 ``threads_meta`` row를 쓰거나 갱신한다.

    모든 메서드는 에러를 삼킨다. persistence 가시성은 편의 기능일 뿐 대화를 깨뜨릴 이유가 되지
    않는다.
    """

    def __init__(self, loop: _LoopThread, store: Any) -> None:
        self._loop = loop
        self._store = store
        self.user_id = DEFAULT_USER_ID

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def ensure_created(self, thread_id: str, *, assistant_id: str | None = None, metadata: dict | None = None) -> None:
        if not self._store or not thread_id:
            return
        try:
            self._loop.run(self._ensure_created(thread_id, assistant_id, metadata))
        except Exception:  # noqa: BLE001 - best-effort로 무시한다
            pass

    async def _ensure_created(self, thread_id: str, assistant_id: str | None, metadata: dict | None) -> None:
        existing = await self._store.get(thread_id, user_id=self.user_id)
        if existing is None:
            await self._store.create(
                thread_id,
                assistant_id=assistant_id,
                user_id=self.user_id,
                metadata=metadata or {"source": "tui"},
            )

    def set_title(self, thread_id: str, title: str) -> None:
        if not self._store or not thread_id or not title:
            return
        try:
            self._loop.run(self._store.update_display_name(thread_id, title, user_id=self.user_id))
        except Exception:  # noqa: BLE001 - best-effort로 무시한다
            pass


def build_persistence() -> tuple[_LoopThread, ThreadMetaWriter]:
    """background loop에서 공유 engine을 초기화하고 writer를 반환한다.

    설정된 데이터베이스 backend가 ``memory``이거나(SQL session factory가 없다) 초기화에 실패하면
    no-op인 ``ThreadMetaWriter``를 반환한다.
    """
    loop = _LoopThread()
    store = None
    try:
        from deerflow.config.app_config import get_app_config
        from deerflow.persistence.engine import get_session_factory, init_engine_from_config
        from deerflow.persistence.thread_meta import make_thread_store

        config = get_app_config()
        loop.run(init_engine_from_config(config.database))
        session_factory = get_session_factory()
        if session_factory is not None:
            store = make_thread_store(session_factory)
    except Exception:  # noqa: BLE001 - no-op writer로 낮춘다
        store = None
    return loop, ThreadMetaWriter(loop, store)
