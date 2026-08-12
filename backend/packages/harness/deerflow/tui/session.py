"""TUI용 embedded session 배선.

영속 checkpointer를 붙인 ``DeerFlowClient`` 생성, ``--continue`` / ``--resume``의 thread 해석
(id **또는** 제목 기준), 그리고 터미널 session을 Web UI에 보이게 하는 공유 persistence writer를
담당한다(``deerflow.tui.persistence`` 참고).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 순수 planning 단계에서 무거운 client를 import하지 않기 위함
    from deerflow.client import DeerFlowClient

    from .cli import LaunchPlan
    from .persistence import ThreadMetaWriter, _LoopThread


@dataclass
class Session:
    client: DeerFlowClient
    writer: ThreadMetaWriter | None = None
    _loop: _LoopThread | None = None

    def resolve_thread(self, plan: LaunchPlan) -> str | None:
        """--resume / --continue를 반영해 실행 대상 thread id를 해석한다."""
        if plan.thread_id:
            return self.resolve_ref(plan.thread_id)
        if plan.continue_recent:
            threads = self.client.list_threads(limit=1).get("thread_list", [])
            if threads:
                return threads[0].get("thread_id")
        return None

    def resolve_ref(self, ref: str) -> str:
        """thread 참조(id 또는 제목)를 thread id로 해석한다.

        먼저 id로 기존 thread를 찾고, 그다음 제목이 정확히 일치하는지 본다. 아무것도 맞지 않으면
        ref 문자열 자체를 id로 간주해 fallback하므로, 알 수 없는 id도 표준 thread ID 계약을
        만족하는 한 그 네임스페이스를 이어가거나 새로 만든다.
        """
        try:
            threads = self.client.list_threads(limit=100).get("thread_list", [])
        except Exception:  # noqa: BLE001 - 해석은 best-effort다
            return self._validated_literal_ref(ref)
        if any(t.get("thread_id") == ref for t in threads):
            return ref
        for thread in threads:
            if (thread.get("title") or "") == ref:
                return thread.get("thread_id") or self._validated_literal_ref(ref)
        return self._validated_literal_ref(ref)

    @staticmethod
    def _validated_literal_ref(ref: str) -> str:
        """ref 문자열을 thread id로 채택하기 전에 검증한다."""
        from deerflow.utils.thread_id import validate_thread_id

        try:
            return validate_thread_id(ref)
        except ValueError as exc:
            raise ValueError(f"Thread reference {ref!r} matches no existing thread and is not a valid thread id (expected 1-64 ASCII letters, digits, hyphens, or underscores).") from exc

    def recent_threads(self, limit: int = 20) -> list[dict]:
        return self.client.list_threads(limit=limit).get("thread_list", [])

    def close(self) -> None:
        """background DB loop를 멈추고 engine을 정리한다(best-effort)."""
        loop = self._loop
        if loop is None:
            return
        self._loop = None
        try:
            from deerflow.persistence.engine import close_engine

            loop.run(close_engine())
        except Exception:  # noqa: BLE001 - teardown은 best-effort다
            pass
        loop.close()


def open_session(persistence: bool = True) -> Session:
    """설정된 checkpointer를 백엔드로 쓰는 embedded session을 만든다.

    ``persistence``는 공유 ``threads_meta`` writer(및 그 background DB loop/engine)를 제어한다.
    headless 일회성 실행은 writer를 쓰지 않으므로 ``persistence=False``를 넘겨, event loop와
    connection pool을 세웠다가 바로 버리는 일을 피한다.
    """
    from deerflow.client import DeerFlowClient
    from deerflow.runtime.checkpointer.provider import get_checkpointer

    checkpointer = get_checkpointer()
    client = DeerFlowClient(checkpointer=checkpointer)
    if not persistence:
        return Session(client=client)

    from .persistence import build_persistence

    loop, writer = build_persistence()
    return Session(client=client, writer=writer, _loop=loop)
