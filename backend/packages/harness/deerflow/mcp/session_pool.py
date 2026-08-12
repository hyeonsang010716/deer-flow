"""stateful tool call을 위한 영속 MCP session pool.

langchain-mcp-adapters로 MCP 도구를 ``session=None``으로 로드하면 tool call마다
새 MCP session이 생성된다. Playwright 같은 stateful 서버에서는 호출 사이에
브라우저 상태(열린 페이지, 입력한 폼)가 사라진다는 뜻이다.

이 모듈은 ``(server_name, scope_key)``로 스코프된 영속 MCP session을 유지하는
session pool을 제공한다(scope_key는 보통 thread_id). 덕분에 연속된 tool call이
같은 session과 서버 측 상태를 공유한다. pool이 용량에 도달하면 LRU 순서로
session을 evict한다.

Lifecycle 모델 (owner task)
----------------------------
MCP ``ClientSession``은 ``anyio`` task group 위에 구현되어 있고, anyio는 cancel
scope를 *진입한 task와 같은 task*에서 빠져나올 것을 강제한다. ``cm.__aenter__``를
실행한 task가 아닌 곳에서 ``cm.__aexit__``를 호출하면 다음이 발생한다::

    RuntimeError: Attempted to exit cancel scope in a different task than it
    was entered in

sync tool 경로(``make_sync_tool_wrapper``)는 호출마다 새 ``asyncio.run`` event
loop에서 실행되므로, 한 호출을 처리하며 진입한 session이 다른 호출을 처리하는
도중에 — 즉 다른 task에서 — 종료되어 크래시가 난다(GitHub issue #3379).

이를 원천 차단하기 위해 pool의 모든 session은 전용 ``_run_session`` task가
소유한다. 그 task가 context manager에 진입하고, 살아 있는 session을 caller에게
넘긴 뒤 close event를 *기다린다*. 모든 종료 경로는 그 event를 **신호**만 보내고,
``__aexit__``는 owner task가 직접 수행하므로 진입과 종료가 항상 같은 task에서
일어난다.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

from mcp import ClientSession

logger = logging.getLogger(__name__)


class MCPSessionPool:
    """``(server_name, scope_key)``로 스코프된 영속 MCP session을 관리한다."""

    MAX_SESSIONS = 256
    SESSION_CLOSE_TIMEOUT = 5.0  # 다른 loop의 session을 닫을 때 대기하는 초

    def __init__(self) -> None:
        # 각 항목: (session, owning_loop, owner_task, close_event).
        self._entries: OrderedDict[
            tuple[str, str],
            tuple[
                ClientSession,
                asyncio.AbstractEventLoop,
                asyncio.Task[Any],
                asyncio.Event,
            ],
        ] = OrderedDict()
        # 진행 중인 생성 작업. key는 (server, scope). 같은 loop의 동시 caller들이
        # 각자 중복 session을 만들지 않고 하나의 생성 작업을 공유하게 한다.
        # 값: (loop, ready_future, owner_task, close_event).
        self._inflight: dict[
            tuple[str, str],
            tuple[
                asyncio.AbstractEventLoop,
                asyncio.Future[ClientSession],
                asyncio.Task[Any],
                asyncio.Event,
            ],
        ] = {}
        # threading.Lock은 어떤 event loop에도 묶이지 않으므로 async 경로와
        # sync/worker-thread 경로 양쪽에서 안전하게 획득할 수 있다.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session owner task
    # ------------------------------------------------------------------

    async def _run_session(
        self,
        connection: dict[str, Any],
        ready: asyncio.Future[ClientSession],
        close_evt: asyncio.Event,
    ) -> None:
        """MCP session 하나를 그 수명 전체 동안 소유한다.

        session context manager에 진입해 초기화하고, ``ready``로 살아 있는 session을
        공개한 뒤 ``close_evt``가 set될 때까지 블로킹한다. context manager는 *항상*
        이 task에서 빠져나오므로 anyio의 cancel scope same-task 요구사항을 만족한다.
        """
        from langchain_mcp_adapters.sessions import create_session

        cm = create_session(connection)
        try:
            session = await cm.__aenter__()
        except BaseException as e:
            # cancel scope에 진입한 적이 없으므로 빠져나올 것도 없다.
            if not ready.done():
                ready.set_exception(e)
            return

        # 이제 context manager에 진입했다. 이후로는 초기화 실패, 취소, close 신호
        # 어느 경우든 __aexit__가 반드시 이 task에서 실행되어야 anyio의 same-task
        # cancel scope 요구사항을 만족하고 session/subprocess 누수를 막을 수 있다.
        try:
            await session.initialize()
            if not ready.done():
                ready.set_result(session)
            await close_evt.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
        finally:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                logger.warning("Error closing MCP session", exc_info=True)

    async def get_session(
        self,
        server_name: str,
        scope_key: str,
        connection: dict[str, Any],
    ) -> ClientSession:
        """영속 MCP session을 가져오거나 생성한다.

        기존 session이 다른(또는 이미 닫힌) event loop에서 만들어진 것이면 evict하고,
        현재 loop의 task가 소유하는 새 session으로 교체한다.

        Args:
            server_name: MCP 서버 이름.
            scope_key: 격리 key (보통 thread_id).
            connection: ``create_session``에 넘길 연결 설정.

        Returns:
            초기화된 ``ClientSession``.
        """
        key = (server_name, scope_key)
        current_loop = asyncio.get_running_loop()

        # Phase 1: thread lock 아래에서 registry를 조회/변경한다(await 없음).
        # 세 결과 중 하나를 원자적으로 결정한다: 기존 session 반환, 진행 중인 생성에
        # 합류, 또는 이 key의 생성자가 되기.
        # 각 항목: (loop, owner_task, close_event, cancel). ``cancel``은 진행 중인
        # 생성 작업에 대해 True다. 그 owner는 close_evt로 깨울 수 없는
        # ``initialize()`` 안에서 블로킹되어 있을 수 있어 취소해야 한다.
        evicted: list[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any], asyncio.Event, bool]] = []
        join: asyncio.Future[ClientSession] | None = None
        ready: asyncio.Future[ClientSession] | None = None
        close_evt: asyncio.Event | None = None
        task: asyncio.Task[Any] | None = None
        with self._lock:
            if key in self._entries:
                session, loop, ent_task, ent_close = self._entries[key]
                if loop is current_loop and not loop.is_closed():
                    self._entries.move_to_end(key)
                    return session
                # 다른/닫힌 event loop에 속한 session이므로 evict한다.
                self._entries.pop(key)
                evicted.append((loop, ent_task, ent_close, False))

            inflight = self._inflight.get(key)
            if inflight is not None and inflight[0] is current_loop and not inflight[0].is_closed():
                # 이 loop의 다른 caller가 이미 session을 만들고 있으므로, 중복 생성
                # 대신 같은 결과를 기다린다.
                join = inflight[1]
            else:
                if inflight is not None:
                    # 다른/닫힌 loop가 소유한 오래된 진행 중 생성 작업. 기록을 버리고
                    # owner를 정리한다. 그 owner는 close_evt로 깨울 수 없는
                    # initialize() 안에서 블로킹되어 있을 수 있으므로 취소해야 한다.
                    # 그런 다음 여기서 새 session을 만든다.
                    self._inflight.pop(key)
                    evicted.append((inflight[0], inflight[2], inflight[3], True))
                # 생성자가 된다: await 전에 진행 중 기록을 먼저 공개해 동시 caller가
                # 경쟁하지 않고 합류하게 한다.
                ready = current_loop.create_future()
                close_evt = asyncio.Event()
                task = current_loop.create_task(self._run_session(connection, ready, close_evt))
                self._inflight[key] = (current_loop, ready, task, close_evt)

            # 용량에 도달하면 LRU 항목을 evict한다.
            while len(self._entries) >= self.MAX_SESSIONS:
                oldest_key, (_, loop, ent_task, ent_close) = next(iter(self._entries.items()))
                self._entries.pop(oldest_key)
                evicted.append((loop, ent_task, ent_close, False))

        # Phase 2: evict된 session/생성 작업을 종료한다. 같은 loop의 owner는 결정적으로
        # 끝나도록 await하고, 다른 loop의 owner는 자기 loop로 라우팅한다. 어느 경우든
        # __aexit__는 이 task가 아니라 owner task가 실행한다. 진행 중 owner는
        # cancel=True로 취소해 블로킹된 initialize()가 그들을 멈춰 세우지 못하게 한다.
        for loop, ent_task, ent_close, cancel in evicted:
            if loop is current_loop and not loop.is_closed():
                await self._shutdown(ent_close, ent_task, cancel)
            elif cancel:
                await self._shutdown_entry(loop, ent_task, ent_close, cancel=True)
            else:
                self._signal_close(loop, ent_close)

        # Phase 2b: 이 loop에서 같은 key의 생성이 이미 진행 중이므로, 중복 session을
        # 만들지 않고 그 결과를 공유한다.
        if join is not None:
            return await asyncio.shield(join)

        assert ready is not None and close_evt is not None and task is not None

        # Phase 3: owner task가 초기화된 session을 공개할 때까지 기다린다.
        try:
            session = await asyncio.shield(ready)
        except BaseException:
            # 여기에 도달하는 경우는 두 가지다.
            #
            # 1. owner task가 실패해(예: connect/initialize 오류) ready.set_exception()으로
            #    보고한 경우. 이미 자신의 finally 블록에서 자기 task로 cm.__aexit__를
            #    실행 중이므로 취소하면 **안 된다** — 그 정리 작업을 끊게 된다.
            #    풀려서 끝날 때까지 기다리기만 한다.
            # 2. 이 호출 자체가 취소된 경우(CancelledError). shield 덕분에 `ready`는
            #    여전히 pending이고 owner task는 살아서 블로킹 중이다. close를 신호하고
            #    취소해 owner가 자기 task에서 cancel scope를 빠져나오게 한 뒤 종료를
            #    기다린다.
            #
            # session은 아직 등록되지 않아 아무도 닫을 수 없으므로, 여기서 기다리면
            # session이나 owner task가 절대 누수되지 않는다.
            owner_already_failed = ready.done() and not ready.cancelled() and ready.exception() is not None
            if not owner_already_failed:
                close_evt.set()
                task.cancel()
            try:
                await asyncio.shield(task)
            except BaseException:
                logger.debug("Owner task ended during get_session unwind", exc_info=True)
            with self._lock:
                if self._inflight.get(key) == (current_loop, ready, task, close_evt):
                    self._inflight.pop(key)
            raise

        # Phase 4: 진행 중 생성 작업을 등록 항목으로 승격한다. 단 우리 진행 중 기록이
        # 여전히 살아 있는 것일 때만이다. 초기화 도중 동시 close_* / close_all이 그
        # 기록을 지웠을 수 있고, 그때는 session을 _entries로 되살리면 **안 된다**.
        # 대신 우리가 teardown을 책임진다: owner task에 신호를 보내 자기 task에서
        # __aexit__를 실행하도록 기다린 뒤 취소를 표면화한다.
        with self._lock:
            still_ours = self._inflight.get(key) == (current_loop, ready, task, close_evt)
            if still_ours:
                self._inflight.pop(key)
                self._entries[key] = (session, current_loop, task, close_evt)
        if not still_ours:
            await self._shutdown(close_evt, task)
            raise asyncio.CancelledError("MCP session pool was closed while the session was being created")
        logger.info("Created persistent MCP session for %s/%s", server_name, scope_key)
        return session

    # ------------------------------------------------------------------
    # 정리 helper
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_close(loop: asyncio.AbstractEventLoop, close_evt: asyncio.Event) -> None:
        """owner task에 종료를 요청하고 기다리지 않는다.

        ``asyncio.Event.set``은 thread-safe하지 않으므로 소유 loop에 스케줄한다.
        loop가 닫혀 있다면 owner task는 이미 사라진 것이다.
        """
        if loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(close_evt.set)
        except RuntimeError:
            # is_closed() 검사와 지금 사이에 loop가 닫혔다.
            pass

    async def _shutdown(
        self,
        close_evt: asyncio.Event,
        task: asyncio.Task[Any],
        cancel: bool = False,
    ) -> None:
        """owner task에 신호를 보내고 종료를 기다린다(해당 loop 위에서 실행).

        ``cancel=True``는 진행 중인 생성 작업에 쓴다. owner task가 ``close_evt``로
        깨울 수 없는 ``initialize()`` 안에서 블로킹되어 있을 수 있어 취소해야 한다.
        그래도 ``finally`` 블록이 자기 task에서 ``__aexit__``를 실행하므로 anyio의
        same-task cancel scope 요구사항을 만족한다.
        """
        close_evt.set()
        if cancel:
            task.cancel()
        try:
            await task
        except (Exception, asyncio.CancelledError):
            logger.debug("Owner task ended during shutdown", exc_info=True)

    async def _shutdown_entry(
        self,
        loop: asyncio.AbstractEventLoop,
        task: asyncio.Task[Any],
        close_evt: asyncio.Event,
        cancel: bool = False,
    ) -> None:
        """항목 하나를 종료하되, close를 소유 loop로 라우팅한다."""
        if loop.is_closed():
            return
        current_loop = asyncio.get_running_loop()
        if loop is current_loop:
            await self._shutdown(close_evt, task, cancel)
        elif loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(close_evt, task, cancel), loop)
            try:
                await asyncio.wrap_future(future)
            except Exception:
                logger.warning("Error closing MCP session on owning loop", exc_info=True)
        else:
            # 소유 loop가 존재하지만 현재 loop도 아니고 실행 중도 아니다. 여기는 async
            # context 안이므로 run_until_complete()는 "Cannot run the event loop while
            # another loop is running"을 발생시키고, 그 loop가 다른 thread 소유일 수도
            # 있어 여기서 구동하는 것은 안전하지 않다. 실제로는 도달하지 않는 분기다 —
            # session의 소유 loop는 장수 gateway loop(실행 중)이거나 단명
            # asyncio.run loop(닫혀 있어 위에서 걸러짐)이기 때문이다. 그 loop가 다시
            # 돌 때 owner task가 정리되도록 best-effort thread-safe 신호로 폴백한다.
            logger.warning("Owning loop for MCP session is idle; signalling close best-effort. Session may leak until the loop runs again.")
            self._signal_close(loop, close_evt)
            if cancel:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass

    async def close_scope(self, scope_key: str) -> None:
        """주어진 scope(예: thread_id)의 모든 session을 닫는다."""
        with self._lock:
            keys = [k for k in self._entries if k[1] == scope_key]
            entries = [(self._entries.pop(k)) for k in keys]
            inflight_keys = [k for k in self._inflight if k[1] == scope_key]
            inflight = [self._inflight.pop(k) for k in inflight_keys]
        for _session, loop, task, close_evt in entries:
            await self._shutdown_entry(loop, task, close_evt)
        for loop, _ready, task, close_evt in inflight:
            await self._shutdown_entry(loop, task, close_evt, cancel=True)

    async def close_server(self, server_name: str) -> None:
        """주어진 서버의 모든 session을 닫는다."""
        with self._lock:
            keys = [k for k in self._entries if k[0] == server_name]
            entries = [(self._entries.pop(k)) for k in keys]
            inflight_keys = [k for k in self._inflight if k[0] == server_name]
            inflight = [self._inflight.pop(k) for k in inflight_keys]
        for _session, loop, task, close_evt in entries:
            await self._shutdown_entry(loop, task, close_evt)
        for loop, _ready, task, close_evt in inflight:
            await self._shutdown_entry(loop, task, close_evt, cancel=True)

    async def close_all(self) -> None:
        """관리 중인 모든 session을 닫는다."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()
        for _session, loop, task, close_evt in entries:
            await self._shutdown_entry(loop, task, close_evt)
        for loop, _ready, task, close_evt in inflight:
            await self._shutdown_entry(loop, task, close_evt, cancel=True)

    def close_all_sync(self) -> None:
        """모든 session을 각자의 소유 event loop에서 닫는다(동기).

        각 session은 생성된 loop 위에서 자신의 owner task가 닫으므로 cross-loop /
        cross-task 오류를 피한다. 활성 event loop가 없는 어떤 thread에서 호출해도
        안전하다.

        닫기 의미는 소유 loop가 어디서 도는지에 따라 다르다.

        * 소유 loop가 유휴 상태이거나 다른 thread에서 도는 경우 — 이 호출은 teardown이
          끝날 때까지(또는 ``SESSION_CLOSE_TIMEOUT``이 지날 때까지) 블로킹한다.
        * 소유 loop가 *이* thread에서 현재 돌고 있는 loop인 경우 — 블로킹하면 deadlock이
          되므로 여기서는 teardown을 *신호*만 하고, 제어가 그 loop로 돌아간 뒤 비동기로
          완료된다. 따라서 caller는 이후에도 그 loop를 계속 돌려야 한다. 즉시 loop를
          멈추면 owner task의 ``__aexit__``가 실행되지 않을 수 있다. 실행 중인 loop
          안에서 결정적인 close가 필요하면 ``await close_all()``을 쓴다.
        """
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()

        # 등록된 항목은 초기화가 끝난 상태다(부드러운 close_evt 경로). 진행 중인 생성
        # 작업은 초기화 도중 블로킹되어 있을 수 있어 teardown을 풀기 위해 취소한다.
        owners = [(loop, task, close_evt, False) for _s, loop, task, close_evt in entries]
        owners += [(loop, task, close_evt, True) for loop, _r, task, close_evt in inflight]
        try:
            current_running_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_running_loop = None
        for loop, task, close_evt, cancel in owners:
            if loop.is_closed():
                continue
            try:
                if loop is current_running_loop:
                    # 이 loop의 thread 안에서 실행 중이므로
                    # run_coroutine_threadsafe(...).result()를 동기로 기다리면 timeout까지
                    # deadlock에 빠진다. owner task에 직접 신호만 보내고, 이 동기 호출이
                    # 실행 중인 loop로 제어를 돌려준 뒤 끝내게 한다.
                    close_evt.set()
                    if cancel:
                        task.cancel()
                elif loop.is_running():
                    # 이 thread에서 소유 loop에 shutdown을 스케줄한다.
                    future = asyncio.run_coroutine_threadsafe(self._shutdown(close_evt, task, cancel), loop)
                    future.result(timeout=self.SESSION_CLOSE_TIMEOUT)
                else:
                    loop.run_until_complete(self._shutdown(close_evt, task, cancel))
            except Exception:
                logger.debug("Error closing MCP session during sync close", exc_info=True)


# ------------------------------------------------------------------
# 모듈 수준 singleton
# ------------------------------------------------------------------

_pool: MCPSessionPool | None = None
_pool_lock = threading.Lock()


def get_session_pool() -> MCPSessionPool:
    """전역 session pool singleton을 반환한다."""
    global _pool
    # lock 아래에서 생성하고 반환한다. 그래야 cold start에서 경쟁하는 caller들이 pool을
    # 정확히 하나만 만들고, reset_session_pool()이 전역을 읽고 반환하는 사이에 None으로
    # 만들지 못한다(이전에는 None이 반환될 수 있었다). critical section이 아주 짧고
    # await하지 않으므로 async 경로와 sync/worker-thread 경로 양쪽에서 threading.Lock을
    # 잡아도 안전하다.
    with _pool_lock:
        if _pool is None:
            _pool = MCPSessionPool()
        return _pool


def reset_session_pool() -> None:
    """singleton을 리셋한다(테스트와 MCP cache 리셋 경로에서 사용)."""
    global _pool
    with _pool_lock:
        _pool = None
