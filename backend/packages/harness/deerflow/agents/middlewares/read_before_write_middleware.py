"""파일을 수정하는 도구를 위한 결정적 read-before-write gate(issue #3857).

lead agent가 같은 보고서 섹션을 다섯 번 덧붙이던 중복 출력 실패는 "append만 하고 다시 읽지
않는" 쓰기에서 비롯됐다. 이 middleware는 버전 gate를 강제한다: 기존 파일을 수정하려면 대화
앞부분에 그 파일의 *현재* 버전에 대한 ``read_file``이 있어야 한다.

설계 불변식:
- 도구는 stateless로 유지한다. read mark(전체 파일 내용의 ``sha256``)는 ``read_file``
  ToolMessage의 ``additional_kwargs``에 찍히므로, gate의 state는 ``state["messages"]``에 산다.
- summarization이 read 결과를 지우면 mark도 함께 사라진다 — read한 내용이 context에서 없어진
  동안에는 gate가 절대 통과하지 않는다.
- 쓰기는 mark를 갱신하지 않는다. 성공한 쓰기는 파일 hash를 바꾸므로 이전의 모든 read를
  무효화하고, 연속된 수정 사이에 재read를 강제한다.
- gate 확인과 도구 실행은 (scope, path) 단위로 직렬화한다. LangGraph는 한 AIMessage의 tool
  call들을 동시에 실행하므로, critical section이 없으면 같은 턴의 두 쓰기가 어느 쪽 변경도
  반영되기 전에 하나의 오래된 mark로 모두 통과할 수 있다. 같은 lock이 ``read_file``과 mark
  기록도 감싸므로, mark는 항상 모델이 실제로 본 버전을 hash한다.
- fail-open: gate 자체가 파일을 확인할 수 없으면(sandbox 일시 오류, 바이너리 내용, 또는
  AIO/E2B처럼 read 실패를 예외 대신 ``"Error: ..."`` 문자열로 알리는 sandbox) 도구를 그대로
  실행시켜 도구가 스스로 에러를 내게 한다.
"""

import asyncio
import hashlib
import logging
import posixpath
import threading
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.tool_result_meta import normalize_tool_result
from deerflow.sandbox.tools import read_current_file_content

logger = logging.getLogger(__name__)

READ_MARK_KEY = "deerflow_read_mark"

_READ_TOOLS = frozenset({"read_file"})
_GATED_WRITE_TOOLS = frozenset({"write_file", "str_replace"})

# AIO/E2B 계열 sandbox는 read 실패(파일 없음 포함)를 예외가 아니라 "Error: ..." 문자열로
# 바꾼다. 이 prefix로 시작하는 내용은 "확인 불가"로 취급한다 — gate는 fail-open이 되고 mark도
# 찍지 않는다.
_UNINSPECTABLE_CONTENT_PREFIX = "Error:"

_BLOCK_MESSAGE = (
    "Error: {tool_name} blocked — {path} already exists and you have not read its current version. "
    "Any write invalidates earlier reads, so re-read before every modification. "
    "Call read_file on it (a ranged read of the relevant section is enough, e.g. the last ~30 lines "
    "before an append), check what is already there, then retry."
)

# gate 확인과 도구 실행을 직렬화하는 (scope, path) 단위 lock. sandbox/file_operation_lock.py와
# 같은 WeakValueDictionary 패턴이지만 namespace는 분리한다. 도구 내부의 파일 lock은 변경만
# 보호하지만, 이 lock은 그 앞의 권한 확인까지 함께 감싼다.
_GATE_LOCKS: weakref.WeakValueDictionary[tuple[str, str], threading.Lock] = weakref.WeakValueDictionary()
_GATE_LOCKS_GUARD = threading.Lock()


def _get_gate_lock(scope: str, norm_path: str) -> threading.Lock:
    key = (scope, norm_path)
    with _GATE_LOCKS_GUARD:
        lock = _GATE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _GATE_LOCKS[key] = lock
        return lock


def _normalize_mark_path(path: str) -> str:
    return posixpath.normpath(path)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ReadBeforeWriteMiddleware(AgentMiddleware):
    """버전 gate: 현재 버전을 읽지 않은 기존 파일에 대한 쓰기를 막는다."""

    def __init__(self, content_reader: Callable[[Any, str], str] | None = None) -> None:
        super().__init__()
        self._content_reader = content_reader or read_current_file_content

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call.get("name")
        if name in _GATED_WRITE_TOOLS:
            path = self._requested_path(request)
            if path is None:
                return handler(request)
            with self._lock_for(request, path):
                blocked = self._check_write_gate(request)
                if blocked is not None:
                    # 차단된 쓰기는 ToolErrorHandlingMiddleware를 우회하므로,
                    # ToolProgressMiddleware가 분류할 수 있도록 deerflow_tool_meta를 찍는다.
                    return normalize_tool_result(blocked)
                return handler(request)
        if name in _READ_TOOLS:
            path = self._requested_path(request)
            if path is None:
                return handler(request)
            with self._lock_for(request, path):
                result = handler(request)
                self._attach_read_mark(request, result)
                return result
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        name = request.tool_call.get("name")
        if name in _GATED_WRITE_TOOLS:
            path = self._requested_path(request)
            if path is None:
                return await handler(request)
            # threading.Lock은 획득한 thread가 아닌 다른 thread에서 해제해도 되므로, worker
            # thread에서 획득하고 event loop thread에서 해제해도 안전하다.
            lock = self._lock_for(request, path)
            await asyncio.to_thread(lock.acquire)
            try:
                blocked = await asyncio.to_thread(self._check_write_gate, request)
                if blocked is not None:
                    return normalize_tool_result(blocked)
                return await handler(request)
            finally:
                lock.release()
        if name in _READ_TOOLS:
            path = self._requested_path(request)
            if path is None:
                return await handler(request)
            lock = self._lock_for(request, path)
            await asyncio.to_thread(lock.acquire)
            try:
                result = await handler(request)
                await asyncio.to_thread(self._attach_read_mark, request, result)
                return result
            finally:
                lock.release()
        return await handler(request)

    # -- locking ---------------------------------------------------------

    def _lock_for(self, request: ToolCallRequest, path: str) -> threading.Lock:
        return _get_gate_lock(self._lock_scope(request), _normalize_mark_path(path))

    @staticmethod
    def _lock_scope(request: ToolCallRequest) -> str:
        """관련 없는 agent끼리 경합하지 않도록 lock을 thread(또는 sandbox) 단위로 구분한다."""
        context = getattr(request.runtime, "context", None)
        if isinstance(context, dict):
            thread_id = context.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                return thread_id
        state = request.state
        if isinstance(state, dict):
            sandbox_state = state.get("sandbox")
            if isinstance(sandbox_state, dict):
                sandbox_id = sandbox_state.get("sandbox_id")
                if isinstance(sandbox_id, str) and sandbox_id:
                    return sandbox_id
        return "global"

    # -- gate ----------------------------------------------------------

    def _check_write_gate(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_call = request.tool_call
        path = self._requested_path(request)
        if path is None:
            return None
        try:
            current = self._content_reader(request.runtime, path)
        except FileNotFoundError:
            # write_file은 파일을 생성하고, str_replace는 자체 에러를 낸다.
            return None
        except Exception:
            logger.warning("read-before-write gate could not inspect %r; allowing the write (fail-open)", path, exc_info=True)
            return None
        if current.startswith(_UNINSPECTABLE_CONTENT_PREFIX):
            # 에러 문자열을 쓰는 sandbox read 채널(AIO/E2B)에서는 "파일 없음"과 "읽기 불가"를
            # 구분할 수 없으므로 fail-open한다 — 생성은 그대로 진행되고 실제 실패는 도구
            # 자체에서 드러난다.
            logger.debug("read-before-write gate got an error-string read for %r; allowing the write (fail-open)", path)
            return None
        norm_path = _normalize_mark_path(path)
        if self._latest_mark_hash(request.state, norm_path) == _content_hash(current):
            return None
        tool_name = str(tool_call.get("name", "write"))
        return ToolMessage(
            content=_BLOCK_MESSAGE.format(tool_name=tool_name, path=path),
            tool_call_id=str(tool_call.get("id", "")),
            name=tool_name,
            status="error",
        )

    @staticmethod
    def _requested_path(request: ToolCallRequest) -> str | None:
        args = request.tool_call.get("args") or {}
        if not isinstance(args, dict):
            return None
        path = args.get("path")
        return path if isinstance(path, str) and path else None

    @staticmethod
    def _latest_mark_hash(state: Any, norm_path: str) -> str | None:
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                continue
            mark = (message.additional_kwargs or {}).get(READ_MARK_KEY)
            if isinstance(mark, dict) and mark.get("path") == norm_path:
                mark_hash = mark.get("hash")
                return mark_hash if isinstance(mark_hash, str) else None
        return None

    # -- mark stamping ---------------------------------------------------

    def _attach_read_mark(self, request: ToolCallRequest, result: ToolMessage | Command) -> None:
        path = self._requested_path(request)
        if path is None:
            return
        message = self._extract_tool_message(result)
        if message is None or message.status == "error":
            return
        try:
            content = self._content_reader(request.runtime, path)
        except Exception:
            logger.debug("read-before-write mark skipped for %r: file not hashable", path, exc_info=True)
            return
        if content.startswith(_UNINSPECTABLE_CONTENT_PREFIX):
            logger.debug("read-before-write mark skipped for %r: error-string read channel", path)
            return
        message.additional_kwargs[READ_MARK_KEY] = {
            "path": _normalize_mark_path(path),
            "hash": _content_hash(content),
        }

    @staticmethod
    def _extract_tool_message(result: ToolMessage | Command) -> ToolMessage | None:
        if isinstance(result, ToolMessage):
            return result
        if isinstance(result, Command) and isinstance(result.update, dict):
            candidates = [m for m in result.update.get("messages", []) if isinstance(m, ToolMessage)]
            if candidates:
                return candidates[-1]
        return None
