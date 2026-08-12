"""mem0 memory 백엔드 — 상태를 갖지 않는 HTTP MemoryManager.

모든 상태(dedup, 추출, 저장)는 mem0 서버 쪽에 있다. 이 백엔드는 queue도 watermark도
cache도 두지 않으므로 multi-worker Gateway 배포에서도 안전하다. 식별자는 1:1로
대응한다: (user_id, agent_name) -> mem0 (user_id, agent_id), thread_id -> mem0 run_id.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

# ABC 계약 — 이 백엔드 폴더에서 유일하게 허용되는 `from deerflow` import.
from deerflow.agents.memory.manager import MemoryManager, MemoryManagerError

from .client import Mem0APIError, Mem0Client
from .config import Mem0Config
from .message_filtering import extract_message_text, filter_messages_for_memory

logger = logging.getLogger(__name__)

_ROLE_MAP = {"human": "user", "ai": "assistant"}


def _build_filters(
    *,
    user_id: str | None = None,
    agent_name: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """주어진 식별자 조각들로 mem0 ``filters`` 객체를 만든다.

    entity id가 하나도 없으면 None을 반환한다(mem0는 최소 하나를 요구한다).
    """
    parts: list[dict[str, Any]] = []
    if user_id:
        parts.append({"user_id": user_id})
    if agent_name:
        parts.append({"agent_id": agent_name})
    if run_id:
        parts.append({"run_id": run_id})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"AND": parts}


def _to_fact(record: dict[str, Any]) -> dict[str, Any]:
    """mem0 record를 host(agents/memory/tools.py)가 소비하는 백엔드 중립 fact 형태로
    변환한다: id/content/category/confidence/createdAt/source.
    mem0의 관련도 점수를 confidence로 그대로 쓴다."""
    categories = record.get("categories") or []
    metadata = record.get("metadata") or {}
    return {
        "id": str(record.get("id", "")),
        "content": str(record.get("memory", "")),
        "category": str(categories[0]) if categories else "context",
        "confidence": float(record.get("score") or 0.0),
        "createdAt": str(record.get("created_at", "")),
        "source": str(metadata.get("source", "")),
    }


class Mem0Manager(MemoryManager):
    """mem0 Platform API(또는 호환 서버)를 백엔드로 쓰는 MemoryManager."""

    # 아래에서 search()를 override하므로 플래그는 True여야 한다(계약 불변식).
    # 이 값은 memory mode="tool"도 함께 활성화한다.
    supports_search: ClassVar[bool] = True
    # mem0는 add()로 전체 대화에서 fact를 추출하고 중복을 제거한다. fact CRUD 훅은
    # 의도적으로 지원하지 않으므로, tool 모드에서도 passive 쓰기를 유지한 채
    # query 기반 search만 노출한다.
    requires_passive_writes_in_tool_mode: ClassVar[bool] = True

    _config: Mem0Config = PrivateAttr()
    _client: Any = PrivateAttr(default=None)  # Mem0Client. 테스트는 fake를 주입한다.

    def model_post_init(self, __context: Any) -> None:
        self._config = Mem0Config.from_backend_config(self.backend_config)
        self._client = Mem0Client(
            base_url=self._config.base_url,
            api_key=self._config.resolve_api_key(),
            timeout_seconds=self._config.timeout_seconds,
        )

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> Mem0Manager:
        """manager를 만든다. startup policy가 ``fail_fast``면 ping으로 인증을 확인한다."""
        mgr = cls(backend_config=backend_config, mode=mode)
        if mgr._config.startup_policy == "fail_fast":
            mgr._client.ping()
        return mgr

    def close(self) -> None:
        """내부 HTTP connection pool을 해제한다."""
        self._client.close()

    # ── 오류 정책 ────────────────────────────────────────────────────────
    def _read_or_fallback(self, fallback: Any, fn: Any) -> Any:
        try:
            return fn()
        except Mem0APIError as e:
            if self._config.read_policy == "fail_open":
                logger.warning("mem0 read failed (%s); continuing without memory", e)
                return fallback
            raise MemoryManagerError(f"mem0 read failed: {e}") from e

    def _write_or_drop(self, fn: Any) -> None:
        try:
            fn()
        except Mem0APIError as e:
            if self._config.write_policy == "log_and_drop":
                logger.warning("mem0 write failed (%s); dropping update", e)
                return
            raise MemoryManagerError(f"mem0 write failed: {e}") from e

    # ── Tier 1: 쓰기 ─────────────────────────────────────────────────────
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """필터링한 대화를 mem0에 넘겨 서버 측 추출을 맡긴다.

        Fire-and-forget 방식이다. mem0가 비동기로 처리하며 응답의 event_id는 polling하지
        않는다. ``thread_id``는 mem0 ``run_id``에 대응하므로 "entity id가 최소 하나"라는
        mem0 요구 조건을 항상 만족한다.
        """
        kept = filter_messages_for_memory(messages)
        payload = [{"role": _ROLE_MAP[getattr(m, "type", "")], "content": extract_message_text(m).strip()} for m in kept if getattr(m, "type", "") in _ROLE_MAP]
        payload = [p for p in payload if p["content"]]
        if not payload:
            return
        self._write_or_drop(
            lambda: self._client.add_memories(
                messages=payload,
                user_id=user_id,
                agent_id=agent_name,
                run_id=thread_id,
            )
        )

    async def aadd(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.add,
            thread_id,
            messages,
            agent_name=agent_name,
            user_id=user_id,
            trace_id=trace_id,
        )

    # ── Tier 1: 읽기-주입 ────────────────────────────────────────────────
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """query 없는 recall. 규약상 현재 query가 전달되지 않으므로 버킷에서 가장 최근
        memory(top_k)를 주입한다. query 기반 recall은 mode="tool"의 search()로 쓴다."""
        filters = _build_filters(user_id=user_id, agent_name=agent_name, run_id=thread_id)
        if filters is None:
            return ""
        top_k = self._config.top_k
        records = self._read_or_fallback(
            [],
            lambda: self._client.list_memories(
                filters=filters,
                page_size=min(top_k, 200),
                max_items=top_k,
            ),
        )
        budget = self._config.max_injection_chars
        seen: set[str] = set()
        lines: list[str] = []
        used = 0
        shortest_line: int | None = None
        for record in records:
            rid = record.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            text = str(record.get("memory") or "").strip()
            if not text:
                continue
            line = f"- {text}"
            line_len = len(line)
            shortest_line = line_len if shortest_line is None else min(shortest_line, line_len)
            # 항목 경계에서만 자른다. 남은 예산(줄바꿈 1자 포함)에 통째로 들어가는
            # memory만 담아, 주입 결과가 항목 중간에서 끊긴 조각으로 끝나지 않게 한다.
            # 너무 큰 항목은 건너뛴다. 뒤쪽의 짧은 항목은 여전히 들어갈 수 있다.
            added = line_len if not lines else line_len + 1
            if used + added > budget:
                continue
            lines.append(line)
            used += added
        context = "\n".join(lines)
        if not context and shortest_line is not None:
            # recall된 memory가 전부 설정된 예산보다 길었다. 부분 fact를 주입하는 대신
            # 항목 경계 보장을 지키고 설정 문제를 warning으로 드러낸다.
            logger.warning(
                "max_injection_chars=%d is smaller than the shortest recalled memory (%d chars); returning empty context",
                budget,
                shortest_line,
            )
        return context

    async def aget_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.get_context,
            user_id,
            agent_name=agent_name,
            thread_id=thread_id,
        )

    # ── Tier 2: 검색 ─────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = _build_filters(user_id=user_id, agent_name=agent_name)
        if filters is None:
            return []
        if category:
            parts = filters["AND"] if "AND" in filters else [filters]
            filters = {"AND": [*parts, {"categories": {"contains": category}}]}
        results = self._read_or_fallback(
            [],
            lambda: self._client.search_memories(
                query=query,
                filters=filters,
                top_k=top_k,
                threshold=self._config.score_threshold,
            ),
        )
        return [_to_fact(r) for r in results]

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.search,
            query,
            top_k,
            user_id=user_id,
            agent_name=agent_name,
            category=category,
        )

    # ── Tier 2: 관리 ─────────────────────────────────────────────────────
    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        filters = _build_filters(user_id=user_id, agent_name=agent_name)
        if filters is None:
            return {"facts": []}
        records = self._read_or_fallback([], lambda: self._client.list_memories(filters=filters))
        return {"facts": [_to_fact(r) for r in records]}

    def export_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        return self.get_memory(user_id=user_id, agent_name=agent_name)

    def clear_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """버킷을 비운다. agent_name=None이면 해당 user의 memory 전체를,
        agent를 명시하면 그 agent의 버킷만 비운다."""
        if not user_id and not agent_name:
            return {"facts": []}
        self._write_or_drop(lambda: self._client.delete_all_memories(user_id=user_id, agent_id=agent_name, run_id=None))
        return {"facts": []}

    def delete_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        if not user_id and not agent_name:
            return None
        self._write_or_drop(lambda: self._client.delete_all_memories(user_id=user_id, agent_id=agent_name, run_id=None))
