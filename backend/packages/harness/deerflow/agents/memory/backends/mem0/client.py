"""mem0 REST API용 동기 httpx client (v3, delete만 v1).

MemoryManager 규약이 동기이므로(DeerMem의 LLM 호출도 동기다) 이 client는 평범한
``httpx.Client``다. 테스트가 ``httpx.MockTransport``를 주입할 수 있도록
``transport``를 선택 인자로 받는다.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class Mem0APIError(RuntimeError):
    """모든 mem0 요청 실패(transport, 4xx/5xx)."""


class Mem0AuthError(Mem0APIError):
    """401 — API key가 없거나 유효하지 않다."""


class Mem0Client:
    """DeerFlow가 쓰는 mem0 엔드포인트만 감싼 얇은 wrapper."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Token {api_key}", "Accept": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise Mem0APIError(f"mem0 request failed: {e}") from e
        if resp.status_code == 401:
            raise Mem0AuthError("mem0 authentication failed (check the API key)")
        if resp.status_code >= 400:
            raise Mem0APIError(f"mem0 {method} {path} -> {resp.status_code}: {resp.text[:200]}")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise Mem0APIError(f"mem0 {method} {path} returned malformed JSON: {e}") from e

    def add_memories(
        self,
        *,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """추출 작업을 큐에 넣는다(서버 측 비동기 처리, 응답에 event_id가 담긴다)."""
        body: dict[str, Any] = {"messages": messages}
        if user_id:
            body["user_id"] = user_id
        if agent_id:
            body["agent_id"] = agent_id
        if run_id:
            body["run_id"] = run_id
        return self._request("POST", "/v3/memories/add/", json=body)

    def search_memories(
        self,
        *,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        body = {"query": query, "filters": filters, "top_k": top_k, "threshold": threshold}
        return self._request("POST", "/v3/memories/search/", json=body).get("results", [])

    def list_memories(
        self,
        *,
        filters: dict[str, Any],
        page_size: int = 200,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """페이지가 소진되거나 ``max_items``에 도달할 때까지 memory를 나열한다."""
        results: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._request(
                "POST",
                "/v3/memories/",
                params={"page": page, "page_size": page_size},
                json={"filters": filters},
            )
            results.extend(data.get("results", []))
            if not data.get("next") or (max_items is not None and len(results) >= max_items):
                return results[:max_items] if max_items is not None else results
            page += 1

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        params = {k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v}
        self._request("DELETE", "/v1/memories/", params=params)

    def ping(self) -> None:
        """시작 시 인증 확인: sentinel user id로 범위를 좁힌 1건 조회.

        sentinel 버킷은 항상 비어 있으므로 실제 데이터를 건드리지 않고 API key가
        유효한지만 확인한다.
        """
        self.list_memories(filters={"user_id": "__deerflow_startup_check__"}, page_size=1, max_items=1)
