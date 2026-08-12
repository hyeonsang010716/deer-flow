"""GroundRoute community web search·fetch tool 모음.

GroundRoute(https://groundroute.ai)는 meta search 레이어다. 검색 엔진 여섯 개(Serper, Brave,
Exa, Tavily, Firecrawl, Perplexity) 앞에 API 하나를 둔다. 각 query를 품질 기준을 통과하는
가장 저렴한 엔진으로 라우팅하고 반복 query는 cache하므로, 엔진 하나가 죽어도 대량 research
run이 계속 동작하고 비용은 단일 엔진 직접 호출보다 더 들지 않는다. 요금은 gain-share 방식으로,
cache 절감분의 절반가량을 호출자가 가져간다.

이 모듈은 자체 완결적이다(httpx만 쓰고 GroundRoute SDK는 쓰지 않는다). /v1/search 요청과 응답
매핑은 GroundRoute MCP server 및 검증된 Langflow 컴포넌트와 동일하다:
  results[] = {url, title, snippet, content, source_engine, published_at}

`web_search`는 {title, url, snippet, source_engine}의 정규화된 JSON 목록을 반환한다.
`web_fetch`는 GroundRoute mode=page로 URL 하나를 읽어 추출된 텍스트를 반환한다.
"""

import json
import logging
import os

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_GROUNDROUTE_ENDPOINT = "https://api.groundroute.ai/v1/search"
_DEFAULT_MAX_RESULTS = 5
# GroundRoute는 서버 쪽에서 max_results를 1-50으로 제한한다. 동일하게 맞추려고 여기서도 제한한다.
_MAX_RESULTS_CAP = 50
_TIMEOUT_S = 30.0
_FETCH_SNIPPET_LIMIT = 4096
# key 누락 경고는 tool("web_search" / "web_fetch")당 최대 한 번만 낸다.
_api_key_warned: set[str] = set()


def _get_api_key(tool_name: str) -> str | None:
    """해당 tool의 config 블록에서 GroundRoute key를 찾고, 없으면 환경 변수를 본다.

    `tool_name`은 읽을 config 섹션(web_search 또는 web_fetch)이다. fetch는 GroundRoute를 쓰고
    search는 다른 엔진을 쓰는 구성에서도 올바른 key를 읽게 한다. tool 이름을 받는
    serper/exa/firecrawl과 같은 방식이다.
    """
    config = get_app_config().get_tool_config(tool_name)
    if config is not None:
        api_key = (config.model_extra or {}).get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    return os.getenv("GROUNDROUTE_API_KEY")


def _coerce_max_results(value: object, *, default: int = _DEFAULT_MAX_RESULTS) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid GroundRoute max_results=%r; using default %s", value, default)
        coerced = default
    return max(1, min(coerced, _MAX_RESULTS_CAP))


def _missing_key_error(tool_name: str, **context: str) -> str:
    if tool_name not in _api_key_warned:
        _api_key_warned.add(tool_name)
        logger.warning(
            "GroundRoute API key is not set for '%s'. Set GROUNDROUTE_API_KEY in your environment or provide api_key in config.yaml. Get a free key at https://groundroute.ai/keys",
            tool_name,
        )
    return json.dumps({"error": "GROUNDROUTE_API_KEY is not configured", **context}, ensure_ascii=False)


def _post_search(api_key: str, body: dict) -> dict:
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        response = client.post(
            _GROUNDROUTE_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    response.raise_for_status()
    return response.json()


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int | None = None) -> str:
    """GroundRoute를 사용해 web에서 정보를 검색한다.

    GroundRoute는 query를 검색 엔진 여섯 개에 라우팅해 선택된 엔진의 결과 집합을 반환하며,
    엔진 하나를 쓸 수 없으면 다른 엔진으로 failover한다.

    Args:
        query: 찾으려는 내용을 설명하는 검색 키워드. 구체적일수록 결과가 좋다.
        max_results: 반환할 최대 검색 결과 수. 생략하면 설정값(기본 5)을 쓴다. 1-50으로 제한된다.
    """
    # 호출자가 준 max_results를 우선한다. 생략된 경우에만 config로 fallback한다.
    if max_results is None:
        config = get_app_config().get_tool_config("web_search")
        if config is not None:
            max_results = (config.model_extra or {}).get("max_results")
    count = _DEFAULT_MAX_RESULTS if max_results is None else _coerce_max_results(max_results)

    api_key = _get_api_key("web_search")
    if not api_key:
        return _missing_key_error("web_search", query=query)

    try:
        data = _post_search(api_key, {"query": query, "max_results": count})
    except httpx.HTTPStatusError as e:
        logger.error("GroundRoute API returned HTTP %s: %s", e.response.status_code, e.response.text)
        return json.dumps(
            {"error": f"GroundRoute API error: HTTP {e.response.status_code}", "query": query},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("GroundRoute search failed: %s: %s", type(e).__name__, e)
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)

    results = data.get("results") or []
    if not results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    normalized_results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("snippet", ""),
            "source_engine": r.get("source_engine", ""),
        }
        for r in results
    ]
    return json.dumps(normalized_results, indent=2, ensure_ascii=False)


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """GroundRoute를 통해 주어진 URL의 web page 내용을 가져온다.
    사용자가 직접 제공했거나 web_search와 web_fetch 도구의 결과로 반환된 URL만 정확히 그대로 가져와라.
    이 도구는 비공개 Google Docs나 로그인 장벽 뒤의 page처럼 인증이 필요한 콘텐츠에는 접근할 수 없다.
    www.가 없는 URL에 www.를 임의로 붙이지 마라.
    URL에는 schema를 반드시 포함해야 한다. https://example.com은 유효하지만 example.com은 유효하지 않다.

    Args:
        url: 내용을 가져올 URL.
    """
    api_key = _get_api_key("web_fetch")
    if not api_key:
        return _missing_key_error("web_fetch", url=url)

    try:
        data = _post_search(api_key, {"query": url, "mode": "page", "max_results": 1})
    except httpx.HTTPStatusError as e:
        logger.error("GroundRoute fetch returned HTTP %s: %s", e.response.status_code, e.response.text)
        return f"Error: GroundRoute API error: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error("GroundRoute fetch failed: %s: %s", type(e).__name__, e)
        return f"Error: {e}"

    results = data.get("results") or []
    if not results:
        return "Error: No results found"

    result = results[0]
    content = result.get("content") or result.get("snippet") or ""
    title = result.get("title", "")
    return f"# {title}\n\n{content[:_FETCH_SNIPPET_LIMIT]}"
