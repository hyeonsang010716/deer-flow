import json

from langchain.tools import tool
from tavily import TavilyClient

from deerflow.config import get_app_config


def _get_tavily_client() -> TavilyClient:
    config = get_app_config().get_tool_config("web_search")
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return TavilyClient(api_key=api_key)


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """웹을 검색한다.

    Args:
        query: 검색할 질의.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 5
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    client = _get_tavily_client()
    res = client.search(query, max_results=max_results)
    normalized_results = [
        {
            "title": result["title"],
            "url": result["url"],
            "snippet": result["content"],
        }
        for result in res["results"]
    ]
    json_results = json.dumps(normalized_results, indent=2, ensure_ascii=False)
    return json_results


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """지정된 URL의 웹 페이지 내용을 가져온다.
    사용자가 직접 제공했거나 web_search, web_fetch 도구의 결과로 반환된 URL만 정확히 그대로 가져와라.
    이 도구는 비공개 Google Docs나 로그인 뒤에 있는 페이지처럼 인증이 필요한 콘텐츠에는 접근할 수 없다.
    www.가 없는 URL에 www.를 붙이지 마라.
    URL에는 반드시 schema가 포함되어야 한다: https://example.com은 유효한 URL이지만 example.com은 유효하지 않다.

    Args:
        url: 내용을 가져올 URL.
    """
    client = _get_tavily_client()
    res = client.extract([url])
    if "failed_results" in res and len(res["failed_results"]) > 0:
        return f"Error: {res['failed_results'][0]['error']}"
    elif "results" in res and len(res["results"]) > 0:
        result = res["results"][0]
        return f"# {result['title']}\n\n{result['raw_content'][:4096]}"
    else:
        return "Error: No results found"
