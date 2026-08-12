import json

from exa_py import Exa
from langchain.tools import tool

from deerflow.config import get_app_config


def _get_exa_client(tool_name: str = "web_search") -> Exa:
    config = get_app_config().get_tool_config(tool_name)
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return Exa(api_key=api_key)


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """웹을 검색한다.

    Args:
        query: 검색할 질의.
    """
    try:
        config = get_app_config().get_tool_config("web_search")
        max_results = 5
        search_type = "auto"
        contents_max_characters = 1000
        if config is not None:
            max_results = config.model_extra.get("max_results", max_results)
            search_type = config.model_extra.get("search_type", search_type)
            contents_max_characters = config.model_extra.get("contents_max_characters", contents_max_characters)

        client = _get_exa_client()
        res = client.search(
            query,
            type=search_type,
            num_results=max_results,
            contents={"highlights": {"max_characters": contents_max_characters}},
        )

        normalized_results = [
            {
                "title": result.title or "",
                "url": result.url or "",
                "snippet": "\n".join(result.highlights) if result.highlights else "",
            }
            for result in res.results
        ]
        json_results = json.dumps(normalized_results, indent=2, ensure_ascii=False)
        return json_results
    except Exception as e:
        return f"Error: {str(e)}"


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
    try:
        client = _get_exa_client("web_fetch")
        res = client.get_contents([url], text={"max_characters": 4096})

        if res.results:
            result = res.results[0]
            title = result.title or "Untitled"
            text = result.text or ""
            return f"# {title}\n\n{text[:4096]}"
        else:
            return "Error: No results found"
    except Exception as e:
        return f"Error: {str(e)}"
