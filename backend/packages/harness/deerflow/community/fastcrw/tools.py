import json
import os

from firecrawl import FirecrawlApp
from langchain.tools import tool

from deerflow.community.url_safety import validate_public_http_url
from deerflow.config import get_app_config

# fastCRW는 Firecrawl 호환 웹 데이터 엔진이다(단일 Rust 바이너리, self-host 또는 클라우드).
# REST API가 Firecrawl과 호환되므로 이 provider는 Firecrawl client를 그대로 재사용하고
# base URL만 바꾼다. 기본값은 관리형 클라우드 서비스를 가리킨다. self-host를 쓰려면
# tool 설정의 `base_url`을 덮어쓰거나 CRW_API_URL을 설정한다.
DEFAULT_BASE_URL = "https://fastcrw.com/api"


def _get_fastcrw_client(tool_name: str = "web_search") -> FirecrawlApp:
    config = get_app_config().get_tool_config(tool_name)
    api_key = None
    base_url = None
    if config is not None:
        if "api_key" in config.model_extra:
            api_key = config.model_extra.get("api_key")
        if "base_url" in config.model_extra:
            base_url = config.model_extra.get("base_url")
    if api_key is None:
        api_key = os.getenv("CRW_API_KEY")
    if base_url is None:
        base_url = os.getenv("CRW_API_URL", DEFAULT_BASE_URL)
    return FirecrawlApp(api_key=api_key, api_url=base_url)  # type: ignore[arg-type]


def _get_tool_config_extra(tool_name: str) -> dict:
    config = get_app_config().get_tool_config(tool_name)
    return dict(config.model_extra or {}) if config is not None else {}


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """웹을 검색한다.

    Args:
        query: 검색할 질의.
    """
    try:
        config = get_app_config().get_tool_config("web_search")
        max_results = 5
        if config is not None:
            max_results = config.model_extra.get("max_results", max_results)

        client = _get_fastcrw_client("web_search")
        result = client.search(query, limit=max_results)

        # result.web은 SearchResultWeb 객체의 리스트다
        web_results = result.web or []
        normalized_results = [
            {
                "title": getattr(item, "title", "") or "",
                "url": getattr(item, "url", "") or "",
                "snippet": getattr(item, "description", "") or "",
            }
            for item in web_results
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
        cfg = _get_tool_config_extra("web_fetch")
        allow_private_addresses = _coerce_bool(cfg.get("allow_private_addresses"), False)
        url_error = validate_public_http_url(url, allow_private_addresses=allow_private_addresses)
        if url_error:
            return url_error
        client = _get_fastcrw_client("web_fetch")
        result = client.scrape(url, formats=["markdown"])

        markdown_content = result.markdown or ""
        metadata = result.metadata
        title = metadata.title if metadata and metadata.title else "Untitled"

        if not markdown_content:
            return "Error: No content found"
    except Exception as e:
        return f"Error: {str(e)}"

    return f"# {title}\n\n{markdown_content[:4096]}"
