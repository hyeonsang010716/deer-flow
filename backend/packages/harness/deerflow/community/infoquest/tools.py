from langchain.tools import tool

from deerflow.config import get_app_config
from deerflow.utils.readability import ReadabilityExtractor

from .infoquest_client import InfoQuestClient

readability_extractor = ReadabilityExtractor()


def _get_infoquest_client() -> InfoQuestClient:
    search_config = get_app_config().get_tool_config("web_search")
    search_time_range = -1
    if search_config is not None and "search_time_range" in search_config.model_extra:
        search_time_range = search_config.model_extra.get("search_time_range")

    fetch_config = get_app_config().get_tool_config("web_fetch")
    fetch_time = -1
    if fetch_config is not None and "fetch_time" in fetch_config.model_extra:
        fetch_time = fetch_config.model_extra.get("fetch_time")
    fetch_timeout = -1
    if fetch_config is not None and "timeout" in fetch_config.model_extra:
        fetch_timeout = fetch_config.model_extra.get("timeout")
    navigation_timeout = -1
    if fetch_config is not None and "navigation_timeout" in fetch_config.model_extra:
        navigation_timeout = fetch_config.model_extra.get("navigation_timeout")

    image_search_config = get_app_config().get_tool_config("image_search")
    image_search_time_range = -1
    if image_search_config is not None and "image_search_time_range" in image_search_config.model_extra:
        image_search_time_range = image_search_config.model_extra.get("image_search_time_range")
    image_size = "i"
    if image_search_config is not None and "image_size" in image_search_config.model_extra:
        image_size = image_search_config.model_extra.get("image_size")

    return InfoQuestClient(
        search_time_range=search_time_range,
        fetch_timeout=fetch_timeout,
        fetch_navigation_timeout=navigation_timeout,
        fetch_time=fetch_time,
        image_search_time_range=image_search_time_range,
        image_size=image_size,
    )


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """web을 검색한다.

    Args:
        query: 검색할 query.
    """

    client = _get_infoquest_client()
    return client.web_search(query)


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """주어진 URL의 web page 내용을 가져온다.
    사용자가 직접 제공했거나 web_search와 web_fetch 도구의 결과로 반환된 URL만 정확히 그대로 가져와라.
    이 도구는 비공개 Google Docs나 로그인 장벽 뒤의 page처럼 인증이 필요한 콘텐츠에는 접근할 수 없다.
    www.가 없는 URL에 www.를 임의로 붙이지 마라.
    URL에는 schema를 반드시 포함해야 한다. https://example.com은 유효하지만 example.com은 유효하지 않다.

    Args:
        url: 내용을 가져올 URL.
    """
    client = _get_infoquest_client()
    result = client.fetch(url)
    if result.startswith("Error: "):
        return result
    article = readability_extractor.extract_article(result)
    return article.to_markdown()[:4096]


@tool("image_search", parse_docstring=True)
def image_search_tool(query: str) -> str:
    """온라인에서 이미지를 검색한다. 인물, 초상, 사물, 장면 등 시각적 정확도가 필요한 대상의 참고 이미지를 찾으려면 이미지 생성 전에 이 도구를 사용하라.

    **사용 시점:**
    - 캐릭터/인물 이미지 생성 전: 비슷한 포즈, 표정, 스타일을 검색한다
    - 특정 사물/제품 이미지 생성 전: 정확한 시각적 레퍼런스를 검색한다
    - 장면/장소 이미지 생성 전: 건축이나 환경 레퍼런스를 검색한다
    - 패션/의상 이미지 생성 전: 스타일과 디테일 레퍼런스를 검색한다

    반환된 이미지 URL을 이미지 생성의 참고 이미지로 사용하면 품질이 크게 향상된다.

    Args:
        query: 이미지를 검색할 query.
    """
    client = _get_infoquest_client()
    return client.image_search(query)
