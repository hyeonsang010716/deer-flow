"""
Image Search Tool - 이미지 생성 시 참고할 이미지를 DuckDuckGo로 검색한다.
"""

import json
import logging

from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)


def _search_images(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    size: str | None = None,
    color: str | None = None,
    type_image: str | None = None,
    layout: str | None = None,
    license_image: str | None = None,
) -> list[dict]:
    """
    DuckDuckGo로 이미지 검색을 실행한다.

    Args:
        query: 검색 키워드
        max_results: 최대 결과 수
        region: 검색 지역
        safesearch: safe search 수준
        size: 이미지 크기(Small/Medium/Large/Wallpaper)
        color: 색상 필터
        type_image: 이미지 타입(photo/clipart/gif/transparent/line)
        layout: 레이아웃(Square/Tall/Wide)
        license_image: 라이선스 필터

    Returns:
        검색 결과 list
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs library not installed. Run: pip install ddgs")
        return []

    ddgs = DDGS(timeout=30)

    try:
        kwargs = {
            "region": region,
            "safesearch": safesearch,
            "max_results": max_results,
        }

        if size:
            kwargs["size"] = size
        if color:
            kwargs["color"] = color
        if type_image:
            kwargs["type_image"] = type_image
        if layout:
            kwargs["layout"] = layout
        if license_image:
            kwargs["license_image"] = license_image

        results = ddgs.images(query, **kwargs)
        return list(results) if results else []

    except Exception as e:
        logger.error(f"Failed to search images: {e}")
        return []


@tool("image_search", parse_docstring=True)
def image_search_tool(
    query: str,
    max_results: int = 5,
    size: str | None = None,
    type_image: str | None = None,
    layout: str | None = None,
) -> str:
    """온라인에서 이미지를 검색한다. 인물, 초상, 사물, 장면 등 시각적 정확도가 필요한 대상의 참고 이미지를 찾으려면 이미지 생성 전에 이 도구를 사용하라.

    **사용 시점:**
    - 캐릭터/인물 이미지 생성 전: 비슷한 포즈, 표정, 스타일을 검색한다
    - 특정 사물/제품 이미지 생성 전: 정확한 시각적 레퍼런스를 검색한다
    - 장면/장소 이미지 생성 전: 건축이나 환경 레퍼런스를 검색한다
    - 패션/의상 이미지 생성 전: 스타일과 디테일 레퍼런스를 검색한다

    반환된 이미지 URL을 이미지 생성의 참고 이미지로 사용하면 품질이 크게 향상된다.

    Args:
        query: 찾으려는 이미지를 설명하는 검색 키워드. 구체적일수록 결과가 좋다(예: 그냥 "woman" 대신 "Japanese woman street photography 1990s").
        max_results: 반환할 최대 이미지 수. 기본값은 5.
        size: 이미지 크기 필터. 옵션: "Small", "Medium", "Large", "Wallpaper". 참고 이미지에는 "Large"를 쓰라.
        type_image: 이미지 타입 필터. 옵션: "photo", "clipart", "gif", "transparent", "line". 사실적인 레퍼런스에는 "photo"를 쓰라.
        layout: 레이아웃 필터. 옵션: "Square", "Tall", "Wide". 생성 목적에 맞춰 고르라.
    """
    config = get_app_config().get_tool_config("image_search")

    # config에 설정되어 있으면 max_results를 덮어쓴다
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results", max_results)

    results = _search_images(
        query=query,
        max_results=max_results,
        size=size,
        type_image=type_image,
        layout=layout,
    )

    if not results:
        return json.dumps({"error": "No images found", "query": query}, ensure_ascii=False)

    normalized_results = [
        {
            "title": r.get("title", ""),
            "image_url": r.get("image", ""),
            "thumbnail_url": r.get("thumbnail", ""),
        }
        for r in results
    ]

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
        "usage_hint": "Use the 'image_url' values as reference images in image generation. Download them first if needed.",
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
