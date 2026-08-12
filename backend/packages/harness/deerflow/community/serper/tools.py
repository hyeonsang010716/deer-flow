"""
Serper(Google Search API) 기반 웹/이미지 검색 도구.

Serper는 JSON API로 실시간 Google Search와 Google Images 결과를 제공한다.
API key가 필요하며 https://serper.dev 에서 발급받는다.
"""

import json
import logging
import os
from ipaddress import IPv4Address, ip_address
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
_SERPER_IMAGES_ENDPOINT = "https://google.serper.dev/images"
_SERPER_MAX_RESULTS = 10
_api_key_warned: set[str] = set()


def _get_api_key(tool_name: str) -> str | None:
    config = get_app_config().get_tool_config(tool_name)
    if config is not None:
        api_key = config.model_extra.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    env_key = os.getenv("SERPER_API_KEY")
    if isinstance(env_key, str) and env_key.strip():
        return env_key.strip()
    return None


def _coerce_max_results(value: object, default: int = 5, max_allowed: int = _SERPER_MAX_RESULTS) -> int:
    """설정/파라미터 입력을 상한이 있는 양의 결과 개수로 변환한다."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    if count <= 0:
        return default
    return min(count, max_allowed)


def _missing_key_error(query: str, tool_name: str) -> str:
    if tool_name not in _api_key_warned:
        _api_key_warned.add(tool_name)
        logger.warning("Serper API key is not set for '%s'. Set SERPER_API_KEY in your environment or provide api_key in config.yaml. Sign up at https://serper.dev", tool_name)
    return json.dumps(
        {"error": "SERPER_API_KEY is not configured", "query": query},
        ensure_ascii=False,
    )


def _unexpected_format_error(query: str) -> str:
    return json.dumps(
        {"error": "Serper returned an unexpected response format", "query": query},
        ensure_ascii=False,
    )


def _response_items(data: dict, field: str, query: str) -> tuple[list[dict] | None, str | None]:
    items = data.get(field)
    # 필드가 없거나 null이면 잘못된 payload가 아니라 "결과 없음"으로 본다.
    # (일부 API는 이를 ``{"organic": null}`` 로 표현한다.)
    if items is None:
        return [], None
    if not isinstance(items, list):
        logger.error("Serper returned unexpected '%s' payload type: %s", field, type(items).__name__)
        return None, _unexpected_format_error(query)
    return [item for item in items if isinstance(item, dict)], None


def _clean_query(query: str) -> str:
    """원본 질의를 Serper에 실제로 보낼 값으로 정규화한다."""
    query = query.strip()
    if len(query) > 500:
        query = query[:500]
    return query


def _decode_ipv4(host: str) -> IPv4Address | None:
    """``ip_address``가 거부하는 난독화된 IPv4 리터럴을 디코딩한다.

    많은 HTTP client가 쓰는 관대한 ``inet_aton`` 파싱을 흉내내어 정수
    (``2130706433``), 16진(``0x7f000001``), 8진(``0177.0.0.1``) 표기를 인식한다.
    host가 주소로 해석되면 ``IPv4Address``를, 아니면 ``None``을 반환한다.
    (``cafe.com`` 같은 실제 도메인은 디코딩에 실패하므로 호출자가 host로 다룬다.)
    """
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values: list[int] = []
    for part in parts:
        if not part:
            return None
        try:
            if part.startswith(("0x", "0X")):
                values.append(int(part, 16))
            elif part.startswith("0") and len(part) > 1:
                values.append(int(part, 8))
            else:
                values.append(int(part, 10))
        except ValueError:
            return None

    *leading, last = values
    for value in leading:
        if not 0 <= value <= 0xFF:
            return None
    max_last = (1 << (8 * (4 - len(leading)))) - 1
    if not 0 <= last <= max_last:
        return None

    result = 0
    for value in leading:
        result = (result << 8) | value
    result = (result << (8 * (4 - len(leading)))) | last
    return ip_address(result)


def _is_url_present(value: object) -> bool:
    """*value*가 비어 있지 않은 URL 문자열이면 ``True``를 반환한다.

    필드가 *없어서* 교차 fallback 대상인 경우와, *있었지만* SSRF guard에 걸러진
    경우를 구분하는 데 쓴다. 후자는 상대 필드로 대체하지 않고 빈 값으로 남겨야 한다.
    """
    return isinstance(value, str) and bool(value.strip())


def _safe_public_url(value: object) -> str:
    """``value``가 안전한 공개 http(s) URL일 때만 그대로 반환하고, 아니면 ""를 반환한다.

    best-effort SSRF guard다. http(s)가 아닌 scheme, ``localhost``, private/non-global
    IP 리터럴(난독화된 10/16/8진 표기 포함)을 거부한다. URL 문자열만 검사하므로
    내부 IP로 resolve되는 공개 hostname(DNS rebinding 등)은 잡지 못한다.
    이 URL을 실제로 다운로드하는 consumer는 fetch 시점에 resolve된 IP를 다시 검증해야 한다.
    """
    if not isinstance(value, str):
        return ""
    url = value.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        return ""

    # 끝의 점 하나(FQDN root label)를 제거한다. ``localhost.`` 와 ``127.0.0.1.`` 은
    # 흔한 resolver에서 loopback으로 해석되지만, 그대로 두면 아래 localhost/IP 검사를 빠져나간다.
    host = parsed.hostname.lower().rstrip(".")
    if not host:
        return ""
    if host == "localhost" or host.endswith(".localhost"):
        return ""

    try:
        ip = ip_address(host)
    except ValueError:
        ip = _decode_ipv4(host)
        if ip is None:
            return url
    return url if ip.is_global else ""


def _serper_post(endpoint: str, api_key: str, query: str, max_results: int) -> tuple[dict | None, str | None]:
    """Serper endpoint로 POST 요청을 보낸다.

    ``query``는 :func:`_clean_query`로 이미 정규화되어 있다고 가정한다.

    ``(data, error_json)`` 튜플을 반환한다. 성공하면 ``data``는 파싱된 JSON 응답이고
    ``error_json``은 ``None``이다. 실패하면 ``data``는 ``None``이고 ``error_json``은
    그대로 반환 가능한 직렬화된 구조화 에러다.
    """
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": max_results}

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            logger.error("Serper returned an unexpected payload type: %s", type(data).__name__)
            return None, _unexpected_format_error(query)
        return data, None
    except httpx.HTTPStatusError as e:
        resp_text = (e.response.text or "")[:500]
        logger.error("Serper API returned HTTP %s: %s", e.response.status_code, resp_text)
        return None, json.dumps(
            {"error": f"Serper API error: HTTP {e.response.status_code}", "query": query},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("Serper request failed: %s: %s", type(e).__name__, str(e)[:500])
        return None, json.dumps({"error": str(e)[:500], "query": query}, ensure_ascii=False)


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Serper를 통해 Google Search로 웹에서 정보를 검색한다.

    Args:
        query: 찾으려는 내용을 설명하는 검색 키워드. 구체적으로 쓸수록 결과가 좋아진다.
        max_results: 반환할 최대 검색 결과 개수. 기본값은 5이고 최대 10으로 제한된다.
    """
    config = get_app_config().get_tool_config("web_search")
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results", max_results)
    max_results = _coerce_max_results(max_results)
    query = _clean_query(query)

    api_key = _get_api_key("web_search")
    if not api_key:
        return _missing_key_error(query, "web_search")

    data, error_json = _serper_post(_SERPER_SEARCH_ENDPOINT, api_key, query, max_results)
    if error_json is not None:
        return error_json

    organic, error_json = _response_items(data, "organic", query)
    if error_json is not None:
        return error_json
    if not organic:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    # 검색 결과 링크는 _safe_public_url을 거치지 않고 그대로 반환한다.
    # image_search의 이미지 URL과 달리 이 도구가 직접 fetch/다운로드하지 않고
    # 모델이 읽을 인용으로만 노출되기 때문이다.
    normalized_results = [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "content": r.get("snippet", ""),
        }
        for r in organic[:max_results]
    ]

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


@tool("image_search", parse_docstring=True)
def image_search_tool(query: str, max_results: int = 5) -> str:
    """Serper를 통해 Google Images로 온라인 이미지를 검색한다. 인물, 초상, 사물, 장면 등 시각적 정확도가 필요한 콘텐츠의 레퍼런스 이미지를 찾으려면 이미지를 생성하기 전에 반드시 이 도구를 사용하라.

    반환된 이미지 URL은 이미지 생성 시 레퍼런스 이미지로 사용할 수 있으며, 품질을 크게 향상시킨다.

    Args:
        query: 찾으려는 이미지를 설명하는 검색 키워드. 구체적으로 쓸수록 결과가 좋아진다(예: 그냥 "woman"이 아니라 "Japanese woman street photography 1990s").
        max_results: 반환할 최대 이미지 개수. 기본값은 5이고 최대 10으로 제한된다.
    """
    config = get_app_config().get_tool_config("image_search")
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results", max_results)
    max_results = _coerce_max_results(max_results)
    query = _clean_query(query)

    api_key = _get_api_key("image_search")
    if not api_key:
        return _missing_key_error(query, "image_search")

    data, error_json = _serper_post(_SERPER_IMAGES_ENDPOINT, api_key, query, max_results)
    if error_json is not None:
        return error_json

    images, error_json = _response_items(data, "images", query)
    if error_json is not None:
        return error_json
    if not images:
        return json.dumps({"error": "No images found", "query": query}, ensure_ascii=False)

    normalized_results = []
    for r in images:
        raw_image = r.get("imageUrl")
        raw_thumb = r.get("thumbnailUrl")
        # 비용이 적지 않은 SSRF guard를 필드당 두 번이 아니라 한 번만 평가한다.
        safe_image = _safe_public_url(raw_image)
        safe_thumb = _safe_public_url(raw_thumb)
        # 상대 필드가 *없을 때만* 교차 fallback한다. 있었지만 SSRF 필터에 걸린 필드는
        # 상대 값으로 대체하지 않고 빈 값으로 남긴다. 그래야 버려진 고해상도 URL이
        # preview로 둔갑하는 일(그 반대도)이 없고, 호출자가 의존하는
        # 고해상도/preview 계약이 유지된다.
        image_url = safe_image or (safe_thumb if not _is_url_present(raw_image) else "")
        thumbnail_url = safe_thumb or (safe_image if not _is_url_present(raw_thumb) else "")
        if not image_url and not thumbnail_url:
            continue
        normalized_results.append(
            {
                "title": r.get("title", ""),
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
            }
        )
        if len(normalized_results) >= max_results:
            break

    if not normalized_results:
        return json.dumps({"error": "No safe image URLs found", "query": query}, ensure_ascii=False)

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
        "usage_hint": "Use the 'image_url' values as reference images in image generation. Download them first if needed.",
    }
    return json.dumps(output, indent=2, ensure_ascii=False)
