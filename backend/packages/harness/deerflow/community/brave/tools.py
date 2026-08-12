"""
Brave Search API 기반 웹/이미지 검색 도구.

Brave Search는 독립적인 검색 인덱스의 웹/이미지 결과를 REST API로 제공한다.
API key가 필요하며 https://brave.com/search/api/ 에서 발급받는다.

DDGS aggregator로 결과를 스크래핑하는 DuckDuckGo의 ``backend: brave`` 옵션과 달리,
이 provider는 공식 Brave Search API를 직접 호출한다. 구조화된 결과, 인증된 quota,
문서화된 SLA를 얻는다.
"""

import json
import logging
import os
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_IMAGES_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"
_DEFAULT_MAX_RESULTS = 5
# Brave Search API는 `count` 파라미터를 요청당 20개 결과로 제한한다.
_BRAVE_WEB_MAX_COUNT = 20
# Brave Image Search는 웹 검색보다 큰 배치를 지원한다.
_BRAVE_IMAGE_MAX_COUNT = 200
# NAT64 well-known prefix(RFC 6052). IPv4 주소를 품은 IPv6 리터럴이다.
_NAT64_PREFIX = ip_network("64:ff9b::/96")
_api_key_warned: set[str] = set()


def _get_api_key(tool_name: str = "web_search") -> str | None:
    config = get_app_config().get_tool_config(tool_name)
    if config is not None:
        api_key = (config.model_extra or {}).get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    env_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if isinstance(env_key, str) and env_key.strip():
        return env_key.strip()
    return None


def _coerce_max_results(
    value: object,
    *,
    default: int = _DEFAULT_MAX_RESULTS,
    max_allowed: int = _BRAVE_WEB_MAX_COUNT,
) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid Brave Search max_results=%r; using default %s",
            value,
            default,
        )
        coerced = default

    return max(1, min(coerced, max_allowed))


def _clean_query(query: str, *, max_length: int = 400) -> str:
    query = query.strip()
    if len(query) > max_length:
        query = query[:max_length]
    return query


def _missing_key_error(query: str, tool_name: str) -> str:
    if tool_name not in _api_key_warned:
        _api_key_warned.add(tool_name)
        logger.warning(
            "Brave Search API key is not set for '%s'. Set BRAVE_SEARCH_API_KEY in your environment or provide api_key in config.yaml. Sign up at https://brave.com/search/api/",
            tool_name,
        )
    return json.dumps(
        {"error": "BRAVE_SEARCH_API_KEY is not configured", "query": query},
        ensure_ascii=False,
    )


def _unexpected_format_error(query: str, *, service_name: str = "Brave Search") -> str:
    return json.dumps(
        {"error": f"{service_name} returned an unexpected response format", "query": query},
        ensure_ascii=False,
    )


def _decode_ipv4(host: str) -> IPv4Address | None:
    """``ip_address``가 거부하는 난독화된 IPv4 리터럴을 디코딩한다.

    많은 HTTP client가 쓰는 관대한 ``inet_aton`` 파싱을 흉내 내어 정수(``2130706433``),
    16진수(``0x7f000001``), 8진수(``0177.0.0.1``) 표기를 모두 인식한다.
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
    return IPv4Address(result)


def _is_url_present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _embedded_ipv4(ip: IPv6Address) -> IPv4Address | None:
    """IPv6 리터럴에 박혀 있는 IPv4 주소가 있으면 꺼낸다.

    IPv4-mapped(``::ffff:a.b.c.d``), 6to4(``2002::/16``), NAT64(``64:ff9b::/96``),
    IPv4-compatible(``::a.b.c.d``) 형태를 모두 다룬다. 이들은 IPv6 경로로 v4 목적지를
    숨겨 들여오며, v6 리터럴만 ``is_global``로 검사하면 loopback/사설 대상이 안전한 것으로
    보고된다.
    """
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip in _NAT64_PREFIX:
        return IPv4Address(int(ip) & 0xFFFFFFFF)
    # IPv4-compatible ``::a.b.c.d`` (상위 96비트가 0이며 ::/::1은 제외).
    packed = int(ip)
    if packed >> 32 == 0 and packed > 1:
        return IPv4Address(packed & 0xFFFFFFFF)
    return None


def _safe_public_url(value: object) -> str:
    """``value``가 안전한 공개 http(s) URL일 때만 그대로 반환하고, 아니면 ""를 반환한다.

    best-effort SSRF 방어다. http(s)가 아닌 scheme, ``localhost``, 사설/비공개 IP 리터럴
    (난독화된 10/16/8진수 표기와 비공개 IPv4를 품은 IPv6 리터럴 포함)을 거부한다.
    URL 문자열만 검사하므로 내부 IP로 resolve되는 공개 hostname은 잡지 못한다. 이 URL을
    실제로 다운로드하는 쪽은 fetch 시점에 resolve된 IP를 다시 검증해야 한다.
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
    if isinstance(ip, IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None and not embedded.is_global:
            return ""
    return url if ip.is_global else ""


def _brave_get(
    endpoint: str,
    api_key: str,
    query: str,
    params: dict[str, object],
    *,
    service_name: str,
) -> tuple[dict | None, str | None]:
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            logger.error("%s returned an unexpected payload type: %s", service_name, type(data).__name__)
            return None, _unexpected_format_error(query, service_name=service_name)
        return data, None
    except httpx.HTTPStatusError as e:
        logger.error("%s API returned HTTP %s: %s", service_name, e.response.status_code, e.response.text)
        return None, json.dumps(
            {"error": f"{service_name} API error: HTTP {e.response.status_code}", "query": query},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("%s request failed: %s: %s", service_name, type(e).__name__, e)
        return None, json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Brave Search로 웹에서 정보를 검색한다.

    Args:
        query: 찾으려는 내용을 설명하는 검색 키워드. 구체적으로 쓸수록 결과가 좋아진다.
        max_results: 반환할 최대 검색 결과 개수. 기본값은 5.
    """
    config = get_app_config().get_tool_config("web_search")
    if config is not None and "max_results" in (config.model_extra or {}):
        max_results = config.model_extra["max_results"]

    count = _coerce_max_results(max_results, max_allowed=_BRAVE_WEB_MAX_COUNT)
    query = _clean_query(query)

    api_key = _get_api_key("web_search")
    if not api_key:
        return _missing_key_error(query, "web_search")

    params = {"q": query, "count": count, "text_decorations": False}

    data, error_json = _brave_get(_BRAVE_WEB_ENDPOINT, api_key, query, params, service_name="Brave Search")
    if error_json is not None:
        return error_json

    web_results = (data.get("web") or {}).get("results", [])
    if not web_results:
        return json.dumps({"error": "No results found", "query": query}, ensure_ascii=False)

    normalized_results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("description", ""),
        }
        for r in web_results
    ]

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


@tool("image_search", parse_docstring=True)
def image_search_tool(query: str, max_results: int = 5) -> str:
    """Brave Image Search로 온라인 이미지를 검색한다. 인물, 초상, 사물, 장면 등 시각적 정확도가 필요한 콘텐츠의 레퍼런스 이미지를 찾으려면 이미지를 생성하기 전에 반드시 이 도구를 사용하라.

    반환된 이미지 URL은 이미지 생성 시 레퍼런스 이미지로 사용할 수 있으며, 품질을 크게 향상시킨다.

    Args:
        query: 찾으려는 이미지를 설명하는 검색 키워드. 구체적으로 쓸수록 결과가 좋아진다.
        max_results: 반환할 최대 이미지 개수. 기본값은 5이고 최대 200으로 제한된다.
    """
    config = get_app_config().get_tool_config("image_search")
    extra = (config.model_extra or {}) if config is not None else {}
    if "max_results" in extra:
        max_results = extra["max_results"]
    count = _coerce_max_results(max_results, max_allowed=_BRAVE_IMAGE_MAX_COUNT)
    query = _clean_query(query)

    api_key = _get_api_key("image_search")
    if not api_key:
        return _missing_key_error(query, "image_search")

    params: dict[str, object] = {"q": query, "count": count}
    for key in ("country", "search_lang", "safesearch", "spellcheck"):
        if key in extra:
            params[key] = extra[key]

    data, error_json = _brave_get(
        _BRAVE_IMAGES_ENDPOINT,
        api_key,
        query,
        params,
        service_name="Brave Image Search",
    )
    if error_json is not None:
        return error_json

    images = data.get("results")
    if images is None:
        images = []
    if not isinstance(images, list):
        logger.error("Brave Image Search returned unexpected 'results' payload type: %s", type(images).__name__)
        return _unexpected_format_error(query, service_name="Brave Image Search")
    if not images:
        return json.dumps({"error": "No images found", "query": query}, ensure_ascii=False)

    normalized_results = []
    for item in images:
        if not isinstance(item, dict):
            continue
        thumbnail = item.get("thumbnail") if isinstance(item.get("thumbnail"), dict) else {}
        properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        raw_image = properties.get("url")
        raw_thumb = thumbnail.get("src")
        raw_source = item.get("url")

        safe_image = _safe_public_url(raw_image)
        safe_thumb = _safe_public_url(raw_thumb)
        safe_source = _safe_public_url(raw_source)

        # URL을 노출하면서 그 URL이 어느 dict에서 왔는지 함께 기억한다. 그래야 보고되는
        # width/height가 버려진 URL이 아니라 실제로 반환하는 URL을 설명한다.
        if safe_image:
            image_url, image_dims = safe_image, properties
        elif not _is_url_present(raw_image):
            image_url, image_dims = safe_thumb, thumbnail
        else:
            image_url, image_dims = "", {}

        if safe_thumb:
            thumbnail_url, thumb_dims = safe_thumb, thumbnail
        elif not _is_url_present(raw_thumb):
            thumbnail_url, thumb_dims = safe_image, properties
        else:
            thumbnail_url, thumb_dims = "", {}

        if not image_url and not thumbnail_url:
            continue

        dims = image_dims if image_url else thumb_dims

        normalized_results.append(
            {
                "title": item.get("title", ""),
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "source_url": safe_source,
                "source": item.get("source", ""),
                "width": dims.get("width"),
                "height": dims.get("height"),
            }
        )
        if len(normalized_results) >= count:
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
