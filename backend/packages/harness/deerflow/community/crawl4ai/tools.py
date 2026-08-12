import logging

from langchain.tools import tool

from deerflow.community.url_safety import validate_public_http_url
from deerflow.config import get_app_config

from .crawl4ai_client import Crawl4AiClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11235"
DEFAULT_TIMEOUT_S = 30
DEFAULT_FILTER = "fit"
VALID_FILTERS = ("fit", "raw", "bm25", "llm")


def _get_tool_config(tool_name: str) -> dict | None:
    """도구 설정의 extra(model_extra) dict를 반환한다. 설정이 없으면 None을 반환한다."""
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return None
    extras = config.model_extra
    return extras if extras is not None else {}


def _coerce_timeout(value: object, default: int) -> float:
    """설정의 timeout을 초 단위로 변환한다. 잘못된 입력이면 ``default``로 되돌린다.

    ``jina_ai._coerce_timeout``과 같은 방식이다. bool과 숫자가 아닌 문자열은 기본값으로
    떨어뜨려, ``timeout: off``(YAML의 ``False``) 같은 값이 ``0.0``이 되어 정상 서버에 대한
    모든 요청을 timeout시키지 않게 한다.
    """
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            logger.warning("Crawl4AI web_fetch: invalid timeout %r in config; using %ss", value, default)
    return float(default)


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


def _coerce_filter(value: object) -> str:
    """markdown 필터를 정규화하고 검증하며, 잘못되면 기본값으로 되돌린다.

    오타나 오래된 값(예: ``FIt``, ``fit_content``)을 설정 읽기 시점에 잡아, 정체를 알기 어려운
    HTTP 400으로 서버까지 흘러가지 않게 한다.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_FILTERS:
            return normalized
        logger.warning("Crawl4AI web_fetch: unknown filter %r in config; using %r (valid: %s)", value, DEFAULT_FILTER, ", ".join(VALID_FILTERS))
    return DEFAULT_FILTER


def _build_client(cfg: dict | None) -> Crawl4AiClient:
    """이미 읽어 둔 ``web_fetch`` 설정 dict로 ``Crawl4AiClient``를 만든다.

    설정을 다시 읽지 않고 인자로 받는다. 그래야 한 번의 호출이 ``get_app_config()``를 정확히
    한 번만 읽고, 동시에 일어나는 hot-reload를 걸치지 않는다.
    """
    base_url = DEFAULT_BASE_URL
    token = ""
    timeout_s: float = float(DEFAULT_TIMEOUT_S)
    if cfg is not None:
        base_url = cfg.get("base_url", base_url)
        token = cfg.get("token", token)
        timeout_s = _coerce_timeout(cfg.get("timeout"), DEFAULT_TIMEOUT_S)
    return Crawl4AiClient(base_url=base_url, token=token, timeout_s=timeout_s)


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """지정된 URL의 웹 페이지 내용을 가져온다.
    사용자가 직접 제공했거나 web_search, web_fetch 도구의 결과로 반환된 URL만 정확히 그대로 가져와라.
    이 도구는 비공개 Google Docs나 로그인 뒤에 있는 페이지처럼 인증이 필요한 콘텐츠에는 접근할 수 없다.
    www.가 없는 URL에 www.를 붙이지 마라.
    URL에는 반드시 schema가 포함되어야 한다: https://example.com은 유효한 URL이지만 example.com은 유효하지 않다.

    Args:
        url: 내용을 가져올 URL.
    """
    try:
        cfg = _get_tool_config("web_fetch")  # 설정은 한 번만 읽고 값을 아래로 전달한다
        allow_private_addresses = _coerce_bool(cfg.get("allow_private_addresses") if cfg is not None else None, False)
        url_error = validate_public_http_url(url, allow_private_addresses=allow_private_addresses)
        if url_error:
            return url_error
        filter_mode = _coerce_filter(cfg.get("filter") if cfg is not None else None)
        client = _build_client(cfg)
        markdown = await client.fetch_markdown(url, filter_mode=filter_mode)

        if markdown.startswith("Error:"):
            return markdown

        return markdown[:4096]

    except Exception as e:
        logger.error(f"Error in web_fetch_tool: {e}")
        return f"Error: {str(e)}"
