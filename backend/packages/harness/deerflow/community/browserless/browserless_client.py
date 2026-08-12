import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrowserlessFetchResult:
    html: str
    target_status_code: str
    target_status: str


@dataclass(frozen=True)
class BrowserlessScreenshotResult:
    content: bytes
    content_type: str
    target_status_code: str
    target_status: str
    final_url: str


def _get_header(headers: Any, name: str) -> str:
    value = headers.get(name)
    if value:
        return str(value)
    return str(headers.get(name.lower(), ""))


class BrowserlessClient:
    """Browserless headless Chrome API 클라이언트."""

    def __init__(self, base_url: str, token: str = "", timeout_s: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    async def fetch_html(
        self,
        url: str,
        wait_for_event: str = "",
        wait_for_timeout_ms: int = 0,
        wait_for_selector: str = "",
        wait_for_selector_timeout_ms: int = 5000,
        reject_resource_types: list[str] | None = None,
        reject_request_pattern: list[str] | None = None,
    ) -> str:
        """Browserless로 page의 렌더링된 HTML을 가져온다.

        공개 문자열 계약: 항상 렌더링된 HTML 또는 "Error: ..." 문자열을 반환하며, 더 풍부한
        결과 객체를 반환하지 않는다. 대상 page의 실제 status까지 필요하면
        fetch_html_with_status()를 쓴다. Browserless는 대상 page가 4xx/5xx나 anti-bot 차단
        page를 응답해도 렌더 요청 자체에는 HTTP 200을 준다.

        Args:
            url: 가져올 URL.
            wait_for_event: 기다릴 page 이벤트(예: "networkidle", "load").
            wait_for_timeout_ms: page load 후 추가 대기 시간.
            wait_for_selector: 기다릴 CSS selector.
            wait_for_selector_timeout_ms: selector 대기 timeout.
            reject_resource_types: 차단할 리소스 타입(예: ["image"]).
            reject_request_pattern: 차단할 URL 패턴.

        Returns:
            렌더링된 HTML. 실패 시 "Error: ..." 문자열.
        """
        result = await self.fetch_html_with_status(
            url=url,
            wait_for_event=wait_for_event,
            wait_for_timeout_ms=wait_for_timeout_ms,
            wait_for_selector=wait_for_selector,
            wait_for_selector_timeout_ms=wait_for_selector_timeout_ms,
            reject_resource_types=reject_resource_types,
            reject_request_pattern=reject_request_pattern,
        )
        return result.html if isinstance(result, BrowserlessFetchResult) else result

    async def fetch_html_with_status(
        self,
        url: str,
        wait_for_event: str = "",
        wait_for_timeout_ms: int = 0,
        wait_for_selector: str = "",
        wait_for_selector_timeout_ms: int = 5000,
        reject_resource_types: list[str] | None = None,
        reject_request_pattern: list[str] | None = None,
    ) -> BrowserlessFetchResult | str:
        """Browserless로 page의 렌더링된 HTML을 대상 status와 함께 가져온다.

        요청/응답 처리는 fetch_html()과 같지만, 성공 시 문자열 대신 대상 page의 실제 status
        헤더를 담은 BrowserlessFetchResult를 반환한다. 덕분에 호출자가 진짜 200과
        "렌더는 성공했지만 대상이 에러(또는 anti-bot 차단)"인 응답을 구분할 수 있다.
        HTML/에러 문자열만 필요하면 fetch_html()을 쓴다.

        현재 Browserless API 버전이 받는 파라미터만 보낸다. query param으로 기본 navigation
        timeout(30s)을 설정한다.

        Args:
            url: 가져올 URL.
            wait_for_event: 기다릴 page 이벤트(예: "networkidle", "load").
            wait_for_timeout_ms: page load 후 추가 대기 시간.
            wait_for_selector: 기다릴 CSS selector.
            wait_for_selector_timeout_ms: selector 대기 timeout.
            reject_resource_types: 차단할 리소스 타입(예: ["image"]).
            reject_request_pattern: 차단할 URL 패턴.

        Returns:
            렌더링된 HTML과 대상 page status 헤더를 담은 결과. 실패 시 "Error: ..." 문자열.
        """
        payload: dict[str, Any] = {
            "url": url,
        }

        if self.token:
            payload["token"] = self.token
        if wait_for_event:
            payload["waitForEvent"] = wait_for_event
        if wait_for_timeout_ms > 0:
            payload["waitForTimeout"] = wait_for_timeout_ms
        if wait_for_selector:
            payload["waitForSelector"] = {
                "selector": wait_for_selector,
                "timeout": wait_for_selector_timeout_ms,
            }
        if reject_resource_types:
            payload["rejectResourceTypes"] = reject_resource_types
        if reject_request_pattern:
            payload["rejectRequestPattern"] = reject_request_pattern

        logger.debug(f"Fetching URL via Browserless: {url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/content",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                    },
                )

                code = resp.status_code
                target_code = resp.headers.get("X-Response-Code", "")
                target_status = resp.headers.get("X-Response-Status", "")

                logger.debug(f"Browserless response: code={code}, target_code={target_code}, target_status={target_status}")

                if code != 200:
                    return f"Error: Browserless HTTP {code}: {resp.text[:200]}"

                html = resp.text
                if not html or not html.strip():
                    return "Error: Browserless returned empty response"

                return BrowserlessFetchResult(
                    html=html,
                    target_status_code=_get_header(resp.headers, "X-Response-Code"),
                    target_status=_get_header(resp.headers, "X-Response-Status"),
                )

        except httpx.TimeoutException:
            return f"Error: Browserless request timed out after {self.timeout_s}s"
        except httpx.RequestError as e:
            logger.error(f"Browserless request failed: {e}")
            return f"Error: Browserless request failed: {e!s}"
        except Exception as e:
            logger.error(f"Browserless fetch failed: {e}")
            return f"Error: Browserless fetch failed: {e!s}"

    async def capture_screenshot(
        self,
        url: str,
        full_page: bool = True,
        output_format: str = "png",
        quality: int | None = None,
        viewport: dict[str, int] | None = None,
        wait_for_selector: str = "",
        wait_for_selector_timeout_ms: int = 5000,
        wait_for_timeout_ms: int = 0,
        best_attempt: bool = False,
    ) -> BrowserlessScreenshotResult | str:
        """Browserless로 URL을 렌더링해 screenshot을 캡처한다.

        Args:
            url: 렌더링할 URL.
            full_page: viewport만이 아니라 page 전체를 캡처한다.
            output_format: 이미지 포맷. png, jpeg, webp 중 하나다.
            quality: jpeg/webp 출력에 대한 선택적 품질값.
            viewport: 선택적 browser viewport 딕셔너리.
            wait_for_selector: 캡처 전에 기다릴 CSS selector.
            wait_for_selector_timeout_ms: selector 대기 timeout.
            wait_for_timeout_ms: navigation 후 추가 대기 시간.
            best_attempt: 대기가 timeout되어도 계속 진행한다.

        Returns:
            바이너리 내용을 담은 screenshot 결과. 실패 시 에러 문자열.
        """
        payload: dict[str, Any] = {
            "url": url,
            "options": {
                "fullPage": full_page,
                "type": output_format,
            },
        }
        if quality is not None:
            payload["options"]["quality"] = quality
        if viewport:
            payload["viewport"] = viewport
        if wait_for_selector:
            payload["waitForSelector"] = {
                "selector": wait_for_selector,
                "timeout": wait_for_selector_timeout_ms,
            }
        if wait_for_timeout_ms > 0:
            payload["waitForTimeout"] = wait_for_timeout_ms
        if best_attempt:
            payload["bestAttempt"] = True

        params = {"token": self.token} if self.token else None

        logger.debug(f"Capturing URL screenshot via Browserless: {url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/screenshot",
                    json=payload,
                    params=params,
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                    },
                )

                code = resp.status_code
                logger.debug(
                    "Browserless screenshot response: code=%s, target_code=%s, target_status=%s",
                    code,
                    resp.headers.get("X-Response-Code", ""),
                    resp.headers.get("X-Response-Status", ""),
                )

                if code != 200:
                    return f"Error: Browserless HTTP {code}: {resp.text[:200]}"

                content = resp.content
                if not content:
                    return "Error: Browserless returned empty screenshot response"

                return BrowserlessScreenshotResult(
                    content=content,
                    content_type=_get_header(resp.headers, "Content-Type"),
                    target_status_code=_get_header(resp.headers, "X-Response-Code"),
                    target_status=_get_header(resp.headers, "X-Response-Status"),
                    final_url=_get_header(resp.headers, "X-Response-URL"),
                )

        except httpx.TimeoutException:
            return f"Error: Browserless screenshot request timed out after {self.timeout_s}s"
        except httpx.RequestError as e:
            logger.error(f"Browserless screenshot request failed: {e}")
            return f"Error: Browserless screenshot request failed: {e!s}"
        except Exception as e:
            logger.error(f"Browserless screenshot failed: {e}")
            return f"Error: Browserless screenshot failed: {e!s}"
