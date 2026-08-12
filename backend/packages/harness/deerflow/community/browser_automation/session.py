"""Playwright 기반의 상태 유지형, loop 종속 browser session.

Playwright의 async 객체(``Browser``/``BrowserContext``/``Page``)는 자신을 생성한 event loop에
종속된다. DeerFlow 도구는 Gateway loop, TUI loop, 새 테스트 loop 어디에서든 await될 수 있고,
browser session은 같은 thread의 여러 turn에 걸쳐 살아 있어야 한다. Playwright의 loop를 호출자
loop와 분리하기 위해 모든 Playwright 연산은 전용 daemon event loop 하나에서 실행한다
(BoxLite provider가 loop 종속 box handle에 쓰는 방식과 같다). async 도구는
:func:`asyncio.wrap_future`로 결과를 await한다.

Playwright 자체는 optional 의존성이다. 전용 loop 안에서 지연 import하므로 core harness는
Playwright 없이도 설치된다.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)

T = TypeVar("T")

# page snapshot을 만들 때 interactive로 취급하는 element role/tag. 모델은 이 snapshot이
# 찍어 두는 ``data-df-ref`` 인덱스로 element를 지정하므로, CSS selector를 추측하거나
# 오래된 element handle을 들고 있을 필요가 없다.
_SNAPSHOT_JS = r"""
() => {
  // Clear ref stamps from any previous snapshot first. GitHub-style SPAs keep
  // stale (now-hidden) nodes in the DOM carrying old data-df-ref values; if we
  // don't strip them, a later click selector like [data-df-ref="5"] can match
  // the hidden leftover ahead of the current visible element in DOM order and
  // time out waiting for it to become actionable.
  for (const stale of document.querySelectorAll("[data-df-ref]")) {
    stale.removeAttribute("data-df-ref");
  }
  const INTERACTIVE = new Set(["A", "BUTTON", "INPUT", "TEXTAREA", "SELECT"]);
  const results = [];
  let ref = 0;
  const nodes = document.querySelectorAll(
    "a, button, input, textarea, select, [role=button], [role=link], [role=tab], [role=checkbox], [onclick]"
  );
  for (const el of nodes) {
    const rect = el.getBoundingClientRect();
    const visible = rect.width > 0 && rect.height > 0 &&
      window.getComputedStyle(el).visibility !== "hidden" &&
      window.getComputedStyle(el).display !== "none";
    if (!visible) continue;
    ref += 1;
    el.setAttribute("data-df-ref", String(ref));
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute("role") || "";
    const type = el.getAttribute("type") || "";
    let name = (el.getAttribute("aria-label") || el.getAttribute("name") ||
      el.getAttribute("placeholder") || el.innerText || el.value || "").trim();
    if (name.length > 120) name = name.slice(0, 120) + "…";
    results.push({ ref, tag, role, type, name });
    if (results.length >= 200) break;
  }
  return { url: location.href, title: document.title, elements: results };
}
"""


_WHEEL_SCROLL_JS = r"""
({ x, y, dx, dy }) => {
  const root = document.scrollingElement || document.documentElement;
  const candidates = [];
  let node = document.elementFromPoint(x, y);

  while (node && node !== document.documentElement) {
    if (node instanceof Element) {
      candidates.push(node);
    }
    node = node.parentElement;
  }
  candidates.push(root);

  const canScroll = (el, axis, delta) => {
    if (!delta || !el) {
      return false;
    }
    const max =
      axis === "y"
        ? el.scrollHeight - el.clientHeight
        : el.scrollWidth - el.clientWidth;
    if (max <= 0) {
      return false;
    }
    const current = axis === "y" ? el.scrollTop : el.scrollLeft;
    return delta < 0 ? current > 0 : current < max;
  };

  for (const el of candidates) {
    if (!canScroll(el, "y", dy) && !canScroll(el, "x", dx)) {
      continue;
    }
    const beforeLeft = el.scrollLeft;
    const beforeTop = el.scrollTop;
    el.scrollBy({ left: dx, top: dy, behavior: "auto" });
    return el.scrollLeft !== beforeLeft || el.scrollTop !== beforeTop;
  }

  const beforeX = window.scrollX;
  const beforeY = window.scrollY;
  window.scrollBy({ left: dx, top: dy, behavior: "auto" });
  return window.scrollX !== beforeX || window.scrollY !== beforeY;
}
"""


# click 액션별 timeout. session 기본값(30s)보다 훨씬 짧게 두어, 오래되거나 잘못된 ref가
# 빨리 실패하고 모델이 다시 snapshot을 뜨게 한다. 그렇지 않으면 browsing loop 전체가 막히고
# agent의 loop-detection 안전 중단까지 걸린다.
_CLICK_TIMEOUT_MS = 8000
# click 이후의 짧은 best-effort 안정화 대기. SPA(클라이언트 측) 이동은 새 load 이벤트를
# 발생시키지 않으므로 이 대기가 액션을 막아서는 안 된다.
_POST_CLICK_LOAD_TIMEOUT_MS = 3000
_LIVE_FRAME_JPEG_QUALITY = 85
_MANUAL_LIVE_FRAME_MIN_INTERVAL_S = 0.75
_LIVE_FRAME_INPUT_INTERVAL_S = 0.05
_LIVE_FRAME_SETTLE_DELAYS_S = (0.8, 2.0)

# 장기 실행 다중 사용자 gateway에서 thread별 Chromium이 무한정 쌓이는 것을 막는다.
# idle timeout을 넘긴 session은 다음 get_session 호출 때 lazy하게 제거하고,
# 상한을 넘으면 LRU session을 닫는다.
_DEFAULT_MAX_SESSIONS = 32
_DEFAULT_IDLE_TIMEOUT_S = 30 * 60.0


def browser_multi_worker_error(workers: int | None = None) -> str | None:
    """browser session이 프로세스 로컬이라 fail-closed해야 하는 사유를 반환한다."""
    if workers is None:
        try:
            workers = int(os.environ.get("GATEWAY_WORKERS", "1"))
        except (TypeError, ValueError):
            workers = 1
    if workers <= 1:
        return None
    return f"GATEWAY_WORKERS={workers} cannot enable agentic browser tools: browser sessions are process-local and uvicorn does not provide thread affinity. Set GATEWAY_WORKERS=1 or disable the browser_navigate tool."


def ensure_browser_worker_compatibility() -> None:
    """요청이 다른 worker로 갈 수 있는 환경에서는 런타임 browser 사용을 거부한다."""
    error = browser_multi_worker_error()
    if error is not None:
        raise RuntimeError(error)


class BrowserSessionCapacityError(RuntimeError):
    """browser session 상한에 밀어낼 수 있는 슬롯이 없을 때 발생한다."""


class BrowserLiveViewerError(RuntimeError):
    """두 번째 Live viewer가 같은 session에 붙으려 할 때 발생한다."""


def _is_playwright_timeout_error(exc: Exception) -> bool:
    """import 시점에 Playwright를 요구하지 않고 Playwright timeout을 식별한다."""
    return exc.__class__.__name__ == "TimeoutError" and exc.__class__.__module__.startswith("playwright.")


def redact_browser_url(url: str) -> str:
    """차단된 URL 로그가 token/PII를 흘리지 않도록 query와 fragment를 제거한다."""
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return "<unparsable-url>"


class _PlaywrightLoopThread:
    """전용 daemon thread에서 도는 전용 asyncio event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="deerflow-browser-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """*coro*를 전용 loop에 예약하고 어떤 loop에서든 await한다."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        """호출자를 block하지 않고 *coro*를 전용 loop에 예약한다."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _log_failure(done: Any) -> None:
            try:
                done.result()
            except Exception as exc:
                logger.debug("browser background task failed: %s", exc)

        future.add_done_callback(_log_failure)

    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


@dataclass
class SnapshotElement:
    ref: int
    tag: str
    role: str
    type: str
    name: str

    def render(self) -> str:
        label = self.role or self.tag
        detail = f" type={self.type}" if self.type else ""
        name = self.name or "(no text)"
        return f"[{self.ref}] {label}{detail}: {name}"


@dataclass
class PageSnapshot:
    url: str
    title: str
    elements: list[SnapshotElement] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}", ""]
        if not self.elements:
            lines.append("No interactive elements detected.")
        else:
            lines.append("Interactive elements (address them by [ref] number):")
            lines.extend(el.render() for el in self.elements)
        return "\n".join(lines)


@dataclass
class BrowserTab:
    index: int
    url: str
    title: str
    active: bool


class BrowserSession:
    """전용 loop에 묶인 Playwright browser+page 한 쌍."""

    def __init__(
        self,
        loop: _PlaywrightLoopThread,
        *,
        headless: bool,
        timeout_ms: int,
        viewport: dict[str, int],
        cdp_url: str | None = None,
        url_guard: Callable[[str], str | None] | None = None,
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        self._loop = loop
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._viewport = viewport
        # browser request 경계에 적용하는 선택적 SSRF guard. URL(redirect/popup/subresource)을
        # 차단할 때는 에러 문자열을, 허용할 때는 None을 반환한다. 명시적 navigate URL은 호출자가
        # 검사하지만, Playwright는 redirect를 따라가고 subresource/popup 요청을 내보내며 그
        # 한 번의 검사를 우회한다. 그래서 page가 만드는 모든 요청을 여기서도 검증해,
        # 사설 호스트나 cloud metadata로 30x redirect되는 공개 URL을 잡아낸다.
        self._url_guard = url_guard
        self._request_guard_bound = False
        # 설정되면 전용 headless 인스턴스를 띄우는 대신 DevTools Protocol로 이미 실행 중인
        # Chrome에 붙는다(Codex의 "connect to your real browser"와 같다). 사용자는 자신의
        # 실제 로그인 session이 살아 있는 보이는 browser를 agent가 조작하는 걸 지켜본다.
        self._cdp_url = cdp_url
        self._on_activity = on_activity
        self._activity_lock = threading.Lock()
        self._active_refs = 0
        self._connected_over_cdp = False
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        # 도구 호출과 Live WebSocket이 이 session을 공유하므로 닫힌 page를 동시에 볼 수 있다.
        # browser 계층 재구성은 한 호출자만 수행하고, lock 안의 두 번째 검사가 그 결과를 재사용한다.
        self._ensure_page_lock = asyncio.Lock()
        # Live screencast 상태. streaming 중에는 ``_on_frame``을 유지해 screencast를 새 page에
        # 다시 bind할 수 있게 한다. 로그인/OAuth 흐름은 흔히 popup이나 새 tab을 여는데,
        # 사용자가 그것을 보고 조작해야 한다.
        self._on_frame: Callable[[bytes], None] | None = None
        # live screencast의 CDP session이 현재 bind된 page. frame은 ``self._page``(현재 활성
        # page)에서 캡처하지만 CDP repaint 신호는 특정 page에 묶여 있다. 활성 page가 이 값과
        # 어긋나면 다시 bind해야 새 page의 repaint가 계속 frame을 만든다.
        self._screencast_page: Page | None = None
        # 재진입 rebind 방지 플래그. screencast를 (재)bind하는 동안 _ensure_page를 호출할 수
        # 있고 이는 _set_active_page를 거친다. 이 플래그가 없으면 또 다른 rebind가 예약되어
        # 재귀에 빠진다.
        self._screencast_binding = False
        self._last_manual_live_frame_at = 0.0
        self._settle_live_frames_pending = False
        self._input_live_frame_generation = 0
        self._input_live_frame_pending = False
        self._page_listener_bound = False

    @property
    def active_refs(self) -> int:
        with self._activity_lock:
            return self._active_refs

    def _pin(self) -> None:
        """호출자가 점유하는 동안 이 session을 manager에 붙잡아 둔다."""
        with self._activity_lock:
            self._active_refs += 1

    def _unpin(self) -> None:
        with self._activity_lock:
            self._active_refs = max(0, self._active_refs - 1)

    @contextlib.contextmanager
    def _activity(self):
        """실제 browser 연산을 참조 카운트에 잡고 최근 사용 시각을 갱신한다."""
        self._pin()
        if self._on_activity is not None:
            self._on_activity()
        try:
            yield
        finally:
            self._unpin()

    async def _ensure_page(self) -> Page:
        if self._page is not None and not self._page.is_closed():
            return self._page
        async with self._ensure_page_lock:
            if self._page is not None and not self._page.is_closed():
                return self._page
            from playwright.async_api import async_playwright

            if self._playwright is None:
                self._playwright = await async_playwright().start()

            if self._cdp_url:
                # 사용자가 --remote-debugging-port로 띄운 Chrome에 붙는다. 기본 context와
                # 기존 tab을 재사용한다. CDP로 붙은 실제 Chrome에서 new_context()/new_page()를
                # 호출하면 "Browser context management is not supported"가 나므로,
                # Chrome이 이미 열어 둔 tab을 그대로 쓴다.
                if self._browser is None or not self._browser.is_connected():
                    self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)
                    self._connected_over_cdp = True
                    # CDP로 붙은 실제 Chrome은 자체 browsing context를 소유하므로 SSRF request
                    # guard를 의도적으로 설치하지 않는다(_install_request_guard 참고).
                    # 운영자가 알 수 있게 경고를 남긴다. 이 session에서는 사설/metadata 호스트로의
                    # redirect나 subresource가 차단되지 않으며, cdp_url은 로컬/신뢰 환경 전용이다.
                    logger.warning(
                        "browser SSRF request guard is disabled for CDP-attached session (cdp_url=%s)",
                        redact_browser_url(self._cdp_url),
                    )
                self._context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
                self._context.set_default_timeout(self._timeout_ms)
                existing = self._context.pages
                self._set_active_page(existing[-1] if existing else await self._context.new_page())
                self._bind_new_page_listener()
                return self._page

            if self._browser is None or not self._browser.is_connected():
                self._browser = await self._playwright.chromium.launch(headless=self._headless)
            # device_scale_factor=2로 screenshot을 retina 밀도로 렌더링해, 이미지를 확대해
            # 화면을 채워도 panel이 선명하게 유지된다.
            self._context = await self._browser.new_context(viewport=self._viewport, device_scale_factor=2)
            self._context.set_default_timeout(self._timeout_ms)
            await self._install_request_guard()
            self._set_active_page(await self._context.new_page())
            self._bind_new_page_listener()
            return self._page

    def _set_active_page(self, page: Page) -> None:
        """*page*를 활성 page로 채택하고 live screencast를 그 page에 유지한다.

        활성 page를 바꾸는 모든 경로(최초/재구성 page, popup과 새 tab, 명시적 tab 전환)가
        여기를 거치므로 live stream이 오래된 page로 흘러갈 수 없다. CDP screencast의 repaint
        신호는 page 하나에 묶여 있으므로, 활성 page가 screencast가 bind된 page와 어긋나면
        다시 bind한다. rebind는 에러 처리와 함께 예약해, 일시적 실패로 callback이 await되지
        않은 채 조용히 삼켜지지 않게 한다. 이전의 fire-and-forget rebind는 Live를 옛 page에
        묶어 둘 수 있었다. 주소창과 snapshot은 넘어갔는데 frame만 그대로였다.
        """
        self._page = page
        if self._on_frame is not None and not self._screencast_binding and page is not self._screencast_page:
            asyncio.ensure_future(self._rebind_screencast_safe())

    def _bind_new_page_listener(self) -> None:
        """popup과 새 tab을 따라가 인증 흐름이 계속 보이고 조작 가능하게 한다.

        로그인과 OAuth 동의 화면은 흔히 popup이나 새 tab을 연다. 따라가지 않으면 사용자는
        멈춘 화면만 보고 승인할 수 없다. 새 page가 열리면 그것을 활성 page로 채택하고,
        screencast가 돌고 있으면 다시 bind해 stream이 사용자가 조작해야 할 tab을 따라가게 한다.
        """
        if self._context is None or self._page_listener_bound:
            return

        def _on_new_page(page: Page) -> None:
            self._set_active_page(page)

        self._context.on("page", _on_new_page)
        self._page_listener_bound = True

    async def _install_request_guard(self) -> None:
        """URL이 SSRF guard를 통과하지 못하는 요청을 모두 중단시킨다.

        context 수준에서 동작하므로 최상위 navigation, 모든 redirect hop, popup/새 tab,
        iframe, subresource fetch까지 덮는다. 최초 URL 한 번 검사로는 볼 수 없는 경로들이다.
        ``http://169.254.169.254/...`` 로 redirect되는 공개 URL은 응답이 snapshot이나 텍스트로
        노출되기 전에 중단된다. 자체 browsing context를 소유하는 CDP 연결 Chrome에서는 건너뛴다.
        """
        if self._url_guard is None or self._context is None or self._request_guard_bound or self._cdp_url:
            return

        guard = self._url_guard

        async def _route(route: Any) -> None:
            url = ""
            with contextlib.suppress(Exception):
                url = route.request.url
            if url.startswith(("http://", "https://")) and guard(url) is not None:
                logger.warning("browser request blocked by SSRF guard: %s", redact_browser_url(url))
                with contextlib.suppress(Exception):
                    await route.abort("blockedbyclient")
                return
            with contextlib.suppress(Exception):
                await route.continue_()

        with contextlib.suppress(Exception):
            await self._context.route("**/*", _route)
            self._request_guard_bound = True

    async def _rebind_screencast(self) -> None:
        if self._on_frame is not None:
            await self._start_screencast(self._on_frame)

    async def _rebind_screencast_safe(self) -> None:
        try:
            await self._rebind_screencast()
        except Exception as exc:
            logger.debug("browser live screencast rebind failed: %s", exc)

    async def _navigate(self, url: str) -> PageSnapshot:
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded")
        return await self._snapshot_impl(page)

    async def _snapshot_impl(self, page: Page) -> PageSnapshot:
        data = await page.evaluate(_SNAPSHOT_JS)
        elements = [SnapshotElement(ref=int(e["ref"]), tag=e["tag"], role=e["role"], type=e["type"], name=e["name"]) for e in data["elements"]]
        return PageSnapshot(url=data["url"], title=data["title"], elements=elements)

    async def _snapshot(self) -> PageSnapshot:
        page = await self._ensure_page()
        return await self._snapshot_impl(page)

    async def _click(self, ref: int) -> PageSnapshot:
        page = await self._ensure_page()
        selector = f'[data-df-ref="{ref}"]'
        base = page.locator(selector)

        # 오래된 ref는 session 기본값 30초를 기다리지 않고 바로 실패시킨다. SPA가 다시
        # 렌더링하면 ref가 사라질 수 있고, 이때는 browsing loop가 멎어 agent의 loop-detection
        # 안전 중단까지 가기보다 모델이 다시 snapshot을 뜨고 재시도하는 편이 낫다.
        if await base.count() == 0:
            raise RuntimeError(f"element [{ref}] is no longer on the page; call browser_snapshot to get fresh refs")

        locator = base.first
        # 대상을 화면 안으로 스크롤한다. 이미 보이는 상태여도 무해하다.
        try:
            await locator.scroll_into_view_if_needed(timeout=_CLICK_TIMEOUT_MS)
        except Exception as exc:
            if not _is_playwright_timeout_error(exc):
                raise

        try:
            await locator.click(timeout=_CLICK_TIMEOUT_MS)
        except Exception as exc:
            if not _is_playwright_timeout_error(exc):
                raise
            raise RuntimeError(f"element [{ref}] was not clickable within {_CLICK_TIMEOUT_MS // 1000}s; the page may have changed — call browser_snapshot and retry") from exc

        # SPA(클라이언트 측) 이동은 새 load 이벤트를 발생시키지 않으므로, 이 안정화 대기는
        # best-effort이며 snapshot을 막아서는 안 된다.
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=_POST_CLICK_LOAD_TIMEOUT_MS)
        except Exception as exc:
            if not _is_playwright_timeout_error(exc):
                raise

        return await self._snapshot_impl(page)

    async def _type(self, ref: int, text: str, submit: bool) -> PageSnapshot:
        page = await self._ensure_page()
        selector = f'[data-df-ref="{ref}"]'
        await page.fill(selector, text)
        if submit:
            await page.press(selector, "Enter")
            # best-effort 안정화 대기. 클라이언트 측 검색/제출은 새 load 이벤트를 발생시키지
            # 않을 수 있으므로 snapshot을 30초씩 막아서는 안 된다.
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=_POST_CLICK_LOAD_TIMEOUT_MS)
            except Exception as exc:
                if not _is_playwright_timeout_error(exc):
                    raise
        return await self._snapshot_impl(page)

    async def _get_text(self, max_chars: int) -> str:
        page = await self._ensure_page()
        text = await page.inner_text("body")
        return text[:max_chars]

    async def _screenshot_bytes(self, full_page: bool) -> bytes:
        page = await self._ensure_page()
        return await page.screenshot(full_page=full_page, type="png")

    async def _live_frame(self) -> bytes:
        page = await self._ensure_page()
        return await page.screenshot(type="jpeg", quality=_LIVE_FRAME_JPEG_QUALITY)

    async def _emit_live_frame(self) -> None:
        if self._on_frame is None:
            return
        with self._activity():
            self._on_frame(await self._live_frame())
            self._last_manual_live_frame_at = time.monotonic()

    async def _settle_live_frames(self) -> None:
        previous_delay = 0.0
        try:
            for delay in _LIVE_FRAME_SETTLE_DELAYS_S:
                await asyncio.sleep(max(0.0, delay - previous_delay))
                previous_delay = delay
                await self._emit_live_frame()
        finally:
            self._settle_live_frames_pending = False

    def _schedule_settle_live_frames(self) -> None:
        if self._settle_live_frames_pending:
            return
        self._settle_live_frames_pending = True
        asyncio.ensure_future(self._settle_live_frames())

    async def _push_live_frame(self) -> None:
        if self._on_frame is None:
            return
        elapsed = time.monotonic() - self._last_manual_live_frame_at
        if elapsed >= _MANUAL_LIVE_FRAME_MIN_INTERVAL_S:
            await self._emit_live_frame()
        # SPA에서는 frame 하나만으로는 대개 너무 이르다. page body가 아직 렌더링 중인데
        # URL만 바뀌었을 수 있다. 연속 screencast로 되돌아가는 대신 짧고 제한된 안정화
        # burst를 덧붙인다.
        self._schedule_settle_live_frames()

    async def _flush_input_live_frames(self) -> None:
        try:
            # 첫 burst는 합쳐 보내고, 입력이 이어지는 동안에는 제한된 주기로 계속 갱신한다.
            # trailing debounce 방식이면 wheel/키보드 제스처가 끝날 때까지 화면이 멈춘다.
            await asyncio.sleep(_LIVE_FRAME_INPUT_INTERVAL_S)
            while self._on_frame is not None:
                generation = self._input_live_frame_generation
                await self._emit_live_frame()
                if generation == self._input_live_frame_generation:
                    self._schedule_settle_live_frames()
                    return
                await asyncio.sleep(_LIVE_FRAME_INPUT_INTERVAL_S)
        finally:
            self._input_live_frame_pending = False

    def _schedule_input_live_frame(self) -> None:
        self._input_live_frame_generation += 1
        if self._input_live_frame_pending:
            return
        self._input_live_frame_pending = True
        asyncio.ensure_future(self._flush_input_live_frames())

    async def _back(self) -> PageSnapshot:
        page = await self._ensure_page()
        await page.go_back(wait_until="domcontentloaded")
        return await self._snapshot_impl(page)

    async def _current_url(self) -> str | None:
        page = await self._ensure_page()
        try:
            return page.url
        except Exception:
            return None

    async def _tabs(self) -> list[BrowserTab]:
        await self._ensure_page()
        if self._context is None:
            return []
        pages = [page for page in self._context.pages if not page.is_closed()]
        tabs: list[BrowserTab] = []
        for index, page in enumerate(pages):
            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                url = page.url
            except Exception:
                url = ""
            tabs.append(BrowserTab(index=index, url=url, title=title, active=page == self._page))
        return tabs

    async def _activate_tab(self, index: int) -> None:
        await self._ensure_page()
        if self._context is None:
            return
        pages = [page for page in self._context.pages if not page.is_closed()]
        if index < 0 or index >= len(pages):
            return
        target = pages[index]
        await target.bring_to_front()
        # 이 경로는 async이므로 _set_active_page의 예약 rebind를 거치지 않고 여기서 바로
        # await하며 rebind한다. 그래야 이 호출이 반환되기 전에 frame 소스가 전환된 tab과 맞는다.
        self._page = target
        if self._on_frame is not None and target is not self._screencast_page:
            await self._rebind_screencast()

    async def _close(self) -> None:
        try:
            with contextlib.suppress(Exception):
                await self._stop_screencast()
            if self._connected_over_cdp:
                # 사용자의 실제 Chrome에 붙어 있다. browser, context, 채택한 tab을 절대 닫지
                # 않고 Playwright 연결만 끊는다.
                if self._browser is not None:
                    with contextlib.suppress(Exception):
                        await self._browser.close()
            else:
                if self._context is not None:
                    with contextlib.suppress(Exception):
                        await self._context.close()
                if self._browser is not None:
                    with contextlib.suppress(Exception):
                        await self._browser.close()
            if self._playwright is not None:
                with contextlib.suppress(Exception):
                    await self._playwright.stop()
        finally:
            self._playwright = None
            self._browser = None
            self._context = None
            self._page = None
            self._screencast_page = None
            self._connected_over_cdp = False
            self._on_frame = None
            self._settle_live_frames_pending = False
            self._input_live_frame_generation += 1
            self._page_listener_bound = False
            self._request_guard_bound = False

    async def _start_screencast(self, on_frame: Callable[[bytes], None]) -> None:
        """Live 모드를 시작하고 첫 JPEG frame을 보낸다.

        예전에는 Chrome의 CDP screencast에 붙어 모든 repaint를 고품질 Playwright screenshot으로
        바꿨다. 그러면 거의 정적인 panel도 headless renderer/GPU를 계속 붙잡아 GitHub 같은
        page를 여는 비용이 커졌다. 지금은 필요할 때만 frame을 만든다. 최초 연결, browser 도구
        완료, 사용자 입력이 각각 throttle된 screenshot을 밀어 넣는다.
        """
        page = await self._ensure_page()
        if self._on_frame is not None and self._on_frame is not on_frame:
            raise BrowserLiveViewerError("Browser live stream already has an active viewer")
        await self._stop_screencast()
        self._on_frame = on_frame
        self._screencast_page = page
        await self._emit_live_frame()
        self._schedule_settle_live_frames()

    async def _stop_screencast(self, on_frame: Callable[[bytes], None] | None = None) -> None:
        if on_frame is not None and self._on_frame is not on_frame:
            return
        self._on_frame = None
        self._settle_live_frames_pending = False
        self._input_live_frame_generation += 1
        self._screencast_page = None

    async def _dispatch_input(self, event: dict) -> None:
        page = await self._ensure_page()
        vw = self._viewport.get("width", 1280)
        vh = self._viewport.get("height", 720)
        etype = event.get("type")
        if etype in {"click", "move", "down", "up"}:
            x = float(event.get("nx", 0)) * vw
            y = float(event.get("ny", 0)) * vh
            if etype == "click":
                await page.mouse.click(x, y)
            elif etype == "move":
                await page.mouse.move(x, y)
            elif etype == "down":
                await page.mouse.move(x, y)
                await page.mouse.down()
            elif etype == "up":
                await page.mouse.up()
        elif etype == "wheel":
            dx = float(event.get("dx", 0))
            dy = float(event.get("dy", 0))
            nx = event.get("nx")
            ny = event.get("ny")
            x = vw / 2
            y = vh / 2
            if nx is not None and ny is not None:
                try:
                    x = max(0.0, min(1.0, float(nx))) * vw
                    y = max(0.0, min(1.0, float(ny))) * vh
                    await page.mouse.move(x, y)
                except (TypeError, ValueError) as e:
                    logger.debug("invalid browser wheel coordinates: %s", e)
            try:
                handled = bool(await page.evaluate(_WHEEL_SCROLL_JS, {"x": x, "y": y, "dx": dx, "dy": dy}))
            except Exception as e:
                logger.debug("browser js scroll failed: %s", e)
                handled = False
            if not handled:
                await page.mouse.wheel(dx, dy)
        elif etype == "key":
            key = event.get("key")
            if key:
                await page.keyboard.press(key)
        elif etype == "text":
            text = event.get("text")
            if text:
                await page.keyboard.type(text)
        elif etype == "navigate":
            url = event.get("url")
            if url:
                await page.goto(url, wait_until="domcontentloaded")
        elif etype == "back":
            await page.go_back(wait_until="domcontentloaded")
        elif etype == "forward":
            await page.go_forward(wait_until="domcontentloaded")
        elif etype == "activate_tab":
            index = event.get("index")
            if isinstance(index, int) and not isinstance(index, bool):
                await self._activate_tab(index)
        if etype != "move":
            self._schedule_input_live_frame()

    # 공개 API. 각각 전용 loop로 작업을 넘긴다.
    async def navigate(self, url: str) -> PageSnapshot:
        with self._activity():
            return await self._loop.run(self._navigate(url))

    async def snapshot(self) -> PageSnapshot:
        with self._activity():
            return await self._loop.run(self._snapshot())

    async def click(self, ref: int) -> PageSnapshot:
        with self._activity():
            return await self._loop.run(self._click(ref))

    async def type_text(self, ref: int, text: str, submit: bool = False) -> PageSnapshot:
        with self._activity():
            return await self._loop.run(self._type(ref, text, submit))

    async def get_text(self, max_chars: int = 8000) -> str:
        with self._activity():
            return await self._loop.run(self._get_text(max_chars))

    async def screenshot_bytes(self, full_page: bool = False) -> bytes:
        with self._activity():
            return await self._loop.run(self._screenshot_bytes(full_page))

    async def live_frame(self) -> bytes:
        with self._activity():
            return await self._loop.run(self._live_frame())

    async def push_live_frame(self) -> None:
        with self._activity():
            await self._loop.run(self._push_live_frame())

    def schedule_live_frames(self) -> None:
        self._loop.submit(self._push_live_frame())

    async def back(self) -> PageSnapshot:
        with self._activity():
            return await self._loop.run(self._back())

    async def current_url(self) -> str | None:
        with self._activity():
            return await self._loop.run(self._current_url())

    async def tabs(self) -> list[BrowserTab]:
        with self._activity():
            return await self._loop.run(self._tabs())

    async def start_screencast(self, on_frame: Callable[[bytes], None]) -> None:
        with self._activity():
            await self._loop.run(self._start_screencast(on_frame))

    async def stop_screencast(self, on_frame: Callable[[bytes], None] | None = None) -> None:
        with self._activity():
            await self._loop.run(self._stop_screencast(on_frame))

    async def dispatch_input(self, event: dict) -> None:
        with self._activity():
            await self._loop.run(self._dispatch_input(event))

    async def close(self) -> None:
        await self._loop.run(self._close())


class BrowserSessionManager:
    """thread별 browser session의 프로세스 로컬 registry.

    session은 ``thread_id``로 키를 잡고 각각 headless Chromium 프로세스를 소유한다. 그대로
    두면 장기 실행 다중 사용자 gateway에서 도구를 한 번이라도 쓴 thread마다 browser가 쌓여
    실제 메모리/FD 누수가 된다. 이를 막기 위해 ``get_session``은 ``idle_timeout_s``를 넘겨
    유휴 상태인 session을 lazy하게 제거하고, ``max_sessions`` 상한을 초과하면 가장 오래
    쓰이지 않은 비고정 session을 닫는다. 실행 중인 browser 연산과 Live WebSocket lease는
    참조 카운트로 관리되므로 사용 중인 session이 제거되는 일은 없다. 수용은 하드 상한이다.
    기존 session이 모두 고정되어 있으면 ``max_sessions``를 넘기는 대신 새 thread를 거부한다.
    제거는 전용 Playwright loop에서 fire-and-forget으로 수행되어 호출자를 막지 않으며,
    방금 요청된 thread는 항상 유지된다.
    """

    def __init__(self, *, max_sessions: int = _DEFAULT_MAX_SESSIONS, idle_timeout_s: float = _DEFAULT_IDLE_TIMEOUT_S) -> None:
        self._loop: _PlaywrightLoopThread | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._last_used: dict[str, float] = {}
        self._max_sessions = max_sessions
        self._idle_timeout_s = idle_timeout_s
        self._lock = threading.Lock()

    def _touch_session(self, key: str) -> None:
        with self._lock:
            if key in self._sessions:
                self._last_used[key] = time.monotonic()

    def _ensure_loop(self) -> _PlaywrightLoopThread:
        if self._loop is None:
            self._loop = _PlaywrightLoopThread()
        return self._loop

    def get_session(
        self,
        thread_id: str | None,
        *,
        headless: bool = True,
        timeout_ms: int = 30000,
        viewport: dict[str, int] | None = None,
        cdp_url: str | None = None,
        allow_unguarded_cdp: bool = False,
        url_guard: Callable[[str], str | None] | None = None,
        pin: bool = False,
    ) -> BrowserSession:
        ensure_browser_worker_compatibility()
        if cdp_url and not allow_unguarded_cdp:
            raise RuntimeError("cdp_url uses a browser context where DeerFlow cannot enforce its SSRF request guard; set allow_unguarded_cdp: true only for an explicitly trusted local Chrome session")
        key = thread_id or "default"
        now = time.monotonic()
        evicted: list[BrowserSession] = []
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                evicted.extend(self._collect_idle_locked(keep_key=key, now=now))
                if self._max_sessions > 0 and len(self._sessions) >= self._max_sessions:
                    lru = self._pop_lru_unpinned_locked(excluded_keys={key})
                    if lru is None:
                        raise BrowserSessionCapacityError(f"Browser session capacity is full ({self._max_sessions}); close an active Live browser before opening another")
                    evicted.append(lru)
                session = BrowserSession(
                    self._ensure_loop(),
                    headless=headless,
                    timeout_ms=timeout_ms,
                    viewport=viewport or {"width": 1280, "height": 720},
                    cdp_url=cdp_url,
                    url_guard=url_guard,
                    on_activity=lambda: self._touch_session(key),
                )
                self._sessions[key] = session
            if pin:
                session._pin()
            self._last_used[key] = now
            evicted.extend(self._collect_evictable_locked(keep_key=key, now=now))
        for evicted_session in evicted:
            self._schedule_close(evicted_session)
        return session

    @contextlib.contextmanager
    def acquire_session(self, thread_id: str | None, **kwargs: Any):
        """browser 연산 하나를 위해 원자적으로 고정된 session을 획득한다."""
        key = thread_id or "default"
        session = self.get_session(key, pin=True, **kwargs)
        try:
            yield session
        finally:
            self.release_session(key, session)

    def release_session(self, thread_id: str | None, session: BrowserSession) -> None:
        """lease를 반납하고, 제거 가능해지면 idle/LRU 상한을 다시 적용한다."""
        key = thread_id or "default"
        session._unpin()
        now = time.monotonic()
        with self._lock:
            evicted = self._collect_evictable_locked(keep_key=None, now=now) if self._sessions.get(key) is session else []
        for evicted_session in evicted:
            self._schedule_close(evicted_session)

    def _collect_evictable_locked(self, *, keep_key: str | None, now: float) -> list[BrowserSession]:
        """유휴이거나 상한을 넘은 session을 registry에서 빼고, 닫을 대상으로 반환한다.

        반드시 ``self._lock``을 잡은 상태에서 호출한다. ``keep_key``(방금 사용된 thread)가
        지정되면 절대 제거하지 않으므로 새 요청이 자신의 session을 잃지 않는다. lease 반납
        경로는 ``None``을 넘겨, 방금 고정 해제된 session이 다음 요청을 기다리지 않고 설정된
        상한을 회복하게 한다.
        """
        to_close: list[BrowserSession] = []
        to_close.extend(self._collect_idle_locked(keep_key=keep_key, now=now))
        if self._max_sessions > 0:
            excluded = {keep_key} if keep_key is not None else set()
            while len(self._sessions) > self._max_sessions:
                session = self._pop_lru_unpinned_locked(excluded_keys=excluded)
                if session is None:
                    break
                to_close.append(session)
        return to_close

    def _collect_idle_locked(self, *, keep_key: str | None, now: float) -> list[BrowserSession]:
        """유휴 상태이고 고정되지 않은 session을 registry에서 제거한다."""
        to_close: list[BrowserSession] = []
        if self._idle_timeout_s > 0:
            for other_key, last_used in list(self._last_used.items()):
                if other_key == keep_key:
                    continue
                session = self._sessions.get(other_key)
                if session is not None and session.active_refs:
                    continue
                if now - last_used >= self._idle_timeout_s:
                    session = self._sessions.pop(other_key, None)
                    self._last_used.pop(other_key, None)
                    if session is not None:
                        to_close.append(session)
        return to_close

    def _pop_lru_unpinned_locked(self, *, excluded_keys: set[str]) -> BrowserSession | None:
        candidates = [(last_used, other_key) for other_key, last_used in self._last_used.items() if other_key not in excluded_keys and (session := self._sessions.get(other_key)) is not None and not session.active_refs]
        if not candidates:
            return None
        _, lru_key = min(candidates)
        session = self._sessions.pop(lru_key, None)
        self._last_used.pop(lru_key, None)
        return session

    def _schedule_close(self, session: BrowserSession) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(Exception):
            loop.submit(session._close())

    async def close_session(self, thread_id: str | None) -> bool:
        key = thread_id or "default"
        with self._lock:
            session = self._sessions.pop(key, None)
            self._last_used.pop(key, None)
        if session is None:
            return False
        await session.close()
        return True

    async def close_all_sessions(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._last_used.clear()
        for session in sessions:
            await session.close()
        return len(sessions)


_manager: BrowserSessionManager | None = None
_manager_lock = threading.Lock()


def get_browser_session_manager() -> BrowserSessionManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = BrowserSessionManager()
    return _manager


def reset_browser_session_manager() -> None:
    """테스트용 hook. session을 닫지 않고 프로세스 로컬 manager만 버린다."""
    global _manager
    with _manager_lock:
        _manager = None
