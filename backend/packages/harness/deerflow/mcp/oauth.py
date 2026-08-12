"""MCP HTTP/SSE 서버를 위한 OAuth token 지원."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from deerflow.config.extensions_config import ExtensionsConfig, McpOAuthConfig

logger = logging.getLogger(__name__)


@dataclass
class _OAuthToken:
    """캐시된 OAuth token."""

    access_token: str
    token_type: str
    expires_at: datetime


class OAuthTokenManager:
    """MCP 서버용 OAuth token을 발급/캐시/갱신한다."""

    def __init__(self, oauth_by_server: dict[str, McpOAuthConfig]):
        self._oauth_by_server = oauth_by_server
        self._tokens: dict[str, _OAuthToken] = {}
        # asyncio.Lock이 아니라 평범한 threading.Lock을 쓴다. 임베디드/TUI의 동기 tool-call
        # 경로(DeerFlowClient.stream() -> LangGraph ToolNode._func -> ThreadPoolExecutor ->
        # deerflow.tools.sync.make_sync_tool_wrapper의 호출별 asyncio.run())는 동시 tool call
        # 마다 새 OS thread의 새 event loop에서 get_authorization_header를 호출한다.
        # asyncio.Lock은 처음 경합한 loop에 묶이므로, 두 번째 호출자의 release/wake-up이
        # call_soon_threadsafe 없이 loop를 넘나들면 조용히 deadlock에 빠지거나 "bound to a
        # different event loop" 예외가 난다. threading.Lock은 loop 종속성이 없어 같은 서버의
        # lock을 몇 개의 event loop/thread가 호출하든 안전하게 공유할 수 있다.
        self._locks: dict[str, threading.Lock] = {name: threading.Lock() for name in oauth_by_server}

    @classmethod
    def from_extensions_config(cls, extensions_config: ExtensionsConfig) -> OAuthTokenManager:
        oauth_by_server: dict[str, McpOAuthConfig] = {}
        for server_name, server_config in extensions_config.get_enabled_mcp_servers().items():
            if server_config.oauth and server_config.oauth.enabled:
                oauth_by_server[server_name] = server_config.oauth
        return cls(oauth_by_server)

    def has_oauth_servers(self) -> bool:
        return bool(self._oauth_by_server)

    def oauth_server_names(self) -> list[str]:
        return list(self._oauth_by_server.keys())

    async def get_authorization_header(self, server_name: str) -> str | None:
        oauth = self._oauth_by_server.get(server_name)
        if not oauth:
            return None

        token = self._tokens.get(server_name)
        if token and not self._is_expiring(token, oauth):
            return f"{token.token_type} {token.access_token}"

        lock = self._locks[server_name]
        # blocking 대기가 이 event loop를 막지 않도록 OS 수준 lock을 별도 thread에서 획득하고,
        # 해제는 동기적으로 한다(release()는 블로킹하지 않는다). 이렇게 하면 예전
        # `async with lock:`의 중복 제거 동작(서버당 동시 호출자 중 하나만 실제로 token을
        # 가져온다)을 유지하면서, 호출자가 서로 다른 event loop/thread에 있어도 안전하다.
        #
        # 획득 자체는 명시적 Task로 실행하며 이 coroutine의 취소로부터 shield한다. 맨
        # `await asyncio.to_thread(lock.acquire)`는 안전하게 취소할 수 없다. executor thread가
        # lock.acquire()를 시작한 뒤에는 Python이 그것을 멈출 방법이 없으므로, 그 await에서
        # 취소가 전달돼도 thread는 나중에(현재 보유자가 해제할 때) 계속 lock을 획득한다. 그때는
        # 이 coroutine이 이미 사라져 release()를 호출할 주체가 없어 lock이 영원히 잠긴 채로
        # 남고, 이 서버에 대한 이후 모든 호출이 같은 줄에서 영구히 막힌다. 획득 task를 shield하면
        # 취소된 호출자도 (멈출 수 없는) 획득이 실제로 끝날 때까지 기다렸다가 곧바로 lock을
        # 해제할 수 있어 소유권이 새지 않는다.
        acquire_task = asyncio.create_task(asyncio.to_thread(lock.acquire), name=f"oauth-lock-acquire:{server_name}")
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            # 정리 도중 이 coroutine이 다시 취소되더라도, 획득이 실제로 끝날 때까지 매번
            # shield하면서 계속 기다린다. 바탕 thread는 중단할 수 없으므로, lock이 우리 것이
            # 되는 시점을 알아내 영원히 잠긴 채 두지 않고 즉시 해제하는 유일한 방법이다.
            while not acquire_task.done():
                try:
                    await asyncio.shield(acquire_task)
                except asyncio.CancelledError:
                    continue
            lock.release()
            raise
        try:
            token = self._tokens.get(server_name)
            if token and not self._is_expiring(token, oauth):
                return f"{token.token_type} {token.access_token}"

            fresh = await self._fetch_token(oauth)
            self._tokens[server_name] = fresh
            logger.info(f"Refreshed OAuth access token for MCP server: {server_name}")
            return f"{fresh.token_type} {fresh.access_token}"
        finally:
            lock.release()

    @staticmethod
    def _is_expiring(token: _OAuthToken, oauth: McpOAuthConfig) -> bool:
        now = datetime.now(UTC)
        return token.expires_at <= now + timedelta(seconds=max(oauth.refresh_skew_seconds, 0))

    async def _fetch_token(self, oauth: McpOAuthConfig) -> _OAuthToken:
        import httpx  # pyright: ignore[reportMissingImports]

        data: dict[str, str] = {
            "grant_type": oauth.grant_type,
            **oauth.extra_token_params,
        }

        if oauth.scope:
            data["scope"] = oauth.scope
        if oauth.audience:
            data["audience"] = oauth.audience

        if oauth.grant_type == "client_credentials":
            if not oauth.client_id or not oauth.client_secret:
                raise ValueError("OAuth client_credentials requires client_id and client_secret")
            data["client_id"] = oauth.client_id
            data["client_secret"] = oauth.client_secret
        elif oauth.grant_type == "refresh_token":
            if not oauth.refresh_token:
                raise ValueError("OAuth refresh_token grant requires refresh_token")
            data["refresh_token"] = oauth.refresh_token
            if oauth.client_id:
                data["client_id"] = oauth.client_id
            if oauth.client_secret:
                data["client_secret"] = oauth.client_secret
        else:
            raise ValueError(f"Unsupported OAuth grant type: {oauth.grant_type}")

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(oauth.token_url, data=data)
            response.raise_for_status()
            payload = response.json()

        access_token = payload.get(oauth.token_field)
        if not access_token:
            raise ValueError(f"OAuth token response missing '{oauth.token_field}'")

        # 이후 갱신이 최신 값을 쓰도록 회전된 refresh_token을 보관한다. 프로세스 내부 갱신일
        # 뿐이며 의도적으로 extensions_config.json에 다시 쓰지 않는다. refresh token을 회전시키는
        # provider(Auth0, Okta, Google 등)는 갱신할 때마다 새 refresh_token을 돌려주므로, 그것을
        # 버리면 다음 갱신이 invalid_grant로 실패한다.
        if oauth.grant_type == "refresh_token":
            rotated = payload.get("refresh_token")
            if isinstance(rotated, str) and rotated:
                oauth.refresh_token = rotated

        token_type = str(payload.get(oauth.token_type_field, oauth.default_token_type) or oauth.default_token_type)

        expires_in_raw = payload.get(oauth.expires_in_field, 3600)
        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600

        expires_at = datetime.now(UTC) + timedelta(seconds=max(expires_in, 1))
        return _OAuthToken(access_token=access_token, token_type=token_type, expires_at=expires_at)


def build_oauth_tool_interceptor(extensions_config: ExtensionsConfig) -> Any | None:
    """OAuth Authorization 헤더를 주입하는 tool interceptor를 만든다."""
    token_manager = OAuthTokenManager.from_extensions_config(extensions_config)
    if not token_manager.has_oauth_servers():
        return None

    async def oauth_interceptor(request: Any, handler: Any) -> Any:
        header = await token_manager.get_authorization_header(request.server_name)
        if not header:
            return await handler(request)

        updated_headers = dict(request.headers or {})
        updated_headers["Authorization"] = header
        return await handler(request.override(headers=updated_headers))

    return oauth_interceptor


async def get_initial_oauth_headers(extensions_config: ExtensionsConfig) -> dict[str, str]:
    """MCP 서버 연결에 사용할 초기 OAuth Authorization 헤더를 가져온다."""
    token_manager = OAuthTokenManager.from_extensions_config(extensions_config)
    if not token_manager.has_oauth_servers():
        return {}

    headers: dict[str, str] = {}
    for server_name in token_manager.oauth_server_names():
        try:
            value = await token_manager.get_authorization_header(server_name)
        except Exception:
            logger.warning(
                "Skipping initial OAuth header for MCP server '%s' after token fetch failed",
                server_name,
                exc_info=True,
            )
            continue
        if value:
            headers[server_name] = value

    return {name: value for name, value in headers.items() if value}
