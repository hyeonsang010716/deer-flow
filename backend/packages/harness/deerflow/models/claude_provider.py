"""OAuth Bearer 인증, prompt caching, smart thinking을 지원하는 커스텀 Claude provider.

두 가지 인증 모드를 지원한다:
  1. 표준 API key (x-api-key 헤더) — 기본 ChatAnthropic 동작
  2. Claude Code OAuth token (Authorization: Bearer 헤더)
     - sk-ant-oat 접두사로 감지한다
     - anthropic-beta: oauth-2025-04-20,claude-code-20250219 이 필요하다
     - 모든 OAuth 요청은 system prompt에 billing 헤더가 있어야 한다

명시적인 runtime handoff 경로에서 credential을 자동으로 읽어온다:
  - $ANTHROPIC_API_KEY 환경 변수
  - $CLAUDE_CODE_OAUTH_TOKEN 또는 $ANTHROPIC_AUTH_TOKEN
  - $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
  - $CLAUDE_CODE_CREDENTIALS_PATH
  - ~/.claude/.credentials.json
"""

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
THINKING_BUDGET_RATIO = 0.8

# OAuth token 접근 시 Anthropic API가 요구하는 billing 헤더.
# 반드시 첫 번째 system prompt 블록이어야 한다. 형식은 Claude Code CLI를 따른다.
# 하드코딩된 버전이 어긋나면 ANTHROPIC_BILLING_HEADER 환경 변수로 덮어쓴다.
_DEFAULT_BILLING_HEADER = "x-anthropic-billing-header: cc_version=2.1.85.351; cc_entrypoint=cli; cch=6c6d5;"
OAUTH_BILLING_HEADER = os.environ.get("ANTHROPIC_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


class ClaudeChatModel(ChatAnthropic):
    """OAuth Bearer 인증, prompt caching, smart thinking을 갖춘 ChatAnthropic.

    설정 예시:
        - name: claude-sonnet-4.6
          use: deerflow.models.claude_provider:ClaudeChatModel
          model: claude-sonnet-4-6
          max_tokens: 16384
          enable_prompt_caching: true
    """

    # 커스텀 필드
    enable_prompt_caching: bool = True
    prompt_cache_size: int = 3
    auto_thinking_budget: bool = True
    retry_max_attempts: int = MAX_RETRIES
    _is_oauth: bool = PrivateAttr(default=False)
    _oauth_access_token: str = PrivateAttr(default="")

    model_config = {"arbitrary_types_allowed": True}

    def _validate_retry_config(self) -> None:
        if self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be >= 1")

    def model_post_init(self, __context: Any) -> None:
        """credential을 자동으로 읽고, 필요하면 OAuth를 설정한다."""
        from pydantic import SecretStr

        from deerflow.models.credential_loader import (
            OAUTH_ANTHROPIC_BETAS,
            is_oauth_token,
            load_claude_code_credential,
        )

        self._validate_retry_config()

        # 실제 key 값을 꺼낸다(SecretStr.str()은 '**********'를 반환한다).
        current_key = ""
        if self.anthropic_api_key:
            if hasattr(self.anthropic_api_key, "get_secret_value"):
                current_key = self.anthropic_api_key.get_secret_value()
            else:
                current_key = str(self.anthropic_api_key)

        # 유효한 key가 없으면 명시적인 Claude Code OAuth handoff 소스를 시도한다.
        if not current_key or current_key in ("your-anthropic-api-key",):
            cred = load_claude_code_credential()
            if cred:
                current_key = cred.access_token
                logger.info(f"Using Claude Code CLI credential (source: {cred.source})")
            else:
                logger.warning("No Anthropic API key or explicit Claude Code OAuth credential found.")

        # OAuth token을 감지하고 Bearer 인증을 설정한다.
        if is_oauth_token(current_key):
            self._is_oauth = True
            self._oauth_access_token = current_key
            # token을 임시로 api_key에 넣는다(client에서 auth_token으로 교체된다).
            self.anthropic_api_key = SecretStr(current_key)
            # OAuth에 필요한 beta 헤더를 추가한다.
            self.default_headers = {
                **(self.default_headers or {}),
                "anthropic-beta": OAUTH_ANTHROPIC_BETAS,
            }
            # OAuth token은 cache_control 블록이 4개로 제한되므로 prompt caching을 끈다.
            self.enable_prompt_caching = False
            logger.info("OAuth token detected — will use Authorization: Bearer header")
        else:
            if current_key:
                self.anthropic_api_key = SecretStr(current_key)

        # api_key가 SecretStr인지 확인한다.
        if isinstance(self.anthropic_api_key, str):
            self.anthropic_api_key = SecretStr(self.anthropic_api_key)

        super().model_post_init(__context)

        # OAuth Bearer 인증을 위해 client 생성 직후 patch한다.
        # client가 lazy하게 생성되므로 super() 이후여야 한다.
        if self._is_oauth:
            self._patch_client_oauth(self._client)
            self._patch_client_oauth(self._async_client)

    def _patch_client_oauth(self, client: Any) -> None:
        """OAuth Bearer 인증을 위해 Anthropic SDK client의 api_key를 auth_token으로 바꾼다."""
        if hasattr(client, "api_key") and hasattr(client, "auth_token"):
            client.api_key = None
            client.auth_token = self._oauth_access_token

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """prompt caching, thinking budget, OAuth billing을 주입하기 위한 override."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if self._is_oauth:
            self._apply_oauth_billing(payload)

        if self.enable_prompt_caching:
            self._apply_prompt_caching(payload)

        if self.auto_thinking_budget:
            self._apply_thinking_budget(payload)

        return payload

    def _apply_oauth_billing(self, payload: dict) -> None:
        """모든 OAuth 요청에 필요한 billing 헤더 블록을 주입한다.

        billing 블록은 항상 system 리스트의 맨 앞에 놓고, 중복이나 순서 어긋남을 피하기 위해
        기존에 있던 것은 제거한다.
        """
        billing_block = {"type": "text", "text": OAUTH_BILLING_HEADER}

        system = payload.get("system")
        if isinstance(system, list):
            # 기존 billing 블록을 모두 제거한 뒤 index 0에 하나만 삽입한다.
            filtered = [b for b in system if not (isinstance(b, dict) and OAUTH_BILLING_HEADER in b.get("text", ""))]
            payload["system"] = [billing_block] + filtered
        elif isinstance(system, str):
            if OAUTH_BILLING_HEADER in system:
                payload["system"] = [billing_block]
            else:
                payload["system"] = [billing_block, {"type": "text", "text": system}]
        else:
            payload["system"] = [billing_block]

        # OAuth billing 검증을 위해 API가 요구하는 metadata.user_id를 추가한다.
        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = {}
        if "user_id" not in payload["metadata"]:
            # 머신의 hostname에서 안정적인 device_id를 만든다.
            hostname = socket.gethostname()
            device_id = hashlib.sha256(f"deerflow-{hostname}".encode()).hexdigest()
            session_id = str(uuid.uuid4())
            payload["metadata"]["user_id"] = json.dumps(
                {
                    "device_id": device_id,
                    "account_uuid": "deerflow",
                    "session_id": session_id,
                }
            )

    def _apply_prompt_caching(self, payload: dict) -> None:
        """system, 최근 메시지, 마지막 tool 정의에 ephemeral cache_control을 적용한다.

        Anthropic API와 AWS Bedrock이 모두 강제하는 하드 리밋인 MAX_CACHE_BREAKPOINTS(4)개의
        breakpoint 예산을 쓴다. breakpoint는 조건에 맞는 블록 중 *마지막* 것들에 붙인다.
        뒤쪽 breakpoint일수록 더 긴 prefix를 덮어 cache hit율이 좋기 때문이다.

        system prompt는 완전히 정적이라고 가정한다(사용자별 memory나 현재 날짜가 없다).
        동적 context는 DynamicContextMiddleware가 첫 HumanMessage 안에 <system-reminder>로
        턴마다 주입한다.
        """
        MAX_CACHE_BREAKPOINTS = 4

        # 후보 블록을 문서 순서대로 모은다:
        #   1. system text 블록
        #   2. 최근 prompt_cache_size개 메시지의 content 블록
        #   3. 마지막 tool 정의
        candidates: list[dict] = []

        # 1. system 블록
        system = payload.get("system")
        if system and isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    candidates.append(block)
        elif system and isinstance(system, str):
            new_block: dict = {"type": "text", "text": system}
            payload["system"] = [new_block]
            candidates.append(new_block)

        # 2. 최근 메시지 블록
        messages = payload.get("messages", [])
        cache_start = max(0, len(messages) - self.prompt_cache_size)
        for i in range(cache_start, len(messages)):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        candidates.append(block)
            elif isinstance(content, str) and content:
                new_block = {"type": "text", "text": content}
                msg["content"] = [new_block]
                candidates.append(new_block)

        # 3. 마지막 tool 정의
        tools = payload.get("tools", [])
        if tools and isinstance(tools[-1], dict):
            candidates.append(tools[-1])

        # API 제한을 넘지 않도록 마지막 MAX_CACHE_BREAKPOINTS개 후보에만
        # cache_control을 적용한다.
        for block in candidates[-MAX_CACHE_BREAKPOINTS:]:
            block["cache_control"] = {"type": "ephemeral"}

    def _apply_thinking_budget(self, payload: dict) -> None:
        """thinking budget을 자동 배정한다(max_tokens의 80%)."""
        thinking = payload.get("thinking")
        if not thinking or not isinstance(thinking, dict):
            return
        if thinking.get("type") != "enabled":
            return
        if thinking.get("budget_tokens"):
            return

        max_tokens = payload.get("max_tokens", 8192)
        thinking["budget_tokens"] = int(max_tokens * THINKING_BUDGET_RATIO)

    @staticmethod
    def _strip_cache_control(payload: dict) -> None:
        """OAuth 요청이 Anthropic에 도달하기 전에 cache_control 마커를 제거한다."""
        for section in ("system", "messages"):
            items = payload.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item.pop("cache_control", None)
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block.pop("cache_control", None)

        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool.pop("cache_control", None)

    def _create(self, payload: dict) -> Any:
        if self._is_oauth:
            self._strip_cache_control(payload)
        return super()._create(payload)

    async def _acreate(self, payload: dict) -> Any:
        if self._is_oauth:
            self._strip_cache_control(payload)
        return await super()._acreate(payload)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """OAuth patch와 retry 로직을 더한 override."""
        if self._is_oauth:
            self._patch_client_oauth(self._client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return super()._generate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
        raise last_error

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """OAuth patch와 retry 로직을 더한 async override."""
        import asyncio

        if self._is_oauth:
            self._patch_client_oauth(self._async_client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return await super()._agenerate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
        raise last_error

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        """고정 20% 버퍼를 더한 exponential backoff."""
        backoff_ms = 2000 * (1 << (attempt - 1))
        jitter_ms = int(backoff_ms * 0.2)
        total_ms = backoff_ms + jitter_ms

        if hasattr(error, "response") and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    total_ms = int(retry_after) * 1000
                except (ValueError, TypeError):
                    pass

        return total_ms
