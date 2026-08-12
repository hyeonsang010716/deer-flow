"""신뢰할 수 없는 도구 결과의 prompt-injection 제어 토큰을 무력화한다.

DeerFlow는 이미 실제 사용자 메시지를 신뢰하지 않고 그 안의 framework/injection 태그를
무력화한다(``InputSanitizationMiddleware`` 참고). 에이전트가 *가져오는* 원격 콘텐츠도
똑같이 신뢰할 수 없다. ``web_fetch`` / ``web_search`` / ``image_search``가 돌려주는 웹 페이지
본문과 검색 snippet, ``web_capture``가 노출하는 대상 사이트의 response-status 텍스트가
그렇다. 그런데도 이들은 그대로 모델 context에 들어갔다. 공격자가 통제하는 페이지가 위조된
``<system-reminder>`` 블록(또는 ``--- END USER INPUT ---`` 마커)을 심어 모델에게 권위 있는
framework context로 전달할 수 있었다.

이 middleware는 자체 네트워크 도구의 결과에 *동일한* 구조적 무력화
(``neutralize_untrusted_tags``)를 적용해 그 틈을 좁힌다. 가져온 ``<system-reminder>``는
사용자 입력에서와 똑같이 ``&lt;system-reminder&gt;``로 escape된다. 대상은 의도적으로
원격 콘텐츠 도구로 한정한다. 로컬 도구 출력(bash, 파일 읽기)은 건드리지 않아 정상적인
코드/로그 내용이 망가지지 않는다.

범위 주의: 매칭이 이름 기반 allowlist라 다른 이름으로 등록된 MCP 원격 콘텐츠 도구는 아직
포함되지 않는다. ``_REMOTE_CONTENT_TOOL_NAMES``를 참고한다.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace as dc_replace
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

# 결과가 공격자의 영향을 받을 수 있는 원격 콘텐츠인 도구 이름들. 자체 search/fetch provider는
# 모두 ``web_fetch`` / ``web_search`` / ``image_search``로 정규화되므로(community/*/tools.py
# 참고) 이 집합은 provider와 무관하다. ``web_capture``(Browserless 스크린샷)는 대상 사이트의
# response-status 텍스트(``X-Response-Status``, 캡처 대상 서버가 통제하는 자유 형식 문구)를
# 결과 메시지에 노출하므로 역시 신뢰할 수 없는 원격 콘텐츠이고 여기에 속한다.
#
# 알려진 한계: 이 gate는 이름 기반이다. MCP 서버는 원격 콘텐츠 도구를 임의의 이름
# (예: ``fetch_url`` / ``scrape_page``)으로 노출할 수 있고, 그 결과도 똑같이 신뢰할 수 없지만
# 여기서 매칭되지 않아 무력화 없이 모델에 도달한다. fetch/search/crawl 부분 문자열을 보는
# 이름 휴리스틱은 정상적인 *로컬* 도구 출력(예: ``file_search`` 결과)까지 망가뜨리므로
# 의도적으로 피한다. 견고한 MCP 대응은 이름이 아니라 등록 시 메타데이터로 원격 콘텐츠 도구를
# 표시하는 방식이어야 하며 후속 작업으로 추적 중이다.
_REMOTE_CONTENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "web_fetch",
        "web_search",
        "image_search",
        "web_capture",
    }
)


def _neutralize_content(content: object) -> object:
    """*content*의 형태를 유지한 채 신뢰할 수 없는 태그를 무력화해 반환한다.

    ToolMessage content가 가질 수 있는 두 형태를 처리한다.

    * 평범한 ``str``(현재 모든 web 도구가 반환하는 형태).
    * 콘텐츠 블록 리스트. 맨 ``str`` 요소와 ``{"type": "text", "text": ...}`` 텍스트 블록은
      재작성하고, 텍스트가 아닌 블록(이미지 등)은 그대로 통과시킨다. 맨 ``str`` 처리는
      콘텐츠 리스트 안의 ``str`` 항목을 이미 고려하는
      ``ToolOutputBudgetMiddleware._message_text``와 같다.
    """
    # 테스트가 input-sanitization 모듈을 stub해도 이 모듈을 로드할 수 있도록, 그리고
    # 코드베이스의 지연 import 방식에 맞추기 위해 lazy import한다.
    from deerflow.agents.middlewares.input_sanitization_middleware import neutralize_untrusted_tags

    if isinstance(content, str):
        return neutralize_untrusted_tags(content)
    if isinstance(content, list):
        rebuilt: list[object] = []
        for block in content:
            if isinstance(block, str):
                rebuilt.append(neutralize_untrusted_tags(block))
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                rebuilt.append({**block, "text": neutralize_untrusted_tags(block["text"])})
            else:
                rebuilt.append(block)
        return rebuilt
    return content


def _sanitize_tool_message(message: ToolMessage) -> ToolMessage:
    """content를 무력화한 *message* 사본을 반환한다. 바뀐 게 없으면 원본을 반환한다."""
    new_content = _neutralize_content(message.content)
    if new_content == message.content:
        return message
    return message.model_copy(update={"content": new_content})


def _sanitize_result(result: ToolMessage | Command) -> ToolMessage | Command:
    """도구 호출 결과(``ToolMessage`` 또는 ``Command``)를 무력화한다."""
    if isinstance(result, ToolMessage):
        return _sanitize_tool_message(result)
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list) and any(isinstance(m, ToolMessage) for m in messages):
            new_messages = [_sanitize_tool_message(m) if isinstance(m, ToolMessage) else m for m in messages]
            if new_messages != messages:
                return dc_replace(result, update={**update, "messages": new_messages})
    return result


class ToolResultSanitizationMiddleware(AgentMiddleware[AgentState]):
    """모델이 보기 전에 원격 도구 결과의 injection/framework 태그를 escape한다.

    자체 네트워크 도구(``web_fetch`` / ``web_search`` / ``image_search`` /
    ``web_capture``)의 결과만 재작성하고 다른 도구의 출력은 그대로 반환한다. 사용자 입력
    guardrail을 그대로 반영해 신뢰할 수 없는 원격 콘텐츠와 사용자 입력이 동일한 구조적
    무력화를 받게 한다.

    범위는 이름 기반 allowlist(``_REMOTE_CONTENT_TOOL_NAMES``)다. 로컬 도구에 오탐 없이
    내장 web 도구를 확실히 덮는다. 다른 이름으로 등록된 MCP 원격 콘텐츠 도구는 덮지 않는다.
    이름 휴리스틱을 피하는 이유와 메타데이터 표시 후속 작업은
    ``_REMOTE_CONTENT_TOOL_NAMES``의 설명을 참고한다.
    """

    def _should_sanitize(self, request: ToolCallRequest) -> bool:
        return request.tool_call.get("name") in _REMOTE_CONTENT_TOOL_NAMES

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if not self._should_sanitize(request):
            return result
        return _sanitize_result(result)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if not self._should_sanitize(request):
            return result
        return _sanitize_result(result)
