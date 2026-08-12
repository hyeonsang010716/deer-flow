"""Phase 1B authorization 집행 공용 헬퍼."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from deerflow.authz.provider import AuthorizationProvider, Principal

logger = logging.getLogger(__name__)


def filter_tools_by_authorization(
    tools: Sequence[BaseTool],
    *,
    provider: AuthorizationProvider | None,
    principal: Principal,
    fail_closed: bool,
) -> list[BaseTool]:
    """*tools* 중 policy상 보이는 것만 원래 순서 그대로 반환한다.

    호출자는 deferred-tool 조립 전에 이 함수를 호출해야 한다. ``fail_closed``가 true면
    provider 오류나 잘못된 필터 결과는 모든 tool을 거부하고, fail-open으로 명시 설정된
    정책이면 원래 집합을 그대로 유지한다.
    """
    original_tools = list(tools)
    if provider is None:
        return original_tools

    candidates = [tool.name for tool in original_tools]
    try:
        allowed = provider.filter_resources(principal, "tool", candidates)
        if not isinstance(allowed, list) or any(not isinstance(name, str) for name in allowed):
            raise TypeError("AuthorizationProvider.filter_resources must return list[str]")
    except Exception:
        logger.exception("Authorization provider failed while filtering tools")
        return [] if fail_closed else original_tools

    allowed_names = set(allowed)
    return [tool for tool in original_tools if tool.name in allowed_names]
