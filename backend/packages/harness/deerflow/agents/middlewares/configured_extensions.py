"""config에 선언된 agent middleware 로딩."""

import logging
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware

from deerflow.reflection import resolve_class

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)


def load_configured_extension_middlewares(app_config: "AppConfig") -> list[AgentMiddleware]:
    """config에 선언된 agent middleware를 인스턴스화한다.

    각 항목은 ``module.path:ClassName`` 형식의 무인자 ``AgentMiddleware`` 클래스 경로다.
    import·속성·하위 클래스 검증은 의도적으로 공용 reflection resolver를 거친다. 그래야 실패
    메시지가 model, tool, sandbox provider, guardrail provider와 동일하게 실행 가능한 의존성
    힌트를 담는다.
    """
    middlewares: list[AgentMiddleware] = []
    for middleware_path in list(app_config.extensions.middlewares or []):
        middleware_cls = resolve_class(middleware_path, AgentMiddleware)
        try:
            middleware = middleware_cls()
        except Exception:
            logger.exception("Failed to instantiate configured extension middleware %s", middleware_path)
            raise
        middlewares.append(middleware)
    return middlewares
