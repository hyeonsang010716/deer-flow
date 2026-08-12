"""thread 제목을 자동 생성하는 미들웨어."""

import logging
import re
from typing import TYPE_CHECKING, Any, NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder
from deerflow.config.title_config import get_title_config
from deerflow.models import create_chat_model

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.config.title_config import TitleConfig

logger = logging.getLogger(__name__)


class TitleMiddlewareState(AgentState):
    """`ThreadState` 스키마와 호환된다."""

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """첫 사용자 메시지 이후 thread 제목을 자동으로 생성한다."""

    state_schema = TitleMiddlewareState

    def __init__(self, *, app_config: "AppConfig | None" = None, title_config: "TitleConfig | None" = None):
        super().__init__()
        self._app_config = app_config
        self._title_config = title_config

    def _get_title_config(self):
        if self._title_config is not None:
            return self._title_config
        if self._app_config is not None:
            return self._app_config.title
        return get_title_config()

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        return ""

    @staticmethod
    def _message_type(message: object) -> str | None:
        message_type = getattr(message, "type", None)
        if message_type is None and isinstance(message, dict):
            message_type = message.get("type") or message.get("role")
        if message_type == "user":
            return "human"
        if message_type == "assistant":
            return "ai"
        return message_type if isinstance(message_type, str) else None

    @staticmethod
    def _message_content(message: object) -> object:
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    @staticmethod
    def _is_dynamic_context_reminder_message(message: object) -> bool:
        if is_dynamic_context_reminder(message):
            return True
        if isinstance(message, dict):
            additional_kwargs = message.get("additional_kwargs")
            return isinstance(additional_kwargs, dict) and bool(additional_kwargs.get("dynamic_context_reminder"))
        return False

    @staticmethod
    def _is_user_message_for_title(message: object) -> bool:
        return TitleMiddleware._message_type(message) == "human" and not TitleMiddleware._is_dynamic_context_reminder_message(message)

    def _get_title_user_message(self, state: TitleMiddlewareState) -> str:
        messages = state.get("messages") or []
        user_msg_content = next((self._message_content(m) for m in messages if self._is_user_message_for_title(m)), "")
        return self._normalize_content(user_msg_content)

    def _should_generate_title(self, state: TitleMiddlewareState, *, allow_partial_exchange: bool = False) -> bool:
        """이 thread에 제목을 생성해야 하는지 판단한다."""
        config = self._get_title_config()
        if not config.enabled:
            return False

        # state에 이미 제목이 있는지 확인한다.
        if state.get("title"):
            return False

        # 첫 턴인지 확인한다(사용자 메시지 하나와 assistant 응답 하나 이상).
        # 부분 초기화된 checkpoint를 읽을 때 ``messages`` 채널이 None일 수 있으므로, ``len()``이 안전하도록
        # 방어적으로 빈 리스트로 바꾼다.
        messages = state.get("messages") or []
        min_messages = 1 if allow_partial_exchange else 2
        if len(messages) < min_messages:
            return False

        # 사용자 메시지와 assistant 메시지를 센다.
        user_messages = [m for m in messages if self._is_user_message_for_title(m)]
        assistant_messages = [m for m in messages if self._message_type(m) == "ai"]

        # 일반 경로에서는 첫 완전한 주고받기 이후에만 제목을 만든다. 중단 경로
        # (``allow_partial_exchange=True``)는 첫 턴의 사용자 메시지 하나만으로도 허용해, AI chunk가
        # checkpoint에 닿기 전에 run이 취소되어도 fallback 제목을 남길 수 있게 한다.
        return len(user_messages) == 1 and (len(assistant_messages) >= 1 or allow_partial_exchange)

    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        """사용자/assistant 메시지를 뽑아 제목 prompt를 만든다.

        호출자가 user_msg를 fallback으로 쓸 수 있도록 (prompt_string, user_msg)를 반환한다.
        """
        config = self._get_title_config()
        messages = state.get("messages") or []

        assistant_msg_content = next((self._message_content(m) for m in messages if self._message_type(m) == "ai"), "")

        user_msg = self._get_title_user_message(state)
        assistant_msg = self._strip_think_tags(self._normalize_content(assistant_msg_content))

        prompt = config.prompt_template.format(
            max_words=config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg

    def _strip_think_tags(self, text: str) -> str:
        """reasoning 모델(예: minimax, DeepSeek-R1)이 내보내는 <think>...</think> 블록을 제거한다."""
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    def _parse_title(self, content: object) -> str:
        """모델 출력을 정돈된 제목 문자열로 정규화한다."""
        config = self._get_title_config()
        title_content = self._normalize_content(content)
        title_content = self._strip_think_tags(title_content)
        title = title_content.strip().strip('"').strip("'")
        return title[: config.max_chars] if len(title) > config.max_chars else title

    def _fallback_title(self, user_msg: str) -> str:
        config = self._get_title_config()
        fallback_chars = min(config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            # 말줄임표 자리를 미리 확보해, 이 경로도 모델 경로의 ``_parse_title``과 똑같이
            # ``max_chars``를 지키게 한다.
            ellipsis = "..."
            body = min(fallback_chars, config.max_chars - len(ellipsis))
            return user_msg[:body].rstrip() + ellipsis
        return user_msg if user_msg else "New Conversation"

    def _get_runnable_config(self) -> dict[str, Any]:
        """부모 RunnableConfig를 상속하고 미들웨어 tag를 추가한다.

        덕분에 RunJournal이 이 미들웨어의 LLM 호출을 ``lead_agent``가 아니라 ``middleware:title``로 식별한다.
        """
        try:
            parent = get_config()
        except Exception:
            parent = {}
        config = {**parent}
        config["run_name"] = "title_agent"
        config["tags"] = [
            *(config.get("tags") or []),
            "middleware:title",
            TAG_NOSTREAM,
        ]
        return config

    def _generate_title_result(self, state: TitleMiddlewareState, *, allow_partial_exchange: bool = False) -> dict | None:
        """LLM 호출로 블로킹하지 않고 로컬 fallback 제목을 만든다."""
        if not self._should_generate_title(state, allow_partial_exchange=allow_partial_exchange):
            return None

        user_msg = self._get_title_user_message(state)
        return {"title": self._fallback_title(user_msg)}

    async def _agenerate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """설정된 LLM으로 제목을 비동기 생성하고, 실패하면 로컬 fallback을 쓴다."""
        if not self._should_generate_title(state):
            return None

        config = self._get_title_config()
        if not config.model_name:
            user_msg = self._get_title_user_message(state)
            return {"title": self._fallback_title(user_msg)}

        user_msg = self._get_title_user_message(state)

        try:
            prompt, user_msg = self._build_title_prompt(state)
            # ``_get_runnable_config()``가 graph 수준 RunnableConfig(``_make_lead_agent``에서 설정)를
            # 상속하고 그 callbacks에 이미 tracing handler가 들어 있으므로 attach_tracing=False를 쓴다.
            # model 수준에서 다시 바인딩하면 span이 중복 생성된다.
            model_kwargs = {"thinking_enabled": False, "attach_tracing": False}
            if self._app_config is not None:
                model_kwargs["app_config"] = self._app_config
            model = create_chat_model(name=config.model_name, **model_kwargs)
            response = await model.ainvoke(prompt, config=self._get_runnable_config())
            title = self._parse_title(response.content)
            if title:
                return {"title": title}
        except Exception:
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)
        return {"title": self._fallback_title(user_msg)}

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return self._generate_title_result(state)

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return await self._agenerate_title_result(state)
