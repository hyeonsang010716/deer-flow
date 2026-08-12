"""multi-turn 대화에서 reasoning_content를 보존하도록 patch한 ChatDeepSeek.

메시지를 API로 되돌려 보낼 때 reasoning_content를 제대로 처리하는 ChatDeepSeek 변형을 제공한다.
원본 구현은 reasoning_content를 additional_kwargs에 저장하지만 이후 API 호출에는 포함하지 않는다.
그래서 thinking mode가 켜졌을 때 모든 assistant 메시지에 reasoning_content를 요구하는 API에서
에러가 난다.
"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek

from deerflow.models.assistant_payload_replay import restore_assistant_payloads, restore_reasoning_content


class PatchedChatDeepSeek(ChatDeepSeek):
    """reasoning_content를 제대로 보존하는 ChatDeepSeek.

    thinking/reasoning이 켜진 model을 쓰면 API는 multi-turn 대화의 모든 assistant 메시지에
    reasoning_content가 있기를 기대한다. 이 patch 버전은 additional_kwargs의
    reasoning_content가 request payload에 포함되도록 보장한다.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "DEEPSEEK_API_KEY", "openai_api_key": "DEEPSEEK_API_KEY"}

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """reasoning_content를 보존한 request payload를 반환한다.

        부모 메서드를 override해, additional_kwargs의 reasoning_content를 payload의 assistant
        메시지에 주입한다.
        """
        # 변환 전 원본 메시지를 가져온다
        original_messages = self._convert_input(input_).to_messages()

        # 부모를 호출해 기본 payload를 얻는다
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        restore_assistant_payloads(
            payload.get("messages", []),
            original_messages,
            restore_reasoning_content,
        )

        return payload
