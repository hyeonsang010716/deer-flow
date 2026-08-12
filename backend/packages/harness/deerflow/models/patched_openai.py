"""Gemini thinking 모델을 위해 thought_signature를 보존하는 patched ChatOpenAI.

OpenAI 호환 gateway(예: Vertex AI, Google AI Studio, 각종 proxy)를 통해 thinking을 켠 Gemini를
쓰면, API는 tool-call 객체의 ``thought_signature`` 필드를 이후 모든 요청에 원문 그대로 되돌려
보낼 것을 요구한다.

OpenAI 호환 gateway는 raw tool-call dict(``thought_signature`` 포함)를
``additional_kwargs["tool_calls"]``에 저장하지만, 표준 ``langchain_openai.ChatOpenAI``는 나가는
payload에 표준 필드(``id``, ``type``, ``function``)만 직렬화하면서 signature를 조용히 버린다.
그러면 HTTP 400 ``INVALID_ARGUMENT`` 오류가 난다:

    Unable to submit request because function call `<tool>` in the N. content
    block is missing a `thought_signature`.

이 모듈은 ``_get_request_payload``를 재정의해, 원래 signature를 갖고 있던 assistant message에
대해 tool-call signature를 나가는 payload에 다시 주입하는 방식으로 문제를 해결한다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from deerflow.models.assistant_payload_replay import restore_assistant_payloads


class PatchedChatOpenAI(ChatOpenAI):
    """OpenAI gateway 경유 Gemini thinking을 위해 ``thought_signature``를 보존하는 ChatOpenAI.

    OpenAI 호환 gateway로 thinking을 켠 Gemini를 쓰면, API는 multi-turn 대화의 tool-call 객체에
    ``thought_signature``가 있기를 기대한다. 이 patch 버전은 API로 보내기 전에
    ``AIMessage.additional_kwargs["tool_calls"]``에서 signature를 꺼내 직렬화된 request payload에
    복원한다.

    ``config.yaml``에서의 사용 예::

        - name: gemini-2.5-pro-thinking
          display_name: Gemini 2.5 Pro (Thinking)
          use: deerflow.models.patched_openai:PatchedChatOpenAI
          model: google/gemini-2.5-pro-preview
          api_key: $GEMINI_API_KEY
          base_url: https://<your-openai-compat-gateway>/v1
          max_tokens: 16384
          supports_thinking: true
          supports_vision: true
          when_thinking_enabled:
            extra_body:
              thinking:
                type: enabled
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """tool-call 객체의 ``thought_signature``를 보존한 request payload를 반환한다.

        부모 메서드를 재정의해서, LangChain이 ``additional_kwargs["tool_calls"]``에 저장했지만
        직렬화 과정에서 버려진 ``thought_signature`` 필드를 tool-call 객체에 다시 주입한다.
        """
        # 직렬화가 버릴 수 있는 필드에 접근하기 위해, 변환 *전에* 원본 LangChain message를
        # 확보한다.
        original_messages = self._convert_input(input_).to_messages()

        # 부모 구현에서 기본 payload를 얻는다.
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        restore_assistant_payloads(payload.get("messages", []), original_messages, _restore_tool_call_signatures)

        return payload


def _restore_tool_call_signatures(payload_msg: dict, orig_msg: AIMessage) -> None:
    """*payload_msg*의 tool-call 객체에 ``thought_signature``를 다시 주입한다.

    Gemini OpenAI 호환 gateway가 function call이 담긴 응답을 반환할 때, 각 tool-call 객체는
    ``thought_signature``를 가질 수 있다. LangChain은 raw tool-call dict를
    ``additional_kwargs["tool_calls"]``에 저장하지만, 나가는 payload에는 표준 필드
    (``id``, ``type``, ``function``)만 직렬화하면서 signature를 조용히 버린다.

    이 함수는 raw tool-call 항목을 ``id``로 매칭하고(없으면 위치 순서로 폴백) signature를
    직렬화된 payload 항목에 되돌려 복사한다.
    """
    raw_tool_calls: list[dict] = orig_msg.additional_kwargs.get("tool_calls") or []
    payload_tool_calls: list[dict] = payload_msg.get("tool_calls") or []

    if not raw_tool_calls or not payload_tool_calls:
        return

    # 효율적인 매칭을 위해 id → raw_tc 조회 테이블을 만든다.
    raw_by_id: dict[str, dict] = {}
    for raw_tc in raw_tool_calls:
        tc_id = raw_tc.get("id")
        if tc_id:
            raw_by_id[tc_id] = raw_tc

    for idx, payload_tc in enumerate(payload_tool_calls):
        # 먼저 id로 매칭하고, 안 되면 위치로 폴백한다.
        raw_tc = raw_by_id.get(payload_tc.get("id", ""))
        if raw_tc is None and idx < len(raw_tool_calls):
            raw_tc = raw_tool_calls[idx]

        if raw_tc is None:
            continue

        # gateway는 snake_case를 쓸 수도, camelCase를 쓸 수도 있다.
        sig = raw_tc.get("thought_signature") or raw_tc.get("thoughtSignature")
        if sig:
            payload_tc["thought_signature"] = sig
