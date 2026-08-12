"""prompt injection 방어를 위한 입력 guardrail middleware (issue #3630).

마지막 진짜 user 메시지에서 차단 대상 XML 유사 태그를 escape해(예: ``<system>`` →
``&lt;system&gt;``) 구조화된 context 마커가 아니라 리터럴 텍스트로 렌더링되게 한다.
덕분에 사용자 의도("DeerFlow의 <think> 태그는 어떻게 쓰나요?")는 보존하면서 injection
시도는 무력화한다 — AWS Bedrock의 PII ANONYMIZE와 같은, 거부하지 않고 비식별화하는 전략이다.

차단 대상: 시스템 예약 태그(memory, analysis 등) + 흔한 injection 태그(system,
instruction, role 등). 일반 HTML/XML 태그(<div>, <span>)는 escape하지 **않는다**.

정상 입력은 2차 의미 방어로 평문 boundary 마커로 감싼다(OWASP structured-prompt 지침).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphBubbleUp

from deerflow.agents.human_input import read_human_input_response
from deerflow.utils.messages import ORIGINAL_USER_CONTENT_KEY, message_content_to_text

logger = logging.getLogger(__name__)

_SUMMARY_MESSAGE_NAME = "summary"

# 차단 대상 태그 이름의 유한 집합: 시스템 예약 태그 + 흔한 injection 패턴.
#
# 유지보수: 시스템이 모델 입력으로 내보내는 framework 블록 태그를 추가할 때는
# test_input_sanitization_middleware.py::test_denylist_covers_framework_authority_blocks의
# 기대 개수도 반드시 갱신해야 한다. 이 테스트가 차단 태그의 정확한 개수를 고정하고 있어,
# 대응하는 regression guard 없이는 새 framework 태그를 추가할 수 없다.
_BLOCKED_TAG_NAMES: frozenset[str] = frozenset(
    {
        # framework가 주입하는 구조화/권위 블록. lead-agent 시스템 프롬프트의
        # "System-Context Confidentiality" 절(agents/lead_agent/prompt.py)은 그런 태그
        # *전부*를 신뢰된 내부 데이터로 선언한다 — 몇 개를 나열한 뒤 "and all other
        # structured tags"라고 한다. 따라서 denylist는 임의로 고른 부분집합이 아니라
        # framework의 권위 블록 전체를 하나의 부류로 다뤄야 한다. 그중 어느 하나라도
        # 신뢰할 수 없는 입력에서 위조되면 신뢰된 framework context를 흉내 내기 때문이다.
        # 목록은 framework가 실제로 모델 입력에 내보내는 블록 태그(시스템 프롬프트 +
        # hidden-context/reminder middleware)에서 뽑았고,
        # test_input_sanitization_middleware.py::test_denylist_covers_framework_authority_blocks가
        # drift를 막는다. reminder 블록의 두 표기를 모두 포함한다: "system-reminder"
        # (dynamic-context)와 "system_reminder"(todo/terminal middleware).
        #
        # subagent도 이 denylist를 공유한다. build_subagent_runtime_middlewares가 같은
        # _build_runtime_middlewares base를 재사용하므로 두 sanitization 경로 모두 subagent
        # 모델 입력도 보호한다. 따라서 subagent 시스템 프롬프트 블록
        # (file_editing_workflow / guidelines / output_format / working_directory)도
        # lead-agent 쪽과 같은 부류의 권위 블록이다.
        "system-reminder",
        "system_reminder",
        "memory",
        "current_date",
        "think",
        "analysis",
        "role",
        "soul",
        "self_update",
        "thinking_style",
        "clarification_system",
        "critical_reminders",
        "response_style",
        "citations",
        "uploaded_files",  # 예전 uploads 태그 — 하위 호환을 위해 deermem이 아직 처리한다
        "current_uploads",
        "subagent_system",
        "skill_system",
        "skill_index",
        "available_skills",
        "disabled_skills",
        "memory_tool_system",
        "todo_list_system",
        "durable_context_data",
        "slash_skill_activation",
        "mcp_routing_hints",
        "available-deferred-tools",
        "goal_continuation",
        "file_editing_workflow",
        "guidelines",
        "output_format",
        "working_directory",
        # subagent 시스템 프롬프트 블록(general_purpose.py): task 도구를 사용 금지로
        # 선언한다. 신뢰할 수 없는 입력에서 이를 위조하면 실제로는 없는 도구 제약이
        # 있다고(또는 있는 제약이 없다고) 모델을 속일 수 있다.
        "tool_restrictions",
        # 흔한 prompt injection 태그 패턴
        "system",
        "instruction",
        "important",
        "override",
        "ignore",
        "prompt",
    }
)

# 차단 태그 전체를 매칭한다: <tag>, </tag>, <tag attrs>, <tag/>, 닫히지 않은 <tag
_BLOCKED_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(?:" + "|".join(re.escape(t) for t in sorted(_BLOCKED_TAG_NAMES)) + r")\b[^>]*>?",
    re.IGNORECASE,
)

# 평문 boundary 마커 (OWASP structured-prompt 지침).
_USER_INPUT_BEGIN = "--- BEGIN USER INPUT ---"
_USER_INPUT_END = "--- END USER INPUT ---"

# 사용자 텍스트에 이미 마커가 들어 있을 때 대신 넣는 무력화 형태.
# 시각적으로는 비슷해 보이지만 실제 boundary 구분자와는 매칭되지 않는다.
_NEUTRALIZED_BEGIN = "[BEGIN USER INPUT]"
_NEUTRALIZED_END = "[END USER INPUT]"

# 두 boundary 토큰을 독립된 줄로든 텍스트 안에 박힌 형태로든 매칭한다.
_BOUNDARY_TOKEN_RE = re.compile(
    re.escape(_USER_INPUT_BEGIN) + r"|" + re.escape(_USER_INPUT_END),
)


def _escape_tag_match(match: re.Match) -> str:
    """차단 태그 매칭에서 < 와 > 를 escape해 리터럴 텍스트로 렌더링되게 한다."""
    return match.group(0).replace("<", "&lt;").replace(">", "&gt;")


def _neutralize_boundary_tokens(text: str) -> str:
    """실제 BEGIN/END USER INPUT 마커를 비슷하게 생긴 무해한 형태로 치환한다."""
    return _BOUNDARY_TOKEN_RE.sub(
        lambda m: _NEUTRALIZED_BEGIN if m.group(0) == _USER_INPUT_BEGIN else _NEUTRALIZED_END,
        text,
    )


def neutralize_untrusted_tags(text: str) -> str:
    """신뢰할 수 없는 텍스트의 framework/injection 제어 토큰을 무력화한다.

    신뢰 경계 밖에서 온 뒤 *데이터*로서 모델 context에 들어가려는 모든 내용에 쓰는 공용
    primitive다. 현재는 진짜 user 메시지(:func:`_check_user_content` 경유)와 원격 tool
    결과(web_fetch / web_search 등, :class:`ToolResultSanitizationMiddleware` 경유)가 대상이다.

    구조적 방어 두 가지만 적용하고 그 외에는 아무것도 하지 않는다.

    * 차단 대상 framework/injection 태그(예: ``<system-reminder>``)를
      ``&lt;system-reminder&gt;``로 HTML escape해, 사람이 읽을 수는 있되 구조적 의미는
      잃게 한다.
    * 평문 ``--- BEGIN/END USER INPUT ---`` boundary 마커를 무력화해, 신뢰할 수 없는
      내용이 user-input 경계를 위조하거나 빠져나가지 못하게 한다.

    boundary 마커로 텍스트를 감싸지는 **않는다**. 그 framing은 user 메시지에만 해당하기
    때문이다. 비어 있거나 공백뿐인 텍스트는 그대로 반환해 caller가 불필요한 마커 잡음을
    만들지 않게 한다.
    """
    if not text.strip():
        return text
    text = _BLOCKED_TAG_PATTERN.sub(_escape_tag_match, text)
    return _neutralize_boundary_tokens(text)


def _is_genuine_user_message(message: object) -> bool:
    """시스템이 주입한 HumanMessage를 제외한 진짜 user 메시지에 대해 True를 반환한다.

    ``hide_from_ui``는 HumanInputCard의 숨겨진 UI 응답에도 쓰이므로, 유효한 user 응답을
    담고 있지 않은 숨겨진 HumanMessage만 건너뛴다.
    """
    if not isinstance(message, HumanMessage):
        return False
    if message.name == _SUMMARY_MESSAGE_NAME:
        return False
    if message.additional_kwargs.get("hide_from_ui") and read_human_input_response(message.additional_kwargs) is None:
        return False
    return True


def _check_user_content(text: str) -> str:
    """user 내용을 sanitize한다: 차단 태그를 escape한 뒤 boundary 마커로 감싼다.

    * 비어 있거나 공백뿐 → 그대로 반환(마커 잡음 없음).
    * 차단 태그 → ``<``/``>``를 HTML escape(예: ``<system>`` → ``&lt;system&gt;``).
    * user 텍스트 안의 boundary 토큰 → 경계를 위조하지 못하도록 무력화.
    * 이미 감싸진 경우(정확히 prefix+suffix) → 텍스트를 그대로 반환(idempotent).
    * 그 외 → boundary 마커로 감싼다.
    """
    if not text.strip():
        return text
    text = _BLOCKED_TAG_PATTERN.sub(_escape_tag_match, text)
    # Idempotency: 텍스트가 *정확히* 감싸진 경우(prefix+suffix)에만 건너뛴다. 사용자가
    # 단순히 begin 토큰을 어딘가에 입력한 경우는 해당되지 않는다.
    if text.startswith(_USER_INPUT_BEGIN) and text.endswith(_USER_INPUT_END):
        # 안쪽 내용의 boundary 토큰은 그래도 무력화한다. 사용자가 바깥 감싸기를 위조해
        # 아래의 무력화를 우회하고 안쪽 boundary 마커를 주입할 수 있기 때문이다
        # (break-out 공격).
        inner = text[len(_USER_INPUT_BEGIN) : -len(_USER_INPUT_END)]
        neutralized_inner = _neutralize_boundary_tokens(inner)
        if neutralized_inner == inner:
            return text
        return f"{_USER_INPUT_BEGIN}{neutralized_inner}{_USER_INPUT_END}"
    # 사용자가 박아 넣었을 수 있는 boundary 토큰을 무력화한다. self-suppression
    # (begin 토큰으로 감싸기를 건너뛰기)과 break-out(end 토큰으로 payload 안에 이른
    # 경계를 만들기)을 모두 막는다.
    text = _neutralize_boundary_tokens(text)
    return f"{_USER_INPUT_BEGIN}\n{text}\n{_USER_INPUT_END}"


class InputSanitizationMiddleware(AgentMiddleware[AgentState]):
    """user 입력의 prompt injection 태그를 escape하는 guardrail middleware.

    차단 태그는 거부하지 않고 HTML escape하므로 사용자 의도는 보존되고 태그는 의미적
    영향력을 잃는다. 정상 입력은 평문 boundary 마커로 감싼다. 변환은 일시적이며
    (wrap_model_call) state에는 절대 기록되지 않는다.
    """

    @staticmethod
    def _extract_text_from_content(content: str | list) -> tuple[str, list | None]:
        """평문 문자열 또는 content block 리스트에서 텍스트를 이어 붙여 추출한다.

        ``(text, extracted_blocks)``를 반환한다. *content*가 문자열이면
        *extracted_blocks*는 None이고, 리스트면 텍스트 content block들의 리스트다.

        리스트는 content block dict 옆에 맨 ``str`` 항목을 담을 수 있으므로
        (``message_content_to_text``가 둘 다 텍스트로 취급하고, 일부 IM/SDK 클라이언트가
        정확히 그 형태를 보낸다) 맨 문자열도 함께 수집한다. 이를 건너뛰면 그 메시지의
        sanitization이 통째로 생략된다.
        """
        if isinstance(content, str):
            return content, None
        if not isinstance(content, list):
            return "", None
        text_parts: list[str] = []
        text_blocks: list[dict | str] = []
        for block in content:
            if isinstance(block, str):
                if not block:  # 빈 항목은 건너뛴다 — message_content_to_text 동작과 일치
                    continue
                text_parts.append(block)
                text_blocks.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if not text:  # 빈 block은 건너뛴다 — message_content_to_text 동작과 일치
                    continue
                text_parts.append(text)
                text_blocks.append(block)
        return "\n".join(text_parts), text_blocks

    @staticmethod
    def _rebuild_content(
        original_content: list,
        processed_text: str,
        text_blocks: list,
    ) -> list:
        """텍스트 block들을 병합된 단일 텍스트 block으로 치환하되, 사이에 끼어 있는 비텍스트
        block은 보존한다.

        ``[text, image, text]``에서는 두 텍스트 block 사이의 image block이 제자리에 남는다.
        텍스트 block만 하나로 합쳐진다.
        """
        text_block_ids = {id(b) for b in text_blocks}
        first = last = None
        for i, block in enumerate(original_content):
            if id(block) in text_block_ids:
                if first is None:
                    first = i
                last = i
        if first is None:
            return original_content
        result: list = [*original_content[:first], {"type": "text", "text": processed_text}]
        # 텍스트 block 사이에 있던 비텍스트 block을 다시 넣는다
        for i in range(first + 1, last + 1):
            if id(original_content[i]) not in text_block_ids:
                result.append(original_content[i])
        result.extend(original_content[last + 1 :])
        return result

    def _process_request(self, request: ModelRequest) -> ModelRequest:
        """마지막 진짜 user 메시지를 sanitize한 request를 반환한다.

        차단 태그는 거부하지 않고 HTML escape하므로 사용자 의도는 보존되고 태그는 의미적
        영향력을 잃는다. 변환은 일시적이며 원본 request는 절대 변경되지 않는다.
        """
        messages = list(request.messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not _is_genuine_user_message(msg):
                if isinstance(msg, HumanMessage):
                    logger.debug(
                        "_process_request: skipping non-genuine HumanMessage at pos=%d name=%s hide_from_ui=%s content_preview=%.80r",
                        i,
                        msg.name,
                        msg.additional_kwargs.get("hide_from_ui"),
                        msg.content,
                    )
                continue
            content = msg.content
            logger.debug("_process_request: found genuine user message at pos=%d content=%.120r", i, content)

            text_content, text_blocks = self._extract_text_from_content(content)

            # 텍스트가 전혀 없으면(예: 이미지만 있는 메시지) 그대로 통과시킨다
            if not text_content and not isinstance(content, str):
                logger.debug("_process_request: no text content in message — passing through")
                return request

            # 사용 가능하면 사용자의 원본 입력만 sanitize한다(UploadsMiddleware가
            # <current_uploads> 블록을 앞에 붙이기 전에 설정한다). 그래야 서버가 주입한
            # 신뢰된 블록이 차단 태그 검사 대상이 되지 않는다. 마커가 없을 때만 전체 내용
            # 스캔으로 폴백한다 — UploadsMiddleware는 업로드 턴에만 마커를 설정하므로
            # 업로드 없는 평문 메시지에는 마커가 없다. 그런 경우 전체 내용 스캔은 안전하다.
            # 실수로 escape할 서버 주입 <current_uploads> 블록이 아예 없기 때문이다.
            preserved_kwargs = dict(msg.additional_kwargs or {})
            original_user_content = preserved_kwargs.get(ORIGINAL_USER_CONTENT_KEY)
            if isinstance(original_user_content, str) and original_user_content:
                processed_user = _check_user_content(original_user_content)
                if processed_user != original_user_content:
                    # 전체 내용 중 사용자 텍스트 suffix만 치환한다. 서버가 앞에 붙인
                    # 블록은 건드리지 않는다.
                    idx = text_content.rfind(original_user_content)
                    if idx >= 0:
                        processed = text_content[:idx] + processed_user
                    else:
                        # _extract_text_from_content와 message_content_to_text의 텍스트
                        # 추출 결과가 어긋나 rfind가 실패했다(multimodal 리스트 content에서만
                        # 도달 가능. Decision 18 참고).
                        if isinstance(content, list) and len(content) >= 2:
                            # content[0]은 서버가 주입한 <current_uploads> 블록이다
                            # (UploadsMiddleware가 리스트 content의 첫 요소로 앞에 붙인다).
                            # 사용자 블록(content[1:])만 sanitize하고 직접 재구성한다.
                            # _rebuild_content는 type:"text" 블록만 처리하므로
                            # message_content_to_text가 보는 raw 문자열이나 비표준 dict
                            # 블록을 놓치기 때문이다.
                            logger.warning(
                                "rfind failed on multimodal content; sanitizing user content blocks individually",
                            )
                            new_content: list = [content[0]]
                            for block in content[1:]:
                                if isinstance(block, str):
                                    new_content.append(neutralize_untrusted_tags(block))
                                elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                                    sanitized = neutralize_untrusted_tags(block["text"])
                                    if sanitized != block["text"]:
                                        new_content.append({**block, "text": sanitized})
                                    else:
                                        new_content.append(block)
                                else:
                                    new_content.append(block)
                            messages[i] = HumanMessage(
                                content=new_content,
                                id=msg.id,
                                name=msg.name,
                                additional_kwargs=preserved_kwargs,
                            )
                            return request.override(messages=messages)
                        else:
                            # 서버 블록과 사용자 블록을 구분할 수 없다(리스트가 아닌
                            # content이거나 len(content) < 2). 전체 내용 sanitization으로
                            # 낮춘다. 서버 블록이 escape될 수 있지만(UX 저하) 사용자의
                            # 위조는 여전히 무력화된다(보안 회귀 없음).
                            logger.warning(
                                "rfind failed with original_user_content set; cannot distinguish blocks, falling back to full-content sanitization",
                            )
                            processed = _check_user_content(text_content)
                else:
                    processed = text_content  # 변경 불필요
            elif isinstance(original_user_content, str):
                # key는 있지만 빈 문자열인 경우(예: 텍스트 입력 없는 파일 업로드).
                # sanitize할 사용자 텍스트가 없고, 서버가 주입한 블록은 손대지 않고
                # 그대로 살아남아야 한다.
                processed = text_content
            else:
                processed = _check_user_content(text_content)  # fallback

            if processed == text_content:
                # 이미 깨끗하거나 이미 감싸져 있다 — override 불필요
                return request

            if text_blocks:
                new_content = self._rebuild_content(content, processed, text_blocks)
            else:
                new_content = processed

            # sanitize 이전의 사용자 텍스트를 보존해, 진짜 입력을 봐야 하는 하위 소비자
            # (slash skill activation, regenerate)가 BEGIN/END 감싸기 이후에도 복원할 수
            # 있게 한다. UploadsMiddleware나 IM channel이 설정한 유효한 값은 유지하되,
            # 잘못된 메타데이터는 고쳐서 persistence가 모델에게 보이는 감싸진 내용으로
            # 폴백하지 않게 한다.
            if not isinstance(original_user_content, str):
                if ORIGINAL_USER_CONTENT_KEY in preserved_kwargs:
                    logger.warning(
                        "InputSanitizationMiddleware replaced non-string %s metadata: type=%s",
                        ORIGINAL_USER_CONTENT_KEY,
                        type(original_user_content).__name__,
                    )
                preserved_kwargs[ORIGINAL_USER_CONTENT_KEY] = message_content_to_text(content)
            messages[i] = HumanMessage(
                content=new_content,
                id=msg.id,
                name=msg.name,
                additional_kwargs=preserved_kwargs,
            )
            logger.debug(
                "InputSanitizationMiddleware: original=%r -> processed=%r",
                content if isinstance(content, str) else "[content-blocks]",
                processed,
            )
            return request.override(messages=messages)
        return request

    def _try_process(self, request: ModelRequest) -> ModelRequest:
        """request를 sanitize한다. 예상치 못한 오류에서는 fail-open한다.

        GraphBubbleUp은 전파하고, 그 밖의 예외는 원본 request를 반환한다.
        """
        try:
            return self._process_request(request)
        except GraphBubbleUp:
            raise
        except Exception:
            logger.warning(
                "Input guardrail processing failed; passing original request to model",
                exc_info=True,
            )
            return request

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._try_process(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._try_process(request))
