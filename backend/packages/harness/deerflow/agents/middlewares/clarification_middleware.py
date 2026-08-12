"""clarification 요청을 가로채 사용자에게 제시하는 middleware."""

import json
import logging
import re
from collections.abc import Callable
from hashlib import sha256
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

# 허용된 form field 타입. 그 외에는 "text"로 낮춰서, 모델이 잘못된 타입을 줘도 렌더링할 수
# 없는 카드가 나오지 않게 한다.
FORM_FIELD_TYPES = frozenset({"text", "textarea", "number", "select", "multi_select", "checkbox", "date"})
_OPTION_FIELD_TYPES = frozenset({"select", "multi_select"})

# JavaScript Object.prototype 속성과 충돌하는 field 이름들. frontend는 form 값을 field 이름을
# 키로 하는 평범한 객체에 저장하므로, 이런 이름은 사용자 입력 대신 상속된 prototype 멤버를
# 읽게 된다.
_RESERVED_FIELD_NAMES = frozenset(
    {
        "__proto__",
        "constructor",
        "prototype",
        "toString",
        "toLocaleString",
        "valueOf",
        "hasOwnProperty",
        "isPrototypeOf",
        "propertyIsEnumerable",
        "__defineGetter__",
        "__defineSetter__",
        "__lookupGetter__",
        "__lookupSetter__",
    }
)

# 폭주한 모델이 무한정 큰 form을 만들지 못하도록 하는 hard cap. 상한을 넘는 것은 구조적
# 오류이므로, 업무 field를 조용히 잘라내는 대신 form 전체를 legacy 모드로 낮춘다.
MAX_FORM_FIELDS = 16
MAX_FIELD_OPTIONS = 24
MAX_FIELD_TEXT_CHARS = 200
# 정규화된 field 직렬화 결과에 대한 총 예산(UTF-8 바이트). 항목별 상한만으로는 평문 IM
# fallback이 채널 전송 한도를 넘는 form도 통과한다(Slack은 메시지당 40k자에서 자르고, Feishu는
# 카드당 약 30KB를 권장한다). 그러면 뒤쪽 field가 조용히 사라지는데, 이는 원자적 검증이 막으려는
# 바로 그 상황이다. 16KB면 허용된 어떤 form의 fallback 텍스트도 가장 빡빡한 채널 안에 여유 있게
# 들어가고 question/context 여유분도 남는다.
MAX_FORM_SERIALIZED_BYTES = 16_384

_XML_TAG_RE = re.compile(r"</?[A-Za-z_][\w:.-]*(?:\s[^<>]*?)?\s*/?>")


class ClarificationMiddlewareState(AgentState):
    """`ThreadState` 스키마와 호환된다."""

    pass


class ClarificationMiddleware(AgentMiddleware[ClarificationMiddlewareState]):
    """clarification 도구 호출을 가로채고 실행을 중단해 사용자에게 질문을 제시한다.

    모델이 `ask_clarification` 도구를 호출하면 이 middleware는 다음을 수행한다.

    1. 실행 전에 도구 호출을 가로챈다.
    2. clarification 질문과 metadata를 추출한다.
    3. 사용자가 읽기 좋은 메시지로 포매팅한다.
    4. 실행을 중단하고 질문을 제시하는 Command를 반환한다.
    5. 사용자 응답을 기다린 뒤에 진행한다.

    clarification이 대화 흐름을 그대로 이어가던 기존 도구 기반 방식을 대체한다.
    """

    state_schema = ClarificationMiddlewareState

    def _stable_message_id(self, tool_call_id: str, formatted_message: str) -> str:
        """재시도된 clarification 호출이 추가가 아니라 교체되도록 결정적 message ID를 만든다."""
        if tool_call_id:
            return f"clarification:{tool_call_id}"
        digest = sha256(formatted_message.encode("utf-8")).hexdigest()[:16]
        return f"clarification:{digest}"

    def _normalize_options(self, raw_options: Any) -> list[str]:
        """도구가 준 options를 표시 가능한 문자열 값으로 정규화한다."""
        options = raw_options

        # 일부 모델(예: Qwen3-Max)은 배열 파라미터를 네이티브 배열이 아니라 JSON 문자열로
        # 직렬화한다. 아래 렌더링 로직에서 `options`가 항상 list가 되도록 역직렬화하고 정규화한다.
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                options = [options]

        if options is None:
            return []
        if isinstance(options, dict):
            options = self._flatten_dict_option_values(options)
        elif not isinstance(options, list):
            options = [options]

        # 공백 제거, 빈 값 삭제, 순서를 유지한 중복 제거. frontend 파서는 빈 option label이
        # 하나라도 있으면 payload 전체를 거부하므로 절대 내보내면 안 된다.
        normalized: list[str] = []
        seen: set[str] = set()
        for option in options:
            text = _XML_TAG_RE.sub("", str(option)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _flatten_dict_option_values(value: dict[str, Any]) -> list[str | int | float]:
        """XML을 dict로 변환한 option payload에서 스칼라 leaf를 원본 순서대로 평탄화한다."""
        flattened: list[str | int | float] = []

        def collect(nested: Any) -> None:
            if isinstance(nested, dict):
                for item in nested.values():
                    collect(item)
            elif isinstance(nested, list):
                for item in nested:
                    collect(item)
            elif isinstance(nested, str | int | float):
                flattened.append(nested)

        collect(value)
        return flattened

    @staticmethod
    def _normalize_bool(raw: Any) -> bool:
        """모델이 준 boolean을 강제 변환한다. 일부 모델은 boolean을 문자열이나 1/0으로 직렬화한다."""
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int | float):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() == "true"
        return False

    def _normalize_fields(self, raw_fields: Any) -> list[dict[str, Any]]:
        """도구가 준 form field를 검증된 v2 field 스키마로 정규화한다.

        검증은 원자적이다. 구조적으로 깨진 항목(dict가 아님, 잘못되거나 예약되거나 중복된
        이름, 개수·길이 상한 초과)이 하나라도 있으면 form 전체를 무효화한다. 필수 업무 field가
        조용히 빠진 채로 카드가 "완전한" 것처럼 보이는 일을 막기 위해서다. 무해한 문제는 국소
        강등으로 처리한다. 알 수 없는 타입과 option 없는 select는 ``text``가 된다.
        """
        fields = raw_fields
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except (json.JSONDecodeError, TypeError):
                return []
        if not isinstance(fields, list):
            return []
        if len(fields) > MAX_FORM_FIELDS:
            return []

        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for entry in fields:
            if not isinstance(entry, dict):
                return []
            raw_name = entry.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                return []
            name = raw_name.strip()
            if name in _RESERVED_FIELD_NAMES or name in seen_names or len(name) > MAX_FIELD_TEXT_CHARS:
                return []
            seen_names.add(name)

            raw_label = entry.get("label")
            label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else name
            if len(label) > MAX_FIELD_TEXT_CHARS:
                return []

            field_type = entry.get("type")
            # isinstance 검사를 먼저 한다. 모델이 보내는 `type: []` / `type: {}`도 적법한
            # JSON이라, 해시 불가 값으로 멤버십 검사를 하면 강등 대신 TypeError가 난다.
            if not isinstance(field_type, str) or field_type not in FORM_FIELD_TYPES:
                field_type = "text"

            options = self._normalize_options(entry.get("options")) if field_type in _OPTION_FIELD_TYPES else []
            if len(options) > MAX_FIELD_OPTIONS or any(len(option) > MAX_FIELD_TEXT_CHARS for option in options):
                return []
            if field_type in _OPTION_FIELD_TYPES and not options:
                field_type = "text"

            field: dict[str, Any] = {
                "name": name,
                "label": label,
                "type": field_type,
                "required": self._normalize_bool(entry.get("required")),
            }
            if field_type in _OPTION_FIELD_TYPES:
                field["options"] = [
                    {
                        "id": f"{name}-option-{index}",
                        "label": option,
                        "value": option,
                    }
                    for index, option in enumerate(options, 1)
                ]
            placeholder = entry.get("placeholder")
            if isinstance(placeholder, str) and placeholder.strip():
                if len(placeholder.strip()) > MAX_FIELD_TEXT_CHARS:
                    return []
                field["placeholder"] = placeholder.strip()
            normalized.append(field)

        if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_FORM_SERIALIZED_BYTES:
            return []

        return normalized

    def _build_human_input_payload(self, args: dict[str, Any], *, tool_call_id: str, request_id: str, fields: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """ToolMessage.content를 fallback으로 남긴 채 구조화된 UI payload를 만든다.

        프로토콜 버전 관리: legacy 모드(``free_text`` / ``choice_with_other``)는 wire 포맷을
        그대로 두기 위해 ``version: 1``을 유지하고, v2 ``form`` 모드는 ``version: 2``를 실어
        구버전 frontend가 payload를 거부하고 평문 ToolMessage content로 강등되게 한다. 응답은
        v1 응답 프로토콜(``text`` / ``option``)을 유지한다. form 카드가 읽기 좋은 ``value``
        요약을 제출하므로 새 응답 종류를 도입하지 않는다.

        ``fields``는 이미 정규화된 리스트를 받는다. payload와 텍스트 fallback을 함께 렌더링하는
        호출자가 정규화를 한 번만 하도록 하기 위해서다.
        """
        if fields is None:
            fields = self._normalize_fields(args.get("fields"))
        options = self._normalize_options(args.get("options", []))
        clarification_type = str(args.get("clarification_type", "missing_info"))

        if fields:
            version, input_mode = 2, "form"
        elif options:
            version, input_mode = 1, "choice_with_other"
        else:
            version, input_mode = 1, "free_text"

        payload: dict[str, Any] = {
            "version": version,
            "kind": "human_input_request",
            "source": "ask_clarification",
            "request_id": request_id,
            "clarification_type": clarification_type,
            "question": str(args.get("question") or ""),
            "input_mode": input_mode,
        }

        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        if "context" in args:
            context = args.get("context")
            payload["context"] = None if context is None else str(context)

        if input_mode == "form":
            payload["fields"] = fields
        elif options:
            payload["options"] = [
                {
                    "id": f"option-{index}",
                    "label": option,
                    "value": option,
                }
                for index, option in enumerate(options, 1)
            ]

        return payload

    def _is_chinese(self, text: str) -> bool:
        """텍스트에 한자가 포함되어 있는지 확인한다.

        Args:
            text: 검사할 텍스트

        Returns:
            한자가 포함되어 있으면 True
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _format_clarification_message(self, args: dict, fields: list[dict[str, Any]] | None = None) -> str:
        """clarification 인자를 사용자가 읽기 좋은 메시지로 포매팅한다.

        Args:
            args: clarification 상세 정보를 담은 도구 호출 인자
            fields: 이미 정규화된 form field. payload와 이 fallback을 함께 렌더링하는
                호출자가 정규화를 한 번만 하도록 하기 위해 받는다.

        Returns:
            포매팅된 메시지 문자열
        """
        question = args.get("question", "")
        # str()로 강제 변환해 아이콘 조회 키를 해시 가능하게 유지한다. 모델이 보내는
        # `clarification_type: []`도 적법한 JSON이라 dict 키로 쓰면 TypeError가 난다.
        clarification_type = str(args.get("clarification_type", "missing_info"))
        context = args.get("context")
        if fields is None:
            fields = self._normalize_fields(args.get("fields"))
        options = self._normalize_options(args.get("options", []))

        # 타입별 아이콘
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "🤔",
            "approach_choice": "🔀",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }

        icon = type_icons.get(clarification_type, "❓")

        # 메시지를 자연스럽게 조립한다.
        message_parts = []

        # 흐름이 자연스럽도록 아이콘과 질문을 함께 붙인다.
        if context:
            # context가 있으면 배경 설명으로 먼저 보여 준다.
            message_parts.append(f"{icon} {context}")
            message_parts.append(f"\n{question}")
        else:
            # 아이콘과 질문만 보여 준다.
            message_parts.append(f"{icon} {question}")

        # payload 로직과 동일하게 form field가 options보다 우선한다.
        if fields:
            message_parts.append("")  # 줄 간격용 빈 줄
            for i, field in enumerate(fields, 1):
                line = f"  {i}. {field['label']}"
                if field["required"]:
                    line += " (required)"
                field_options = field.get("options")
                if field_options:
                    line += " — options: " + " / ".join(option["label"] for option in field_options)
                    if field["type"] == "multi_select":
                        line += " (multiple allowed)"
                message_parts.append(line)
            message_parts.append("")
            message_parts.append("Please reply with a value for each field.")
        elif options and len(options) > 0:
            message_parts.append("")  # 줄 간격용 빈 줄
            for i, option in enumerate(options, 1):
                message_parts.append(f"  {i}. {option}")

        return "\n".join(message_parts)

    def _is_disabled(self, request: ToolCallRequest) -> bool:
        """이 run에서 clarification이 억제되는지 여부.

        비대화형 채널(예: GitHub webhook)은 run context에 ``disable_clarification``을 설정한다.
        clarification이 run을 막다른 길로 만들기 때문이다. 사람은 나중에 도착하는 webhook으로만
        "답변"하는데, 그때는 이미 에이전트의 turn이 한참 전에 끝난 뒤다. 이 값이 설정되어 있으면
        중단하지 않고, 에이전트가 스스로 판단해 진행하도록 유도하는 ToolMessage를 반환한다.
        """
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        if not context:
            return False
        return bool(context.get("disable_clarification"))

    def _handle_disabled_clarification(self, request: ToolCallRequest) -> ToolMessage:
        """clarification을 억제하고 에이전트에게 계속 진행하라고 알린다.

        ``Command(goto=END)``가 아니라 평범한 ToolMessage를 반환하므로 에이전트 루프가 끝나지
        않고 이어진다. 에이전트는 이를 도구 결과로 받아 다시 생성하며, 되묻는 대신 실제로
        행동하는 것이 바람직한 동작이다.
        """
        tool_call_id = request.tool_call.get("id", "")
        logger.info("ask_clarification suppressed (disable_clarification set); instructing agent to proceed")
        return ToolMessage(
            id=self._stable_message_id(tool_call_id, "proceed-without-clarification"),
            content=(
                "Clarification is disabled in this context — the human is not present "
                "to answer synchronously. Do not ask for confirmation. Proceed with your "
                "best judgment, carry out the requested action, and state any assumptions "
                "you made in your final response."
            ),
            tool_call_id=tool_call_id,
            name="ask_clarification",
        )

    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """clarification 요청을 처리하고 실행을 중단하는 Command를 반환한다.

        Args:
            request: 도구 호출 요청

        Returns:
            포매팅된 clarification 메시지와 함께 실행을 중단하는 Command
        """
        # clarification 인자를 추출한다.
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        logger.info("Intercepted clarification request")
        logger.debug("Clarification question: %s", question)

        # form field는 한 번만 정규화한다. 텍스트 fallback과 payload가 같은 결과를 쓴다.
        fields = self._normalize_fields(args.get("fields"))

        # clarification 메시지를 포매팅한다.
        formatted_message = self._format_clarification_message(args, fields=fields)

        # 도구 호출 ID를 가져온다.
        tool_call_id = request.tool_call.get("id", "")

        request_id = self._stable_message_id(tool_call_id, formatted_message)
        human_input_payload = self._build_human_input_payload(args, tool_call_id=tool_call_id, request_id=request_id, fields=fields)

        # 포매팅된 질문을 담은 ToolMessage를 만든다. 메시지 이력에 추가된다.
        tool_message = ToolMessage(
            id=request_id,
            content=formatted_message,
            tool_call_id=tool_call_id,
            name="ask_clarification",
            artifact={"human_input": human_input_payload},
        )

        # 다음을 수행하는 Command를 반환한다.
        # 1. 포매팅된 tool message를 추가한다.
        # 2. __end__로 이동해 실행을 중단한다.
        # 참고: 여기서 별도의 AIMessage를 추가하지 않는다. frontend가 ask_clarification tool
        # message를 직접 감지해 표시한다.
        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """ask_clarification 도구 호출을 가로채 실행을 중단한다(sync 버전).

        Args:
            request: 도구 호출 요청
            handler: 원래의 도구 실행 handler

        Returns:
            포매팅된 clarification 메시지와 함께 실행을 중단하는 Command
        """
        # ask_clarification 도구 호출인지 확인한다.
        if request.tool_call.get("name") != "ask_clarification":
            # clarification 호출이 아니면 평소대로 실행한다.
            return handler(request)

        if self._is_disabled(request):
            return self._handle_disabled_clarification(request)

        return self._handle_clarification(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """ask_clarification 도구 호출을 가로채 실행을 중단한다(async 버전).

        Args:
            request: 도구 호출 요청
            handler: 원래의 도구 실행 handler(async)

        Returns:
            포매팅된 clarification 메시지와 함께 실행을 중단하는 Command
        """
        # ask_clarification 도구 호출인지 확인한다.
        if request.tool_call.get("name") != "ask_clarification":
            # clarification 호출이 아니면 평소대로 실행한다.
            return await handler(request)

        if self._is_disabled(request):
            return self._handle_disabled_clarification(request)

        return self._handle_clarification(request)
