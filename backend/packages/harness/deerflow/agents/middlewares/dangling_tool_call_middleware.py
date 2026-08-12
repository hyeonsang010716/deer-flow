"""메시지 이력의 dangling tool call과 orphan tool 결과를 바로잡는 middleware.

dangling tool call은 AIMessage에 tool_calls가 있는데 이력에 대응하는 ToolMessage가 없는
상태다(사용자 중단이나 요청 취소 등으로 발생한다). orphan ToolMessage는 짝이 되는 AIMessage
tool_call 없이 tool 결과만 남은 상태다(summarization/branching이 상위 AIMessage를 지운 뒤 등).
둘 다 엄격한 provider의 요청 거부를 유발한다.

이 middleware는 model 호출을 가로채 다음을 수행한다.

- provider 직렬화 전에 잘못된 tool-call 이름과 인자를 정리한다.
- dangling한 AIMessage tool_call마다 에러 표시가 담긴 합성 ToolMessage를 올바른 위치에 넣는다.
- 요청에 더 이상 원본 tool_call이 없는 orphan ToolMessage를 제거해, 엄격한 OpenAI 호환
  backend가 HTTP 400을 반환하지 않게 한다.

참고: before_model이 아니라 wrap_model_call을 쓴다. before_model + add_messages reducer는
패치를 메시지 목록 끝에 덧붙이지만, 여기서는 각 dangling AIMessage 바로 뒤라는 정확한 위치에
삽입해야 하기 때문이다.
"""

import json
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# 이슈 #2894 우회: 잘못된 write_file 호출은 invalid tool-call args에 거대한 Markdown payload를
# 담을 수 있다. 합성 ToolMessage가 크거나 깨진 내용을 모델에게 되돌려 주지 않도록 복구 에러
# 상세를 짧게 유지한다.
_MAX_RECOVERY_ERROR_DETAIL_LEN = 500
_UNKNOWN_TOOL_NAME = "unknown_tool"
_EMPTY_TOOL_NAME_ERROR = "Tool call could not be executed because its name was missing or empty."
_SYNTHETIC_TOOL_CALL_ID_PREFIX = "deerflow_synthetic_tool_call_"


def _valid_tool_name(name: object) -> bool:
    return isinstance(name, str) and bool(name.strip())


def _valid_tool_call_id(tool_call_id: object) -> bool:
    return isinstance(tool_call_id, str) and bool(tool_call_id.strip())


def _tool_call_name(tool_call: dict) -> object:
    """호출이 선언한 이름을 반환한다. _message_tool_calls의 raw-payload fallback과 동일하게 동작한다."""
    name = tool_call.get("name")
    if _valid_tool_name(name):
        return name
    function = tool_call.get("function")
    return function.get("name") if isinstance(function, dict) else name


def _names_can_pair(call_name: object, result_name: object) -> bool:
    """결과의 이름이 호출의 이름과 모순되지 않는지 판단한다.

    양쪽 모두 정당하게 비어 있을 수 있고(빈 이름 형제 복구가 바로 그것을 위해 존재한다),
    없는 이름은 아무것과도 모순될 수 없다. 사용 가능한 두 이름이 서로 다를 때만 짝짓기를
    배제한다.
    """
    if not _valid_tool_name(call_name) or not _valid_tool_name(result_name):
        return True
    return call_name.strip() == result_name.strip()


def _relabel_tool_call_ids(tool_calls: list, msg_index: int, source: str) -> tuple[list, list[dict], bool]:
    """tool-call 목록의 잘못된 id를 안정적인 합성 id로 교체한다.

    id는 호출의 위치에서 파생하므로 짝짓기 단계와 model에 실릴 메시지가 상태를 주고받지
    않고도 같은 값에 도달한다.

    Returns:
        재작성된 목록, 재라벨된 호출마다 하나씩의 ``{original, synthetic, name}`` 항목,
        그리고 변경 여부.
    """
    relabeled: list = []
    assigned: list[dict] = []
    changed = False
    for position, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict) or _valid_tool_call_id(tool_call.get("id")):
            relabeled.append(tool_call)
            continue
        synthetic = f"{_SYNTHETIC_TOOL_CALL_ID_PREFIX}{msg_index}_{source}_{position}"
        relabeled.append({**tool_call, "id": synthetic})
        changed = True
        assigned.append({"original": tool_call.get("id"), "synthetic": synthetic, "name": _tool_call_name(tool_call)})
    return relabeled, assigned, changed


def _turn_malformed_result_count(messages: list, start: int) -> int:
    """``start``에서 시작된 turn이 낸 잘못된 결과의 개수를 센다."""
    count = 0
    for msg in messages[start + 1 :]:
        if getattr(msg, "type", None) == "ai":
            break
        if isinstance(msg, ToolMessage) and not _valid_tool_call_id(msg.tool_call_id):
            count += 1
    return count


def _claim_synthetic_id(open_calls: list[dict], result: ToolMessage, positional: bool) -> str | None:
    """``result``가 답하는 열린 malformed 호출을 소비하고 그 새 id를 반환한다.

    malformed 원본은 모두 똑같이 비어 있어서 자기 결과를 식별할 수 없다. ``open_calls``는 이미
    호출을 낸 turn으로 범위가 좁혀져 있다. 그 turn 안에서 결과의 이름이 후보를 좁히고,
    *강제되는* 선택만 취한다.

    * 호환 호출이 하나 — 이름이나 turn의 유일한 호출이라는 사실이 그것을 식별한다.
    * 호환 호출이 여럿 — 위치가 식별하지만 ``positional``이 성립할 때만, 즉 turn의 모든 열린
      호출에 결과가 있을 때만 그렇다. 동일한 병렬 호출(``bash`` 두 개)은 다른 무엇으로도
      구분할 수 없고, 여기서의 순서는 provider에 대한 가정이 아니라 구성상의 보장이다.
      LangGraph의 ``ToolNode``는 ``tool_calls``에 대해 ``asyncio.gather`` / ``executor.map``으로
      결과를 만들고, 둘 다 도구가 어떻게 뒤섞이든 입력 순서로 내놓는다. 결과가 *없다*는 것은
      호출이 중단되었다는 뜻이고 — 이 middleware가 존재하는 이유 자체다 — 그러면 살아남은
      결과가 호출과 나란히 맞는다고 더는 신뢰할 수 없다.

    ``None``을 반환하면 결과는 malformed인 채로 남아 orphan 단계에서 제거된다. 이는 귀속 불가능한
    결과가 현재 받는 처리이며, 없는 짝을 지어내는 것보다 낫다.
    """
    candidates = [position for position, entry in enumerate(open_calls) if entry["original"] == result.tool_call_id and _names_can_pair(entry["name"], result.name)]
    if not candidates or (len(candidates) > 1 and not positional):
        return None
    return open_calls.pop(candidates[0])["synthetic"]


def _normalize_tool_name(name: object) -> str:
    return name.strip() if _valid_tool_name(name) else _UNKNOWN_TOOL_NAME


def _has_invalid_tool_name(name: object) -> bool:
    return not _valid_tool_name(name)


def _parse_json_object(value: object) -> dict | None:
    """JSON object 문자열을 파싱한다. 그 외 입력에는 None을 반환한다."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_tool_arguments(arguments: object) -> str:
    """OpenAI 호환 재전송에 안전한 JSON object 문자열을 반환한다."""
    if isinstance(arguments, dict):
        try:
            return json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return "{}"
    return arguments if _parse_json_object(arguments) is not None else "{}"


class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """dangling tool call에 placeholder ToolMessage를 넣고, orphan ToolMessage(출처 AIMessage
    tool_call이 사라진 tool 결과)를 제거한다.

    메시지 이력에서 다음을 찾는다.
    - tool_calls에 대응하는 ToolMessage가 없는 AIMessage — 문제의 AIMessage 바로 뒤에 합성
      에러 응답을 주입한다
    - 짝이 되는 AIMessage tool_call이 없는 ToolMessage(orphan) — 엄격한 OpenAI 호환 backend가
      요청을 거부하지 않도록 제거한다
    """

    @staticmethod
    def _message_tool_calls(msg) -> list[dict]:
        """구조화된 필드나 raw provider payload에서 정규화된 tool call을 반환한다.

        LangChain은 잘못된 provider function call을 ``invalid_tool_calls``에 저장한다. 이들은
        실행되지 않지만, provider adapter가 호출 id/이름을 다음 요청에 충분히 직렬화해서
        엄격한 OpenAI 호환 validator가 짝이 되는 ToolMessage를 기대하게 만들 수 있다. 그래서
        dangling 호출로 취급해 다음 model 요청이 올바른 형태를 유지하고, 모델이 또 다른
        provider 400 대신 복구 가능한 tool 에러를 보게 한다.
        """
        normalized: list[dict] = []

        tool_calls = getattr(msg, "tool_calls", None) or []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                logger.debug("Skipping malformed non-dict tool_call in AIMessage: %r", tool_call)
                continue
            original_name = tool_call.get("name")
            normalized_call = dict(tool_call)
            normalized_call["name"] = _normalize_tool_name(original_name)
            if _has_invalid_tool_name(original_name):
                normalized_call["invalid_tool_name"] = True
            normalized.append(normalized_call)

        raw_tool_calls = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []
        # raw payload는 같은 호출들의 fallback 직렬화다. OpenAI serializer는 구조화된 두 view가
        # 모두 비었을 때만 이걸 찾는다(_normalize_tool_call_ids의 게이팅과 동일). invalid_tool_calls가
        # 비어 있지 않은데 이걸 수집하면 같은 호출을 두 번 세어 하나의 id에 ToolMessage를 둘
        # 만들게 되고, 이는 엄격한 provider가 거부하는 바로 그 중복 id 형태다.
        if not tool_calls and not getattr(msg, "invalid_tool_calls", None):
            for raw_tc in raw_tool_calls:
                if not isinstance(raw_tc, dict):
                    continue

                function = raw_tc.get("function")
                name = raw_tc.get("name")
                if not name and isinstance(function, dict):
                    name = function.get("name")

                args = raw_tc.get("args", {})
                if not args and isinstance(function, dict):
                    parsed_args = _parse_json_object(function.get("arguments"))
                    args = parsed_args if parsed_args is not None else {}

                normalized_call = {
                    "id": raw_tc.get("id"),
                    "name": _normalize_tool_name(name),
                    "args": args if isinstance(args, dict) else {},
                }
                if _has_invalid_tool_name(name):
                    normalized_call["invalid_tool_name"] = True
                normalized.append(normalized_call)

        for invalid_tc in getattr(msg, "invalid_tool_calls", None) or []:
            if not isinstance(invalid_tc, dict):
                continue
            original_name = invalid_tc.get("name")
            normalized_call = {
                "id": invalid_tc.get("id"),
                "name": _normalize_tool_name(original_name),
                "args": {},
                "invalid": True,
                "error": invalid_tc.get("error"),
            }
            if _has_invalid_tool_name(original_name):
                normalized_call["invalid_tool_name"] = True
            normalized.append(normalized_call)

        return normalized

    @staticmethod
    def _synthetic_tool_message_content(tool_call: dict) -> str:
        if tool_call.get("invalid_tool_name"):
            return f"[{_EMPTY_TOOL_NAME_ERROR} Use one of the available tool names when retrying.]"
        if tool_call.get("invalid"):
            name = tool_call.get("name")
            error = tool_call.get("error")
            error_text = error[:_MAX_RECOVERY_ERROR_DETAIL_LEN] if isinstance(error, str) and error else ""
            # 이슈 #2894 우회: 잘못된 write_file 호출은 invalid tool-call args에 거대한 Markdown
            # payload를 담을 수 있다. 크거나 깨진 내용을 모델에게 되돌려 주지 않으면서 복구
            # 안내는 실행 가능하게 유지한다.
            if name == "write_file":
                details = f" Parser error: {error_text}" if error_text else ""
                return (
                    "[write_file failed before execution: the tool-call arguments were not valid JSON, "
                    "so no file was written. This often happens when the model tries to write a very "
                    "large Markdown file in a single tool call, especially when `content` contains "
                    "unescaped quotes, inline JSON, backslashes, or code fences. Do not retry the same "
                    "large `write_file` payload for this artifact; provide the report/content directly "
                    "as normal assistant text in your next response. If a file write is still needed "
                    f"later, split the file into smaller sections instead of one large payload.{details}]"
                )
            if error_text:
                return f"[Tool call could not be executed because its arguments were invalid: {error_text}]"
            return "[Tool call could not be executed because its arguments were invalid.]"
        return "[Tool call was interrupted and did not return a result.]"

    @staticmethod
    def _sanitize_ai_message_tool_calls(msg):
        """model에 실릴 tool call을 직렬화해도 안전하게 만든 AIMessage를 반환한다."""
        if getattr(msg, "type", None) != "ai":
            return msg

        changed = False
        update: dict = {}

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            structured_changed = False
            sanitized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    sanitized_tool_calls.append(tool_call)
                    continue
                name = tool_call.get("name")
                sanitized = dict(tool_call)
                normalized_name = _normalize_tool_name(name)
                if sanitized.get("name") != normalized_name:
                    sanitized["name"] = normalized_name
                    structured_changed = True
                sanitized_tool_calls.append(sanitized)
            if structured_changed:
                update["tool_calls"] = sanitized_tool_calls
                changed = True

        invalid_tool_calls = getattr(msg, "invalid_tool_calls", None)
        if invalid_tool_calls:
            invalid_changed = False
            sanitized_invalid_tool_calls = []
            for invalid_tool_call in invalid_tool_calls:
                if not isinstance(invalid_tool_call, dict):
                    sanitized_invalid_tool_calls.append(invalid_tool_call)
                    continue
                sanitized = dict(invalid_tool_call)
                normalized_name = _normalize_tool_name(sanitized.get("name"))
                normalized_arguments = _normalize_tool_arguments(sanitized.get("args"))
                if sanitized.get("name") != normalized_name:
                    sanitized["name"] = normalized_name
                    invalid_changed = True
                if sanitized.get("args") != normalized_arguments:
                    sanitized["args"] = normalized_arguments
                    invalid_changed = True
                sanitized_invalid_tool_calls.append(sanitized)
            if invalid_changed:
                update["invalid_tool_calls"] = sanitized_invalid_tool_calls
                changed = True

        additional_kwargs = dict(getattr(msg, "additional_kwargs", {}) or {})
        raw_tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            raw_changed = False
            sanitized_raw_tool_calls = []
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    sanitized_raw_tool_calls.append(raw_tool_call)
                    continue

                sanitized_raw = dict(raw_tool_call)
                function = sanitized_raw.get("function")
                if isinstance(function, dict):
                    sanitized_function = dict(function)
                    normalized_name = _normalize_tool_name(sanitized_function.get("name"))
                    normalized_arguments = _normalize_tool_arguments(sanitized_function.get("arguments"))
                    if sanitized_function.get("name") != normalized_name:
                        sanitized_function["name"] = normalized_name
                        raw_changed = True
                    if sanitized_function.get("arguments") != normalized_arguments:
                        sanitized_function["arguments"] = normalized_arguments
                        raw_changed = True
                    if sanitized_function != function:
                        sanitized_raw["function"] = sanitized_function
                else:
                    normalized_name = _normalize_tool_name(sanitized_raw.get("name"))
                    if sanitized_raw.get("name") != normalized_name:
                        sanitized_raw["name"] = normalized_name
                        raw_changed = True
                sanitized_raw_tool_calls.append(sanitized_raw)

            if raw_changed:
                additional_kwargs["tool_calls"] = sanitized_raw_tool_calls
                update["additional_kwargs"] = additional_kwargs
                changed = True

        if not changed:
            return msg
        return msg.model_copy(update=update)

    @staticmethod
    def _normalize_tool_call_ids(messages: list) -> list:
        """malformed tool-call id를 합성 id로 교체한 메시지를 반환한다.

        tool-call id를 생략하는 provider는 id가 비었거나 ``None``인 올바른 형태의 ``tool_calls``
        항목으로 파싱된다. 그런 id는 아래 짝짓기 집합에 들어갈 수 없어서 해당 호출의 결과는
        orphan으로 제거되고 placeholder도 대신 들어가지 않는다. 그러면 요청은 빈 id를 단 채
        tool 결과 없이 provider에 도달한다. id를 먼저 정규화하면 짝짓기와 placeholder 로직이
        그 호출을 다른 호출과 똑같이 다룰 수 있고, 이는 빈 이름 복구와 같은 방식이다.
        """
        rewritten: dict[int, object] = {}
        # 가장 최근 AIMessage에서 나온, 아직 응답되지 않은 malformed 호출들. 문서 순서대로 훑으며
        # 여기서 리셋하는 것이 결과를 그것을 낸 turn으로 한정한다. 결과는 이전 turn의 호출에
        # 응답하지 않으므로, 앞선 dangling 호출이 나중 turn의 결과를 소비해서는 안 된다.
        open_calls: list[dict] = []
        # 이 turn의 결과가 malformed 호출과 1:1로 맞는지 여부. 다른 방법으로는 구분할 수 없는
        # 형제 호출 사이의 동점을 위치로 깨뜨릴 수 있게 해준다.
        positional = False

        for index, msg in enumerate(messages):
            if getattr(msg, "type", None) == "ai":
                update: dict = {}
                assigned: list[dict] = []
                structured = getattr(msg, "tool_calls", None) or []
                additional_kwargs = getattr(msg, "additional_kwargs", None) or {}
                raw_tool_calls = additional_kwargs.get("tool_calls")

                invalid = getattr(msg, "invalid_tool_calls", None) or []
                sources: list[tuple[str, list, str]] = [
                    ("call", structured, "tool_calls"),
                    ("invalid", invalid, "invalid_tool_calls"),
                ]
                # raw payload는 같은 호출들의 fallback view이므로, 실제로 직렬화되는 view일 때만
                # 재라벨한다. OpenAI serializer는 구조화된 두 view가 *모두* 비었을 때만 이걸
                # 찾는다. 가려진 raw view에 id를 발급하면 provider가 결코 보지 않는 호출에
                # placeholder를 빚지게 되어, orphan tool 결과를 wire에 올리게 된다.
                if not structured and not invalid and isinstance(raw_tool_calls, list):
                    sources.append(("raw", raw_tool_calls, "additional_kwargs"))

                for source, tool_calls, field in sources:
                    relabeled, source_assigned, changed = _relabel_tool_call_ids(tool_calls, index, source)
                    assigned.extend(source_assigned)
                    if not changed:
                        continue
                    update[field] = {**additional_kwargs, "tool_calls": relabeled} if field == "additional_kwargs" else relabeled

                open_calls = assigned
                positional = _turn_malformed_result_count(messages, index) == len(assigned)
                if update:
                    rewritten[index] = msg.model_copy(update=update)
                continue

            # 이미 짝지어진 결과를 해당 호출의 새 id로 다시 가리키게 해서, 아래 짝짓기가 이를
            # orphan으로 버리지 않고 유지하게 한다.
            if not isinstance(msg, ToolMessage) or _valid_tool_call_id(msg.tool_call_id):
                continue
            synthetic = _claim_synthetic_id(open_calls, msg, positional)
            if synthetic is not None:
                rewritten[index] = msg.model_copy(update={"tool_call_id": synthetic})

        if not rewritten:
            return messages
        return [rewritten.get(index, msg) for index, msg in enumerate(messages)]

    def _build_patched_messages(self, messages: list) -> list | None:
        """tool 결과를 해당 tool-call AIMessage 뒤에 모아 놓은 메시지를 반환한다.

        provider 직렬화 전에 model에 실릴 인과 순서를 정규화하되, 이미 유효한 대화 기록은
        그대로 둔다.
        """
        normalized = self._normalize_tool_call_ids(messages)

        tool_messages_by_id: dict[str, deque[ToolMessage]] = defaultdict(deque)
        for msg in normalized:
            if isinstance(msg, ToolMessage):
                tool_messages_by_id[msg.tool_call_id].append(msg)

        tool_call_ids: set[str] = set()
        for msg in normalized:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                if tc_id:
                    tool_call_ids.add(tc_id)

        patched: list = []
        patch_count = 0
        drop_count = 0
        for msg in normalized:
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id in tool_call_ids:
                    continue  # 해당 AIMessage 뒤에서 다시 내보낸다
                # orphan: 출처 AIMessage tool_call이 더 이상 요청에 없는 ToolMessage
                # (예: summarization이 제거한 경우). 엄격한 provider가 HTTP 400으로 거부하지
                # 않도록 model 요청에서 조용히 제거한다. 영속 state는 건드리지 않으며 이 한 번의
                # model 호출에만 영향을 준다.
                drop_count += 1
                continue

            sanitized_msg = self._sanitize_ai_message_tool_calls(msg)
            patched.append(sanitized_msg)
            if getattr(msg, "type", None) != "ai":
                continue

            # 정리된 메시지가 대체하기 전에 빈 이름을 분류할 수 있도록 의도적으로 원본
            # 메시지를 검사한다.
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                if not tc_id:
                    continue

                tool_msg_queue = tool_messages_by_id.get(tc_id)
                existing_tool_msg = tool_msg_queue.popleft() if tool_msg_queue else None
                if existing_tool_msg is not None:
                    if tc.get("invalid_tool_name") and _has_invalid_tool_name(existing_tool_msg.name):
                        existing_tool_msg = existing_tool_msg.model_copy(update={"name": tc["name"]})
                    patched.append(existing_tool_msg)
                else:
                    patched.append(
                        ToolMessage(
                            content=self._synthetic_tool_message_content(tc),
                            tool_call_id=tc_id,
                            name=tc.get("name", "unknown"),
                            status="error",
                        )
                    )
                    patch_count += 1

        if patched == messages and not drop_count:
            return None
        if drop_count or patch_count:
            logger.warning(
                "DanglingToolCallMiddleware: %d orphan(s) dropped, %d placeholder(s) injected",
                drop_count,
                patch_count,
            )
        return patched

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return await handler(request)
