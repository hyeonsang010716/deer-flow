"""streaming과 영속화에 쓰는 간결한 subagent step payload를 만든다.

이슈 #3779: subagent(subtask) 실행 step은 가장 최근 stream 프레임으로만 보였고 영속화되지
않아서, 사용자가 새로고침 후 subagent가 어떤 도구를 돌렸고 각 step이 무엇을 만들었는지 확인할
수 없었다.

이 모듈은 순수 데이터 정형 레이어다. 캡처된 subagent 메시지 dict — ``AIMessage``(텍스트 +
tool-call 요청인 assistant turn)나 ``ToolMessage``(도구 출력)의 ``model_dump()`` — 를 작고
JSON 직렬화 가능한 ``step`` payload로 변환한다. 이 payload는:

- ``task_running`` custom event(``task_tool.py``) 안에서 실시간으로 stream되고,
- ``subagent.step`` run event(``runtime/runs/worker.py``)로 영속화된다.

순수하게 유지하면 graph를 띄우지 않고 단위 테스트할 수 있고, streaming과 영속화 호출 지점이
"step"의 정의 하나를 공유한다.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from deerflow.runtime.events.catalog import (
    SUBAGENT_END_EVENT,
    SUBAGENT_START_EVENT,
    SUBAGENT_STEP_EVENT,
)
from deerflow.utils.messages import message_content_to_text

from .status_contract import normalize_token_usage

#: ``text`` 필드의 step당 기본 문자 수 상한. 도구 출력(웹 검색 결과, 파일 내용)은 커질 수
#: 있으므로 이 상한이 영속화되는 run-event 행과 stream 프레임을 제한한다. 표시/저장에만
#: 영향을 주며, subagent 자신의 LLM context는 ToolOutputBudgetMiddleware가 따로 제한한다.
SUBAGENT_STEP_MAX_CHARS = 8192

#: 영속화되는 subagent step의 ``RunEvent.category``. ``"message"``가 아닌 전용 category를 쓰면
#: 이 event들이 ``list_messages``(thread 메시지 피드)에서 빠지면서도 펼칠 때 가져오는
#: ``list_events``에서는 반환된다(#3779).
SUBAGENT_EVENT_CATEGORY = SUBAGENT_START_EVENT.category

#: ``task_*`` 종료 custom-event 타입을 영속화 status로 매핑한다.
_TERMINAL_EVENT_STATUS: dict[str, str] = {
    "task_completed": "completed",
    "task_failed": "failed",
    "task_cancelled": "cancelled",
    "task_timed_out": "timed_out",
}


def capture_step_message(
    message: BaseMessage,
    captured: list[dict[str, Any]],
    seen_ids: set[str],
) -> bool:
    """새 step이면 ``message.model_dump()``를 ``captured``에 추가한다.

    "step"은 assistant turn(``AIMessage``)이거나 도구 결과(``ToolMessage``)다. 후자는 도구
    출력이 살아남도록 이슈 #3779에서 추가했다. 그 외 메시지 타입(예: ``HumanMessage``)은
    무시한다. 중복 제거는 id가 있으면 id로, 없으면 전체 dict 비교로 한다. 그래야
    ``stream_mode="values"``가 같은 마지막 메시지를 다시 내보내도 O(1)로 유지된다.

    메시지를 추가했으면 ``True``를 반환한다.
    """
    if not isinstance(message, (AIMessage, ToolMessage)):
        return False

    message_dict = message.model_dump()
    message_id = message_dict.get("id")
    if message_id:
        if message_id in seen_ids:
            return False
    elif message_dict in captured:
        return False

    captured.append(message_dict)
    if message_id:
        seen_ids.add(message_id)
    return True


def capture_new_step_messages(
    messages: list[BaseMessage],
    captured: list[dict[str, Any]],
    seen_ids: set[str],
    processed_count: int,
) -> int:
    """``processed_count`` 이후 추가된 모든 step 메시지를 캡처한다(#3779).

    ``stream_mode="values"``는 chunk마다 전체 메시지 history를 다시 내보내고, LangGraph
    super-step 하나가 여러 메시지를 한 번에 추가할 수 있다. 특히 모델이 한 turn에 여러 tool
    call을 내면 tool call마다 ``ToolMessage``가 하나씩 붙는다. ``messages[-1]``만 캡처하던
    이전 동작은 마지막 하나를 뺀 모든 도구 출력을 조용히 버렸다.

    history가 늘어났으면 새로 추가된 모든 메시지를 훑는다. 늘지 않았으면 마지막 메시지만 다시
    검사해서 id 없는 in-place 교체(길이는 같고 내용이 새로운 경우)도 캡처되게 한다. 변화가
    없는 재전송은 ``capture_step_message``의 중복 제거가 no-op으로 만든다. 새 cursor를 반환한다.

    history가 *줄어든* 경우(``total < processed_count``) — ``DeerFlowSummarizationMiddleware``가
    ``RemoveMessage(id=REMOVE_ALL_MESSAGES)``로 채널을 다시 쓸 때 발생한다(#3875 Phase 3) —
    cursor를 새 tail로 리셋하고, compaction 이전에 캡처한 step의 재전송은
    ``capture_step_message``의 id/내용 중복 제거에 맡긴다. 이 리셋이 없으면 compaction 지점
    이후에 추가된 모든 step은 ``total``이 낡은 cursor를 따라잡을 때까지 버려진다.

    INVARIANT: 리셋 후 무증가 분기는 ``messages[-1]``만 다시 검사하므로, compaction된 목록에서
    리셋 cursor보다 아래 인덱스에 진짜 새로운 AIMessage/ToolMessage가 삽입되면 놓친다. 현재는
    도달 불가능하다. summarization middleware는 요약을 별도의 ``summary_text`` state 키에 넣고,
    compaction 후 messages 채널에는 이미 본 보존된 tail 메시지만 남는다 — compaction은 cursor
    아래에 캡처 대상이 되는 새 메시지를 삽입하지 않는다. 향후 middleware가 이 invariant를 깨면
    리셋 분기는 전체 재스캔이 필요하다.
    """
    total = len(messages)
    if total < processed_count:
        processed_count = total
    if total > processed_count:
        for message in messages[processed_count:total]:
            capture_step_message(message, captured, seen_ids)
        return total
    if messages:
        capture_step_message(messages[-1], captured, seen_ids)
    return max(processed_count, total)


def truncate_step_text(text: str, max_chars: int) -> tuple[str, bool]:
    """``(text, truncated)``를 반환하며, ``max_chars``보다 길면 잘라낸다."""
    if max_chars >= 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def _bounded_tool_call(call: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """캡처된 tool call에 대해 ``{name, args}``를 반환하며 큰 args는 잘라낸다(#3779).

    ``build_subagent_step``은 ``text`` 필드를 제한하지만 tool-call ``args``는 그대로 복사되어,
    큰 payload(파일 전체 내용, heredoc)를 실은 ``write_file``/``bash`` 호출이 무제한 크기의
    ``subagent.step`` 행과 stream 프레임을 만들었다. JSON 직렬화한 args가 ``max_chars``를 넘으면
    구조화된 값을 잘라낸 직렬화 미리보기로 바꾸고 ``args_truncated``로 표시한다. 작은 args는
    카드가 살펴볼 수 있도록 구조화된 상태로 둔다.
    """
    name = call.get("name")
    args = call.get("args")
    serialized = args if isinstance(args, str) else json.dumps(args, default=str, ensure_ascii=False)
    if max_chars >= 0 and len(serialized) > max_chars:
        return {"name": name, "args": serialized[:max_chars], "args_truncated": True}
    return {"name": name, "args": args}


def build_subagent_step(
    message: dict[str, Any],
    *,
    task_id: str,
    message_index: int,
    max_chars: int = SUBAGENT_STEP_MAX_CHARS,
) -> dict[str, Any]:
    """캡처된 subagent 메시지 dict에서 간결한 step payload를 만든다.

    ``kind``는 ToolMessage(``type == "tool"``)면 ``"tool"``, 그 외에는 ``"ai"``다. AI step은
    ``tool_calls``(name + args만, 큰 args는 ``max_chars``로 제한 — ``_bounded_tool_call`` 참고)를
    싣고, tool step은 출처 ``tool_name``을 싣는다. ``text``는 ``max_chars``로 잘리며 그에 맞춰
    ``truncated`` 플래그가 설정된다.
    """
    kind = "tool" if message.get("type") == "tool" else "ai"
    # ``... or ""``는 tool-call만 있는 turn의 content=None이 ""로 렌더링되게 유지한다
    # (그렇지 않으면 message_content_to_text가 str()로 "None"을 만든다).
    text, truncated = truncate_step_text(message_content_to_text(message.get("content") or ""), max_chars)

    step: dict[str, Any] = {
        "task_id": task_id,
        "message_index": message_index,
        "kind": kind,
        "text": text,
        "truncated": truncated,
    }

    if kind == "tool":
        step["tool_name"] = message.get("name")
    else:
        step["tool_calls"] = [_bounded_tool_call(call, max_chars) for call in (message.get("tool_calls") or [])]

    return step


def subagent_run_event(chunk: Any) -> dict[str, Any] | None:
    """``task_*`` custom stream chunk를 ``RunEventStore.put`` kwargs로 매핑한다.

    영속화 가능한 subagent lifecycle event면 ``event_type`` / ``category`` / ``content`` /
    ``metadata``를 반환하고, 유효한 subagent event가 아닌 chunk에는 ``None``을 반환한다
    (그래서 worker는 인식한 것만 영속화한다). ``thread_id`` / ``run_id``는 호출자가 채운다.
    """
    if not isinstance(chunk, dict):
        return None

    event = chunk.get("type")
    if not isinstance(event, str) or not event.startswith("task_"):
        return None

    task_id = chunk.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None

    if event == "task_started":
        description = chunk.get("description")
        if description is not None and not isinstance(description, str):
            return None
        return {
            "event_type": SUBAGENT_START_EVENT.event_type,
            "category": SUBAGENT_START_EVENT.category,
            "content": {"task_id": task_id, "description": description},
            "metadata": {"task_id": task_id},
        }

    if event == "task_running":
        message_index = chunk.get("message_index")
        message = chunk.get("message")
        if isinstance(message_index, bool) or not isinstance(message_index, int) or message_index < 0:
            return None
        if not isinstance(message, dict):
            return None
        return {
            "event_type": SUBAGENT_STEP_EVENT.event_type,
            "category": SUBAGENT_STEP_EVENT.category,
            "content": build_subagent_step(message, task_id=task_id, message_index=message_index),
            "metadata": {"task_id": task_id, "message_index": message_index},
        }

    status = _TERMINAL_EVENT_STATUS.get(event)
    if status is not None:
        content: dict[str, Any] = {"task_id": task_id, "status": status}
        model_name = chunk.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            content["model_name"] = model_name.strip()
        usage = normalize_token_usage(chunk.get("usage"))
        if usage is not None:
            content["usage"] = usage
        # 최종 result/error는 여러 페이지짜리 보고서일 수 있으므로 잘라서 영속화되는 run-event
        # 행 크기를 제한한다(원문은 종료 ToolMessage에 그대로 남고, 카드는 그쪽을 따로 읽는다).
        if chunk.get("result") is not None:
            result, result_truncated = truncate_step_text(str(chunk["result"]), SUBAGENT_STEP_MAX_CHARS)
            content["result"] = result
            if result_truncated:
                content["result_truncated"] = True
        if chunk.get("error") is not None:
            error, error_truncated = truncate_step_text(str(chunk["error"]), SUBAGENT_STEP_MAX_CHARS)
            content["error"] = error
            if error_truncated:
                content["error_truncated"] = True
        return {
            "event_type": SUBAGENT_END_EVENT.event_type,
            "category": SUBAGENT_END_EVENT.category,
            "content": content,
            "metadata": {"task_id": task_id},
        }

    return None
