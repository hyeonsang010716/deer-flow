"""구조화된 subagent 결과 metadata에 대한 backend↔frontend 계약.

``task`` 도구의 결과 텍스트는 모델에게 보이는 표시용 콘텐츠다. runtime 소비자는
``ToolMessage.additional_kwargs``에 실린 구조화된 사실을 읽는다:

- ``subagent_status``: ``SUBAGENT_STATUS_VALUES`` 중 하나.
- ``subagent_stop_reason`` (선택): guardrail cap이 run을 조기 종료시켰을 때
  ``SUBAGENT_STOP_REASON_VALUES``(``token_capped`` / ``turn_capped`` /
  ``loop_capped``) 중 하나. 추가 전용 필드다(#3875 Phase 2). cap이 걸렸지만 최종 답변을
  만들어낸 run은 ``status=completed``를 유지하면서 여기에 cap을 싣고, 쓸 수 있는 출력이
  없는 run은 ``status=failed`` + ``stop_reason``이 된다. 구버전 frontend는 모르는 필드를
  무시한다.
- ``subagent_error`` (선택): backend가 기록한 사람이 읽을 수 있는 에러 blob.
- ``subagent_result_brief`` / ``subagent_result_sha256`` (선택): 길이를 제한한 완료 결과
  metadata와 전체 결과의 digest.
- ``subagent_model_name`` (선택): 이 위임 run이 사용한 실제 DeerFlow 모델 식별자.
- ``subagent_token_usage`` (선택): provider가 보고한 경우의 최종 누적 ``input_tokens`` /
  ``output_tokens`` / ``total_tokens`` snapshot.

``contracts/subagent_status_contract.json``의 공용 fixture가 Python과 TypeScript 양쪽에서
enum 값을 고정한다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

SUBAGENT_STATUS_KEY = "subagent_status"
SUBAGENT_STOP_REASON_KEY = "subagent_stop_reason"
SUBAGENT_ERROR_KEY = "subagent_error"
SUBAGENT_RESULT_BRIEF_KEY = "subagent_result_brief"
SUBAGENT_RESULT_SHA256_KEY = "subagent_result_sha256"
SUBAGENT_MODEL_NAME_KEY = "subagent_model_name"
SUBAGENT_TOKEN_USAGE_KEY = "subagent_token_usage"
SUBAGENT_METADATA_TEXT_MAX_CHARS = 2000

#: 생산자는 항상 ``hashlib.sha256(...).hexdigest()`` — 소문자 hex 64자 — 를 내보낸다.
#: reader도 같은 형태를 강제해서, 손상된 중계 값이 digest 행세를 하지 못하게 한다.
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")

SubagentStatusValue = Literal[
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "polling_timed_out",
]

#: ``subagent_status``가 가질 수 있는 모든 값의 열거. 공용 fixture의
#: ``valid_status_values`` 배열과 대응하며, contract 테스트가 서로를 고정한다. cap이 걸린
#: run에는 별도 status 값을 주지 않는다(#3875 Phase 2). 출력이 있으면 ``completed``,
#: 없으면 ``failed``이고 이유는 추가 필드 ``subagent_stop_reason``에 실어서 구버전 소비자도
#: 계속 동작하게 한다.
SUBAGENT_STATUS_VALUES: tuple[SubagentStatusValue, ...] = (
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "polling_timed_out",
)

#: guardrail cap이 run을 조기 종료시킨 이유. status enum 값이 아니라 추가 필드
#: ``subagent_stop_reason``에 싣는다.
SubagentStopReasonValue = Literal["token_capped", "turn_capped", "loop_capped"]

SUBAGENT_STOP_REASON_VALUES: tuple[SubagentStopReasonValue, ...] = (
    "token_capped",
    "turn_capped",
    "loop_capped",
)

#: cap이 발동했을 때 모델에게 보이는 결과 텍스트에 끼워 넣는 사람이 읽을 수 있는 label.
#: 예: ``Task Succeeded (capped: token budget). Result: ...``.
_STOP_REASON_LABELS: dict[SubagentStopReasonValue, str] = {
    "token_capped": "token budget",
    "turn_capped": "turn budget",
    "loop_capped": "repeated tool-call loop",
}

#: ``subagent_result_brief`` / ``subagent_result_sha256``에 복구 가능한 결과를 싣는 status.
#: ``completed``뿐이다. cap이 걸렸어도 쓸 만한 부분 결과를 낸 run은 ``completed``
#: (+ ``stop_reason``)로 드러나므로, 그 작업물은 깨끗한 성공과 똑같이 wire를 타고 살아남는다.
#: 그 외 비완료 status는 ``subagent_error``만 싣는다.
_RESULT_BEARING_STATUSES: frozenset[SubagentStatusValue] = frozenset({"completed"})

#: 예전 checkpoint된 thread history에는 있지만 더 이상 생산되지 않는 status 값을 읽는 쪽에서
#: 정규화한다. ``max_turns_reached``는 Phase 1(#3949)이 내보냈고 영속화된
#: ``ToolMessage.additional_kwargs``에 남아 있다. #3980이 생산자와 contract fixture에서
#: 제거했지만, reader는 여전히 Phase 2의 cap 등가물로 매핑한다. 그래야 과거 데이터가
#: delegation ledger에서 ``in_progress``로 묶이지 않고 (cap은 ``stop_reason``에 실린 채)
#: 종료 상태로 해석된다. frontend ``subtask-result.ts``도 같은 이유로 대응되는 deprecated
#: alias를 유지한다.
_LEGACY_STATUS_NORMALIZATION: dict[str, SubagentStopReasonValue] = {
    "max_turns_reached": "turn_capped",
}


class StructuredSubagentResult(TypedDict):
    status: SubagentStatusValue
    stop_reason: NotRequired[SubagentStopReasonValue]
    result_brief: NotRequired[str]
    result_sha256: NotRequired[str]
    error: NotRequired[str]


def _bound_metadata_text(text: str, cap: int = SUBAGENT_METADATA_TEXT_MAX_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= cap:
        return cleaned
    marker = "\n...\n"
    if cap <= len(marker):
        return cleaned[:cap]
    head = cap * 2 // 3
    tail = cap - head - len(marker)
    if tail <= 0:
        return cleaned[:cap]
    return f"{cleaned[:head]}{marker}{cleaned[-tail:]}"


def make_subagent_additional_kwargs(
    status: SubagentStatusValue,
    *,
    result: str | None = None,
    error: str | None = None,
    stop_reason: SubagentStopReasonValue | None = None,
    model_name: str | None = None,
    token_usage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """middleware가 찍는 ``additional_kwargs`` payload를 만든다.

    error 필드가 비어 있으면 아예 빼서, JSON wire 포맷이 오해를 부르는 빈
    ``subagent_error: ""``를 싣지 않게 한다. ``stop_reason``은 guardrail cap이 run을
    끝냈을 때만 찍는다(:data:`SUBAGENT_STOP_REASON_VALUES` 참고).

    Raises:
        ValueError: ``status``가 :data:`SUBAGENT_STATUS_VALUES`에 없거나,
            ``stop_reason``이 :data:`SUBAGENT_STOP_REASON_VALUES`에 없을 때.
            임의 문자열은 받지 않는다. 오타가 있으면 생산자 경계에서 크게 실패하는 대신
            metadata 누락으로 소비자에게 조용히 흘러가기 때문이다.
    """
    if status not in SUBAGENT_STATUS_VALUES:
        raise ValueError(f"invalid subagent status {status!r}; expected one of {SUBAGENT_STATUS_VALUES}")
    if stop_reason is not None and stop_reason not in SUBAGENT_STOP_REASON_VALUES:
        raise ValueError(f"invalid subagent stop_reason {stop_reason!r}; expected one of {SUBAGENT_STOP_REASON_VALUES}")
    payload: dict[str, object] = {SUBAGENT_STATUS_KEY: status}
    if status in _RESULT_BEARING_STATUSES and isinstance(result, str) and result.strip():
        payload[SUBAGENT_RESULT_BRIEF_KEY] = _bound_metadata_text(result)
        payload[SUBAGENT_RESULT_SHA256_KEY] = hashlib.sha256(result.encode("utf-8")).hexdigest()
    # ``completed``(깨끗한 성공, 또는 부분 결과가 살아남은 cap된 run)만 error blob을 억제한다.
    # 나머지 status는 모두 싣는다.
    if status != "completed" and isinstance(error, str) and error.strip():
        payload[SUBAGENT_ERROR_KEY] = _bound_metadata_text(error)
    if stop_reason is not None:
        payload[SUBAGENT_STOP_REASON_KEY] = stop_reason
    if isinstance(model_name, str) and model_name.strip():
        payload[SUBAGENT_MODEL_NAME_KEY] = model_name.strip()
    normalized_usage = normalize_token_usage(token_usage)
    if normalized_usage is not None:
        payload[SUBAGENT_TOKEN_USAGE_KEY] = normalized_usage
    return payload


def normalize_token_usage(value: Any) -> dict[str, int] | None:
    """누적 token-usage mapping을 검증해 계약 형태로 만든다.

    두 metadata 표면 — 종료 ``ToolMessage`` metadata(여기)와 영속화되는
    ``subagent.step`` / ``subagent.end`` run event(``step_events.py``) — 가 공유하는 단일
    validator다. 함수를 하나로 유지해야 둘이 어긋나지 않는다(예: 한쪽만 추가 token 필드를
    받아들여 한쪽 경로에서 usage가 조용히 사라지는 경우). 세 키 모두 음수가 아닌 ``int``를
    요구하며 ``bool``은 거부한다. mapping이 아니거나 형식이 잘못된 입력에는 ``None``을 반환한다.
    """
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        amount = value.get(key)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            return None
        normalized[key] = amount
    return normalized


def format_subagent_result_message(
    status: SubagentStatusValue,
    *,
    result: str | None = None,
    error: str | None = None,
    stop_reason: SubagentStopReasonValue | None = None,
) -> tuple[str, str | None]:
    """모델에게 보이는 task 콘텐츠와 정규화된 metadata 에러를 반환한다.

    ``stop_reason``이 설정되면 짧은 ``(capped: ...)`` 주석을 텍스트에 끼워 넣어, lead agent가
    metadata를 파싱하지 않고도 run이 guardrail cap으로 끝났음을 알 수 있게 한다. 쓸 만한 결과를
    낸 cap된 run은 ``status=completed``(+ 부분 결과)이고, 쓸 만한 출력이 없으면
    ``status=failed``이다.
    """
    result_text = "" if result is None else str(result)
    error_text = str(error).strip() if isinstance(error, str) else ""
    capped = _STOP_REASON_LABELS.get(stop_reason) if stop_reason is not None else None

    if status == "completed":
        if capped:
            return f"Task Succeeded (capped: {capped}). Result: {result_text}", None
        return f"Task Succeeded. Result: {result_text}", None

    if status == "cancelled":
        detail = error_text or "Task cancelled by user."
        if detail == "Task cancelled by user.":
            return detail, detail
        return f"Task cancelled by user. Error: {detail}", detail

    if status == "timed_out":
        detail = error_text or "Task timed out."
        if detail == "Task timed out.":
            return detail, detail
        return f"Task timed out. Error: {detail}", detail

    if status == "polling_timed_out":
        detail = error_text or "Task polling timed out."
        return detail, detail

    # ``failed`` — 쓸 만한 출력을 내지 못한 turn-capped run(``stop_reason=turn_capped``) 포함.
    # cap 주석을 끼워 넣어 lead가 고장 난 subagent와 단순히 turn budget이 떨어진 subagent를
    # 구분할 수 있게 한다.
    detail = error_text or "Task failed."
    if capped:
        if detail == "Task failed.":
            return f"Task failed (capped: {capped}).", detail
        return f"Task failed (capped: {capped}). Error: {detail}", detail
    if detail == "Task failed.":
        return detail, detail
    return f"Task failed. Error: {detail}", detail


def read_subagent_result_metadata(
    additional_kwargs: Mapping[str, object] | None,
) -> StructuredSubagentResult | None:
    if not additional_kwargs:
        return None
    raw_status = additional_kwargs.get(SUBAGENT_STATUS_KEY)
    # 예전 checkpoint 값(#3949)은 더 이상 생산되지 않지만(#3980) 영속화된 history에는 남아 있다.
    # 유효성 검사 전에 정규화해서 ``None``을 반환하는 대신(그러면 delegation 항목이
    # ``in_progress``로 묶인다) 종료 상태로 해석되게 한다. 예전 ``max_turns_reached``는 복구된
    # 부분 결과를 실었으므로, ``result_brief``가 아직 있는 payload는 Phase 2의
    # ``completed + turn_capped`` 형태로 매핑되고(부분 결과가 wire에서 살아남는다), 결과가 없는
    # payload는 ``failed + turn_capped``로 매핑된다.
    legacy_stop_reason = _LEGACY_STATUS_NORMALIZATION.get(raw_status) if isinstance(raw_status, str) else None
    if legacy_stop_reason is not None:
        raw_result_brief = additional_kwargs.get(SUBAGENT_RESULT_BRIEF_KEY)
        status = "completed" if (isinstance(raw_result_brief, str) and raw_result_brief.strip()) else "failed"
    elif raw_status in SUBAGENT_STATUS_VALUES:
        status = raw_status
    else:
        return None
    payload: StructuredSubagentResult = {"status": status}
    raw_result = additional_kwargs.get(SUBAGENT_RESULT_BRIEF_KEY)
    raw_hash = additional_kwargs.get(SUBAGENT_RESULT_SHA256_KEY)
    raw_error = additional_kwargs.get(SUBAGENT_ERROR_KEY)
    if status in _RESULT_BEARING_STATUSES and isinstance(raw_result, str) and raw_result.strip():
        payload["result_brief"] = _bound_metadata_text(raw_result)
        if isinstance(raw_hash, str) and _SHA256_HEX_RE.fullmatch(raw_hash):
            payload["result_sha256"] = raw_hash
    if status != "completed" and isinstance(raw_error, str) and raw_error.strip():
        payload["error"] = _bound_metadata_text(raw_error)
    # wire에 실린 명시적 stop_reason이 우선하고, 없으면 합성한 legacy 이유를 쓴다.
    raw_stop_reason = additional_kwargs.get(SUBAGENT_STOP_REASON_KEY)
    if isinstance(raw_stop_reason, str) and raw_stop_reason in SUBAGENT_STOP_REASON_VALUES:
        payload["stop_reason"] = raw_stop_reason
    elif legacy_stop_reason is not None:
        payload["stop_reason"] = legacy_stop_reason
    return payload
