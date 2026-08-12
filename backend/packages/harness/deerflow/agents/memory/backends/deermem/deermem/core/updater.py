"""memory 데이터를 읽고 쓰고 갱신하는 memory updater."""

import asyncio
import atexit
import concurrent.futures
import copy
import html
import json
import logging
import math
import re
import uuid
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import DeerMemConfig
from .message_processing import detect_signals, extract_message_text
from .prompt import (
    format_conversation_for_update,
    load_prompt,
    load_prompt_messages,
)
from .storage import (
    MemoryManifestRevisionConflict,
    MemoryStorage,
    create_empty_memory,
    utc_now_iso_z,
)

logger = logging.getLogger(__name__)


# async context에서 호출될 때 동기 memory 갱신을 떠넘길 thread pool. 이전의
# asyncio.run() 방식과 달리 *동기* model.invoke()를 실행하므로 event loop가 만들어지지
# 않는다. 따라서 (@lru_cache로 전역 캐시되는) langchain async httpx client pool을
# 건드리지 않고, loop 간 connection 재사용도 발생할 수 없다.
_SYNC_MEMORY_UPDATER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="memory-updater-sync",
)
atexit.register(lambda: _SYNC_MEMORY_UPDATER_EXECUTOR.shutdown(wait=False))


# 데이터 접근 + fact CRUD 함수(_save_memory_to_file / get_memory_data /
# reload_memory_data / import_memory_data / clear_memory_data / create_memory_fact /
# delete_memory_fact / update_memory_fact)는 MemoryUpdater의 인스턴스 메서드로
# 옮겨졌다(self._storage를 쓴다). 아래 클래스를 참고한다.
def _validate_confidence(confidence: float) -> float:
    """저장된 JSON이 표준을 지키도록 fact의 confidence를 검증한다."""
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError("confidence")
    return confidence


def _coerce_source_confidence(fact: dict[str, Any]) -> float:
    """저장된 fact의 confidence를 [0, 1] 범위의 유한한 float로 반환한다. 기본값은 0.5다.

    dict.get(key, default)는 키가 있으면 저장된 값을(None 포함) 그대로 반환하므로,
    "confidence": null로 기록된 fact는 None을 산술 연산에 흘려보내 max()를 터뜨린다.
    이 헬퍼는 손상되거나 수동 편집된 memory 파일에서 오는 null, bool, 비숫자,
    비유한 값을 모두 막는다.
    """
    raw = fact.get("confidence")
    if raw is None or isinstance(raw, bool):
        return 0.5
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(val, 1.0)) if math.isfinite(val) else 0.5


def _trim_facts_to_max(facts: list[dict[str, Any]], max_facts: int) -> list[dict[str, Any]]:
    """confidence가 높은 순으로 ``max_facts``개만 남긴다(confidence는 강제 변환한다).

    confidence는 :func:`_coerce_source_confidence`로 읽으므로 ``null``이나 비숫자
    confidence를 가진 legacy/import된 fact가 정렬을 터뜨리지 않는다. #4023 이전의
    ``key=lambda f: f.get("confidence", 0)`` 형태는 ``None``/``str``을 ``float``와
    비교해 ``len(facts) > max_facts``인 순간 ``TypeError``를 냈다. upstream의
    ``_trim_facts_to_max``(#4023에서 도입)와 동일하게 맞춰, monolithic->vendored 이름
    변경 과정에서 조용히 누락됐던 강제 변환 수정이 vendored 사본에도 반영되게 한다.
    """
    if len(facts) <= max_facts:
        return facts
    return sorted(facts, key=_coerce_source_confidence, reverse=True)[:max_facts]


def _extract_text(content: Any) -> str:
    """LLM 응답 content(str 또는 content block 리스트)에서 평문 텍스트를 뽑는다.

    최신 LLM은 평문 대신 [{"type": "text", "text": "..."}] 같은 block 리스트로 구조화된
    content를 반환할 수 있다. 그런 값에 str()을 쓰면 실제 텍스트가 아니라 Python repr이
    나와 이후 JSON 파싱이 깨진다.

    문자열 조각은 구분자 없이 이어 붙인다. 조각난 JSON/텍스트 payload를 망가뜨리지 않기
    위해서다. dict 기반 text block은 완결된 텍스트 블록으로 보고 가독성을 위해 줄바꿈으로
    잇는다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        pending_str_parts: list[str] = []

        def flush_pending_str_parts() -> None:
            if pending_str_parts:
                pieces.append("".join(pending_str_parts))
                pending_str_parts.clear()

        for block in content:
            if isinstance(block, str):
                pending_str_parts.append(block)
            elif isinstance(block, dict):
                flush_pending_str_parts()
                text_val = block.get("text")
                if isinstance(text_val, str):
                    pieces.append(text_val)

        flush_pending_str_parts()
        return "\n".join(pieces)
    return str(content)


_REQUIRED_MEMORY_UPDATE_TOP_LEVEL_KEYS = frozenset({"user", "history", "newFacts"})
_FACT_CLASSIFICATION_FIELDS = ("scope", "durability", "authority")


def _normalize_gate_label(value: Any) -> str | None:
    """모델이 만든 scope gate 라벨을 정규화한다. 정책 검증은 하지 않는다."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _fact_scope_gate_reason(fact: dict[str, Any]) -> str | None:
    """모델이 추출한 fact의 결정적 거부 사유를 반환한다."""
    if any(_normalize_gate_label(fact.get(field)) is None for field in _FACT_CLASSIFICATION_FIELDS):
        return "missing"
    if _normalize_gate_label(fact.get("scope")) != "user":
        return "scope"
    if _normalize_gate_label(fact.get("durability")) != "durable":
        return "durability"
    if _normalize_gate_label(fact.get("authority")) != "descriptive":
        return "authority"
    return None


def _summary_scope_gate_reason(section_data: dict[str, Any]) -> str | None:
    """summary 갱신의 결정적 거부 사유를 반환한다."""
    scope = _normalize_gate_label(section_data.get("scope"))
    authority = _normalize_gate_label(section_data.get("authority"))
    if scope is None or authority is None:
        return "missing"
    if scope != "user":
        return "scope"
    if authority != "descriptive":
        return "authority"
    return None


def _removal_scope_gate_reason(removal: dict[str, Any]) -> str | None:
    """모순으로 인한 삭제 요청의 결정적 거부 사유를 반환한다."""
    scope = _normalize_gate_label(removal.get("scope"))
    reason = removal.get("reason")
    if scope is None or not isinstance(reason, str) or not reason.strip():
        return "missing"
    if scope != "user":
        return "scope"
    return None


def _normalize_memory_update_fact(fact: Any) -> dict[str, Any] | None:
    """모델이 만든 memory 갱신 결과에서 fact 항목 하나를 정규화한다."""
    if not isinstance(fact, dict):
        return None

    raw_content = fact.get("content")
    if not isinstance(raw_content, str):
        return None
    content = raw_content.strip()
    if not content:
        return None

    raw_category = fact.get("category")
    category = raw_category.strip() if isinstance(raw_category, str) and raw_category.strip() else "context"

    raw_confidence = fact.get("confidence", 0.5)
    if isinstance(raw_confidence, bool):
        return None
    if isinstance(raw_confidence, str):
        raw_confidence = raw_confidence.strip()
        if not raw_confidence:
            return None
        try:
            raw_confidence = float(raw_confidence)
        except ValueError:
            return None
    elif isinstance(raw_confidence, (int, float)):
        raw_confidence = float(raw_confidence)
    else:
        return None

    if not math.isfinite(raw_confidence):
        return None

    normalized_fact = {
        "content": content,
        "category": category,
        "confidence": raw_confidence,
    }
    source_error = fact.get("sourceError")
    if isinstance(source_error, str):
        normalized_source_error = source_error.strip()
        if normalized_source_error:
            normalized_fact["sourceError"] = normalized_source_error

    # fact 수명(expected_valid_days): LLM이 선택적으로 지정하는 재검토 주기.
    # 공용 _read_expected_valid_days 규칙으로 검증한다(bool 거부, 유한값 요구, int로
    # 강제 변환, 양수만 유지). 상한은 _apply_updates에서 적용한다.
    evd = _read_expected_valid_days(fact)
    if evd is not None:
        normalized_fact["expected_valid_days"] = evd

    # scope 분류는 추출 단계 전용 메타데이터다. _apply_updates가 항목별로 fail closed할
    # 수 있도록 구조 정규화 과정에서는 유지하되, 저장되는 fact_entry에는 절대 복사하지
    # 않는다.
    for field in _FACT_CLASSIFICATION_FIELDS:
        normalized_value = _normalize_gate_label(fact.get(field))
        if normalized_value is not None:
            normalized_fact[field] = normalized_value

    return normalized_fact


def _normalize_memory_update_data(update_data: dict[str, Any]) -> dict[str, Any]:
    """파싱한 memory 갱신 데이터를 _apply_updates가 소비하는 형태로 변환한다."""
    user = update_data.get("user")
    history = update_data.get("history")
    new_facts = update_data.get("newFacts")
    facts_to_remove = update_data.get("factsToRemove")
    normalized_facts_to_remove: list[dict[str, Any]] = []
    if isinstance(facts_to_remove, list):
        for entry in facts_to_remove:
            # legacy 문자열 형태는 분류 없는 삭제 요청으로 남긴다. apply 계층의 gate가
            # 이를 missing으로 거부하므로, scope 없는 파괴적 변경이 계속 허용되지 않는다.
            if isinstance(entry, str):
                fact_id = entry.strip()
                if fact_id:
                    normalized_facts_to_remove.append({"id": fact_id})
                continue
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            normalized_removal: dict[str, Any] = {"id": raw_id.strip()}
            scope = _normalize_gate_label(entry.get("scope"))
            if scope is not None:
                normalized_removal["scope"] = scope
            reason = entry.get("reason")
            if isinstance(reason, str) and reason.strip():
                normalized_removal["reason"] = reason.strip()
            if "replacementFactIndex" in entry:
                # 잘못된 값도 그대로 보존한다. apply 계층이 잘못된 의존 관계를 거부해야
                # 하며, 조용히 순수 삭제로 처리해 기존 fact를 지워서는 안 된다.
                normalized_removal["replacementFactIndex"] = entry.get("replacementFactIndex")
            normalized_facts_to_remove.append(normalized_removal)
    normalized_new_facts = []
    dropped_new_fact = not isinstance(new_facts, list)
    if isinstance(new_facts, list):
        for fact in new_facts:
            normalized_fact = _normalize_memory_update_fact(fact)
            if normalized_fact is not None:
                normalized_new_facts.append(normalized_fact)
            else:
                dropped_new_fact = True

    if normalized_facts_to_remove and dropped_new_fact:
        raise json.JSONDecodeError(
            "Unsafe partial memory update: factsToRemove with malformed newFacts",
            json.dumps(update_data, ensure_ascii=False),
            0,
        )

    # ── staleness 검토에 따른 삭제 요청 정규화 ──
    stale_removals_raw = update_data.get("staleFactsToRemove")
    normalized_stale_removals: list[dict[str, str]] = []
    if isinstance(stale_removals_raw, list):
        for entry in stale_removals_raw:
            if not isinstance(entry, dict):
                continue
            fact_id = entry.get("id")
            if not isinstance(fact_id, str) or not fact_id:
                continue
            reason = entry.get("reason", "")
            normalized_stale_removals.append(
                {
                    "id": fact_id,
                    "reason": reason if isinstance(reason, str) else "",
                }
            )

    # ── staleness 검토에 따른 수명 연장 요청 정규화 ──
    stale_extensions_raw = update_data.get("staleFactsToExtend")
    normalized_stale_extensions: list[dict[str, Any]] = []
    if isinstance(stale_extensions_raw, list):
        for entry in stale_extensions_raw:
            if not isinstance(entry, dict):
                continue
            fact_id = entry.get("id")
            if not isinstance(fact_id, str) or not fact_id:
                continue
            # extend_by_days: int/float만 받고(bool 거부) int로 변환한 뒤 0보다 큰 값만
            # 남긴다. (0, 1) 사이 소수는 0으로 변환되어 여기서 버려지므로, apply 경로가
            # 증분 0인 연장을 조용히 기록하는 일이 없다.
            raw_extend = entry.get("extend_by_days")
            if isinstance(raw_extend, (int, float)) and not isinstance(raw_extend, bool):
                extend_by = int(raw_extend)
                if extend_by > 0:
                    reason = entry.get("reason", "")
                    normalized_stale_extensions.append(
                        {
                            "id": fact_id,
                            "extend_by_days": extend_by,
                            "reason": reason if isinstance(reason, str) else "",
                        }
                    )

    # ── consolidation 결정 정규화 ──
    consolidation_raw = update_data.get("factsToConsolidate")
    normalized_consolidation: list[dict[str, Any]] = []
    if isinstance(consolidation_raw, list):
        for entry in consolidation_raw:
            if not isinstance(entry, dict):
                continue
            source_ids = entry.get("sourceIds")
            if not isinstance(source_ids, list) or not source_ids:
                continue
            # dict.fromkeys는 순서를 유지하며 중복을 제거하므로 ["f1","f1"]은 ["f1"]로
            # 줄고, source가 하나뿐인 merge로 올바르게 거부된다.
            clean_ids = list(dict.fromkeys(sid for sid in source_ids if isinstance(sid, str) and sid))
            if len(clean_ids) < 2:
                continue
            consolidated = entry.get("consolidated")
            if not isinstance(consolidated, dict):
                continue
            content = consolidated.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            # confidence 정규화: bool을 거부하고(bool은 int의 하위 클래스라 isinstance
            # 검사만으로는 True/False가 조용히 통과한다) float로 변환한 뒤 비유한 값을
            # 거부한다. _normalize_memory_update_fact와 동일한 규칙이다.
            _raw_conf = consolidated.get("confidence", 0.9)
            if isinstance(_raw_conf, bool) or not isinstance(_raw_conf, (int, float)):
                _norm_conf = 0.9
            else:
                _f = float(_raw_conf)
                _norm_conf = _f if math.isfinite(_f) else 0.9
            _raw_cat = consolidated.get("category")
            _norm_cat = _raw_cat.strip() if isinstance(_raw_cat, str) and _raw_cat.strip() else "context"
            normalized_consolidation.append(
                {
                    "sourceIds": clean_ids,
                    "consolidated": {
                        "content": content.strip(),
                        "category": _norm_cat,
                        "confidence": _norm_conf,
                        **{field: normalized for field in _FACT_CLASSIFICATION_FIELDS if (normalized := _normalize_gate_label(consolidated.get(field))) is not None},
                    },
                }
            )

    return {
        "user": user if isinstance(user, dict) else {},
        "history": history if isinstance(history, dict) else {},
        "newFacts": normalized_new_facts,
        "factsToRemove": normalized_facts_to_remove,
        "staleFactsToRemove": normalized_stale_removals,
        "staleFactsToExtend": normalized_stale_extensions,
        "factsToConsolidate": normalized_consolidation,
    }


def _parse_memory_update_response(response_content: Any) -> dict[str, Any]:
    """LLM 응답에서 유효한 첫 memory 갱신 JSON 객체를 파싱한다.

    일부 provider는 JSON만 반환하라고 지시해도 thinking trace, 산문, markdown fence로
    JSON을 감싼다. 이 파서는 안전하게 추출 가능한 JSON 객체만 받아들이며, 잘리거나
    형식이 깨진 JSON을 복구하지는 않는다.
    """
    response_text = _extract_text(response_content).strip()
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", response_text):
        try:
            parsed, _end = decoder.raw_decode(response_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _REQUIRED_MEMORY_UPDATE_TOP_LEVEL_KEYS.issubset(parsed):
            return _normalize_memory_update_data(parsed)

    raise json.JSONDecodeError("No valid memory update JSON object found", response_text, 0)


# 일반적인 파일 관련 작업이 아니라 파일 업로드 *이벤트*를 서술한 문장에만 매칭한다.
# "User works with CSV files"나 "prefers PDF export" 같은 정상 fact를 지우지 않도록
# 의도적으로 좁게 잡았다.
_UPLOAD_SENTENCE_RE = re.compile(
    r"[^.!?]*\b(?:"
    r"upload(?:ed|ing)?(?:\s+\w+){0,3}\s+(?:file|files?|document|documents?|attachment|attachments?)"
    r"|file\s+upload"
    r"|/mnt/user-data/uploads/"
    r"|<(?:uploaded_files|current_uploads)>"
    r")[^.!?]*[.!?]?\s*",
    re.IGNORECASE,
)


def _strip_upload_mentions_from_memory(memory_data: dict[str, Any]) -> dict[str, Any]:
    """모든 memory summary와 fact에서 파일 업로드 관련 문장을 제거한다.

    업로드된 파일은 session 범위다. 업로드 이벤트를 장기 memory에 남기면 에이전트가 이후
    session에서 존재하지 않는 파일을 찾게 된다.
    """
    # user/history 섹션의 summary를 정리한다.
    for section in ("user", "history"):
        section_data = memory_data.get(section, {})
        for _key, val in section_data.items():
            if isinstance(val, dict) and "summary" in val:
                cleaned = _UPLOAD_SENTENCE_RE.sub("", val["summary"]).strip()
                cleaned = re.sub(r"  +", " ", cleaned)
                val["summary"] = cleaned

    # 업로드 이벤트를 서술하는 fact도 함께 제거한다.
    facts = memory_data.get("facts", [])
    if facts:
        memory_data["facts"] = [f for f in facts if not _UPLOAD_SENTENCE_RE.search(f.get("content", ""))]

    return memory_data


def _fact_content_key(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    return stripped.casefold()


def _raise_if_duplicate_fact_content(memory_data: dict[str, Any], content_key: str | None) -> None:
    """정규화된 content가 이미 존재하는 후보 fact를 거부한다.

    호출자는 read-check-write 임계 구역 안에서 가장 최신 snapshot을 대상으로(즉 모든
    revision 충돌 재시도마다) 이 검사를 수행해야 한다. 그래야 같은 content를 동시에
    만들려는 두 주체가 둘 다 검사를 통과해 중복 fact를 저장하는 일이 없다."""
    if content_key is None:
        return
    for fact in memory_data.get("facts", []):
        if isinstance(fact, dict) and _fact_content_key(fact.get("content")) == content_key:
            raise ValueError("Duplicate fact")


# ── staleness 검토 헬퍼 ───────────────────────────────────────────────────


def _parse_fact_datetime(raw: str) -> datetime | None:
    """fact의 createdAt 필드에 있는 ISO-8601 datetime 문자열을 파싱한다.

    파싱에 실패하면 ``None``을 반환해, 호출자가 형식이 깨진 fact를 안전하게 건너뛰게 한다.
    """
    if not raw:
        return None
    try:
        result = datetime.fromisoformat(raw)
        # tzinfo 없는 naive datetime은 timezone 인식 cutoff와 비교할 때 TypeError를
        # 낸다. 안전하게 UTC로 가정한다.
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result
    except (ValueError, TypeError):
        return None


def _read_expected_valid_days(fact: dict[str, Any]) -> int | None:
    """fact의 ``expected_valid_days``를 양의 int로 반환하거나 ``None``을 반환한다.

    int/float만 받고(``int``의 하위 클래스인 ``bool``은 거부) 양수 검사 *전에* int로
    변환한다. 원래의 ``_normalize_memory_update_fact`` 규칙과 같다. 먼저 변환하는 것이
    (0, 1) 구간 값에서 중요하다. ``0.5``는 원본 그대로의 ``> 0`` 검사를 통과하지만 int로
    자르면 ``0``이 되어, 그대로면 ``None`` 대신 양수가 아닌 수명이 반환된다. 비유한
    float(``NaN``, ``+/-inf``)는 거부하고, 아주 큰 int는 ``float()``를 거치지 않고
    (``10**400``에서 ``OverflowError``가 난다) 그대로 반환한다. Python JSON decoder는
    소수점 없는 정수 리터럴을 임의 정밀도 ``int``로 파싱하므로, 손으로 편집한
    ``memory.json``이 그런 값을 담을 수 있다. ``datetime`` 연산에 쓰기엔 너무 큰 int는
    호출자가 :func:`_safe_add_days`로 제한한다. 이 헬퍼의 역할은 타입과 양수 여부 검증이지
    datetime 범위 검증이 아니다. 안전한 상한은 값 자체가 아니라 더해질 ``datetime``에
    달려 있기 때문이다. ``None``을 반환하면 호출자가 전역 age로 fallback하거나 필드를
    생략할 수 있어, 0/음수/비유한 수명이 조용히 기록되지 않는다.
    """
    raw = fact.get("expected_valid_days")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        evd = raw  # 임의 정밀도 int. 절대 float()를 거치지 않는다.
    elif isinstance(raw, float) and math.isfinite(raw):
        evd = int(raw)  # 양수 검사 전에 변환한다.
    else:
        return None
    return evd if evd > 0 else None


def _safe_add_days(dt: datetime, days: int) -> datetime | None:
    """``dt + timedelta(days=days)``를 반환하고, overflow하면 ``None``을 반환한다.

    저장된 ``expected_valid_days``가 아주 크면(예: ``10**12``) ``timedelta.max.days``를
    넘거나 결과가 ``datetime.max``를 초과하거나 ``datetime.min`` 아래로 내려가며, 모두
    ``OverflowError``를 낸다. staleness와 consolidation 경로는 재검토 기한을 구하려고
    evd를 ``datetime``에 더하는데, ``None``을 반환하면 호출자가 갱신 주기 전체를 중단하지
    않고 설정된 전역 수명으로 fallback할 수 있다.
    """
    try:
        return dt + timedelta(days=days)
    except (OverflowError, ValueError):
        return None


def _effective_fact_staleness_age(fact: dict[str, Any], config: Any) -> int:
    """*fact*의 실효 staleness 검토 주기를 일 단위로 반환한다.

    저장된 ``expected_valid_days``가 있고 유효하면 그 값을 그대로 쓴다.
    ``staleness_max_lifetime_multiplier`` 상한은 fact가 처음 생성되는 *기록 시점*에 한 번만
    적용되어 검토 주기를 처음부터 제한한다. 여기서 다시 적용하면 수명 연장 작업이 그 최초
    상한을 넘어 검토 주기를 옮길 수 없게 되어 ``staleFactsToExtend``의 목적이 사라진다.
    이 기능 도입 이전의 fact이거나 LLM이 추정치를 주지 않은 경우에는 전역
    ``staleness_age_days``로 fallback한다.
    """
    evd = _read_expected_valid_days(fact)
    return evd if evd is not None else config.staleness_age_days


def _select_stale_candidates(
    current_memory: dict[str, Any],
    config: Any,
) -> list[dict[str, Any]]:
    """각자의 검토 주기를 넘긴 fact들을 반환한다.

    fact별 실효 검토 주기는 ``_effective_fact_staleness_age``가 결정한다. LLM이 지정한
    ``expected_valid_days``가 있으면 그 값을, 없으면 전역 ``staleness_age_days``를 쓴다.
    보호 category(기본값 ``correction``)는 제외한다. 명시적인 사용자 피드백이라 나이만으로
    자동 정리해서는 안 되기 때문이다.
    """
    now = datetime.now(UTC)
    protected = frozenset(config.staleness_protected_categories)
    candidates: list[dict[str, Any]] = []
    for fact in current_memory.get("facts", []):
        if not isinstance(fact, dict):
            continue
        category = fact.get("category", "")
        if isinstance(category, str) and category in protected:
            continue
        created_at = _parse_fact_datetime(fact.get("createdAt", ""))
        if created_at is None:
            continue
        effective_age = _effective_fact_staleness_age(fact, config)
        # effective_age가 아주 큰 저장값이면 now - timedelta(days=effective_age)가
        # datetime.min을 넘어 overflow한다. 그렇게 큰 주기라면 fact가 아직 stale일 수
        # 없으므로 주기를 중단하지 말고 이 fact만 건너뛴다.
        cutoff = _safe_add_days(now, -effective_age)
        if cutoff is not None and created_at < cutoff:
            candidates.append(fact)
    return candidates


def _build_staleness_section(
    stale_candidates: list[dict[str, Any]],
    config: Any,
    *,
    prompts_dir: str | None = None,
    agent_name: str | None = None,
) -> str:
    """후보 fact들로 staleness 검토 prompt 섹션을 구성한다.

    각 fact 줄에는 그 fact의 실효 검토 주기를 나타내는 ``valid:Nd`` 표기가 붙는다. LLM이
    보수성을 조절하는 근거가 된다. 30일 만에 검토되는 fact는 생성 당시 변동성이 크다고
    판단된 것이고, 365일 만에 검토되는 fact는 안정적이라고 판단된 것이다.
    """
    if not stale_candidates:
        return ""
    lines: list[str] = []
    for fact in stale_candidates:
        fid = fact.get("id", "?")
        cat = html.escape(str(fact.get("category", "context")).strip() or "context", quote=False)
        conf = _coerce_source_confidence(fact)
        created_raw = fact.get("createdAt", "")
        created_short = created_raw[:10] if isinstance(created_raw, str) and len(created_raw) >= 10 else created_raw
        # quote=False: content는 element 텍스트 위치에 들어간다(<stale_facts> 태그 안이며
        # 속성값이 되는 일은 없다). 따라서 구조를 깨뜨릴 수 있는 문자는 <, >, & 뿐이므로
        # '와 "는 건드리지 않는다. prompt.py #4028의 관례와 동일하다.
        content = html.escape(str(fact.get("content", "")), quote=False)
        effective_age = _effective_fact_staleness_age(fact, config)
        lines.append(f'- [{fid} | {cat} | {conf:.2f} | {created_short} | valid:{effective_age}d] "{content}"')
    return load_prompt("staleness_review", prompts_dir=prompts_dir, agent_name=agent_name).format(stale_facts="\n".join(lines))


# ── consolidation 헬퍼 ──────────────────────────────────────────────────


def _select_consolidation_candidates(
    current_memory: dict[str, Any],
    config: Any,
) -> dict[str, list[dict[str, Any]]]:
    """파편화 임계값을 넘은 fact category들을 반환한다.

    fact를 category로 묶고, 항목이 ``consolidation_min_facts`` 이상인 category만 반환한다.
    ``staleness_protected_categories``에 속한 category는 제외한다. staleness 규약과 동일하게,
    명시적인 사용자 피드백이 merge 후보로 노출되지 않게 한다.
    """
    facts = current_memory.get("facts", [])
    if not facts:
        return {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        cat = fact.get("category", "context")
        if isinstance(cat, str) and cat.strip():
            by_category.setdefault(cat.strip(), []).append(fact)
    threshold = config.consolidation_min_facts
    protected = set(config.staleness_protected_categories)
    return {cat: group for cat, group in by_category.items() if len(group) >= threshold and cat not in protected}


def _build_consolidation_section(
    candidates: dict[str, list[dict[str, Any]]],
    max_groups: int = 3,
    max_sources: int = 8,
    *,
    prompts_dir: str | None = None,
    agent_name: str | None = None,
) -> str:
    """consolidation 후보 그룹을 prompt 섹션 형태로 만든다.

    category는 최대 ``max_groups``개(파편화가 심한 순), 그룹당 fact는 최대
    ``max_sources``개만 노출한다. apply 시점에 적용되는 상한과 같게 맞춰, LLM에게 실제로는
    처리할 수 없는 그룹을 보여주지 않는다.
    """
    if not candidates:
        return ""
    # 가장 파편화된 category를 우선한다. 동률은 안정성을 위해 사전순으로 가른다.
    sorted_candidates = sorted(candidates.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    parts: list[str] = []
    for cat, group in sorted_candidates[:max_groups]:
        lines: list[str] = []
        for fact in group[:max_sources]:
            fid = fact.get("id", "?")
            conf = _coerce_source_confidence(fact)
            content = html.escape(str(fact.get("content", "")))
            lines.append(f'- [{fid} | {conf:.2f}] "{content}"')
        shown = min(len(group), max_sources)
        parts.append(f'<consolidation_candidates category="{html.escape(cat)}" count="{shown}">\n' + "\n".join(lines) + "\n</consolidation_candidates>")
    return load_prompt("consolidation", prompts_dir=prompts_dir, agent_name=agent_name).format(consolidation_groups="\n\n".join(parts), max_groups=max_groups)


def _escape_memory_for_prompt(memory: Any) -> Any:
    """모든 문자열 leaf를 HTML 이스케이프한 ``memory`` 사본을 반환한다.

    memory_update prompt는 전체 memory state를 ``json.dumps`` blob으로
    ``<current_memory>...</current_memory>`` 블록 안에 넣는다. ``json.dumps``는 ``"``와
    ``\\``만 이스케이프하고 ``<``, ``>``, ``&``는 그대로 두므로, 사용자 영향을 받는 필드(예:
    fact ``content``가 ``</current_memory><evil>...``)가 모델에 그대로 전달되어 블록을
    탈출할 수 있다(prompt injection, #4044).

    직렬화된 blob이 아니라 각 문자열 *값*을 직렬화 전에 이스케이프하면 JSON 구조는 깨지지
    않는다. ``json.dumps``가 이미 안전해진 값을 다시 따옴표로 감싸기 때문이다. 알려진 필드만이
    아니라 모든 leaf를 이스케이프하면 현재든 앞으로든 사용자 영향을 받는 어떤 필드도 raw
    ``<``/``>``/``&``를 실을 수 없다. id나 timestamp 같은 통제된 필드에는 그런 문자가 없으므로
    이스케이프해도 무해한 no-op다. staleness/consolidation 섹션에 이미 적용된 ``html.escape``
    처리와 동일한 방식이다(#4028).
    """
    if isinstance(memory, str):
        return html.escape(memory)
    if isinstance(memory, dict):
        return {key: _escape_memory_for_prompt(value) for key, value in memory.items()}
    if isinstance(memory, list):
        return [_escape_memory_for_prompt(item) for item in memory]
    return memory


def _memory_with_manual_markers(memory: Any) -> Any:
    """수동 작성 fact(``source.type == "manual"``)의 content 앞에 ``[MANUAL]``을 붙인
    ``memory`` deep copy를 반환한다.

    이 표시는 prompt 전용 신호로, 해당 fact가 신뢰도 높은 사용자 편집임을 추출 LLM에게
    알린다. 저장된 memory는 건드리지 않는다(이 사본은 prompt에만 들어간다). 멱등이므로 이미
    접두사가 붙은 fact에 다시 표시하지 않는다.
    """
    display = copy.deepcopy(memory)
    if not isinstance(display, dict):
        return display
    for fact in display.get("facts", []):
        if not isinstance(fact, dict):
            continue
        src = fact.get("source")
        src_type = src.get("type") if isinstance(src, dict) else None
        if src_type == "manual":
            content = fact.get("content")
            if isinstance(content, str) and not content.startswith("[MANUAL]"):
                fact["content"] = "[MANUAL] " + content
    return display


def _message_identity(msg: Any) -> tuple[str, ...] | None:
    """watermark 추적용으로 ``msg``의 해시 가능한 identity를 반환한다.

    watermark는 index가 아니라 content/identity 기반이므로, summarization이 대화 앞부분을
    지워도 유효하다(index 기반이면 앞부분 제거 후 엉뚱한 메시지를 가리켜 아직 추출되지 않은
    turn을 조용히 건너뛴다). langgraph 메시지 ``id``를 우선 쓰고(고유하며 중복 content에도
    강하다), id가 없으면 ``(type, content)``로 fallback한다(예: 테스트의 순수
    ``HumanMessage(content=...)``). id도 없고 추출 가능한 텍스트도 없으면 ``None``을 반환한다.
    그러면 호출자가 전체 리스트를 넘기는데, 이는 안전한 과다 추출이며 손실은 없다.
    """
    mid = getattr(msg, "id", None)
    if isinstance(mid, str) and mid:
        return ("id", mid)
    text = extract_message_text(msg)
    if not text:
        return None
    msg_type = getattr(msg, "type", "") or ""
    return ("content", msg_type, text)


class MemoryUpdater:
    """대화 context를 바탕으로 LLM을 써서 memory를 갱신한다."""

    def __init__(self, config: DeerMemConfig, storage: MemoryStorage, llm: Any = None, *, prompts_dir: str | None = None, callbacks: Any = None):
        """주입된 config + storage + llm(DI)으로 memory updater를 초기화한다.

        Args:
            config: DeerMem 내부 설정.
            storage: memory storage 인스턴스(DeerMem이 소유하고 여기로 주입한다).
            llm: memory 추출용 chat model(DeerMem이 소유하고 여기로 주입한다). LLM이 설정되지
                않았으면 None이며, 그 경우 갱신 시 예외가 난다.
            prompts_dir: ``load_prompt`` / ``load_prompt_messages``로 전달되는 선택적 커스텀
                prompt 템플릿 디렉터리. None이면 번들 기본값을 쓴다.
            callbacks: 선택적 ``MemoryCallbacks``(DeerMem이 소유하고 여기로 주입한다).
                ``on_memory_llm_call``은 LLM 호출 전에 trace 메타데이터를 ``invoke_config``에
                병합하려고 호출된다. None이면 tracing하지 않는다.
        """
        self._config = config
        self._storage = storage
        self._llm = llm
        self._prompts_dir = prompts_dir
        self._callbacks = callbacks
        # Watermark: (thread_id, user_id, agent_name)별 마지막 추출 메시지 identity.
        # 메모리에만 두므로 재시작하면 한 batch를 다시 추출한다. 캐시는 크기 제한 LRU
        # (config.watermark_max_keys)라 thread를 많이 다루는 장수 gateway에서도 무한히 커지지
        # 않는다. 밀려난 키는 해당 thread의 다음 turn에서 한 batch를 다시 추출한다.
        self._watermarks: OrderedDict[tuple[str | None, str | None, str | None], tuple[str, ...] | None] = OrderedDict()

    # ── 데이터 접근 + fact CRUD (이전 모듈 레벨 함수. self._storage를 쓴다) ──

    def _save_memory_to_file(
        self,
        memory_data: dict[str, Any],
        agent_name: str | None = None,
        *,
        user_id: str | None = None,
        expected_revision: int | None = None,
    ) -> bool:
        """주입된 storage를 통해 memory 데이터를 저장한다."""
        kwargs: dict[str, Any] = {"user_id": user_id}
        if expected_revision is not None:
            kwargs["expected_revision"] = expected_revision
        return self._storage.save(memory_data, agent_name, **kwargs)

    def get_memory_data(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """주입된 storage를 통해 현재 memory 데이터를 가져온다."""
        return self._storage.load(agent_name, user_id=user_id)

    def reload_memory_data(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """주입된 storage를 통해 memory 데이터를 다시 읽는다."""
        return self._storage.reload(agent_name, user_id=user_id)

    def import_memory_data(self, memory_data: dict[str, Any], agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """주입된 storage를 통해 import된 memory 데이터를 저장한다."""
        if not isinstance(memory_data, dict):
            raise ValueError("memory_data")
        memory_data = copy.deepcopy(memory_data)
        empty = create_empty_memory()
        for section in ("user", "history"):
            incoming_section = memory_data.get(section, {})
            if not isinstance(incoming_section, dict):
                raise ValueError(f"memory_data.{section}")
            complete_section = copy.deepcopy(empty[section])
            for key, value in incoming_section.items():
                if key in complete_section and isinstance(complete_section[key], dict) and isinstance(value, dict):
                    complete_section[key].update(copy.deepcopy(value))
                else:
                    complete_section[key] = copy.deepcopy(value)
            memory_data[section] = complete_section
        if agent_name is not None and getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            current = self.get_memory_data(agent_name, user_id=user_id)
            incoming_facts = copy.deepcopy(memory_data.get("facts", []))
            if not isinstance(incoming_facts, list) or any(not isinstance(fact, dict) for fact in incoming_facts):
                raise ValueError("memory_data.facts")
            for fact in incoming_facts:
                fact["id"] = str(fact.get("id") or f"fact_{uuid.uuid4().hex}")
                fact["confidence"] = _coerce_source_confidence(fact)
            current_by_id = {str(fact.get("id")): fact for fact in current.get("facts", []) if isinstance(fact, dict)}
            incoming_ids = {str(fact.get("id")) for fact in incoming_facts}
            self._storage.apply_changes(
                {
                    "upserts": incoming_facts,
                    "upsertRevisions": {str(fact.get("id")): (int(current_by_id[str(fact.get("id"))].get("revision") or 1) if str(fact.get("id")) in current_by_id else None) for fact in incoming_facts},
                    "deletes": [fact_id for fact_id in current_by_id if fact_id not in incoming_ids],
                    "deleteRevisions": {fact_id: int(fact.get("revision") or 1) for fact_id, fact in current_by_id.items() if fact_id not in incoming_ids},
                    "summaries": {"user": copy.deepcopy(memory_data.get("user", {})), "history": copy.deepcopy(memory_data.get("history", {}))},
                },
                agent_name=agent_name,
                user_id=user_id,
                expected_manifest_revision=int(current.get("revision") or 0),
            )
            return self._storage.load(agent_name, user_id=user_id)
        if agent_name is None:
            memory_data["facts"] = []
        if not self._storage.save(memory_data, agent_name, user_id=user_id):
            raise OSError("Failed to save imported memory data")
        return self._storage.load(agent_name, user_id=user_id)

    def clear_memory_data(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """공유 summary는 유지한 채 선택한 agent 하나의 fact만 비운다."""
        if agent_name is not None and getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            for attempt in range(3):
                current = self.get_memory_data(agent_name, user_id=user_id) if attempt == 0 else self.reload_memory_data(agent_name, user_id=user_id)
                facts = [fact for fact in current.get("facts", []) if isinstance(fact, dict)]
                try:
                    self._storage.apply_changes(
                        {
                            "deletes": [str(fact.get("id")) for fact in facts],
                            "deleteRevisions": {str(fact.get("id")): int(fact.get("revision") or 1) for fact in facts},
                        },
                        agent_name=agent_name,
                        user_id=user_id,
                        expected_manifest_revision=int(current.get("revision") or 0),
                    )
                    return self.reload_memory_data(agent_name, user_id=user_id)
                except MemoryManifestRevisionConflict:
                    if attempt == 2:
                        raise
                    logger.info("Retrying scoped memory clear from a fresh snapshot after a revision conflict")
            raise AssertionError("bounded scoped-clear retry did not return or raise")
        current = self.get_memory_data(agent_name, user_id=user_id)
        cleared_memory = copy.deepcopy(current)
        cleared_memory["facts"] = []
        if not self._save_memory_to_file(cleared_memory, agent_name, user_id=user_id, expected_revision=int(current.get("revision") or 0)):
            raise OSError("Failed to save cleared memory data")
        return cleared_memory

    def clear_all_memory_data(self, *, user_id: str | None = None) -> dict[str, Any]:
        """한 사용자의 전역 summary와 모든 agent fact 버킷을 비운다."""
        if getattr(type(self._storage), "clear_all", None) is not MemoryStorage.clear_all:
            return self._storage.clear_all(user_id=user_id)
        current = self.get_memory_data(user_id=user_id)
        cleared_memory = create_empty_memory()
        if not self._save_memory_to_file(
            cleared_memory,
            user_id=user_id,
            expected_revision=int(current.get("revision") or 0),
        ):
            raise OSError("Failed to save cleared memory data")
        return cleared_memory

    def create_memory_fact(self, content: str, category: str = "context", confidence: float = 0.5, agent_name: str | None = None, *, user_id: str | None = None) -> tuple[dict[str, Any], str | None]:
        """새 fact를 만들어 저장하고 ``(updated_memory, fact_id)``를 반환한다.

        fact_id를 직접 반환하므로 호출자(예: memory_add 도구)가 content 매칭으로 memory
        데이터에서 다시 유도할 필요가 없다. 그렇게 하면 backend의 content 정규화 방식에
        결합되고, 정규화가 다른 backend에서는 저장 상한을 잘못 보고할 수 있다.

        새 fact는 이후 :func:`_trim_facts_to_max`로 잘린다(confidence가 높은 쪽이 남고,
        confidence는 강제 변환된다). 상한 때문에 방금 추가한(confidence가 낮은) fact가 밀려나면
        ``fact_id``는 ``None``이 되어, 호출자가 잘못된 "added" 상태에 매달린 id를 붙이는 대신
        "not stored - cap reached"를 보고한다. 이는 vendored 사본이 매달린 id를 피하려고 함께
        빼버렸던 max_facts 상한과 trim 이후 존재 확인(upstream의
        ``create_memory_fact_with_created_fact``)을 다시 살린 것이다.

        중복 거부도 호출자만이 아니라 여기서 강제한다. 두 storage 경로(apply_changes와 legacy
        단일 파일 save) 모두 revision 충돌 재시도 루프 안에서 후보의 정규화된 content 키를 최신
        memory snapshot과 대조하므로, 동시에 생성하는 두 주체가 같은 content를 둘 다 저장할 수
        없다. 정규화된 content가 일치하면 ``ValueError("Duplicate fact")``를 낸다.
        """
        if agent_name is None:
            raise ValueError("agent_name")
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("content")
        normalized_category = category.strip() or "context"
        validated_confidence = _validate_confidence(confidence)
        candidate_key = _fact_content_key(normalized_content)
        now = utc_now_iso_z()
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        candidate = {
            "id": fact_id,
            "content": normalized_content,
            "category": normalized_category,
            "confidence": validated_confidence,
            "createdAt": now,
            "source": "manual",
        }
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            for attempt in range(3):
                memory_data = self.get_memory_data(agent_name, user_id=user_id) if attempt == 0 else self.reload_memory_data(agent_name, user_id=user_id)
                # 중복 거부는 충돌 재시도 루프 안에 두어, revision 충돌이 날 때마다 최신
                # snapshot을 기준으로 다시 평가되게 한다. 같은 content를 만드는 두 주체가 둘
                # 다 저장할 수는 없다(진 쪽은 다시 읽어 이긴 쪽의 fact를 보고 여기서
                # 거부된다).
                _raise_if_duplicate_fact_content(memory_data, candidate_key)
                updated_memory = dict(memory_data)
                updated_memory["facts"] = _trim_facts_to_max([*memory_data.get("facts", []), copy.deepcopy(candidate)], self._config.max_facts)
                kept_ids = {str(fact.get("id")) for fact in updated_memory["facts"]}
                deletions = [str(fact.get("id")) for fact in memory_data.get("facts", []) if str(fact.get("id")) not in kept_ids]
                try:
                    self._storage.apply_changes(
                        {
                            "upserts": [fact for fact in updated_memory["facts"] if fact.get("id") == fact_id],
                            "upsertRevisions": {fact_id: None},
                            "deletes": deletions,
                            "deleteRevisions": {str(fact.get("id")): int(fact.get("revision") or 1) for fact in memory_data.get("facts", []) if str(fact.get("id")) in deletions},
                        },
                        agent_name=agent_name,
                        user_id=user_id,
                        expected_manifest_revision=int(memory_data.get("revision") or 0),
                    )
                    fresh_memory = self.reload_memory_data(agent_name, user_id=user_id)
                    stored = any(fact.get("id") == fact_id for fact in fresh_memory.get("facts", []))
                    return fresh_memory, (fact_id if stored else None)
                except MemoryManifestRevisionConflict:
                    if attempt == 2:
                        raise
                    logger.info("Retrying capped fact creation from a fresh snapshot after a revision conflict")
            raise AssertionError("bounded create retry did not return or raise")
        # legacy 단일 파일 경로. 위 apply_changes 경로와 동일한 중복 거부 계약을 지킨다.
        # revision 충돌로 save가 False를 반환하면 최신 snapshot을 다시 읽고 중복 검사를 재실행하므로,
        # 동시 생성자의 commit은 일반적인 저장 실패가 아니라 ValueError("Duplicate fact")로
        # 거부된다.
        for attempt in range(3):
            memory_data = self.get_memory_data(agent_name, user_id=user_id) if attempt == 0 else self.reload_memory_data(agent_name, user_id=user_id)
            _raise_if_duplicate_fact_content(memory_data, candidate_key)
            updated_memory = dict(memory_data)
            updated_memory["facts"] = _trim_facts_to_max([*memory_data.get("facts", []), copy.deepcopy(candidate)], self._config.max_facts)
            if self._save_memory_to_file(updated_memory, agent_name, user_id=user_id, expected_revision=int(memory_data.get("revision") or 0)):
                # 상한 때문에 방금 추가한(confidence가 낮은) fact가 밀려났다면 None으로
                # 알려서, 호출자가 매달린 id를 "added"로 보고하지 않게 한다.
                stored = any(f.get("id") == fact_id for f in updated_memory["facts"])
                return updated_memory, (fact_id if stored else None)
            logger.info("Retrying capped fact creation from a fresh snapshot after a revision conflict")
        raise OSError("Failed to save memory data after creating fact")

    def delete_memory_fact(self, fact_id: str, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """id로 fact를 삭제하고 갱신된 memory 데이터를 저장한다."""
        if agent_name is None:
            raise ValueError("agent_name")
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes and hasattr(self._storage, "get_fact"):
            deleted = self._storage.get_fact(fact_id, agent_name=agent_name, user_id=user_id)
            if deleted is None:
                raise KeyError(fact_id)
            global_memory = self.get_memory_data(user_id=user_id)
            self._storage.apply_changes(
                {"deletes": [fact_id], "deleteRevisions": {fact_id: int(deleted.get("revision") or 1)}},
                agent_name=agent_name,
                user_id=user_id,
                expected_manifest_revision=int(global_memory.get("revision") or 0),
                allow_manifest_rebase=True,
            )
            return self.get_memory_data(agent_name, user_id=user_id)
        memory_data = self.get_memory_data(agent_name, user_id=user_id)
        facts = memory_data.get("facts", [])
        updated_facts = [fact for fact in facts if fact.get("id") != fact_id]
        if len(updated_facts) == len(facts):
            raise KeyError(fact_id)
        deleted = next(fact for fact in facts if fact.get("id") == fact_id)
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            self._storage.apply_changes(
                {"deletes": [fact_id], "deleteRevisions": {fact_id: int(deleted.get("revision") or 1)}},
                agent_name=agent_name,
                user_id=user_id,
                expected_manifest_revision=int(memory_data.get("revision") or 0),
                allow_manifest_rebase=True,
            )
            return self.get_memory_data(agent_name, user_id=user_id)
        updated_memory = dict(memory_data)
        updated_memory["facts"] = updated_facts
        if not self._save_memory_to_file(updated_memory, agent_name, user_id=user_id, expected_revision=int(memory_data.get("revision") or 0)):
            raise OSError(f"Failed to save memory data after deleting fact '{fact_id}'")
        return updated_memory

    def update_memory_fact(self, fact_id: str, content: str | None = None, category: str | None = None, confidence: float | None = None, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """기존 fact를 수정하고 갱신된 memory 데이터를 저장한다."""
        if agent_name is None:
            raise ValueError("agent_name")
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes and hasattr(self._storage, "get_fact"):
            updated_fact = self._storage.get_fact(fact_id, agent_name=agent_name, user_id=user_id)
            if updated_fact is None:
                raise KeyError(fact_id)
            if content is not None:
                normalized_content = content.strip()
                if not normalized_content:
                    raise ValueError("content")
                updated_fact["content"] = normalized_content
            if category is not None:
                updated_fact["category"] = category.strip() or "context"
            if confidence is not None:
                updated_fact["confidence"] = _validate_confidence(confidence)
            global_memory = self.get_memory_data(user_id=user_id)
            self._storage.apply_changes(
                {"upserts": [updated_fact], "upsertRevisions": {fact_id: int(updated_fact.get("revision") or 1)}},
                agent_name=agent_name,
                user_id=user_id,
                expected_manifest_revision=int(global_memory.get("revision") or 0),
                allow_manifest_rebase=True,
            )
            return self.get_memory_data(agent_name, user_id=user_id)
        memory_data = self.get_memory_data(agent_name, user_id=user_id)
        updated_memory = dict(memory_data)
        updated_facts: list[dict[str, Any]] = []
        found = False
        for fact in memory_data.get("facts", []):
            if fact.get("id") == fact_id:
                found = True
                updated_fact = dict(fact)
                if content is not None:
                    normalized_content = content.strip()
                    if not normalized_content:
                        raise ValueError("content")
                    updated_fact["content"] = normalized_content
                if category is not None:
                    updated_fact["category"] = category.strip() or "context"
                if confidence is not None:
                    updated_fact["confidence"] = _validate_confidence(confidence)
                updated_facts.append(updated_fact)
            else:
                updated_facts.append(fact)
        if not found:
            raise KeyError(fact_id)
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            changed = next(fact for fact in updated_facts if fact.get("id") == fact_id)
            self._storage.apply_changes(
                {"upserts": [changed], "upsertRevisions": {fact_id: int(changed.get("revision") or 1)}},
                agent_name=agent_name,
                user_id=user_id,
                expected_manifest_revision=int(memory_data.get("revision") or 0),
                allow_manifest_rebase=True,
            )
            return self.get_memory_data(agent_name, user_id=user_id)
        updated_memory["facts"] = updated_facts
        if not self._save_memory_to_file(updated_memory, agent_name, user_id=user_id, expected_revision=int(memory_data.get("revision") or 0)):
            raise OSError(f"Failed to save memory data after updating fact '{fact_id}'")
        return updated_memory

    def _build_signal_hints(self, signals: frozenset[str]) -> str:
        """탐지된 signal 클래스에 대한 선택적 prompt hint를 만든다.

        존재하는 signal마다 추출 LLM이 올바른 category와 confidence를 고르도록 유도하는 지시문을
        하나씩 더한다. 이 변수는 여전히 템플릿의 ``{correction_hint}`` 자리에 렌더링된다(이름은
        과거의 잔재다. 지금은 전체 signal hint 집합과 :meth:`_prepare_update_prompt`가 덧붙이는
        manual fact 안내까지 담는다).
        """
        hints: list[str] = []
        if "correction" in signals:
            hints.append(
                "IMPORTANT: Explicit correction signals were detected in this conversation. "
                "Record a correction with confidence >= 0.95 only when it describes a durable, user-level "
                "working preference that is safe to reuse across unrelated tasks. A correction to facts, files, "
                "directions, or constraints in the current task is thread- or project-scoped and must not be stored."
            )
        if "reinforcement" in signals:
            hints.append(
                "IMPORTANT: Positive reinforcement signals were detected in this conversation. "
                "Record the confirmed approach, style, or preference with high confidence only if it is a durable, "
                "user-level pattern. Approval of the current result or current task is thread-scoped and must not be stored."
            )
        if "preference" in signals:
            hints.append("IMPORTANT: A preference signal was detected. Record it with high confidence only when it is a durable, user-level preference; a one-off choice for the current task is thread-scoped and must not be stored.")
        if "identity" in signals:
            hints.append("IMPORTANT: An identity signal was detected. Record the user's stated role, profession, or background only when it is user-level and durable across tasks.")
        if "goal" in signals:
            hints.append("IMPORTANT: A goal signal was detected. Record only a durable, user-level goal that remains useful across unrelated tasks; the objective of the current task, sprint, PR, or thread must not be stored.")
        if "decision" in signals:
            hints.append("IMPORTANT: A decision signal was detected. Record only a durable, user-level decision or working pattern; a choice made for the current task, file, PR, or thread must not be stored.")
        return "\n".join(hints)

    def _prepare_update_prompt(
        self,
        messages: list[Any],
        agent_name: str | None,
        signals: frozenset[str],
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], list[Any]] | None:
        """memory를 읽어 대화에 대한 갱신 prompt를 구성한다."""
        config = self._config
        if not messages:
            return None

        current_memory = self.get_memory_data(agent_name, user_id=user_id)
        conversation_text = format_conversation_for_update(messages)
        if not conversation_text.strip():
            return None

        correction_hint = self._build_signal_hints(signals)

        # manual fact signal: prompt의 current_memory에서 신뢰도 높은 사용자 작성 fact에
        # [MANUAL] 접두사를 붙이고, 새 대화가 명시적이고 분명한 정정이 아닌 한 그대로
        # 보존하도록 모델에 지시한다.
        display_memory = current_memory
        if self._has_manual_facts(current_memory):
            display_memory = _memory_with_manual_markers(current_memory)
            manual_hint = "NOTE: Facts marked [MANUAL] are high-trust user-authored edits. Update them only when the new conversation is an explicit, unambiguous correction; otherwise preserve them as-is."
            correction_hint = (correction_hint + "\n" + manual_hint).strip() if correction_hint else manual_hint

        # ── staleness 검토 섹션 구성 ──
        staleness_section = ""
        if config.staleness_review_enabled:
            stale_candidates = _select_stale_candidates(current_memory, config)
            if len(stale_candidates) >= config.staleness_min_candidates:
                staleness_section = _build_staleness_section(stale_candidates, config, prompts_dir=self._prompts_dir, agent_name=agent_name)

        # ── consolidation 섹션 구성 ──
        consolidation_section = ""
        if config.consolidation_enabled:
            consolidation_candidates = _select_consolidation_candidates(current_memory, config)
            if consolidation_candidates:
                consolidation_section = _build_consolidation_section(
                    consolidation_candidates,
                    max_groups=config.consolidation_max_groups_per_cycle,
                    max_sources=config.consolidation_max_sources,
                    prompts_dir=self._prompts_dir,
                    agent_name=agent_name,
                )

        variables = {
            "current_memory": json.dumps(_escape_memory_for_prompt(display_memory), indent=2, ensure_ascii=False),
            "conversation": conversation_text,
            "correction_hint": correction_hint,
            "staleness_review_section": staleness_section,
            "consolidation_section": consolidation_section,
        }
        prompt = load_prompt_messages("memory_update", variables, agent_name=agent_name, prompts_dir=self._prompts_dir)
        return current_memory, prompt

    def _has_manual_facts(self, memory: dict[str, Any]) -> bool:
        """``memory``에 사용자 작성(manual) fact가 하나라도 있는지 반환한다."""
        return any(isinstance(f, dict) and isinstance(f.get("source"), dict) and f.get("source", {}).get("type") == "manual" for f in memory.get("facts", []))

    def _emit_extraction_metrics(
        self,
        metrics: dict[str, Any],
        *,
        thread_id: str | None,
        user_id: str | None,
        trace_id: str | None,
        model_name: str | None,
        response: Any,
        success: bool,
    ) -> None:
        """추출 이후 observability callback(Langfuse span 등)을 호출한다.

        ``extraction_callback``이 설정되지 않았으면(기본값) no-op다. callback에서 나온 예외는
        로그만 남기고 삼켜서, observability가 갱신 경로를 깨뜨리지 않게 한다.
        """
        callback = self._config.extraction_callback
        if callback is None:
            return
        usage = getattr(response, "usage_metadata", None)
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "trace_id": trace_id,
            "model_name": model_name,
            "success": success,
            "token_usage": usage if isinstance(usage, dict) else None,
        }
        payload.update(metrics)
        try:
            callback(payload)
        except Exception:
            logger.warning("extraction_callback raised; ignoring", exc_info=True)

    def _finalize_update(
        self,
        current_memory: dict[str, Any],
        response_content: Any,
        thread_id: str | None,
        agent_name: str | None,
        user_id: str | None = None,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """모델 응답을 파싱해 갱신을 적용하고 memory를 저장한다."""
        update_data = _parse_memory_update_response(response_content)
        if metrics is not None:
            extracted = update_data.get("newFacts", [])
            extracted_list = extracted if isinstance(extracted, list) else []
            metrics["facts_extracted"] = len(extracted_list)
            # facts_passed_confidence / rejected_low_confidence는 실제 confidence 필터가
            # 있는 _apply_updates 안에서 채운다. 여기서 다시 유도한 값이 아니라 실제 필터를
            # 추적하기 위해서다.
        if getattr(type(self._storage), "apply_changes", None) is not MemoryStorage.apply_changes:
            for attempt in range(3):
                # 제자리 변경 전에 deep copy해서, commit이 실패해도 캐시된 snapshot이
                # 손상되지 않게 한다. manifest 충돌 시에는 추출 결과 전체를 새 문서에 다시
                # 적용한다. trim/consolidation/delete 결정은 snapshot 전체를 대상으로 하므로
                # 서로 떨어진 point write로 재생해서는 절대 안 된다.
                updated_memory = self._apply_updates(copy.deepcopy(current_memory), update_data, thread_id, metrics=metrics)
                updated_memory = _strip_upload_mentions_from_memory(updated_memory)
                current_by_id = {str(fact.get("id")): fact for fact in current_memory.get("facts", [])}
                updated_by_id = {str(fact.get("id")): fact for fact in updated_memory.get("facts", [])}
                change_set = {
                    "upserts": [copy.deepcopy(fact) for fact_id, fact in updated_by_id.items() if current_by_id.get(fact_id) != fact],
                    "upsertRevisions": {fact_id: (int(current_by_id[fact_id].get("revision") or 1) if fact_id in current_by_id else None) for fact_id, fact in updated_by_id.items() if current_by_id.get(fact_id) != fact},
                    "deletes": [fact_id for fact_id in current_by_id if fact_id not in updated_by_id],
                    "deleteRevisions": {fact_id: int(current_by_id[fact_id].get("revision") or 1) for fact_id in current_by_id if fact_id not in updated_by_id},
                }
                summaries_changed = updated_memory.get("user", {}) != current_memory.get("user", {}) or updated_memory.get("history", {}) != current_memory.get("history", {})
                if summaries_changed:
                    change_set["summaries"] = {
                        "user": copy.deepcopy(updated_memory.get("user", {})),
                        "history": copy.deepcopy(updated_memory.get("history", {})),
                    }
                try:
                    self._storage.apply_changes(
                        change_set,
                        agent_name=agent_name,
                        user_id=user_id,
                        expected_manifest_revision=int(current_memory.get("revision") or 0),
                    )
                    return True
                except MemoryManifestRevisionConflict:
                    if attempt == 2:
                        raise
                    current_memory = self.reload_memory_data(agent_name, user_id=user_id)
                    logger.info("Retrying extracted memory update from a fresh snapshot after a revision conflict")
            raise AssertionError("bounded extracted-update retry did not return or raise")
        # 제자리 변경 전에 deep copy해서, 이후 save() 실패가 아직 캐시된 원본 객체 참조를
        # 손상시키지 않게 한다.
        updated_memory = self._apply_updates(copy.deepcopy(current_memory), update_data, thread_id, metrics=metrics)
        updated_memory = _strip_upload_mentions_from_memory(updated_memory)
        return self._storage.save(
            updated_memory,
            agent_name,
            user_id=user_id,
            expected_revision=int(current_memory.get("revision") or 0),
        )

    async def aupdate_memory(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        signals: frozenset[str] = frozenset(),
        user_id: str | None = None,
        trace_id: str | None = None,
        *,
        bypass_watermark: bool = False,
    ) -> bool:
        """동기 경로에 위임해 memory를 비동기로 갱신한다.

        ``asyncio.to_thread``로 *동기* ``model.invoke()`` 경로를 worker thread에서 실행하므로
        두 번째 event loop가 만들어지지 않고, (lead agent와 공유하는) langchain async httpx
        client pool도 건드리지 않는다. issue #2615에서 설명한 loop 간 connection 재사용 버그를
        없앤다.
        """
        return await asyncio.to_thread(
            self._do_update_memory_sync,
            messages=messages,
            thread_id=thread_id,
            agent_name=agent_name,
            signals=signals,
            user_id=user_id,
            trace_id=trace_id,
            bypass_watermark=bypass_watermark,
        )

    def _do_update_memory_sync(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        signals: frozenset[str] = frozenset(),
        user_id: str | None = None,
        trace_id: str | None = None,
        *,
        bypass_watermark: bool = False,
    ) -> bool:
        """순수 동기 memory 갱신. worker thread의 request-trace ContextVar에 ``trace_id``를
        바인딩한 뒤 구현부에 위임한다.

        갱신은 request ContextVar를 상속받지 않는 Timer / executor thread에서 실행되므로,
        그대로 두면 여기서 남는 로그 레코드가 request trace id를 잃는다(이전에는 LLM 호출 직전
        tracing hook까지만 전달됐다). host가 주입하는 ``trace_context_manager`` hook(deer-flow
        factory 밖에서 DeerMem을 단독 실행하면 ``None``)이 호출 동안 ``trace_id``를 바인딩하고
        종료 시 이전 바인딩을 복원한다. trace_id가 ``None``이면 ContextVar를 건드리지 않는다
        (id를 꾸며내지 않는다).
        """
        cm = self._config.trace_context_manager
        if cm is not None and trace_id is not None:
            with cm(trace_id):
                return self._do_update_memory_sync_impl(
                    messages=messages,
                    thread_id=thread_id,
                    agent_name=agent_name,
                    signals=signals,
                    user_id=user_id,
                    trace_id=trace_id,
                    bypass_watermark=bypass_watermark,
                )
        return self._do_update_memory_sync_impl(
            messages=messages,
            thread_id=thread_id,
            agent_name=agent_name,
            signals=signals,
            user_id=user_id,
            trace_id=trace_id,
            bypass_watermark=bypass_watermark,
        )

    def _watermark_get(self, key: tuple[str | None, str | None, str | None]) -> tuple[str, ...] | None:
        """``key``의 watermark를 반환하고 가장 최근 사용으로 표시한다.

        값의 truthiness가 아니라 키 존재 여부로 판단하므로, 저장된 ``None`` identity도 LRU
        순서상 살아 있는 항목으로 취급된다.
        """
        if key not in self._watermarks:
            return None
        self._watermarks.move_to_end(key)
        return self._watermarks[key]

    def _watermark_set(
        self,
        key: tuple[str | None, str | None, str | None],
        value: tuple[str, ...] | None,
    ) -> None:
        """``key``에 ``value``를 저장하고, 크기 제한 LRU 캐시가
        ``config.watermark_max_keys``를 넘으면 가장 오래 안 쓴 항목을 밀어낸다.

        키가 밀려나도 안전하다. 해당 thread의 다음 turn에서 watermark를 찾지 못해 한 batch를
        다시 추출할 뿐이다(문서화된 재시작 동작과 같다). ``0``이면 무제한(축출 없음)이다.
        """
        self._watermarks[key] = value
        self._watermarks.move_to_end(key)
        cap = self._config.watermark_max_keys
        if cap > 0 and len(self._watermarks) > cap:
            self._watermarks.popitem(last=False)

    def _feed_after_watermark(
        self,
        watermark_key: tuple[str | None, str | None, str | None],
        messages: list[Any],
    ) -> list[Any]:
        """``messages`` 중 아직 추출되지 않은 구간을 반환한다.

        watermark는 마지막으로 추출한 메시지의 identity를 저장한다(:func:`_message_identity`
        참고). 그 메시지가 아직 남아 있으면 그 *뒤*를 전부 넘기고, 없으면(summarization이
        앞부분을 지웠거나 이 키의 첫 추출이면) 전체 리스트를 넘긴다. 다시 넘기는 것은 안전한
        과다 추출이다. turn을 건너뛰지 않으며, fact를 잃는 실패 방향은 그것뿐이다.
        """
        last_id = self._watermark_get(watermark_key)
        if last_id is None:
            return messages
        for i, msg in enumerate(messages):
            if _message_identity(msg) == last_id:
                return messages[i + 1 :]
        return messages

    def _do_update_memory_sync_impl(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        signals: frozenset[str] = frozenset(),
        user_id: str | None = None,
        trace_id: str | None = None,
        *,
        bypass_watermark: bool = False,
    ) -> bool:
        """``model.invoke()``를 쓰는 순수 동기 memory 갱신.

        *동기* LLM 호출 경로를 쓰므로 event loop가 만들어지지 않는다. 덕분에 langchain
        provider가 전역 캐시하는 async httpx ``AsyncClient`` / connection pool(lead agent와
        공유하는 것)을 절대 건드리지 않고, loop 간 connection 재사용도 불가능하다.

        watermark: middleware가 매 turn 전체 대화를 넘기므로, 이미 추출한 turn을 건너뛰지
        않으면 갱신마다 옛 메시지를 다시 넘기게 된다. watermark는 마지막으로 추출한 메시지의
        identity를 저장하므로(content/id 기반, 메모리 전용) summarization이 대화 앞부분을 지워도
        정확하다. 재시작하면 사라지고 한 batch를 다시 추출한다. ``bypass_watermark``는 긴급
        (summarization) flush 경로가 설정한다. 그 경로가 넘기는 부분집합은 "제거 전에 추출"하는
        일회성 snapshot이므로 전체를 그대로 넘기고 대화 watermark를 읽지도 전진시키지도 않는다
        (부분집합 자체의 길이로 전진시키면 watermark가 뒤로 밀려, 다음 정상 feed에서 아직
        추출되지 않은 뒷부분 turn을 건너뛴다).
        """
        metrics: dict[str, Any] = {}
        response: Any = None
        model_name: str | None = None
        success = False
        attempted = False
        try:
            watermark_key = (thread_id, user_id, agent_name)
            if bypass_watermark:
                # 긴급 flush: 넘어온 부분집합을 통째로 추출한다.
                feed_messages = messages
            else:
                feed_messages = self._feed_after_watermark(watermark_key, messages)
            if not feed_messages:
                logger.debug("Memory update skipped: no new messages since watermark (thread=%s)", thread_id)
                return True
            # watermark 이후 feed에서 signal을 다시 탐지해, 추출 hint가 LLM이 실제로 볼 turn만
            # 가리키게 한다. 유입 시점의 ``signals``(DeerMem에서 전체 대화 기준으로 탐지)는
            # 이미 제 역할(enqueue 시 backpressure 판단)을 다했다. hint는 부드러운 유도일 뿐이며
            # watermark가 제외한 turn을 가리켜서는 안 된다.
            feed_signals = detect_signals(feed_messages, patterns_dir=self._config.patterns_dir)
            prepared = self._prepare_update_prompt(
                messages=feed_messages,
                agent_name=agent_name,
                signals=feed_signals,
                user_id=user_id,
            )
            if prepared is None:
                return False

            current_memory, prompt = prepared
            model_name = self._config.model.model
            model = self._llm
            if model is None:
                raise RuntimeError("DeerMem memory update requested but no LLM is configured (set memory.backend_config.model in config).")
            invoke_config: dict[str, Any] = {"run_name": "memory_agent"}
            # LLM 호출 직전 observability hook(예: langfuse). 호출 전에 trace 메타데이터를
            # invoke_config에 병합해, tracer가 LLM 경계에서 span을 남기게 한다. None이면
            # tracing하지 않는다(langfuse는 필수가 아니다). 예전의
            # backend_config.tracing_callback을 대체한다.
            if self._callbacks is not None:
                self._callbacks.on_memory_llm_call(
                    invoke_config,
                    thread_id=thread_id,
                    user_id=user_id,
                    trace_id=trace_id,
                    model_name=model_name,
                )
            logger.info("Invoking memory-update LLM (thread=%s trace_id=%s)", thread_id, trace_id)
            attempted = True
            response = model.invoke(prompt, config=invoke_config)
            success = self._finalize_update(
                current_memory=current_memory,
                response_content=response.content,
                thread_id=thread_id,
                agent_name=agent_name,
                user_id=user_id,
                metrics=metrics,
            )
            if success and not bypass_watermark:
                # watermark를 마지막으로 넘긴 메시지까지 전진시킨다(feed는 suffix이므로
                # messages[-1]이다). 긴급 경로에서는 건너뛴다. 부분집합의 마지막 메시지는
                # 대화의 최신 메시지보다 오래됐으므로, 그것을 기준으로 전진시키면 뒤로 밀린다.
                self._watermark_set(watermark_key, _message_identity(messages[-1]))
            return success
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response for memory update: %s", e)
            return False
        except Exception as e:
            logger.exception("Memory update failed: %s", e)
            return False
        finally:
            # _finalize_update(또는 invoke)가 예외를 내도 metric을 남겨서, observability
            # callback이 정상 경로뿐 아니라 예외 실패(파싱 오류, 재시도 후 storage 오류)도 보게
            # 한다. 시도 이전의 조기 반환(새 메시지 없음, 빈 대화, 모델 없음)은 기존 동작대로
            # metric을 남기지 않는다.
            if attempted:
                self._emit_extraction_metrics(
                    metrics,
                    thread_id=thread_id,
                    user_id=user_id,
                    trace_id=trace_id,
                    model_name=model_name,
                    response=response,
                    success=success,
                )

    def update_memory(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        signals: frozenset[str] = frozenset(),
        user_id: str | None = None,
        trace_id: str | None = None,
        *,
        bypass_watermark: bool = False,
    ) -> bool:
        """동기 LLM 경로로 memory를 동기 갱신한다.

        lead agent가 공유하는 async ``AsyncClient``와 완전히 분리된 connection pool을 쓰는
        ``model.invoke()``(동기 HTTP)를 사용한다. issue #2615에서 설명한 loop 간 connection
        재사용 버그를 없앤다.

        실행 중인 event loop 안에서 호출되면(예: LangGraph 노드) 블로킹 동기 호출을 thread
        pool로 넘겨 호출자의 loop를 막지 않는다.

        Args:
            messages: 대화 메시지 리스트.
            thread_id: 출처 추적용 선택적 thread ID.
            agent_name: 주어지면 해당 agent의 memory를 갱신하고, None이면 전역 memory를 갱신한다.
            signals: 대화에서 탐지된 signal 클래스(correction / reinforcement / preference /
                ...). 추출 hint로 쓰인다.
            user_id: 주어지면 memory를 특정 사용자 범위로 한정한다.

        Returns:
            갱신이 저장되면 True. 실패(내용 없음, 파싱 불가 응답, LLM 오류) 시 False다. 실패는
            best-effort로 삼킨다. watermark가 실패 시 전진하지 않으므로, 실패한 갱신은 다음 대화
            turn에서 다시 넘어간다.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            try:
                future = _SYNC_MEMORY_UPDATER_EXECUTOR.submit(
                    self._do_update_memory_sync,
                    messages=messages,
                    thread_id=thread_id,
                    agent_name=agent_name,
                    signals=signals,
                    user_id=user_id,
                    trace_id=trace_id,
                    bypass_watermark=bypass_watermark,
                )
                return future.result()
            except Exception:
                logger.exception("Failed to offload memory update to executor")
                return False

        return self._do_update_memory_sync(
            messages=messages,
            thread_id=thread_id,
            agent_name=agent_name,
            signals=signals,
            user_id=user_id,
            trace_id=trace_id,
            bypass_watermark=bypass_watermark,
        )

    def _apply_updates(
        self,
        current_memory: dict[str, Any],
        update_data: dict[str, Any],
        thread_id: str | None = None,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """LLM이 만든 갱신을 memory에 적용한다.

        Args:
            current_memory: 현재 memory 데이터.
            update_data: LLM이 만든 갱신 내용.
            thread_id: 추적용 선택적 thread ID.
            metrics: 선택적 observability dict. 주어지면 confidence와 scope gate 카운터를 실제
                필터 지점에서 세어 채우므로, observability가 실제 수용 결과와 어긋날 수 없다.

        Returns:
            갱신된 memory 데이터.
        """
        config = self._config
        now = utc_now_iso_z()
        scope_gate_rejections: dict[str, dict[str, int]] = {
            "facts": {"missing": 0, "scope": 0, "durability": 0, "authority": 0},
            "summaries": {"missing": 0, "scope": 0, "authority": 0},
            "removals": {"missing": 0, "scope": 0, "replacement": 0},
            "consolidations": {"missing": 0, "scope": 0, "durability": 0, "authority": 0},
        }

        def reject_by_scope_gate(kind: str, reason: str) -> None:
            scope_gate_rejections[kind][reason] += 1

        # user 섹션 갱신
        user_updates = update_data.get("user", {})
        for section in ["workContext", "personalContext", "topOfMind"]:
            section_data = user_updates.get(section, {})
            if not isinstance(section_data, dict) or not section_data.get("shouldUpdate") or not section_data.get("summary"):
                continue
            rejection_reason = _summary_scope_gate_reason(section_data)
            if rejection_reason is not None:
                reject_by_scope_gate("summaries", rejection_reason)
            else:
                current_memory["user"][section] = {
                    "summary": section_data["summary"],
                    "updatedAt": now,
                }

        # history 섹션 갱신
        history_updates = update_data.get("history", {})
        for section in ["recentMonths", "earlierContext", "longTermBackground"]:
            section_data = history_updates.get(section, {})
            if not isinstance(section_data, dict) or not section_data.get("shouldUpdate") or not section_data.get("summary"):
                continue
            rejection_reason = _summary_scope_gate_reason(section_data)
            if rejection_reason is not None:
                reject_by_scope_gate("summaries", rejection_reason)
            else:
                current_memory["history"][section] = {
                    "summary": section_data["summary"],
                    "updatedAt": now,
                }

        # ── staleness 검토: 삭제 + 수명 연장 ──
        # 두 작업은 staleness 후보 guardrail 통과 과정과 candidate_ids 집합을 공유한다.
        # proposed_remove_ids는 삭제 하위 블록 밖으로 끌어올려, 주기별 상한이 실제로 지운 것뿐
        # 아니라 LLM이 제안한 *모든* 삭제를 포괄하게 한다. LLM이 지우려 한 fact는 상한 덕분에
        # 살아남았더라도 조용히 연장되어서는 안 된다.
        stale_removals = update_data.get("staleFactsToRemove", [])
        stale_extensions = update_data.get("staleFactsToExtend", [])
        has_staleness_ops = (isinstance(stale_removals, list) and stale_removals) or (isinstance(stale_extensions, list) and stale_extensions)
        if has_staleness_ops:
            # 결정적 guardrail: 실제 staleness 후보와 교집합을 취해, LLM이 보호 category나
            # 나이가 차지 않은 fact id를 내보내는 실수를 조용히 거부한다. 무조건 실행하므로
            # apply 계층의 보호는 모델 동작과 staleness_review_enabled 플래그 양쪽에 독립적이다.
            # id 필드 이전의 legacy / 수동 편집 fact도 방어한다. 나이가 찼고 보호 대상도 아닌데
            # "id"가 없는 fact는 유효한 staleness 후보지만 교집합에 쓸 id가 없으므로, KeyError를
            # 내지 말고 여기서 건너뛴다.
            candidate_ids = {f["id"] for f in _select_stale_candidates(current_memory, config) if f.get("id") is not None}

            # ── 삭제 ──
            proposed_remove_ids: set[str] = set()
            if isinstance(stale_removals, list) and stale_removals:
                proposed_remove_ids = {entry["id"] for entry in stale_removals if isinstance(entry, dict) and "id" in entry}
                stale_ids_to_remove = proposed_remove_ids & candidate_ids

                if not stale_ids_to_remove:
                    stale_removals = []
                else:
                    # 안전 상한: 주기당 staleness 삭제 개수를 제한한다. LLM이 상한보다 많이
                    # 반환하면 confidence가 낮은 항목만 상한까지 남겨서, 가장 의심스러운 fact가
                    # 먼저 지워지게 한다.
                    max_stale = config.staleness_max_removals_per_cycle
                    if len(stale_ids_to_remove) > max_stale:
                        stale_facts = [f for f in current_memory.get("facts", []) if f.get("id") in stale_ids_to_remove]
                        stale_facts.sort(key=_coerce_source_confidence)
                        stale_ids_to_remove = {f["id"] for f in stale_facts[:max_stale]}

                    current_memory["facts"] = [f for f in current_memory.get("facts", []) if f.get("id") not in stale_ids_to_remove]

                # observability를 위해 삭제 내역을 로그로 남긴다
                for entry in stale_removals:
                    if isinstance(entry, dict) and entry.get("id") in stale_ids_to_remove:
                        logger.info(
                            "Staleness review removed fact %s: %s",
                            entry["id"],
                            entry.get("reason", "no reason provided"),
                        )

            # ── 수명 연장 ──
            # LLM이 유지하기로 한 fact의 expected_valid_days를 재조정한다. 대상은 LLM이 삭제를
            # 제안하지 *않은* stale 후보이며, 주기별 상한 덕분에 살아남은 것도 포함한다. 새 창은
            # min(days_since + extend_by_days, staleness_max_extension_days)다. 연장은 생성
            # 시점의 배수 상한이 아니라 절대 상한을 쓴다. 연장은 의도적인 검토 결정이므로 원래
            # 생성 상한을 넘어 창을 옮길 수 있어야 하지만, 절대 상한이 timedelta overflow와 LLM
            # 오작동을 막는다.
            if isinstance(stale_extensions, list) and stale_extensions:
                # 잘라낸 집합뿐 아니라 LLM이 제안한 모든 삭제를 제외해, 상한 덕에 살아남은
                # 삭제 제안 fact가 연장되는 일이 없게 한다.
                extendable_ids = candidate_ids - proposed_remove_ids
                ext_by_id = {e["id"]: e for e in stale_extensions if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"] in extendable_ids}
                if ext_by_id:
                    now_utc = datetime.now(UTC)
                    max_ext = config.staleness_max_extension_days
                    updated_facts: list[dict[str, Any]] = []
                    for fact in current_memory.get("facts", []):
                        fid = fact.get("id")
                        ext = ext_by_id.get(fid) if fid else None
                        if ext is not None:
                            extend_by = ext.get("extend_by_days")
                            if isinstance(extend_by, (int, float)) and not isinstance(extend_by, bool):
                                extend_by_int = int(extend_by)  # 검사 전에 강제 변환한다
                                if extend_by_int > 0:
                                    created = _parse_fact_datetime(fact.get("createdAt", ""))
                                    if created is None:
                                        # 도달 불가: _select_stale_candidates가 파싱 불가
                                        # createdAt을 가진 fact를 이미 제외한다.
                                        updated_facts.append(fact)
                                        continue
                                    days_since = int((now_utc - created).total_seconds() // 86400)
                                    new_evd = min(days_since + extend_by_int, max_ext)
                                    fact = {**fact, "expected_valid_days": new_evd}
                                    logger.info(
                                        "Staleness review extended fact %s by %d days (new expected_valid_days: %d): %s",
                                        fid,
                                        extend_by_int,
                                        new_evd,
                                        ext.get("reason", "no reason provided"),
                                    )
                        updated_facts.append(fact)
                    current_memory["facts"] = updated_facts

        # 새 fact 추가
        existing_fact_keys = {fact_key for fact_key in (_fact_content_key(fact.get("content")) for fact in current_memory.get("facts", [])) if fact_key is not None}
        new_facts = update_data.get("newFacts", [])
        # 아래 consolidation 경로와 공유하는 생성 시점 수명 상한. 두 fact 생성 지점이 한 곳에서
        # 동일한 상한을 적용하게 한다.
        creation_cap = int(config.staleness_age_days * config.staleness_max_lifetime_multiplier)
        # 새 fact는 서로 독립인 두 수용 필터의 지배를 받는다. 결정적 scope gate와 이 confidence
        # 임계값이다. 각각 자기 필터 지점에서 세므로 어느 metric도 자신이 보고하는 필터와
        # 어긋날 수 없다. ``facts_passed_confidence``는 scope gate가 거부하더라도 임계값을 넘은
        # 항목을 세고, scope gate 카운터는 confidence 검사 통과 여부와 무관하게 증가한다.
        # 임계값을 통과한 중복/빈/상한 초과 fact도 여기서 센다. 이 metric은 저장된 fact 개수가
        # 아니라 confidence gate 신호이기 때문이다(host의 거부율 경고는 dedup이나 상한이 아니라
        # confidence 필터링을 감시한다).
        passed_threshold = 0
        replacement_fact_keys: dict[int, str] = {}
        for fact_index, fact in enumerate(new_facts):
            confidence = fact.get("confidence", 0.5)
            if confidence >= config.fact_confidence_threshold:
                passed_threshold += 1
            rejection_reason = _fact_scope_gate_reason(fact)
            if rejection_reason is not None:
                reject_by_scope_gate("facts", rejection_reason)
                continue
            if confidence < config.fact_confidence_threshold:
                continue
            raw_content = fact.get("content", "")
            if not isinstance(raw_content, str):
                continue
            normalized_content = raw_content.strip()
            fact_key = _fact_content_key(normalized_content)
            if fact_key is None:
                # 빈 content나 공백뿐인 content. 위의 비문자열 방어와 동일하게 건너뛴다.
                # content가 비어 있지 않아야 한다는 불변식을 어기는 빈 fact를 추가하지 않는다.
                continue
            # 이미 존재하더라도 자격을 갖춘 replacement의 content 키는 모두 기억해 둔다. 짝지어진
            # 삭제는 trim 이후 memory가 이 content를 대상 ID가 아닌 다른 ID로 갖고 있을 때만
            # 안전하다.
            replacement_fact_keys[fact_index] = fact_key
            if fact_key in existing_fact_keys:
                continue

            fact_entry = {
                "id": f"fact_{uuid.uuid4().hex[:8]}",
                "content": normalized_content,
                "category": fact.get("category", "context"),
                "confidence": confidence,
                "createdAt": now,
                "source": thread_id or "unknown",
            }
            source_error = fact.get("sourceError")
            if isinstance(source_error, str):
                normalized_source_error = source_error.strip()
                if normalized_source_error:
                    fact_entry["sourceError"] = normalized_source_error
            evd = _read_expected_valid_days(fact)
            if evd is not None:
                # 생성 시점 상한을 적용해, LLM이 무한한 수명을 지정해 staleness 검토를 영원히
                # 미루지 못하게 한다. 연장(staleFactsToExtend)은 검증 없는 최초 지정이 아니라
                # 의도적인 검토 결정이므로, 자체 staleness_max_extension_days 상한을 통해 이
                # 상한을 우회한다.
                fact_entry["expected_valid_days"] = min(evd, creation_cap)
            current_memory["facts"].append(fact_entry)
            existing_fact_keys.add(fact_key)

        # 최대 fact 수 제한을 적용한다(confidence 강제 변환은 _trim_facts_to_max 참고).
        current_memory["facts"] = _trim_facts_to_max(current_memory["facts"], config.max_facts)

        # 모순된 fact는 replacement가 두 gate를 모두 통과하고 중복 제거/trim에서도 살아남은
        # 뒤에만 지운다. task 범위의 모순은 사용자 memory를 지울 수 없고, 짝지어진 replacement가
        # 실패해도 삭제만 하는 갱신으로 전락하지 않는다.
        fact_ids_to_remove: set[str] = set()
        for removal in update_data.get("factsToRemove", []):
            if not isinstance(removal, dict):
                reject_by_scope_gate("removals", "missing")
                continue
            rejection_reason = _removal_scope_gate_reason(removal)
            if rejection_reason is not None:
                reject_by_scope_gate("removals", rejection_reason)
                continue
            fact_id = removal.get("id")
            if not isinstance(fact_id, str) or not fact_id:
                reject_by_scope_gate("removals", "missing")
                continue
            if "replacementFactIndex" in removal:
                replacement_index = removal.get("replacementFactIndex")
                if not isinstance(replacement_index, int) or isinstance(replacement_index, bool) or replacement_index < 0:
                    reject_by_scope_gate("removals", "replacement")
                    continue
                replacement_key = replacement_fact_keys.get(replacement_index)
                if replacement_key is None or not any(fact.get("id") != fact_id and _fact_content_key(fact.get("content")) == replacement_key for fact in current_memory.get("facts", [])):
                    reject_by_scope_gate("removals", "replacement")
                    continue
            fact_ids_to_remove.add(fact_id)

        if fact_ids_to_remove:
            current_memory["facts"] = [fact for fact in current_memory.get("facts", []) if fact.get("id") not in fact_ids_to_remove]

        # ── memory consolidation 적용 ──
        # max_facts trim 이후에 실행한다. 방금 밀려난 source fact(confidence가 낮아 높은
        # newFacts에 밀린 것)는 fact_index에 없어 존재 여부 guardrail에서 거부되므로, source는
        # 지워졌는데 병합된 대체 fact 자체도 잘려 나가는 유일한 실제 데이터 손실 시나리오를
        # 막는다. consolidation은 항상 fact를 2개 이상 지우고 1개를 추가하므로, trim 이후에
        # 실행해도 총합이 max_facts를 넘을 수 없다.
        # apply 시점에 feature flag를 확인해, debounce된 갱신과 경합하는 config 변경이 운영자가
        # 따로 두려던 fact를 조용히 병합하지 않게 한다.
        if config.consolidation_enabled:
            consolidation_decisions = update_data.get("factsToConsolidate", [])
            if isinstance(consolidation_decisions, list) and consolidation_decisions:
                fact_index = {f.get("id"): f for f in current_memory.get("facts", []) if isinstance(f, dict)}
                max_groups = config.consolidation_max_groups_per_cycle
                max_sources = config.consolidation_max_sources
                # newFacts 경로와 공유하는 생성 시점 수명 상한. 상속된 expected_valid_days를
                # 제한해, 오래 사는 source들의 병합이 최초 검토를 무한정 미루지 못하게 한다.
                creation_cap = int(config.staleness_age_days * config.staleness_max_lifetime_multiplier)
                ids_consumed: set[str] = set()
                new_consolidated: list[dict[str, Any]] = []
                merge_count = 0

                # staleness 통과 guardrail을 그대로 따른다. LLM이 후보로 볼 수 있었던 정당한 ID
                # 집합을 만든다(보호 category와 임계값 미만 category는 제외). 보호 대상이거나
                # 자격 없는 fact ID를 제안하는 LLM의 실수는 모델 동작과 무관하게 여기서
                # 거부된다. staleness가 삭제를 적용하기 전에 _select_stale_candidates와 교집합을
                # 취하는 것과 동일한 방식이다. id 없는 legacy fact는 건너뛴다(id 기반 source
                # 집합의 대상이 될 수 없다).
                allowed_source_ids = {f["id"] for group in _select_consolidation_candidates(current_memory, config).values() for f in group if f.get("id") is not None}

                # 미리 잘라내지 않고 모든 결정을 순회하며 성공 횟수를 센다. 앞쪽 결정의 guard
                # 실패 때문에 뒤쪽의 유효한 결정이 설정된 merge 예산을 못 쓰는 일이 없게 한다.
                for decision in consolidation_decisions:
                    if merge_count >= max_groups:
                        break

                    source_ids = decision.get("sourceIds", [])
                    consolidated = decision.get("consolidated", {})

                    # guardrail: 모든 source ID는 trim 이후 index에 존재해야 하고, 이번 주기의
                    # 앞선 merge에 이미 소비되지 않았어야 하며, allowed_source_ids에 있어야
                    # 한다. 그 집합은 _select_consolidation_candidates로 만들며
                    # staleness_protected_categories(기본값 "correction")에 속한 category를
                    # 제외한다. staleness의 apply 시점 검사와 동일한 방식으로, 명시적인 사용자
                    # 피드백이 모델 동작과 무관하게 조용히 병합되지 않게 한다.
                    if any(sid in ids_consumed or sid not in fact_index or sid not in allowed_source_ids for sid in source_ids):
                        continue
                    # guardrail: 그룹당 2..max_sources개
                    if not (2 <= len(source_ids) <= max_sources):
                        continue

                    content = consolidated.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    rejection_reason = _fact_scope_gate_reason(consolidated)
                    if rejection_reason is not None:
                        reject_by_scope_gate("consolidations", rejection_reason)
                        continue

                    source_confidences = [_coerce_source_confidence(fact_index[sid]) for sid in source_ids]
                    # _coerce_source_confidence가 각 값을 [0, 1]로 제한하므로 계약상
                    # max(source_confidences) ≤ 1.0이다.
                    max_source_conf = max(source_confidences)

                    # LLM이 반환한 confidence를 쓰되 source 최댓값으로 상한을 둬서,
                    # consolidation이 confidence를 부풀리지 못하게 한다. 먼저 [0, 1]로 제한해,
                    # 나중에 상한이 완화되더라도 범위를 벗어난 값(예: 1.5)이 새어 나가지 않게
                    # 한다. 값이 없거나 형식이 깨졌으면 max_source_conf로 fallback한다.
                    raw_llm_conf = consolidated.get("confidence")
                    if isinstance(raw_llm_conf, (int, float)) and not isinstance(raw_llm_conf, bool) and math.isfinite(float(raw_llm_conf)):
                        fact_confidence = min(max(0.0, min(float(raw_llm_conf), 1.0)), max_source_conf)
                    else:
                        fact_confidence = max_source_conf

                    # 결과가 저장 임계값 미만이 되는 merge는 건너뛴다. newFacts에 적용되는 것과
                    # 같은 gate이므로, 일반 유입 경로가 거부할 fact를 consolidation이 받아들이는
                    # 일이 없다.
                    if fact_confidence < config.fact_confidence_threshold:
                        continue

                    # 가장 최근 source의 createdAt을 물려받아, staleness 시계가 합성 시점이
                    # 아니라 원 정보의 나이를 반영하게 한다. consolidatedAt은 staleness 자격을
                    # 초기화하지 않고 감사용으로 merge 시각만 기록한다.
                    # 비교는 _parse_fact_datetime으로 한다. 충돌 없이 timezone을 인식하기
                    # 위해서다. createdAt이 숫자면 문자열 max()가 TypeError를 내고, Z와 +00:00이
                    # 섞이면 사전순 정렬이 틀린다.
                    _fallback_dt = _parse_fact_datetime(now) or datetime.now(UTC)
                    _source_dts = [_parse_fact_datetime(fact_index[sid].get("createdAt") or "") or _fallback_dt for sid in source_ids]
                    _newest_dt = max(_source_dts)
                    source_created_at = _newest_dt.isoformat().removesuffix("+00:00") + "Z"
                    new_fact: dict[str, Any] = {
                        "id": f"fact_{uuid.uuid4().hex[:8]}",
                        "content": content.strip(),
                        "category": consolidated.get("category", "context"),
                        "confidence": fact_confidence,
                        "createdAt": source_created_at,
                        "consolidatedAt": now,
                        "source": "consolidation",
                        "consolidatedFrom": list(source_ids),
                    }
                    # source fact의 sourceError를 전파해, 정정 context(무엇이 왜 잘못됐는지)가
                    # 조용히 사라지지 않게 한다.
                    source_errors = list(dict.fromkeys(e for sid in source_ids if isinstance((e := fact_index[sid].get("sourceError")), str) and e.strip()))
                    if source_errors:
                        new_fact["sourceError"] = "\n".join(source_errors)

                    # source들로부터 expected_valid_days를 물려받아, 병합된 fact가 전역
                    # staleness_age_days로 조용히 떨어지지 않고 원 정보의 수명 신호를 유지하게
                    # 한다. 병합된 fact는 source들의 검토 기한(createdAt + 실효 수명) 중 *가장
                    # 이른* 시점에 재검토된다. merge는 모든 source의 세부 내용을 합치는데,
                    # 변동성 큰 세부 항목(예: evd=7)이 안정적인 source의 3650일 창을 물려받아
                    # 수년간 staleness 검토를 빠져나가서는 안 되기 때문이다. 병합된 fact를 다시
                    # 검증하는 경로는 staleness KEEP/REMOVE뿐이므로, 가장 이른 기한 쪽으로
                    # 기울이면 불확실한 merge가 더 빨리 재검토된다. 명시적 evd가 없는 legacy
                    # fact를 포함해 모든 source가 참여한다. 그런 fact의 실효 수명은 설정된 전역
                    # staleness_age_days다(_effective_fact_staleness_age의 읽기 시점 fallback과
                    # 동일). 덕분에 legacy source의 기본 90일 창이 오래 사는 형제에게 조용히
                    # 삼켜지지 않는다. 기한은 병합된 fact의 createdAt(가장 최근 source의 것)을
                    # 기준으로 표현하므로, 이미 기한이 지난 source는 전역 fallback 대신 최소한의
                    # 양수 창을 낳는다(다음 주기 검토). 전역 fallback이었다면 밀린 검토를 더
                    # 미뤘을 것이다. 다른 새 fact와 마찬가지로 생성 시점 배수 상한(루프 밖으로
                    # 끌어올렸다)으로 제한해, consolidation이 최초 검토를 무한정 미루지 못하게
                    # 한다.
                    # 각 source의 절대 검토 기한(createdAt + 실효 수명)을 계산한다. 저장된 evd가
                    # 아주 크면 datetime 연산이 overflow할 수 있는데, 그때 _safe_add_days가
                    # None을 반환하고 해당 source는 전역 수명 기준 기한으로 fallback한다. legacy
                    # (evd 없음) source와 같은 처리라, 필드 하나가 깨졌다고 merge가 중단되지
                    # 않는다.
                    global_age = config.staleness_age_days
                    source_deadlines: list[datetime] = []
                    for sid, dt in zip(source_ids, _source_dts):
                        eff = _effective_fact_staleness_age(fact_index[sid], config)
                        deadline = _safe_add_days(dt, eff)
                        if deadline is None:
                            deadline = _safe_add_days(dt, global_age) or _newest_dt
                        source_deadlines.append(deadline)
                    earliest_deadline = min(source_deadlines)
                    # int(total_seconds() // 86400)은 #4143에서 지적된 .days의 0 방향 절삭
                    # 불일치를 피한다. 음수 결과(이미 기한이 지난 source)는 아래에서 제한한다.
                    days_until_earliest = int((earliest_deadline - _newest_dt).total_seconds() // 86400)
                    # 0 이하면 어떤 source의 기한이 이미 지났다는 뜻이다(merge 자체가 밀린
                    # 검토였다). 전역 fallback을 물려받는 대신 최소한의 양수 창을 줘서, 병합된
                    # fact가 다음 주기에 재검토되게 한다.
                    inherited_evd = max(days_until_earliest, 1)
                    new_fact["expected_valid_days"] = min(inherited_evd, creation_cap)

                    ids_consumed.update(source_ids)
                    new_consolidated.append(new_fact)
                    merge_count += 1
                    logger.info(
                        "Consolidation merged %d facts into: %s",
                        len(source_ids),
                        content.strip()[:80],
                    )

                if ids_consumed:
                    current_memory["facts"] = [f for f in current_memory.get("facts", []) if f.get("id") not in ids_consumed]
                    current_memory["facts"].extend(new_consolidated)

        if metrics is not None:
            metrics["facts_passed_confidence"] = passed_threshold
            metrics["rejected_low_confidence"] = len(new_facts) - passed_threshold
            metrics["facts_passed_scope_gate"] = len(new_facts) - sum(scope_gate_rejections["facts"].values())
            metrics["rejected_by_scope_gate"] = sum(count for reasons in scope_gate_rejections.values() for count in reasons.values())
            metrics["scope_gate_rejections"] = scope_gate_rejections

        return current_memory
