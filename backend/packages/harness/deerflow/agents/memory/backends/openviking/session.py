"""안정적인 OpenViking session 식별자와 transcript cursor 헬퍼."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .config import GENERATED_PEER_PREFIX, is_safe_peer_id

_SESSION_NAMESPACE = "deerflow-openviking-adapter-v1"
_DEFAULT_AGENT_SCOPE = "__default__"
_CURSOR_SCHEMA_VERSION = 1


def _canonical_peer_id(
    agent_name: str | None,
    default_peer_id: str,
) -> str:
    """대소문자를 구분하지 않는 DeerFlow agent 이름을 서로 겹치지 않는 peer ID로 매핑한다."""

    if agent_name is None:
        return default_peer_id

    value = str(agent_name).strip().lower()
    if not value or value == _DEFAULT_AGENT_SCOPE:
        raise ValueError(f"Invalid OpenViking peer scope: {agent_name!r}")
    if is_safe_peer_id(value) and value != default_peer_id and not value.startswith(GENERATED_PEER_PREFIX):
        return value

    # 생성된 네임스페이스는 예약되어 있어 호환 가능한 이름, 기본 peer, 해시 fallback이
    # 서로 같은 값이 될 수 없다. 128비트 digest는 agent 이름을 정규화하거나 잘라내면서
    # 생기는 충돌도 막는다.
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{GENERATED_PEER_PREFIX}{digest}"


def _session_id(
    owner_user_id: str,
    peer_id: str,
    thread_id: str,
) -> str:
    """DeerFlow thread 하나에 대응하는 안정적인 OpenViking session을 유도한다."""

    digest = hashlib.sha256(f"{_SESSION_NAMESPACE}\0{owner_user_id}\0{peer_id}\0{thread_id}".encode()).hexdigest()
    return f"df_{digest[:48]}"


def _memory_target_uris(peer_id: str) -> list[str]:
    """요청에 쓸 self 및 현재 peer의 memory 루트를 반환한다."""

    return [
        "viking://user/memories",
        f"viking://user/peers/{peer_id}/memories",
    ]


def _captureable_messages(
    messages: list[Any],
    should_keep_hidden_message: Any,
) -> list[Any]:
    """메시지를 OpenViking에 넘기기 전에 DeerFlow가 주입한 context를 제거한다."""

    selected: list[Any] = []
    for message in messages:
        additional_kwargs = _message_value(
            message,
            "additional_kwargs",
            {},
        )
        if not isinstance(additional_kwargs, dict):
            additional_kwargs = {}
        if additional_kwargs.get("hide_from_ui") and not (should_keep_hidden_message and should_keep_hidden_message(additional_kwargs)):
            continue
        selected.append(message)
    return selected


def _message_signature(message: Any) -> str:
    """transcript 내용을 보관하지 않고 메시지의 안정적인 의미만 해싱한다."""

    additional_kwargs = _message_value(message, "additional_kwargs", {})
    if not isinstance(additional_kwargs, Mapping):
        additional_kwargs = {}
    tool_calls = _message_value(message, "tool_calls", None)
    if not tool_calls:
        tool_calls = additional_kwargs.get("tool_calls") or []

    value = {
        "id": _message_value(message, "id", None),
        "role": _message_value(message, "type", None) or _message_value(message, "role", None),
        "content": _message_value(message, "content", ""),
        "tool_calls": tool_calls,
        "tool_call_id": _message_value(message, "tool_call_id", None) or _message_value(message, "tool_id", None),
        "tool_name": _message_value(message, "name", None) or _message_value(message, "tool_name", None),
        "tool_status": _message_value(message, "status", None) or _message_value(message, "tool_status", None),
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matching_prefix_count(
    state: dict[str, Any],
    signatures: list[str],
) -> int | None:
    """이미 제출된 prefix 길이를 반환한다. compaction 이후에도 동작한다."""

    count = state.get("submitted_prefix_count")
    digest = state.get("submitted_prefix_digest")
    if isinstance(count, int) and 0 <= count <= len(signatures) and isinstance(digest, str):
        if _sequence_digest(signatures[:count]) == digest:
            return count
        return None

    submitted = _string_list(state.get("submitted_signatures"))
    if submitted and len(submitted) <= len(signatures):
        width = len(submitted)
        for start in range(len(signatures) - width, -1, -1):
            if signatures[start : start + width] == submitted:
                return start + width
    return 0 if not state else None


def _advanced_cursor(
    previous: dict[str, Any],
    prefix_signatures: list[str] | None,
    newly_submitted: list[str],
    *,
    max_seen: int,
    commit_pending: bool,
) -> dict[str, Any]:
    """메시지 내용을 저장하지 않고 확정된 capture 진행 상황만 전진시킨다."""

    recent = [
        *_string_list(previous.get("submitted_signatures")),
        *newly_submitted,
    ][-max_seen:]
    state: dict[str, Any] = {
        "schema_version": _CURSOR_SCHEMA_VERSION,
        "submitted_signatures": recent,
        "commit_pending": commit_pending,
    }
    if prefix_signatures is not None:
        state["submitted_prefix_count"] = len(prefix_signatures)
        state["submitted_prefix_digest"] = _sequence_digest(prefix_signatures)
    else:
        state["submitted_prefix_count"] = previous.get("submitted_prefix_count")
        state["submitted_prefix_digest"] = previous.get("submitted_prefix_digest")
    return state


def _sequence_digest(signatures: list[str]) -> str:
    digest = hashlib.sha256()
    for signature in signatures:
        encoded = signature.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _message_value(message: Any, key: str, default: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)
