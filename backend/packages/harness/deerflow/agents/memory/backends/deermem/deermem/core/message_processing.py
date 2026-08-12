"""대화를 memory 갱신 입력으로 바꾸는 공용 헬퍼."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_UPLOAD_BLOCK_RE = re.compile(r"<(?P<tag>uploaded_files|current_uploads)>[\s\S]*?</(?P=tag)>\n*", re.IGNORECASE)

_PATTERN_CACHE: dict[tuple[str, str | None], list[re.Pattern[str]]] = {}


def load_patterns(name: str, *, patterns_dir: str | None = None) -> list[re.Pattern[str]]:
    """YAML 파일에서 signal 패턴을 읽어 컴파일한다.

    ``name``은 ``"correction"`` 또는 ``"reinforcement"``다. ``patterns_dir``은
    번들된 ``core/message_patterns/`` 디렉터리를 대체하며, ``None``(기본값)이면
    번들 기본값을 쓴다. 기본값은 외부화 이전의 하드코딩 패턴과 동일하므로
    무설정 동작이 그대로 유지된다. 컴파일 결과는 ``(name, patterns_dir)`` 단위로
    캐시한다.

    YAML 리스트의 각 항목은 문자열(플래그 없이 컴파일)이거나
    ``{pattern: <regex>, flags: [...]}`` 매핑이며 ``flags``에는 ``"ignorecase"``를
    넣을 수 있다. 잘못된 YAML, 리스트가 아닌 최상위 값, 잘못된 regex는
    ``ValueError``를 던진다(모두 메시지에 파일 경로를 포함한다).
    *patterns_dir*를 명시한 경우 파일이 없으면 ``FileNotFoundError``를 던지고
    읽기 실패(OSError)는 그대로 다시 던진다. 형식이 깨진 항목과 알 수 없는 플래그
    이름은 WARNING만 남기고 건너뛴다. 번들 기본값(*patterns_dir*가 ``None``)일 때는
    파일이 없거나 읽지 못하면 WARNING을 남기고 ``[]``를 반환한다. 설정 오류가 아니라
    패키징 버그이기 때문이다.
    """
    cache_key = (name, patterns_dir)
    cached = _PATTERN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base = Path(patterns_dir) if patterns_dir else Path(__file__).parent / "message_patterns"
    path = base / f"{name}.yaml"
    if not path.exists():
        if patterns_dir is not None:
            raise FileNotFoundError(f"Signal patterns file not found: {path}")
        logger.warning("Signal patterns file not found (%s); %s detection disabled.", path, name)
        _PATTERN_CACHE[cache_key] = []
        return []

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    except OSError as e:
        if patterns_dir is not None:
            raise OSError(f"Failed to read signal patterns file {path}: {e}") from e
        logger.warning("Failed to read signal patterns %s: %s; %s detection disabled.", path, e, name)
        _PATTERN_CACHE[cache_key] = []
        return []

    if not isinstance(data, list):
        raise ValueError(f"Signal patterns file {path} must contain a list, not {type(data).__name__}")

    compiled: list[re.Pattern[str]] = []
    for i, entry in enumerate(data):
        if isinstance(entry, str):
            pattern_text, flag_names = entry, []
        elif isinstance(entry, Mapping):
            pattern_text = entry.get("pattern")
            flag_names = entry.get("flags", []) or []
        else:
            logger.warning("Skipping non-string/non-mapping entry %d in %s (type %s)", i, path, type(entry).__name__)
            continue
        if not isinstance(pattern_text, str) or not pattern_text:
            logger.warning("Skipping entry %d in %s: missing or empty 'pattern'", i, path)
            continue
        flags = 0
        for flag_name in flag_names:
            if flag_name == "ignorecase":
                flags |= re.IGNORECASE
            else:
                logger.warning("Ignoring unknown flag %r in entry %d of %s", flag_name, i, path)
        try:
            compiled.append(re.compile(pattern_text, flags))
        except re.error as e:
            raise ValueError(f"Invalid regex in {path} entry {i}: {e} (pattern={pattern_text!r})") from e

    _PATTERN_CACHE[cache_key] = compiled
    return compiled


def extract_message_text(message: Any) -> str:
    """필터링과 signal 감지에 쓸 평문 텍스트를 메시지 content에서 뽑아낸다."""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        return " ".join(text_parts)
    return str(content)


def _non_empty_str(value: object) -> str | None:
    """``value``가 공백 제거 후에도 비어 있지 않은 문자열이면 그대로, 아니면 None을 반환한다."""
    return value if isinstance(value, str) and value.strip() else None


def _is_human_clarification_response(additional_kwargs: Any) -> bool:
    """``additional_kwargs``가 형식에 맞는 human clarification 응답을 담고 있을 때만 True를 반환한다.

    사용자가 직접 쓴, 기억할 가치가 있는 답변인지 판단한다. deer-flow의
    ``read_human_input_response``(운영에서는 host가 ``should_keep_hidden_message``로
    주입한다)를 host 비의존 구조 검사로 옮긴 것이다. 조건은 version 1 + kind
    ``human_input_response``인 ``human_input_response`` 매핑, 비어 있지 않은
    source/request_id/value, 그리고 option 응답이면 비어 있지 않은 option_id다.
    형식이 깨졌거나 일부만 있는 payload는 False를 반환해 다른 hide_from_ui
    프레임워크 메시지처럼 제외된다. host를 import하지 않고 인라인으로 둔 이유는
    ``filter_messages_for_memory``가 단독 실행이나 테스트에서도 올바르게 동작하게
    하기 위해서다. NOTE: human_input_response 포맷이 바뀌면 운영 경로인
    ``read_human_input_response``와 반드시 함께 맞춰야 한다.
    """
    if not isinstance(additional_kwargs, Mapping):
        return False
    raw = additional_kwargs.get("human_input_response")
    if not isinstance(raw, Mapping):
        return False
    if raw.get("version") != 1 or raw.get("kind") != "human_input_response":
        return False
    if _non_empty_str(raw.get("source")) is None or _non_empty_str(raw.get("request_id")) is None or _non_empty_str(raw.get("value")) is None:
        return False
    response_kind = raw.get("response_kind")
    if response_kind == "text":
        return True
    if response_kind == "option":
        return _non_empty_str(raw.get("option_id")) is not None
    return False


def filter_messages_for_memory(messages: list[Any], *, should_keep_hidden_message: Any = None) -> list[Any]:
    """memory 갱신을 위해 사용자 입력과 최종 assistant 응답만 남긴다.

    ``hide_from_ui`` 프레임워크 메시지는 건너뛰지만, 사용자가 직접 쓴 clarification
    답변(형식에 맞는 ``human_input_response``)은 host 비의존 구조 검사로 기본 유지한다
    (deer-flow의 ``read_human_input_response``와 동일한 판정).
    ``should_keep_hidden_message(additional_kwargs) -> bool`` 훅을 넘기면 유지 여부
    판단을 대체할 수 있다. 운영에서는 host가 권위 있는
    ``read_human_input_response``에 위임하는 훅을 주입한다.
    """
    filtered = []
    skip_next_ai = False
    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            # middleware가 주입한 숨김 메시지(TodoMiddleware.todo_reminder,
            # ViewImageMiddleware, p0 DynamicContextMiddleware.__memory 등)는
            # hide_from_ui를 달고 있으며 memory 갱신 LLM에 절대 도달하면 안 된다.
            # 프레임워크 내부 텍스트가 장기 memory를 오염시키고, p0 __memory
            # payload는 자기 증폭 루프를 유발할 수 있다.
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if additional_kwargs.get("hide_from_ui"):
                # 프레임워크가 주입한 숨김 메시지(TodoMiddleware 리마인더,
                # ViewImage payload, p0 __memory 자기 증폭 방지)는 제외한다.
                # 사용자가 직접 쓴 clarification 답변(형식에 맞는
                # human_input_response)은 기억할 가치가 있는 실제 내용이므로
                # host 비의존 구조 검사로 기본 유지한다.
                # host가 ``should_keep_hidden_message`` 훅을 주면 그 판단이 우선한다
                # (운영 DeerMem은 read_human_input_response에 위임하는 훅을 주입한다).
                if should_keep_hidden_message is not None:
                    keep = should_keep_hidden_message(additional_kwargs)
                else:
                    keep = _is_human_clarification_response(additional_kwargs)
                if not keep:
                    continue
            content_str = extract_message_text(msg)
            if "<uploaded_files>" in content_str.lower() or "<current_uploads>" in content_str.lower():
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                if not stripped:
                    skip_next_ai = True
                    continue
                clean_msg = copy(msg)
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                filtered.append(msg)
                skip_next_ai = False
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                if skip_next_ai:
                    skip_next_ai = False
                    continue
                filtered.append(msg)

    return filtered


def detect_correction(messages: list[Any], *, patterns: list[re.Pattern[str]] | None = None) -> bool:
    """최근 대화 턴에서 사용자의 명시적 정정을 감지한다.

    ``patterns``를 주면 로드된 패턴 대신 그것을 쓴다(호출자가 이미
    ``DeerMemConfig.patterns_dir``를 해석한 경우에 유용하다). ``None``이면
    :func:`load_patterns`로 번들 기본값을 읽는다. 스캔 범위는 최근 human 턴인
    ``messages[-6:]``로 고정한다.
    """
    if patterns is None:
        patterns = load_patterns("correction")
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        content = extract_message_text(msg).strip()
        if content and any(pattern.search(content) for pattern in patterns):
            return True

    return False


def detect_reinforcement(messages: list[Any], *, patterns: list[re.Pattern[str]] | None = None) -> bool:
    """최근 대화 턴에서 명시적인 긍정 강화 signal을 감지한다.

    ``patterns``를 주면 로드된 패턴 대신 그것을 쓴다(호출자가 이미
    ``DeerMemConfig.patterns_dir``를 해석한 경우에 유용하다). ``None``이면
    :func:`load_patterns`로 번들 기본값을 읽는다. 스캔 범위는 ``messages[-6:]``로
    고정한다.
    """
    if patterns is None:
        patterns = load_patterns("reinforcement")
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        content = extract_message_text(msg).strip()
        if content and any(pattern.search(content) for pattern in patterns):
            return True

    return False


# :func:`detect_signals`가 감지하는 signal 종류. 이름은 fact의 ``category``
# enum(CORE_CATEGORIES)과 맞춰 두어 signal이 추출 category 힌트로 바로 쓰인다.
# 예외는 ``reinforcement``로, 같은 이름의 category가 없고 추출 힌트에서는
# preference/behavior로 매핑된다. 새 signal 이름을 추가하기 전에 반드시
# CORE_CATEGORIES와 맞춘다.
SIGNAL_NAMES: tuple[str, ...] = (
    "correction",
    "reinforcement",
    "preference",
    "identity",
    "goal",
    "decision",
)


def detect_signals(
    messages: list[Any],
    *,
    patterns_dir: str | None = None,
) -> set[str]:
    """최근 대화 턴에서 signal 종류를 감지한다.

    최근 human 턴 6개 중 하나라도 패턴이 일치하는 signal 이름의 집합을 반환한다.
    :func:`detect_correction` / :func:`detect_reinforcement`(하위 호환용으로 남아 있다)를
    전체 signal 집합으로 일반화한 함수다. 스캔 범위는 ``messages[-6:]``로 고정한다.
    """
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]
    if not recent_user_msgs:
        return set()

    hits: set[str] = set()
    for name in SIGNAL_NAMES:
        patterns = load_patterns(name, patterns_dir=patterns_dir)
        if not patterns:
            continue
        for msg in recent_user_msgs:
            content = extract_message_text(msg).strip()
            if content and any(pattern.search(content) for pattern in patterns):
                hits.add(name)
                break
    return hits


# 메시지 전체를 trivial로 판정하기 전에 잘라내는 후행 문자. 뒤에 문장부호가 붙은
# 단순 응답("ok.", "好的！")도 여전히 trivial이다.
_TRIVIAL_TRAIL = " \t\n\r.。,，!！?？;；"


def filter_trivial(
    messages: list[Any],
    *,
    patterns: list[re.Pattern[str]] | None = None,
    patterns_dir: str | None = None,
) -> list[Any]:
    """단순 응답뿐인 human 턴과 그에 대한 AI 답변을 제거한다.

    공백을 제거한 전체 텍스트가 trivial 패턴("嗯", "ok", "好的", "谢谢" 등)과
    일치하면 그 human 턴은 "trivial"이다. ``fullmatch``로 판정하므로 "ok"가 포함된
    실질적인 턴은 절대 버려지지 않는다. 일치한 human 턴과 바로 뒤따르는 assistant
    답변을 함께 제거한다(:func:`filter_messages_for_memory`의 ``skip_next_ai`` 방식을
    그대로 쓴다). 모든 턴이 trivial이면 결과가 비고, 호출자는 이를 "enqueue하지 않음"
    으로 처리해 추출 LLM 호출을 아낀다.
    """
    if patterns is None:
        patterns = load_patterns("trivial", patterns_dir=patterns_dir)
    if not patterns:
        return list(messages)

    result: list[Any] = []
    skip_next_ai = False
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            content = extract_message_text(msg).strip().rstrip(_TRIVIAL_TRAIL)
            is_trivial = bool(content) and any(pattern.fullmatch(content) for pattern in patterns)
            if is_trivial:
                skip_next_ai = True
                continue
            result.append(msg)
            skip_next_ai = False
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                if skip_next_ai:
                    skip_next_ai = False
                    continue
                result.append(msg)
    return result
