"""memory 갱신 및 주입에 쓰는 prompt 템플릿."""

from __future__ import annotations

import html
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class PromptConfigurationError(ValueError):
    """prompt 템플릿 설정 오류(잘못된 yaml, 누락된 키, 잘못된 placeholder).

    :func:`load_prompt`과 :func:`load_prompt_messages`가 맨 :class:`ValueError` 대신
    이 예외를 던져, 호출자가 영구적인 설정 실패와 복구 가능한 런타임 오류를 구분할 수 있게 한다."""


try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# ── 외부화된 prompt 템플릿 ───────────────────────────────────────
#
# memory prompt 4종은 ``core/prompts/`` 아래 yaml 파일로 존재하며
# :func:`load_prompt`가 읽는다. 덕분에 코드 수정 없이 agent별로 또는 외부 디렉터리로
# 덮어쓸 수 있다. 번들 기본값은 예전 모듈 상수와 바이트 단위로 동일하므로 무설정
# 동작이 그대로다. 템플릿은 ``.format`` 문법을 쓴다(``{var}`` 치환, 리터럴 중괄호는
# ``{{``/``}}``). html escape는 조립 계층(updater.py의 ``_escape_memory_for_prompt``와
# 여기의 ``format_conversation_for_update``)에서만 하고 템플릿 문자열 안에서는 하지
# 않는다. 이중 escape를 막기 위해서다.

_PROMPTS_DEFAULT_DIR = Path(__file__).resolve().parent / "prompts"

# load_prompt용 캐시. 같은 (name, agent, dir)로 다시 호출하면 yaml을 다시 읽지 않고
# 캐시된 템플릿 문자열을 반환한다. 아래 shim 상수들도 import 시점에 번들 기본값으로
# 이 캐시를 채운다.
_PROMPT_CACHE: dict[tuple[str, str | None, str | None], str] = {}

# load_prompt_messages용 캐시. 파싱된 원본 템플릿({role, content} dict 리스트)을
# (name, agent, dir) 키로 저장한다. 캐시가 맞으면 호출자의 변수로 렌더링만 하며,
# 파일은 키마다 한 번씩만 읽는다.
_CHAT_TEMPLATE_CACHE: dict[tuple[str, str | None, str | None], tuple[list[dict[str, str]], str]] = {}


def _render_messages(
    raw_templates: list[dict[str, str]],
    variables: dict[str, Any],
    source_path: str,
) -> list[BaseMessage]:
    """캐시된 chat 템플릿을 새 *variables*로 렌더링한다."""
    messages: list[BaseMessage] = []
    for tmpl in raw_templates:
        content = tmpl["content"]
        try:
            content = content.format(**variables)
        except (KeyError, ValueError) as e:
            raise PromptConfigurationError(f"Invalid placeholder in {source_path!r} (content of role={tmpl['role']!r}): {e}") from e
        if tmpl["role"] == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def load_prompt(
    name: str,
    *,
    agent_name: str | None = None,
    prompts_dir: str | None = None,
) -> str:
    """이름으로 prompt 템플릿을 읽는다(agent 재정의가 기본값보다 우선한다).

    ``{prompts_dir}/{agent_name}/{name}.yaml``이 있으면 그것을, 없으면
    ``{prompts_dir}/{name}.yaml``을 읽는다. ``prompts_dir`` 기본값은 패키지에 번들된
    ``core/prompts/``다. 반환값은 ``.format`` 문법의 원본 ``template`` 문자열이며,
    렌더링(``.format(**vars)``)은 호출자가 한다.

    결과는 ``(name, agent_name, prompts_dir)`` 조합마다 캐시하므로 파일 시스템 읽기는
    조합당 최대 한 번만 일어난다(번들 기본값이면 보통 프로세스당 한 번).
    """
    cache_key = (name, agent_name, prompts_dir)
    cached = _PROMPT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base = Path(prompts_dir) if prompts_dir else _PROMPTS_DEFAULT_DIR
    candidates: list[Path] = [base / f"{name}.yaml"]
    if agent_name:
        candidates.insert(0, base / agent_name / f"{name}.yaml")
    for path in candidates:
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                raise PromptConfigurationError(f"Invalid YAML in {path}: {e}") from e
            data = data or {}
            fmt = data.get("format", "text")
            if fmt != "text":
                raise PromptConfigurationError(f"Expected format='text' in {path}, got {fmt!r}; use load_prompt_messages() for chat-format templates")
            template = data.get("template")
            if not isinstance(template, str) or not template:
                raise PromptConfigurationError(f"Missing or empty 'template' key in {path}")
            _PROMPT_CACHE[cache_key] = template
            return template
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"prompt template not found: {name} (searched: {searched})")


def load_prompt_messages(
    name: str,
    variables: dict[str, Any],
    *,
    agent_name: str | None = None,
    prompts_dir: str | None = None,
) -> list[BaseMessage]:
    """chat 형식 prompt 템플릿을 읽고 렌더링해 ``list[BaseMessage]``를 반환한다.

    ``{prompts_dir}/{agent_name}/{name}.chat.yaml``이 있으면 그것을, 없으면
    ``{prompts_dir}/{name}.chat.yaml``을 읽는다. 각 메시지의 ``content``는
    ``.format(**variables)``로 렌더링한다. system content에는 변수가 없고 리터럴
    ``{{ }}`` JSON 중괄호만 있어 매 호출마다 바이트 단위로 동일하게 렌더링된다.
    lead agent의 정적 system prompt와 마찬가지로 prefix cache에 유리하다.

    치환 전 원본 템플릿(role + content)은 ``(name, agent, prompts_dir)`` 단위로
    캐시해 yaml 파일은 한 번만 읽고, 호출마다 렌더링만 수행한다.

    단일 문자열인 text 형식은 :func:`load_prompt`를 쓴다.
    """
    cache_key = (name, agent_name, prompts_dir)
    cached_chat = _CHAT_TEMPLATE_CACHE.get(cache_key)
    if cached_chat is not None:
        raw_templates, source_path = cached_chat
        return _render_messages(raw_templates, variables, source_path)

    base = Path(prompts_dir) if prompts_dir else _PROMPTS_DEFAULT_DIR
    candidates: list[Path] = [base / f"{name}.chat.yaml"]
    if agent_name:
        candidates.insert(0, base / agent_name / f"{name}.chat.yaml")
    for path in candidates:
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                raise PromptConfigurationError(f"Invalid YAML in {path}: {e}") from e
            data = data or {}
            fmt = data.get("format", "chat")
            if fmt != "chat":
                raise PromptConfigurationError(f"Expected format='chat' in {path}, got {fmt!r}; use load_prompt() for text-format templates")
            msg_list = data.get("messages")
            if not isinstance(msg_list, list) or not msg_list:
                raise PromptConfigurationError(f"Missing or empty 'messages' key in {path}")
            raw_templates: list[dict[str, str]] = []
            for msg in msg_list:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                raw_templates.append({"role": role, "content": content})
            _CHAT_TEMPLATE_CACHE[cache_key] = (raw_templates, str(path))
            return _render_messages(raw_templates, variables, str(path))
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"chat prompt template not found: {name} (searched: {searched})")


# 주입되는 텍스트 섹션(staleness_review / consolidation / fact_extraction)의
# 모듈 레벨 별칭. 각각 import 시점에 번들 yaml 템플릿을 한 번 읽는다.
# ``memory_update``는 여기 없다. lead agent의 정적 system prompt처럼 system/user를
# 분리한 chat 형식이라 :func:`load_prompt_messages`를 쓴다.
STALENESS_REVIEW_PROMPT = load_prompt("staleness_review")
CONSOLIDATION_PROMPT = load_prompt("consolidation")
FACT_EXTRACTION_PROMPT = load_prompt("fact_extraction")


# 모듈 레벨 tiktoken encoding 캐시. 최초 사용 시 지연 로딩하고 이후 호출은 dict
# 조회로 끝난다(네트워크 I/O 없음). 시작 시 :func:`warm_tiktoken_cache`로 미리
# 데워 두면 느릴 수 있는 첫 ``get_encoding`` 호출에 요청이 묶이지 않는다.
#
# 로드 *실패*는 ``(None, monotonic_timestamp)`` 튜플로 캐시한다. 네트워크가 제한된
# 환경에서 이후 호출마다 blocking BPE 다운로드를 다시 시도하지 않기 위해서다.
# ``_TIKTOKEN_RETRY_COOLDOWN_S``가 지나면 실패 캐시가 만료되므로, 일시적인 네트워크
# 장애는 프로세스 재시작 없이 정확한 tiktoken 계산으로 복구된다. 진행 중인 로드는
# ``_TIKTOKEN_ENCODING_LOADING``으로 캐시해 동시 호출자가 blocking
# ``tiktoken.get_encoding`` 스레드를 더 만들지 않고 즉시 fallback하게 한다.
# tiktoken을 아예 쓰지 않으려면 ``memory.token_counting: char`` 설정을 쓴다.
_TIKTOKEN_ENCODING_MISSING = object()
_TIKTOKEN_ENCODING_LOADING = object()
# tiktoken 로드 *실패* 후 재시도까지의 대기 시간. 사용자 설정이 아니라 내부 튜닝
# 상수이며, 기본 ``tiktoken`` 모드가 일시적 네트워크 장애에서 얼마나 빨리 복구되는지만
# 좌우한다. tiktoken의 네트워크 의존성 자체를 피하려면 이 값을 조정하지 말고
# ``memory.token_counting: char``를 설정한다.
_TIKTOKEN_RETRY_COOLDOWN_S = 600.0
_tiktoken_encoding_cache: dict[str, Any] = {}
_tiktoken_encoding_cache_lock = threading.Lock()


def _get_tiktoken_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding | None:
    """캐시된 tiktoken encoding을 반환하고, 실패하거나 쓸 수 없으면 ``None``을 반환한다.

    특정 *encoding_name*에 대한 최초 호출에서는 tiktoken이
    ``openaipublic.blob.core.windows.net``에서 BPE 데이터를 내려받아야 할 수 있다.
    네트워크가 제한된 환경(예: GFW 뒤 배포)에서는 OS TCP 타임아웃이 걸릴 때까지
    수십 분 동안 블로킹될 수 있다. 따라서 호출자는 블로킹을 감안해 event loop 밖에서
    실행해야 한다(예: ``asyncio.to_thread``).

    로드 실패는 타임스탬프와 함께 기억해 두어, 이후 호출은 blocking 다운로드를 다시
    유발하지 않고 즉시 문자 기반 추정으로 fallback한다. 실패 기록은
    ``_TIKTOKEN_RETRY_COOLDOWN_S`` 후 만료되므로 일시적 장애는 재시작 없이 복구된다.
    진행 중인 로드도 기억해, 타임아웃된 호출자 때문에 이후 요청들이 blocking
    ``get_encoding`` 호출을 더 시작하는 구간이 생기지 않게 한다.
    """
    if not TIKTOKEN_AVAILABLE:
        return None

    with _tiktoken_encoding_cache_lock:
        cached = _tiktoken_encoding_cache.get(encoding_name, _TIKTOKEN_ENCODING_MISSING)
        if cached is _TIKTOKEN_ENCODING_LOADING:
            return None
        if isinstance(cached, tuple):
            # 캐시된 실패 기록 (None, failed_at). cooldown이 지난 뒤에만 재시도한다.
            _, failed_at = cached
            if time.monotonic() - failed_at < _TIKTOKEN_RETRY_COOLDOWN_S:
                return None
            cached = _TIKTOKEN_ENCODING_MISSING
        if cached is not _TIKTOKEN_ENCODING_MISSING:
            return cast("tiktoken.Encoding", cached)
        _tiktoken_encoding_cache[encoding_name] = _TIKTOKEN_ENCODING_LOADING

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        logger.warning("Failed to load tiktoken encoding %r; falling back to char-based estimation", encoding_name, exc_info=True)
        with _tiktoken_encoding_cache_lock:
            _tiktoken_encoding_cache[encoding_name] = (None, time.monotonic())
        return None

    with _tiktoken_encoding_cache_lock:
        _tiktoken_encoding_cache[encoding_name] = encoding
    return encoding


def _char_based_token_estimate(text: str) -> int:
    """CJK 밀도를 반영한 네트워크 불필요 token 추정.

    단순한 ``len(text) // 4`` 휴리스틱은 영어/코드(token당 약 4자)에는 무난하지만
    중국어·일본어·한국어에서는 token당 1.5~2자에 가까워 token 수를 크게 과소평가한다.
    CJK 문자를 따로 세면(token당 약 2자) CJK 비중이 큰 memory 내용에서 주입 예산이
    넘치는 것을 막는다.
    """
    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"  # CJK \ud1b5\ud569 \ud55c\uc790
        or "\u3040" <= ch <= "\u30ff"  # \ud788\ub77c\uac00\ub098 + \uac00\ud0c0\uce74\ub098
        or "\uac00" <= ch <= "\ud7a3"  # \ud55c\uae00 \uc74c\uc808
    )
    return (len(text) - cjk) // 4 + cjk // 2


def _count_tokens(text: str, encoding_name: str = "cl100k_base", *, use_tiktoken: bool = True) -> int:
    """tiktoken으로 텍스트의 token 수를 센다.

    Args:
        text: token 수를 셀 텍스트.
        encoding_name: 사용할 encoding(기본값: GPT-4/3.5용 cl100k_base).
        use_tiktoken: ``False``면 tiktoken을 아예 쓰지 않고 네트워크가 필요 없는
            문자 기반 추정을 쓴다. BPE 다운로드를 절대 시도하지 않음을 보장한다
            (``memory.token_counting`` 설정 참고).

    Returns:
        텍스트의 token 수.
    """
    if not use_tiktoken:
        return _char_based_token_estimate(text)

    encoding = _get_tiktoken_encoding(encoding_name)
    if encoding is None:
        # tiktoken을 쓸 수 없거나 encoding 로드에 실패하면 CJK를 고려한
        # 문자 기반 추정으로 fallback한다.
        return _char_based_token_estimate(text)

    try:
        return len(encoding.encode(text))
    except Exception:
        # 오류가 나면 CJK를 고려한 문자 기반 추정으로 fallback한다.
        return _char_based_token_estimate(text)


def warm_tiktoken_cache() -> bool:
    """tiktoken encoding 캐시를 미리 데운다.

    첫 요청이 BPE 다운로드에 묶이지 않도록 시작 시점에 event loop 밖에서 호출한다.
    encoding을 성공적으로 로드했거나 이미 캐시돼 있으면 ``True``, tiktoken을 쓸 수
    없거나 다운로드가 실패하면 ``False``를 반환한다.
    """
    return _get_tiktoken_encoding("cl100k_base") is not None


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    """confidence 값을 [0, 1] 범위의 float로 변환한다.

    유한하지 않은 값(NaN, inf, -inf)은 잘못된 값으로 보고 clamp 전에 기본값으로
    되돌려, 이런 값이 순위를 지배하지 못하게 한다. ``default``는 유한한 값이라고
    가정한다.
    """
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return max(0.0, min(1.0, default))
    if not math.isfinite(confidence):
        return max(0.0, min(1.0, default))
    return max(0.0, min(1.0, confidence))


def _format_fact_line(fact: dict[str, Any]) -> str | None:
    """포맷된 fact 한 줄을 만든다. 잘못된 fact면 ``None``을 반환한다.

    guaranteed 주입 경로와 일반 주입 경로가 동일한 줄 포맷을 내도록 공용 헬퍼로
    분리했다.
    """
    content_value = fact.get("content")
    if not isinstance(content_value, str):
        return None
    content = content_value.strip()
    if not content:
        return None
    category = str(fact.get("category", "context")).strip() or "context"
    confidence = _coerce_confidence(fact.get("confidence"), default=0.0)
    source_error = fact.get("sourceError")
    # 이 필드들은 사용자가 편집할 수 있고(POST/PATCH /api/memory, import)
    # lead-agent system prompt의 <memory> 블록에 렌더링된다. escape하지 않으면
    # "</memory></system-reminder>" 같은 값이 블록을 닫고 뒤 텍스트를 prompt가 선언한
    # 사용자 관리 신뢰 영역 밖으로 옮길 수 있다. #4028/#4060의 memory_update prompt
    # escape와 같은 방어다. quote=False인 이유는 이 값들이 element-text 위치에만
    # 놓이고(속성 값이 아니다) <, >, &만 탈출에 쓰이기 때문이다. fact 안의 ' 와 "는
    # 그대로 둔다.
    content = html.escape(content, quote=False)
    category = html.escape(category, quote=False)
    if category == "correction" and isinstance(source_error, str) and source_error.strip():
        source_error = html.escape(source_error.strip(), quote=False)
        return f"- [{category} | {confidence:.2f}] {content} (avoid: {source_error})"
    return f"- [{category} | {confidence:.2f}] {content}"


def _escape_summary(value: Any) -> str:
    """사용자가 편집 가능한 context summary를 ``<memory>`` 블록용으로 escape한다.

    context summary(``workContext``/``personalContext``/``topOfMind``와 history 섹션)는
    ``/api/memory`` import로 사용자가 편집할 수 있고 fact와 같은 ``<memory>`` 블록에
    렌더링된다. escape하지 않은 ``</memory>`` 값이 블록을 닫고 뒤 텍스트를 lead-agent
    prompt가 선언한 사용자 관리 신뢰 영역 밖으로 옮길 수 있다.
    ``_format_fact_line``의 escape와 짝을 이루는 방어다(#4097). ``str(...)``은 import가
    드물게 심을 수 있는 비문자열 summary에 대해 기존 f-string 변환 동작을 유지한다.
    ``quote=False``인 이유는 summary가 element-text 위치에만 놓이고(속성 값이 아니다)
    ``<``, ``>``, ``&``만 탈출에 쓰이기 때문이다. ``'``와 ``"``는 그대로 둔다.
    """
    return html.escape(str(value), quote=False)


def _select_fact_lines(
    ranked_facts: list[dict[str, Any]],
    *,
    token_budget: int,
    use_tiktoken: bool,
) -> tuple[list[str], int]:
    """*줄만* 계산하는 token 예산 안에서 포맷된 fact 줄을 greedy하게 고른다.

    이 함수는 의도적으로 **header를 모른다**. fact 줄 자체(줄 사이 ``\\n`` 구분자 포함)만
    센다. ``"Facts:\\n"`` header와 섹션 사이 ``"\\n\\n"`` 구분자의 token은 호출자가 이
    함수를 부르기 *전에* 확보하고, 남은 용량을 *token_budget*으로 넘겨야 한다.

    예산을 넘기는 첫 fact에서 멈추므로 호출자가 미리 정렬한 순서(보통 confidence
    내림차순)가 엄격히 유지된다. 더 짧고 순위가 낮은 fact가 건너뛴 상위 fact를
    앞지르는 일은 없다.

    Args:
        ranked_facts: 호출자가 원하는 기준으로 미리 정렬한 fact 목록.
        token_budget: fact 줄에만 쓸 수 있는 최대 token 수.
        use_tiktoken: 계산에 tiktoken을 쓸지 여부.

    Returns:
        ``(selected_lines, consumed_tokens)``. *consumed_tokens*는 반환된 줄들의
        정확한 token 비용이며 줄 사이 ``\\n`` 구분자는 포함하고 앞의 header는 포함하지
        않는다.
    """
    lines: list[str] = []
    consumed = 0
    for fact in ranked_facts:
        formatted = _format_fact_line(fact)
        if formatted is None:
            continue
        line_text = ("\n" + formatted) if lines else formatted
        line_tokens = _count_tokens(line_text, use_tiktoken=use_tiktoken)
        if consumed + line_tokens > token_budget:
            break
        lines.append(formatted)
        consumed += line_tokens
    return lines, consumed


def _fallback_format_facts(
    valid_facts: list[dict[str, Any]],
    *,
    preceding_section_cost: int,
    max_tokens: int,
    use_tiktoken: bool,
) -> tuple[str, list[str]] | tuple[None, None]:
    """주 경로가 예외를 던졌을 때 쓰는 confidence 단독 순위 계산.

    ``(section_text, fact_lines)`` 튜플을 반환한다. ``section_text``는 포맷된
    ``"Facts:\\n..."`` 섹션 문자열이며 앞의 섹션 구분자는 포함하지 않는다(구분자는
    호출자 몫이다). ``fact_lines``는 facts 블록을 이루는 개별 줄이다. 남는 fact가
    없으면 둘 다 ``None``이다.

    줄을 따로 반환하는 이유는 호출자가 구조 인지 안전 절단에서 이를 추적해,
    fallback fact도 주 경로가 만든 fact와 동일하게 보호되는 접미부로 취급하기
    위해서다.

    *valid_facts*는 주 경로가 이미 걸러 놓은 목록이라 fallback에서 검증을 다시 하지
    않는다. *preceding_section_cost*는 user-context / history 섹션이 이미 쓴 token
    수로, 남은 예산 계산에 쓴다.
    """
    ranked = sorted(valid_facts, key=lambda f: _coerce_confidence(f.get("confidence"), default=0.0), reverse=True)

    header = "Facts:\n"
    overhead = _count_tokens(header, use_tiktoken=use_tiktoken)
    line_budget = max_tokens - preceding_section_cost - overhead
    if line_budget <= 0:
        return None, None

    lines, _ = _select_fact_lines(ranked, token_budget=line_budget, use_tiktoken=use_tiktoken)
    if not lines:
        return None, None
    return header + "\n".join(lines), lines


def format_memory_for_injection(
    memory_data: dict[str, Any],
    max_tokens: int = 2000,
    *,
    use_tiktoken: bool = True,
    guaranteed_categories: list[str] | None = None,
    guaranteed_token_budget: int = 500,
) -> str:
    """system prompt에 주입할 memory 데이터를 포맷한다.

    Args:
        memory_data: memory 데이터 딕셔너리.
        max_tokens: 사용할 최대 token 수(정확도를 위해 tiktoken으로 센다).
        use_tiktoken: ``False``면 모든 token 계산에 tiktoken 대신 네트워크가 필요 없는
            문자 기반 추정을 쓴다(``memory.token_counting`` 설정 참고). 기본값은 ``True``.
        guaranteed_categories: 일반 token 예산과 무관하게 항상 주입해야 하는 fact
            category. 이 fact들은 별도의 *guaranteed_token_budget*에서 가져간다.
            ``None``이거나 비어 있으면 모든 fact가 같은 예산을 두고 경쟁한다(원래 동작).
        guaranteed_token_budget: guaranteed 섹션의 token 상한. 보통은 guaranteed 줄이
            *max_tokens* 안에서 일반 줄을 *밀어내므로* 전체 출력은 ``max_tokens``
            이하로 유지된다. guaranteed 줄만으로 조립 결과가 *max_tokens*를 넘길 때에만
            예산이 실제로 가산되며, 이때 안전 절단 상한이
            ``max_tokens + guaranteed_actual_usage``로 올라가 guaranteed 줄을 보호한다.
            *guaranteed_categories*가 ``None``이거나 비어 있으면 무시된다.

    Returns:
        system prompt 주입용으로 포맷된 memory 문자열.
    """
    if not memory_data:
        return ""

    # 맨 문자열은 명시적으로 거부한다. ``str``를 순회하면 낱글자가 나오고, 의미 없는
    # 글자 frozenset이 조용히 만들어져 경고 없이 guarantee가 꺼진다. 설정 계층 호출자는
    # Pydantic(``list[str]`` 강제)을 거치므로 이 검사는 공개 헬퍼 표면만 보호한다.
    if isinstance(guaranteed_categories, str):
        raise TypeError("guaranteed_categories must be an iterable of strings, not a bare str")
    effective_guaranteed: frozenset[str] = frozenset(c.strip() for c in guaranteed_categories if isinstance(c, str) and c.strip()) if guaranteed_categories else frozenset()

    sections: list[str] = []

    # user context 포맷
    user_data = memory_data.get("user", {})
    if user_data:
        user_sections = []

        work_ctx = user_data.get("workContext", {})
        if work_ctx.get("summary"):
            user_sections.append(f"Work: {_escape_summary(work_ctx['summary'])}")

        personal_ctx = user_data.get("personalContext", {})
        if personal_ctx.get("summary"):
            user_sections.append(f"Personal: {_escape_summary(personal_ctx['summary'])}")

        top_of_mind = user_data.get("topOfMind", {})
        if top_of_mind.get("summary"):
            user_sections.append(f"Current Focus: {_escape_summary(top_of_mind['summary'])}")

        if user_sections:
            sections.append("User Context:\n" + "\n".join(f"- {s}" for s in user_sections))

    # history 포맷
    history_data = memory_data.get("history", {})
    if history_data:
        history_sections = []

        recent = history_data.get("recentMonths", {})
        if recent.get("summary"):
            history_sections.append(f"Recent: {_escape_summary(recent['summary'])}")

        earlier = history_data.get("earlierContext", {})
        if earlier.get("summary"):
            history_sections.append(f"Earlier: {_escape_summary(earlier['summary'])}")

        background = history_data.get("longTermBackground", {})
        if background.get("summary"):
            history_sections.append(f"Background: {_escape_summary(background['summary'])}")

        if history_sections:
            sections.append("History:\n" + "\n".join(f"- {s}" for s in history_sections))

    # ── Facts ────────────────────────────────────────────────────────────────
    #
    # 설계 노트
    # ~~~~~~~~~~~~
    # • ``"Facts:\\n"`` header는 최대 한 번만 출력한다.
    # • guaranteed category fact를 전용 *guaranteed_token_budget*에서 먼저 골라
    #   Facts 블록 앞에 두므로 일반 fact에 밀려나지 않는다. 보통은 guaranteed 줄이
    #   일반 줄을 밀어내 전체 출력이 *max_tokens* 안에 들어간다. guaranteed 줄만으로
    #   출력이 *max_tokens*를 넘길 때에만 예산이 실제로 가산되고, 그에 맞춰 안전 절단
    #   상한도 올라간다.
    # • 일반 fact는 *max_tokens*에서만 가져간다.
    # • token 계산(header, 구분자, 줄)은 전부 호출자인 여기서 한다.
    #   ``_select_fact_lines`` 헬퍼는 header를 모른다.
    # • 주 경로가 예외를 던지면 ``_fallback_format_facts``가 confidence만으로
    #   단일 패스 순위를 매긴다.
    facts_data = memory_data.get("facts", [])
    guaranteed_line_tokens = 0  # 뒤에서 실제 절단 한도를 계산할 때 쓴다
    # facts 블록 마커를 위 ``guaranteed_line_tokens``와 함께 함수 스코프에서 초기화한다.
    # fact가 없어 아래 블록이 실행되지 않아도 맨 아래 구조 인지 절단이 이 값들을 참조할
    # 수 있어야 하기 때문이다. 그렇지 않으면 context/history는 크고 ``facts``는 빈
    # 사용자에서 overflow 경로가 ``UnboundLocalError``를 낸다.
    facts_header = "Facts:\n"
    all_fact_lines: list[str] = []
    if isinstance(facts_data, list) and facts_data:
        # 위에서 만든 섹션(user context, history)의 token 비용.
        base_text = "\n\n".join(sections)
        base_tokens = _count_tokens(base_text, use_tiktoken=use_tiktoken) if base_text else 0

        # try 진입 *전에* 유효한 fact를 미리 걸러 둔다. except 경로가 같은 목록을
        # 그대로 fallback에 넘겨, 뜨거운 prompt 주입 경로에서 검증을 반복하지 않게 한다.
        valid_facts = [f for f in facts_data if isinstance(f, dict) and isinstance(f.get("content"), str) and f.get("content", "").strip()]

        try:
            # 유효한 fact를 guaranteed 그룹과 일반 그룹으로 나눈다.
            # category 필드는 기본값(``or "context"``) 없이 *원본*을 쓴다. 운영자가
            # ``guaranteed_categories=["context"]``로 설정했을 때 category가 없는 레거시
            # fact가 조용히 guaranteed 풀로 승격되지 않게 하기 위해서다. category가 없는
            # fact는 언제나 일반 경로로 넘어간다.
            def _confidence_key(fact: dict[str, Any]) -> float:
                return _coerce_confidence(fact.get("confidence"), default=0.0)

            if effective_guaranteed:

                def _category_match(fact: dict[str, Any]) -> bool:
                    raw = fact.get("category")
                    if not isinstance(raw, str):
                        return False
                    cat = raw.strip()
                    return bool(cat) and cat in effective_guaranteed

                guaranteed = sorted(
                    [f for f in valid_facts if _category_match(f)],
                    key=_confidence_key,
                    reverse=True,
                )
                regular = sorted(
                    [f for f in valid_facts if not _category_match(f)],
                    key=_confidence_key,
                    reverse=True,
                )
            else:
                guaranteed = []
                regular = sorted(valid_facts, key=_confidence_key, reverse=True)

            # ── 1단계: guaranteed 줄 선택 ──────────────────────────
            header_cost = _count_tokens(facts_header, use_tiktoken=use_tiktoken)

            guaranteed_lines: list[str] = []
            if guaranteed:
                guaranteed_line_budget = guaranteed_token_budget
                guaranteed_lines, guaranteed_line_tokens = _select_fact_lines(
                    guaranteed,
                    token_budget=guaranteed_line_budget,
                    use_tiktoken=use_tiktoken,
                )

            # ── 2단계: 일반 줄 선택 ────────────────────────────
            # 일반 fact는 주 예산인 *max_tokens*를 두고 경쟁한다.
            # 이미 소모한 몫을 모두 뺀다:
            #   기본 섹션 + 섹션 구분자 + header + guaranteed 줄
            #   + (둘 다 있을 때) 일반 블록과 guaranteed 블록을 잇는 그룹 사이 ``\n``.
            regular_lines: list[str] = []
            if regular:
                inter_group_newline_tokens = _count_tokens("\n", use_tiktoken=use_tiktoken) if guaranteed_lines else 0
                used_before_regular = base_tokens + header_cost + guaranteed_line_tokens + inter_group_newline_tokens
                regular_line_budget = max_tokens - used_before_regular
                if regular_line_budget > 0:
                    regular_lines, _ = _select_fact_lines(
                        regular,
                        token_budget=regular_line_budget,
                        use_tiktoken=use_tiktoken,
                    )

            # ── "Facts:" 섹션 하나만 출력 ───────────────────────────
            # 앞쪽 섹션 구분자는 여기에 넣지 않는다. 섹션 간 간격은 마지막
            # ``"\n\n".join(sections)``만이 결정하며, 이것이 예전의 ``\n\n`` 중복 버그를
            # 막는다.
            all_fact_lines = guaranteed_lines + regular_lines
            if all_fact_lines:
                section_text = facts_header + "\n".join(all_fact_lines)
                sections.append(section_text)

        except Exception:
            # ── fallback: confidence만으로 순위, 단일 예산 ─────────
            # 분할 / guaranteed 경로의 예기치 못한 오류가 memory 주입 자체를 막아서는
            # 안 된다. 원래의 단일 패스 confidence 순위로 되돌아간다. 뜨거운 fallback
            # 경로에서 검증을 반복하지 않도록 미리 걸러 둔 ``valid_facts``를 재사용한다.
            logger.warning(
                "Memory injection: guaranteed-category path failed, falling back to confidence-only ranking",
                exc_info=True,
            )
            fallback, fallback_lines = _fallback_format_facts(
                valid_facts,
                preceding_section_cost=base_tokens,
                max_tokens=max_tokens,
                use_tiktoken=use_tiktoken,
            )
            if fallback:
                sections.append(fallback)
                # fallback이 만든 줄을 ``all_fact_lines``에 올려, 아래 구조 인지 절단이
                # fallback fact도 보호되는 접미부로 취급하게 한다. 이렇게 하지 않으면
                # user-context가 큰 경우 기존 앞부분 절단 방식이 fallback fact를
                # 조용히 잘라 버린다.
                all_fact_lines = fallback_lines

    if not sections:
        return ""

    result = "\n\n".join(sections)

    token_count = _count_tokens(result, use_tiktoken=use_tiktoken)
    effective_limit = max_tokens + guaranteed_line_tokens
    if token_count > effective_limit:
        # 구조 인지 절단. ``Facts:\n...`` 블록을 *보호되는 접미부*로 취급해,
        # 이 기능이 지키려는 guaranteed category fact가 overflow 시 앞부분 절단으로
        # 조용히 사라지지 않게 한다. 절단 대상은 앞선 user-context / history 섹션뿐이며,
        # Facts 블록을 확보하고 남은 예산을 이들만으로 넘기면 뒤에서부터 잘라낸다.
        # *guaranteed_line_tokens*가 0이면(guaranteed category 미설정이거나 남은 fact가
        # 없는 경우) 식이 ``max_tokens`` 기준의 원래 앞부분 절단으로 환원되므로
        # 하위 호환이 유지된다.
        facts_block = (facts_header + "\n".join(all_fact_lines)) if all_fact_lines else ""
        facts_block_tokens = _count_tokens(facts_block, use_tiktoken=use_tiktoken)
        separator_tokens = _count_tokens("\n\n", use_tiktoken=use_tiktoken)
        budget_for_non_facts = max(
            0,
            effective_limit - facts_block_tokens - (separator_tokens if facts_block else 0),
        )

        # *sections*에서 뒤쪽 Facts 블록을 뺀 앞부분(fact가 아닌 영역)을 만든다.
        preceding_sections = sections[:-1] if all_fact_lines else sections
        preceding = "\n\n".join(preceding_sections)

        if preceding:
            preceding_tokens = _count_tokens(preceding, use_tiktoken=use_tiktoken)
            if preceding_tokens > budget_for_non_facts:
                char_per_token = len(preceding) / max(preceding_tokens, 1)
                target_chars = int(budget_for_non_facts * char_per_token * 0.95)
                preceding = preceding[:target_chars].rstrip() + "\n..."
            result = (preceding + "\n\n" + facts_block) if facts_block else preceding
        else:
            result = facts_block

    return result


def format_conversation_for_update(messages: list[Any]) -> str:
    """memory 갱신 prompt에 넣을 대화 메시지를 포맷한다.

    Args:
        messages: 대화 메시지 목록.

    Returns:
        포맷된 대화 문자열.
    """
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))

        # content가 리스트(멀티모달)일 수 있으므로 처리한다
        if isinstance(content, list):
            text_parts = []
            for p in content:
                if isinstance(p, str):
                    text_parts.append(p)
                elif isinstance(p, dict):
                    text_val = p.get("text")
                    if isinstance(text_val, str):
                        text_parts.append(text_val)
            content = " ".join(text_parts) if text_parts else str(content)

        # 일시적인 파일 경로 정보가 장기 memory에 남지 않도록 human 메시지에서
        # uploaded_files 태그를 제거한다. 제거 후 남는 내용이 없으면(업로드만 있는
        # 메시지) 해당 턴 전체를 건너뛴다.
        if role == "human":
            content = re.sub(r"<(?P<tag>uploaded_files|current_uploads)>[\s\S]*?</(?P=tag)>\n*", "", str(content)).strip()
            if not content:
                continue

        # 아주 긴 메시지는 앞부분(주제 / 도입)과 뒷부분(결론 / "X를 기억해라" 지시)을
        # 남기고 가운데를 버린다. 앞만 자르면 뒤쪽 지시가 사라지지만 앞+뒤를 남기면
        # 둘 다 보존된다. 구분자는 순수 ASCII(< > & 없음)라 아래 html.escape가 그대로
        # 두며, LLM에게 어디가 잘렸는지 알려준다. escape는 절단 후에 하므로 경계가
        # entity를 쪼갤 일이 없다(entity는 escape 후에만 생긴다).
        if len(str(content)) > 1000:
            s = str(content)
            content = s[:500] + "\n...[truncated]...\n" + s[-500:]

        # memory_update prompt의 <conversation> 블록에 넣기 전에 < > &를 escape한다.
        # 이 원본 사용자 턴은 prompt에서 공격자 영향이 가장 큰 입력이라,
        # "</conversation><current_memory>..." 같은 값을 escape하지 않으면 블록을 닫고
        # 추출 LLM용 <current_memory> 권위 섹션을 위조할 수 있다. 같은 템플릿의
        # current_memory 슬롯에 적용한 블록 탈출 방어 #4044, 그리고 <memory> 블록의
        # _escape_summary/_format_fact_line escape(#4097)와 같은 계열이다. 절단 후에
        # escape하므로 끝의 "..."가 entity를 쪼갤 수 없다. content는 element-text
        # 위치에만 놓이고 속성 값이 아니므로 quote=False다.
        content = html.escape(str(content), quote=False)

        if role == "human":
            lines.append(f"User: {content}")
        elif role == "ai":
            lines.append(f"Assistant: {content}")

    return "\n\n".join(lines)
