"""구조화 파싱 전에 LLM 응답 텍스트를 정규화하는 유틸리티."""

from __future__ import annotations

import re

# 완결된 <think>...</think> 블록에 매칭한다(대소문자 무시, 줄바꿈 포함).
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
# 닫히지 않고 남은 <think>에 매칭한다(모델이 생각 도중 max_tokens로 잘린 경우).
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def strip_think_blocks(text: str, *, truncate_unclosed: bool = True) -> str:
    """모델 응답에서 inline reasoning ``<think>`` 블록을 제거한다.

    완결된 ``<think>...</think>`` 블록은 항상 제거한다. 닫히지 않은 ``<think>`` 여는 태그는
    모델이 생각 도중 잘린 것으로 본다. ``truncate_unclosed``가 True면(기본값이며, 뒤쪽 쓰레기
    문자를 버려야 하는 suggestions/goal 같은 JSON 파서가 쓴다) 그 태그에서 텍스트를 자른다.
    출력에 ``<think>`` 문자열을 정당하게 그대로 담을 수 있는 호출자(예: 그 태그를 언급하는 초안을
    다시 쓰는 input polisher)는 ``truncate_unclosed=False``를 넘겨, 뒤 텍스트를 조용히 버리는
    대신 태그를 보존한다.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    if truncate_unclosed:
        open_match = _OPEN_THINK_RE.search(text)
        if open_match:
            text = text[: open_match.start()]
    return text.strip()


def strip_markdown_code_fence(text: str) -> str:
    """감싸고 있는 markdown code fence가 하나 있으면 제거한다."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_response_text(content: object) -> str:
    """흔한 chat model 응답 content 형태에서 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)
