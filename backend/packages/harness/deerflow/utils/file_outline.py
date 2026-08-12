"""공용 문서 outline 추출.

middleware와 ``list_uploaded_files`` 도구가 같은 코드를 쓸 수 있도록
``file_conversion.py``와 ``uploads_middleware.py``에서 분리했다.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# pymupdf4llm이 bold 텍스트를 Markdown # heading으로 승격시키지 못할 때 만들어지는
# bold 구조 heading용 regex(SEC 서류에서 흔하다).
#
# 중국어 heading(第三节...)은 pymupdf4llm이 이미 표준 # heading으로 잡아내므로
# 이 패턴이 필요 없다.
_BOLD_HEADING_RE = re.compile(r"^\*\*((ITEM|PART|SECTION|SCHEDULE|EXHIBIT|APPENDIX|ANNEX|CHAPTER)\b[A-Z0-9 .,\-]*)\*\*\s*$")

# heading이 PDF에서 여러 text span에 걸쳐 있을 때(예: 섹션 번호와 제목이 별도 span)
# pymupdf4llm이 만들어내는 분리된 bold heading용 regex.
# 예:  **1** **Introduction**  또는  **3.2** **Multi-Head Attention**
# 요구 조건:
#   1. 줄 전체가 공백으로 구분된 **...** 블록으로만 이뤄질 것(다른 산문 없음)
#   2. 첫 블록이 섹션 번호일 것(숫자와 점, 예: "1", "3.2", "A.1")
#   3. 두 번째 블록이 순수 숫자/구두점이 아닐 것. **2023** **2022** **2021** 같은 재무 표
#      헤더는 제외하면서 **1** **概述**나 악센트가 붙은 단어 같은 비ASCII 제목은 허용한다
#      ([A-Za-z] 대신 negative lookahead 사용)
#   4. 추가 블록은 최대 2개(총 4개)까지, 내부에 *가 없는 [^*]+로 제한해 regex를 선형으로
#      유지하고 공격자가 통제하는 내용에서 ReDoS를 피한다
_SPLIT_BOLD_HEADING_RE = re.compile(r"^\*\*[\dA-Z][\d\.]*\*\*\s+\*\*(?!\d[\d\s.,\-–—/:()%]*\*\*)[^*]+\*\*(?:\s+\*\*[^*]+\*\*){0,2}\s*$")

# agent context에 주입되는 outline 항목의 최대 개수.
# 아주 긴 문서에서도 prompt 크기를 제한한다.
MAX_OUTLINE_ENTRIES = 50

_OUTLINE_PREVIEW_LINES = 5


def _clean_bold_title(raw: str) -> str:
    """pymupdf4llm의 bold 잔재가 섞일 수 있는 제목 문자열을 정규화한다.

    pymupdf4llm은 인접한 bold span을 하나의 ``**A B**`` 블록 대신 ``**A** **B**``로 내보내기도
    한다. 이 헬퍼는 그런 조각들을 합친 뒤 가장 바깥 ``**...**`` 래퍼를 제거해 caller가 평문을
    받게 한다.

    예시::

        "**Overview**"                       → "Overview"
        "**UNITED STATES** **SECURITIES**"   → "UNITED STATES SECURITIES"
        "plain text"                         → "plain text"  (변경 없음)
    """
    # 인접한 bold span을 합친다: "** **" → " "
    merged = re.sub(r"\*\*\s*\*\*", " ", raw).strip()
    # 문자열 전체가 감싸여 있으면 가장 바깥 **...**를 제거한다.
    if m := re.fullmatch(r"\*\*(.+?)\*\*", merged, re.DOTALL):
        return m.group(1).strip()
    return merged


def extract_outline(md_path: Path) -> list[dict]:
    """Markdown 파일에서 문서 outline(heading)을 추출한다.

    pymupdf4llm이 만들어내는 세 가지 heading 스타일을 인식한다:

    1. 표준 Markdown heading: '#'로 시작하는 줄. 제목이 평문이 되도록 인라인 ``**...**``
       래퍼와 인접한 bold span(``** **``)을 정리한다.

    2. bold 전용 구조 heading: ``**ITEM 1. BUSINESS**``, ``**PART II**`` 등. SEC 서류는
       본문과 같은 폰트 크기에 bold+대문자로 섹션 heading을 쓰므로 pymupdf4llm이 이를
       # heading으로 승격시키지 못한다.

    3. 분리된 bold heading: ``**1** **Introduction**``, ``**3.2** **Attention**``.
       원본 PDF에서 섹션 번호와 제목 텍스트가 별도 span일 때 pymupdf4llm이 이렇게 내보낸다
       (학술 논문에서 흔하다).

    Args:
        md_path: .md 파일 경로.

    Returns:
        title(str), line(int, 1부터 시작) 키를 가진 dict 리스트.
        outline이 MAX_OUTLINE_ENTRIES에서 잘리면 마지막 요소로 sentinel 항목
        ``{"truncated": True}``를 덧붙이므로, caller가 파일을 다시 훑지 않고도 "앞의 N개
        heading만 표시" 힌트를 렌더링할 수 있다.
        파일을 읽을 수 없거나 heading이 없으면 빈 리스트를 반환한다.
    """
    outline: list[dict] = []
    try:
        with md_path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue

                # 스타일 1: 표준 Markdown heading
                if stripped.startswith("#"):
                    title = _clean_bold_title(stripped.lstrip("#").strip())
                    if title:
                        outline.append({"title": title, "line": lineno})

                # 스타일 2: SEC 구조 키워드를 담은 단일 bold 블록
                elif m := _BOLD_HEADING_RE.match(stripped):
                    title = m.group(1).strip()
                    if title:
                        outline.append({"title": title, "line": lineno})

                # 스타일 3: 분리된 bold heading — **<num>** **<title>**
                # regex가 이미 블록 최대 4개와 두 번째 블록이 숫자가 아님을 강제한다.
                elif _SPLIT_BOLD_HEADING_RE.match(stripped):
                    title = " ".join(re.findall(r"\*\*([^*]+)\*\*", stripped))
                    if title:
                        outline.append({"title": title, "line": lineno})

                if len(outline) > MAX_OUTLINE_ENTRIES:
                    outline.pop()
                    outline.append({"truncated": True})
                    break
    except Exception:
        return []

    return outline


def extract_outline_for_file(file_path: Path) -> tuple[list[dict], list[str]]:
    """*file_path*에 대한 문서 outline과 대체 preview를 반환한다.

    upload 변환 파이프라인이 만든 형제 ``<stem>.md`` 파일을 찾는다.

    Returns:
        (outline, preview):
        - outline: ``{title, line}`` dict 리스트(선택적 sentinel 포함). heading이 없거나
          .md가 없으면 비어 있다.
        - preview: .md의 앞쪽 비어 있지 않은 몇 줄. outline이 비었을 때 agent가 최소한의
          context를 갖도록 내용 anchor로 쓴다. outline이 비어 있지 않으면 비어 있다
          (fallback이 필요 없다).
    """
    md_path = file_path.with_suffix(".md")
    if not md_path.is_file():
        return [], []

    outline = extract_outline(md_path)
    if outline:
        logger.debug("Extracted %d outline entries from %s", len(outline), file_path.name)
        return outline, []

    # outline이 비었으므로 앞쪽 비어 있지 않은 몇 줄을 내용 preview로 읽는다.
    preview: list[str] = []
    try:
        with md_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    preview.append(stripped)
                if len(preview) >= _OUTLINE_PREVIEW_LINES:
                    break
    except Exception:
        logger.debug("Failed to read preview lines from %s", md_path, exc_info=True)
    return [], preview
