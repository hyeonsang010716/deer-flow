"""파일 변환 유틸리티.

문서 파일(PDF, PPT, Excel, Word)을 Markdown으로 변환한다.

PDF 변환 전략(auto 모드):
  1. pymupdf4llm이 설치돼 있으면 먼저 시도한다. heading 인식이 더 좋고 대부분의 파일에서 더 빠르다.
  2. 출력이 의심스럽게 짧으면(페이지당 < _MIN_CHARS_PER_PAGE, 페이지 수를 알 수 없으면 전체
     < 200자) 이미지 기반으로 보고 MarkItDown으로 fallback한다.
  3. pymupdf4llm이 없으면 MarkItDown을 바로 사용한다(기존 동작).

큰 파일(> ASYNC_THRESHOLD_BYTES)은 event loop를 막지 않도록 asyncio.to_thread()로
thread pool에서 변환한다(#1569 수정).

FastAPI나 HTTP 의존성이 없는 순수 유틸리티 함수들이다.
"""

import asyncio
import logging
from pathlib import Path

from deerflow.config.app_config import get_app_config

# 하위 호환용 re-export. outline 추출은 file_outline.py로 옮겼다.
from deerflow.utils.file_outline import (  # noqa: F401
    MAX_OUTLINE_ENTRIES,
    extract_outline,
)

logger = logging.getLogger(__name__)

# markdown으로 변환해야 하는 파일 확장자
CONVERTIBLE_EXTENSIONS = {
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".doc",
    ".docx",
}

# 이 임계값보다 큰 파일은 background thread에서 변환한다.
# 작은 파일은 동기적으로 1초 안에 끝나며, thread를 띄우면 불필요한 scheduling
# overhead만 늘어난다.
_ASYNC_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MB

# pymupdf4llm이 *페이지당* 이 임계값보다 적은 문자를 만들면 해당 PDF는 이미지 기반이거나
# 암호화된 것으로 보고 MarkItDown으로 fallback한다.
# 근거: 일반 텍스트 PDF는 페이지당 200-2000자를 내지만 이미지 기반 PDF는 0에 가깝다.
# 페이지당 50자면 안전 마진이 충분하다.
# 페이지 수를 알 수 없으면 절대값 200자 검사로 대체한다.
_MIN_CHARS_PER_PAGE = 50


def _pymupdf_output_too_sparse(text: str, file_path: Path) -> bool:
    """pymupdf4llm 출력이 의심스럽게 짧으면(이미지 기반 PDF) True를 반환한다.

    절대 임계값 대신 페이지당 문자 수를 쓰므로, 짧은 문서(적은 페이지·적은 문자)와
    긴 문서(많은 페이지·많은 문자) 모두 올바르게 처리된다.
    """
    chars = len(text.strip())
    doc = None
    pages: int | None = None
    try:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pages = len(doc)
    except Exception:
        pass
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
    if pages is not None and pages > 0:
        return (chars / pages) < _MIN_CHARS_PER_PAGE
    # fallback: 페이지 수를 알 수 없으면 절대 임계값을 쓴다.
    return chars < 200


def _convert_pdf_with_pymupdf4llm(file_path: Path) -> str | None:
    """pymupdf4llm으로 PDF 변환을 시도한다.

    markdown 텍스트를 반환하며, pymupdf4llm이 설치돼 있지 않거나 변환이 실패하면
    (예: 암호화·손상된 PDF) None을 반환한다.
    """
    try:
        import pymupdf4llm
    except ImportError:
        return None

    try:
        return pymupdf4llm.to_markdown(str(file_path))
    except Exception:
        logger.exception("pymupdf4llm failed to convert %s; falling back to MarkItDown", file_path.name)
        return None


def _convert_with_markitdown(file_path: Path) -> str:
    """MarkItDown으로 지원되는 모든 파일을 markdown 텍스트로 변환한다."""
    from markitdown import MarkItDown

    md = MarkItDown()
    return md.convert(str(file_path)).text_content


def _do_convert(file_path: Path, pdf_converter: str) -> str:
    """동기 변환. 직접 호출되거나 asyncio.to_thread를 통해 호출된다.

    Args:
        file_path: 파일 경로.
        pdf_converter: "auto" | "pymupdf4llm" | "markitdown"
    """
    is_pdf = file_path.suffix.lower() == ".pdf"

    if is_pdf and pdf_converter != "markitdown":
        # pymupdf4llm을 먼저 시도한다(auto 또는 명시적 지정).
        pymupdf_text = _convert_pdf_with_pymupdf4llm(file_path)

        if pymupdf_text is not None:
            # pymupdf4llm이 설치돼 있다.
            if pdf_converter == "pymupdf4llm":
                # 명시적 지정이므로 출력 길이와 상관없이 그대로 쓴다.
                return pymupdf_text
            # auto 모드: 출력이 파싱 실패처럼 보이면 fallback한다.
            # 페이지당 문자 수로 이미지 기반 PDF(0에 가까움)와 원래 짧은 문서를 구분한다.
            if not _pymupdf_output_too_sparse(pymupdf_text, file_path):
                return pymupdf_text
            logger.warning(
                "pymupdf4llm produced only %d chars for %s (likely image-based PDF); falling back to MarkItDown",
                len(pymupdf_text.strip()),
                file_path.name,
            )
        # pymupdf4llm이 없거나 fallback이 걸린 경우 → MarkItDown을 쓴다.

    return _convert_with_markitdown(file_path)


async def convert_file_to_markdown(file_path: Path, output_path: Path | None = None) -> Path | None:
    """지원되는 문서 파일을 Markdown으로 변환한다.

    PDF는 두 converter 전략으로 처리한다(모듈 docstring 참고). 큰 파일(> 1 MB)은
    event loop를 막지 않도록 thread pool로 넘긴다.

    Args:
        file_path: 변환할 파일 경로.
        output_path: 생성될 ``.md`` 파일의 목적지(선택). 생략하면 ``file_path``에
            ``.md`` 확장자를 붙여 쓴다. 요청 단위로 파일명 유일성을 관리하는 caller는
            미리 확보한 경로를 넘겨야 동반 markdown이 다른 upload를 덮어쓰지 않는다.

    Returns:
        생성된 .md 파일 경로. 변환이 실패하면 None.
    """
    try:
        pdf_converter = _get_pdf_converter()
        file_size = file_path.stat().st_size

        if file_size > _ASYNC_THRESHOLD_BYTES:
            text = await asyncio.to_thread(_do_convert, file_path, pdf_converter)
        else:
            text = _do_convert(file_path, pdf_converter)

        md_path = output_path if output_path is not None else file_path.with_suffix(".md")
        md_path.write_text(text, encoding="utf-8")

        logger.info("Converted %s to markdown: %s (%d chars)", file_path.name, md_path.name, len(text))
        return md_path
    except Exception as e:
        logger.error("Failed to convert %s to markdown: %s", file_path.name, e)
        return None


# 섹션 heading처럼 보이는 bold 전용 줄을 위한 regex.
# pymupdf4llm이 # Markdown heading이 아니라 **bold**로 렌더링하는 SEC 서류의 구조적
# heading을 대상으로 한다(본문과 같은 폰트 크기라서 bold+대문자 서식으로만 구분된다).
#
# 패턴은 다음을 모두 요구한다:
#   1. 줄 전체가 하나의 **...** 블록일 것(주변에 다른 산문이 없을 것)
#   2. 알려진 구조 키워드로 시작할 것:
#      - ITEM / PART / SECTION (뒤에 번호나 문자가 올 수 있음)
#      - SCHEDULE, EXHIBIT, APPENDIX, ANNEX, CHAPTER

_ALLOWED_PDF_CONVERTERS = {"auto", "pymupdf4llm", "markitdown"}


def _get_uploads_config_value(key: str, default: object) -> object:
    """uploads config에서 값을 읽는다. dict 접근과 속성 접근을 모두 지원한다."""
    cfg = get_app_config()
    uploads_cfg = getattr(cfg, "uploads", None)
    if isinstance(uploads_cfg, dict):
        return uploads_cfg.get(key, default)
    return getattr(uploads_cfg, key, default)


def _get_pdf_converter() -> str:
    """app config에서 pdf_converter 설정을 읽는다. 기본값은 'auto'.

    값을 소문자로 정규화하고 허용 집합에 대해 검증하므로, config.yaml의 'AUTO'나
    'MarkItDown' 같은 값이 조용히 예상 밖 동작으로 흘러가지 않는다.
    """
    try:
        raw = str(_get_uploads_config_value("pdf_converter", "auto")).strip().lower()
        if raw not in _ALLOWED_PDF_CONVERTERS:
            logger.warning("Invalid pdf_converter value %r; falling back to 'auto'", raw)
            return "auto"
        return raw
    except Exception:
        pass
    return "auto"
