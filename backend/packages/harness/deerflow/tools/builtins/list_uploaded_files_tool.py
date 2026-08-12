"""현재 thread에 과거 업로드된 파일을 찾는 도구.

이번 run에서 새로 업로드된 파일만 나열하는 ``<current_uploads>``와 달리, 이 도구는 agent가
이전 턴에 업로드된 파일을 필요할 때 탐색할 수 있게 한다.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import tool
from langgraph.config import get_config

from deerflow.agents.middlewares.input_sanitization_middleware import neutralize_untrusted_tags
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tools.types import Runtime
from deerflow.uploads.manager import is_upload_staging_file
from deerflow.utils.file_outline import extract_outline_for_file

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 20
_MAX_MAX_RESULTS = 100


def _extension_label(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    return neutralize_untrusted_tags(suffix) or "(no extension)"


def _format_omitted_summary(omitted: list[str]) -> str:
    counts = Counter(_extension_label(Path(f)) for f in omitted)
    parts = [f"{count} {ext}" for ext, count in sorted(counts.items())]
    return neutralize_untrusted_tags(", ".join(parts))


def _resolve_thread_id(runtime: Runtime) -> str | None:
    """runtime context 또는 RunnableConfig에서 현재 thread id를 해석한다."""
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id:
        return thread_id

    runtime_config = getattr(runtime, "config", None) or {}
    thread_id = runtime_config.get("configurable", {}).get("thread_id")
    if thread_id:
        return thread_id

    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None


def _resolve_user_id(runtime: Runtime) -> str:
    """현재 user id를 해석한다."""
    from deerflow.runtime.user_context import resolve_runtime_user_id

    return resolve_runtime_user_id(runtime) or get_effective_user_id()


def _list_uploaded_files_impl(
    include_outline: bool | list[str] = False,
    max_results: int = _DEFAULT_MAX_RESULTS,
    runtime: Runtime | None = None,
    *,
    _paths: Any | None = None,
) -> dict:
    """핵심 구현. @tool 래퍼 없이도 테스트할 수 있다."""
    if runtime is None:
        return {"files": [], "message": "No runtime context available."}

    thread_id = _resolve_thread_id(runtime)
    if thread_id is None:
        return {"files": [], "message": "Thread not found."}

    user_id = _resolve_user_id(runtime)
    paths = _paths or get_paths()
    uploads_dir = paths.sandbox_uploads_dir(thread_id, user_id=user_id)

    if not uploads_dir.exists():
        return {"files": [], "message": "No uploads directory for this thread."}

    # 현재 run에서 업로드된 파일 이름 집합을 구해서 제외한다.
    current_run_filenames: set[str] = set()
    try:
        state = runtime.state
        uploaded = state.get("uploaded_files") if isinstance(state, dict) else getattr(state, "uploaded_files", None)
        if isinstance(uploaded, list):
            for entry in uploaded:
                if isinstance(entry, dict) and entry.get("filename"):
                    current_run_filenames.add(entry["filename"])
    except Exception:
        logger.warning(
            "Failed to read uploaded_files from runtime.state; current-run files may appear in list_uploaded_files results",
            exc_info=True,
        )

    # max_results 정규화
    max_results = max(1, min(max_results, _MAX_MAX_RESULTS))

    # include_outline 정규화
    if isinstance(include_outline, bool):
        outline_for_all: bool = include_outline
        outline_filenames: set[str] = set()
    else:
        outline_for_all = False
        outline_filenames = set(include_outline)

    # 과거 파일을 수집한다(mtime 내림차순 정렬).
    # 변환 결과물인 .md 파일(같은 stem의 .md가 아닌 형제 파일이 있는 경우)은 건너뛴다.
    candidates: list[tuple[float, Path, int]] = []
    try:
        # 이름 집합을 만들고 순회하기 위해 파일 entry를 한 번만 수집한다.
        entries = [e for e in os.scandir(uploads_dir) if e.is_file() and not e.is_symlink() and not is_upload_staging_file(e.name)]
        all_names: set[str] = {e.name for e in entries}

        for entry in entries:
            if entry.name in current_run_filenames:
                continue
            # 다른 파일의 변환 결과물인 .md 파일은 건너뛴다.
            # 알려진 한계: 사용자가 report.pdf와 report.md를 직접 함께 업로드하면 .md가
            # "변환 결과물"로 취급되어 숨겨진다. MVP에서는 허용 가능한 수준이다. 이 상황은
            # 변환된 문서와 stem이 겹치는 파일을 업로드해야 발생하므로 드물다.
            if entry.name.endswith(".md"):
                stem = entry.name[:-3]  # ".md" 제거
                non_md_siblings = {n for n in all_names if n != entry.name and Path(n).stem == stem}
                if non_md_siblings:
                    continue
            stat = entry.stat()
            candidates.append((stat.st_mtime, Path(entry.path), stat.st_size))
    except OSError:
        return {"files": [], "message": f"Failed to read uploads directory: {uploads_dir}"}

    if not candidates:
        return {"files": [], "message": "No historical uploaded files in this thread."}

    # mtime 내림차순 정렬(최신 우선)
    candidates.sort(key=lambda item: item[0], reverse=True)

    total_count = len(candidates)
    truncated = total_count > max_results
    visible = candidates[:max_results]
    omitted_paths = [p.name for _, p, _ in candidates[max_results:]]

    files: list[dict] = []
    for _, file_path, st_size in visible:
        filename = file_path.name
        file_info: dict = {
            "filename": neutralize_untrusted_tags(filename),
            "size": st_size,
            "path": neutralize_untrusted_tags(f"/mnt/user-data/uploads/{filename}"),
            "extension": neutralize_untrusted_tags(file_path.suffix),
        }

        should_include_outline = outline_for_all or filename in outline_filenames
        if should_include_outline:
            outline, preview = extract_outline_for_file(file_path)
            if outline:
                file_info["outline"] = [{**entry, "title": neutralize_untrusted_tags(entry["title"])} if "title" in entry else entry for entry in outline]
            if preview:
                file_info["outline_preview"] = [neutralize_untrusted_tags(p) for p in preview]

        files.append(file_info)

    result: dict = {
        "files": files,
        "total_count": total_count,
    }

    if truncated:
        result["truncated"] = True
        result["omitted_summary"] = _format_omitted_summary(omitted_paths)

    if files:
        result["message"] = f"Found {total_count} historical file(s)."
    else:
        result["message"] = "No historical uploaded files in this thread."

    return result


@tool
def list_uploaded_files(
    runtime: Runtime,
    include_outline: Annotated[
        bool | list[str],
        "Control which files get their document outline (headings/preview) returned. "
        "False (default): no outline for any file — just filename, size, and path. "
        "True: include outline/preview for every .md-convertible file. "
        'list of filenames: include outline/preview only for those specific files (e.g. ["report.md", "data.csv"]).',
    ] = False,
    max_results: Annotated[
        int,
        "Maximum number of files to return (default 20, max 100).",
    ] = _DEFAULT_MAX_RESULTS,
) -> dict:
    """이 thread에서 사용할 수 있는 과거 업로드 파일을 조회한다.

    이전 턴에 업로드된 파일을 반환한다 — 현재 run에서 업로드된 파일은 제외된다
    (이미 <current_uploads>에 나열되어 있다).

    이 tool을 사용해야 할 때:
    - 사용자가 이름을 밝히지 않고 이전에 업로드한 파일을 언급할 때(예: "전에 업로드한 그 PDF들 분석해줘")
    - 이 thread에 어떤 파일이 있는지 확인해야 할 때
    - thread에서 작업을 시작하며 사용할 수 있는 데이터를 개괄해야 할 때

    이 tool을 건너뛰어야 할 때:
    - 사용자가 특정 파일을 지목했을 때 — 그 경로로 read_file이나 grep을 바로 사용하라
    - 현재 run에서 업로드된 파일일 때 — 이미 <current_uploads>에 있다
    """
    return _list_uploaded_files_impl(
        include_outline=include_outline,
        max_results=max_results,
        runtime=runtime,
    )
