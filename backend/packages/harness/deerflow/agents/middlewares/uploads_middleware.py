"""현재 run에서 업로드된 파일을 에이전트 context에 주입하는 middleware.

과거 업로드는 더 이상 매 턴 주입하지 않는다. 에이전트가 ``list_uploaded_files`` 도구로
필요할 때 찾는다.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import run_in_executor
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.input_sanitization_middleware import neutralize_untrusted_tags
from deerflow.config.paths import Paths, get_paths
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.uploads.manager import is_upload_staging_file
from deerflow.utils.file_outline import extract_outline_for_file
from deerflow.utils.messages import ORIGINAL_USER_CONTENT_KEY, message_content_to_text

logger = logging.getLogger(__name__)

_MAX_FILES_PER_CONTEXT_SECTION = 10


def _extension_label(file: dict) -> str:
    extension = str(file.get("extension") or Path(str(file.get("filename") or "")).suffix).lower()
    return neutralize_untrusted_tags(extension) or "(no extension)"


def _format_omitted_file_types(files: list[dict]) -> str:
    counts = Counter(_extension_label(file) for file in files)
    parts = [f"{count} {extension}" for extension, count in sorted(counts.items())]
    return neutralize_untrusted_tags(", ".join(parts))


class UploadsMiddlewareState(AgentState):
    """uploads middleware의 state 스키마."""

    uploaded_files: NotRequired[list[dict] | None]


class UploadsMiddleware(AgentMiddleware[UploadsMiddlewareState]):
    """현재 run에서 업로드된 파일을 에이전트 context에 주입하는 middleware.

    현재 메시지의 additional_kwargs.files(업로드 후 프론트엔드가 설정)에서 파일 메타데이터를
    읽어, 마지막 human 메시지 앞에 <current_uploads> 블록을 붙인다. 그래야 모델이 방금 어떤
    파일이 업로드됐는지 안다.

    과거 업로드는 주입하지 않는다. 에이전트가 ``list_uploaded_files`` 도구로 필요할 때 찾는다.
    """

    state_schema = UploadsMiddlewareState

    def __init__(
        self,
        base_dir: str | None = None,
        *,
        max_files_per_context_section: int = _MAX_FILES_PER_CONTEXT_SECTION,
    ):
        """middleware를 초기화한다.

        Args:
            base_dir: thread 데이터의 기본 디렉터리. 기본값은 Paths 해석 결과다.
            max_files_per_context_section: 업로드 파일 prompt 섹션마다 나열할 최대 파일 수.
        """
        super().__init__()
        if max_files_per_context_section < 1:
            raise ValueError("max_files_per_context_section must be at least 1")
        self._paths = Paths(base_dir) if base_dir else get_paths()
        self._max_files_per_context_section = max_files_per_context_section

    def _format_file_entry(self, file: dict, lines: list[str]) -> None:
        """파일 항목 하나(이름, 크기, 경로, 선택적 outline)를 lines에 추가한다.

        사용자에서 온 값(파일명, 경로, outline 제목, 미리보기 텍스트)은
        ``neutralize_untrusted_tags``로 무력화한다. 조작된 파일명이나 문서가 신뢰 영역인
        ``<current_uploads>`` 안에 차단 대상 authority 태그를 심지 못하게 하기 위해서다.
        """
        size_kb = file["size"] / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
        lines.append(f"- {neutralize_untrusted_tags(file['filename'])} ({size_str})")
        lines.append(f"  Path: {neutralize_untrusted_tags(file['path'])}")
        if file.get("selection_reason") == "query_match":
            lines.append("  Selected because: matched the current query.")
        outline = file.get("outline") or []
        if outline:
            truncated = outline[-1].get("truncated", False)
            visible = [e for e in outline if not e.get("truncated")]
            lines.append("  Document outline (use `read_file` with line ranges to read sections):")
            for entry in visible:
                lines.append(f"    L{entry['line']}: {neutralize_untrusted_tags(entry['title'])}")
            if truncated:
                lines.append(f"    ... (showing first {len(visible)} headings; use `read_file` to explore further)")
        else:
            preview = file.get("outline_preview") or []
            if preview:
                lines.append("  No structural headings detected. Document begins with:")
                for text in preview:
                    lines.append(f"    > {neutralize_untrusted_tags(text)}")
            lines.append("  Use `grep` to search for keywords (e.g. `grep(pattern='keyword', path='/mnt/user-data/uploads/')`).")
        lines.append("")

    def _select_files_for_context(
        self,
        files: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """업로드 순서대로 개수를 제한한 context 파일을 반환한다."""
        selected = [dict(f) for f in files[: self._max_files_per_context_section]]
        omitted = [dict(f) for f in files[self._max_files_per_context_section :]]
        return selected, omitted

    def _create_files_message(
        self,
        files: list[dict],
        *,
        omitted_files: list[dict] | None = None,
    ) -> str:
        """현재 run에서 업로드된 파일을 나열하는 메시지를 만든다.

        Args:
            files: 현재 메시지에서 업로드된 파일.
            omitted_files: 상한을 넘어 prompt context에서 제외된 파일.

        Returns:
            <current_uploads> 태그로 감싼 포맷 문자열.
        """
        lines = ["<current_uploads>"]

        lines.append("The following files were uploaded in this message:")
        lines.append("")
        if files:
            for file in files:
                self._format_file_entry(file, lines)
            if omitted_files:
                lines.append(f"... ({len(omitted_files)} more file(s) from this message omitted from this context.)")
                lines.append(f"  Omitted file types: {_format_omitted_file_types(omitted_files)}")
                lines.append("  Use `glob(pattern='**/*', path='/mnt/user-data/uploads/')` to list all uploads.")
                lines.append("  Use `grep(pattern='keyword', path='/mnt/user-data/uploads/')` to search across uploads.")
                lines.append("")
        else:
            lines.append("(empty)")
            lines.append("")

        lines.append("To work with these files:")
        lines.append("- Read from the file first — use the outline line numbers and `read_file` to locate relevant sections.")
        lines.append("- Use `grep` to search for keywords when you are not sure which section to look at")
        lines.append("  (e.g. `grep(pattern='revenue', path='/mnt/user-data/uploads/')`).")
        lines.append("- Use `glob` to find files by name pattern")
        lines.append("  (e.g. `glob(pattern='**/*.md', path='/mnt/user-data/uploads/')`).")
        lines.append("- Only fall back to web search if the file content is clearly insufficient to answer the question.")
        lines.append("</current_uploads>")

        return "\n".join(lines)

    def _files_from_kwargs(self, message: HumanMessage, uploads_dir: Path | None = None) -> list[dict] | None:
        """메시지의 additional_kwargs.files에서 파일 정보를 추출한다.

        프론트엔드는 업로드 성공 후 additional_kwargs.files에 파일 메타데이터를 보낸다.
        각 항목은 filename, size(바이트), path(virtual path), status를 갖는다.

        Args:
            message: 검사할 human 메시지.
            uploads_dir: 파일 존재 확인에 쓰는 실제 uploads 디렉터리.
                         주어지면 파일이 더 이상 없는 항목은 건너뛴다.

        Returns:
            virtual path를 담은 파일 dict 리스트. 필드가 없거나 비어 있으면 None.
        """
        kwargs_files = (message.additional_kwargs or {}).get("files")
        if not isinstance(kwargs_files, list) or not kwargs_files:
            return None

        files = []
        for f in kwargs_files:
            if not isinstance(f, dict):
                continue
            filename = f.get("filename") or ""
            if not filename or Path(filename).name != filename or is_upload_staging_file(filename):
                continue
            if uploads_dir is not None and not (uploads_dir / filename).is_file():
                continue
            files.append(
                {
                    "filename": filename,
                    "size": int(f.get("size") or 0),
                    "path": f"/mnt/user-data/uploads/{filename}",
                    "extension": Path(filename).suffix,
                }
            )
        return files if files else None

    @override
    def before_agent(self, state: UploadsMiddlewareState, runtime: Runtime) -> dict | None:
        """에이전트 실행 전에 현재 run의 업로드를 주입한다.

        현재 메시지의 additional_kwargs.files에 있는 파일만 나열한다. 과거 업로드는
        ``list_uploaded_files``로 필요할 때 찾는다.

        마지막 human 메시지 내용 앞에 <current_uploads> context를 붙인다.
        """
        messages = list(state.get("messages", []))
        if not messages:
            return {"uploaded_files": []}

        last_message_index = len(messages) - 1
        last_message = messages[last_message_index]

        if not isinstance(last_message, HumanMessage):
            return {"uploaded_files": []}

        # 존재 확인용 uploads 디렉터리를 해석한다
        thread_id = (runtime.context or {}).get("thread_id")
        if thread_id is None:
            try:
                from langgraph.config import get_config

                thread_id = get_config().get("configurable", {}).get("thread_id")
            except RuntimeError:
                pass
        uploads_dir = self._paths.sandbox_uploads_dir(thread_id, user_id=resolve_runtime_user_id(runtime)) if thread_id else None

        # 현재 메시지의 additional_kwargs.files에서 새로 업로드된 파일을 가져온다
        new_files = self._files_from_kwargs(last_message, uploads_dir) or []
        if not new_files:
            if (last_message.additional_kwargs or {}).get("files"):
                logger.info(
                    "UploadsMiddleware: files metadata was present but no files were found on disk (thread_id=%s, uploads_dir=%s)",
                    thread_id,
                    uploads_dir,
                )
            # 오래된 uploaded_files를 비운다. 그래야 list_uploaded_files가 직전 턴 이후 과거
            # 업로드가 된 파일을 제외하지 않는다.
            return {"uploaded_files": []}

        context_files, omitted_files = self._select_files_for_context(new_files)

        # context 파일에 outline을 붙인다
        if uploads_dir:
            for file in context_files:
                phys_path = uploads_dir / file["filename"]
                outline, preview = extract_outline_for_file(phys_path)
                file["outline"] = outline
                file["outline_preview"] = preview

        logger.debug(f"Current uploads: {[f['filename'] for f in new_files]}")

        # 파일 메시지를 만들어 마지막 human 메시지 내용 앞에 붙인다
        files_message = self._create_files_message(
            context_files,
            omitted_files=omitted_files if omitted_files else None,
        )

        original_content = last_message.content
        additional_kwargs = dict(last_message.additional_kwargs or {})
        original_user_content = additional_kwargs.get(ORIGINAL_USER_CONTENT_KEY)
        if not isinstance(original_user_content, str):
            if ORIGINAL_USER_CONTENT_KEY in additional_kwargs:
                logger.warning(
                    "UploadsMiddleware replaced non-string %s metadata: type=%s",
                    ORIGINAL_USER_CONTENT_KEY,
                    type(original_user_content).__name__,
                )
            additional_kwargs[ORIGINAL_USER_CONTENT_KEY] = message_content_to_text(original_content)
        if isinstance(original_content, str):
            updated_content = f"{files_message}\n\n{original_content}"
        elif isinstance(original_content, list):
            files_block = {"type": "text", "text": f"{files_message}\n\n"}
            updated_content = [files_block, *original_content]
        else:
            updated_content = original_content

        updated_message = HumanMessage(
            content=updated_content,
            id=last_message.id,
            name=last_message.name,
            additional_kwargs=additional_kwargs,
        )

        messages[last_message_index] = updated_message

        return {
            "uploaded_files": new_files,
            "messages": messages,
        }

    @override
    async def abefore_agent(self, state: UploadsMiddlewareState, runtime: Runtime) -> dict | None:
        """동기 uploads 스캔을 event loop 밖으로 넘기는 async hook.

        ``before_agent``는 blocking 파일시스템 IO(디렉터리 나열, ``stat``, 형제 ``.md`` outline
        읽기)를 한다. graph가 async로 돌면 langgraph가 동기 hook을 event loop 위에서 그대로
        실행하므로, ``run_in_executor``로 worker thread에 넘긴다. ``run_in_executor``는 현재
        context를 복사하므로 LangGraph의 runnable config와 DeerFlow의 요청 ContextVar fallback이
        모두 유지된다. 권위 있는 ``runtime.context["user_id"]`` 채널을 위해 runtime 자체도
        명시적으로 전달한다.
        """
        return await run_in_executor(None, self.before_agent, state, runtime)
