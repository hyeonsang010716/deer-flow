"""langchain-mcp-adapters로 MCP 도구를 로드하고 stdio session pooling을 적용한다."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config

from deerflow.config.extensions_config import ExtensionsConfig, resolve_effective_mcp_routing
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, Paths, get_paths
from deerflow.constants import DEFAULT_MCP_SESSION_INIT_TIMEOUT
from deerflow.mcp.client import build_servers_config
from deerflow.mcp.oauth import build_oauth_tool_interceptor, get_initial_oauth_headers
from deerflow.mcp.session_pool import get_session_pool
from deerflow.reflection import resolve_variable
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.mcp_metadata import tag_mcp_routing, tag_mcp_tool
from deerflow.tools.sync import make_sync_tool_wrapper
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

# MCP tool 이름은 외부(잠재적으로 적대적이거나 침해된) 서버에서 그대로 전달된다. tool 이름은
# 어디까지나 함수 식별자이므로, provider의 function-calling API가 bind 시점에 동일한 문자
# 집합으로 검증한다. 하지만 deferred(tool_search) MCP tool은 bind에서 제외되므로 그 이름에는
# provider 검증이 전혀 돌지 않는다 — system prompt 문자열 안에만 존재하기 때문에, 조작된
# 이름(개행, markdown, 꺾쇠 괄호)이 framework prompt 구조를 위조할 수 있다. 로드 경계에서
# 정규화하면 bind된 이름과 deferred 이름이 모두 같은 안전한 식별자 문자 집합으로 제한되며,
# 이는 skill 이름이 받는 로드 시점 검증(skills/storage/skill_storage.py)과 같은 방식이다.
_VALID_MCP_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

# stdio MCP subprocess의 temp dir로 쓰는 thread workspace 하위 디렉터리. 프로세스 temp dir을
# (cwd와 함께) 여기에 고정하면 ``os.tmpdir()`` / ``tempfile.gettempdir()``에 쓰는 도구의 출력이
# 도달할 수 없는 host temp 경로가 아니라, sandbox/artifact API가 해석할 수 있는 mount된
# user-data 트리 안에 떨어진다.
_MCP_TMP_SUBDIR = ".mcp/tmp"

# MCP 서버가 반환한 자유 텍스트에 포함된 로컬 파일 참조를 매칭한다. 일부 서버(특히 Playwright의
# ``browser_take_screenshot``)는 저장한 파일을 ``ResourceLink`` 블록이 아니라 text/markdown
# 링크로만 알려준다. 그 참조는 절대 경로, ``file://`` URI, 또는 서버 프로세스 cwd 기준 상대
# 경로(예: ``temp/page.yml``, ``./shot.png``)일 수 있다. 각 매치는 thread의 user-data 트리 안에
# 실재하는 파일로 해석될 때만 재작성되므로, 과하게 잡힌 매치는 무해하다(그대로 둔다).
_LOCAL_PATH_IN_TEXT_RE = re.compile(r"(?:file://)?/[^\s'\"<>|*?]+|(?:\.{0,2}/|[\w.-]+/)[^\s'\"<>|*?]+")

# 경로의 일부가 아니라 문장 부호/마크업에 해당하는 후행 문자들.
_TEXT_PATH_TRAILING_CHARS = ".,;:!?)]}>\"'`"

_FILE_SNAPSHOT = dict[Path, tuple[int, int]]


def _local_path_from_uri(uri: str, *, base_dir: Path | None = None) -> Path | None:
    """*uri*가 로컬 파일을 가리키면 절대 경로 ``Path``를, 아니면 ``None``을 반환한다.

    맨 경로와 ``file://`` URI를 받는다. 원격 URI(``http``/``https``/``data``/...)는 ``None``을
    반환해 호출자가 그대로 두게 한다. 상대 경로는 *base_dir*가 주어졌을 때만 해석한다.
    """
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
    except ValueError:
        return None
    if parsed.scheme == "file":
        raw = unquote(parsed.path)
    elif parsed.scheme == "":
        raw = uri
    else:
        return None
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        if base_dir is None:
            return None
        path = base_dir / path
    return path


def _local_uri_to_virtual_path(
    uri: str,
    *,
    thread_id: str,
    user_id: str,
    source_base_dir: Path | None = None,
) -> str | None:
    """로컬 파일 참조를 ``/mnt/user-data/...`` virtual path로 변환한다.

    stdio MCP 서버는 cwd와 temp dir이 thread의 mount된 user-data 트리 안에 고정된 채로
    실행되므로(:func:`_make_session_pool_tool` 참고), 서버가 만든 파일은 이미
    sandbox/artifact API가 서빙할 수 있는 위치에 있다 — 빠진 것은 DeerFlow의 나머지가 파일을
    지칭하는 virtual prefix뿐이다. 여기서는 그 host→virtual 매핑만 결정적으로 수행한다. 복사도,
    신뢰 루트 목록도 없고, thread 자기 트리 바깥의 파일을 노출하지도 않는다.

    URI가 원격이거나, 해석할 수 없거나, 이 thread의 user-data 트리 바깥을 가리키거나, 실재하지
    않는 파일이면 ``None``을 반환해 호출자가 참조를 그대로 두게 한다. 상대 참조는
    *source_base_dir*(서버의 cwd) 기준으로 해석한다.
    """
    src = _local_path_from_uri(uri, base_dir=source_base_dir)
    if src is None:
        return None

    try:
        real = src.resolve()
        if not real.is_file():
            return None
    except OSError:
        return None

    try:
        user_data_root = get_paths().sandbox_user_data_dir(thread_id, user_id=user_id).resolve()
    except OSError:
        return None

    try:
        relative = real.relative_to(user_data_root)
    except ValueError:
        # 파일이 이 thread의 user-data mount 바깥에 있어 virtual path로 표현할 수 없으므로
        # 원래 참조를 그대로 둔다.
        logger.debug("MCP path rewrite skipped outside user-data tree: %s", real)
        return None

    virtual_path = f"{VIRTUAL_PATH_PREFIX}/{relative.as_posix()}"
    logger.debug("MCP path rewrite: %s -> %s", real, virtual_path)
    return virtual_path


def _snapshot_workspace_files(root: Path) -> _FILE_SNAPSHOT:
    """*root* 아래 일반 파일들의 가벼운 snapshot을 반환한다."""
    snapshot: _FILE_SNAPSHOT = {}
    if not root.exists():
        return snapshot

    try:
        candidates = root.rglob("*")
        for path in candidates:
            try:
                stat = path.stat()
            except OSError:
                continue
            if path.is_file():
                snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return snapshot
    return snapshot


def _changed_workspace_files(root: Path, before: _FILE_SNAPSHOT) -> list[Path]:
    """*before* 이후 *root* 아래에서 생성되거나 수정된 파일들을 반환한다."""
    after = _snapshot_workspace_files(root)
    return [path for path, signature in after.items() if before.get(path) != signature]


def _prepare_stdio_workspace(paths: Paths, *, thread_id: str, user_id: str) -> tuple[Path, Path, _FILE_SNAPSHOT]:
    """고정된 stdio MCP subprocess를 위해 thread workspace를 준비한다.

    동기 파일시스템 작업(디렉터리 생성, temp dir 준비, 호출 전 snapshot)을 한 헬퍼로 묶어
    호출자가 :func:`asyncio.to_thread`로 event loop 바깥에서 돌릴 수 있게 한다. workspace cwd,
    고정된 temp dir, 호출 전 파일 snapshot을 반환한다.
    """
    paths.ensure_thread_dirs(thread_id, user_id=user_id)
    source_base_dir = paths.sandbox_work_dir(thread_id, user_id=user_id)
    tmp_dir = source_base_dir / _MCP_TMP_SUBDIR
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.chmod(0o700)
    except OSError:
        logger.warning("Failed to prepare MCP temp dir: %s", tmp_dir, exc_info=True)
    before_files = _snapshot_workspace_files(source_base_dir)
    return source_base_dir, tmp_dir, before_files


def _result_has_text_content(call_tool_result: Any) -> bool:
    """MCP 결과에 text content가 하나라도 있으면 ``True``를 반환한다.

    호출 후 snapshot diff는 자유 텍스트 안의 맨 파일명 대조에만 쓰인다. 결과에 text 블록이 없으면
    재작성할 것도 없으므로 호출자는 두 번째 재귀 순회를 통째로 건너뛸 수 있다.
    """
    from mcp.types import EmbeddedResource, TextContent, TextResourceContents

    content = getattr(call_tool_result, "content", None)
    if not content:
        return False
    for item in content:
        if isinstance(item, TextContent):
            return True
        if isinstance(item, EmbeddedResource) and isinstance(item.resource, TextResourceContents):
            return True
    return False


def _rewrite_unique_bare_filenames(
    text: str,
    *,
    changed_files: Iterable[Path],
    thread_id: str,
    user_id: str,
    source_base_dir: Path | None = None,
) -> str:
    """이번 호출이 유일한 매치를 만들어 냈을 때만 맨 파일명을 재작성한다.

    ``Saved as page-2026.yml`` 같은 응답은 구조적으로 경로가 아니다. 안전하게 해석하는 유일한
    방법은 그 파일명을 바로 이 tool call이 생성/수정한 파일들과 대조해서, basename이 이 thread의
    mount된 user-data 트리 안 파일 정확히 하나에 대응할 때만 재작성하는 것이다.
    """
    candidates: dict[str, list[str]] = {}
    for path in changed_files:
        virtual_path = _local_uri_to_virtual_path(
            str(path),
            thread_id=thread_id,
            user_id=user_id,
            source_base_dir=source_base_dir,
        )
        if virtual_path is None:
            continue
        candidates.setdefault(path.name, []).append(virtual_path)

    unique = {name: paths[0] for name, paths in candidates.items() if len(set(paths)) == 1}
    if not unique:
        if candidates:
            logger.debug("MCP bare filename rewrite skipped: no unique candidate in %s", sorted(candidates))
        else:
            logger.debug("MCP bare filename rewrite skipped: no snapshot candidates")
        return text

    rewritten = text
    for name in sorted(unique, key=len, reverse=True):
        # 더 긴 경로/단어 안에서는 재작성하지 않는다. 문장 끝 마침표는 허용하지만 ".bak"이나
        # 다른 경로 세그먼트는 허용하지 않는다.
        pattern = re.compile(rf"(?<![\w./-]){re.escape(name)}(?!(?:[\w/-]|\.[\w]))")
        rewritten_text, count = pattern.subn(unique[name], rewritten)
        if count:
            logger.debug("MCP bare filename rewrite: %s -> %s", name, unique[name])
        rewritten = rewritten_text
    return rewritten


def _rewrite_local_paths_in_text(
    text: str,
    *,
    thread_id: str,
    user_id: str,
    source_base_dir: Path | None = None,
    changed_files: Iterable[Path] | None = None,
) -> str:
    """자유 텍스트에서 발견된 로컬 파일 참조를 best-effort로 재작성한다.

    일부 MCP 서버(특히 Playwright의 ``browser_take_screenshot``)는 저장한 파일을
    ``ResourceLink``가 아니라 자유 텍스트로만 알려준다 — 예: ``Took the screenshot and saved it
    as temp/page-2026.png``. 자유 텍스트는 신뢰할 만한 protocol이 아니므로 일부러 보수적으로
    처리한다. 모든 후보 토큰을 :func:`_local_uri_to_virtual_path`에 넘기고, 그 함수는 이
    thread의 user-data 트리 안 실재 파일로 해석될 때만 재작성한다. 실제 경로가 아니거나 다른 곳을
    가리키는 토큰은 그대로 두므로, 과하게 잡힌 regex 매치도 해를 끼치지 않는다.
    """
    translated_by_source: dict[str, str | None] = {}

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        # 경로가 문장 끝에 올 수 있으므로("saved as temp/a.png.") 후행 문장 부호를 떼어 두었다가
        # (재작성됐을 수도 있는) 경로 뒤에 다시 붙인다.
        stripped = token.rstrip(_TEXT_PATH_TRAILING_CHARS)
        trailing = token[len(stripped) :]
        if stripped not in translated_by_source:
            translated_by_source[stripped] = _local_uri_to_virtual_path(
                stripped,
                thread_id=thread_id,
                user_id=user_id,
                source_base_dir=source_base_dir,
            )
        rewritten = translated_by_source[stripped]
        if rewritten is None:
            return token
        return f"{rewritten}{trailing}"

    rewritten = _LOCAL_PATH_IN_TEXT_RE.sub(_replace, text)
    if changed_files is None:
        return rewritten
    return _rewrite_unique_bare_filenames(
        rewritten,
        changed_files=changed_files,
        thread_id=thread_id,
        user_id=user_id,
        source_base_dir=source_base_dir,
    )


def _extract_thread_id(runtime: Runtime | None) -> str:
    """주입된 tool runtime 또는 LangGraph config에서 thread_id를 추출한다."""
    if runtime is not None:
        tid = runtime.context.get("thread_id") if runtime.context else None
        if tid is not None:
            return str(tid)
        config = runtime.config or {}
        tid = config.get("configurable", {}).get("thread_id")
        if tid is not None:
            return str(tid)

    try:
        tid = get_config().get("configurable", {}).get("thread_id")
        return str(tid) if tid is not None else "default"
    except RuntimeError:
        return "default"


def _convert_call_tool_result(
    call_tool_result: Any,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    source_base_dir: Path | None = None,
    changed_files: Iterable[Path] | None = None,
) -> Any:
    """MCP CallToolResult를 LangChain ``content_and_artifact`` 형식으로 변환한다.

    private 심볼 ``langchain_mcp_adapters.tools._convert_call_tool_result``에 의존하지 않고
    adapter와 동일한 변환 로직을 구현한다.

    ``thread_id``와 ``user_id``가 주어지면, ``ResourceLink`` 블록이나 평문 텍스트가 참조하는
    로컬 파일(예: Playwright MCP가 저장한 스크린샷)의 참조를 host 경로에서
    ``/mnt/user-data/...`` virtual path로 변환해 sandbox와 artifact API가 해석할 수 있게 한다.
    파일 자체는 복사하지 않는다 — stdio 서버는 cwd/temp가 mount된 트리 안에 고정된 채로
    실행되므로 이미 서빙 가능한 위치에 있다. 원격 URI와 thread의 user-data 트리 바깥 파일은
    그대로 둔다.
    """
    from langchain_core.messages import ToolMessage
    from langchain_core.messages.content import create_file_block, create_image_block, create_text_block
    from langchain_core.tools import ToolException
    from mcp.types import EmbeddedResource, ImageContent, ResourceLink, TextContent, TextResourceContents

    # ToolMessage는 그대로 통과시킨다(interceptor short-circuit).
    if isinstance(call_tool_result, ToolMessage):
        return call_tool_result, None

    # langgraph가 설치돼 있으면 LangGraph Command도 그대로 통과시킨다.
    try:
        from langgraph.types import Command

        if isinstance(call_tool_result, Command):
            return call_tool_result, None
    except ImportError:
        # langgraph는 선택 의존성이다. 없으면 표준 MCP content 변환을 그대로 진행한다.
        pass

    def _resolve_link_url(uri: str) -> str:
        if thread_id is None or user_id is None:
            return uri
        rewritten = _local_uri_to_virtual_path(uri, thread_id=thread_id, user_id=user_id, source_base_dir=source_base_dir)
        return rewritten if rewritten is not None else uri

    def _resolve_text(text: str) -> str:
        # Playwright 같은 서버는 저장한 파일을 평문 텍스트로만 알려주고 붙잡을 ResourceLink를
        # 주지 않는다. 텍스트에서 로컬 경로를 찾아 변환해, 생성된 파일을 sandbox/artifact API로
        # 읽을 수 있게 한다.
        if thread_id is None or user_id is None:
            return text
        return _rewrite_local_paths_in_text(
            text,
            thread_id=thread_id,
            user_id=user_id,
            source_base_dir=source_base_dir,
            changed_files=changed_files,
        )

    # MCP content 블록을 LangChain content 블록으로 변환한다.
    lc_content = []
    for item in call_tool_result.content:
        if isinstance(item, TextContent):
            lc_content.append(create_text_block(text=_resolve_text(item.text)))
        elif isinstance(item, ImageContent):
            lc_content.append(create_image_block(base64=item.data, mime_type=item.mimeType))
        elif isinstance(item, ResourceLink):
            mime = item.mimeType or None
            url = _resolve_link_url(str(item.uri))
            if mime and mime.startswith("image/"):
                lc_content.append(create_image_block(url=url, mime_type=mime))
            else:
                lc_content.append(create_file_block(url=url, mime_type=mime))
        elif isinstance(item, EmbeddedResource):
            from mcp.types import BlobResourceContents

            res = item.resource
            if isinstance(res, TextResourceContents):
                lc_content.append(create_text_block(text=_resolve_text(res.text)))
            elif isinstance(res, BlobResourceContents):
                mime = res.mimeType or None
                if mime and mime.startswith("image/"):
                    lc_content.append(create_image_block(base64=res.blob, mime_type=mime))
                else:
                    lc_content.append(create_file_block(base64=res.blob, mime_type=mime))
            else:
                lc_content.append(create_text_block(text=str(res)))
        else:
            lc_content.append(create_text_block(text=str(item)))

    if call_tool_result.isError:
        error_parts = [item["text"] for item in lc_content if isinstance(item, dict) and item.get("type") == "text"]
        raise ToolException("\n".join(error_parts) if error_parts else str(lc_content))

    artifact = None
    if call_tool_result.structuredContent is not None:
        artifact = {"structured_content": call_tool_result.structuredContent}

    return lc_content, artifact


def _resolve_session_init_timeout(server_cfg: Any) -> float | None:
    """*server_cfg*에 적용될 실제 session-init timeout을 반환한다.

    ``None``(명시적 opt-out)은 ``None``으로 유지한다. 그 외 숫자가 아닌 값은 ``asyncio.wait_for``에
    그대로 넘기거나(예외가 난다) 조용히 상한을 없애는 대신 기본값으로 되돌린다. 실제 config는
    pydantic이 float를 보장하지만 테스트에서 mock으로 만든 config는 무엇이든 줄 수 있고, 이
    fallback이 hang 방지를 유지한다.
    """
    value = server_cfg.session_init_timeout if server_cfg is not None else DEFAULT_MCP_SESSION_INIT_TIMEOUT
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MCP_SESSION_INIT_TIMEOUT
    return float(value)


def _make_session_pool_tool(
    tool: BaseTool,
    server_name: str,
    connection: dict[str, Any],
    tool_interceptors: list[Any] | None = None,
    tool_call_timeout: float | None = None,
    session_init_timeout: float | None = None,
    tool_name_prefix: bool = True,
) -> BaseTool:
    """MCP tool을 감싸서 pool의 persistent session을 재사용하게 한다.

    호출마다 session을 만드는 대신 ``(server_name, user_id:thread_id)`` 범위로 pool이 관리하는
    session을 쓴다. 덕분에 stateful MCP 서버(예: Playwright)는 같은 thread 안의 tool call들
    사이에서 state를 유지하면서도 사용자별로 격리된다.

    설정된 ``tool_interceptors``(OAuth, custom)는 그대로 보존되어, pool session을 호출하기 전에
    매 호출마다 적용된다.
    """
    # adapter가 붙인 prefix만 떼어낸다. prefix를 쓰지 않는 서버가 이름 자체가
    # ``<server_name>_``로 시작하는 tool을 노출할 수도 있기 때문이다.
    original_name = tool.name
    prefix = f"{server_name}_"
    if tool_name_prefix and original_name.startswith(prefix):
        original_name = original_name[len(prefix) :]

    pool = get_session_pool()

    async def call_with_persistent_session(
        runtime: Runtime | None = None,
        **arguments: Any,
    ) -> Any:
        thread_id = _extract_thread_id(runtime)
        user_id = resolve_runtime_user_id(runtime)
        # pool session의 범위를 user *와* thread로 잡는다. 파일시스템 격리가
        # (user_id, thread_id) 단위이므로, thread_id만 쓰면 thread_id가 겹치는 두 사용자가 하나의
        # stateful MCP session을 공유하게 된다.
        scope_key = f"{user_id}:{thread_id}"
        session_connection = dict(connection)
        # cwd/temp 고정과 workspace snapshot은 실제 파일시스템에 쓰는 로컬 subprocess로 도는
        # stdio 서버에만 의미가 있다. SSE/HTTP 서버는 고정할 로컬 cwd가 없으므로 파일시스템
        # 작업을 통째로 건너뛴다(불필요한 디렉터리 생성과 재귀 순회를 피한다).
        is_stdio = session_connection.get("transport", "stdio") == "stdio"
        source_base_dir: Path | None = None
        process_cwd: Path | None = None
        before_files: _FILE_SNAPSHOT | None = None
        if is_stdio:
            paths = get_paths()
            # 동기 파일시스템 준비(디렉터리 생성, temp dir 설정, 호출 전 snapshot)를 묶어
            # event loop 바깥에서 실행한다 — snapshot이 workspace 전체를 순회하므로 그대로 두면
            # event loop를 막는다.
            source_base_dir, tmp_dir, before_files = await asyncio.to_thread(_prepare_stdio_workspace, paths, thread_id=thread_id, user_id=user_id)
            # stdio MCP 서버는 상대 출력 링크를 자기 프로세스 cwd 기준으로 해석한다. 그 cwd를
            # thread의 mount된 user-data 트리 안에 두어야 Playwright 같은 도구가 만든 파일이
            # sandbox/artifact API가 서빙할 수 있는 위치에 떨어지고, 그 참조도 virtual path로
            # 변환할 수 있다.
            configured_cwd = session_connection.get("cwd", str(source_base_dir))
            session_connection["cwd"] = str(configured_cwd)
            process_cwd = Path(configured_cwd)
            # subprocess temp dir도 같은 mount 트리 아래로 고정한다. 그러면 OS temp dir을
            # 기본으로 쓰는 도구들(Node의 os.tmpdir(), Python의 tempfile, 다수의 CLI)이 도달할 수
            # 없는 host 경로 대신 user-data 안에 쓴다 — cwd 고정의 도구 비의존적 대응물이다.
            # 운영자가 준 env는 교체하지 않고 병합한다.
            session_env = dict(session_connection.get("env") or {})
            session_env.setdefault("TMPDIR", str(tmp_dir))
            session_env.setdefault("TMP", str(tmp_dir))
            session_env.setdefault("TEMP", str(tmp_dir))
            session_connection["env"] = session_env
        if session_init_timeout is not None:
            # 여기서의 취소는 안전하다. 생성 도중 멈춘 session의 teardown은
            # MCPSessionPool.get_session이 책임진다(close를 알리고 owner task의 __aexit__이 자기
            # task에서 실행되기를 기다린다). 따라서 멈춘 서버가 session을 누수시키거나 turn을
            # 막을 수 없다.
            try:
                session = await asyncio.wait_for(
                    pool.get_session(server_name, scope_key, session_connection),
                    timeout=session_init_timeout,
                )
            except TimeoutError:
                # discovery timeout과 같은 로그 레벨로 노출한다. tool call은 여전히 모델이
                # 반응할 수 있는 TimeoutError로 실패하지만, 운영자는 멈춘 MCP session 때문에
                # 생긴 tool call 실패를 진단하려면 WARNING이 필요하다.
                logger.warning(
                    "MCP session initialization for server '%s' timed out after %.1fs",
                    server_name,
                    session_init_timeout,
                )
                raise
        else:
            session = await pool.get_session(server_name, scope_key, session_connection)

        # 공통 call_tool kwargs를 한 번만 만든다. 필요할 때만 키를 추가해서, 정확한 인자를
        # 검증하는 기존 호출부가 영향을 받지 않게 한다.
        call_kwargs: dict[str, Any] = {}
        if tool_call_timeout:
            call_kwargs["read_timeout_seconds"] = timedelta(seconds=tool_call_timeout)

        if tool_interceptors:
            from langchain_mcp_adapters.interceptors import MCPToolCallRequest

            async def base_handler(request: MCPToolCallRequest) -> Any:
                # interceptor가 주입한 header를 MCP call meta로 전달해 stdio MCP 호출에서도
                # 보존한다.
                kwargs = dict(call_kwargs)
                if request.headers:
                    if isinstance(request.headers, Mapping):
                        kwargs["meta"] = {"headers": dict(request.headers)}
                    else:
                        logger.warning("Ignoring MCP interceptor headers with unsupported type: %s", type(request.headers).__name__)
                return await session.call_tool(
                    request.name,
                    request.args,
                    **kwargs,
                )

            handler = base_handler
            for interceptor in reversed(tool_interceptors):
                outer = handler

                async def wrapped(req: Any, _i: Any = interceptor, _h: Any = outer) -> Any:
                    return await _i(req, _h)

                handler = wrapped

            request = MCPToolCallRequest(
                name=original_name,
                args=arguments,
                server_name=server_name,
                runtime=runtime,
            )
            call_tool_result = await handler(request)
        else:
            call_tool_result = await session.call_tool(
                original_name,
                arguments,
                **call_kwargs,
            )

        # 호출 후 snapshot diff는 자유 텍스트 안의 맨 파일명 대조에만 쓰이므로, 재작성할 text
        # content가 없으면 두 번째 재귀 순회를 건너뛴다. diff와 _convert_call_tool_result 안의
        # 토큰별 경로 해석 모두 파일시스템을 건드리므로 event loop 바깥에서 실행한다.
        changed_files: list[Path] | None = None
        if is_stdio and before_files is not None and _result_has_text_content(call_tool_result):
            changed_files = await asyncio.to_thread(_changed_workspace_files, source_base_dir, before_files)
        return await asyncio.to_thread(
            _convert_call_tool_result,
            call_tool_result,
            thread_id=thread_id,
            user_id=user_id,
            source_base_dir=process_cwd,
            changed_files=changed_files,
        )

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=call_with_persistent_session,
        response_format="content_and_artifact",
        metadata=tool.metadata,
    )


async def get_mcp_tools() -> list[BaseTool]:
    """활성화된 모든 MCP 서버의 tool을 가져온다.

    stdio transport를 쓰는 tool은 persistent-session 로직으로 감싸서, 같은 thread 안의 연속 호출이
    동일한 MCP session을 재사용하게 한다. HTTP/SSE tool은 task 간 TaskGroup 정리 오류를 피하려고
    감싸지 않고 그대로 반환한다.

    Returns:
        활성화된 모든 MCP 서버에서 온 LangChain tool 목록.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed. Install it to enable MCP tools: pip install langchain-mcp-adapters")
        return []

    # NOTE: 항상 디스크에서 최신 설정을 읽으려고 get_extensions_config() 대신
    # ExtensionsConfig.from_file()을 쓴다. 그래야 (별도 프로세스로 도는) Gateway API를 통한
    # 변경이 MCP tool 초기화 시점에 즉시 반영된다.
    extensions_config = ExtensionsConfig.from_file()
    servers_config = build_servers_config(extensions_config)

    if not servers_config:
        logger.info("No enabled MCP servers configured")
        return []

    try:
        # multi-server MCP client를 만든다
        logger.info(f"Initializing MCP client with {len(servers_config)} server(s)")

        # 서버 연결(tool discovery / session init)에 쓸 초기 OAuth header를 주입한다
        initial_oauth_headers = await get_initial_oauth_headers(extensions_config)
        for server_name, auth_header in initial_oauth_headers.items():
            if server_name not in servers_config:
                continue
            if servers_config[server_name].get("transport") in ("sse", "http"):
                existing_headers = dict(servers_config[server_name].get("headers", {}))
                existing_headers["Authorization"] = auth_header
                servers_config[server_name]["headers"] = existing_headers

        tool_interceptors: list[Any] = []
        oauth_interceptor = build_oauth_tool_interceptor(extensions_config)
        if oauth_interceptor is not None:
            tool_interceptors.append(oauth_interceptor)

        # extensions_config.json에 선언된 custom interceptor를 로드한다.
        # 형식: "mcpInterceptors": ["pkg.module:builder_func", ...]
        raw_interceptor_paths = (extensions_config.model_extra or {}).get("mcpInterceptors")
        if isinstance(raw_interceptor_paths, str):
            raw_interceptor_paths = [raw_interceptor_paths]
        elif not isinstance(raw_interceptor_paths, list):
            if raw_interceptor_paths is not None:
                logger.warning(f"mcpInterceptors must be a list of strings, got {type(raw_interceptor_paths).__name__}; skipping")
            raw_interceptor_paths = []
        for interceptor_path in raw_interceptor_paths:
            try:
                builder = resolve_variable(interceptor_path)
                interceptor = builder()
                if callable(interceptor):
                    tool_interceptors.append(interceptor)
                    logger.info(f"Loaded MCP interceptor: {interceptor_path}")
                elif interceptor is not None:
                    logger.warning(f"Builder {interceptor_path} returned non-callable {type(interceptor).__name__}; skipping")
            except Exception as e:
                logger.warning(
                    f"Failed to load MCP interceptor {interceptor_path}: {e}",
                    exc_info=True,
                )

        client = MultiServerMCPClient(
            servers_config,
            tool_interceptors=tool_interceptors,
            tool_name_prefix=True,
        )

        async def load_server_tools(server_name: str) -> list[BaseTool]:
            try:
                server_cfg = extensions_config.mcp_servers.get(server_name)
                tool_name_prefix = server_cfg.tool_name_prefix if server_cfg is not None else True
                session_init_timeout = _resolve_session_init_timeout(server_cfg)
                if tool_name_prefix:
                    discovery = client.get_tools(server_name=server_name)
                else:
                    discovery = load_mcp_tools(
                        None,
                        connection=servers_config[server_name],
                        callbacks=client.callbacks,
                        server_name=server_name,
                        tool_interceptors=client.tool_interceptors,
                        tool_name_prefix=False,
                    )
                if session_init_timeout is not None:
                    # tool discovery(subprocess 생성 + initialize + tools/list)에 timeout을
                    # 걸어, 멈춘 stdio 서버가 agent 생성을 무한정 막지 못하게 한다. 아래 gather가
                    # 서버마다 독립적으로 실행되므로 서버 단위로 건다 — 느린 서버 하나가 다른
                    # 서버들의 tool 기여를 막아서는 안 된다.
                    #
                    # 여기서의 취소는 안전하다. discovery는 adapter의 중첩된 async context
                    # manager(load_mcp_tools → create_session → _create_stdio_session →
                    # stdio_client) 안에서 실행되고, wait_for의 CancelledError가 그것들을 풀어
                    # 준다. stdio_client의 finally는 stdin을 닫고 정상 종료를 기다린 뒤
                    # _terminate_process_tree(POSIX는 SIGTERM→SIGKILL, Windows는 프로세스 트리
                    # 종료)로 확대하므로, npx subprocess와 그것이 만든 자식들이 모두 회수된다 —
                    # timeout이 반복돼도 고아 프로세스가 쌓이지 않는다.
                    try:
                        return await asyncio.wait_for(discovery, timeout=session_init_timeout)
                    except TimeoutError:
                        # "timed out"으로 로깅하는 것은 우리가 건 상한뿐이다. 분기 조건이 값이
                        # None이 아님을 보장하므로 %.1f 포맷이 실패할 수 없다. discovery 자체가
                        # 던진 TimeoutError(예: opt-out 경로에서 SDK 내부 timeout)는 대신 아래
                        # 일반 실패 handler로 흘러간다.
                        logger.warning(
                            "Skipping MCP server '%s' after tool discovery timed out (%.1fs)",
                            server_name,
                            session_init_timeout,
                        )
                        return []
                return await discovery
            except Exception as e:
                logger.warning(
                    f"Skipping MCP server '{server_name}' after tool discovery failed: {e}",
                    exc_info=True,
                )
                return []

        # 서버마다 독립적으로 tool을 가져와, 망가진 MCP 서버 하나가 정상 서버들의 tool 기여를
        # 막지 못하게 한다.
        tools_by_server = await asyncio.gather(*(load_server_tools(name) for name in servers_config))
        tools = [tool for server_tools in tools_by_server for tool in server_tools]
        logger.info(f"Successfully loaded {len(tools)} tool(s) from MCP servers")

        # 각 tool을 persistent-session 로직으로 감싼다.
        # pool에 넣는 것은 stdio session뿐이다. HTTP/SSE transport는 내부적으로 anyio
        # TaskGroup을 쓰는데 다른 async task에서 닫을 수 없어서, pool에 넣으면 정리 시
        # RuntimeError가 난다(#3203 참고).
        wrapped_tools: list[BaseTool] = []
        # 각 tool을 실제로 만들어 낸 서버 기준으로 라우팅한다. tools_by_server[i]는
        # servers_config의 i번째 서버에 대응한다. servers_config에서 이름 prefix를 훑어 출처
        # 서버를 추정하면, 한 서버 이름이 다른 서버 이름의 prefix일 때 모호해진다(예: "web" vs
        # "web_scraper" → "web_scraper_search".startswith("web_")가 "web"에 먼저 걸린다). 그러면
        # tool이 엉뚱한 서버 아래 pool된다. 출처 그룹핑을 쓰면 서버가 이름 prefix를 끄더라도
        # 라우팅이 정확하다.
        for source_name, server_tools in zip(servers_config.keys(), tools_by_server, strict=True):
            transport = servers_config[source_name].get("transport", "stdio")
            server_cfg = extensions_config.mcp_servers.get(source_name)
            tool_name_prefix = server_cfg.tool_name_prefix if server_cfg is not None else True
            for tool in server_tools:
                if not _VALID_MCP_TOOL_NAME.fullmatch(tool.name or ""):
                    logger.warning(
                        "Dropping MCP tool from server '%s' with invalid name %r: tool names must match %s. A name outside this charset cannot be bound as a function tool and could forge prompt structure when listed as a deferred tool.",
                        source_name,
                        tool.name,
                        _VALID_MCP_TOOL_NAME.pattern,
                    )
                    continue
                tag_mcp_tool(tool)
                prefix = f"{source_name}_"
                original_name = tool.name[len(prefix) :] if tool_name_prefix and tool.name.startswith(prefix) else tool.name
                routing = resolve_effective_mcp_routing(server_cfg, original_name)
                if routing.get("mode") != "off":
                    tag_mcp_routing(tool, routing)
                if transport == "stdio":
                    _timeout = server_cfg.tool_call_timeout if server_cfg else None
                    _init_timeout = _resolve_session_init_timeout(server_cfg)
                    wrapped_tools.append(
                        _make_session_pool_tool(
                            tool,
                            source_name,
                            servers_config[source_name],
                            tool_interceptors,
                            tool_call_timeout=_timeout,
                            session_init_timeout=_init_timeout,
                            tool_name_prefix=tool_name_prefix,
                        )
                    )
                else:
                    if transport != "stdio" and server_cfg and server_cfg.tool_call_timeout is not None:
                        logger.warning(
                            "Ignoring tool_call_timeout for MCP server '%s' because transport '%s' is not stdio; configure HTTP/SSE transport-level timeouts instead.",
                            source_name,
                            transport,
                        )
                    wrapped_tools.append(tool)

        # deerflow client가 동기적으로 stream하므로, tool이 sync 호출도 지원하도록 패치한다
        for tool in wrapped_tools:
            if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
                tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)

        return wrapped_tools

    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}", exc_info=True)
        return []
