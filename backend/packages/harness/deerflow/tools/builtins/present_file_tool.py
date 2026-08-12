from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.types import Command

from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"


def _get_thread_id(runtime: Runtime) -> str | None:
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


def _normalize_presented_filepath(
    runtime: Runtime,
    filepath: str,
) -> str:
    """제시된 파일 경로를 `/mnt/user-data/outputs/*` 계약에 맞게 정규화한다.

    다음 두 가지를 받는다:
    - `/mnt/user-data/outputs/report.md` 같은 가상 sandbox 경로
    - `/app/backend/.deer-flow/threads/<thread>/user-data/outputs/report.md` 같은
      host 쪽 thread outputs 경로

    Returns:
        정규화된 가상 경로.

    Raises:
        ValueError: runtime metadata가 없거나 경로가 현재 thread의 outputs 디렉터리
            바깥일 때.
    """
    if runtime.state is None:
        raise ValueError("Thread runtime state is not available")

    thread_id = _get_thread_id(runtime)
    if not thread_id:
        raise ValueError("Thread ID is not available in runtime context or runtime config")

    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("Thread outputs path is not available in runtime state")

    outputs_dir = Path(outputs_path).resolve()
    stripped = filepath.lstrip("/")
    virtual_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")

    if stripped == virtual_prefix or stripped.startswith(virtual_prefix + "/"):
        try:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath, user_id=resolve_runtime_user_id(runtime))
        except TypeError:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath)
    else:
        actual_path = Path(filepath).expanduser().resolve()

    try:
        relative_path = actual_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Only files in {OUTPUTS_VIRTUAL_PREFIX} can be presented: {filepath}") from exc

    return f"{OUTPUTS_VIRTUAL_PREFIX}/{relative_path.as_posix()}"


@tool("present_files", parse_docstring=True)
def present_file_tool(
    runtime: Runtime,
    filepaths: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """client interface에서 사용자가 보고 렌더링할 수 있도록 파일을 노출한다.

    present_files tool을 사용해야 할 때:

    - 사용자가 보거나 다운로드하거나 다룰 수 있게 파일을 제공할 때
    - 관련된 여러 파일을 한 번에 제시할 때
    - 사용자에게 보여줄 파일을 생성한 뒤

    present_files tool을 사용하면 안 되는 때:
    - 내부 처리를 위해 파일 내용을 읽기만 하면 될 때
    - 사용자에게 보여줄 목적이 아닌 임시 파일이나 중간 산출물

    참고:
    - 파일을 생성해 `/mnt/user-data/outputs` 디렉터리로 옮긴 뒤에 이 tool을 호출하라.
    - 이 tool은 다른 tool과 병렬로 안전하게 호출할 수 있다. 상태 갱신은 reducer가 처리해 충돌을 막는다.

    Args:
        filepaths: 사용자에게 제시할 파일의 절대 경로 리스트. `/mnt/user-data/outputs` 안의 파일**만** 제시할 수 있다.
    """
    try:
        normalized_paths = [_normalize_presented_filepath(runtime, filepath) for filepath in filepaths]
    except ValueError as exc:
        return Command(
            update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]},
        )

    # 병합과 중복 제거는 merge_artifacts reducer가 처리한다
    return Command(
        update={
            "artifacts": normalized_paths,
            "messages": [ToolMessage("Successfully presented files", tool_call_id=tool_call_id)],
        },
    )
