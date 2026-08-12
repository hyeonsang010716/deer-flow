import mimetypes
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.tools.types import Runtime

_ALLOWED_IMAGE_VIRTUAL_ROOTS = (
    f"{VIRTUAL_PATH_PREFIX}/workspace",
    f"{VIRTUAL_PATH_PREFIX}/uploads",
    f"{VIRTUAL_PATH_PREFIX}/outputs",
)
_ALLOWED_IMAGE_VIRTUAL_ROOTS_TEXT = ", ".join(_ALLOWED_IMAGE_VIRTUAL_ROOTS)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _is_allowed_image_virtual_path(image_path: str) -> bool:
    return any(image_path == root or image_path.startswith(f"{root}/") for root in _ALLOWED_IMAGE_VIRTUAL_ROOTS)


def _detect_image_mime(image_data: bytes) -> str | None:
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(image_data) >= 12 and image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    if image_data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _sanitize_image_error(error: Exception, thread_data: ThreadDataState | None) -> str:
    from deerflow.sandbox.tools import mask_local_paths_in_output

    return mask_local_paths_in_output(f"{type(error).__name__}: {error}", thread_data)


@tool("view_image", parse_docstring=True)
def view_image_tool(
    runtime: Runtime,
    image_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """이미지 파일을 읽는다.

    이미지 파일을 읽어 표시할 수 있게 만들 때 이 tool을 사용하라.

    view_image tool을 사용해야 할 때:
    - 이미지 파일을 직접 봐야 할 때.

    view_image tool을 사용하면 안 되는 때:
    - 이미지가 아닌 파일(대신 present_files를 사용하라)
    - 여러 파일을 한 번에 다룰 때(대신 present_files를 사용하라)

    Args:
        image_path: 이미지 파일의 /mnt/user-data 절대 virtual path. 지원하는 주요 형식: jpg, jpeg, png, webp, gif.
    """
    from deerflow.sandbox.exceptions import SandboxRuntimeError
    from deerflow.sandbox.tools import (
        get_thread_data,
        resolve_and_validate_user_data_path,
        validate_local_tool_path,
    )

    thread_data = get_thread_data(runtime)

    if not _is_allowed_image_virtual_path(image_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Only image paths under {_ALLOWED_IMAGE_VIRTUAL_ROOTS_TEXT} are allowed",
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    try:
        validate_local_tool_path(image_path, thread_data, read_only=True)
        actual_path = resolve_and_validate_user_data_path(image_path, thread_data)
    except (PermissionError, SandboxRuntimeError) as e:
        return Command(
            update={"messages": [ToolMessage(f"Error: {str(e)}", tool_call_id=tool_call_id)]},
        )

    path = Path(actual_path)

    # 파일이 존재하는지 검증한다
    if not path.exists():
        return Command(
            update={"messages": [ToolMessage(f"Error: Image file not found: {image_path}", tool_call_id=tool_call_id)]},
        )

    # 디렉터리가 아니라 파일인지 검증한다
    if not path.is_file():
        return Command(
            update={"messages": [ToolMessage(f"Error: Path is not a file: {image_path}", tool_call_id=tool_call_id)]},
        )

    # 이미지 확장자를 검증한다
    expected_mime_type = _EXTENSION_TO_MIME.get(path.suffix.lower())
    if expected_mime_type is None:
        return Command(
            update={"messages": [ToolMessage(f"Error: Unsupported image format: {path.suffix}. Supported formats: {', '.join(_EXTENSION_TO_MIME)}", tool_call_id=tool_call_id)]},
        )

    # 파일 확장자로 MIME 타입을 판별한다
    mime_type, _ = mimetypes.guess_type(actual_path)
    if mime_type is None:
        mime_type = expected_mime_type

    try:
        image_size = path.stat().st_size
    except OSError as e:
        return Command(
            update={"messages": [ToolMessage(f"Error reading image metadata: {_sanitize_image_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )
    if image_size > _MAX_IMAGE_BYTES:
        return Command(
            update={"messages": [ToolMessage(f"Error: Image file is too large: {image_size} bytes. Maximum supported size is {_MAX_IMAGE_BYTES} bytes", tool_call_id=tool_call_id)]},
        )

    # 내용을 검증하기 위해 이미지 파일을 읽는다(magic byte + 크기)
    try:
        with open(actual_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(f"Error reading image file: {_sanitize_image_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )

    if len(image_data) != image_size:
        # stat()과 read() 사이에 파일이 바뀌었다. 안전을 위해 거부한다.
        return Command(
            update={"messages": [ToolMessage("Error: Image file changed during read", tool_call_id=tool_call_id)]},
        )

    detected_mime_type = _detect_image_mime(image_data)
    if detected_mime_type is None:
        return Command(
            update={"messages": [ToolMessage("Error: File contents do not match a supported image format", tool_call_id=tool_call_id)]},
        )
    if detected_mime_type != expected_mime_type:
        return Command(
            update={"messages": [ToolMessage(f"Error: Image contents are {detected_mime_type}, but file extension indicates {expected_mime_type}", tool_call_id=tool_call_id)]},
        )
    mime_type = detected_mime_type

    # 큰 payload가 checkpoint마다 중복되지 않도록 state에는 base64가 아니라 가벼운 metadata만
    # 저장한다(#4138 참고). 모델이 필요로 할 때 middleware가 파일을 그때그때 읽는다.
    new_viewed_images = {
        image_path: {
            "mime_type": mime_type,
            "size": image_size,
            "actual_path": str(actual_path),
        }
    }

    return Command(
        update={"viewed_images": new_viewed_images, "messages": [ToolMessage("Successfully read image", tool_call_id=tool_call_id)]},
    )
