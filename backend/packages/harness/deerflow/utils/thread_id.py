"""DeerFlow backend 전체가 공유하는 표준 thread 식별자 검증."""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from pydantic import AfterValidator, StringConstraints

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_THREAD_ID_RE = re.compile(THREAD_ID_PATTERN)


def validate_thread_id(thread_id: str) -> str:
    """유효한 thread ID를 반환하거나 ``ValueError``를 던진다.

    thread ID는 호출자가 정하는 불투명 식별자이며 반드시 UUID일 필요는 없다. 다만 모든
    persistence·파일시스템 backend에서 안전한 값이어야 한다.
    """
    if not isinstance(thread_id, str) or _THREAD_ID_RE.fullmatch(thread_id) is None:
        raise ValueError("Invalid thread_id: expected 1-64 ASCII letters, digits, hyphens, or underscores")
    return thread_id


def resolve_thread_id(thread_id: str | None) -> str:
    """전달된 ID를 검증하고, ``None``일 때만 UUID를 생성한다."""
    if thread_id is None:
        return str(uuid.uuid4())
    return validate_thread_id(thread_id)


ThreadId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=THREAD_ID_PATTERN),
    AfterValidator(validate_thread_id),
]
