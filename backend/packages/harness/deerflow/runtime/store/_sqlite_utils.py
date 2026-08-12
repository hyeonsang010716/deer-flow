"""store와 checkpointer provider가 공유하는 SQLite 연결 유틸리티."""

from __future__ import annotations

import pathlib

from deerflow.config.paths import resolve_path


def resolve_sqlite_conn_str(raw: str) -> str:
    """store/checkpointer backend에 바로 쓸 수 있는 SQLite 연결 문자열을 반환한다.

    SQLite 특수 문자열(``":memory:"``과 ``file:`` URI)은 그대로 돌려준다. 상대든 절대든
    일반 파일시스템 경로는 :func:`resolve_path`로 절대 경로 문자열로 변환한다.
    """
    if raw == ":memory:" or raw.startswith("file:"):
        return raw
    return str(resolve_path(raw))


def ensure_sqlite_parent_dir(conn_str: str) -> None:
    """SQLite 파일 경로의 부모 디렉터리를 만든다.

    in-memory DB(``":memory:"``)와 ``file:`` URI에서는 아무것도 하지 않는다.
    """
    if conn_str != ":memory:" and not conn_str.startswith("file:"):
        pathlib.Path(conn_str).parent.mkdir(parents=True, exist_ok=True)
