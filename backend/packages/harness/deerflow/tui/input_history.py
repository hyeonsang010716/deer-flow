"""위/아래 탐색을 지원하는 크기 제한 composer 입력 히스토리(순수 로직).

여기에는 persistence도 Textual 의존성도 없다. 항목의 초기 적재나 저장은 앱이 다른 곳에서
할 수 있다. 탐색 중에는 작성 중이던 draft를 따로 보관해, 히스토리를 거슬러 올라갔다가 다시
내려오면 사용자가 입력하던 내용이 복원된다.
"""

from __future__ import annotations

DEFAULT_LIMIT = 200


class InputHistory:
    def __init__(self, entries: list[str] | None = None, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = max(1, limit)
        self._entries: list[str] = list(entries or [])[-self._limit :]
        self._cursor: int | None = None  # None이면 탐색 중이 아니다
        self._draft: str = ""

    def entries(self) -> list[str]:
        return list(self._entries)

    def add(self, text: str) -> None:
        """제출된 항목을 기록한다. 빈 줄과 연속 중복 줄은 무시한다."""
        self._cursor = None
        self._draft = ""
        if not text.strip():
            return
        if self._entries and self._entries[-1] == text:
            return
        self._entries.append(text)
        if len(self._entries) > self._limit:
            self._entries = self._entries[-self._limit :]

    def up(self, draft: str = "") -> str:
        """한 항목 이전으로 이동한다. 그 항목을 반환하며, 히스토리가 비었으면 ``draft``를 반환한다."""
        if not self._entries:
            return draft
        if self._cursor is None:
            self._draft = draft
            self._cursor = len(self._entries) - 1
        elif self._cursor > 0:
            self._cursor -= 1
        return self._entries[self._cursor]

    def down(self) -> str:
        """한 항목 이후로 이동한다. 가장 최근 항목을 지나면 draft를 복원한다."""
        if self._cursor is None:
            return self._draft
        if self._cursor < len(self._entries) - 1:
            self._cursor += 1
            return self._entries[self._cursor]
        self._cursor = None
        return self._draft

    def reset(self) -> None:
        self._cursor = None
        self._draft = ""
