"""thread metadata 저장소의 추상 인터페이스.

구현체:
- ThreadMetaRepository: SQL 기반(SQLAlchemy를 통한 sqlite / postgres)
- MemoryThreadMetaStore: LangGraph BaseStore를 감싼다(memory 모드)

모든 변경/조회 메서드는 3가지 상태를 갖는 ``user_id`` 파라미터를 받는다
(:mod:`deerflow.runtime.user_context` 참고):

- ``AUTO``(기본값): request 스코프 contextvar에서 해석한다.
- 명시적 ``str``: 주어진 값을 그대로 쓴다.
- 명시적 ``None``: owner 필터링을 우회한다(migration/CLI 전용).
"""

from __future__ import annotations

import abc
from typing import Any

from deerflow.runtime.user_context import AUTO, _AutoSentinel

# 컴포넌트 간에 공유하는 metadata 키. ``frontend/src/core/threads/utils.ts``와
# ``frontend/tests/e2e/utils/mock-api.ts``와 동기화를 유지한다.
THREAD_PINNED_METADATA_KEY = "deerflow_pinned"


class InvalidMetadataFilterError(ValueError):
    """client가 넘긴 metadata 필터 키가 전부 거부되면 발생한다."""


class ThreadMetaStore(abc.ABC):
    @abc.abstractmethod
    async def create(
        self,
        thread_id: str,
        *,
        assistant_id: str | None = None,
        user_id: str | None | _AutoSentinel = AUTO,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        pass

    @abc.abstractmethod
    async def get(self, thread_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> dict | None:
        pass

    @abc.abstractmethod
    async def search(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> list[dict[str, Any]]:
        """thread를 검색한다.

        결과는 pin된 thread(``metadata.deerflow_pinned is True``)를 먼저 두고,
        각 그룹 안에서는 ``updated_at``과 ``thread_id`` 내림차순으로 정렬한다.
        """
        pass

    @abc.abstractmethod
    async def update_display_name(self, thread_id: str, display_name: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        pass

    @abc.abstractmethod
    async def update_status(self, thread_id: str, status: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        pass

    @abc.abstractmethod
    async def update_metadata(self, thread_id: str, metadata: dict, *, touch: bool = True, user_id: str | None | _AutoSentinel = AUTO) -> None:
        """``metadata``를 thread의 metadata 필드에 병합한다.

        기존 키는 새 값으로 덮어쓰고, ``metadata``에 없는 키는 보존한다. thread가
        없거나 owner 검사에 실패하면 아무것도 하지 않는다.

        ``touch``가 ``True``(기본값)이면 해당 row의 ``updated_at``을 갱신해서
        변경이 최신순 정렬에 반영되게 한다. 대화 활동이 아닌 metadata 변경(예: pin/unpin)은
        ``touch=False``를 넘겨서 thread가 ``updated_at`` 정렬 목록에서 자리를 유지하게 한다.
        """
        pass

    @abc.abstractmethod
    async def update_owner(self, thread_id: str, owner_user_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        """thread metadata row를 새 owner에게 옮긴다.

        신뢰된 내부 복구/migration 경로용이다. row가 없거나 호출자가 owner 검사에
        실패하면 아무것도 하지 않는다.
        """
        pass

    @abc.abstractmethod
    async def check_access(self, thread_id: str, user_id: str, *, require_existing: bool = False) -> bool:
        """``user_id``가 ``thread_id``에 접근할 수 있는지 확인한다."""
        pass

    @abc.abstractmethod
    async def delete(self, thread_id: str, *, user_id: str | None | _AutoSentinel = AUTO) -> None:
        pass
