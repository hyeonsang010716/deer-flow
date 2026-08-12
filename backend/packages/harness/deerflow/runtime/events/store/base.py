"""run event 저장을 위한 추상 인터페이스.

RunEventStore는 run event stream의 통합 저장 인터페이스다. message(frontend 표시용)와 실행
trace(디버깅/감사용)가 같은 인터페이스를 거치며, ``category`` 필드로 구분된다.

구현체:
- MemoryRunEventStore: 메모리 dict(개발, 테스트용)
- DbRunEventStore: SQLAlchemy ORM 기반 영속화
- JsonlRunEventStore: 로컬/디버그용 JSONL 파일 영속화
"""

from __future__ import annotations

import abc

from deerflow.runtime.user_context import AUTO, _AutoSentinel


class RunEventStore(abc.ABC):
    """run event stream 저장 인터페이스.

    모든 구현체는 다음을 보장해야 한다:
    1. put()한 event는 이후 조회에서 다시 읽을 수 있다
    2. seq는 같은 thread 안에서 순증가한다
    3. list_messages()는 category="message" event만 반환한다
    4. list_events()는 지정한 run의 모든 event를 반환한다
    5. 반환 dict는 필수 RunEvent envelope 필드를 담는다. backend는
       DbRunEventStore.user_id처럼 문서화된 필드를 추가할 수 있다
    """

    @abc.abstractmethod
    async def put(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        category: str,
        content: str | dict = "",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> dict:
        """event를 쓰고 seq를 자동 부여한 뒤 완성된 레코드를 반환한다."""

    @abc.abstractmethod
    async def put_batch(self, events: list[dict]) -> list[dict]:
        """event를 배치로 쓴다. RunJournal의 flush buffer가 쓴다.

        각 dict의 키는 put()의 키워드 인자와 같다.
        seq가 부여된 완성 레코드를 반환한다.
        """

    @abc.abstractmethod
    async def put_if_absent(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        category: str,
        content: str | dict = "",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> tuple[dict, bool]:
        """해당 run에 같은 event type이 아직 없을 때만 event 하나를 쓴다.

        검사와 쓰기는 그 thread의 일반 writer와 직렬화되어야 한다. ``(record, created)``를
        반환한다. terminal run receipt가 쓰는 durability primitive이며, 그 복구 경로는 worker가
        죽은 뒤에도 안전하게 재시도할 수 있다.
        """

    @abc.abstractmethod
    async def list_messages(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> list[dict]:
        """thread의 표시 가능한 message(category=message)를 seq 오름차순으로 반환한다.

        양방향 cursor 페이지네이션을 지원한다:
        - before_seq: seq < before_seq인 마지막 ``limit``개(오름차순)
        - after_seq: seq > after_seq인 처음 ``limit``개(오름차순)
        - 둘 다 없으면: 최신 ``limit``개(오름차순)

        요청과 무관한 호출자는 ``user_id``를 명시적으로 넘길 수 있다. user 범위를 두는 backend는
        자신의 격리 모델에 맞게 이를 적용해야 한다.
        """

    @abc.abstractmethod
    async def list_events(
        self,
        thread_id: str,
        run_id: str,
        *,
        event_types: list[str] | None = None,
        task_id: str | None = None,
        limit: int = 500,
        after_seq: int | None = None,
    ) -> list[dict]:
        """run의 전체 event stream을 seq 오름차순으로 반환한다.

        ``event_types``와 ``task_id``(``metadata["task_id"]``에 매칭)로 선택적으로 필터링한다.
        ``after_seq``는 seq > after_seq인 처음 ``limit``개를 반환하는 전진 cursor다. 덕분에
        run 전체에 걸린 ``limit``이 꼬리를 잘라내지 않고도 subagent task 하나의 event를 페이지
        단위로 훑을 수 있다(#3779).
        """

    @abc.abstractmethod
    async def list_messages_by_run(
        self,
        thread_id: str,
        run_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict]:
        """특정 run의 표시 가능한 message(category=message)를 seq 오름차순으로 반환한다.

        양방향 cursor 페이지네이션을 지원한다:
        - after_seq: seq > after_seq인 처음 ``limit``개(오름차순)
        - before_seq: seq < before_seq인 마지막 ``limit``개(오름차순)
        - 둘 다 없으면: 최신 ``limit``개(오름차순)
        """

    @abc.abstractmethod
    async def get_last_visible_ai_seq_by_run(
        self,
        thread_id: str,
        run_ids: set[str],
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> dict[str, int]:
        """각 run의 마지막 non-middleware AI message의 seq를 반환한다.

        ``user_id``는 :meth:`list_messages`와 같은 명시적 호출자 의미를 따른다.
        """

    @abc.abstractmethod
    async def count_messages(self, thread_id: str) -> int:
        """thread 안의 표시 가능한 message(category=message) 수를 센다."""

    @abc.abstractmethod
    async def delete_by_thread(self, thread_id: str) -> int:
        """thread의 모든 event를 삭제하고 삭제된 event 수를 반환한다."""

    @abc.abstractmethod
    async def delete_by_run(self, thread_id: str, run_id: str) -> int:
        """특정 run의 모든 event를 삭제하고 삭제된 event 수를 반환한다."""
