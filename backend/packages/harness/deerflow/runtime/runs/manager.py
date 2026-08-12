"""in-memory run registry. 선택적으로 영속 RunStore를 백엔드로 둔다."""

from __future__ import annotations

import asyncio
import logging
import socket
import sqlite3
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import IntegrityError as SAIntegrityError

from deerflow.runtime.user_context import AUTO, _AutoSentinel, resolve_user_id
from deerflow.utils.time import is_lease_expired
from deerflow.utils.time import now_iso as _now_iso

from .schemas import DisconnectMode, RunStatus, ThreadOperationKind
from .store.base import EditReplayVisibility

if TYPE_CHECKING:
    from deerflow.config.run_ownership_config import RunOwnershipConfig
    from deerflow.runtime.events.store.base import RunEventStore
    from deerflow.runtime.runs.store.base import RunStore

logger = logging.getLogger(__name__)

ORPHAN_RECOVERY_STOP_REASON = "orphan_recovered"
STARTUP_ORPHAN_RECOVERY_ERROR = "Gateway restarted before this run reached a durable final state."
LEASE_ORPHAN_RECOVERY_ERROR = "Run lease expired — owning worker is unreachable."

_RETRYABLE_SQLITE_MESSAGES = (
    "database is locked",
    "database table is locked",
    "database is busy",
)

_RETRYABLE_SQLITE_ERROR_CODES = {
    sqlite3.SQLITE_BUSY,
    sqlite3.SQLITE_LOCKED,
}

# driver 고유의 unique-constraint 신호. driver와 SQLAlchemy 버전이 바뀌어도 안정적이다 —
# 메시지 텍스트는 그렇지 않다(SQLite는 "UNIQUE constraint failed", Postgres는
# "duplicate key value violates unique constraint"라고 한다).
_UNIQUE_PGCODE = "23505"
_SQLITE_UNIQUE_ERRORCODE = sqlite3.SQLITE_CONSTRAINT_UNIQUE


def _generate_worker_id() -> str:
    """고유한 worker 식별자 ``hostname:hex_uuid``를 생성한다."""
    return f"{socket.gethostname()}:{uuid.uuid4().hex}"


def _is_unique_violation(exc: BaseException) -> bool:
    """*exc*(또는 그 cause chain)가 unique-constraint 위반이면 True를 반환한다.

    SQLAlchemy는 driver의 IntegrityError를 감싸며, 감싸인 driver 예외는 ``exc.orig``
    (그리고 ``__cause__`` / ``__context__``)로 접근할 수 있다. 메시지 매칭보다
    driver 고유 신호 — psycopg ``pgcode`` / ``sqlcode`` = "23505", sqlite3
    ``sqlite_errorcode`` = ``SQLITE_CONSTRAINT_UNIQUE`` — 를 우선하고, chain으로
    driver 예외에 닿을 수 없는 경우에만 메시지 부분 문자열로 fallback한다.

    메시지 텍스트는 driver와 로케일에 따라 달라지므로(SQLite는
    ``UNIQUE constraint failed: <table>.<index>``, Postgres는
    ``duplicate key value violates unique constraint``를 낸다) 코드/속성 검사가
    핵심 경로다.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if getattr(current, "pgcode", None) == _UNIQUE_PGCODE:
            return True
        if getattr(current, "sqlcode", None) == _UNIQUE_PGCODE:
            return True
        if getattr(current, "sqlstate", None) == _UNIQUE_PGCODE:
            return True
        if getattr(current, "sqlite_errorcode", None) == _SQLITE_UNIQUE_ERRORCODE:
            return True

        # 메시지 fallback은 chain으로 고유 코드 속성에 닿을 수 없는 driver를 위한
        # 이중 안전장치다. IntegrityError 타입 노드로 제한해서, ``str()``에 우연히
        # "duplicate key" / "unique" + "violat"이 들어간 무관한 애플리케이션 예외
        # (CHECK constraint 메시지, 검증 오류, 임의의 서브시스템 문자열)가 unique
        # 위반으로 오분류되어 500 대신 HTTP 409로 조용히 나가지 않게 한다.
        if isinstance(current, (SAIntegrityError, sqlite3.IntegrityError)):
            message = str(current).lower()
            if "unique constraint failed" in message:
                return True
            if "unique" in message and "violat" in message:
                return True
            if "duplicate key" in message:
                return True

        for attr in ("orig", "__cause__", "__context__"):
            inner = getattr(current, attr, None)
            if isinstance(inner, BaseException):
                pending.append(inner)
    return False


def _is_retryable_persistence_error(exc: BaseException) -> bool:
    """일시적인 SQLite 영속화 실패면 True를 반환한다.

    SQLite lock 경합은 보통 sqlite3 예외나 SQLAlchemy wrapper로 드러난다. 여기의
    짧은 bounded retry는 영구적 실패를 영원히 감추지 않으면서 run status 확정을
    일시적인 writer 부하로부터 보호한다.
    """

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        message = str(current).lower()
        if any(fragment in message for fragment in _RETRYABLE_SQLITE_MESSAGES):
            return True
        if isinstance(current, (sqlite3.OperationalError, sqlite3.DatabaseError)):
            error_code = getattr(current, "sqlite_errorcode", None)
            if error_code in _RETRYABLE_SQLITE_ERROR_CODES:
                return True
        for chained in (getattr(current, "orig", None), current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


@dataclass(frozen=True)
class PersistenceRetryPolicy:
    """짧은 run-store 쓰기에 적용하는 bounded retry 정책."""

    max_attempts: int = 5
    initial_delay: float = 0.05
    max_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class RunRecord:
    """단일 run에 대한 변경 가능한 record."""

    run_id: str
    thread_id: str
    assistant_id: str | None
    status: RunStatus
    on_disconnect: DisconnectMode
    operation_kind: ThreadOperationKind = ThreadOperationKind.run
    multitask_strategy: str = "reject"
    metadata: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)
    user_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    task: asyncio.Task | None = field(default=None, repr=False)
    # 승인된 run이 둘 이상의 worker 경로로 넘어가는 경우 startup을 직렬화한다.
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    abort_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    abort_action: str = "interrupt"
    error: str | None = None
    model_name: str | None = None
    store_only: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    lead_agent_tokens: int = 0
    subagent_tokens: int = 0
    middleware_tokens: int = 0
    # 모델별 token 사용량 분해
    token_usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    message_count: int = 0
    last_ai_message: str | None = None
    first_human_message: str | None = None
    finalizing: bool = False
    owner_worker_id: str | None = None
    lease_expires_at: str | None = None
    # 프로세스 로컬 fencing 신호. 한 번 설정되면 이 worker는 더 이상 durable한
    # run/thread 확정을 수행하면 안 된다. lease 소유권을 잃은 것이 확인됐거나
    # 만료 전에 확인하지 못했기 때문이다.
    ownership_lost: bool = False
    stop_reason: str | None = None


class RunStartOutcome(StrEnum):
    """pending에서 running으로 넘어가는 startup barrier의 결과."""

    started = "started"
    cancelled = "cancelled"


class RunStartupError(RuntimeError):
    """durable한 startup을 안전하게 결정할 수 없을 때 발생한다."""


OrphanRecoveryCallback = Callable[[list[RunRecord]], Awaitable[None]]


class RunManager:
    """in-memory run registry. 선택적으로 영속 RunStore를 백엔드로 둔다.

    모든 변경은 asyncio lock으로 보호된다. ``store``가 주어지면 직렬화 가능한
    metadata도 store에 영속화되어 run history가 프로세스 재시작을 견딘다.
    """

    def __init__(
        self,
        store: RunStore | None = None,
        *,
        persistence_retry_policy: PersistenceRetryPolicy | None = None,
        worker_id: str | None = None,
        run_ownership_config: RunOwnershipConfig | None = None,
        event_store: RunEventStore | None = None,
        on_orphans_recovered: OrphanRecoveryCallback | None = None,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        # 보조 index: thread_id -> 삽입 순서를 유지하는 run_id 집합(dict를 ordered
        # set으로 사용). ``_runs``와 lockstep으로 갱신되므로 thread 단위 조회가
        # O(전체 in-memory run) 전체 스캔을 피하면서도 ``_runs``의 순회 순서를
        # 유지한다(``_thread_records_locked`` 참고).
        self._runs_by_thread: dict[str, dict[str, None]] = {}
        self._lock = asyncio.Lock()
        self._store = store
        self._persistence_retry_policy = persistence_retry_policy or PersistenceRetryPolicy()
        self._worker_id = worker_id or _generate_worker_id()
        self._run_ownership_config = run_ownership_config
        self._event_store = event_store
        self._on_orphans_recovered = on_orphans_recovered
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_stop: asyncio.Event | None = None
        self._orphan_recovery_task: asyncio.Task[None] | None = None

    def _index_run_locked(self, record: RunRecord) -> None:
        """*record*를 thread index에 등록한다. 호출자가 ``self._lock``을 잡고 있어야 한다."""
        self._runs_by_thread.setdefault(record.thread_id, {})[record.run_id] = None

    def _unindex_run_locked(self, run_id: str, thread_id: str) -> None:
        """thread index에서 *run_id*를 제거한다. 호출자가 ``self._lock``을 잡고 있어야 한다."""
        bucket = self._runs_by_thread.get(thread_id)
        if bucket is not None:
            bucket.pop(run_id, None)
            if not bucket:
                self._runs_by_thread.pop(thread_id, None)

    def _thread_records_locked(self, thread_id: str) -> list[RunRecord]:
        """*thread_id*의 살아 있는 in-memory record를 반환한다. 호출자가 ``self._lock``을 잡아야 한다.

        모든 in-memory run을 스캔하는 대신 ``_runs_by_thread`` index로 O(thread 내 run)
        조회를 한다. 정확성은 index와 ``_runs``가 ``self._lock`` 아래에서 lockstep으로
        갱신된다는 점(두 쓰기 사이에 ``await``가 없다)에 기대므로, lock을 잡은 쪽은
        항상 둘이 일치하는 상태를 본다. ``self._runs.get`` 필터는 정합성 복구가 아니라
        이중 안전장치다. index에는 남아 있지만 ``_runs``에서 이미 사라진 낡은 id는
        걸러내지만, ``_runs``에 있는데 index에 없는 run은 복구하지 못한다(그런 run은
        조용히 누락된다). 향후 리팩터링이 lockstep 불변식을 깨뜨릴 경우를 대비해 한쪽
        방향만 보호한다.
        """
        run_ids = self._runs_by_thread.get(thread_id)
        if not run_ids:
            return []
        return [record for run_id in run_ids if (record := self._runs.get(run_id)) is not None]

    @staticmethod
    def _store_put_payload(record: RunRecord, *, error: str | None = None, stop_reason: str | None = None) -> dict[str, Any]:
        payload = {
            "thread_id": record.thread_id,
            "assistant_id": record.assistant_id,
            "status": record.status.value,
            "operation_kind": record.operation_kind.value,
            "multitask_strategy": record.multitask_strategy,
            "metadata": record.metadata or {},
            "kwargs": record.kwargs or {},
            "error": error if error is not None else record.error,
            "created_at": record.created_at,
            "model_name": record.model_name,
            "owner_worker_id": record.owner_worker_id,
            "lease_expires_at": record.lease_expires_at,
        }
        if record.user_id is not None:
            payload["user_id"] = record.user_id
        if record.stop_reason is not None:
            payload["stop_reason"] = record.stop_reason
        return payload

    async def _call_store_with_retry(
        self,
        operation_name: str,
        run_id: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """SQLite 부하를 대비해 bounded retry를 붙여 짧은 store 연산을 실행한다."""
        policy = self._persistence_retry_policy
        attempt = 1
        delay = policy.initial_delay
        while True:
            try:
                return await operation()
            except Exception as exc:
                retryable = _is_retryable_persistence_error(exc)
                if attempt >= policy.max_attempts or not retryable:
                    raise
                logger.warning(
                    "Transient persistence failure during %s for run %s (attempt %d/%d); retrying",
                    operation_name,
                    run_id,
                    attempt,
                    policy.max_attempts,
                    exc_info=True,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                delay = min(policy.max_delay, delay * policy.backoff_factor if delay else policy.initial_delay)
                attempt += 1

    async def _persist_snapshot_to_store(self, run_id: str, payload: dict[str, Any]) -> bool:
        """앞서 캡처해 둔 run snapshot을 best-effort로 영속화한다."""
        if self._store is None:
            return True
        try:
            await self._call_store_with_retry(
                "put",
                run_id,
                lambda: self._store.put(run_id, **payload),
            )
            return True
        except Exception:
            logger.warning("Failed to persist run %s to store", run_id, exc_info=True)
            return False

    async def _persist_new_run_to_store(self, record: RunRecord) -> None:
        """새로 생성된 run record를 backing store에 영속화한다.

        최초 run 생성은 run 가시성 경계의 일부다. backing store row가 없으면
        호출자가 메모리에서 그 run을 관측해서는 안 된다. 이후의 status/model 업데이트와
        달리 실패는 그대로 전파되므로 호출자가 생성 실패로 처리할 수 있다. record를
        ``_runs``에 넣은 뒤의 rollback은 호출자 책임이다.
        """
        if self._store is None:
            return
        await self._call_store_with_retry(
            "put",
            record.run_id,
            lambda: self._store.put(record.run_id, **self._store_put_payload(record)),
        )

    async def _persist_to_store(self, record: RunRecord, *, error: str | None = None) -> bool:
        """run record를 backing store에 best-effort로 영속화한다."""
        return await self._persist_snapshot_to_store(
            record.run_id,
            self._store_put_payload(record, error=error),
        )

    async def _persist_status(self, record: RunRecord, status: RunStatus, *, error: str | None = None, stop_reason: str | None = None) -> bool:
        """status 전이를 backing store에 best-effort로 영속화한다."""
        if record.ownership_lost:
            logger.warning(
                "Skipped status update to %s for run %s after lease ownership was lost",
                status.value,
                record.run_id,
            )
            return False
        if self._store is None:
            return True
        row_recovery_payload = self._store_put_payload(record, error=error, stop_reason=stop_reason)
        try:
            updated = await self._call_store_with_retry(
                "update_status",
                record.run_id,
                lambda: self._store.update_status(record.run_id, status.value, error=error, stop_reason=stop_reason),
            )
            if updated is False:
                # 이제 ``update_status``는 ``status IN ('pending','running')``으로 보호된다.
                # False가 뜻하는 경우는 둘 중 하나다:
                #   (a) row가 애초에 영속화되지 않았다(최초 ``put()`` 실패) → 다시 만든다.
                #   (b) row가 terminal이다 — peer takeover(``error``)이거나 로컬
                #       cancel/completion 경쟁(``interrupted`` / ``success``).
                #       어느 쪽인지에 따라 로그 심각도가 갈린다.
                existing = await self._store.get(record.run_id)
                if existing is not None:
                    existing_status = existing.get("status")
                    if existing_status == status.value:
                        logger.info(
                            "Run %s status update to %s was already persisted",
                            record.run_id,
                            status.value,
                        )
                        return True
                    if existing_status == "error":
                        logger.warning(
                            "Run %s status update to %s skipped: store row already at error (peer takeover)",
                            record.run_id,
                            status.value,
                        )
                        if self.heartbeat_enabled and not record.store_only:
                            await self._mark_ownership_lost(
                                record,
                                reason="A peer terminalized the run before this worker could persist its outcome.",
                                require_active=False,
                            )
                    else:
                        logger.info(
                            "Run %s status update to %s skipped: store row already at %s (local cancel/completion race)",
                            record.run_id,
                            status.value,
                            existing_status,
                        )
                    return False
                return await self._persist_snapshot_to_store(record.run_id, row_recovery_payload)
            return True
        except Exception:
            logger.warning("Failed to persist status update for run %s", record.run_id, exc_info=True)
            return False

    @staticmethod
    def _record_from_store(row: dict[str, Any]) -> RunRecord:
        """직렬화된 store row로부터 읽기 전용 runtime record를 만든다.

        NULL인 status/on_disconnect 컬럼(예: 해당 컬럼 추가 이전에 기록된 row)은 각각
        ``pending``과 ``cancel``을 기본값으로 쓴다.
        """
        return RunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            assistant_id=row.get("assistant_id"),
            status=RunStatus(row.get("status") or RunStatus.pending.value),
            on_disconnect=DisconnectMode(row.get("on_disconnect") or DisconnectMode.cancel.value),
            operation_kind=ThreadOperationKind(row.get("operation_kind") or ThreadOperationKind.run.value),
            multitask_strategy=row.get("multitask_strategy") or "reject",
            metadata=row.get("metadata") or {},
            kwargs=row.get("kwargs") or {},
            created_at=row.get("created_at") or "",
            updated_at=row.get("updated_at") or "",
            user_id=row.get("user_id"),
            error=row.get("error"),
            model_name=row.get("model_name"),
            store_only=True,
            total_input_tokens=row.get("total_input_tokens") or 0,
            total_output_tokens=row.get("total_output_tokens") or 0,
            total_tokens=row.get("total_tokens") or 0,
            llm_call_count=row.get("llm_call_count") or 0,
            lead_agent_tokens=row.get("lead_agent_tokens") or 0,
            subagent_tokens=row.get("subagent_tokens") or 0,
            middleware_tokens=row.get("middleware_tokens") or 0,
            token_usage_by_model=row.get("token_usage_by_model") or {},
            message_count=row.get("message_count") or 0,
            last_ai_message=row.get("last_ai_message"),
            first_human_message=row.get("first_human_message"),
            owner_worker_id=row.get("owner_worker_id"),
            lease_expires_at=row.get("lease_expires_at"),
            stop_reason=row.get("stop_reason"),
        )

    async def update_run_completion(self, run_id: str, **kwargs) -> None:
        """token 사용량과 완료 데이터를 backing store에 영속화한다."""
        row_recovery_payload: dict[str, Any] | None = None
        record: RunRecord | None = None
        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None and record.ownership_lost:
                logger.warning("Skipped completion persistence for run %s after lease ownership was lost", run_id)
                return
            if record is not None:
                for key, value in kwargs.items():
                    if key == "status":
                        continue
                    if hasattr(record, key) and value is not None:
                        setattr(record, key, value)
                record.updated_at = _now_iso()
                row_recovery_payload = self._store_put_payload(record, error=kwargs.get("error"))
        if self._store is None:
            return
        try:
            updated = await self._call_store_with_retry(
                "update_run_completion",
                run_id,
                lambda: self._store.update_run_completion(run_id, **kwargs),
            )
            if updated is False:
                existing = await self._store.get(run_id)
                requested_status = kwargs.get("status")
                if existing is not None and existing.get("status") != requested_status:
                    existing_status = existing.get("status")
                    logger.warning(
                        "Run completion update for %s skipped because store row is already at %s",
                        run_id,
                        existing_status,
                    )
                    if existing_status == "error" and record is not None and self.heartbeat_enabled:
                        await self._mark_ownership_lost(
                            record,
                            reason="A peer terminalized the run before completion data was persisted.",
                            require_active=False,
                        )
                    return
                if row_recovery_payload is None:
                    logger.warning("Failed to recreate missing run %s for completion persistence", run_id)
                    return
                if not await self._persist_snapshot_to_store(run_id, row_recovery_payload):
                    return
                recovered = await self._call_store_with_retry(
                    "update_run_completion",
                    run_id,
                    lambda: self._store.update_run_completion(run_id, **kwargs),
                )
                if recovered is False:
                    logger.warning("Run completion update for %s affected no rows after row recreation", run_id)
        except Exception:
            logger.warning("Failed to persist run completion for %s", run_id, exc_info=True)

    async def update_run_progress(self, run_id: str, **kwargs) -> None:
        """status는 바꾸지 않고 진행 중인 token/message snapshot을 영속화한다."""
        should_persist = True
        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                should_persist = record.status == RunStatus.running and not record.ownership_lost
            if record is not None and should_persist:
                for key, value in kwargs.items():
                    if hasattr(record, key) and value is not None:
                        setattr(record, key, value)
                record.updated_at = _now_iso()
        if should_persist and self._store is not None:
            try:
                await self._store.update_run_progress(run_id, **kwargs)
            except Exception:
                logger.warning("Failed to persist run progress for %s", run_id, exc_info=True)

    async def create(
        self,
        thread_id: str,
        assistant_id: str | None = None,
        *,
        on_disconnect: DisconnectMode = DisconnectMode.cancel,
        metadata: dict | None = None,
        kwargs: dict | None = None,
        multitask_strategy: str = "reject",
        user_id: str | None = None,
    ) -> RunRecord:
        """새 pending run을 만들어 등록한다.

        주의: 이 메서드는 해당 thread에 active run이 없다고 가정한다. 원자적
        ``create_thread_operation_atomic`` 대신 ``store.put``(upsert)으로 영속화하므로,
        같은 thread에 대한 동시 insert는 partial unique index에 걸려
        ``ConflictError``가 아니라 raw ``IntegrityError``로 드러난다. 프로덕션
        호출자는 :meth:`create_or_reject`를 써야 한다.
        """
        run_id = str(uuid.uuid4())
        now = _now_iso()
        lease_expires_at = self._compute_lease_expires_at()
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            status=RunStatus.pending,
            on_disconnect=on_disconnect,
            multitask_strategy=multitask_strategy,
            metadata=metadata or {},
            kwargs=kwargs or {},
            user_id=user_id,
            created_at=now,
            updated_at=now,
            owner_worker_id=self._worker_id,
            lease_expires_at=lease_expires_at,
        )
        async with self._lock:
            self._runs[run_id] = record
            self._index_run_locked(record)
            persisted = False
            try:
                await self._persist_new_run_to_store(record)
                persisted = True
            except Exception:
                logger.warning("Failed to persist run %s; rolled back in-memory record", run_id, exc_info=True)
                raise
            finally:
                # ``except Exception``을 우회하는 cancellation까지 함께 처리한다.
                if not persisted:
                    self._runs.pop(run_id, None)
                    self._unindex_run_locked(run_id, record.thread_id)
        logger.info("Run created: run_id=%s thread_id=%s", run_id, thread_id)
        return record

    async def get(self, run_id: str, *, user_id: str | None = None) -> RunRecord | None:
        """ID로 run record를 반환하거나 ``None``을 반환한다.

        Args:
            run_id: 조회할 run ID.
            user_id: store에서 hydrate할 때 권한 필터링에 쓰는 선택적 user ID.
        """
        async with self._lock:
            record = self._runs.get(run_id)
        if record is not None:
            return record
        if self._store is None:
            return None
        try:
            row = await self._store.get(run_id, user_id=user_id)
        except Exception:
            logger.warning("Failed to hydrate run %s from store", run_id, exc_info=True)
            return None
        # store await 이후 재확인: store 호출이 진행되는 동안 동시 create()가
        # in-memory record를 넣었을 수 있다.
        async with self._lock:
            record = self._runs.get(run_id)
        if record is not None:
            return record
        if row is None:
            return None
        try:
            return self._record_from_store(row)
        except Exception:
            logger.warning("Failed to map store row for run %s", run_id, exc_info=True)
            return None

    async def aget(self, run_id: str, *, user_id: str | None = None) -> RunRecord | None:
        """ID로 run record를 반환하며, 없으면 영속 store를 fallback으로 확인한다.

        하위 호환을 위한 :meth:`get`의 별칭이다.
        """
        return await self.get(run_id, user_id=user_id)

    async def list_by_thread(self, thread_id: str, *, user_id: str | None = None, limit: int = 100) -> list[RunRecord]:
        """주어진 thread의 run을 최신순으로 최대 ``limit``개 반환한다.

        같은 ``run_id``가 메모리와 backing store 양쪽에 있을 때만 in-memory run이
        우선한다. 병합된 결과는 ``created_at`` 기준 최신순으로 정렬한 뒤 ``limit``
        (기본 100)까지 잘라낸다.

        Args:
            thread_id: 필터링할 thread ID.
            user_id: store에서 hydrate할 때 권한 필터링에 쓰는 선택적 user ID.
            limit: 반환할 run의 최대 개수.
        """
        async with self._lock:
            memory_records = [record for record in self._thread_records_locked(thread_id) if record.operation_kind == ThreadOperationKind.run]
        if self._store is None:
            return sorted(memory_records, key=lambda r: r.created_at, reverse=True)[:limit]
        records_by_id = {record.run_id: record for record in memory_records}
        # 요청한 페이지와 가능한 모든 in-memory/store 중복을 함께 덮을 만큼 row를
        # 조회한다. 로컬 record가 영속 row보다 오래됐을 수 있으므로 store limit에서
        # 그만큼 빼면 병합 전에 실제 최신 run이 가려질 수 있고, ``limit``만 조회하면
        # 그 페이지가 중복된 로컬 record로 채워졌을 때 별개의 row를 놓칠 수 있다.
        store_limit = limit + len(memory_records)
        try:
            rows = await self._store.list_by_thread(thread_id, user_id=user_id, limit=store_limit)
        except Exception:
            logger.warning("Failed to hydrate runs for thread %s from store", thread_id, exc_info=True)
            return sorted(memory_records, key=lambda r: r.created_at, reverse=True)[:limit]
        for row in rows:
            run_id = row.get("run_id")
            if run_id and run_id not in records_by_id:
                try:
                    records_by_id[run_id] = self._record_from_store(row)
                except Exception:
                    logger.warning("Failed to map store row for run %s", run_id, exc_info=True)
        return sorted(records_by_id.values(), key=lambda record: record.created_at, reverse=True)[:limit]

    async def list_successful_regenerate_sources(
        self,
        thread_id: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> set[str]:
        """성공한 regeneration으로 대체된 모든 source run을 반환한다.

        :meth:`list_by_thread`와 달리 이 조회는 의도적으로 개수 제한이 없다. 현재
        프로세스의 record가 영속된 status를 덮어쓴다. 최신 in-memory 실패가 더 오래된
        성공 store snapshot을 물려받아서는 안 되기 때문이다. 대체 여부 필터링은
        올바른 pagination에 필수이므로 store 실패는 그대로 전파한다.
        """
        resolved_user_id = resolve_user_id(user_id, method_name="RunManager.list_successful_regenerate_sources")
        async with self._lock:
            memory_records = [record for record in self._thread_records_locked(thread_id) if record.operation_kind == ThreadOperationKind.run and (resolved_user_id is None or record.user_id == resolved_user_id)]

        sources = set(await self._store.list_successful_regenerate_sources(thread_id, user_id=resolved_user_id)) if self._store is not None else set()
        # _thread_records_locked는 thread index의 삽입 순서를 유지한다. record를
        # 오래된 것부터 적용하면 여러 시도가 같은 source run을 참조할 때(예: 성공 후
        # 실패한 재시도) 최신 in-memory regeneration 시도가 최종 권위를 갖는다.
        for record in memory_records:
            source = record.metadata.get("regenerate_from_run_id")
            if not isinstance(source, str) or not source:
                continue
            sources.discard(source)
            if record.status == RunStatus.success:
                sources.add(source)
        return sources

    @staticmethod
    def _record_status_value(record: RunRecord) -> str:
        status = record.status
        return status.value if isinstance(status, RunStatus) else str(status)

    @staticmethod
    def _compute_edit_replay_visibility(records: list[RunRecord]) -> EditReplayVisibility:
        latest_attempt_by_source: dict[str, tuple[str, str]] = {}
        failed_attempts: set[str] = set()
        for record in sorted(records, key=lambda item: item.created_at):
            metadata = record.metadata or {}
            if metadata.get("replay_kind") != "edit":
                continue
            source = metadata.get("regenerate_from_run_id")
            if not isinstance(source, str) or not source:
                continue
            status = RunManager._record_status_value(record)
            latest_attempt_by_source[source] = (record.run_id, status)
            if status in {RunStatus.error.value, RunStatus.timeout.value, RunStatus.interrupted.value}:
                failed_attempts.add(record.run_id)

        hidden_sources: set[str] = set()
        for source, (_, status) in latest_attempt_by_source.items():
            if status in {RunStatus.pending.value, RunStatus.running.value, RunStatus.success.value}:
                hidden_sources.add(source)
        return EditReplayVisibility(
            hidden_source_run_ids=hidden_sources,
            hidden_attempt_run_ids=failed_attempts,
        )

    async def list_edit_replay_visibility(
        self,
        thread_id: str,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> EditReplayVisibility:
        """edit-and-rerun 시도에 대한 run-id 가시성 규칙을 반환한다.

        store row는 reload / multi-worker 이력을 덮는다. 현재 프로세스의 record가
        같은 run id를 덮어쓰는데, 이 조회를 시작한 시점의 영속 snapshot보다 더 최신의
        terminal status를 가질 수 있기 때문이다.
        """
        resolved_user_id = resolve_user_id(user_id, method_name="RunManager.list_edit_replay_visibility")
        records_by_id: dict[str, RunRecord] = {}
        if self._store is not None:
            rows = await self._store.list_edit_regenerate_runs(thread_id, user_id=resolved_user_id)
            for row in rows:
                try:
                    record = self._record_from_store(row)
                except Exception:
                    logger.warning("Failed to map edit replay run row for %s", row.get("run_id"), exc_info=True)
                    continue
                records_by_id[record.run_id] = record

        async with self._lock:
            memory_records = [record for record in self._thread_records_locked(thread_id) if resolved_user_id is None or record.user_id == resolved_user_id]
        for record in memory_records:
            records_by_id[record.run_id] = record

        return self._compute_edit_replay_visibility(list(records_by_id.values()))

    async def try_start(self, run_id: str) -> RunStartOutcome:
        """agent를 만들기 전에 취소되지 않은 pending run을 running으로 전이시킨다."""
        async with self._lock:
            record = self._runs.get(run_id)
        if record is None:
            raise RunStartupError(f"Cannot start unknown run {run_id}")

        async with record.start_lock:
            async with self._lock:
                if record.abort_event.is_set() or record.status != RunStatus.pending:
                    return RunStartOutcome.cancelled

            if self._store is not None:
                try:
                    updated = await self._call_store_with_retry(
                        "start_run",
                        run_id,
                        lambda: self._store.start_run(run_id),
                    )
                except Exception as exc:
                    raise RunStartupError(f"Failed to start run {run_id}: {exc}") from exc
                if updated is False:
                    async with self._lock:
                        if record.status == RunStatus.pending:
                            record.status = RunStatus.interrupted
                            record.abort_event.set()
                            record.updated_at = _now_iso()
                    return RunStartOutcome.cancelled

            async with self._lock:
                if record.abort_event.is_set() or record.status != RunStatus.pending:
                    restore_status = record.status
                    restore_error = record.error
                    restore_stop_reason = record.stop_reason
                else:
                    record.status = RunStatus.running
                    record.updated_at = _now_iso()
                    logger.info("Run %s -> %s", run_id, RunStatus.running.value)
                    return RunStartOutcome.started

            if self._store is not None:
                await self._persist_status(
                    record,
                    restore_status,
                    error=restore_error,
                    stop_reason=restore_stop_reason,
                )
            return RunStartOutcome.cancelled

    async def fail_start_if_pending(self, run_id: str, *, error: str) -> bool:
        """승인된 run에 worker task를 붙이지 못했으면 실패로 표시한다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None or record.status != RunStatus.pending:
                return False
            record.status = RunStatus.error
            record.error = error
            record.abort_event.set()
            record.updated_at = _now_iso()

        await self._persist_status(record, RunStatus.error, error=error)
        return True

    async def get_many_by_thread(
        self,
        thread_id: str,
        run_ids: set[str],
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ) -> dict[str, RunRecord]:
        """지정한 thread run들을 일괄 로드하며 in-memory record를 우선한다."""
        if not run_ids:
            return {}
        resolved_user_id = resolve_user_id(user_id, method_name="RunManager.get_many_by_thread")
        async with self._lock:
            records_by_id = {
                record.run_id: record for record in self._thread_records_locked(thread_id) if record.operation_kind == ThreadOperationKind.run and record.run_id in run_ids and (resolved_user_id is None or record.user_id == resolved_user_id)
            }
        if self._store is None:
            return records_by_id

        remaining = run_ids - records_by_id.keys()
        if not remaining:
            return records_by_id
        try:
            rows = await self._store.get_many_by_thread(thread_id, set(remaining), user_id=resolved_user_id)
        except Exception:
            logger.warning("Failed to batch-hydrate runs for thread %s", thread_id, exc_info=True)
            return records_by_id
        for run_id, row in rows.items():
            if run_id in records_by_id:
                continue
            try:
                records_by_id[run_id] = self._record_from_store(row)
            except Exception:
                logger.warning("Failed to map store row for run %s", run_id, exc_info=True)
        return records_by_id

    async def set_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
        persist: bool = True,
    ) -> None:
        """run을 새로운 status로 전이시킨다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                logger.warning("set_status called for unknown run %s", run_id)
                return
            if record.ownership_lost:
                logger.warning(
                    "Skipped local status transition to %s for run %s after lease ownership was lost",
                    status.value,
                    run_id,
                )
                return
            record.status = status
            record.updated_at = _now_iso()
            if error is not None:
                record.error = error
            if stop_reason is not None:
                record.stop_reason = stop_reason
        if persist:
            persisted = await self._persist_status(record, status, error=error, stop_reason=stop_reason)
            if not persisted and self.heartbeat_enabled and status == RunStatus.success and not record.ownership_lost:
                await self._mark_ownership_lost(
                    record,
                    reason="Successful completion could not be confirmed in the durable run store.",
                    require_active=False,
                )
        if record.ownership_lost:
            return
        logger.info("Run %s -> %s", run_id, status.value)

    async def persist_current_status(self, run_id: str) -> bool:
        """in-memory run record에 이미 반영된 status를 영속화한다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                logger.warning("persist_current_status called for unknown run %s", run_id)
                return False
            status = record.status
            error = record.error
            stop_reason = record.stop_reason
        persisted = await self._persist_status(record, status, error=error, stop_reason=stop_reason)
        if not persisted and self.heartbeat_enabled and status == RunStatus.success and not record.ownership_lost:
            await self._mark_ownership_lost(
                record,
                reason="Successful completion could not be confirmed in the durable run store.",
                require_active=False,
            )
        return persisted

    async def set_status_if_not_cancelled(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
        persist: bool = True,
    ) -> str | None:
        """durable한 cancellation이 먼저 이기지 않았다면 terminal status를 설정한다."""
        if not persist or not self.heartbeat_enabled or self._store is None:
            await self.set_status(
                run_id,
                status,
                error=error,
                stop_reason=stop_reason,
                persist=persist,
            )
            return None

        try:
            result = await self._call_store_with_retry(
                "finalize_if_not_cancelled",
                run_id,
                lambda: self._store.finalize_if_not_cancelled(
                    run_id,
                    status=status.value,
                    error=error,
                    stop_reason=stop_reason,
                ),
            )
        except Exception:
            async with self._lock:
                record = self._runs.get(run_id)
            if record is not None:
                await self._mark_ownership_lost(
                    record,
                    reason=("The durable store could not confirm whether cancellation or completion won."),
                    require_active=False,
                )
            return None

        if result.cancel_action is not None:
            async with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record.abort_action = result.cancel_action
                    record.abort_event.set()
            return result.cancel_action

        await self.set_status(
            run_id,
            status,
            error=error,
            stop_reason=stop_reason,
            persist=not result.finalized,
        )
        return None

    async def _ensure_delivery_receipt(self, record: RunRecord) -> bool:
        """recovery 도중 zero-delivery receipt를 멱등하게 영속화한다."""
        if self._event_store is None:
            return True
        try:
            await self._event_store.put_if_absent(
                thread_id=record.thread_id,
                run_id=record.run_id,
                event_type="run.delivery",
                category="outputs",
                content={"presented": 0, "paths": [], "by_tool": {}},
            )
            return True
        except Exception:
            logger.warning(
                "Failed to backfill delivery receipt for recovered run %s; preserving its terminal status",
                record.run_id,
                exc_info=True,
            )
            return False

    async def set_finalizing(self, run_id: str, finalizing: bool) -> None:
        """run이 취소 후 cleanup을 수행 중인지 여부를 표시한다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                logger.warning("set_finalizing called for unknown run %s", run_id)
                return
            record.finalizing = finalizing
            record.updated_at = _now_iso()

    async def wait_for_prior_finalizing(
        self,
        thread_id: str,
        run_id: str,
        *,
        poll_interval: float = 0.01,
        abort_event: asyncio.Event | None = None,
    ) -> None:
        """같은 thread의 더 오래된 run들이 취소 후 cleanup을 끝낼 때까지 기다린다."""
        while True:
            async with self._lock:
                found_current = False
                prior_finalizing = False
                for record in self._thread_records_locked(thread_id):
                    if record.run_id == run_id:
                        found_current = True
                        break
                    if record.finalizing:
                        prior_finalizing = True

                if not found_current or not prior_finalizing:
                    return

            if abort_event is None:
                await asyncio.sleep(poll_interval)
                continue
            try:
                await asyncio.wait_for(abort_event.wait(), timeout=poll_interval)
            except TimeoutError:
                continue
            return

    async def has_later_run(self, thread_id: str, run_id: str) -> bool:
        """해당 thread에 더 새로운 in-memory run이 승인됐는지 여부를 반환한다."""
        async with self._lock:
            seen_current = False
            for record in self._thread_records_locked(thread_id):
                if record.run_id == run_id:
                    seen_current = True
                    continue
                if seen_current:
                    return True
        return False

    async def has_later_started_run(self, thread_id: str, run_id: str) -> bool:
        """같은 thread의 더 새로운 run이 이미 state를 진행시켰을 수 있는지 반환한다."""
        async with self._lock:
            seen_current = False
            for record in self._thread_records_locked(thread_id):
                if record.run_id == run_id:
                    seen_current = True
                    continue
                if seen_current and (record.status != RunStatus.pending or record.finalizing):
                    return True
        return False

    async def _persist_model_name(self, run_id: str, model_name: str | None) -> None:
        """model_name 갱신을 backing store에 best-effort로 영속화한다."""
        if self._store is None:
            return
        try:
            await self._call_store_with_retry(
                "update_model_name",
                run_id,
                lambda: self._store.update_model_name(run_id, model_name),
            )
        except Exception:
            logger.warning("Failed to persist model_name update for run %s", run_id, exc_info=True)

    async def update_model_name(self, run_id: str, model_name: str | None) -> None:
        """run의 model name을 갱신한다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                logger.warning("update_model_name called for unknown run %s", run_id)
                return
            record.model_name = model_name
            record.updated_at = _now_iso()
        await self._persist_model_name(run_id, model_name)
        logger.info("Run %s model_name=%s", run_id, model_name)

    async def _request_durable_cancel(
        self,
        run_id: str,
        *,
        action: str,
    ) -> tuple[CancelOutcome, str | None]:
        """cancellation을 기록하고 먼저 이긴 action을 반환한다."""
        if self._store is None:
            return CancelOutcome.unknown, None
        try:
            winning_action = await self._call_store_with_retry(
                "request_cancel",
                run_id,
                lambda: self._store.request_cancel(run_id, action=action),
            )
        except NotImplementedError:
            # durable cancellation 이전의 서드파티 store는 owner에게 알렸다고
            # 속이지 말고 예전의 안전한 동작을 그대로 유지한다.
            logger.info(
                "Run store does not support cross-worker cancellation for run %s",
                run_id,
            )
            return CancelOutcome.lease_valid_elsewhere, None
        except Exception:
            logger.warning(
                "Failed to persist cancellation request for run %s",
                run_id,
                exc_info=True,
            )
            return CancelOutcome.unknown, None

        if winning_action is not None:
            logger.info(
                "Run %s cancellation requested (requested=%s,winner=%s)",
                run_id,
                action,
                winning_action,
            )
            return CancelOutcome.requested, winning_action

        # 호출자의 read와 보호된 cancellation UPDATE 사이의 경쟁에서 completion이
        # 이겼을 수 있다. 요청이 수락됐다고 주장하는 대신 정확한 terminal 결과를
        # API가 보고하도록 다시 읽는다.
        try:
            fresh = await self._store.get(run_id)
        except Exception:
            fresh = None
        if fresh is None:
            return CancelOutcome.unknown, None
        if fresh.get("status") not in ("pending", "running"):
            return CancelOutcome.not_cancellable, None
        # legacy/부분 구현 store는 owner가 살아 있는 상태에서도 요청을 거절할 수
        # 있다. 기존의 lease-conflict 신호를 그대로 유지한다.
        return CancelOutcome.lease_valid_elsewhere, None

    async def _request_remote_cancel(
        self,
        run_id: str,
        *,
        action: str,
    ) -> CancelOutcome:
        """task가 다른 worker에 속한 run의 cancellation을 기록한다."""
        outcome, _ = await self._request_durable_cancel(
            run_id,
            action=action,
        )
        return outcome

    async def _signal_local_cancel(
        self,
        run_id: str,
        *,
        action: str,
    ) -> None:
        """status 영속화나 cleanup 없이 프로세스 로컬 abort 상태만 설정한다."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None or record.status not in (RunStatus.pending, RunStatus.running) or record.abort_event.is_set():
                return

            record.abort_action = action
            record.abort_event.set()
            task_active = record.task is not None and not record.task.done()
            record.finalizing = task_active
            if task_active and record.status == RunStatus.running:
                record.task.cancel()
        logger.info("Run %s cancellation signalled locally (action=%s)", run_id, action)

    async def cancel(self, run_id: str, *, action: str = "interrupt") -> CancelOutcome:
        """run의 cancellation을 요청한다.

        호출이 소유 worker에 도달하면 기존과 동일하게 로컬에서 취소한다(in-memory
        abort + status를 store에 영속화).

        heartbeat가 켜진 multi-worker 배포에서 호출이 비소유 worker에 도달한 경우:

        - **lease 만료** — run의 lease가 grace 임계를 넘었으므로 이 worker가
          소유권을 가져와 ``error``로 표시한다. 소유 worker는 죽은 것으로
          간주한다(heartbeat 갱신이 멈췄다).

        - **lease 유효** — cancellation action을 durable하게 기록한다. owner는
          다음 heartbeat에서 이를 관측하고, 직접 라우팅된 요청과 동일한 로컬
          abort/확정 경로를 수행한다.

        single-worker 모드(``heartbeat_enabled=False``)에서는 메모리에 없는
        store-only hydrate run이 ``not_active_locally``를 반환해 기존 409 동작을
        유지한다.

        Args:
            run_id: 취소할 run ID.
            action: ``"interrupt"``는 checkpoint를 유지하고, ``"rollback"``은
                    run 이전 state로 되돌린다.

        Returns:
            무슨 일이 일어났는지 나타내는 :class:`CancelOutcome` enum.
        """
        # ------------------------------------------------------------------
        # 로컬 경로 — 이 worker가 메모리에서 해당 run을 소유한다.
        # ------------------------------------------------------------------
        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                if record.status == RunStatus.interrupted:
                    return CancelOutcome.cancelled  # 멱등
                if record.status not in (RunStatus.pending, RunStatus.running) and (not self.heartbeat_enabled or self._store is None):
                    return CancelOutcome.not_cancellable

        durable_cancel_won = False
        if record is not None and self.heartbeat_enabled and self._store is not None:
            outcome, winning_action = await self._request_durable_cancel(
                run_id,
                action=action,
            )
            if outcome == CancelOutcome.requested:
                action = winning_action or action
                durable_cancel_won = True
            elif outcome == CancelOutcome.unknown:
                logger.warning(
                    "Proceeding with local cancellation for run %s after durable cancel persistence failed",
                    run_id,
                )
            elif outcome != CancelOutcome.lease_valid_elsewhere:
                return outcome

        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                if record.status == RunStatus.interrupted or record.abort_event.is_set():
                    return CancelOutcome.cancelled
                if record.status not in (RunStatus.pending, RunStatus.running):
                    return CancelOutcome.cancelled if durable_cancel_won else CancelOutcome.not_cancellable
                record.abort_action = action
                record.abort_event.set()
                task_active = record.task is not None and not record.task.done()
                record.finalizing = task_active
                if task_active and record.status == RunStatus.running:
                    record.task.cancel()
                record.status = RunStatus.interrupted
                record.updated_at = _now_iso()

        # store 호출이 다른 변경을 막지 않도록 lock 바깥에서 영속화한다.
        if record is not None:
            persisted = await self._persist_status(record, RunStatus.interrupted)
            if not persisted and self._store is not None:
                # ``_persist_status``는 내부적으로 이미 ``existing``을 조회했다.
                # in-memory cancel과 보호된 ``update_status`` 사이에 peer takeover가
                # row를 ``error``로 바꿨는지 store를 다시 확인한다. 그렇다면
                # ``taken_over``를 반환해 클라이언트가 store와 일관된 status를 보게 한다.
                try:
                    existing = await self._store.get(run_id)
                except Exception:
                    existing = None
                if existing is not None and existing.get("status") == "error":
                    # in-memory ``record.status``는 여전히 ``interrupted``(위 lock
                    # 아래에서 설정)인 반면 store row는 이제 ``error``다. 이 일시적인
                    # 불일치는 무해하다. ``_persist_status``의 guard가 뒤늦은 확정
                    # 쓰기가 takeover를 덮어쓰는 것을 막고, 이후 읽기에서는 store가
                    # 권위 있는 출처다.
                    logger.info("Run %s local cancel superseded by peer takeover", run_id)
                    return CancelOutcome.taken_over
            logger.info("Run %s cancelled (action=%s)", run_id, action)
            return CancelOutcome.cancelled

        if durable_cancel_won:
            return CancelOutcome.cancelled

        # ------------------------------------------------------------------
        # 비로컬 경로 — in-memory record가 없으므로 store를 조회해야 한다.
        # ------------------------------------------------------------------

        if not self.heartbeat_enabled:
            return CancelOutcome.not_active_locally

        if self._store is None:
            return CancelOutcome.unknown

        try:
            row = await self._store.get(run_id)
        except Exception:
            logger.warning("Failed to fetch run %s from store during cancel", run_id, exc_info=True)
            return CancelOutcome.unknown

        if row is None:
            return CancelOutcome.unknown

        store_status = row.get("status")
        if store_status == "interrupted":
            return CancelOutcome.requested
        if store_status not in ("pending", "running"):
            return CancelOutcome.not_cancellable

        grace_seconds = self.grace_seconds
        lease_expires_at: str | None = row.get("lease_expires_at")

        if not is_lease_expired(lease_expires_at, grace_seconds=grace_seconds):
            return await self._request_remote_cancel(run_id, action=action)

        take_over_msg = f"Run reclaimed by worker {self._worker_id}: the owning worker ({row.get('owner_worker_id') or 'unknown'}) stopped renewing its lease and is presumed dead."
        try:
            taken = await self._call_store_with_retry(
                "claim_for_takeover",
                run_id,
                lambda: self._store.claim_for_takeover(
                    run_id,
                    grace_seconds=grace_seconds,
                    error=take_over_msg,
                ),
            )
        except Exception:
            logger.warning("Take-over claim for run %s failed with exception", run_id, exc_info=True)
            return CancelOutcome.unknown

        if taken:
            logger.warning("Run %s taken over by worker %s (action=%s)", run_id, self._worker_id, action)
            return CancelOutcome.taken_over

        # 조건부 UPDATE가 0개 row에 매칭됐다. 원인은 둘이다:
        #   (a) owner가 lease를 갱신했다 → cancellation 요청을 영속화한다.
        #   (b) 우리의 read와 claim 사이에 row가 terminal이 됐다(run이 끝났거나
        #       다른 worker가 이미 가져갔다) → not_cancellable 또는 taken_over.
        # 둘을 구분하려고 다시 읽는다.
        try:
            fresh = await self._store.get(run_id)
        except Exception:
            fresh = None
        if fresh is None:
            return CancelOutcome.unknown
        fresh_status = fresh.get("status")
        if fresh_status not in ("pending", "running"):
            if fresh_status == "error":
                logger.info("Run %s takeover lost to another worker already at error", run_id)
                return CancelOutcome.taken_over
            return CancelOutcome.not_cancellable
        # row가 아직 active다 — takeover가 경쟁하는 동안 owner가 lease를 갱신했다.
        # 라우팅 문제를 409로 드러내는 대신 그 owner에게 알린다.
        return await self._request_remote_cancel(run_id, action=action)

    def _compute_lease_expires_at(self) -> str | None:
        """새로 만든 run의 lease 만료 ISO timestamp를 반환한다.

        heartbeat가 꺼져 있으면(single-worker 모드) ``None``을 반환하므로,
        reconciliation이 죽은 run을 orphan(NULL lease)으로 보고 즉시 회수해
        ownership 도입 이전 동작을 유지한다. multi-worker 배포는 heartbeat를 켜서
        lease를 사용한다.
        """
        if self._run_ownership_config is None:
            return None
        if not self._run_ownership_config.heartbeat_enabled:
            return None
        lease_seconds = self._run_ownership_config.lease_seconds
        return (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()

    async def create_or_reject(
        self,
        thread_id: str,
        assistant_id: str | None = None,
        *,
        on_disconnect: DisconnectMode = DisconnectMode.cancel,
        metadata: dict | None = None,
        kwargs: dict | None = None,
        multitask_strategy: str = "reject",
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> RunRecord:
        """thread에 대한 일반 agent run을 원자적으로 승인한다."""
        return await self._admit_thread_operation(
            thread_id,
            assistant_id,
            operation_kind=ThreadOperationKind.run,
            on_disconnect=on_disconnect,
            metadata=metadata,
            kwargs=kwargs,
            multitask_strategy=multitask_strategy,
            model_name=model_name,
            user_id=user_id,
        )

    async def _close_cancelled_admission(self, record: RunRecord) -> None:
        """호출자에게 전달되지 않은 대체 run을 terminal로 만들고 durable 상태를 확인한다."""
        await self.cancel(record.run_id)
        if self._store is None:
            return

        stored = await self._call_store_with_retry(
            "verify cancelled admission",
            record.run_id,
            lambda: self._store.get(record.run_id, user_id=record.user_id),
        )
        active_statuses = (RunStatus.pending.value, RunStatus.running.value)
        if stored is not None and stored.get("status") in active_statuses:
            # `_persist_status`는 의도적으로 best-effort다. 이 보상 경로는 엄격한
            # 두 번째 CAS 시도가 필요하다. 호출자가 record를 결코 받지 못하고,
            # 반환 이후에는 어떤 worker도 붙을 수 없기 때문이다. peer의 terminal
            # 전이가 CAS에서 이기면 아래에서 그대로 보존된다.
            await self._call_store_with_retry(
                "terminalize cancelled admission",
                record.run_id,
                lambda: self._store.update_status(record.run_id, RunStatus.interrupted.value),
            )
            stored = await self._call_store_with_retry(
                "verify terminal cancelled admission",
                record.run_id,
                lambda: self._store.get(record.run_id, user_id=record.user_id),
            )
            if stored is not None and stored.get("status") in active_statuses:
                raise RuntimeError(f"Cancelled admission {record.run_id} remains active in the run store")

        if stored is None:
            async with self._lock:
                if self._runs.get(record.run_id) is record:
                    self._runs.pop(record.run_id, None)
                    self._unindex_run_locked(record.run_id, record.thread_id)
            return

        stored_status = RunStatus(stored.get("status") or RunStatus.pending.value)
        async with self._lock:
            if self._runs.get(record.run_id) is record:
                record.status = stored_status
                record.error = stored.get("error")
                record.stop_reason = stored.get("stop_reason")
                record.updated_at = _now_iso()

    async def _admit_thread_operation(
        self,
        thread_id: str,
        assistant_id: str | None = None,
        *,
        operation_kind: ThreadOperationKind,
        on_disconnect: DisconnectMode = DisconnectMode.cancel,
        metadata: dict | None = None,
        kwargs: dict | None = None,
        multitask_strategy: str = "reject",
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> RunRecord:
        """진행 중인 run을 원자적으로 확인하고 새 run을 만든다.

        ``reject`` 전략에서는 thread에 이미 pending/running run이 있으면
        ``ConflictError``를 발생시킨다. ``interrupt``/``rollback``에서는 생성 전에
        진행 중인 run을 취소한다.

        lock 순서 불변식: 로컬 확인, store insert, 로컬 등록 전체에 걸쳐 로컬
        ``self._lock``을 잡고 있으므로, 같은 worker에서 ConflictError가 발생하려는
        상황에 store insert가 성공하는 일(= store에 pending row가 새는 일)은 없다.
        프로세스 간 경합은 ``(thread_id) WHERE status IN ('pending','running')``에
        대한 partial unique index로 store 레벨에서 해소된다.
        """
        run_id = str(uuid.uuid4())
        now = _now_iso()

        _supported_strategies = ("reject", "interrupt", "rollback")
        if multitask_strategy not in _supported_strategies:
            raise UnsupportedStrategyError(f"Multitask strategy '{multitask_strategy}' is not yet supported. Supported strategies: {', '.join(_supported_strategies)}")

        lease_expires_at = self._compute_lease_expires_at()
        grace_seconds = self._run_ownership_config.grace_seconds if self._run_ownership_config else 10

        interrupted_records: list[RunRecord] = []
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            status=RunStatus.pending,
            on_disconnect=on_disconnect,
            operation_kind=operation_kind,
            multitask_strategy=multitask_strategy,
            metadata=metadata or {},
            kwargs=kwargs or {},
            user_id=user_id,
            created_at=now,
            updated_at=now,
            model_name=model_name,
            owner_worker_id=self._worker_id,
            lease_expires_at=lease_expires_at,
        )

        async with self._lock:
            # 1) 로컬 진행 중 run 확인(같은 worker용 guard. worker 간에는 아래
            #    store의 partial unique index가 담당한다).
            local_inflight = [r for r in self._thread_records_locked(thread_id) if r.status in (RunStatus.pending, RunStatus.running) or r.finalizing]

            if multitask_strategy in ("interrupt", "rollback") and any(record.operation_kind != ThreadOperationKind.run for record in local_inflight):
                raise ConflictError(f"Thread {thread_id} has an active checkpoint write")

            if multitask_strategy == "reject" and local_inflight:
                raise ConflictError(f"Thread {thread_id} already has an active run")

            if multitask_strategy in ("interrupt", "rollback") and local_inflight:
                logger.info(
                    "Preparing to cancel %d inflight run(s) on thread %s (strategy=%s)",
                    len(local_inflight),
                    thread_id,
                    multitask_strategy,
                )

            # 2) 로컬 lock을 잡은 채로 store에 영속화한다. 프로세스 간 원자성의
            #    source of truth는 store다.
            if self._store is not None:
                if multitask_strategy == "reject":
                    try:
                        await self._call_store_with_retry(
                            "create_thread_operation_atomic",
                            run_id,
                            lambda: self._store.create_thread_operation_atomic(
                                run_id=run_id,
                                thread_id=thread_id,
                                owner_worker_id=self._worker_id,
                                lease_expires_at=lease_expires_at,
                                operation_kind=operation_kind.value,
                                multitask_strategy="reject",
                                assistant_id=assistant_id,
                                user_id=user_id,
                                model_name=model_name,
                                metadata=metadata,
                                kwargs=kwargs,
                                created_at=now,
                                grace_seconds=grace_seconds,
                            ),
                        )
                    except ConflictError:
                        raise
                    except Exception as exc:
                        if _is_unique_violation(exc):
                            raise ConflictError(f"Thread {thread_id} already has an active run") from exc
                        raise
                else:
                    # Interrupt / rollback: store 쪽 claim + insert를 한 transaction
                    # 안에서 처리한다. SELECT FOR UPDATE와 INSERT 사이에 다른 worker가
                    # 끼어들 수 있으므로 IntegrityError에는 재시도한다.
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            await self._call_store_with_retry(
                                "create_thread_operation_atomic",
                                run_id,
                                lambda: self._store.create_thread_operation_atomic(
                                    run_id=run_id,
                                    thread_id=thread_id,
                                    owner_worker_id=self._worker_id,
                                    lease_expires_at=lease_expires_at,
                                    operation_kind=operation_kind.value,
                                    multitask_strategy=multitask_strategy,
                                    assistant_id=assistant_id,
                                    user_id=user_id,
                                    model_name=model_name,
                                    metadata=metadata,
                                    kwargs=kwargs,
                                    created_at=now,
                                    grace_seconds=grace_seconds,
                                ),
                            )
                            break
                        except Exception as exc:
                            is_unique = _is_unique_violation(exc)
                            if is_unique and attempt + 1 < max_retries:
                                continue
                            if is_unique:
                                # unique 위반으로 재시도를 모두 소진했다 — reject
                                # 분기의 계약(500이 아니라 409)에 맞추어
                                # ConflictError로 드러낸다. 근본 원인은 같다:
                                # 다른 worker가 이 thread의 경쟁에서 이겼다.
                                raise ConflictError(f"Thread {thread_id} already has an active run") from exc
                            raise
                    # ``create_thread_operation_atomic``이 같은 transaction 안에서
                    # 선점한 store row를 이미 interrupted로 표시했으므로, 그것들에
                    # 대한 추가 store 쓰기는 필요 없다.

            # 3) store insert가 성공했으니 이제서야 로컬 등록이 안전하다.
            self._runs[run_id] = record
            self._index_run_locked(record)

            # 4) 로컬 in-memory 진행 중 run을 취소한다(interrupt/rollback). store 쪽
            #    대응 row는 2단계에서 이미 취소됐다.
            if multitask_strategy in ("interrupt", "rollback"):
                for r in local_inflight:
                    if r.finalizing:
                        continue
                    r.abort_action = multitask_strategy
                    r.abort_event.set()
                    task_active = r.task is not None and not r.task.done()
                    r.finalizing = task_active
                    if task_active:
                        r.task.cancel()
                    r.status = RunStatus.interrupted
                    r.updated_at = now
                    interrupted_records.append(r)

        # lock 바깥: 로컬에서 취소된 run의 interrupted status를 영속화한다. store 쪽
        # 선점 row는 이미 확정됐다. 이 시점의 cancellation은 대체 run이 승인된 뒤에
        # 발생하므로, 호출자에게 cancellation을 전파하기 전에 그 새 run을 닫는다.
        try:
            for interrupted_record in interrupted_records:
                await self._persist_status(interrupted_record, RunStatus.interrupted)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._close_cancelled_admission(record))
            cleanup.set_name(f"deerflow-close-cancelled-admission-{record.run_id}")
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    break
            try:
                cleanup.result()
            except asyncio.CancelledError:
                logger.error("Cancelled admission cleanup task was itself cancelled for run %s", record.run_id)
            except Exception:
                logger.exception("Failed to close run %s after admission was cancelled", record.run_id)
            raise

        logger.info("Run created: run_id=%s thread_id=%s", run_id, thread_id)
        return record

    @asynccontextmanager
    async def reserve_thread_operation(
        self,
        thread_id: str,
        *,
        kind: ThreadOperationKind,
        user_id: str | None = None,
    ) -> AsyncIterator[None]:
        """run이 아닌 thread operation에 대해 durable한 배타적 승인을 유지한다.

        예약은 수명이 짧은 pending row이므로, ``create_or_reject``가 쓰는 것과 같은
        durable uniqueness constraint가 Gateway worker 간 경쟁의 양쪽을 모두 막는다.
        """
        if kind == ThreadOperationKind.run:
            raise ValueError("Normal runs must be admitted with create_or_reject()")
        record = await self._admit_thread_operation(
            thread_id,
            operation_kind=kind,
            multitask_strategy="reject",
            user_id=user_id,
        )
        try:
            reservation_task = asyncio.current_task()
            if reservation_task is None:
                raise RuntimeError("Thread operation reservation requires an active asyncio task")
            lease_lost = True
            async with self._lock:
                if self._runs.get(record.run_id) is record:
                    record.task = reservation_task
                    lease_lost = record.abort_event.is_set()
            if lease_lost:
                raise asyncio.CancelledError()
            yield
        except asyncio.CancelledError:
            if record.abort_event.is_set():
                raise ConflictError(f"Thread {thread_id} reservation lease was lost") from None
            raise
        finally:
            try:
                if self._store is not None:
                    try:
                        await self._call_store_with_retry(
                            "release thread operation",
                            record.run_id,
                            lambda: self._store.delete_thread_operation(record.run_id, user_id=record.user_id),
                        )
                    except Exception:
                        logger.warning(
                            "Failed to release persisted thread operation %s; leaving it for orphan reconciliation",
                            record.run_id,
                            exc_info=True,
                        )
            finally:
                async with self._lock:
                    removed = self._runs.pop(record.run_id, None)
                    if removed is not None:
                        self._unindex_run_locked(record.run_id, removed.thread_id)

    async def reconcile_orphaned_inflight_runs(
        self,
        *,
        error: str,
        before: str | None = None,
        stop_reason: str | None = None,
    ) -> list[RunRecord]:
        """lease가 만료된 영속 active run을 실패로 표시한다.

        multi-worker 배포(Postgres)에서 Worker A가 소유한 run이 lease 만료 후에도
        ``pending`` / ``running``으로 남아 있다면 Worker A가 죽었거나 네트워크가
        분리된 것이다. lease가 갱신되지 않았으므로 이 worker(B)가 안전하게 선점해
        error 처리할 수 있다.

        lease가 아직 유효한 row는 건너뛴다 — 살아 있는 다른 worker의 것이다. lease가
        NULL인 row(ownership 도입 이전 데이터)도 회수해서 기존 single-worker 복구
        동작과 맞춘다. 후보 스캔은 최적화일 뿐이다. 각 row는 lease를 고려한 조건부
        update로 선점하므로, 스캔 이후의 heartbeat 갱신이 항상 reconciliation을 이긴다.
        """
        if self._store is None:
            return []
        grace_seconds = self._run_ownership_config.grace_seconds if self._run_ownership_config else 10
        try:
            rows = await self._call_store_with_retry(
                "list_inflight_with_expired_lease",
                "*",
                lambda: self._store.list_inflight_with_expired_lease(before=before, grace_seconds=grace_seconds),
            )
        except Exception:
            logger.warning("Failed to list orphaned inflight runs for reconciliation", exc_info=True)
            return []

        recovered: list[RunRecord] = []
        now = _now_iso()
        for row in rows:
            try:
                record = self._record_from_store(row)
            except Exception:
                logger.warning("Failed to map orphaned run row during reconciliation", exc_info=True)
                continue

            async with self._lock:
                live_record = self._runs.get(record.run_id)
                if live_record is not None and live_record.status in (RunStatus.pending, RunStatus.running):
                    # 아직 로컬 task가 소유 중이므로 건너뛴다
                    continue

            try:
                claimed = await self._call_store_with_retry(
                    "claim_for_takeover",
                    record.run_id,
                    lambda: self._store.claim_for_takeover(
                        record.run_id,
                        grace_seconds=grace_seconds,
                        error=error,
                        stop_reason=stop_reason,
                    ),
                )
            except Exception:
                logger.warning("Failed to claim orphaned run %s for reconciliation", record.run_id, exc_info=True)
                continue
            if not claimed:
                logger.info(
                    "Skipped orphaned run %s recovery because the takeover claim no longer matched",
                    record.run_id,
                )
                continue
            record.status = RunStatus.error
            record.error = error
            record.stop_reason = stop_reason
            record.updated_at = now
            if record.operation_kind == ThreadOperationKind.run:
                # zero-delivery receipt를 쓰기 전에 위의 원자적 takeover가 먼저
                # 이겨야 한다. 그렇지 않으면 낡은 스캔이 heartbeat 갱신과 경쟁해
                # 살아 있는 run의 이후 상세 receipt를 영구히 덮어쓸 수 있다.
                # receipt 자체는 best-effort로, event store를 쓸 수 없을 때의 일반
                # terminal delivery와 동일하게 동작한다.
                await self._ensure_delivery_receipt(record)
                recovered.append(record)

        if recovered:
            logger.warning("Recovered %d orphaned inflight run(s) as error", len(recovered))
        return recovered

    async def has_inflight(self, thread_id: str) -> bool:
        """*thread_id*에 pending 또는 running run이 있으면 ``True``를 반환한다."""
        async with self._lock:
            return any(r.operation_kind == ThreadOperationKind.run and (r.status in (RunStatus.pending, RunStatus.running) or r.finalizing) for r in self._thread_records_locked(thread_id))

    async def cleanup(self, run_id: str, *, delay: float = 300) -> None:
        """선택적인 지연 후 run record를 제거한다."""
        if delay > 0:
            await asyncio.sleep(delay)
        async with self._lock:
            record = self._runs.pop(run_id, None)
            if record is not None:
                self._unindex_run_locked(run_id, record.thread_id)
        logger.debug("Run record %s cleaned up", run_id)

    # ------------------------------------------------------------------
    # Lease heartbeat
    # ------------------------------------------------------------------

    @property
    def worker_id(self) -> str:
        """이 worker의 고유 식별자를 반환한다."""
        return self._worker_id

    @property
    def heartbeat_enabled(self) -> bool:
        """heartbeat 백그라운드 task를 돌려야 하면 ``True``를 반환한다."""
        if self._run_ownership_config is None:
            return False
        return self._run_ownership_config.heartbeat_enabled

    @property
    def grace_seconds(self) -> int:
        """설정된 grace seconds를 반환한다.

        현재 모든 호출자는 ``heartbeat_enabled`` 아래에 있고, 이 값은
        ``_run_ownership_config``가 None이면 항상 False다. fallback 값은 Pydantic
        모델 기본값과 같으며, 그 guard 없이 이 property에 도달할 수 있는 미래의
        호출자를 대비한 방어책이다.
        """
        return self._run_ownership_config.grace_seconds if self._run_ownership_config else 10

    @staticmethod
    def _parse_lease_deadline(lease_expires_at: str | None) -> datetime | None:
        """마지막으로 durable하게 확인된 lease 만료 시각을 파싱한다.

        heartbeat 모드에서 deadline이 없거나 형식이 잘못된 것은 안전하지 않다.
        로컬 worker가 소유권을 증명할 수 있는 유한한 구간이 없기 때문이다.
        """
        if lease_expires_at is None:
            return None
        try:
            deadline = datetime.fromisoformat(lease_expires_at)
        except (TypeError, ValueError):
            return None
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        return deadline

    async def _mark_ownership_lost(
        self,
        record: RunRecord,
        *,
        reason: str,
        require_active: bool = True,
    ) -> bool:
        """로컬 run 하나를 fencing하고 실행 task를 취소한다.

        여기서는 store 쓰기를 시도하지 않는다. 마지막으로 확인된 lease가 만료되면
        이 worker는 더 이상 terminal 결과를 발행할 권한이 없다. durable한 terminal
        처리는 peer reconciler가 담당한다.
        """
        task_to_cancel: asyncio.Task | None = None
        async with self._lock:
            current = self._runs.get(record.run_id)
            if current is not record:
                return False
            if require_active:
                if record.status not in (RunStatus.pending, RunStatus.running):
                    return False
                if record.task is not None and record.task.done():
                    return False
            if record.ownership_lost:
                return True
            record.ownership_lost = True
            record.abort_event.set()
            record.status = RunStatus.error
            record.error = reason
            record.updated_at = _now_iso()
            if record.task is not None and not record.task.done() and record.task is not asyncio.current_task():
                task_to_cancel = record.task

        if task_to_cancel is not None:
            task_to_cancel.cancel()
        logger.error("Run %s lost lease ownership; local execution was fenced: %s", record.run_id, reason)
        return True

    async def start_heartbeat(self) -> None:
        """백그라운드 lease 갱신 task를 시작한다.

        ``heartbeat_enabled``가 ``False``이거나 task가 이미 돌고 있으면 아무것도 하지 않는다.
        """
        if not self.heartbeat_enabled:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        self._heartbeat_stop = asyncio.Event()
        task = asyncio.create_task(self._heartbeat_loop())
        task.set_name("deerflow-run-lease-heartbeat")
        self._heartbeat_task = task
        logger.info("Run lease heartbeat started for worker %s", self._worker_id)

    async def stop_heartbeat(self, *, timeout: float = 5.0) -> None:
        """``timeout``초 안에 백그라운드 heartbeat task를 중지한다."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            _, pending = await asyncio.wait(
                (self._heartbeat_task,),
                timeout=max(0.0, timeout),
            )
            if pending:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
        self._heartbeat_task = None
        self._heartbeat_stop = None
        logger.info("Run lease heartbeat stopped for worker %s", self._worker_id)

    async def _heartbeat_loop(self) -> None:
        """주기적으로 lease를 갱신하고 죽은 peer가 남긴 orphan run을 회수한다.

        lease 갱신은 ``lease_seconds / 3``마다 실행한다. reconciliation(죽은 worker가
        소유한 만료 lease 청소)은 ``lease_seconds``마다(3번째 cycle마다) 실행해
        pod 재시작을 기다리지 않고 orphan run을 복구한다.

        두 연산 모두 예외로부터 보호된다. 일시적 실패가 heartbeat task를 죽이면
        어떤 lease도 다시 갱신되지 않고, 결국 모든 active run이 peer에게 orphan으로
        보이기 때문이다.
        """
        if self._run_ownership_config is None or self._heartbeat_stop is None:
            return
        lease_seconds = self._run_ownership_config.lease_seconds
        interval = max(1, lease_seconds // 3)
        stop = self._heartbeat_stop
        cycle = 0

        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                break  # stop event가 설정됐다
            except TimeoutError:
                pass  # interval이 경과했다

            cycle += 1
            try:
                await self._renew_leases()
            except Exception:
                logger.warning("Heartbeat renewal cycle failed", exc_info=True)

            # 3번째 cycle마다(= lease_seconds마다) reconcile한다. 최초 스윕은
            # startup reconciliation(langgraph_runtime)이 담당하고, 이 주기적
            # pass는 재시작 사이에 lease가 만료되는 orphan을 잡는다 — 예를 들어
            # Worker A가 죽고 lease 만료 전에 대체 worker가 시작되면 startup
            # pass는 아직 유효한 lease를 건너뛴다.
            if cycle % 3 == 0:
                self._schedule_orphan_reconciliation()

    async def _renew_leases(self) -> None:
        """로컬이 소유한 lease를 갱신하고, deadline에서는 fail-closed로 동작한다.

        ``RunRecord.lease_expires_at``은 durable 갱신에 성공한 뒤에만 전진하므로
        마지막으로 확인된 소유권 deadline이다. 그 deadline 이전의 일시적 예외는
        허용하지만, deadline을 넘길 때까지 블록되거나 계속 실패하는 호출은 로컬
        run을 fencing한다.
        """
        if self._store is None or self._run_ownership_config is None:
            return
        lease_seconds = self._run_ownership_config.lease_seconds
        cancellations: list[tuple[str, str]] = []

        async with self._lock:
            # 백그라운드 task가 이미 끝난 경우를 빼고, 이 worker가 소유한 모든
            # pending/running run을 갱신한다. task가 아직 생성되지 않은
            # (``task is None``) pending run도 이 worker 관점에서는 여전히 살아
            # 있다 — ``create_thread_operation_atomic``이 row를 넣는 시점과 worker
            # 계층이 agent task를 띄우는 시점 사이에는 짧은 구간이 있다. 여기서
            # 그런 record를 제외했는데 그 구간이 ``lease_seconds``를 넘기면(예:
            # event loop 포화, 새 worker에서의 느린 checkpoint hydrate) 이 worker가
            # 여전히 실행할 의도가 있는데도 peer reconciliation이 run을 orphan으로
            # 회수해 ``error``로 표시한다.
            active_runs = [(rid, record) for rid, record in self._runs.items() if record.status in (RunStatus.pending, RunStatus.running) and record.owner_worker_id == self._worker_id and (record.task is None or not record.task.done())]

        for run_id, record in active_runs:
            confirmed_deadline = self._parse_lease_deadline(record.lease_expires_at)
            if confirmed_deadline is None or confirmed_deadline <= datetime.now(UTC):
                await self._mark_ownership_lost(
                    record,
                    reason="Lease ownership could not be confirmed before the last confirmed lease expired.",
                )
                continue

            remaining = (confirmed_deadline - datetime.now(UTC)).total_seconds()
            new_expiry = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()
            try:
                async with asyncio.timeout(remaining):
                    renewal = await self._call_store_with_retry(
                        "renew_lease",
                        run_id,
                        lambda: self._store.renew_lease(
                            run_id,
                            owner_worker_id=self._worker_id,
                            lease_expires_at=new_expiry,
                        ),
                    )
                if renewal.renewed:
                    if confirmed_deadline <= datetime.now(UTC):
                        await self._mark_ownership_lost(
                            record,
                            reason="Lease renewal completed after the last confirmed lease had already expired.",
                        )
                        continue
                    # lock 없이 쓰는 것은 무해하다. 이 경로가 기존 record에서
                    # 변경하는 필드는 ``lease_expires_at``뿐이라 경쟁할 동시 writer가
                    # 없다(``set_status`` / ``_persist_status``는 다른 필드를 건드린다).
                    # 여기서 ``self._lock``을 다시 잡으면 아무 이득 없이 무관한 run
                    # 변경들과 직렬화될 뿐이다.
                    record.lease_expires_at = new_expiry
                    if renewal.cancel_action is not None:
                        action = renewal.cancel_action
                        if action not in ("interrupt", "rollback"):
                            logger.warning(
                                "Run %s has invalid durable cancel action %r; using interrupt",
                                run_id,
                                action,
                            )
                            action = "interrupt"
                        cancellations.append((run_id, action))
                else:
                    # ``renew_lease``가 False를 반환했다 — 다른 worker가 row를
                    # 선점했다(status가 더 이상 pending/running이 아니거나
                    # ``owner_worker_id``가 바뀌었다). CPU를 낭비하거나 확정 시점에
                    # takeover status를 덮어쓰지 않도록 로컬 task를 멈춘다.
                    async with self._lock:
                        still_active = self._runs.get(run_id) is record and record.status in (RunStatus.pending, RunStatus.running) and record.owner_worker_id == self._worker_id and (record.task is None or not record.task.done())
                    if still_active:
                        logger.warning(
                            "Run %s lease renewal failed (status=%s,owner=%s) – worker likely taken over; aborting local task",
                            run_id,
                            record.status.value,
                            record.owner_worker_id,
                        )
                        await self._mark_ownership_lost(
                            record,
                            reason="The durable store rejected lease renewal for this worker.",
                        )
            except Exception:
                if confirmed_deadline <= datetime.now(UTC):
                    await self._mark_ownership_lost(
                        record,
                        reason="Lease ownership could not be confirmed before the last confirmed lease expired.",
                    )
                else:
                    logger.warning(
                        "Failed to renew lease for run %s before its confirmed deadline; will retry",
                        run_id,
                        exc_info=True,
                    )

        # cancellation status 쓰기와 cleanup은 단 하나뿐인 갱신 loop 바깥에 둔다.
        # 모든 로컬 lease가 갱신 기회를 가진 뒤에는 소유 worker task에 신호만 보내고,
        # 그 task가 정상적인 terminal 처리를 수행한다.
        for run_id, action in cancellations:
            await self._signal_local_cancel(
                run_id,
                action=action,
            )

    async def _reconcile_orphans_periodic(self) -> None:
        """죽은 peer가 소유한 만료 lease를 청소한다.

        ``_heartbeat_loop``이 single-flight 백그라운드 task로 스케줄한다. 덕분에
        store 스캔/status 쓰기와 Gateway callback 모두 lease 갱신 loop 바깥에서
        돈다. 최초 스윕은 startup reconciliation이 처리하고, 이 주기적 pass는
        재시작 사이에 lease가 만료되는 orphan을 잡는다.
        """
        recovered = await self.reconcile_orphaned_inflight_runs(
            error=LEASE_ORPHAN_RECOVERY_ERROR,
            stop_reason=ORPHAN_RECOVERY_STOP_REASON,
        )
        if recovered:
            logger.warning(
                "Periodic reconciliation recovered %d orphaned run(s) as error",
                len(recovered),
            )
            if self._on_orphans_recovered is not None:
                try:
                    await self._on_orphans_recovered(recovered)
                except Exception:
                    logger.warning(
                        "Periodic orphan recovery callback failed for %d run(s): run_ids=%s",
                        len(recovered),
                        [record.run_id for record in recovered],
                        exc_info=True,
                    )

    def _schedule_orphan_reconciliation(self) -> None:
        """이미 실행 중인 pass가 없으면 감독되는 recovery pass를 하나 시작한다."""
        task = self._orphan_recovery_task
        if task is not None and not task.done():
            logger.debug("Skipping periodic orphan reconciliation: previous pass is still running")
            return
        task = asyncio.create_task(self._reconcile_orphans_periodic())
        task.set_name("deerflow-periodic-orphan-recovery")
        self._orphan_recovery_task = task
        task.add_done_callback(self._orphan_reconciliation_done)

    def _orphan_reconciliation_done(self, task: asyncio.Task[None]) -> None:
        """감독되는 single-flight recovery task를 정리하고 결과를 확인한다."""
        if self._orphan_recovery_task is task:
            self._orphan_recovery_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.warning("Periodic orphan reconciliation failed", exc_info=True)

    async def _drain_orphan_recovery_task(self, *, timeout: float) -> None:
        """shutdown 중 감독되는 recovery pass를 제한된 시간 안에서 기다린다."""
        task = self._orphan_recovery_task
        if task is None or task.done():
            return
        _, pending = await asyncio.wait((task,), timeout=max(0.0, timeout))
        if pending:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.warning(
                "Orphan recovery drain exceeded %.1fs on shutdown; cancelled the active pass",
                timeout,
            )

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """프로세스 shutdown 시 진행 중인 모든 run을 취소하고 제한된 시간 동안 기다린다.

        active run에 먼저 신호를 보내서 그들의 취소/cleanup이 제한된 heartbeat
        중지와 겹쳐 진행되게 한다. heartbeat는 stop event를 관측하기 전에 무해한
        마지막 lease 갱신을 한 번 수행할 수 있다.

        chat run은 fire-and-forget 백그라운드 ``asyncio`` task로 실행되며 공유
        checkpointer를 통해 checkpoint를 쓴다. shutdown 시 checkpointer의 자원(예:
        gateway ``AsyncExitStack``이 소유한 postgres connection pool)이 해제되는데,
        그 시점에 run task가 아직 graph 중간이면 langgraph의
        ``AsyncPregelLoop._checkpointer_put_after_previous``가 닫힌 pool을 상대로
        ``finally: await checkpointer.aput(...)``을 실행한다. 그 put은
        langgraph 내부 task에서 돌기 때문에(``run_agent``의 호출 스택이 아니다)
        발생한 ``psycopg_pool.PoolClosed``를 worker가 잡을 수 없고, ``asyncio.run()``
        shutdown 중 처리되지 않은 예외로 드러난다(bytedance/deer-flow issue #3373).

        checkpointer가 닫히기 *전에* 진행 중인 run을 배출하면, ``timeout`` 안에
        정리되는 run은 자원이 아직 열려 있는 동안 마지막 checkpoint를 flush할 수
        있다. 스스로 정리되지 **않은** run만 ``interrupted``로 표시하므로, 배출 중
        완료된(예: ``success``) run은 일괄로 덮어써지지 않고 실제 terminal status를
        유지한다. 마지막 status 영속화를 포함한 배출 전체는 ``timeout``으로
        제한된다. cleanup에 걸린 run이나 DB 부하로 느려진 store가 worker shutdown을
        멈추게 하면 ``app.gateway.app._SHUTDOWN_HOOK_TIMEOUT_SECONDS``가 막고 있는
        signal 재진입 deadlock의 전제 조건이 되기 때문이다. ``timeout`` 이후에도
        active인 run은 로그로 남으며 teardown과 경쟁할 수 있다.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        async with self._lock:
            inflight = [record for record in self._runs.values() if record.status in (RunStatus.pending, RunStatus.running) and record.task is not None and not record.task.done()]
            for record in inflight:
                record.abort_action = "interrupt"
                record.abort_event.set()
                record.task.cancel()  # type: ignore[union-attr]  # filtered above
                # status는 여기가 아니라 배출 이후(아래)에 정한다. 배출 중 스스로
                # 완료된 run은 실제 status를 유지해야 하기 때문이다.

        await self.stop_heartbeat(timeout=max(0.0, deadline - loop.time()))

        if not inflight:
            await self._drain_orphan_recovery_task(timeout=max(0.0, deadline - loop.time()))
            return

        tasks = [record.task for record in inflight]
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - loop.time()))

        # 스스로 정리되지 않은 run(timeout 후에도 pending이거나 cancelled로 끝난 run)에
        # 대해서만 ``interrupted``를 표시/영속화한다. 배출 중 정상적으로 끝난 run은
        # 스스로 설정한 status를 유지한다.
        to_persist: list[RunRecord] = []
        async with self._lock:
            for record in inflight:
                task = record.task
                if task not in pending and not task.cancelled():
                    # 스스로 완료됐다 — 발생한 예외를 꺼내 "never retrieved"로
                    # 보고되지 않게 하고, status는 그대로 유지한다.
                    task.exception()  # type: ignore[union-attr]  # done & not cancelled
                    continue
                if record.status in (RunStatus.pending, RunStatus.running):
                    record.status = RunStatus.interrupted
                    record.updated_at = _now_iso()
                to_persist.append(record)

        # 느린 store(``_call_store_with_retry``는 DB 부하에서 backoff할 수 있다)가
        # shutdown을 ``timeout`` 너머로 밀지 못하도록, 마지막 status 영속화를 남은
        # 예산 안으로 제한한다.
        if to_persist:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning("Run drain budget exhausted before persisting %d interrupted run(s) on shutdown", len(to_persist))
            else:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*(self._persist_status(record, RunStatus.interrupted) for record in to_persist), return_exceptions=True),
                        timeout=remaining,
                    )
                except TimeoutError:
                    logger.warning("Run drain status persistence exceeded the %.1fs budget; %d record(s) may not be persisted", timeout, len(to_persist))
                else:
                    # ``_persist_status``는 best-effort다. 자체 실패를 잡아 로그로
                    # 남기고 ``False``를 반환한다. 부분 실패가 gather에 조용히
                    # 삼켜지지 않고 shutdown 수준에서 run_id와 함께 드러나도록
                    # 집계 결과를 확인한다.
                    for record, result in zip(to_persist, results):
                        if isinstance(result, Exception):
                            logger.warning("Unexpected error persisting interrupted status for run %s during shutdown: %r", record.run_id, result)
                        elif result is False:
                            logger.warning("Could not persist interrupted status for run %s during shutdown", record.run_id)

        if pending:
            logger.warning("Run drain exceeded %.1fs on shutdown; %d run task(s) still active and may race checkpointer teardown", timeout, len(pending))
        logger.info("Drained %d in-flight run(s) on shutdown (%d settled within %.1fs)", len(inflight), len(inflight) - len(pending), timeout)
        await self._drain_orphan_recovery_task(timeout=max(0.0, deadline - loop.time()))


class CancelOutcome(StrEnum):
    """:meth:`RunManager.cancel` 호출의 결과."""

    cancelled = "cancelled"
    requested = "requested"
    taken_over = "taken_over"
    lease_valid_elsewhere = "lease_valid_elsewhere"
    not_cancellable = "not_cancellable"
    not_active_locally = "not_active_locally"
    unknown = "unknown"


class ConflictError(Exception):
    """multitask_strategy=reject인데 thread에 진행 중인 run이 있을 때 발생한다."""


class UnsupportedStrategyError(Exception):
    """아직 구현되지 않은 multitask_strategy 값일 때 발생한다."""
