"""메모리 기반 RunStore. database.backend=memory(기본값)일 때와 테스트에서 쓴다.

원래의 RunManager._runs dict 동작과 동등하다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from deerflow.runtime.runs.store.base import LeaseRenewal, RunStore, StatusFinalization


class MemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        # 보조 index: thread_id -> 삽입 순서를 유지하는 run_id 집합(dict를 ordered set으로
        # 쓴다). ``_runs``와 항상 함께 갱신되므로 thread 단위 조회가 메모리에 있는 전체 run을
        # 훑지 않아도 된다. ``RunManager``가 자체 메모리 레코드에 두는 index와 같은 구조다.
        self._runs_by_thread: dict[str, dict[str, None]] = {}

    def _index_run(self, run_id: str, thread_id: str) -> None:
        """보조 index에서 *run_id*를 *thread_id* 아래에 등록한다."""
        self._runs_by_thread.setdefault(thread_id, {})[run_id] = None

    def _unindex_run(self, run_id: str, thread_id: str) -> None:
        """*thread_id* bucket에서 *run_id*를 빼고, bucket이 비면 bucket도 제거한다."""
        bucket = self._runs_by_thread.get(thread_id)
        if bucket is not None:
            bucket.pop(run_id, None)
            if not bucket:
                self._runs_by_thread.pop(thread_id, None)

    async def put(
        self,
        run_id,
        *,
        thread_id,
        assistant_id=None,
        user_id=None,
        model_name=None,
        status="pending",
        operation_kind="run",
        multitask_strategy="reject",
        metadata=None,
        kwargs=None,
        error=None,
        stop_reason=None,
        created_at=None,
        owner_worker_id=None,
        lease_expires_at=None,
    ):
        now = datetime.now(UTC).isoformat()
        existing = self._runs.get(run_id)
        self._runs[run_id] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "user_id": user_id,
            "model_name": model_name,
            "status": status,
            "operation_kind": operation_kind,
            "multitask_strategy": multitask_strategy,
            "metadata": metadata or {},
            "kwargs": kwargs or {},
            "error": error,
            "stop_reason": stop_reason,
            "created_at": created_at or now,
            "updated_at": now,
            "owner_worker_id": owner_worker_id,
            "lease_expires_at": lease_expires_at,
            # ``put``은 멱등한 snapshot 쓰기다. 이전 snapshot 재시도와 경합했을 수 있는
            # 취소 요청을 보존한다.
            "cancel_action": existing.get("cancel_action") if existing else None,
            "cancel_requested_at": existing.get("cancel_requested_at") if existing else None,
        }
        self._index_run(run_id, thread_id)

    async def get(self, run_id, *, user_id=None):
        run = self._runs.get(run_id)
        if run is None:
            return None
        if user_id is not None and run.get("user_id") != user_id:
            return None
        return run

    async def list_by_thread(self, thread_id, *, user_id=None, limit=100):
        # 전체 run을 훑는 대신 thread index로 O(thread 내 run 수) 조회를 한다.
        # ``self._runs.get``은 방어책이다. index에는 남아 있지만 ``_runs``에서는 이미 사라진
        # 낡은 id를 걸러낸다.
        run_ids = self._runs_by_thread.get(thread_id)
        if not run_ids:
            return []
        results = [run for run_id in run_ids if (run := self._runs.get(run_id)) is not None and run.get("operation_kind", "run") == "run" and (user_id is None or run.get("user_id") == user_id)]
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return results[:limit]

    async def list_successful_regenerate_sources(self, thread_id, *, user_id=None):
        run_ids = self._runs_by_thread.get(thread_id) or ()
        sources: set[str] = set()
        for run_id in run_ids:
            run = self._runs.get(run_id)
            if run is None or run.get("operation_kind", "run") != "run" or run.get("status") != "success":
                continue
            if user_id is not None and run.get("user_id") != user_id:
                continue
            source = (run.get("metadata") or {}).get("regenerate_from_run_id")
            if isinstance(source, str) and source:
                sources.add(source)
        return sources

    async def list_edit_regenerate_runs(self, thread_id, *, user_id=None):
        run_ids = self._runs_by_thread.get(thread_id) or ()
        results = []
        for run_id in run_ids:
            run = self._runs.get(run_id)
            if run is None:
                continue
            if user_id is not None and run.get("user_id") != user_id:
                continue
            metadata = run.get("metadata") or {}
            source = metadata.get("regenerate_from_run_id")
            if metadata.get("replay_kind") == "edit" and isinstance(source, str) and source:
                results.append(run)
        results.sort(key=lambda r: r["created_at"])
        return results

    async def get_many_by_thread(self, thread_id, run_ids, *, user_id=None):
        thread_run_ids = self._runs_by_thread.get(thread_id) or ()
        return {run_id: run for run_id in thread_run_ids if run_id in run_ids and (run := self._runs.get(run_id)) is not None and run.get("operation_kind", "run") == "run" and (user_id is None or run.get("user_id") == user_id)}

    async def update_status(self, run_id, status, *, error=None, stop_reason=None):
        run = self._runs.get(run_id)
        if run is None:
            return False
        # 가드: 아직 활성인 row만 전이시킨다. rollback 경로(``interrupted → error`` 확정)를
        # 위해 ``interrupted``도 포함한다.
        if run["status"] not in ("pending", "running", "interrupted"):
            return False
        run["status"] = status
        if error is not None:
            run["error"] = error
        if stop_reason is not None:
            run["stop_reason"] = stop_reason
        run["updated_at"] = datetime.now(UTC).isoformat()
        return True

    async def start_run(self, run_id) -> bool:
        run = self._runs.get(run_id)
        if run is None or run["status"] != "pending":
            return False
        run["status"] = "running"
        run["updated_at"] = datetime.now(UTC).isoformat()
        return True

    async def update_model_name(self, run_id, model_name):
        if run_id in self._runs:
            self._runs[run_id]["model_name"] = model_name
            self._runs[run_id]["updated_at"] = datetime.now(UTC).isoformat()

    async def delete(self, run_id, *, user_id=None):
        run = self._runs.pop(run_id, None)
        if run is not None:
            self._unindex_run(run_id, run["thread_id"])

    async def update_run_completion(self, run_id, *, status, **kwargs):
        run = self._runs.get(run_id)
        if run is None:
            return False
        current_status = run.get("status")
        allowed_sources = {"pending", "running", status}
        if status == "error":
            allowed_sources.add("interrupted")
        if current_status not in allowed_sources:
            return False
        run["status"] = status
        for key, value in kwargs.items():
            if value is not None:
                run[key] = value
        run["updated_at"] = datetime.now(UTC).isoformat()
        return True

    async def update_run_progress(self, run_id, **kwargs):
        if run_id in self._runs and self._runs[run_id].get("status") == "running":
            for key, value in kwargs.items():
                if value is not None:
                    self._runs[run_id][key] = value
            self._runs[run_id]["updated_at"] = datetime.now(UTC).isoformat()

    async def list_pending(self, *, before=None):
        now = before or datetime.now(UTC).isoformat()
        results = [r for r in self._runs.values() if r.get("operation_kind", "run") == "run" and r["status"] == "pending" and r["created_at"] <= now]
        results.sort(key=lambda r: r["created_at"])
        return results

    async def list_inflight(self, *, before=None):
        now = before or datetime.now(UTC).isoformat()
        results = [r for r in self._runs.values() if r["status"] in ("pending", "running") and r["created_at"] <= now]
        results.sort(key=lambda r: r["created_at"])
        return results

    async def aggregate_tokens_by_thread(self, thread_id: str, *, include_active: bool = False) -> dict[str, Any]:
        statuses = ("success", "error", "running") if include_active else ("success", "error")
        # 프로세스 안의 모든 run을 훑는 대신 thread index로 O(thread 내 run 수) 조회를 한다
        # (``list_by_thread``와 같은 방식).
        run_ids = self._runs_by_thread.get(thread_id) or ()
        completed = [run for run_id in run_ids if (run := self._runs.get(run_id)) is not None and run.get("operation_kind", "run") == "run" and run.get("status") in statuses]
        by_model: dict[str, dict] = {}
        for r in completed:
            usage_by_model = r.get("token_usage_by_model") or {}
            if usage_by_model:
                for model, usage in usage_by_model.items():
                    entry = by_model.setdefault(model, {"tokens": 0, "runs": 0})
                    entry["tokens"] += usage.get("total_tokens", 0)
                    entry["runs"] += 1
            else:
                # 모델별 집계가 도입되기 전에 쓰인 row를 위한 폴백. run 전체를 단일
                # ``model_name``에 귀속시킨다. 옛 데이터를 조용히 버리는 대신 레거시
                # lead-only 동작을 유지한다.
                model = r.get("model_name") or "unknown"
                entry = by_model.setdefault(model, {"tokens": 0, "runs": 0})
                entry["tokens"] += r.get("total_tokens", 0)
                entry["runs"] += 1
        return {
            "total_tokens": sum(r.get("total_tokens", 0) for r in completed),
            "total_input_tokens": sum(r.get("total_input_tokens", 0) for r in completed),
            "total_output_tokens": sum(r.get("total_output_tokens", 0) for r in completed),
            "total_runs": len(completed),
            "by_model": by_model,
            "by_caller": {
                "lead_agent": sum(r.get("lead_agent_tokens", 0) for r in completed),
                "subagent": sum(r.get("subagent_tokens", 0) for r in completed),
                "middleware": sum(r.get("middleware_tokens", 0) for r in completed),
            },
        }

    # ------------------------------------------------------------------
    # multi-worker run ownership 메서드
    # ------------------------------------------------------------------

    async def update_lease(
        self,
        run_id: str,
        *,
        owner_worker_id: str,
        lease_expires_at: str,
    ) -> bool:
        run = self._runs.get(run_id)
        if run is None:
            return False
        if run["status"] not in ("pending", "running"):
            return False
        if run.get("owner_worker_id") != owner_worker_id:
            return False
        run["owner_worker_id"] = owner_worker_id
        run["lease_expires_at"] = lease_expires_at
        run["updated_at"] = datetime.now(UTC).isoformat()
        return True

    async def renew_lease(
        self,
        run_id: str,
        *,
        owner_worker_id: str,
        lease_expires_at: str,
    ) -> LeaseRenewal:
        # ``update_lease``를 거쳐 위임한다. 그래야 레거시 primitive를 override 하는 가벼운
        # subclass와 테스트가 같은 동작을 유지한다.
        renewed = await self.update_lease(
            run_id,
            owner_worker_id=owner_worker_id,
            lease_expires_at=lease_expires_at,
        )
        if not renewed:
            return LeaseRenewal(renewed=False)
        run = self._runs.get(run_id)
        return LeaseRenewal(
            renewed=True,
            cancel_action=run.get("cancel_action") if run is not None else None,
        )

    async def request_cancel(self, run_id: str, *, action: str) -> str | None:
        if action not in ("interrupt", "rollback"):
            raise ValueError(f"Unsupported cancellation action: {action}")
        run = self._runs.get(run_id)
        if run is None or run["status"] not in ("pending", "running"):
            return None
        if run.get("cancel_action") is None:
            run["cancel_action"] = action
            run["cancel_requested_at"] = datetime.now(UTC).isoformat()
        run["updated_at"] = datetime.now(UTC).isoformat()
        return run["cancel_action"]

    async def finalize_if_not_cancelled(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> StatusFinalization:
        run = self._runs.get(run_id)
        if run is None:
            return StatusFinalization(finalized=False)
        if run.get("cancel_action") is not None:
            return StatusFinalization(
                finalized=False,
                cancel_action=run["cancel_action"],
            )
        if run["status"] not in ("pending", "running"):
            return StatusFinalization(finalized=False)
        run["status"] = status
        if error is not None:
            run["error"] = error
        if stop_reason is not None:
            run["stop_reason"] = stop_reason
        run["updated_at"] = datetime.now(UTC).isoformat()
        return StatusFinalization(finalized=True)

    async def claim_for_takeover(
        self,
        run_id: str,
        *,
        grace_seconds: int,
        error: str,
        stop_reason: str | None = None,
    ) -> bool:
        from deerflow.utils.time import is_lease_expired

        run = self._runs.get(run_id)
        if run is None:
            return False
        if run["status"] not in ("pending", "running"):
            return False
        lease = run.get("lease_expires_at")
        if not is_lease_expired(lease, grace_seconds=grace_seconds):
            return False
        run["status"] = "error"
        run["error"] = error
        if stop_reason is not None:
            run["stop_reason"] = stop_reason
        run["updated_at"] = datetime.now(UTC).isoformat()
        return True

    async def list_inflight_with_expired_lease(
        self,
        *,
        before: str | None = None,
        grace_seconds: int = 10,
    ) -> list[dict[str, Any]]:
        now_dt = datetime.fromisoformat(before) if before else datetime.now(UTC)
        cutoff = datetime.now(UTC) - timedelta(seconds=grace_seconds)
        results = []
        for r in self._runs.values():
            if r["status"] not in ("pending", "running"):
                continue
            created_at = r.get("created_at", "")
            if not created_at:
                continue
            try:
                created_dt = datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                continue
            if created_dt > now_dt:
                continue
            lease = r.get("lease_expires_at")
            if lease is None:
                # ownership 도입 이전 row: lease가 없으면 orphan이다
                results.append(r)
            else:
                try:
                    lease_dt = datetime.fromisoformat(lease)
                    # naive 값은 UTC로 취급한다. SQL store의 ``coerce_iso``와 같은 규칙이며,
                    # SQLite(읽을 때 tzinfo를 버린다)에서 heartbeat를 켰을 때 aware인
                    # ``cutoff``와 비교하다 ``TypeError``가 나지 않게 한다.
                    if lease_dt.tzinfo is None:
                        lease_dt = lease_dt.replace(tzinfo=UTC)
                    if lease_dt < cutoff:
                        results.append(r)
                except (ValueError, TypeError):
                    results.append(r)
        results.sort(key=lambda r: r["created_at"])
        return results

    async def create_thread_operation_atomic(
        self,
        run_id: str,
        *,
        thread_id: str,
        owner_worker_id: str,
        lease_expires_at: str | None,
        operation_kind: str = "run",
        multitask_strategy: str = "reject",
        assistant_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        created_at: str | None = None,
        grace_seconds: int = 10,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from deerflow.runtime.runs.manager import ConflictError

        now = datetime.now(UTC).isoformat()
        cutoff = datetime.now(UTC) - timedelta(seconds=grace_seconds)

        # reject 전략: 활성 run이 하나라도 있는지 확인한다
        if multitask_strategy == "reject":
            for r in self._runs.values():
                if r["thread_id"] == thread_id and r["status"] in ("pending", "running"):
                    raise ConflictError(f"Thread {thread_id} already has an active run")

        # interrupt/rollback 전략: 진행 중인 run을 claim 한다.
        # 메모리 경로가 SQL store의 트랜잭션 의미를 그대로 따르도록 2-pass로 처리한다. 후보 중
        # 하나라도 다른 worker가 소유한 살아 있는 run이면, 앞선 후보를 이미 변경한 상태가 아닌
        # 채로 ConflictError를 raise 해야 한다. 즉시 변경하면 raise 시점에 store가 절반만
        # interrupt된 상태로 남아, raise가 트랜잭션 전체를 되돌리는 SQL과 달라진다.
        claimed = []
        if multitask_strategy in ("interrupt", "rollback"):
            candidates: list[dict[str, Any]] = []
            for r in self._runs.values():
                if r["thread_id"] != thread_id:
                    continue
                if r["status"] not in ("pending", "running"):
                    continue
                lease_expired = False
                existing_lease = r.get("lease_expires_at")
                if existing_lease is not None:
                    try:
                        lease_dt = datetime.fromisoformat(existing_lease)
                        # naive 값은 UTC로 취급한다. SQL store 및 ``coerce_iso``와 같은
                        # 규칙이며, aware인 ``cutoff``와 비교할 때 ``TypeError``가 나지
                        # 않게 한다.
                        if lease_dt.tzinfo is None:
                            lease_dt = lease_dt.replace(tzinfo=UTC)
                        lease_expired = lease_dt < cutoff
                        if lease_dt >= cutoff and r.get("owner_worker_id") != owner_worker_id:
                            # 다른 worker가 소유한 살아 있는 run이다. interrupt 할 수 없고,
                            # partial unique index도 어차피 INSERT를 거부한다. 호출자가 깔끔한
                            # 신호를 받도록 ConflictError로 노출한다. store가 그대로 남도록
                            # 어떤 변경보다 먼저 raise 한다.
                            raise ConflictError(f"Thread {thread_id} already has an active run owned by another worker")
                    except (ValueError, TypeError):
                        pass
                if r.get("operation_kind", "run") != "run" and not lease_expired:
                    raise ConflictError(f"Thread {thread_id} has an active checkpoint write")
                candidates.append(r)
            for r in candidates:
                r["status"] = "interrupted"
                r["error"] = "Cancelled by newer run"
                r["owner_worker_id"] = owner_worker_id
                r["updated_at"] = now
                claimed.append(r)

        new_row = {
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "user_id": user_id,
            "model_name": model_name,
            "status": "pending",
            "operation_kind": operation_kind,
            "multitask_strategy": multitask_strategy,
            "metadata": metadata or {},
            "kwargs": kwargs or {},
            "error": None,
            "owner_worker_id": owner_worker_id,
            "lease_expires_at": lease_expires_at,
            "cancel_action": None,
            "cancel_requested_at": None,
            "created_at": created_at or now,
            "updated_at": now,
        }
        self._runs[run_id] = new_row
        self._index_run(run_id, thread_id)
        return new_row, claimed
