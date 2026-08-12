"""run 메타데이터 저장소의 추상 인터페이스.

RunManager가 이 인터페이스에 의존한다. 구현체:
- MemoryRunStore: in-memory dict (개발, 테스트)
- 향후: SQLAlchemy ORM 기반 RunRepository

모든 메서드는 사용자 격리를 위해 선택적 user_id를 받는다. user_id가 None이면 사용자
필터링을 적용하지 않는다(단일 사용자 모드).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EditReplayVisibility:
    hidden_source_run_ids: set[str] = field(default_factory=set)
    hidden_attempt_run_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LeaseRenewal:
    """run lease 갱신 결과.

    ``cancel_action``은 lease 소유권을 넘기지 않고 소유 worker에게 durable한 취소 요청을
    전달한다.
    """

    renewed: bool
    cancel_action: str | None = None


@dataclass(frozen=True)
class StatusFinalization:
    """취소가 이기지 않은 경우에만 run을 완료 처리한 결과."""

    finalized: bool
    cancel_action: str | None = None


class RunStore(abc.ABC):
    @abc.abstractmethod
    async def put(
        self,
        run_id: str,
        *,
        thread_id: str,
        assistant_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        status: str = "pending",
        operation_kind: str = "run",
        multitask_strategy: str = "reject",
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        error: str | None = None,
        stop_reason: str | None = None,
        created_at: str | None = None,
        owner_worker_id: str | None = None,
        lease_expires_at: str | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    async def get(
        self,
        run_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        pass

    @abc.abstractmethod
    async def list_by_thread(
        self,
        thread_id: str,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        pass

    async def list_successful_regenerate_sources(
        self,
        thread_id: str,
        *,
        user_id: str | None = None,
    ) -> set[str]:
        """성공한 regenerate로 대체된 원본 run ID들을 반환한다.

        구현체는 thread 전체를 조사해야 하며 평소의 제한된 run 목록 limit을 적용해서는
        안 된다.
        """
        raise NotImplementedError

    async def list_edit_regenerate_runs(
        self,
        thread_id: str,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """thread 하나의 edit-regenerate 시도 run 전부를 오래된 순으로 반환한다."""
        raise NotImplementedError

    async def get_many_by_thread(
        self,
        thread_id: str,
        run_ids: set[str],
        *,
        user_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """thread 하나에 속한 지정된 run들을 일괄 로드한다."""
        raise NotImplementedError

    @abc.abstractmethod
    async def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> bool | None:
        """run 상태를 갱신한다.

        갱신된 row가 없음을 store가 확인할 수 있으면 ``False``를 반환한다. 오래되거나
        가벼운 store는 rowcount를 보고할 수 없을 때 ``None``을 반환할 수 있다.
        """
        pass

    @abc.abstractmethod
    async def start_run(self, run_id: str) -> bool:
        """pending 상태의 run을 원자적으로 running으로 전이시킨다.

        row가 없거나 더 이상 pending이 아니면 ``False``를 반환한다.
        """
        pass

    @abc.abstractmethod
    async def delete(self, run_id: str) -> None:
        pass

    async def delete_thread_operation(self, run_id: str, *, user_id: str | None) -> None:
        """기록된 소유자를 기준으로, 승인된 thread operation을 해제한다.

        기본 구현은 레거시 store 호환을 위한 것이다: 예전 구현은 ``run_id``만 받았다.
        사용자를 인지하는 store는 정리 작업이 주변 request context에 의존하지 않도록 이
        메서드를 override해야 한다.
        """
        await self.delete(run_id)

    @abc.abstractmethod
    async def update_model_name(
        self,
        run_id: str,
        model_name: str | None,
    ) -> None:
        """기존 run의 model_name 필드를 갱신한다."""
        pass

    @abc.abstractmethod
    async def update_run_completion(
        self,
        run_id: str,
        *,
        status: str,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_tokens: int = 0,
        llm_call_count: int = 0,
        lead_agent_tokens: int = 0,
        subagent_tokens: int = 0,
        middleware_tokens: int = 0,
        token_usage_by_model: dict[str, dict[str, int]] | None = None,
        message_count: int = 0,
        last_ai_message: str | None = None,
        first_human_message: str | None = None,
        error: str | None = None,
    ) -> bool | None:
        """최종 완료 필드를 저장한다.

        구현체는 다른 종료 상태를 덮어써서는 안 된다. row가 없거나 이미 충돌하는 종료
        결과를 갖고 있으면 ``False``를 반환한다.
        """
        pass

    async def update_run_progress(
        self,
        run_id: str,
        *,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_tokens: int | None = None,
        llm_call_count: int | None = None,
        lead_agent_tokens: int | None = None,
        subagent_tokens: int | None = None,
        middleware_tokens: int | None = None,
        token_usage_by_model: dict[str, dict[str, int]] | None = None,
        message_count: int | None = None,
        last_ai_message: str | None = None,
        first_human_message: str | None = None,
    ) -> None:
        """run 상태를 바꾸지 않고 실행 중 snapshot을 best-effort로 저장한다."""
        return None

    @abc.abstractmethod
    async def list_pending(self, *, before: str | None = None) -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def list_inflight(self, *, before: str | None = None) -> list[dict[str, Any]]:
        """아직 ``pending`` 또는 ``running``인 저장된 run들을 반환한다."""
        pass

    @abc.abstractmethod
    async def aggregate_tokens_by_thread(self, thread_id: str, *, include_active: bool = False) -> dict[str, Any]:
        """thread 안에서 완료된 run들의 token 사용량을 집계한다.

        다음 키를 가진 dict를 반환한다: total_tokens, total_input_tokens,
        total_output_tokens, total_runs, by_model (model_name → {tokens, runs}),
        by_caller ({lead_agent, subagent, middleware}).
        """
        pass

    @abc.abstractmethod
    async def update_lease(
        self,
        run_id: str,
        *,
        owner_worker_id: str,
        lease_expires_at: str,
    ) -> bool:
        """활성 run의 lease를 갱신한다. 일치하는 row가 없으면 ``False``를 반환한다."""
        pass

    async def renew_lease(
        self,
        run_id: str,
        *,
        owner_worker_id: str,
        lease_expires_at: str,
    ) -> LeaseRenewal:
        """소유권을 갱신하고 durable한 취소 요청이 있으면 함께 반환한다.

        기본 구현은 레거시 ``update_lease`` 메서드를 감싸고 취소 action을 반환하지
        않으므로, 서드파티 store가 백그라운드 읽기를 추가하지 않고도 소스 호환을
        유지한다. 프로세스 간 취소를 지원하는 store는 갱신과 요청 관찰을 원자적으로
        하도록 이 메서드를 override해야 한다.
        """
        renewed = await self.update_lease(
            run_id,
            owner_worker_id=owner_worker_id,
            lease_expires_at=lease_expires_at,
        )
        return LeaseRenewal(renewed=renewed)

    async def request_cancel(self, run_id: str, *, action: str) -> str | None:
        """활성 run의 첫 취소 action을 저장한다.

        구현체는 ``pending``이나 ``running`` row만 갱신해야 하며, 이긴 action을
        반환하거나 일치하는 활성 row가 없으면 ``None``을 반환한다.
        """
        raise NotImplementedError

    async def finalize_if_not_cancelled(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> StatusFinalization:
        """취소가 이기지 않은 한 활성 run을 원자적으로 종료 처리한다.

        호환용 기본 구현은 durable한 취소를 구현하지 않은 store에서도 안전하다.
        """
        updated = await self.update_status(
            run_id,
            status,
            error=error,
            stop_reason=stop_reason,
        )
        return StatusFinalization(finalized=updated is not False)

    @abc.abstractmethod
    async def claim_for_takeover(
        self,
        run_id: str,
        *,
        grace_seconds: int,
        error: str,
        stop_reason: str | None = None,
    ) -> bool:
        """lease가 만료된 활성 run을 원자적으로 ``error``로 표시한다.

        lease가 *grace_seconds*를 넘겨 만료된 row(또는 lease가 NULL인 소유권 도입 이전
        데이터)만 갱신한다. 조건부 WHERE가 호출자의 오래된 lease 읽기와 소유 worker의
        동시 heartbeat 갱신 사이의 경쟁을 막는다. *stop_reason*이 주어지면 같은 원자적
        갱신에서 함께 저장한다.

        다음 경우 ``False``를 반환한다:
          - run이 더 이상 ``pending``/``running``이 아님,
          - lease가 아직 유효함(소유자 heartbeat이 살아 있음),
          - row가 존재하지 않음.
        """
        pass

    @abc.abstractmethod
    async def list_inflight_with_expired_lease(
        self,
        *,
        before: str | None = None,
        grace_seconds: int = 10,
    ) -> list[dict[str, Any]]:
        """lease가 만료된(또는 소유권 도입 이전 row라 NULL인) 활성 run들을 반환한다."""
        pass

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
        """프로세스 간 유일성을 보장하며 활성 thread operation을 원자적으로 생성한다.

        기본 구현은 예전 ``create_run_atomic`` 인터페이스만 구현한 store와의 호환을
        유지한다. 레거시 store는 일반 run row만 지원하므로, 내부 operation 종류를 쓰려면
        이 메서드를 구현해야 한다.

        ``(new_run_dict, claimed_run_dicts)``를 반환한다.
        ``reject`` 전략에서 충돌하면 ``IntegrityError``를 던진다.
        """
        legacy_impl = type(self).create_run_atomic
        if legacy_impl is RunStore.create_run_atomic:
            raise NotImplementedError("RunStore must implement create_thread_operation_atomic() or create_run_atomic()")
        if operation_kind != "run":
            raise NotImplementedError("Legacy RunStore.create_run_atomic() cannot create non-run thread operations")
        return await self.create_run_atomic(
            run_id,
            thread_id=thread_id,
            owner_worker_id=owner_worker_id,
            lease_expires_at=lease_expires_at,
            multitask_strategy=multitask_strategy,
            assistant_id=assistant_id,
            user_id=user_id,
            model_name=model_name,
            metadata=metadata,
            kwargs=kwargs,
            created_at=created_at,
            grace_seconds=grace_seconds,
        )

    async def create_run_atomic(
        self,
        run_id: str,
        *,
        thread_id: str,
        owner_worker_id: str,
        lease_expires_at: str | None,
        multitask_strategy: str = "reject",
        assistant_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        created_at: str | None = None,
        grace_seconds: int = 10,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """일반 run 승인을 위한 deprecated 호환 alias."""
        operation_impl = type(self).create_thread_operation_atomic
        if operation_impl is RunStore.create_thread_operation_atomic:
            raise NotImplementedError("RunStore must implement create_thread_operation_atomic() or create_run_atomic()")
        return await self.create_thread_operation_atomic(
            run_id,
            thread_id=thread_id,
            owner_worker_id=owner_worker_id,
            lease_expires_at=lease_expires_at,
            operation_kind="run",
            multitask_strategy=multitask_strategy,
            assistant_id=assistant_id,
            user_id=user_id,
            model_name=model_name,
            metadata=metadata,
            kwargs=kwargs,
            created_at=created_at,
            grace_seconds=grace_seconds,
        )
