"""Memory manager 계약 + 교체 가능한 backend factory.

memory 패키지의 backend 중립 코어다. 모든 backend가 구현하는
:class:`MemoryManager` 인터페이스(9개 메서드)와, ``MemoryConfig.manager_class``로
활성 backend를 결정하는 싱글턴 factory :func:`get_memory_manager`를 정의한다.

backend 교체는 ``MANAGER_CLASS``를 노출하는 ``backends/<name>/`` 폴더를 넣고
``manager_class: <name>``으로 설정하면 끝이다. deer-flow의 다른 코드는 바뀌지 않는다.

범위 참고: 이 단계는 *교체 가능*까지이며 black-box는 아니다. 에이전트 쪽 규약
(호출 지점의 ``enabled`` 게이팅, ``_get_memory_context``의 ``<memory>`` 래핑)은
그대로 둔다. backend 중립적이라 교체 가능성을 해치지 않는다.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from abc import abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deerflow.config.memory_config import get_memory_config

logger = logging.getLogger(__name__)

# backend 패키지는 <이 디렉터리>/backends/<name>/ 아래에 둔다.
_BACKENDS_DIR = Path(__file__).parent / "backends"
# 각 backend의 __init__이 노출하는 sentinel 속성(MemoryManager 하위 클래스).
_MANAGER_CLASS_ATTR = "MANAGER_CLASS"

# 싱글턴 인스턴스 + backend 레지스트리 캐시(reset_memory_manager가 함께 비운다).
# _manager_lock은 멀티스레드 환경에서 get_memory_manager()의 double-checked 초기화를 보호한다.
_memory_manager: MemoryManager | None = None
_backends_cache: dict[str, type[MemoryManager]] | None = None
_manager_lock = threading.Lock()


class MemoryCallbacks:
    """memory backend용 observability hook 모음.

    기본 구현은 모두 no-op이며 필요한 것만 override한다. LLM 호출 직전 hook인
    ``on_memory_llm_call``은 ``invoke_config``를 변형해 tracer(예: langfuse)가
    LLM 경계에서 span을 남기게 한다. (post-extract / search / inject / error 등
    추가 hook은 호출자가 필요로 할 때 넣는다.)
    """

    def on_memory_llm_call(
        self,
        invoke_config: dict[str, Any],
        *,
        thread_id: str | None,
        user_id: str | None,
        trace_id: str | None,
        model_name: str | None,
    ) -> None:
        """LLM 호출 직전에 ``invoke_config``를 변형한다(예: trace metadata 병합).

        기본 구현은 no-op이다.
        """


class MemoryManagerError(RuntimeError):
    """MemoryManager 경계에서 노출하는 backend 중립 기본 에러."""


class MemoryConflictError(MemoryManagerError):
    """optimistic-concurrency 경쟁에서 밀려난 쓰기다."""


class MemoryCorruptionError(MemoryManagerError):
    """저장된 memory를 안전하게 읽을 수 없다."""


class MemoryManager(BaseModel):
    """backend 중립 memory manager 계약.

    순수 ``ABC``가 아니라 pydantic ``BaseModel``이다. 필드 검증과 직렬화를 공짜로
    얻고 backend config(예: ``DeerMemConfig``)와 pydantic v2 타입 시스템을 공유하기
    위해서다. 하위 클래스는 여전히 ``@abstractmethod``를 구현해야 한다. pydantic의
    ``ModelMetaclass``가 ``ABCMeta``를 상속하므로 미구현 abstractmethod는 인스턴스
    생성 시점에 ``TypeError``를 낸다(memory는 영속 상태라 ``add`` / ``get_context``가
    빠진 backend는 심각한 버그이며 생성 시점에 잡아야 한다). backend 전용 의존성
    (storage / llm / queue 등)은 필드가 아니라 ``model_post_init``(또는
    ``from_config``)에서 설정하는 ``PrivateAttr``이며 검증·직렬화 대상에서 제외된다.

    memory는 ``(agent_name, user_id)`` 단위로 버킷을 나누고, ``thread_id``는 deer-flow
    대화 thread와 일치한다. 서드파티 memory 시스템을 deer-flow 코드 수정 없이 붙일 수
    있도록 계약을 의도적으로 중립적으로 유지한다.

    - :meth:`get_context`는 주입용 평문 텍스트를 반환한다. *포맷*은 구현체가 정하며
      계약의 일부가 아니다(DeerMem은 로드 후 ``format_memory_for_injection``을 쓰지만,
      다른 backend는 자체 search + 포매팅을 해도 된다).
    - :meth:`add` / :meth:`add_nowait`는 원본 대화 메시지를 받는다. 필터링이나
      correction/reinforcement 탐지는 구현체의 내부 사정이며 계약에 없다.
    - facts 모델을 가정하지 않는다. backend가 "fact"를 저장하지 않아도 된다.

    메서드는 계층으로 나뉜다. tier-1(``add`` / ``get_context``)은 ``@abstractmethod``이고,
    tier-2 관리 연산과 tier-3 선택 hook은 기본 구현(``NotImplementedError`` 발생 또는
    no-op)을 갖는다. 덕분에 backend는 지원하는 것만 구현하면 된다. ``delete_memory`` /
    ``export_memory``는 호출자가 없는 죽은 계약이지만(``/memory/export``는 ``get_memory``로
    간다) 기본 raise 형태로 남겨 둔다.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # backend 전용 config(factory가 그대로 넘긴다). 파싱이 필요한 backend
    # (DeerMem -> DeerMemConfig)는 model_post_init / from_config에서 처리한다.
    # None은 {}로 강제 변환해 zero-config ``Backend(backend_config=None)``도 유효하게
    # 둔다(그렇지 않으면 BaseModel이 dict 필드의 None을 거부한다).
    backend_config: dict[str, Any] = Field(default_factory=dict)
    # 동작 모드는 host의 ``MemoryConfig.mode``("middleware" | "tool")를 그대로 반영한다.
    # factory가 ``cfg.mode``를 넘긴다. invariant validator는 tool 모드 backend에
    # search 지원을 요구한다.
    mode: Literal["middleware", "tool"] = "middleware"
    # observability callbacks(선택, ``MemoryCallbacks`` 인스턴스). factory가
    # ``LangfuseMemoryCallbacks``를 주입해 memory-LLM 호출이 langfuse에 드러나게 한다.
    # None이면 callbacks 없음(직접 생성 / standalone). backend는 이를 LLM 경로로 넘기고
    # invoke 전에 ``on_memory_llm_call``을 호출한다.
    callbacks: MemoryCallbacks | None = None

    @field_validator("backend_config", mode="before")
    @classmethod
    def _coerce_backend_config(cls, value: Any) -> dict[str, Any]:
        """None(zero-config)은 빈 dict로 받고, dict는 그대로 둔다."""
        return value or {}

    # search 지원 플래그(필드가 아니라 ClassVar). backend가 search()를 override할 때만
    # True로 둔다. invariant validator가 이 플래그와 실제 override 여부
    # (type(self).search is not MemoryManager.search)의 일치를 검사하므로 둘이 어긋날 수
    # 없다. mode="tool"에 필수다. tool 모드에서 에이전트가 memory_search를 호출하므로
    # search 없는 backend는 설정 오류이며, 빈 결과를 조용히 반환하는 대신 생성 시점에
    # 즉시 실패해야 한다. 기본값 False라 새 backend는 tool 모드를 명시적으로 opt-in한다.
    supports_search: ClassVar[bool] = False
    # fact CRUD 대신 대화 단위 추출에 의존하는 backend는 tool 모드가 query 기반 search를
    # 제공하는 동안에도 MemoryMiddleware 쓰기를 유지할 수 있다. 대부분의 backend는 tool
    # 모드를 전적으로 모델 주도로 둔다.
    requires_passive_writes_in_tool_mode: ClassVar[bool] = False

    @model_validator(mode="after")
    def _check_invariants(self) -> MemoryManager:
        """모든 backend가 인스턴스 생성 시 만족해야 하는 필드 간 invariant.

        base model에 있으므로 factory 경로는 물론 factory를 거치지 않은 직접 생성에도
        적용된다. DeerMem 전용 invariant(예: storage_path가 디렉터리인지)는
        ``DeerMemConfig``에 남긴다.

        ``supports_search``(ClassVar 플래그)는 ``search()``의 실제 override 여부와
        일치해야 한다. 선언과 구현이 어긋나는 것을 막기 위해서다. ``search()``를
        override하고 ``supports_search = True``를 빠뜨렸거나(반대로 override 없이
        플래그만 세웠거나) 하는 것은 버그이며, 오해를 부르는 tool 모드 거부나 첫
        ``memory_search`` 호출 시의 런타임 ``NotImplementedError`` 대신 생성 시점에 잡는다.
        """
        search_overridden = type(self).search is not MemoryManager.search
        if type(self).supports_search != search_overridden:
            raise ValueError(
                f"{type(self).__name__}.supports_search={type(self).supports_search} "
                f"is inconsistent with search(): search() is "
                f"{'overridden' if search_overridden else 'inherited (not implemented)'}. "
                f"Set supports_search={search_overridden} on the backend to match."
            )
        if self.mode == "tool" and not search_overridden:
            raise ValueError(
                f"memory mode='tool' requires a backend that implements search(), but {type(self).__name__} does not override search(). Use mode='middleware' or a backend that overrides search() (and sets supports_search=True)."
            )
        return self

    # ── Tier 1: @abstractmethod ─────────────────────────────────────────
    # 모든 backend가 반드시 구현한다(쓰기 + 읽기-주입은 backend의 기본 책무다).
    # 누락은 심각한 버그이며(memory는 영속 상태다) @abstractmethod가 생성 시점에 잡는다.
    # noop backend는 이들을 no-op / "" 로 구현한다.
    @abstractmethod
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """대화를 memory 갱신 큐에 넣는다(debounce, 비동기).

        Args:
            thread_id: 대화 thread id.
            messages: 원본 대화 메시지. 사용자 입력과 최종 assistant 응답만 남기는
                필터링은 구현체가 직접 한다.
            agent_name: agent별 버킷. ``None``이면 전역 memory다.
            user_id: user별 버킷.
            trace_id: memory-LLM tracing용으로 캡처한 request trace id.
        """

    @abstractmethod
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """해당 버킷의 주입 가능한 memory 텍스트를 반환한다.

        구현체가 알아서 memory를 로드하고 포매팅하며, 반환된 문자열은 호출 지점에서
        그대로 주입된다. 포맷 파라미터는 생성 시 ``backend_config``로 받은 backend
        전용 config이지 이 메서드의 host config가 아니다.
        """

    # ── Tier 2: 기본 구현이 있는 관리 연산 ────────────────────────────
    # 지원하지 않는 연산은 직접 raise를 쓰지 않고 기본 구현(``NotImplementedError``)을
    # 그대로 물려받는다. ``add_nowait``는 기본적으로 ``add``에 위임한다(debounce 큐가
    # 없는 backend에는 "즉시"와 "대기"의 구분이 없다). ``shutdown_flush``는 기본 True라
    # (버퍼가 없는 backend는 비울 것이 없다) host가 조건 없이 호출할 수 있다.
    # ``delete_memory`` / ``export_memory``는 호출자가 없는 죽은 계약이지만
    # (/memory/export는 get_memory로 간다) 기본 raise 덕분에 backend 구현을 강제하지 않고
    # 남겨 둘 수 있다.
    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """대화를 *즉시* memory 갱신 큐에 넣는다(비상 flush).

        summarization이 state에서 메시지를 제거하기 직전에 호출해 내용이 유실되지 않게
        한다. 기본 구현은 :meth:`add`에 위임한다(debounce 큐가 있는 backend는 nowait
        우선순위로 enqueue하도록 override한다).
        """
        self.add(thread_id, messages, agent_name=agent_name, user_id=user_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """버킷의 memory에서 ``query``에 맞는 fact를 찾아 관련도순으로 최대 ``top_k``개 반환한다.

        선택 인자 ``category``는 ``top_k`` 절단 *이전에* 필터링하므로, 다른 카테고리의
        상위 fact에 밀려 category 범위 검색이 빈손이 되지 않는다. 기본 구현은 미지원(raise)이며,
        retrieval을 갖춘 backend는 override와 함께 ``supports_search = True``를 설정한다
        (``mode='tool'``에 필수).
        """
        raise NotImplementedError(f"search not supported by {type(self).__name__}")

    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """버킷의 전체 memory 문서를 반환한다. 기본 구현은 미지원이다."""
        raise NotImplementedError(f"get_memory not supported by {type(self).__name__}")

    def delete_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """버킷의 memory 문서 전체를 삭제한다. 기본 구현은 미지원이다(호출자가 없는 죽은 계약)."""
        raise NotImplementedError(f"delete_memory not supported by {type(self).__name__}")

    def clear_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """버킷의 memory를 비우고, 비워진(이제 빈) 문서를 반환한다.

        ``agent_name=None``이면 해당 user가 소유한 모든 memory를 뜻한다. agent 이름을
        명시하면 그 agent의 memory만 비우고 공유되는 user 단위 summary는 보존해야 한다.
        기본 구현은 미지원(``NotImplementedError``)이며, 지원하는 backend가 override한다.
        """
        raise NotImplementedError(f"clear_memory not supported by {type(self).__name__}")

    def import_memory(
        self,
        memory_data: dict[str, Any],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """memory 문서를 버킷으로 import하고 병합 결과를 반환한다. 기본 구현은 미지원이다."""
        raise NotImplementedError(f"import_memory not supported by {type(self).__name__}")

    def export_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """버킷의 memory 문서를 export한다.

        기본 구현은 미지원이다(호출자가 없는 죽은 계약이며 /memory/export는 get_memory로 간다).
        """
        raise NotImplementedError(f"export_memory not supported by {type(self).__name__}")

    def shutdown_flush(self, timeout: float) -> bool:
        """graceful shutdown 시 대기 중인 갱신을 시간 제한 안에서 best-effort로 비운다.

        Gateway shutdown 경로에서(IM channel과 scheduler가 멈춘 뒤라 drain 도중 새
        IM/scheduler 갱신이 들어오지 않는다) backend의 debounce 버퍼에 남은 갱신을 flush한다.
        이게 없으면 마지막 timer 발화 이후 쌓인 갱신은 재시작 / rolling deploy / SIGTERM 때
        유실된다. 버퍼가 순수 in-memory이고 debounce worker가 프로세스 종료 시 죽는 daemon
        thread이기 때문이다.

        구현체는 ``timeout``을 *엄격하게* 지켜야 한다. drain은 중단할 수 없는 동기 LLM 호출을
        하므로 호출자(Gateway lifespan)에게는 K8s ``terminationGracePeriodSeconds``와 맞물리는
        실질적인 상한이 필요하다. drain이 pod grace window 안에 끝나지 않으면 K8s가 도중에
        SIGKILL하고, drain이 막으려던 유실이 조용히 다시 발생한다.

        Returns:
            ``timeout`` 안에 drain이 실제로 끝났으면(버퍼가 비었고, 실행 중인 worker가 없고,
            예외도 없으면) ``True``. timeout이나 실패면 ``False``(호출자는 경고를 남기고 종료를
            진행한다. 끝내지 못한 잔여분은 버려지지만 아예 flush하지 않는 것보다는 낫다).
            기본값은 ``True``다(버퍼가 없는 backend는 비울 것이 없다). 덕분에 host는 memory가
            켜져 있으면 조건 없이 호출할 수 있고, debounce 큐가 있는 backend는 ``timeout`` 안에
            flush하도록 override한다.
        """
        return True

    # ── Tier 3: 선택 hook ──────────────────────────────────────────
    # A급: 에이전트 쪽에 실제 호출자가 있다(startup warm-up, 수동 reload, fact CRUD).
    # 예전에는 ``hasattr`` 탐지로 접근했지만, 이제 기본 구현과 함께 계약으로 두어
    # 호출자가 직접 호출하고 ``NotImplementedError``를 잡는다.
    # ``warm``의 기본값은 None(warm할 것 없음)이고 나머지는 raise다.
    def warm(self) -> bool | None:
        """startup 시 backend 리소스를 미리 준비한다(예: tiktoken encoding 캐시).

        Gateway lifespan이 event loop 밖에서 호출한다.

        반환 계약은 host가 정확한 로그를 남기도록 3-상태다.

        * ``True``  — warm 성공(또는 이미 캐시됨 / 불필요).
        * ``False`` — warm을 시도했으나 실패(host가 fallback한다).
        * ``None``  — 이 backend는 warm할 것이 없음(기본값). host는 오해를 부르는
          "warmed successfully" 대신 "skipping"을 남기므로, DeerMem이 아닌 backend가
          건드린 적 없는 tiktoken 캐시를 준비했다고 주장하지 않는다.

        일회성 초기화 비용이 큰 backend는 override해서 ``True``/``False``를 반환한다.
        """
        return None

    def reload_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """캐시된 memory 문서를 버리고 storage에서 다시 읽는다.

        기본 구현은 미지원이며 호출자는 :meth:`get_memory`로 fallback한다. 캐시가 있는
        backend가 override한다.
        """
        raise NotImplementedError(f"reload_memory not supported by {type(self).__name__}")

    def create_fact(
        self,
        content: str,
        category: str = "context",
        confidence: float = 0.5,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """fact 하나를 수동으로 추가한다.

        ``(memory_data, fact_id)``를 반환하며, storage 상한 때문에 방금 추가한 fact가
        밀려났으면 ``fact_id``는 None이다. 기본 구현은 미지원이다.
        """
        raise NotImplementedError(f"create_fact not supported by {type(self).__name__}")

    def delete_fact(
        self,
        fact_id: str,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """id로 fact 하나를 삭제한다. 기본 구현은 미지원이다."""
        raise NotImplementedError(f"delete_fact not supported by {type(self).__name__}")

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """id로 fact 하나를 갱신한다(생략된 필드는 보존). 기본 구현은 미지원이다."""
        raise NotImplementedError(f"update_fact not supported by {type(self).__name__}")

    # B급: 아직 에이전트 쪽 호출자가 없다. 향후 시나리오를 위한 시그니처만 둔다.
    # 기본 no-op이라 호출자가 게이팅 없이 호출할 수 있다. (자기 완결형 hook인
    # on_delegation / on_session_end / on_memory_write는 의도적으로 계약에 넣지 않았다.
    # 호출자나 이벤트 소스가 없거나, callbacks 필드가 이미 대체한다.)
    def on_pre_compress(self, messages: list[Any]) -> str:
        """memory -> compressor 피드백(향후 memory 기반 summary 보강용).

        압축 prompt에 주입할 텍스트를 반환한다(기본값은 없음).
        """
        return ""

    def on_turn_start(self, turn_number: int, message: Any, **kwargs: Any) -> None:
        """turn 시작 알림(향후 백그라운드 리뷰용). 기본 구현은 no-op이다."""
        return None

    # ── Async (예비) ──────────────────────────────────────────────
    # 향후 async LLM 클라이언트가 계약 변경 없이 override할 수 있도록 둔 인터페이스
    # 자리표시자다. 기본 구현은 sync 메서드에 위임한다(실제 LLM 호출이 여전히 sync라
    # 동시성 이득은 없다). 현재 호출자들은 sync 경로를 쓴다.
    async def aadd(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        return self.add(thread_id, messages, agent_name=agent_name, user_id=user_id, trace_id=trace_id)

    async def aget_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        return self.get_context(user_id, agent_name=agent_name, thread_id=thread_id)

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.search(query, top_k, user_id=user_id, agent_name=agent_name, category=category)

    # ── 생성 ─────────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        backend_config: dict[str, Any],
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> MemoryManager:
        """backend config와 host가 제공한 hook으로 완전히 배선된 인스턴스를 만든다.

        factory는 클래스를 직접 생성하는 대신 이 메서드를 호출한다. 덕분에 각 backend가
        자체 조립(config 파싱, 의존성 배선, 필요한 host hook 소비)을 책임진다. backend 추가는
        ``from_config`` 구현만으로 끝나고 factory는 그대로다. ``host_hooks``에는 host가
        제공하는 callable/값(tracing, hidden-message 필터, trace-context manager, host-llm
        factory)이 담긴다. 이 중 아무것도 필요 없는 backend(예: noop)는 무시하고
        ``cls(backend_config=backend_config, mode=mode)``를 반환한다.
        """

    def close(self) -> None:
        """graceful shutdown 중 backend 리소스를 해제한다.

        외부 리소스를 소유한 backend가 이 hook을 override한다. 가벼운 구현이나 서드파티
        구현을 위해 기본은 의도적으로 no-op이다.
        """
        return None


# ── backend 탐색(drop-in) ───────────────────────────────────────────
def _scan_backends() -> dict[str, type[MemoryManager]]:
    """``backends/<name>/`` 아래의 교체 가능한 backend를 탐색한다.

    ``MANAGER_CLASS`` 속성(:class:`MemoryManager` 하위 클래스)을 노출하는 하위 패키지를
    폴더 이름으로 등록한다. 결과는 프로세스 단위로 캐시된다. 폴더 이름 == backend 이름 ==
    ``manager_class`` config 값이라는 drop-in 계약을 지킨다. import에 실패한 backend는
    로그만 남기고 건너뛰므로, 깨진 선택 backend가 factory 전체를 망가뜨리지 않는다.
    """
    global _backends_cache
    if _backends_cache is not None:
        return _backends_cache

    registry: dict[str, type[MemoryManager]] = {}
    if not _BACKENDS_DIR.is_dir():
        _backends_cache = registry
        return registry

    for entry in sorted(_BACKENDS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if not (entry / "__init__.py").is_file():
            continue
        dotted = f"deerflow.agents.memory.backends.{entry.name}"
        try:
            module: ModuleType = importlib.import_module(dotted)
        except Exception:  # noqa: BLE001 - 깨진 backend가 factory를 망가뜨리면 안 된다
            logger.exception("Failed to import memory backend %r; skipping", entry.name)
            continue
        cls = getattr(module, _MANAGER_CLASS_ATTR, None)
        if cls is None:
            continue
        if not (isinstance(cls, type) and issubclass(cls, MemoryManager)):
            logger.warning(
                "Memory backend %r exposes MANAGER_CLASS=%r which is not a MemoryManager subclass; skipping",
                entry.name,
                cls,
            )
            continue
        registry[entry.name] = cls

    _backends_cache = registry
    return registry


def _resolve_manager_class(manager_class: str) -> type[MemoryManager]:
    """``manager_class`` config 값을 구체 클래스로 해석한다.

    해석 순서:

    1. 등록된 짧은 이름(:func:`_scan_backends` 결과).
    2. dotted import 경로(``pkg.mod:Cls`` 또는 ``pkg.mod.Cls``).

    둘 다 아니면 config 오류이므로, 다른 storage backend로 조용히 fallback하지 않고 raise한다.
    memory는 영속 상태다. 명시된 ``manager_class`` 해석이 실패했을 때(오타 / import 오류 /
    속성 없음) 조용히 DeerMem으로 대체하면 쓰기가 엉뚱한 저장소로 가는, 눈에 띄지 않는
    데이터 무결성 사고가 된다. manager는 warm-up을 위해 startup에 미리 해석하므로 여기서 크게
    실패시켜 operator가 나중에 불일치를 발견하는 대신 ``memory.manager_class``를 고치게 한다.
    """
    registry = _scan_backends()
    if manager_class in registry:
        return registry[manager_class]

    # dotted 경로로 취급한다. "pkg.mod:Cls"와 "pkg.mod.Cls"를 모두 지원한다.
    dotted_error: str | None = None
    if ":" in manager_class:
        module_path, _, attr = manager_class.partition(":")
    else:
        module_path, _, attr = manager_class.rpartition(".")
    if module_path and attr:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            dotted_error = f"cannot import module {module_path!r}: {e}"
        else:
            cls = getattr(module, attr, None)
            if cls is None:
                dotted_error = f"attribute {attr!r} not found in {module_path!r}"
            elif not (isinstance(cls, type) and issubclass(cls, MemoryManager)):
                dotted_error = f"{manager_class!r} resolved to non-MemoryManager {cls!r}"
            else:
                return cls

    raise ValueError(
        f"memory.manager_class={manager_class!r} is not a registered backend name "
        f"(known: {sorted(registry)}) nor a resolvable 'pkg.mod:Cls' path" + (f": {dotted_error}" if dotted_error else "") + ". Fix memory.manager_class in config; refusing to silently fall back to a "
        "different storage backend (memory is persistent state -- a wrong store is a "
        "silent data-integrity footgun)."
    )


def backend_requires_passive_writes_in_tool_mode(manager_class: str) -> bool:
    """tool 모드에서 backend가 middleware 쓰기를 필요로 하는지 반환한다.

    클래스를 생성하지 않고 해석만 하므로, 에이전트 조립 과정에서 backend startup 검사나
    네트워크 I/O가 발생하지 않는다.
    """
    return _resolve_manager_class(manager_class).requires_passive_writes_in_tool_mode


# ── host 기본 hook 제공자(factory가 from_config로 넘긴다) ────
#
# backend가 소비할 수 있는 슬롯(tracing, hidden-message 필터링, trace-context 바인딩,
# host 기본 LLM)에 대한 host의 기본 구현이다. 이식 가능한 backend 패키지는 deer-flow 개념을
# 이름으로도 언급하지 않으며, host가 여기서(``backends/deermem/`` 바깥의 host 코드) 공급한다.
# factory가 ``cls.from_config(..., **host_hooks)``로 넘기고, 각 backend의 ``from_config``가
# 필요한 것만 소비한다(DeerMem은 쓰고 noop은 무시한다). ``backend_config``에 프로그램적으로
# 설정된 명시값이 우선하며 그대로 유지된다(from_config의 병합이 이미 있는 키를 건너뛴다).
#
# import는 ``runtime_home`` 선례를 따라 lazy로 둔다. 이 모듈의 import 비용을 낮게 유지하고,
# 계약을 벤더링하는 다른 에이전트가 최상위 import가 아니라 이 헬퍼들만 고치면 되게 하기 위해서다.
class LangfuseMemoryCallbacks(MemoryCallbacks):
    """host 기본 callbacks — memory-LLM 경계에서 langfuse span을 남긴다.

    ``on_memory_llm_call``(LLM 호출 직전)에서 langfuse trace metadata를 ``invoke_config``에
    병합한다. 이전 ``_host_default_tracing_callback``과 시그니처·시점·변형이 모두 동일하며,
    langfuse 바인딩을 host 코드에 두고 이식 가능한 backend 패키지가 langfuse를 언급하지 않도록
    callbacks 메서드로 재포장했을 뿐이다. langfuse가 활성 tracing provider가 아니면 no-op이다.
    """

    def on_memory_llm_call(
        self,
        invoke_config: dict[str, Any],
        *,
        thread_id: str | None,
        user_id: str | None,
        trace_id: str | None,
        model_name: str | None,
    ) -> None:
        from deerflow.tracing import inject_langfuse_metadata

        inject_langfuse_metadata(
            invoke_config,
            thread_id=thread_id,
            user_id=user_id,
            assistant_id="memory_agent",
            model_name=model_name,
            environment=os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT"),
            deerflow_trace_id=trace_id,
        )


def _host_default_should_keep_hidden_message(additional_kwargs: Any) -> bool:
    """DeerMem의 ``should_keep_hidden_message`` 슬롯에 대한 deer-flow 기본 구현.

    ``hide_from_ui`` 메시지는 human-input clarification 응답을 담고 있을 때만 남겨서
    사용자의 해명이 memory에 반영되게 한다. 그 외 hidden 메시지(framework 내부 reminder,
    view-image payload 등)는 버린다. ``message_processing``이 ``read_human_input_response``를
    직접 import하던 추상화 이전 동작을 복원한 것이다.
    """
    from deerflow.agents.human_input import read_human_input_response

    return read_human_input_response(additional_kwargs) is not None


def _host_default_llm() -> Any:
    """DeerMem의 ``host_llm`` 슬롯에 대한 deer-flow 기본 구현(zero-config 추출).

    host의 기본 chat model을 만든다(``create_chat_model(name=None)`` -> 앱 기본값,
    ``attach_tracing=True``이므로 callbacks의 ``on_memory_llm_call`` metadata 병합을 통해
    memory LLM 호출이 langfuse에 드러난다). 추상화 이전의 ``model_name: null``과 동일하다.
    사용 가능한 모델이 없으면(모델 미설정) ``None``을 반환해, DeerMem이 startup을 죽이는 대신
    명확한 에러와 함께 추출을 no-op으로 처리하게 한다.
    """
    try:
        from deerflow.models import create_chat_model

        return create_chat_model(name=None)
    except Exception:  # noqa: BLE001 - 기본 모델 부재는 config 상태이지 크래시 사유가 아니다
        logger.warning("Could not build host default model for DeerMem memory extraction; memory extraction will be disabled", exc_info=True)
        return None


def _host_default_extraction_callback(payload: Any) -> None:
    """DeerMem의 ``extraction_callback`` 슬롯에 대한 deer-flow 기본 구현.

    운영 observability를 위해 추출 후 지표(token 사용량, confidence 필터 통과/거부 fact 수,
    gate 거부율)를 로깅하고, 거부율이 60%를 넘으면 경고한다. 덕분에 prompt/임계값 회귀를
    모든 trace를 뒤지지 않고도 알아챌 수 있다. 전용 extraction span을 남기려면 Langfuse를
    인지하는 callback으로 교체하면 된다. 그 인수인계를 위해 metrics 키는 안정적으로 유지한다.
    예외는 절대 던지지 않는다(DeerMem 쪽이 이미 호출을 감싸고 있다).
    """
    if not isinstance(payload, dict):
        return
    extracted = payload.get("facts_extracted")
    passed_confidence = payload.get("facts_passed_confidence")
    rejected = payload.get("rejected_low_confidence", 0)
    rejected_by_scope = payload.get("rejected_by_scope_gate", 0)
    scope_breakdown = payload.get("scope_gate_rejections")
    thread_id = payload.get("thread_id")
    model_name = payload.get("model_name")
    if isinstance(extracted, int) and isinstance(passed_confidence, int) and extracted > 0:
        rejection_rate = (extracted - passed_confidence) / extracted
        logger.info(
            "Memory extraction metrics: thread=%s model=%s extracted=%d passed_confidence=%d rejected=%d rejection_rate=%.2f",
            thread_id,
            model_name,
            extracted,
            passed_confidence,
            rejected,
            rejection_rate,
        )
        if rejection_rate > 0.6:
            logger.warning(
                "Memory extraction rejection rate %.0f%% exceeds 60%% - review extraction prompt / confidence threshold (thread=%s)",
                rejection_rate * 100,
                thread_id,
            )
    else:
        logger.info(
            "Memory extraction metrics: thread=%s model=%s success=%s token_usage=%s",
            thread_id,
            model_name,
            payload.get("success"),
            payload.get("token_usage"),
        )
    if isinstance(scope_breakdown, dict):
        logger.info(
            "Memory scope-gate metrics: thread=%s model=%s rejected=%s breakdown=%s",
            thread_id,
            model_name,
            rejected_by_scope,
            scope_breakdown,
        )
        fact_breakdown = scope_breakdown.get("facts")
        fact_scope_rejected = sum(value for value in fact_breakdown.values() if isinstance(value, int)) if isinstance(fact_breakdown, dict) else 0
        if isinstance(extracted, int) and extracted > 0 and fact_scope_rejected / extracted > 0.6:
            logger.warning(
                "Memory fact scope-gate rejection rate %.0f%% exceeds 60%% - review extraction model classification / prompt (thread=%s)",
                fact_scope_rejected / extracted * 100,
                thread_id,
            )


def _collect_host_hooks() -> dict[str, Any]:
    """backend가 ``from_config``에서 소비할 host hook callable을 제공한다.

    factory는 hook *제공자*이고, 어떤 hook을 쓸지는 각 backend의 ``from_config``가 정하는
    *소비자* 역할이다(그래서 backend를 추가해도 이 factory는 바뀌지 않는다). ``host_llm``은
    이미 만들어진 인스턴스가 아니라 factory callable(``host_llm_factory``)로 준다. backend가
    자체 모델이 없어 실제로 필요할 때만 host 기본 모델을 만들게 하기 위해서다. 매 startup마다
    쓰지도 않을 기본 모델을 만드는 것은 시간 낭비다. 나머지는 값싼 함수 참조라 바로 넘긴다.
    """
    from deerflow.trace_context import request_trace_context

    return {
        "callbacks": LangfuseMemoryCallbacks(),
        "should_keep_hidden_message": _host_default_should_keep_hidden_message,
        "trace_context_manager": request_trace_context,
        "host_llm_factory": _host_default_llm,
        "extraction_callback": _host_default_extraction_callback,
    }


# ── 싱글턴 factory ─────────────────────────────────────────────────────
def get_memory_manager() -> MemoryManager:
    """현재 config에 대한 싱글턴 :class:`MemoryManager`를 반환한다.

    ``MemoryConfig.manager_class``를 읽어 :func:`_resolve_manager_class`로 해석한다.
    인스턴스는 캐시되며, 재해석이 필요하면(테스트 / 런타임 backend 전환)
    :func:`reset_memory_manager`를 호출한다.
    """
    global _memory_manager
    if _memory_manager is not None:
        return _memory_manager

    # deer-flow는 멀티스레드다. memory 주입은 asyncio.to_thread로 돌고, 갱신 큐는 Timer
    # thread에서 발화하며, gateway/agent thread도 모두 여기로 온다. double-checked locking으로
    # 첫 호출이 경쟁해도 인스턴스가 하나만 생성되게 한다. 이제 backend가 __init__에서 만드는
    # 상태 있는 의존성(DeerMem은 storage/queue/updater를 소유하고, 다른 backend는 커넥션을
    # 열 수도 있다)을 갖기 때문에 필수다.
    with _manager_lock:
        if _memory_manager is not None:
            return _memory_manager

        cfg = get_memory_config()
        manager_class = cfg.manager_class
        cls = _resolve_manager_class(manager_class)
        backend_config = dict(cfg.backend_config or {})
        # zero-config UX: host가 storage_path를 명시하지 않으면 DeerMem storage를 deer-flow의
        # state 디렉터리(절대 경로, CWD 무관)로 기본 설정한다. 그러면 memory가 추상화 이전과
        # 동일하게 {runtime_home}/users/{user_id}/memory.json(deer-flow의 base_dir)에 놓인다.
        if not backend_config.get("storage_path"):
            from deerflow.config.runtime_paths import runtime_home

            backend_config["storage_path"] = str(runtime_home())
        elif not Path(backend_config.get("storage_path", "")).is_absolute():
            # 상대 storage_path는 추상화 이전 의미를 유지하도록 runtime_home() 기준으로
            # 해석한다(base_dir 상대, CWD 무관). 그대로 두면 CWD 상대라 취약하다.
            # 이식 가능한 paths.py가 runtime_home에 의존하지 않도록 host 코드인 여기서 해석한다.
            from deerflow.config.runtime_paths import runtime_home

            backend_config["storage_path"] = str((Path(runtime_home()) / backend_config["storage_path"]).resolve())
        # storage_path가 파일인지 검사하는 guard는 이제 DeerMemConfig.model_validator에 있다
        # (DeerMem 내부 의미이며 factory를 우회해도 동작한다).
        # host hook 제공: factory가 kwargs로 넘기고, 어떤 것을 소비할지는 각 backend의
        # from_config가 정한다(factory는 backend 중립을 유지하며 어떤 hook이 필요한지 모른다).
        # backend_config에 이미 있는 명시값이 여전히 우선한다(from_config의 병합이 기존 키를
        # 건너뛴다).
        host_hooks = _collect_host_hooks()
        # ``mode``는 host의 MemoryConfig.mode를 그대로 반영한다. 그래야 invariant validator
        # (mode=="tool"이면 search 필요)가 factory 경로에서도 동작한다.
        _memory_manager = cls.from_config(backend_config, mode=cfg.mode, **host_hooks)
        logger.info("Memory manager resolved: %s (manager_class=%r)", cls.__name__, manager_class)
        return _memory_manager


def reset_memory_manager() -> None:
    """캐시된 싱글턴 manager와 backend 레지스트리를 비운다.

    다음 :func:`get_memory_manager` 호출이 config를 다시 읽고 backend를 다시 스캔한다.
    테스트나 런타임 backend 전환 시 사용한다.
    """
    global _memory_manager, _backends_cache
    with _manager_lock:
        _memory_manager = None
        _backends_cache = None
