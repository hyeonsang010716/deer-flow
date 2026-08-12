"""Noop memory 백엔드 — 동작하는 빈 :class:`MemoryManager`.

플러그인 메커니즘(factory + drop-in 발견 + config 전환)이 끝까지 동작함을 증명하며,
동시에 새 백엔드의 **템플릿** 역할을 한다.

이식성 원칙(전문은 ``config.py`` 참고): 백엔드는 host 정보를 (1) ABC 메서드 인자와
(2) ``backend_config`` dict로만 받는다. 이 폴더에서 허용되는 유일한
``from deerflow`` import는 아래 ABC 계약 줄뿐이며, 다른 에이전트로 이식할 때는 그 한
줄만 바꾼다. deer-flow의 경로 헬퍼, config 싱글턴, model을 import하지 말고 전부
``backend_config``에서 받는다.

새 백엔드를 만드는 절차:
  1. 이 폴더를 ``backends/<yourname>/``로 복사한다.
  2. ``config.py``: 자기 설정과 ``from_backend_config``를 선언한다(``backend_config``를
     파싱하고 ``storage_path``도 deer-flow가 아니라 거기서 읽는다).
  3. ``<yourname>_manager.py``: 클래스 이름을 바꾸고, 의존성을 ``PrivateAttr``로
     선언하고, ``model_post_init``에서 ``self.backend_config``를 자기 config로 파싱한
     뒤, 자기 memory 시스템에 맞춰 ABC 메서드를 구현한다.
  4. (선택) 맨 아래의 tier-3 훅(``create_fact`` / ``delete_fact`` / ``update_fact`` /
     ``reload_memory`` / ``warm``)을 override한다. 기본 구현이 있으므로
     (``warm``=True, 나머지는 ``NotImplementedError``) 지원하는 것만 override하면 되고,
     나머지는 호출자가 ``NotImplementedError``를 잡는다.
  5. ``__init__.py``: ``MANAGER_CLASS = YourManager``로 지정한다(상대 import).
  6. ``config.yaml``: ``manager_class: <yourname>``.

반환 형태 주의: host gateway는 ``get_memory`` / ``export_memory`` / ``clear_memory`` /
``import_memory``의 반환값을 DeerMem 형태 응답
(``version`` / ``lastUpdated`` / ``user`` / ``history`` / ``facts[]``)으로 캐스팅한다.
실제 백엔드는 그 형태로 캐스팅 가능한 dict를 반환한다(DeerMem이 아닌 백엔드는 자기
레코드를 이 형태로 매핑한다). noop은 최소 형태인 ``{"facts": []}``만 반환하고 나머지는
gateway가 기본값으로 채운다.

``manager_class: noop``이면 시스템은 빈 memory로 동작한다. 아무것도 저장하지 않고,
아무것도 주입하지 않으며, 모든 읽기가 빈 결과를 반환한다. 테스트, ``enabled``를
건드리지 않고 memory를 끄는 용도, 기준선 비교에 쓴다.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

# ABC 계약 — 이 백엔드 폴더에서 유일하게 허용되는 `from deerflow`.
# 이식할 때는 이 한 줄만 다른 에이전트의 MemoryManager로 바꾼다.
from deerflow.agents.memory.manager import MemoryManager

from .config import NoopConfig


def _empty_memory() -> dict[str, Any]:
    """새로운 빈 memory 문서를 반환한다(호출자가 변경해도 된다).

    최소 형태이며 ``version`` / ``lastUpdated`` / ``user`` / ``history``는 host
    gateway가 기본값으로 채운다. 실제 백엔드는 완전한 DeerMem 형태 문서를 반환한다
    (모듈 docstring의 반환 형태 주의 참고).
    """
    return {"facts": []}


class NoopMemoryManager(MemoryManager):
    """아무것도 저장하지 않고 아무것도 recall하지 않는 백엔드.

    ``model_post_init``이 ``backend_config``를 :class:`NoopConfig`로 파싱하지만 이는
    패턴을 보여주기 위한 것이고 noop은 모든 필드를 무시한다. 실제 백엔드는 storage
    루트, model 등 자기 설정을 ``self._config``에서 읽는다.
    """

    # 파싱된 config(PrivateAttr이므로 검증·직렬화 대상 필드가 아니다). noop은 모든
    # 필드를 무시하지만 실제 백엔드는 storage 루트, model 등에 self._config.*를 쓴다.
    # storage_path도 여기서(host가 주입) 오며, deer-flow 경로 헬퍼를 import하지 않는다.
    _config: Any = PrivateAttr(default=None)

    # noop은 search()를 override해 []를 반환한다("아무것도 저장·recall하지 않는다"는
    # 설계상 모든 읽기가 빈 결과이며 예외를 던지지 않는다). 따라서 search 가능하며,
    # 플래그 == override라는 불변식에 맞춰 True로 둔다.
    supports_search: ClassVar[bool] = True

    def model_post_init(self, __context: Any) -> None:
        self._config = NoopConfig.from_backend_config(self.backend_config)

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> NoopMemoryManager:
        """noop은 연결할 의존성이 없으므로 host_hooks를 무시한다."""
        return cls(backend_config=backend_config, mode=mode)

    # ── 쓰기 ─────────────────────────────────────────────────────────────
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        return None

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> None:
        return None

    # ── 읽기 ─────────────────────────────────────────────────────────────
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        return ""

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        return []

    # ── 관리 ─────────────────────────────────────────────────────────────
    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        return _empty_memory()

    # delete_memory / export_memory는 base의 tier-2 기본 구현(NotImplementedError)을
    # 그대로 상속한다. 호출자가 하나도 없는 죽은 계약이라 noop은 override하지 않는다.

    def clear_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        return _empty_memory()

    def import_memory(
        self,
        memory_data: dict[str, Any],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        return _empty_memory()

    # ── 생명주기 ────────────────────────────────────────────────────────
    def shutdown_flush(self, timeout: float) -> bool:
        """큐에 쌓이는 것이 없으므로 종료 시 drain은 그냥 성공하는 no-op이다."""
        return True

    # ── Tier 3 훅 (기본 구현 상속. 백엔드가 지원하면 override한다) ──
    # warm / reload_memory / fact CRUD는 base MemoryManager의 선택적 tier-3 훅이며
    # 기본값이 있다: warm=True(warm할 것이 없음), 나머지는 NotImplementedError.
    # noop은 fact CRUD와 reload를 지원하지 않으므로 기본 구현을 그대로 상속한다
    # (호출자가 NotImplementedError를 잡아 501 또는 fallback으로 처리한다).
    # 지원하는 것만 override하되 시그니처는 base와 일치해야 한다
    # (전체 구현은 DeerMem 참고):
    #
    # def create_fact(self, content, category="context", confidence=0.5, *,
    #                 agent_name=None, user_id=None) -> tuple[dict, str | None]:
    #     ...  # (memory_data, fact_id)를 반환. 상한에 밀려 제거되면 fact_id=None
    # def delete_fact(self, fact_id, *, agent_name=None, user_id=None) -> dict: ...
    # def update_fact(self, fact_id, content=None, category=None, confidence=None,
    #                 *, agent_name=None, user_id=None) -> dict: ...
    # def reload_memory(self, *, user_id=None, agent_name=None) -> dict: ...
    # def warm(self) -> bool | None: ...   # 기본 None(warm할 것 없음). 무거운 일회성 초기화가 있으면 override
