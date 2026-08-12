"""custom agent 정의 저장소의 추상 인터페이스.

두 가지 구현이 있다:
- :class:`FileAgentStore` — 기존의 사용자별 on-disk 레이아웃(``config.yaml`` +
  ``SOUL.md``). 여전히 기본값이며 동작도 그대로다.
- :class:`SqlAgentStore` — 공유 SQL persistence 레이어에 agent당 한 row를 두므로,
  다중 인스턴스 배포에서 모든 노드가 같은 agent를 본다.

이 store는 의도적으로 **동기**다. 소비자인 LangGraph graph factory
(``make_lead_agent``), ``setup_agent`` / ``update_agent`` 도구, GitHub agent
registry가 모두 동기이고, event loop 위나 gateway와 별도 프로세스에서 실행될 수 있는데
그런 곳에서는 async engine을 구동할 수 없기 때문이다. async HTTP route는
``asyncio.to_thread``로 store를 호출한다(agents router가 파일시스템 작업에 이미 쓰는 방식).

``user_id`` 의미(리팩터링 전 :mod:`deerflow.config.agents_config`의 자유 함수들과
동일하게 유지했고, 그래서 file backend 이관이 동작 중립적이다): ``None``은
:func:`deerflow.runtime.user_context.get_effective_user_id`를 통해 request context의
유효 사용자로 해석된다(no-auth 모드에서는 ``"default"``). 이것은 파일시스템 버킷 의미이며,
async ``thread_meta`` repository가 쓰는 AUTO/None sentinel과는 다르다.
"""

from __future__ import annotations

import abc
from collections.abc import Hashable
from typing import Any, Literal

from deerflow.config.agents_config import AgentConfig


def parse_agent_config(data: dict[str, Any], name: str) -> AgentConfig:
    """raw config *document*로부터 :class:`AgentConfig`를 만든다. 두 backend가 공유한다.

    document에 ``name``이 없으면 자연키에서 채우고, 검증 전에 알 수 없는 키(예: 예전
    ``prompt_file``)를 제거한다. 리팩터링 전 ``load_agent_config``와 동일하다.
    """
    data = dict(data)
    if "name" not in data:
        data["name"] = name
    known_fields = set(AgentConfig.model_fields.keys())
    data = {k: v for k, v in data.items() if k in known_fields}
    return AgentConfig(**data)


# 삭제 결과. agents router의 결과와 대응한다:
# row/디렉터리를 제거했거나("deleted"), 현재 write 경로가 절대 지우지 않는 legacy 공유
# 레이아웃 항목만 있거나("legacy"), 아무것도 없었거나("missing"), 사용자별 디렉터리가
# memory/facts 데이터를 담고 있지만 custom agent는 아니어서(config.yaml 없음) 사용자의
# memory를 지우는 대신 보존한 경우다("not-custom-agent", #4279).
AgentDeleteOutcome = Literal["deleted", "legacy", "missing", "not-custom-agent"]


class AgentExistsError(Exception):
    """``(user_id, name)``이 이미 존재할 때 :meth:`AgentStore.create`가 raise한다."""


class AgentStore(abc.ABC):
    @abc.abstractmethod
    def get(self, name: str, *, user_id: str | None = None) -> AgentConfig:
        """agent의 config를 반환한다.

        agent가 없으면 :class:`FileNotFoundError`를 raise한다. ``routers/agents.py``와
        ``update_agent``가 404 / "does not exist" 오류를 드러내기 위해 의존하는 기존 계약이다.
        """

    @abc.abstractmethod
    def exists(self, name: str, *, user_id: str | None = None) -> bool:
        """``user_id``에 대해 ``name``이 이미 사용 중인지 반환한다.

        :meth:`create`의 충돌 규칙과 일관되므로 "사용 가능"하던 이름이 나중에 409가 되는 일은
        없다. file backend는 사용자별이든 legacy든 디렉터리가 있으면 사용 중으로 보고,
        db backend는 row 존재 여부를 확인한다.
        """

    @abc.abstractmethod
    def get_soul(self, name: str, *, user_id: str | None = None) -> str | None:
        """agent의 ``SOUL.md`` 내용을 반환한다. 설정되지 않았거나 비었으면 ``None``."""

    @abc.abstractmethod
    def list(self, *, user_id: str | None = None) -> list[AgentConfig]:
        """``user_id``가 소유한 모든 custom agent를 이름순으로 반환한다."""

    @abc.abstractmethod
    def list_all(self) -> list[tuple[str, AgentConfig]]:
        """모든 소유자에 걸친 전체 agent를 ``(user_id, config)``로 반환한다.

        repo 바인딩을 찾기 위해 모든 사용자의 agent를 훑는 GitHub registry가 사용한다.
        순서는 결정적이다(``user_id`` 다음 name).
        """

    @abc.abstractmethod
    def create(self, name: str, config: dict, soul: str, *, user_id: str | None = None) -> None:
        """각 write 지점이 만든 config *document*로부터 새 agent를 저장한다.

        ``config``는 caller가 조립한 raw dict다(예전에 ``config.yaml``에 쓰던 바로 그것).
        재직렬화한 :class:`AgentConfig` 대신 document를 넘기면 on-disk 바이트와 "존재하는
        키만 쓴다"는 동작이 리팩터링 전 writer와 동일하게 유지된다. ``(user_id, name)``이
        이미 있으면 :class:`AgentExistsError`를 raise한다.
        """

    @abc.abstractmethod
    def update(self, name: str, config: dict | None, soul: str | None, *, user_id: str | None = None) -> None:
        """agent의 config와/또는 soul을 쓴다(upsert).

        ``config``와 ``soul``은 각각 독립적으로 선택적이며, ``None``은 "그 부분은 그대로
        둔다"는 뜻이다. agents router는 config 필드가 바뀐 경우에만 config를, soul이 주어진
        경우에만 soul을 갱신한다. 레코드가 없으면 생성한다(``setup_agent`` / 최초 write 경로).
        """

    @abc.abstractmethod
    def delete(self, name: str, *, user_id: str | None = None) -> AgentDeleteOutcome:
        """agent와 같은 위치에 있는 memory를 삭제하고 결과를 반환한다."""

    @abc.abstractmethod
    def signature(self) -> Hashable:
        """cache 무효화를 위한 불투명한 변경 토큰을 반환한다.

        토큰이 같으면 "마지막 읽기 이후 변한 것이 없다"는 뜻이다. GitHub registry는 두 backend
        모두에서 동작하도록 ``stat()`` 대신 이 값으로 cache를 키잉한다(``file``은 mtime
        3-튜플, ``db``는 저장된 agent 내용의 결정적 digest).
        """
