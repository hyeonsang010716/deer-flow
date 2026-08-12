"""공유 sandbox 컨테이너용 ownership store 계약 (#4206).

Gateway instance들은 sandbox 컨테이너를 공유하지만 warm pool은 각자 in-memory로
따로 들고 있다. 공유 ownership 상태가 없으면 한 instance의 startup reconciliation이
다른 instance가 쓰고 있는 컨테이너를 흡수한 뒤 idle이라며 파괴하고, 도구 호출이
502 / connection refused로 실패한다.

lease는 "**이 컨테이너를 누가 회수할 책임인가**"에 답하는 것이지 "누가 써도 되는가"에
답하는 게 아니다. 이 구분이 인터페이스 전체를 결정한다.

* 컨테이너는 (user, thread)마다 결정적이므로, 한 thread의 연속된 turn이 서로 다른
  instance로 가는 것은 정상이다. 지금 그 thread를 처리하는 instance가 기존 보유자에게서
  :meth:`take`로 lease를 가져온다. peer가 아직 쥐고 있다고 거절하면 그 lease가 만료될
  때까지 thread가 멈춘다.
* 회수는 반대다. :meth:`claim`은 컨테이너가 주인이 없거나 이미 우리 것일 때만 성공하므로,
  살아 있는 peer가 책임지는 컨테이너를 흡수했다가 나중에 idle로 파괴하는 일이 생기지
  않는다. 그게 #4206이다.

**lease에는 두 가지 상태가 있고, 이것이 destroy 구간을 안전하게 만든다.**
`own:`은 "이 컨테이너는 내 책임", `del:`은 "지금 이 컨테이너를 내리는 중"이다.
`del:` lease에 대한 인수(:meth:`take`)는 거절되므로, destroy 경로의 claim과 실제
컨테이너 stop 사이에 컨테이너가 재획득되지 않는다. 삭제된 per-sandbox flock guard가
막던 구간이다. 두 상태가 없으면 무조건적인 `take`가 destroyer의 claim을 조용히 덮고,
peer의 stop이 새 소유자가 이미 에이전트에게 넘긴 컨테이너에 떨어진다.

구현자를 위한 계약 노트:

* 모든 메서드는 **동기**다. ``StreamBridge``는 event loop에서 구동되기에 async API를
  갖지만, ownership은 ``AioSandboxProvider.__init__``, 백그라운드 idle/renewal 스레드,
  동기 ``release()`` 경로에서 구동된다. 실제로 event loop에서 도는 sandbox 도구 경로
  (``get()``)는 의도적으로 store를 건드리지 않고, async acquire 경로는 등록 작업을
  ``asyncio.to_thread``로 offload한다.
* backend 실패 시 falsy 값을 반환하지 않고 ``OwnershipBackendError``를 **던진다**.
  호출자는 fail closed 해야 한다. ownership을 게시하지 못한 sandbox는 넘겨주면 안 되고,
  주인이 없음을 증명하지 못한 컨테이너는 파괴하면 안 된다. ``False`` 반환은 "확실히 우리
  것이 아님", 예외는 "알 수 없음"을 뜻한다.
"""

from __future__ import annotations

import abc
import enum


class OwnershipBackendError(RuntimeError):
    """ownership backend가 답하지 못했다.

    확정적인 "우리 것이 아님"(``False``)과 다르다. ownership이 *알 수 없는* 상태이므로
    호출자는 컨테이너가 비었다고 가정하지 말고 fail closed 해야 한다.
    """


class RenewOutcome(enum.Enum):
    """renewal이 성공했는지, 실패했다면 왜인지를 나타낸다.

    ``LAPSED``와 ``LOST``를 하나의 falsy 값으로 뭉뚱그리면 안 된다. lapsed lease는
    *없는* 것이다. 아무도 가져가지 않았으니 재확립이 안전하고, Redis 재시작이 fleet 전체의
    살아 있는 sandbox를 날려버리는 것을 막는 지점이 바로 여기다. lost lease는 peer의
    것이며, 이를 다시 가져오는 것이 #4206의 cross-instance kill이다.
    """

    #: 여전히 우리 것이고 TTL을 갱신했다.
    RENEWED = "renewed"
    #: lease가 없다(만료됐거나 store가 상태를 잃었다). 다시 claim해도 된다.
    LAPSED = "lapsed"
    #: peer가 쥐고 있거나 teardown 중이다. 다시 가져오면 안 된다.
    LOST = "lost"


class SandboxOwnershipStore(abc.ABC):
    """sandbox 컨테이너용 cross-instance ownership lease."""

    #: 이 store가 현재 프로세스 밖의 instance까지 조율하는지 여부.
    #: ``False``면 peer가 우리 lease를 볼 수 없어 모든 컨테이너가 orphan으로 보인다.
    #: 단일 instance 배포에서만 쓴다.
    supports_cross_process: bool = False

    @property
    @abc.abstractmethod
    def owner_id(self) -> str:
        """lease에 기록되는 이 instance의 owner id."""

    @abc.abstractmethod
    def take(self, sandbox_id: str) -> bool:
        """acquire 경로에서 *sandbox_id*의 책임을 가져온다.

        살아 있는 peer로부터 인수한다. 이 컨테이너의 thread turn이 여기로 라우팅됐고,
        이전 소유자는 다음 renewal이 ``LOST``를 보고할 때 추적을 멈춘다. 이전 소유자는
        컨테이너를 파괴하면 안 된다. ``AioSandboxProvider._forget_lost_sandbox`` 참고.

        teardown 중인 컨테이너만 거절한다. 이것이 destroy → 재획득 구간을 막는다.

        Returns:
            이후 이 instance가 lease를 소유하면 ``True``.
            컨테이너가 파괴 중이라 사용하면 안 되면 ``False``.

        Raises:
            OwnershipBackendError: ownership을 게시하지 못했다. 호출자는 fail closed 해야
                한다. 게시되지 않은 sandbox는 peer에게 orphan으로 보이므로 넘겨주면 안 된다.
        """

    @abc.abstractmethod
    def claim(self, sandbox_id: str, *, for_destroy: bool = False) -> bool:
        """*sandbox_id*가 주인이 없거나 이미 우리 것일 때만 소유권을 가져온다.

        배타적이다. 컨테이너가 주인이 없거나 이미 우리 것일 때만 성공하며, 이것이 모든
        adopt/reap 경로의 관문이다.

        단, **peer**에 대해 배타적일 뿐 호출자 자신의 프로세스에 대해서는 아니다. 우리
        ``own:`` lease에 대한 claim은 설계상 성공하고, 그래야 destroy 경로가 이미 소유한
        것을 claim할 수 있다. instance의 reaper 스레드와 자신의 acquire 경로 사이의
        same-process 배제는 이 store가 아니라 provider의 몫이다(``_reserve_local_teardown``).

        ``for_destroy``가 조용히 되감기지 않도록 예외가 하나 있다. 우리 ``del:`` lease에
        대한 **비**-destroy claim은 거절된다. 그 마커가 가리키는 stop은 이미 진행 중이고
        취소할 수 없으므로, 마커를 낮추면 :meth:`take`가 곧 죽을 컨테이너를 넘겨주게 된다.

        read-modify-write는 끼어들면 안 된다. redis에서는 Lua(스크립트 하나, 서버 측)로
        보장하고, memory store는 프로세스 로컬 lock으로 직렬화하며 어차피 단일 instance라
        "서로 다른 instance"가 생길 수 없다. *검증되지 않는* 부분도 짚어 둔다. 계약
        테스트는 순차 호출만 돌리므로 배제 조건은 고정하지만 atomicity는 고정하지 않고,
        CI는 memory tier만 돌리므로 이를 담당하는 Lua는 merge gate에서 실행되지 않는다.

        Args:
            for_destroy: lease를 teardown 진행 중으로 표시해, 유지되는 동안 동시
                :meth:`take`를 거절한다. destroy 경로는 반드시 설정해야 한다. 마커는
                컨테이너가 멈추면 :meth:`release`가 지우고, destroyer가 stop 도중 죽으면
                TTL과 함께 만료된다.

        Returns:
            이후 이 instance가 lease를 소유하면 ``True``.
            살아 있는 peer가 쥐고 있으면 ``False``.

        Raises:
            OwnershipBackendError: ownership을 판정하지 못했다.
        """

    @abc.abstractmethod
    def renew(self, sandbox_id: str) -> RenewOutcome:
        """*sandbox_id*에 대한 우리 lease를 갱신한다.

        의도적으로 스스로 재획득하지 않는다. 안전한 재확립(``LAPSED``)과 cross-instance
        탈취(``LOST``)를 구분할 수 있는 건 호출자뿐이므로 판단은 호출자에게 맡긴다.

        Raises:
            OwnershipBackendError: ownership을 판정하지 못했다.
        """

    @abc.abstractmethod
    def release(self, sandbox_id: str) -> None:
        """상태와 무관하게 *sandbox_id*에 대한 우리 lease를 놓는다.

        우리 lease가 아니면 no-op이므로 peer의 살아 있는 lease를 지우는 일은 없다.
        best-effort다. lease가 만료돼도 같은 상태에 도달한다.

        Raises:
            OwnershipBackendError: release를 게시하지 못했다.
        """

    @abc.abstractmethod
    def owner(self, sandbox_id: str) -> str | None:
        """*sandbox_id*의 현재 owner id를 반환하고, 주인이 없으면 ``None``을 반환한다.

        읽기 전용이다. :meth:`claim`과 달리 소유권을 가져오지 않는다. destroy를 막는
        관문이 아니라 조사(테스트, 로깅)에 쓴다. 읽기 결과는 반환되는 순간 이미 낡지만,
        성공한 claim은 peer를 실제로 막아 준다.

        Raises:
            OwnershipBackendError: ownership을 읽지 못했다.
        """

    def close(self) -> None:
        """backend 자원을 해제한다. 기본 구현은 no-op이다."""
