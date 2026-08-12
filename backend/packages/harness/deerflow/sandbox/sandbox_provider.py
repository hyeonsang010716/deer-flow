import asyncio
import threading
from abc import ABC, abstractmethod

from deerflow.config import get_app_config
from deerflow.reflection import resolve_class
from deerflow.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
    """sandbox provider의 추상 base class"""

    uses_thread_data_mounts: bool = False
    needs_upload_permission_adjustment: bool = True

    @abstractmethod
    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """sandbox 환경을 확보하고 그 ID를 반환한다.

        Returns:
            확보한 sandbox 환경의 ID.
        """
        pass

    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """event loop를 막지 않고 sandbox를 확보한다.

        로컬 Docker/provisioner 작업이 blocking이라 대부분의 sandbox provider는 동기
        라이프사이클 API를 노출한다. async runtime은 이 메서드를 호출해서 그 blocking
        작업이 event loop를 멈추지 않고 worker thread에서 돌게 해야 한다.
        """
        return await asyncio.to_thread(self.acquire, thread_id, user_id=user_id)

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """ID로 sandbox 환경을 가져온다.

        Args:
            sandbox_id: 유지할 sandbox 환경의 ID.
        """
        pass

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """sandbox 환경을 반환한다.

        Args:
            sandbox_id: 파괴할 sandbox 환경의 ID.
        """
        pass

    def reset(self) -> None:
        """provider 인스턴스가 교체돼도 남아 있는 캐시 state를 비운다.

        provider가 override하면 리소스를 반환하면서 인스턴스를 사용 불가로 만들 수 있다.
        """
        pass


_default_sandbox_provider: SandboxProvider | None = None
# `_default_sandbox_provider`의 모든 읽기/쓰기를 보호한다. 이 singleton은 여러 OS
# thread에서 접근 가능하므로(예: 메인 event loop와 자체 loop를 도는 Feishu channel
# thread), 그냥 확인 후 생성하면 provider가 두 번 초기화될 수 있고, 동기화되지 않은
# reset/shutdown이 get과 경쟁하면 호출자에게 `None`이나 깨진 인스턴스를 넘길 수 있다.
# 아래 전역 변수 접근은 `get_sandbox_provider()`의 읽기+반환을 포함해 전부 이 lock을
# 잡는다.
#
# lock은 참조 교체만 보호한다. provider 콜백(`__init__`, `reset()`, `shutdown()`)과
# `resolve_class()`의 동적 import는 lock *바깥*에서 실행한다: 플러그인이 제공하는 코드
# (`config.sandbox.use`는 임의 클래스로 해석된다)라서 느릴 수 있고, 더 나쁘게는 이
# 라이프사이클 함수들에 재진입할 수 있다. 재진입 불가능한 `threading.Lock`을 잡은 채
# 호출하면 그런 provider에서 self-deadlock이 나고, 느린 teardown 동안 동시 `get()`이
# 전부 막힌다. 콜백을 lock 밖에 두면 둘 다 피한다.
_provider_lock = threading.Lock()


def get_sandbox_provider(**kwargs) -> SandboxProvider:
    """sandbox provider singleton을 가져온다.

    캐시된 singleton 인스턴스를 반환한다. 캐시만 비우려면 `reset_sandbox_provider()`,
    제대로 종료하고 비우려면 `shutdown_sandbox_provider()`를 쓴다.

    Returns:
        sandbox provider 인스턴스.
    """
    global _default_sandbox_provider
    # 빠른 경로: lock을 잡고 한 번만 읽어서, 확인과 반환 사이에 동시 reset/shutdown이
    # 전역 변수를 None으로 만들지 못하게 한다.
    with _provider_lock:
        if _default_sandbox_provider is not None:
            return _default_sandbox_provider

    # 콜드 스타트. 해석과 생성은 lock 바깥에서 한다: import와 provider 생성자는 플러그인
    # 코드라 재진입 불가능한 lock 안에서 실행하면 안 된다. 다른 호출자와 경쟁할 수 있으므로
    # lock 안에서 정리한다.
    config = get_app_config()
    cls = resolve_class(config.sandbox.use, SandboxProvider)
    provider = cls(**kwargs)

    with _provider_lock:
        if _default_sandbox_provider is None:
            _default_sandbox_provider = provider
            return provider
        # 설치 경쟁에서 졌다: 다른 thread가 먼저 도달했다. `winner`는 같은 lock 안에서
        # 읽으므로 항상 살아 있는 인스턴스이고 절대 None이 아니다.
        winner = _default_sandbox_provider

    # 방금 만든 인스턴스를 (lock 바깥에서) 버린다. 생성자에 side effect가 있는
    # provider(예: AioSandboxProvider는 idle-checker thread를 띄운다)는 이렇게 정리해야
    # 고아 인스턴스가 새지 않는다 — issue #3721.
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    return winner


def reset_sandbox_provider() -> None:
    """sandbox provider singleton을 리셋한다.

    shutdown을 직접 호출하지 않고 캐시된 인스턴스만 비운다. 다음 `get_sandbox_provider()`
    호출이 새 인스턴스를 만든다. 테스트나 설정 전환 시에 유용하다.

    provider는 `reset()`을 override해서 인스턴스를 넘어 살아 있는 모듈 수준 state를
    비울 수 있다(예: `LocalSandboxProvider`의 캐시된 `LocalSandbox` singleton). 그렇게
    하지 않으면 config/mount 변경이 다음 acquire()에 반영되지 않는다.

    provider override는 reset 중에 활성 sandbox를 반환할 수 있다. 그렇지 않으면 활성
    sandbox는 고아가 된다. reset 이후에는 떼어낸 provider를 재사용하지 않는다. 제대로
    정리하려면 `shutdown_sandbox_provider()`를 쓴다.
    """
    global _default_sandbox_provider
    # lock 안에서 참조만 떼어내고, provider의 `reset()` 콜백은 lock 바깥에서 실행한다
    # (`_provider_lock` 주석 참고).
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is not None:
        provider.reset()


def shutdown_sandbox_provider() -> None:
    """sandbox provider를 종료하고 리셋한다.

    singleton을 비우기 전에 provider를 제대로 종료해서(모든 sandbox 반환) 정리한다.
    애플리케이션 종료 시나 sandbox 시스템을 완전히 초기화해야 할 때 호출한다.
    """
    global _default_sandbox_provider
    # lock 안에서 참조만 떼어내고, (느릴 수 있는) `shutdown()` 콜백은 lock 바깥에서
    # 실행한다(`_provider_lock` 주석 참고).
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is not None and hasattr(provider, "shutdown"):
        provider.shutdown()


def set_sandbox_provider(provider: SandboxProvider) -> None:
    """커스텀 sandbox provider 인스턴스를 설정한다.

    테스트 목적으로 커스텀 또는 mock provider를 주입할 수 있게 한다.

    주의: 기존에 설치된 provider는 교체될 뿐 종료되지 않는다. 덮어쓰는 인스턴스의
    라이프사이클은 호출자가 책임진다.

    Args:
        provider: 사용할 SandboxProvider 인스턴스.
    """
    global _default_sandbox_provider
    with _provider_lock:
        _default_sandbox_provider = provider
