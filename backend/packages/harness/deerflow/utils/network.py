"""thread-safe 네트워크 유틸리티."""

import socket
import threading
from contextlib import contextmanager


class PortAllocator:
    """동시 실행 환경에서 포트 충돌을 막는 thread-safe 포트 allocator.

    예약된 포트 집합을 유지하고 lock으로 포트 할당을 원자적으로 만든다. 한 번 할당된 포트는
    명시적으로 release할 때까지 예약 상태로 남는다.

    사용법:
        allocator = PortAllocator()

        # 방법 1: 수동 할당과 해제
        port = allocator.allocate(start_port=8080)
        try:
            # 포트 사용...
        finally:
            allocator.release(port)

        # 방법 2: context manager (권장)
        with allocator.allocate_context(start_port=8080) as port:
            # 포트 사용...
            # context를 벗어나면 포트가 자동으로 해제된다
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._reserved_ports: set[int] = set()

    def _is_port_available(self, port: int) -> bool:
        """포트를 bind할 수 있는지 확인한다.

        Args:
            port: 확인할 포트 번호.

        Returns:
            사용 가능하면 True, 아니면 False.
        """
        if port in self._reserved_ports:
            return False

        # 검사가 Docker의 동작과 정확히 일치하도록 localhost가 아니라 0.0.0.0(wildcard)에
        # bind한다. Docker는 0.0.0.0:PORT에 bind하므로, 127.0.0.1만 확인하면 Docker가 이미
        # wildcard 주소에서 점유 중인 포트를 사용 가능하다고 잘못 보고할 수 있다.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    def allocate(self, start_port: int = 8080, max_range: int = 100) -> int:
        """thread-safe하게 사용 가능한 포트를 할당한다.

        사용 가능한 포트를 찾아 예약 표시한 뒤 반환한다. release()를 호출할 때까지 예약 상태가
        유지된다.

        Args:
            start_port: 탐색을 시작할 포트 번호.
            max_range: 탐색할 최대 포트 개수.

        Returns:
            사용 가능한 포트 번호.

        Raises:
            RuntimeError: 지정한 범위에서 사용 가능한 포트를 찾지 못한 경우.
        """
        with self._lock:
            for port in range(start_port, start_port + max_range):
                if self._is_port_available(port):
                    self._reserved_ports.add(port)
                    return port

            raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_range}")

    def release(self, port: int) -> None:
        """앞서 할당한 포트를 해제한다.

        Args:
            port: 해제할 포트 번호.
        """
        with self._lock:
            self._reserved_ports.discard(port)

    @contextmanager
    def allocate_context(self, start_port: int = 8080, max_range: int = 100):
        """포트를 할당하고 자동으로 해제하는 context manager.

        Args:
            start_port: 탐색을 시작할 포트 번호.
            max_range: 탐색할 최대 포트 개수.

        Yields:
            사용 가능한 포트 번호.
        """
        port = self.allocate(start_port, max_range)
        try:
            yield port
        finally:
            self.release(port)


# 애플리케이션 전역에서 공유하는 포트 allocator 인스턴스
_global_port_allocator = PortAllocator()


def get_free_port(start_port: int = 8080, max_range: int = 100) -> int:
    """thread-safe하게 비어 있는 포트를 얻는다.

    전역 포트 allocator를 사용해 동시 호출이 같은 포트를 반환하지 않게 한다. 포트는
    release_port()를 호출할 때까지 예약 상태로 표시된다.

    Args:
        start_port: 탐색을 시작할 포트 번호.
        max_range: 탐색할 최대 포트 개수.

    Returns:
        사용 가능한 포트 번호.

    Raises:
        RuntimeError: 지정한 범위에서 사용 가능한 포트를 찾지 못한 경우.
    """
    return _global_port_allocator.allocate(start_port, max_range)


def release_port(port: int) -> None:
    """앞서 할당한 포트를 해제한다.

    Args:
        port: 해제할 포트 번호.
    """
    _global_port_allocator.release(port)
