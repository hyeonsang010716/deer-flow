"""extension에 전달되는 scope별 타입 기반 저장소."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any


class ExtensionData:
    """host 소유 scope 하나에 붙는 extension 전용 상태.

    문자열이 아니라 타입을 key로 쓰므로 서로 무관한 extension끼리 key가 충돌하지 않는다.
    host는 scope(app, task)마다 인스턴스를 하나 만들고 그 scope가 끝나면 버린다. extension이
    stale handle 검사를 할 필요가 없는 이유가 이것이다. store를 붙잡아 두는 대신 매 callback
    마다 현재 scope의 store를 전달받는다.
    """

    __slots__ = ("_scope_id", "_entries", "_lock")

    def __init__(self, scope_id: str) -> None:
        self._scope_id = scope_id
        self._entries: dict[type, Any] = {}
        self._lock = RLock()

    @property
    def scope_id(self) -> str:
        """이 store가 붙어 있는 scope의 host 식별자."""
        return self._scope_id

    def get[T](self, typ: type[T]) -> T | None:
        with self._lock:
            return self._entries.get(typ)

    def get_or_init[T](self, typ: type[T], init: Callable[[], T]) -> T:
        """저장된 값을 반환하고, 없으면 ``init``으로 만들어 반환한다.

        ``init``은 store가 잠긴 상태에서 실행된다. 이 store의 다른 상태를 조합해도 되지만,
        무거운 lazy 작업은 저장되는 값 자체 안에 두어야 한다.
        """
        with self._lock:
            existing = self._entries.get(typ)
            if existing is not None:
                return existing
            created = init()
            self._entries[typ] = created
            return created

    def set[T](self, value: T) -> None:
        with self._lock:
            self._entries[type(value)] = value

    def remove[T](self, typ: type[T]) -> T | None:
        with self._lock:
            return self._entries.pop(typ, None)
