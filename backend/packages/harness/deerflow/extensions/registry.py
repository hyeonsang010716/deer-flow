"""등록 단계 registry와 그 결과물인 불변 런타임 객체.

extension에게는 쓰기 전용 공개 ``ExtensionRegistry`` 계약만 보인다. 실제 host 타입은
추가로 출처 표기(attribution), rollback, 불변 런타임 투영까지 담당한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from deerflow_extension_api import ExtensionData, MiddlewareContributor
from deerflow_extension_api import ExtensionRegistry as ExtensionRegistryContract

_Entry = tuple[str, Any]


@dataclass(frozen=True)
class LoadedExtensions:
    """런타임에서 사용하는 불변 뷰.

    모든 항목이 source 문자열을 함께 들고 다니므로 진단, 출처 추적, 순서 오류 메시지가
    원인 extension을 지목할 수 있다.
    """

    app_store: ExtensionData
    middleware_contributors: tuple[tuple[str, MiddlewareContributor], ...] = ()

    # 메서드가 아니라 미리 계산된 속성이다. hook 지점은 속성 하나만 읽고 빠져나가므로
    # extension이 없는 경로에서는 아무것도 생성하지 않는다.
    has_middleware_contributors: bool = False
    needs_task_store: bool = False


class ExtensionRegistry(ExtensionRegistryContract):
    """등록 단계에서만 쓰이는 가변 registry.

    공개 계약 Protocol을 상속해 host 구현이 extension이 참조하는 타입과 맞는지 검사받는다.
    아래의 host 전용 기능(attribution, discard, mark/rollback_to, build)은 의도적으로
    계약 밖에 둔다.
    """

    def __init__(self) -> None:
        self._middlewares: list[_Entry] = []
        self._current_source: str | None = None

    @contextmanager
    def attributed_to(self, source: str) -> Iterator[None]:
        """블록 안에서 등록된 모든 항목의 출처를 ``source``로 기록한다."""
        previous = self._current_source
        self._current_source = source
        try:
            yield
        finally:
            self._current_source = previous

    def _source(self) -> str:
        if self._current_source is None:
            raise RuntimeError("registration must happen inside ExtensionRegistry.attributed_to(...)")
        return self._current_source

    def middlewares(self, contributor: MiddlewareContributor) -> None:
        self._middlewares.append((self._source(), contributor))

    def discard(self, source: str) -> None:
        """``source``가 등록한 모든 항목을 제거한다.

        install()이 도중에 예외를 던졌을 때 호출한다. 절반만 등록된 extension은
        아예 없는 것보다 위험하다. 만들어내는 데이터가 완전한 것처럼 보이기 때문이다.

        Note: source 문자열로 매칭하므로 두 spec이 같은 ``use``를 다른 config로 쓰는
        경우에는 안전하지 않다. 성공적으로 설치된 다른 인스턴스의 항목까지 지운다.
        install()을 하나씩 처리하는 호출자는 ``mark()``/``rollback_to()``를 쓴다.
        """
        self._middlewares[:] = [entry for entry in self._middlewares if entry[0] != source]

    def mark(self) -> int:
        """install() 하나를 위치 기반으로 되돌릴 수 있도록 버킷 길이를 스냅샷한다."""
        return len(self._middlewares)

    def rollback_to(self, mark: int) -> None:
        """``mark`` 이후의 모든 등록을 되돌린다.

        source가 아니라 위치를 기준으로 한다. 두 spec이 같은 ``use`` 문자열을 다른
        config로 쓰는 경우가 정당하게 존재하는데, source로 지우면 다른 인스턴스의
        성공한 등록까지 함께 사라지기 때문이다.
        """
        del self._middlewares[mark:]

    def build(self) -> LoadedExtensions:
        return LoadedExtensions(
            app_store=ExtensionData("app"),
            middleware_contributors=tuple(self._middlewares),
            has_middleware_contributors=bool(self._middlewares),
            needs_task_store=bool(self._middlewares),
        )


#: extension을 하나도 로드하지 않는 host가 공유하는 빈 인스턴스.
EMPTY_EXTENSIONS = ExtensionRegistry().build()
