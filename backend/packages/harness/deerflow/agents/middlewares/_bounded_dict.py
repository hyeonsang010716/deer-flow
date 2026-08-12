"""guard middleware들이 공유하는 작은 크기 제한 ``OrderedDict``.

guard middleware(``TokenBudgetMiddleware``, ``LoopDetectionMiddleware``)는
``run_id``별 상태를 들고 있는데, 버려지거나 재사용되는 run 때문에 무한정 커지면 안 된다.
이 모듈이 단일 공용 구현을 제공하므로 두 middleware가 동일하게 상한을 두고, 나중에 추가될
guard도 같은 것을 다시 만들지 않는다.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class BoundedDict(OrderedDict):
    """``maxsize``에 도달하면 가장 오래된 항목을 밀어내는 ``OrderedDict``.

    ``run_id``별 상태(stop-reason 플래그, 보류 중인 경고, 사용량 누적치)에 쓴다. lead agent에
    오래 살아 있는 middleware 인스턴스가 많은 run에 걸쳐 메모리를 누수하지 않게 한다.
    삽입 순서를 유지하므로 가장 먼저 삽입된 키부터 밀려난다.
    """

    def __init__(self, maxsize: int = 1000, *args: Any, **kwds: Any) -> None:
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self:
            if len(self) >= self.maxsize:
                self.popitem(last=False)
        super().__setitem__(key, value)
