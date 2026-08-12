"""to_dict를 자동 지원하는 SQLAlchemy declarative base.

모든 DeerFlow ORM model이 이 Base를 상속한다. SQLAlchemy의 inspect()를 이용해 범용 to_dict()
메서드를 제공하므로 각 model이 직렬화 로직을 따로 작성할 필요가 없다.

LangGraph의 checkpointer 테이블은 이 Base가 관리하지 **않는다**.
"""

from __future__ import annotations

from functools import cache

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import DeclarativeBase


@cache
def _column_keys(cls: type) -> tuple[str, ...]:
    """ORM class의 mapped column 키를 mapper 순서대로 반환한다.

    ``to_dict``/``__repr__``는 row마다 실행되므로(예: messages page 직렬화 시 event마다 한 번)
    SQLAlchemy mapper reflection 결과를 class 단위로 캐싱한다. 매핑은 class 정의 시점에 확정되므로
    이 캐시는 절대 낡지 않는다.
    """
    return tuple(c.key for c in sa_inspect(cls).mapper.column_attrs)


class Base(DeclarativeBase):
    """모든 DeerFlow ORM model의 기반 class.

    제공하는 것:
    - SQLAlchemy column inspection을 이용한 자동 to_dict().
    - 모든 column 값을 보여주는 표준 __repr__().
    """

    def to_dict(self, *, exclude: set[str] | None = None) -> dict:
        """ORM 인스턴스를 평범한 dict로 변환한다.

        캐싱된 mapped column 키를 사용한다(:func:`_column_keys` 참고).

        Args:
            exclude: 제외할 column 키 집합(선택).

        Returns:
            모든 mapped column에 대한 {column_key: value} dict.
        """
        keys = _column_keys(type(self))
        if exclude:
            return {k: getattr(self, k) for k in keys if k not in exclude}
        return {k: getattr(self, k) for k in keys}

    def __repr__(self) -> str:
        cols = ", ".join(f"{k}={getattr(self, k)!r}" for k in _column_keys(type(self)))
        return f"{type(self).__name__}({cols})"
