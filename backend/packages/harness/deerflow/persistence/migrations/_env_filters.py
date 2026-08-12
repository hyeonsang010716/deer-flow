"""alembic 범위를 DeerFlow 테이블로 한정하기 위해 ``env.py``가 쓰는 object filter.

LangGraph checkpointer 테이블은 같은 데이터베이스에 있지만 소유자는 LangGraph다. 이 filter가
없으면 ``alembic revision --autogenerate``가 그 테이블들을 리플렉션해 revision마다 엉뚱한
``drop_table`` 연산을 만들어 낸다.

``env.py``에 인라인하지 않고 별도 모듈로 둔 것은 alembic의 import 시점 기계장치를 끌어들이지
않고 단위 테스트할 수 있게 하기 위해서다.
"""

from __future__ import annotations

# LangGraph가 소유한 테이블 -- alembic이 이들에 대한 DDL을 제안해서는 안 된다.
LANGGRAPH_OWNED_TABLES: frozenset[str] = frozenset(
    {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
)


def include_object(object_, name, type_, reflected, compare_to):  # noqa: ARG001
    """LangGraph 소유 테이블이거나 부모 테이블이 LangGraph 소유인 index/constraint면 False를,
    그 외에는 True를 반환한다.

    시그니처는 alembic의 ``include_object`` 호출 계약
    ``(object, name, type_, reflected, compare_to)``와 일치한다.
    """
    if type_ == "table" and name in LANGGRAPH_OWNED_TABLES:
        return False
    parent_table = getattr(object_, "table", None)
    if parent_table is not None and getattr(parent_table, "name", None) in LANGGRAPH_OWNED_TABLES:
        return False
    return True
