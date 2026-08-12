"""alembic column revision을 위한 멱등 헬퍼.

``versions/``의 column revision은 raw ``op.add_column`` / ``op.drop_column`` 대신 이 헬퍼를
써야 한다. 그래야 이미 해당 컬럼이 있는(또는 이미 제거한) DB에 column 변경을 다시 실행해도
안전한 no-op이 된다.

멱등성이 필요한 이유는 두 가지다.

1. **bootstrap 잠금 위에 얹는 defence-in-depth.** ``bootstrap_schema()``는 Postgres를
   advisory lock으로, SQLite를 한 프로세스 안에서 ``asyncio.Lock``으로 직렬화한다. 그럼에도
   재시도가 일어나면(수동 ALTER, 설정 오류, SQLite의 프로세스 간 경합) revision은 여전히
   다시 실행해도 안전해야 한다.

2. **``Base.metadata.create_all``을 관대하게 만든 것과 같은 태도.** ``create_all``은 이미
   있는 테이블을 건너뛴다. column migration도 이미 원하는 상태인 컬럼을 건너뛰어 같은
   관대함을 유지해야 한다.

Drift 경고
---------

이름만 맞춰 보면, 수동 ``ALTER``가 남긴 컬럼이 ``Base.metadata.create_all``이 새 DB에서
만들어낼 형태와 다른데도 가려질 수 있다(예: ``NOT NULL DEFAULT '{}'`` 없이
``ALTER TABLE runs ADD COLUMN token_usage_by_model JSON``을 실행한 #3682 우회, 또는 타입이
틀린 변종 ``ALTER TABLE runs ADD COLUMN token_usage_by_model TEXT NOT NULL DEFAULT '{}'``).
이런 조용한 drift를 드러내기 위해 ``safe_add_column``은 기존 컬럼의 ``nullable`` /
``server_default`` / ``type``을 원하는 ``sa.Column``과 비교하고 불일치 시
``logger.warning``을 낸다. 타입 비교는 ``_type_equivalent``를 거치며, 알려진 dialect 동의어
쌍(예: ``JSON``과 ``JSONB``)을 동등하게 취급해 false positive를 피하면서도 ``TEXT``와
``JSON`` 같은 전면적 타입 불일치는 잡아낸다. 자동 수리는 하지 않는다. 운영자가 알아채고
판단하기에는 경고로 충분하다.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _normalize_default(value: object) -> str | None:
    """서로 다른 출처끼리 비교할 수 있도록 server-default 값을 정규화한다.

    원하는 값은 ``sa.Column.server_default``(``DefaultClause`` / ``TextClause`` 리터럴,
    ``None``, 또는 Python 리터럴)에서 오고, 반영된 값은
    ``Inspector.get_columns()['default']``에서 dialect가 렌더링한 문자열로 온다. 바깥 괄호,
    공백, Postgres 스타일 타입 캐스트를 제거해 텍스트상 동등한 형태가 dialect를 넘어 같게
    비교되도록 한다.
    """
    if value is None:
        return None
    if isinstance(value, sa.sql.elements.TextClause):
        text = value.text
    elif isinstance(value, sa.schema.DefaultClause) and isinstance(value.arg, sa.sql.elements.TextClause):
        text = value.arg.text
    else:
        text = str(value)
    text = text.strip()
    # 일부 dialect가 default를 감싸는 바깥 괄호 한 겹을 제거한다.
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    # ``'{}'::jsonb`` 같은 Postgres 스타일 타입 캐스트를 제거한다.
    if "::" in text:
        text = text.split("::", 1)[0].strip()
    return text or None


def _normalize_type(value: object) -> str:
    """비교를 위해 SQLAlchemy ``TypeEngine``(또는 반영된 타입)을 정규화한다.

    파라미터를 제거한 대문자 타입 클래스 이름을 반환한다(예: ``JSON()`` → ``"JSON"``,
    ``VARCHAR(255)`` → ``"VARCHAR"``). 길이 파라미터는 의도적으로 버린다. drift 경고는
    dialect가 렌더링한 크기 기본값이 아니라 전면적 타입 오설정(JSON 대 TEXT 리뷰 사례)을
    겨냥하기 때문이다. 빈 문자열은 "정보 없음"을 뜻하므로 호출자는 빈 문자열을 동등 비교해서는
    안 된다.
    """
    if value is None:
        return ""
    s = value if isinstance(value, str) else repr(value)
    return s.upper().split("(", 1)[0].strip()


# 타입 drift 경고를 내면 안 되는, 알려진 dialect 동의어 쌍. Postgres는 ``JSON``을 ``JSONB``로
# 반영한다(컬럼을 어떻게 만들었는지에 따라 반대도 있다). 모델의 ``sa.JSON``과 이 allowlist가
# Postgres 배포를 조용하게 유지하면서도 ``TEXT NOT NULL DEFAULT '{}'`` 재추가 같은 진짜 타입
# 오류는 잡아낸다.
#
# 새 쌍은 실제 배포에서 반영값과 모델의 불일치가 false positive임이 입증됐을 때만 추가한다.
# 미리 넣지 않는다. 동등성 범위가 너무 넓어지면 이 헬퍼가 막으려는 조용한 drift 구멍이 다시
# 열리기 때문이다.
_EQUIVALENT_TYPE_FAMILIES: tuple[frozenset[str], ...] = (frozenset({"JSON", "JSONB"}),)


def _type_equivalent(actual: object, desired: object) -> bool:
    """*actual*과 *desired*가 같은 타입이거나 알려진 동등 타입이면 True.

    어느 한쪽에 반영 정보가 없으면 True를 반환해서, 정보 누락이 시끄러운 경고로
    false positive가 되지 않게 한다.
    """
    a = _normalize_type(actual)
    d = _normalize_type(desired)
    if not a or not d:
        return True
    if a == d:
        return True
    pair = frozenset({a, d})
    return any(pair <= fam for fam in _EQUIVALENT_TYPE_FAMILIES)


def _check_column_drift(table: str, desired: sa.Column, actual: dict) -> None:
    """기존 컬럼의 속성이 원하는 모델과 어긋나면 경고한다.

    ``nullable``과 ``server_default``는 직접 동등 비교하고, ``type``은 ``_type_equivalent``로
    비교한다(``JSON``과 ``JSONB`` 같은 알려진 dialect 동의어 쌍을 동등하게 취급한다). 타입이
    문제된 차원인지와 무관하게 반영된 타입과 원하는 타입의 repr을 경고 payload에 함께 실어,
    로그를 분류하는 운영자가 타입 맥락을 한눈에 보게 한다.
    """
    diffs: list[str] = []

    desired_nullable = True if desired.nullable is None else bool(desired.nullable)
    actual_nullable = bool(actual.get("nullable", True))
    if desired_nullable != actual_nullable:
        diffs.append(f"nullable actual={actual_nullable} desired={desired_nullable}")

    desired_default = _normalize_default(desired.server_default)
    actual_default = _normalize_default(actual.get("default"))
    if desired_default != actual_default:
        diffs.append(f"server_default actual={actual_default!r} desired={desired_default!r}")

    if not _type_equivalent(actual.get("type"), desired.type):
        diffs.append(f"type actual={_normalize_type(actual.get('type'))!r} desired={_normalize_type(desired.type)!r}")

    if diffs:
        logger.warning(
            "safe_add_column: %s.%s already exists but drifts from the model definition (%s); actual_type=%r desired_type=%r; leaving as-is -- a manual ALTER may be needed to match the model.",
            table,
            desired.name,
            "; ".join(diffs),
            actual.get("type"),
            desired.type,
        )


def safe_add_column(table: str, column: sa.Column) -> None:
    """테이블이나 컬럼이 없거나 이미 있으면 no-op이 되는 ``op.add_column``.

    - 테이블이 없으면 추가할 대상이 없다. bootstrap은 baseline 테이블 집합을 이미 가진 legacy
      DB만 지원하므로 조용히 건너뛴다.
    - 컬럼이 이미 있으면 no-op. 반환 전에 ``_check_column_drift``가 기존 컬럼의 nullability /
      server_default / type을 원하는 ``column``과 비교하고 불일치 시 ``logger.warning``을 내서,
      수동으로 적용한 우회가 잠재 drift로 조용히 남지 않게 한다.
    """
    insp = _inspector()
    if table not in insp.get_table_names():
        return
    existing = {c["name"]: c for c in insp.get_columns(table)}
    if column.name in existing:
        _check_column_drift(table, column, existing[column.name])
        return
    with op.batch_alter_table(table) as batch:
        batch.add_column(column)


def safe_drop_column(table: str, column_name: str) -> None:
    """테이블이나 컬럼이 이미 사라졌으면 no-op이 되는 ``op.drop_column``."""
    insp = _inspector()
    if table not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns(table)}
    if column_name not in existing:
        return
    with op.batch_alter_table(table) as batch:
        batch.drop_column(column_name)
