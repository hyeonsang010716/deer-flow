"""SQLAlchemy용 dialect 인식 JSON 값 매칭(SQLite + PostgreSQL)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import BigInteger, Float, String, bindparam
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import ColumnElement
from sqlalchemy.sql.visitors import InternalTraversal
from sqlalchemy.types import Boolean, TypeEngine

# key는 컴파일된 SQL에 그대로 삽입되므로, injection을 막기 위해 문자 집합을 제한한다.
_KEY_CHARSET_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# metadata filter 값으로 허용되는 타입(JsonMatch가 받는 집합과 동일).
ALLOWED_FILTER_VALUE_TYPES: tuple[type, ...] = (type(None), bool, int, float, str)

# SQLite는 부호 있는 64비트 범위를 벗어난 값을 바인딩하면 overflow를 낸다. PostgreSQL은
# BIGINT cast 중에 overflow가 난다. 대신 검증 시점에 거부한다.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def validate_metadata_filter_key(key: object) -> bool:
    """*key*를 JSON metadata filter key로 써도 안전하면 True를 반환한다.

    ``[A-Za-z0-9_-]+``에 일치하는 문자열일 때 "안전"하다. key가 컴파일된 SQL path
    표현식(``$."<key>"`` / ``->`` 리터럴)에 그대로 삽입되므로 문자 집합을 제한한다. 더 느슨한
    패턴은 SQL/JSONPath injection 표면을 열게 된다.
    """
    return isinstance(key, str) and bool(_KEY_CHARSET_RE.match(key))


def validate_metadata_filter_value(value: object) -> bool:
    """*value*가 JSON metadata filter에 허용되는 타입이면 True를 반환한다.

    ``_build_clause``가 dialect 이식 가능한 predicate로 컴파일할 줄 아는 타입 집합과 일치한다.
    그 외(list/dict/bytes/...)는 ``str()``로 조용히 변환하지 않고 의도적으로 거부한다. 조용한
    변환은 (a) 잘못된 매칭을 만들고 (b) ``value``가 unhashable일 때 SQLAlchemy의
    ``inherit_cache`` 불변식을 깨뜨린다.

    정수 값은 추가로 부호 있는 64비트 범위 ``[-2**63, 2**63 - 1]``로 제한된다. 더 큰 값은
    SQLite에서 바인딩 시, PostgreSQL에서 ``BIGINT`` cast 중에 overflow가 난다.
    """
    if not isinstance(value, ALLOWED_FILTER_VALUE_TYPES):
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        if not (_INT64_MIN <= value <= _INT64_MAX):
            return False
    return True


class JsonMatch(ColumnElement):
    """JSON column에 대한 dialect 이식 가능한 ``column[key] == value``.

    SQLite에서는 ``json_type``/``json_extract``로, PostgreSQL에서는 ``json_typeof``/``->>``로
    컴파일된다. bool과 int, NULL과 key 부재를 구분하는 타입 안전 비교를 한다.

    *key*는 ``[A-Za-z0-9_-]+``에 일치하는 단일 리터럴 key여야 한다.
    *value*는 ``None``, ``bool``, ``int``(부호 있는 64비트), ``float``, ``str`` 중 하나여야 한다.
    """

    inherit_cache = True
    type = Boolean()
    _is_implicitly_boolean = True

    _traverse_internals = [
        ("column", InternalTraversal.dp_clauseelement),
        ("key", InternalTraversal.dp_string),
        ("value", InternalTraversal.dp_plain_obj),
    ]

    def __init__(self, column: ColumnElement, key: str, value: object) -> None:
        if not validate_metadata_filter_key(key):
            raise ValueError(f"JsonMatch key must match {_KEY_CHARSET_RE.pattern!r}; got: {key!r}")
        if not validate_metadata_filter_value(value):
            if isinstance(value, int) and not isinstance(value, bool):
                raise TypeError(f"JsonMatch int value out of signed 64-bit range [-2**63, 2**63-1]: {value!r}")
            raise TypeError(f"JsonMatch value must be None, bool, int, float, or str; got: {type(value).__name__!r}")
        self.column = column
        self.key = key
        self.value = value
        super().__init__()


@dataclass(frozen=True)
class _Dialect:
    """JSON 타입/값 비교를 생성할 때 쓰는 dialect별 이름들."""

    null_type: str
    num_types: tuple[str, ...]
    num_cast: str
    int_types: tuple[str, ...]
    int_cast: str
    # SQLite에서는 json_type이 이미 'integer'/'real'을 반환하므로 None이다.
    # PostgreSQL에서는 json_typeof가 int와 float 모두에 'number'를 반환하므로, float에서
    # CAST 오류가 나지 않도록 추가 guard용 regex 리터럴을 둔다.
    int_guard: str | None
    string_type: str
    bool_type: str | None


_SQLITE = _Dialect(
    null_type="null",
    num_types=("integer", "real"),
    num_cast="REAL",
    int_types=("integer",),
    int_cast="INTEGER",
    int_guard=None,
    string_type="text",
    bool_type=None,
)

_PG = _Dialect(
    null_type="null",
    num_types=("number",),
    num_cast="DOUBLE PRECISION",
    int_types=("number",),
    int_cast="BIGINT",
    int_guard="'^-?[0-9]+$'",
    string_type="string",
    bool_type="boolean",
)


def _bind(compiler: SQLCompiler, value: object, sa_type: TypeEngine[Any], **kw: Any) -> str:
    param = bindparam(None, value, type_=sa_type)
    return compiler.process(param, **kw)


def _type_check(typeof: str, types: tuple[str, ...]) -> str:
    if len(types) == 1:
        return f"{typeof} = '{types[0]}'"
    quoted = ", ".join(f"'{t}'" for t in types)
    return f"{typeof} IN ({quoted})"


def _build_clause(compiler: SQLCompiler, typeof: str, extract: str, value: object, dialect: _Dialect, **kw: Any) -> str:
    if value is None:
        return f"{typeof} = '{dialect.null_type}'"
    if isinstance(value, bool):
        # Python에서 bool은 int의 하위 클래스이므로 bool 검사가 int 검사보다 먼저 와야 한다
        bool_str = "true" if value else "false"
        if dialect.bool_type is None:
            return f"{typeof} = '{bool_str}'"
        return f"({typeof} = '{dialect.bool_type}' AND {extract} = '{bool_str}')"
    if isinstance(value, int):
        bp = _bind(compiler, value, BigInteger(), **kw)
        if dialect.int_guard:
            # json_typeof = 'number'가 float에도 걸리므로, CASE로 CAST 오류를 막는다
            return f"(CASE WHEN {_type_check(typeof, dialect.int_types)} AND {extract} ~ {dialect.int_guard} THEN CAST({extract} AS {dialect.int_cast}) END = {bp})"
        return f"({_type_check(typeof, dialect.int_types)} AND CAST({extract} AS {dialect.int_cast}) = {bp})"
    if isinstance(value, float):
        bp = _bind(compiler, value, Float(), **kw)
        return f"({_type_check(typeof, dialect.num_types)} AND CAST({extract} AS {dialect.num_cast}) = {bp})"
    bp = _bind(compiler, str(value), String(), **kw)
    return f"({typeof} = '{dialect.string_type}' AND {extract} = {bp})"


@compiles(JsonMatch, "sqlite")
def _compile_sqlite(element: JsonMatch, compiler: SQLCompiler, **kw: Any) -> str:
    if not validate_metadata_filter_key(element.key):
        raise ValueError(f"Key escaped validation: {element.key!r}")
    col = compiler.process(element.column, **kw)
    path = f'$."{element.key}"'
    typeof = f"json_type({col}, '{path}')"
    extract = f"json_extract({col}, '{path}')"
    return _build_clause(compiler, typeof, extract, element.value, _SQLITE, **kw)


@compiles(JsonMatch, "postgresql")
def _compile_pg(element: JsonMatch, compiler: SQLCompiler, **kw: Any) -> str:
    if not validate_metadata_filter_key(element.key):
        raise ValueError(f"Key escaped validation: {element.key!r}")
    col = compiler.process(element.column, **kw)
    typeof = f"json_typeof({col} -> '{element.key}')"
    extract = f"({col} ->> '{element.key}')"
    return _build_clause(compiler, typeof, extract, element.value, _PG, **kw)


@compiles(JsonMatch)
def _compile_default(element: JsonMatch, compiler: SQLCompiler, **kw: Any) -> str:
    raise NotImplementedError(f"JsonMatch supports only sqlite and postgresql; got dialect: {compiler.dialect.name}")


def json_match(column: ColumnElement, key: str, value: object) -> JsonMatch:
    return JsonMatch(column, key, value)
