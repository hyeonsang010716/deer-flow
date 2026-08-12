"""PostgreSQL schema 이름에 대한 공용 검증."""

from __future__ import annotations

import re

# 소문자만 허용하는 것은 의도적이다. schema는 대소문자를 보존하는 *따옴표* 형태로 생성되지만
# (``CREATE SCHEMA IF NOT EXISTS "<schema>"``) *따옴표 없는* ``search_path`` 토큰으로 고정되고,
# PostgreSQL은 이를 소문자로 접는다. 대문자를 허용하면 둘이 어긋나 테이블이 조용히 ``public``에
# 만들어진다.
# 앵커를 쓰지 않는다. 검증에 ``re.fullmatch``를 써서 값 전체가 일치해야 한다.
# ``$`` 앵커를 쓴 ``re.match``는 끝의 개행(``"deerflow\n"``)을 통과시킨다. Python의 ``$``는
# 마지막 ``\n`` 직전에도 매치되므로, 실제로 ``deerflow\n``이라는 이름의 *따옴표* schema가
# 만들어지고 *따옴표 없는* ``search_path``는 ``deerflow``로 접혀 이를 찾지 못해 테이블이
# 조용히 ``public``에 만들어진다.
POSTGRES_SCHEMA_PATTERN = r"[a-z_][a-z0-9_]{0,62}"
_POSTGRES_SCHEMA_RE = re.compile(POSTGRES_SCHEMA_PATTERN)


def validate_postgres_schema(value: str) -> str:
    """v1 평문 식별자 PostgreSQL schema 계약을 검증한다."""
    if value == "":
        return value
    if not _POSTGRES_SCHEMA_RE.fullmatch(value):
        raise ValueError(f"postgres_schema must be a plain lowercase PostgreSQL identifier matching {POSTGRES_SCHEMA_PATTERN}; got {value!r}. Mixed-case and quoted identifiers are not supported.")
    return value
