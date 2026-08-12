"""PostgreSQL schema 헬퍼 (Issue #3380).

connection의 ``search_path``를 대상 schema로 고정하는 driver별 방식을 한곳에 모은다.
DeerFlow가 쓰는 두 PostgreSQL driver는 서로 다른 메커니즘을 기대한다:

- **asyncpg** (app ORM engine): SQLAlchemy ``connect_args``로 전달된
  ``server_settings``만 인식한다. libpq의 ``options=-c ...`` 문법은 이해하지 못한다.
- **psycopg** (LangGraph checkpointer/store): libpq의
  ``options=-c search_path=...`` connection 파라미터를 pool kwarg로 주거나
  DSN query string에 인코딩해서 쓴다.

schema 이름은 상위의
:class:`deerflow.config.database_config.DatabaseConfig`에서 평범한 식별자인지
검증된다. SQL을 만드는 헬퍼는 defense-in-depth로 경계에서 한 번 더 검증하고,
connection 인자 헬퍼는 driver payload를 조립하기만 한다.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


def build_asyncpg_connect_args(schema: str) -> dict:
    """asyncpg의 search_path를 고정하는 SQLAlchemy ``connect_args``를 반환한다.

    *schema*가 비어 있으면 ``{}``를 반환해 engine이 서버 기본값을 그대로 쓰게 한다.
    """
    if not schema:
        return {}
    return {"server_settings": {"search_path": schema}}


def build_psycopg_options(schema: str) -> str | None:
    """psycopg pool kwarg에 쓸 libpq ``options`` 값을 반환한다.

    *schema*가 비어 있으면 ``None``을 반환해 호출자가 kwarg 설정을 건너뛸 수 있게 한다.
    """
    if not schema:
        return None
    return f"-c search_path={schema}"


def _split_libpq_options(options: str) -> list[str]:
    """libpq ``options`` 문자열을 토큰으로 나눈다.

    libpq는 escape되지 않은 공백에서 나누며, backslash는 다음 문자를 escape한다
    (즉 ``\\ ``는 리터럴 공백, ``\\\\``는 리터럴 backslash). POSIX shell quoting이
    아니다 -- 여기서 작은/큰따옴표는 리터럴 문자다.
    """
    tokens: list[str] = []
    current: list[str] = []
    in_token = False
    escaped = False
    for char in options:
        if escaped:
            current.append(char)
            escaped = False
            in_token = True
            continue
        if char == "\\":
            escaped = True
            in_token = True
            continue
        if char.isspace():
            if in_token:
                tokens.append("".join(current))
                current = []
                in_token = False
            continue
        current.append(char)
        in_token = True
    if in_token:
        tokens.append("".join(current))
    return tokens


def _join_libpq_options(tokens: list[str]) -> str:
    """토큰들을 libpq ``options`` 문자열로 합친다.

    토큰 안의 공백과 backslash는 backslash로 escape해서 libpq가 각 토큰을 온전히
    유지하게 한다. ``shlex.join``은 쓸 수 없다: POSIX shell quoting(작은따옴표)을
    내보내는데 libpq는 그것을 리터럴 문자로 취급한다.

    공백 문자 전부를 escape한다. ``_split_libpq_options``가 backslash로 escape된
    TAB/CR/LF를 한 토큰의 일부로 보존하므로, 그냥 공백 바이트로 다시 합치면 libpq가
    그 지점에서 재토큰화해 호출자가 원래 갖고 있던 ``options`` 값을 깨뜨린다.
    """
    escaped = [re.sub(r"([\\\s])", r"\\\1", token) for token in tokens]
    return " ".join(escaped)


def _merge_search_path_option(existing_options: str, schema: str) -> str:
    """다른 옵션은 보존한 채 search_path만 교체한 libpq options를 반환한다."""
    new_option = build_psycopg_options(schema)
    if not new_option:
        return existing_options

    if not existing_options:
        return new_option

    tokens = _split_libpq_options(existing_options)

    merged: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-c" and index + 1 < len(tokens):
            setting = tokens[index + 1]
            if setting.split("=", 1)[0] == "search_path":
                index += 2
                continue
            merged.extend([token, setting])
            index += 2
            continue
        if token.startswith("-csearch_path="):
            index += 1
            continue
        merged.append(token)
        index += 1

    merged.extend(_split_libpq_options(new_option))
    return _join_libpq_options(merged)


def create_schema_sql(schema: str) -> str | None:
    """검증된 평범한 식별자에 대해 안전한 CREATE SCHEMA 문을 반환한다.

    defense-in-depth: 멀리 떨어진 pydantic validator를 믿지 않고 여기서 식별자를
    다시 검증한다. ``create_schema_sql``은 공개 export이고 psycopg는 ``;``로
    구분된 여러 문을 받으므로, ``DatabaseConfig``/``CheckpointerConfig``를 우회하는
    미래의 호출자(예: 테스트 헬퍼)가 이 f-string 경계로 SQL을 주입할 수 없어야 한다.
    """
    if not schema:
        return None
    from deerflow.config.postgres_schema import validate_postgres_schema

    validate_postgres_schema(schema)
    return f'CREATE SCHEMA IF NOT EXISTS "{schema}"'


def normalize_libpq_dsn(dsn: str) -> str:
    """SQLAlchemy ``+driver`` suffix를 제거한 *dsn*을 반환한다.

    ``DatabaseConfig.postgres_url``에는 ``postgresql+asyncpg://`` 같은 SQLAlchemy
    driver suffix가 붙어 있을 수 있다. psycopg의 libpq는 순수한
    ``postgres``/``postgresql`` scheme만 이해하므로 ``+asyncpg``가 붙은 DSN을 그대로
    ``psycopg.connect``에 넘기면 알아보기 힘든 파싱 에러가 난다. URL scheme이 없는
    keyword/DSN 문자열(``host=... dbname=...``)은 그대로 반환한다.

    PostgreSQL 계열이 아닌 URL scheme에는 ``ValueError``를 던진다.
    """
    parts = urlsplit(dsn)
    if not parts.scheme:
        return dsn
    scheme_base = parts.scheme.split("+", 1)[0]
    if scheme_base not in {"postgres", "postgresql"}:
        raise ValueError(f"Unsupported PostgreSQL DSN scheme for schema injection: {parts.scheme!r}")
    if scheme_base == parts.scheme:
        return dsn
    return urlunsplit((scheme_base, parts.netloc, parts.path, parts.query, parts.fragment))


def dsn_with_search_path(dsn: str, schema: str) -> str:
    """``options=-c search_path=<schema>`` query 파라미터를 붙인 *dsn*을 반환한다.

    pool kwarg 대신 DSN 문자열을 받는 psycopg ``from_conn_string`` 호출부에서 쓴다.
    값에 공백과 ``=``가 들어 있어 둘 다 percent-encoding 해야 libpq가 URL을 올바로
    파싱한다.

    libpq는 URI query 값에서 ``%XX`` percent-encoding만 인식하고 ``+``를 공백으로
    취급하지 않는다(그건 HTML form 관례다). 따라서 공백은 반드시 ``+``가 아닌
    ``%20``으로 인코딩해야 한다. 아니면 libpq는 ``-c+search_path=...``라는 깨진 토큰
    하나로 보고 search_path가 아예 적용되지 않는다. 기존 query 파라미터는 보존한다.
    *schema*가 비어 있으면 *dsn*을 그대로 반환한다.
    """
    if not schema:
        return dsn
    parts = urlsplit(dsn)

    if not parts.scheme:
        from psycopg.conninfo import conninfo_to_dict, make_conninfo

        params = conninfo_to_dict(dsn)
        params["options"] = _merge_search_path_option(params.get("options", ""), schema)
        return make_conninfo(**params)

    # DatabaseConfig.postgres_url에는 ``postgresql+asyncpg://`` 같은 SQLAlchemy driver
    # suffix가 붙을 수 있다. psycopg의 libpq는 순수한 ``postgres``/``postgresql``
    # scheme만 이해하므로, 복합 형태는 받아들이되 ``+driver`` 부분을 떼어 psycopg가
    # 소비할 수 있는 DSN을 내보낸다.
    scheme_base = parts.scheme.split("+", 1)[0]
    if scheme_base not in {"postgres", "postgresql"}:
        raise ValueError(f"Unsupported PostgreSQL DSN scheme for schema injection: {parts.scheme!r}")

    options_values: list[str] = []
    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "options":
            options_values.append(value)
        else:
            query_pairs.append((key, value))

    options = _merge_search_path_option(" ".join(options_values), schema)
    query_pairs.append(("options", options))
    # quote_via=quote는 공백을 +(form 방식)가 아니라 %20(libpq에 안전)으로 인코딩한다.
    query = urlencode(query_pairs, quote_via=quote)
    return urlunsplit((scheme_base, parts.netloc, parts.path, query, parts.fragment))


def ensure_postgres_schema(conn_string: str, schema: str, *, install_hint: str) -> None:
    """새 동기 psycopg connection으로 *schema*를 생성한다.

    *schema*가 비어 있으면 아무것도 하지 않는다. ``psycopg`` 의존성이 없으면
    *install_hint*로 매핑해서, 호출자가 backend의 나머지 부분과 동일한 조치 가능한
    메시지를 노출하게 한다. DSN은 SQLAlchemy ``+driver`` suffix가 libpq에 닿지 않도록
    정규화한다.
    """
    statement = create_schema_sql(schema)
    if statement is None:
        return
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(install_hint) from exc

    # psycopg 3의 ``Connection.__exit__``는 commit/rollback만 하고 connection을 닫지
    # 않는다(psycopg2->3에서 문서화된 변경). GC까지 새게 두지 말고 async 쪽과 똑같이
    # try/finally로 libpq connection을 확정적으로 반환한다.
    conn = psycopg.connect(normalize_libpq_dsn(conn_string), autocommit=True)
    try:
        conn.execute(statement)
    finally:
        conn.close()


async def ensure_postgres_schema_async(conn_string: str, schema: str, *, install_hint: str) -> None:
    """:func:`ensure_postgres_schema`의 async 버전."""
    statement = create_schema_sql(schema)
    if statement is None:
        return
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(install_hint) from exc

    conn = await psycopg.AsyncConnection.connect(normalize_libpq_dsn(conn_string), autocommit=True)
    try:
        await conn.execute(statement)
    finally:
        await conn.close()
