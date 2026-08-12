"""sandbox 명령 실행을 위한 환경 변수 정책(issue #3861).

skill script는 sandbox subprocess로 실행된다. 기본적으로 subprocess는 Gateway 프로세스의
``os.environ`` 전체를 상속받는데, 거기에는 플랫폼 credential(``OPENAI_API_KEY``, tracing 키,
community provider 키 등)이 들어 있다. 그러면 범위를 좁힌 request-secret 주입이 무의미해진다.
script가 상속된 플랫폼 secret을 그냥 읽으면 되기 때문이다. 이 모듈은 request-scoped secret을
얹기 전에 상속 환경에서 secret처럼 보이는 변수를 걸러낸다.

패턴 집합은 codex의 ``*KEY*/*SECRET*/*TOKEN*`` 기본 제외와 hermes의 고정 provider blocklist를
따른다. 다만 제외를 기본 *꺼짐*으로 두는 codex와 달리, DeerFlow는 보안 우선으로 기본 제거한다.
"""

from __future__ import annotations

import fnmatch
import os

# secret처럼 보이는 변수 이름에 대한 대소문자 무시 wildcard 패턴. 대문자로 변환한 변수 이름에
# 매칭한다. 무해한 시스템 변수(PATH, HOME, SHELL, LANG, PWD, TMPDIR, VIRTUAL_ENV, PYTHONPATH
# 등)는 이 토큰들을 포함하지 않으므로 그대로 유지된다.
_SECRET_NAME_PATTERNS: tuple[str, ...] = (
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    # ``*PASS*``는 완전한 ``PASSWORD``/``PASSWD`` 표기는 물론, 값 자체가 평문 비밀번호인 흔한
    # 축약형(``DB_PASS``, ``SMTP_PASS``, ``MYSQL_PASS`` 등)까지 포함한다. libpq의 ``.pgpass``
    # 위치를 가리키는 ``PGPASSFILE``도 걸린다.
    #
    # ``*_ASKPASS`` credential helper(``GIT_ASKPASS``, ``SSH_ASKPASS``, ``SUDO_ASKPASS``)도
    # 의도적으로 잡는다. 이들은 secret이 아니라 *프로그램*을 가리키지만, 그 프로그램의 존재
    # 이유가 호출자에게 credential을 건네주는 것이다. 포인터를 상속하는 것도 이 모듈이 막으려는
    # 것과 같은 유출 유형이므로, 제거는 우연이 아니라 의도다.
    #
    # 단지 ``PASS``를 포함할 뿐인 이름(``COMPASS_*``, ``BYPASS_*``)도 함께 제거된다. 이것이 이
    # 모듈의 fail-safe 방향이다. 제거된 이름이 정말 필요한 skill은 required-secrets로 선언하면
    # 된다. 무해한 ``PWD``/``OLDPWD``에는 ``PASS`` 부분 문자열이 없어 영향이 없다.
    "*PASS*",
    "*CREDENTIAL*",
    "*DSN*",  # data source name — 거의 항상 비밀번호가 들어간 connection string이다
)

# KEY/SECRET/TOKEN/DSN 부분 문자열은 없지만 비밀번호를 흔히 품는 connection-string /
# credential 변수 이름(예: ``postgresql://user:pw@host/db``). ``*URL*`` 전면 차단은 의도적으로
# 피한다. skill이 정당하게 읽을 수 있는 무해한 service URL까지 날려버리기 때문이다. 이 중 하나가
# 정말 필요한 skill은 required-secrets로 선언해야 한다(그러면 호출자가 context.secrets로 값을
# 공급하고, 주입이 이긴다).
#
# 해당 클라이언트가 직접 읽는 credential 소스에도 같은 논리가 적용된다. ``MYSQL_PWD``와
# ``REDISCLI_AUTH``는 ``mysql``과 ``redis-cli``의 문서화된 무플래그 credential 소스다.
# ``REDIS_AUTH``는 표준 Redis 클라이언트의 정식 이름이 *아니지만*, 클라이언트 라이브러리와 배포
# chart가 흔히 설정하므로 방어적으로 차단한다. ``PGSERVICEFILE``은 Postgres 쪽 대응물이다.
# libpq가 플래그 없이 그것이 가리키는 ``pg_service.conf``(비밀번호 필드를 담을 수 있다)를 읽기
# 때문이다. 형제격인 ``PGPASSFILE``은 이미 ``*PASS*``가 잡는다. 이들은 정확한 이름으로 넣어야
# 한다. ``PWD``/``AUTH``/``SERVICEFILE``은 wildcard로 만들 수 없는데, ``*PWD*``는
# ``PWD``/``OLDPWD``까지 날리고 이들에만 고유한 공통 토큰도 없기 때문이다. (``*PASS*``가 이미
# ``PGPASSWORD``, ``MYSQL_PASSWORD``, ``DB_PASS``, ``PGPASSFILE`` 등을 포함한다.)
_BLOCKED_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "DATABASE_URL",
        "DATABASE_URI",
        "REDIS_URL",
        "MONGODB_URI",
        "MONGO_URL",
        "AMQP_URL",
        "RABBITMQ_URL",
        "POSTGRES_URL",
        "POSTGRESQL_URL",
        "MYSQL_URL",
        "CLICKHOUSE_URL",
        "CONNECTION_STRING",
        "CONN_STR",
        "GH_PAT",
        "GITHUB_PAT",
        "MYSQL_PWD",
        "REDISCLI_AUTH",
        "REDIS_AUTH",
        "PGSERVICEFILE",
    }
)


def is_blocked_env_name(name: str) -> bool:
    """``name``이 sandbox subprocess가 상속하면 안 되는 credential처럼 보이면 True를 반환한다."""
    upper = name.upper()
    if upper in _BLOCKED_EXACT_NAMES:
        return True
    return any(fnmatch.fnmatchcase(upper, pattern) for pattern in _SECRET_NAME_PATTERNS)


def build_sandbox_env(injected: dict[str, str] | None = None) -> dict[str, str]:
    """sandbox subprocess에 쓸 환경 dict를 만든다.

    ``os.environ``에서 secret처럼 보이는 변수를 뺀 뒤, 명시적으로 주입된 request-scoped secret을
    그 위에 얹는다. 주입된 secret은 이름이 차단 패턴에 걸리더라도 이긴다. 주입은 상위에서 이미
    인가된 것이기 때문이다(skill이 선언했고 값도 host 환경이 아니라 요청에서 왔다).
    """
    env = {key: value for key, value in os.environ.items() if not is_blocked_env_name(key)}
    if injected:
        env.update(injected)
    return env
