"""사용자 기반 authorization을 위한 request-scoped user context.

이 모듈은 gateway의 auth middleware가 인증 성공 후 설정하는
:class:`~contextvars.ContextVar`를 보관한다. repository 메서드는 sentinel 기본값
파라미터를 통해 이 contextvar를 읽으므로, router는 ``user_id`` 보일러플레이트에서
자유로워진다.

repository ``user_id`` 파라미터의 세 가지 상태(이 모듈의 소비자 쪽은
``deerflow.persistence.*``에 있다):

- ``_AUTO``(모듈 전용 sentinel, 기본값): contextvar에서 읽고, 설정되지 않았으면
  :class:`RuntimeError`를 던진다.
- 명시적 ``str``: 주어진 값을 사용하며 contextvar를 덮어쓴다.
- 명시적 ``None``: WHERE 절 없음 — 의도적으로 격리를 우회하는 마이그레이션 스크립트와
  admin CLI에서만 사용한다.

의존 방향
--------------------
``persistence``(하위 레이어)가 이 모듈에서 읽고, ``gateway.auth``(상위 레이어)가 여기에
쓴다. ``CurrentUser``는 :class:`typing.Protocol`로 정의되어 있어서 ``persistence``가
``gateway.auth.models``의 구체 ``User`` 클래스를 import할 필요가 전혀 없다. ``.id: str``
속성을 가진 객체라면 구조적으로 이 protocol을 만족한다.

Asyncio 의미론
-----------------
``ContextVar``는 asyncio에서 thread-local이 아니라 task-local이다. 각 FastAPI 요청은
자체 task에서 실행되므로 context는 자연스럽게 격리된다. ``asyncio.create_task``와
``asyncio.to_thread``는 부모 task의 context를 상속하며 보통 이것이 의도한 동작이다.
background task가 foreground 사용자를 *보면 안 되는* 경우에는 ``contextvars.copy_context()``로
감싸 깨끗한 복사본을 만든다.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Final, Protocol, runtime_checkable


@runtime_checkable
class CurrentUser(Protocol):
    """현재 인증된 사용자를 나타내는 구조적 타입.

    ``.id: str`` 속성을 가진 객체라면 이 protocol을 만족한다. 구체 구현은
    ``app.gateway.auth.models.User``에 있다.
    """

    id: str


_current_user: Final[ContextVar[CurrentUser | None]] = ContextVar("deerflow_current_user", default=None)


def set_current_user(user: CurrentUser) -> Token[CurrentUser | None]:
    """현재 async task의 사용자를 설정한다.

    반환된 reset token은 ``finally`` 블록에서 :func:`reset_current_user`에 넘겨 이전
    context를 복원해야 한다.
    """
    return _current_user.set(user)


def reset_current_user(token: Token[CurrentUser | None]) -> None:
    """``token``이 담고 있는 상태로 context를 복원한다."""
    _current_user.reset(token)


def get_current_user() -> CurrentUser | None:
    """현재 사용자를 반환하고, 설정되어 있지 않으면 ``None``을 반환한다.

    어떤 context에서 호출해도 안전하다. 사용자 없이도 진행할 수 있는 코드 경로
    (예: 마이그레이션 스크립트, 공개 endpoint)에서 사용한다.
    """
    return _current_user.get()


def require_current_user() -> CurrentUser:
    """현재 사용자를 반환하고, 없으면 :class:`RuntimeError`를 던진다.

    요청 인증 context 밖에서 호출되면 안 되는 repository 코드가 사용한다. 에러 메시지는
    stack trace를 디버깅하는 호출자가 문제의 코드 경로를 찾을 수 있도록 작성되어 있다.
    """
    user = _current_user.get()
    if user is None:
        raise RuntimeError("repository accessed without user context")
    return user


# ---------------------------------------------------------------------------
# 실효 user_id 헬퍼 (파일시스템 격리)
# ---------------------------------------------------------------------------

DEFAULT_USER_ID: Final[str] = "default"


def get_effective_user_id() -> str:
    """현재 사용자 id를 문자열로 반환하고, 설정되어 있지 않으면 DEFAULT_USER_ID를 반환한다.

    :func:`require_current_user`와 달리 예외를 던지지 않는다. 항상 유효한 사용자 bucket이
    필요한 파일시스템 경로 해석용이다.
    """
    user = _current_user.get()
    if user is None:
        return DEFAULT_USER_ID
    return str(user.id)


def _storage_user_id_from_auth_identity(identity: object | None) -> str | None:
    """LangGraph auth identity에 대해 저장소에 안전한 안정적 ID를 반환한다."""
    if not isinstance(identity, str) or not identity:
        return None

    # LangGraph는 BaseUser.identity에 임의의 문자열(대개 이메일 주소)을 허용하지만,
    # DeerFlow의 사용자 디렉터리는 더 좁은 문자 집합을 요구한다. graph 구성과 runtime
    # middleware가 항상 같은 bucket을 고르도록 정규화를 auth 경계에 둔다.
    from deerflow.config.paths import make_safe_user_id

    return make_safe_user_id(identity)


def _user_id_from_auth_user(user: object | None) -> str | None:
    if isinstance(user, Mapping):
        identity = user.get("identity")
    else:
        identity = getattr(user, "identity", None)
    return _storage_user_id_from_auth_identity(identity)


def _user_id_from_langgraph_config(config: object | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None

    user_id = _storage_user_id_from_auth_identity(configurable.get("langgraph_auth_user_id"))
    if user_id:
        return user_id
    return _user_id_from_auth_user(configurable.get("langgraph_auth_user"))


def resolve_config_user_id(config: object | None) -> str:
    """LangGraph/Gateway run config에서 실효 사용자를 해석한다.

    server가 소유한 LangGraph 인증 필드가 일반 ``user_id`` 값보다 우선한다. Agent Server가
    auth 필드를 예약하고 덮어쓰는 반면, standalone client는 일반 configurable/context 값을
    줄 수 있기 때문이다. 그다음은 DeerFlow embedded run 경로의 Gateway runtime context이고,
    이어서 legacy configurable 채널, 마지막으로 request ContextVar/기본값 fallback이다.
    """
    langgraph_user_id = _user_id_from_langgraph_config(config)
    if langgraph_user_id:
        return langgraph_user_id

    if isinstance(config, Mapping):
        context = config.get("context")
        if isinstance(context, Mapping):
            context_user_id = context.get("user_id")
            if context_user_id:
                return str(context_user_id)

        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            configurable_user_id = configurable.get("user_id")
            if configurable_user_id:
                return str(configurable_user_id)

    return get_effective_user_id()


def resolve_runtime_user_id(runtime: object | None) -> str:
    """tool/middleware의 실효 user_id에 대한 단일 진실 공급원.

    해석 순서(권위가 높은 것부터):
      1. ``runtime.server_info.user.identity`` — 최신 LangGraph runtime이 Agent Server의
         인증된 사용자로부터 채운다. 일반 run context와 달리 server가 소유한다.
      2. ``config["configurable"]["langgraph_auth_user_id"]`` — LangGraph Server가
         배포의 ``@auth.authenticate`` 결과로 채운다. ``server_info``가 없는 구버전
         runtime과 코드 경로를 지원한다.
      3. ``runtime.context["user_id"]`` — gateway의 ``inject_authenticated_user_context``가
         auth 검증된 ``request.state.user``로부터 설정한다. contextvar가 유실될 수 있는
         경계(요청 task 밖에서 예약된 background task, copy_context를 하지 않는 worker
         pool, 향후 cross-process driver)를 넘어서도 살아남는 유일한 출처다.
      4. ``_current_user`` ContextVar — auth middleware가 요청 진입 시 설정한다. task 내
         작업에는 신뢰할 만하며, ``asyncio`` 자식 task와 ``ContextThreadPoolExecutor``가
         복사해 간다.
      5. ``DEFAULT_USER_ID`` — 최후의 fallback. 인증되지 않은 CLI / 마이그레이션 / 테스트
         경로가 예외 없이 계속 동작하게 한다.

    사용자 범위 상태를 저장하는 tool(custom agent, memory, upload)은 ``setup_agent``가 이미
    의존하는 runtime.context 채널의 이점을 누리도록, ``get_effective_user_id()`` 대신 반드시
    이 함수를 호출해야 한다.
    """
    server_info = getattr(runtime, "server_info", None)
    server_user_id = _user_id_from_auth_user(getattr(server_info, "user", None))
    if server_user_id:
        return server_user_id

    langgraph_user_id = _user_id_from_langgraph_auth()
    if langgraph_user_id:
        return langgraph_user_id

    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping):
        ctx_user_id = context.get("user_id")
        if ctx_user_id:
            return str(ctx_user_id)
    return get_effective_user_id()


def _user_id_from_langgraph_auth() -> str | None:
    """가능하면 인증된 LangGraph Server user id를 반환한다.

    LangGraph Server는 ``langgraph_auth_user``와 ``langgraph_auth_user_id`` configurable
    키를 예약해 두고 배포의 ``@auth.authenticate`` 결과로 채운다. ``get_config()``는
    runnable context 밖에서 호출하면 예외를 던지는데, 이는 단지 이 identity 채널을 쓸 수
    없다는 뜻이다.
    """
    try:
        from langgraph.config import get_config

        config = get_config()
    except RuntimeError:
        return None

    return _user_id_from_langgraph_config(config)


# ---------------------------------------------------------------------------
# sentinel 기반 user_id 해석
# ---------------------------------------------------------------------------
#
# repository 메서드는 기본값이 ``AUTO``인 keyword-only ``user_id`` 인자를 받는다.
# 세 가지 값이 각각 다른 동작을 결정한다. :func:`resolve_user_id`의 docstring을 참고한다.


class _AutoSentinel:
    """'user_id를 contextvar에서 해석하라'는 뜻의 singleton 마커."""

    _instance: _AutoSentinel | None = None

    def __new__(cls) -> _AutoSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<AUTO>"


AUTO: Final[_AutoSentinel] = _AutoSentinel()


def resolve_user_id(
    value: str | None | _AutoSentinel,
    *,
    method_name: str = "repository method",
) -> str | None:
    """repository 메서드에 전달된 user_id 파라미터를 해석한다.

    세 가지 상태:

    - :data:`AUTO`(기본값): contextvar에서 읽고, context에 사용자가 없으면
      :class:`RuntimeError`를 던진다. request-scoped 호출의 일반적인 경우다.
    - 명시적 ``str``: 주어진 id를 그대로 사용하며 contextvar 값을 덮어쓴다. 테스트와
      admin override 흐름에 유용하다.
    - 명시적 ``None``: 필터 없음 — repository는 user_id WHERE 절을 아예 생략해야 한다.
      의도적으로 격리를 우회하는 마이그레이션 스크립트와 CLI 도구 전용이다.
    """
    if isinstance(value, _AutoSentinel):
        user = _current_user.get()
        if user is None:
            raise RuntimeError(f"{method_name} called with user_id=AUTO but no user context is set; pass an explicit user_id, set the contextvar via auth middleware, or opt out with user_id=None for migration/CLI paths.")
        # 경계에서 ``str``로 변환한다. ``User.id``는 API 표면에서는 ``UUID``로 타이핑되지만
        # persistence 레이어는 ``user_id``를 ``String(64)``로 저장하고, aiosqlite는 raw UUID
        # 객체를 VARCHAR 컬럼에 바인딩할 수 없다("type 'UUID' is not supported"). 모든
        # 호출자에게 타입 변경을 전파하는 대신 여기서 문서화된 반환 타입을 지킨다.
        return str(user.id)
    return value
