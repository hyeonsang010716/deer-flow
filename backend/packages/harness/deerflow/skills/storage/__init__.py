"""SkillStorage singleton과 reflection 기반 factory.

``deerflow/sandbox/sandbox_provider.py``와 같은 패턴을 따른다.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from deerflow.skills.storage.local_skill_storage import LocalSkillStorage
from deerflow.skills.storage.skill_storage import SkillStorage
from deerflow.skills.storage.user_scoped_skill_storage import UserScopedSkillStorage
from deerflow.skills.types import SkillCategory

logger = logging.getLogger(__name__)

_default_skill_storage: SkillStorage | None = None
_default_skill_storage_config: object | None = None  # singleton을 만들 때 쓴 AppConfig identity
_skill_storage_lock = threading.Lock()

# cache에 유지할 사용자별 storage 인스턴스의 최대 개수.
# 실제 배포에서 프로세스당 동시 사용자가 몇 명을 넘는 일은 드물다. 64는 메모리가 무한히
# 늘어나는 것을 막으면서도 넉넉한 상한이다.
_MAX_USER_SCOPED_STORAGES = 64

# 동시 생성에 대비해 double-check lock을 쓰는 사용자별 skill storage cache.
# cache가 ``_MAX_USER_SCOPED_STORAGES``를 넘을 때 ``move_to_end`` +
# ``popitem(last=False)``로 가장 오래 안 쓰인 항목을 LRU eviction할 수 있도록
# OrderedDict를 쓴다.
_user_scoped_storages: OrderedDict[str, UserScopedSkillStorage] = OrderedDict()
_user_scoped_storage_lock = threading.Lock()


def get_or_new_skill_storage(**kwargs) -> SkillStorage:
    """``SkillStorage`` 인스턴스를 반환한다 — 새 인스턴스이거나 프로세스 singleton이다.

    다음 경우 **새 인스턴스**를 만든다(캐시하지 않는다):
    - ``skills_path``가 주어진 경우 — 이를 ``host_path`` override로 쓴다(클래스는 여전히
      config로 해석한다).
    - ``app_config``가 주어진 경우 — ``app_config.skills``로 storage를 만들어서, 요청별
      config(예: Gateway ``Depends(get_config)``)를 프로세스 수준 singleton을 오염시키지
      않고 반영한다.

    ``skills_path``와 ``app_config`` 둘 다 없으면 **singleton**을 반환한다(첫 호출에
    생성한 뒤 재사용). 활성 설정은 ``get_app_config()``로 해석한다.

    이 singleton은 **public** skill(전역, 읽기 전용)을 읽는 데 쓴다. 사용자 범위의 custom
    skill 작업에는 :func:`get_or_new_user_skill_storage`를 쓴다.
    """
    global _default_skill_storage, _default_skill_storage_config

    from deerflow.config import get_app_config
    from deerflow.config.skills_config import SkillsConfig

    def _make_storage(skills_config: SkillsConfig, *, host_path: str | None = None, **kwargs) -> SkillStorage:
        from deerflow.reflection import resolve_class

        cls = resolve_class(skills_config.use, SkillStorage)
        return cls(
            host_path=host_path if host_path is not None else str(skills_config.get_skills_path()),
            container_path=skills_config.container_path,
            **kwargs,
        )

    skills_path = kwargs.pop("skills_path", None)
    app_config = kwargs.pop("app_config", None)

    if skills_path is not None:
        if app_config is not None:
            return _make_storage(app_config.skills, host_path=str(skills_path), **kwargs)
        # app_config이 없다: 호출자가 이미 명시적 host path를 준 상황이므로 config.yaml을
        # 읽을 필요가 없도록 기본 SkillsConfig를 쓴다.
        from deerflow.config.skills_config import SkillsConfig

        return _make_storage(SkillsConfig(), host_path=str(skills_path), **kwargs)

    if app_config is not None:
        return _make_storage(app_config.skills, **kwargs)

    # singleton이 config identity 없이 수동 주입된 경우(예: 테스트,
    # _default_skill_storage_config가 None), 디스크의 config.yaml을 요구하지 않도록
    # get_app_config()를 아예 건너뛴다.
    if _default_skill_storage is not None and _default_skill_storage_config is None:
        return _default_skill_storage

    app_config_now = get_app_config()

    # 콜드 스타트가 경쟁해도 인스턴스가 정확히 하나만 만들어지고, reset_skill_storage()가
    # 동시 읽기 중인 전역 변수를 None으로 만들지 못하도록 lock 안에서 double-check로
    # singleton을 만든다. sandbox_provider처럼 lock 밖에서 만들고 진 쪽을 버리는 대신
    # get_memory_storage()와 같이 lock *안에서* 생성한다 — SkillStorage에는 teardown
    # hook이 없어서 경쟁에서 진 쪽의 고아 인스턴스를 정리할 수 없기 때문이다.
    with _skill_storage_lock:
        if _default_skill_storage is None or _default_skill_storage_config is not app_config_now:
            _default_skill_storage = _make_storage(app_config_now.skills, **kwargs)
            _default_skill_storage_config = app_config_now
        return _default_skill_storage


def get_or_new_user_skill_storage(user_id: str, **kwargs) -> SkillStorage:
    """custom skill 격리를 위한 사용자별 ``SkillStorage`` 인스턴스를 반환한다.

    :class:`UserScopedSkillStorage`를 쓴다. 이 클래스는 custom skill 경로를
    ``{base_dir}/users/{user_id}/skills/custom/``로 돌리면서 public skill 읽기는 전역
    루트에서 유지한다.

    ``user_id``는 :func:`make_safe_user_id`로 정규화한다. 그래야 외부 identity(예:
    ``[A-Za-z0-9_-]`` 이외 문자를 포함하는 IM channel id)가 내부적으로
    :func:`_validate_user_id`를 호출하는 :class:`UserScopedSkillStorage`에 닿기 전에
    안전한 버킷으로 들어간다.

    인스턴스는 *정규화된* ``user_id``로 캐시하며, 동시 생성 경쟁을 막기 위해
    double-check locking을 쓴다. cache가 ``_MAX_USER_SCOPED_STORAGES``를 넘으면 가장
    오래 접근하지 않은 항목을 제거한다(FIFO가 아니라 진짜 LRU).
    """
    from deerflow.config.paths import make_safe_user_id

    safe_id = make_safe_user_id(user_id)

    # move_to_end가 안전하도록 항상 lock을 잡는다 — 그래야 FIFO가 아닌 진짜 LRU cache가
    # 된다. dict 연산은 빠르고 이 함수는 agent 생성 주기마다 한 번만 호출되므로 오버헤드는
    # 무시할 수준이다.
    with _user_scoped_storage_lock:
        cached = _user_scoped_storages.get(safe_id)
        if cached is not None:
            _user_scoped_storages.move_to_end(safe_id)
            return cached

        cached = UserScopedSkillStorage(safe_id, **kwargs)
        _user_scoped_storages[safe_id] = cached
        # cache가 상한을 넘으면 가장 오래 안 쓰인 항목을 제거한다.
        # 방금 현재 user_id를 끝으로 옮겼으므로 popitem(last=False)는 가장 오래되고 가장
        # 오래 접근하지 않은 항목을 제거하며, 방금 만든 항목은 절대 제거하지 않는다.
        while len(_user_scoped_storages) > _MAX_USER_SCOPED_STORAGES:
            evicted_key, evicted_val = _user_scoped_storages.popitem(last=False)
            logger.info("Evicted user-scoped skill storage for safe_id=%s (cache ceiling %d)", evicted_key, _MAX_USER_SCOPED_STORAGES)
        return cached


def user_should_see_legacy_skills(user_id: str, **kwargs) -> bool:
    """이 사용자에게 discovery가 LEGACY skill을 하나라도 노출하는지 반환한다.

    sandbox mount는 skill discovery보다 더 허용적이어서는 안 된다. 이 헬퍼가 그 계약을
    한곳에 모아 local, AIO, remote provider가 모두 같은 가시성 규칙을 따르게 한다.
    """
    if kwargs:
        from deerflow.config.paths import make_safe_user_id

        storage = UserScopedSkillStorage(make_safe_user_id(user_id), **kwargs)
    else:
        storage = get_or_new_user_skill_storage(user_id)
    return any((skill.category.value if hasattr(skill.category, "value") else skill.category) == SkillCategory.LEGACY.value for skill in storage.load_skills(enabled_only=False))


def reset_skill_storage() -> None:
    """캐시된 storage 인스턴스를 전부 비운다(테스트와 hot-reload 시나리오에서 쓴다)."""
    global _default_skill_storage, _default_skill_storage_config
    with _skill_storage_lock:
        _default_skill_storage = None
        _default_skill_storage_config = None
    with _user_scoped_storage_lock:
        _user_scoped_storages.clear()


def reset_user_skill_storage(user_id: str | None = None) -> None:
    """특정 사용자 또는 전체 사용자의 사용자별 skill storage cache를 비운다.

    ``user_id``는 :func:`make_safe_user_id`로 정규화해서 cache 키가
    :func:`get_or_new_user_skill_storage`가 쓰는 키와 일치하게 한다. 정규화하지 않으면
    IM channel 사용자 ID(예: ``feishu:xxx``)의 낡은 cache 항목을 비우지 못한다.

    Args:
        user_id: 주어지면 그 사용자의 캐시된 storage만 제거한다.
            ``None``이면 사용자별 cache 전체를 비운다.
    """
    from deerflow.config.paths import make_safe_user_id

    with _user_scoped_storage_lock:
        if user_id is not None:
            safe_id = make_safe_user_id(user_id)
            _user_scoped_storages.pop(safe_id, None)
        else:
            _user_scoped_storages.clear()


__all__ = [
    "LocalSkillStorage",
    "SkillStorage",
    "UserScopedSkillStorage",
    "get_or_new_skill_storage",
    "get_or_new_user_skill_storage",
    "user_should_see_legacy_skills",
    "reset_skill_storage",
    "reset_user_skill_storage",
]
