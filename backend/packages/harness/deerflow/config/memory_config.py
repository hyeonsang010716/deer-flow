"""memory 메커니즘 설정(host 공용 필드만).

DeerMem 전용 필드는 ``backends/deermem/config.py``(``DeerMemConfig``)에 있고,
factory가 backend의 ``__init__``에 넘기는 dict인 ``backend_config``를 통해 전달된다.
이 모듈은 모든 backend/호출부/factory가 읽는 host 공용 필드만 담는다:
``enabled`` / ``injection_enabled`` / ``shutdown_flush_timeout_seconds`` /
``manager_class`` / ``backend_config``.
공용 스키마를 얇게 유지해야 backend를 갈아끼울 수 있다(DeerMem의 설정값이 공용 계약으로
새어나오지 않는다).
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 모든 backend/호출부/factory가 읽는 host 공용 MemoryConfig 필드.
_SHARED_FIELDS = frozenset({"enabled", "mode", "injection_enabled", "shutdown_flush_timeout_seconds", "manager_class", "backend_config"})

# 추상화 이전에 config.yaml의 `memory:` 최상위에 있던 DeerMem 전용 필드들. 로드 시
# `backend_config`로 자동 이관되므로, 업그레이드가 사용자 설정을 조용히 기본값으로
# 되돌리지 않는다. `model_name`은 새로 중첩된 model 하위 설정인
# `backend_config.model.model`로 매핑되고, 나머지는 1:1이다.
_LEGACY_DEERMEM_FIELDS = frozenset(
    {
        "storage_path",
        "storage_class",
        "debounce_seconds",
        "max_facts",
        "fact_confidence_threshold",
        "max_injection_tokens",
        "token_counting",
        "guaranteed_categories",
        "guaranteed_token_budget",
        "staleness_review_enabled",
        "staleness_age_days",
        "staleness_min_candidates",
        "staleness_max_removals_per_cycle",
        "staleness_protected_categories",
        "staleness_max_lifetime_multiplier",
        "staleness_max_extension_days",
        "consolidation_enabled",
        "consolidation_min_facts",
        "consolidation_max_groups_per_cycle",
        "consolidation_max_sources",
        "model_name",
    }
)


class MemoryConfig(BaseModel):
    """backend에 종속되지 않는 host 공용 memory 설정."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable the memory mechanism (call-site gate).",
    )
    mode: Literal["middleware", "tool"] = Field(
        default="middleware",
        description=(
            "Memory operation mode. 'middleware': passive LLM summarization after each turn (current behavior). 'tool': model calls memory tools (memory_search, memory_add, etc.) directly. Mutually exclusive — only one mode runs at a time."
        ),
    )
    injection_enabled: bool = Field(
        default=True,
        description="Whether to inject memory into the system prompt (call-site gate).",
    )
    shutdown_flush_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description=(
            "Hard time budget (seconds) for draining the memory backend's "
            "pending-update buffer during Gateway graceful shutdown. The drain "
            "makes one LLM call per pending item, so large IM batches may need "
            "a higher value. Must fit inside the pod's K8s "
            "terminationGracePeriodSeconds (together with channel/scheduler "
            "stop) or K8s SIGKILLs the drain mid-flight. The drain runs on a "
            "daemon thread, so on timeout the process proceeds to exit and any "
            "unfinished tail is dropped (same failure direction as no flush, "
            "scoped to the tail). Host-shared (not backend-private): the host "
            "owns the lifespan budget and the K8s grace relationship."
        ),
    )
    manager_class: str = Field(
        default="deermem",
        description=(
            "Memory backend selector. Either a registered backend name "
            "(matching a `backends/<name>/` folder that exposes `MANAGER_CLASS`, "
            "e.g. `deermem` / `noop`) or a dotted import path to a "
            "`MemoryManager` subclass. The factory resolves this at "
            "`get_memory_manager()` time and raises `ValueError` on failure "
            "(fail-fast: memory is persistent state, so an unresolved "
            "manager_class is not silently substituted with a different "
            "storage backend)."
        ),
    )
    backend_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-private config (a dict), passed verbatim to the backend's "
            "`__init__(backend_config=...)` by the factory. Each backend "
            "self-interprets it (DeerMem parses it into `DeerMemConfig`). Values "
            "live in the host config file (`config.yaml` `memory.backend_config`); "
            "they do not belong on the shared `MemoryConfig` schema."
        ),
    )


def should_use_memory_tools(config: MemoryConfig) -> bool:
    """memory가 모델 주도 도구를 써야 하면 True를 반환한다."""
    return config.enabled and config.mode == "tool"


# 전역 설정 인스턴스
_memory_config: MemoryConfig = MemoryConfig()


def get_memory_config() -> MemoryConfig:
    """현재 memory 설정을 반환한다.

    ``_memory_config``는 ``get_app_config()`` reload의 부수 효과로만 갱신된다
    (``_apply_singleton_configs`` -> ``load_memory_config_from_dict``). ``get_app_config()``를
    거치지 않고 memory 설정을 읽는 쪽(예: memory 도구를 bind할지 결정하는 agent factory)은
    ``memory.*``가 hot-reload 대상이라고 문서화돼 있음에도 ``config.yaml`` 수정 후 낡은
    ``memory.mode``를 보게 된다. 그래서 여기서 동일한 signature 기반 reload를 트리거해
    singleton이 설정 파일을 따라가게 한다.

    ``get_app_config()``가 한 번도 호출된 적 없으면(``_app_config``가 ``None``) 갱신할
    낡은 설정도 없으므로 기존 동작대로 메모리 상의 singleton을 그대로 반환한다. 이렇게
    해야 첫 ``get_memory_config()`` 호출이 부수적으로 설정 파일을 읽어들여 모듈 기본값을
    기대하는 호출자(예: 단위 테스트)를 깨뜨리지 않는다.
    """
    # lazy import: app_config가 이 모듈을 import하므로 최상위 import는 순환이 된다.
    from .app_config import _app_config, get_app_config

    if _app_config is not None:
        try:
            get_app_config()
        except Exception:
            # 설정 파일이 일시적으로 깨졌으면(잘못된 YAML, 스키마 위반, 누락된 환경변수 등)
            # 마지막 정상 singleton을 유지해서 진행 중인 turn이 죽지 않고 끝나게 한다.
            logger.warning(
                "Failed to reload app config from get_memory_config(); falling back to cached memory config.",
                exc_info=True,
            )
    return _memory_config


def set_memory_config(config: MemoryConfig) -> None:
    """memory 설정을 지정한다."""
    global _memory_config
    _memory_config = config


def load_memory_config_from_dict(config_dict: dict) -> None:
    """dict에서 memory 설정을 읽는다.

    host 공용 필드(``enabled`` / ``mode`` / ``injection_enabled`` / ``manager_class`` /
    ``backend_config``)는 그대로 읽는다. 추상화 이전에 config.yaml의 ``memory:`` 최상위에
    있던 DeerMem 전용 필드(``storage_path``, ``max_facts``, ``debounce_seconds``,
    ``model_name``, ``token_counting``, ``staleness_*``, ``consolidation_*`` 등)는 경고와
    함께 **``backend_config``로 자동 이관**한다. 그래야 추상화 이전 설정에서 업그레이드할
    때 사용자 설정이 조용히 기본값으로 되돌아가지 않는다. 알 수 없는 최상위 키(대개 오타)는
    경고 후 무시한다.
    """
    global _memory_config
    config_dict = dict(config_dict or {})
    backend_config = dict(config_dict.get("backend_config") or {})
    migrated: list[str] = []
    for key in list(config_dict.keys()):
        if key in _SHARED_FIELDS:
            continue
        if key in _LEGACY_DEERMEM_FIELDS:
            value = config_dict.pop(key)
            if value is None or value == "":
                continue  # 기본값이거나 빈 값이라 이관할 필요가 없다.
            if key == "model_name":
                # 예전 최상위 model_name -> backend_config.model.model
                model_cfg = dict(backend_config.get("model") or {})
                if "model" not in model_cfg:
                    model_cfg["model"] = value
                    backend_config["model"] = model_cfg
                    migrated.append(f"{key} -> backend_config.model.model")
            elif key == "storage_path" and str(value).endswith(".json"):
                # 추상화 이전의 storage_path는 파일 경로였다(절대 경로는 per-user를
                # 포기하고 공유 파일을 쓰는 의미였고, 예전 기본값 "memory.json" 같은
                # 상대 경로는 per-user에서 무시됐다). DeerMem은 이제 이를 루트 디렉터리로
                # 취급한다. 파일 형태 값을 그대로 넘기면 디렉터리로 해석돼 per-user memory가
                # 고아가 되거나 저장 시 NotADirectoryError가 난다. 그래서 값을 버리고
                # factory의 무설정 runtime_home이 동작하게 하며(per-user 위치는 그대로
                # {base_dir}/users/{uid}/memory.json) 운영자에게 경고한다.
                logger.warning(
                    "Legacy memory.storage_path=%r looks like a file path; DeerMem now "
                    "treats storage_path as a root DIRECTORY (per-user memory under "
                    "{storage_path}/users/{uid}/memory.json). Dropped -- memory now "
                    "lands under the default root (runtime_home). Set "
                    "memory.backend_config.storage_path to a directory if you want a "
                    "custom location.",
                    value,
                )
            elif key not in backend_config:
                # 명시적으로 지정된 backend_config 값은 덮어쓰지 않는다.
                backend_config[key] = value
                migrated.append(f"{key} -> backend_config.{key}")
        else:
            logger.warning(
                "Unknown memory config key %r at top level (not a shared field %s nor a known legacy DeerMem field); ignored.",
                key,
                sorted(_SHARED_FIELDS),
            )
    if migrated:
        logger.warning(
            "Migrated legacy top-level memory fields into backend_config; move them under memory.backend_config in config.yaml to silence this: %s",
            ", ".join(migrated),
        )
    config_dict["backend_config"] = backend_config
    _memory_config = MemoryConfig(**config_dict)
