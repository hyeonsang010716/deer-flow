"""사용 가능한 subagent를 관리하는 subagent registry."""

import logging
from dataclasses import replace
from typing import Any

from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


def _resolve_subagents_app_config(app_config: Any | None = None):
    if app_config is None:
        from deerflow.config.subagents_config import get_subagents_app_config

        return get_subagents_app_config()
    return getattr(app_config, "subagents", app_config)


def _build_custom_subagent_config(name: str, *, app_config: Any | None = None) -> SubagentConfig | None:
    """config.yaml의 custom_agents 섹션에서 SubagentConfig를 만든다.

    Args:
        name: custom subagent 이름.
        app_config: 값을 해석할 AppConfig 또는 SubagentsAppConfig(선택).

    Returns:
        custom_agents에 있으면 SubagentConfig, 없으면 None.
    """
    subagents_config = _resolve_subagents_app_config(app_config)
    custom = subagents_config.custom_agents.get(name)
    if custom is None:
        return None

    return SubagentConfig(
        name=name,
        description=custom.description,
        system_prompt=custom.system_prompt,
        tools=custom.tools,
        disallowed_tools=custom.disallowed_tools,
        skills=custom.skills,
        model=custom.model,
        max_turns=custom.max_turns,
        timeout_seconds=custom.timeout_seconds,
    )


def get_subagent_config(name: str, *, app_config: Any | None = None) -> SubagentConfig | None:
    """이름으로 subagent 설정을 가져오고 config.yaml override를 적용한다.

    해석 순서(Codex의 config 계층과 동일):
    1. 내장 subagent(general-purpose, bash)
    2. config.yaml custom_agents 섹션의 custom subagent
    3. config.yaml agents 섹션의 agent별 override(timeout, max_turns, model, skills)

    Args:
        name: subagent 이름.
        app_config: override를 해석할 AppConfig 또는 SubagentsAppConfig(선택).

    Returns:
        찾으면 (config.yaml override가 적용된) SubagentConfig, 없으면 None.
    """
    # 1단계: 내장을 찾고, 없으면 custom_agents로 fallback한다
    config = BUILTIN_SUBAGENTS.get(name)
    if config is None:
        config = _build_custom_subagent_config(name, app_config=app_config)
    if config is None:
        return None

    # 2단계: config.yaml agents 섹션의 agent별 override를 적용한다.
    # 여기서는 명시적인 agent별 override만 적용한다. 전역 기본값(최상위 timeout_seconds,
    # max_turns)은 내장 agent에만 적용되며 custom agent 자신의 값을 덮어써서는 안 된다.
    # custom agent는 custom_agents 섹션에서 자체 기본값을 정의하기 때문이다.
    subagents_config = _resolve_subagents_app_config(app_config)
    is_builtin = name in BUILTIN_SUBAGENTS
    agent_override = subagents_config.agents.get(name)

    overrides = {}

    # Timeout: agent별 override > 전역 기본값(내장 전용) > config 자체 값 순
    if agent_override is not None and agent_override.timeout_seconds is not None:
        if agent_override.timeout_seconds != config.timeout_seconds:
            logger.debug("Subagent '%s': timeout overridden (%ss -> %ss)", name, config.timeout_seconds, agent_override.timeout_seconds)
            overrides["timeout_seconds"] = agent_override.timeout_seconds
    elif is_builtin and subagents_config.timeout_seconds != config.timeout_seconds:
        logger.debug("Subagent '%s': timeout from global default (%ss -> %ss)", name, config.timeout_seconds, subagents_config.timeout_seconds)
        overrides["timeout_seconds"] = subagents_config.timeout_seconds

    # Max turns: agent별 override > 전역 기본값(내장 전용) > config 자체 값 순
    if agent_override is not None and agent_override.max_turns is not None:
        if agent_override.max_turns != config.max_turns:
            logger.debug("Subagent '%s': max_turns overridden (%s -> %s)", name, config.max_turns, agent_override.max_turns)
            overrides["max_turns"] = agent_override.max_turns
    elif is_builtin and subagents_config.max_turns is not None and subagents_config.max_turns != config.max_turns:
        logger.debug("Subagent '%s': max_turns from global default (%s -> %s)", name, config.max_turns, subagents_config.max_turns)
        overrides["max_turns"] = subagents_config.max_turns

    # Model: agent별 override만 적용한다(model에는 전역 기본값이 없다)
    effective_model = subagents_config.get_model_for(name)
    if effective_model is not None and effective_model != config.model:
        logger.debug("Subagent '%s': model overridden (%s -> %s)", name, config.model, effective_model)
        overrides["model"] = effective_model

    # Skills: agent별 override만 적용한다(skills에는 전역 기본값이 없다)
    effective_skills = subagents_config.get_skills_for(name)
    if effective_skills is not None and effective_skills != config.skills:
        logger.debug("Subagent '%s': skills overridden (%s -> %s)", name, config.skills, effective_skills)
        overrides["skills"] = effective_skills

    if overrides:
        config = replace(config, **overrides)

    return config


def list_subagents(*, app_config: Any | None = None) -> list[SubagentConfig]:
    """사용 가능한 모든 subagent 설정을 나열한다(config.yaml override 적용).

    Returns:
        등록된 모든 SubagentConfig 인스턴스 목록(내장 + custom).
    """
    configs = []
    for name in get_subagent_names(app_config=app_config):
        config = get_subagent_config(name, app_config=app_config)
        if config is not None:
            configs.append(config)
    return configs


def get_subagent_names(*, app_config: Any | None = None) -> list[str]:
    """사용 가능한 모든 subagent 이름을 가져온다(내장 + custom).

    Returns:
        subagent 이름 목록.
    """
    names = list(BUILTIN_SUBAGENTS.keys())

    # config.yaml의 custom_agents를 병합한다
    subagents_config = _resolve_subagents_app_config(app_config)
    for custom_name in subagents_config.custom_agents:
        if custom_name not in names:
            names.append(custom_name)

    return names


def get_available_subagent_names(*, app_config: Any | None = None) -> list[str]:
    """현재 runtime에 노출해야 할 subagent 이름을 가져온다.

    Returns:
        현재 sandbox 설정에서 보이는 subagent 이름 목록.
    """
    names = get_subagent_names(app_config=app_config)
    try:
        host_bash_allowed = is_host_bash_allowed(app_config) if hasattr(app_config, "sandbox") else is_host_bash_allowed()
    except Exception:
        logger.debug("Could not determine host bash availability; exposing all subagents")
        return names

    if not host_bash_allowed:
        names = [name for name in names if name != "bash"]
    return names
