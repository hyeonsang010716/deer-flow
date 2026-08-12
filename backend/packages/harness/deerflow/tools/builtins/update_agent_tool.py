"""update_agent tool — custom agent가 자기 SOUL.md / config 갱신을 영속화하게 한다.

``runtime.context['agent_name']``이 설정된 경우에만(즉 기존 custom agent의 대화 안에서) lead
agent에 바인딩된다. 기본 agent는 이 tool을 보지 못하며, bootstrap 흐름은 최초 생성 handshake에
계속 ``setup_agent``를 쓴다.

이 tool은 설정된 agent store를 통해 기록한다(file: 사용자별 ``config.yaml``/``SOUL.md``,
db: 공유 ``agents`` 테이블). 따라서 한 사용자가 만든 agent는 다른 사용자에게 절대 보이거나
수정되지 않는다.

필드 간 쓰기 원자성은 backend에 따라 다르다. ``db`` store는 config와 soul을 한 트랜잭션으로
commit하므로, 부분 실패로 한쪽만 갱신되는 일이 없다. ``file`` store는 둘 다 임시 파일에 스테이징한
뒤 두 번의 순차 ``os.replace``로 commit한다(``FileAgentStore._write`` 참고). 각 파일은
all-or-nothing이지만, 두 replace *사이*에 크래시가 나면 갓 기록된 config.yaml 옆에 오래된
SOUL.md가 남을 수 있다(단일 노드, 밀리초 미만의 창). store 도입 이전의 tool은 그 부분 실패 구간을
명시적으로 보고했으나("Partial update for agent 'X': ..."), store를 거치면서 그 *보고*가
사라졌다(replace 도중 크래시는 이제 일반적인 "Failed to update agent"로 나타난다). 이는 의도적인
절충이다. 스테이징 후 replace라는 *안전성*은 그대로이고(손상 없음, 임시 파일 잔여물 없음) 진단
정보만 없어졌다. ``file``에서 파일 간 원자성이 중요해지면 ``FileAgentStore._write``에 그 보고를
복원한다.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, BeforeValidator

from deerflow.config.agents_config import load_agent_config, preserve_non_managed_fields, validate_agent_name
from deerflow.config.app_config import get_app_config
from deerflow.config.paths import get_paths
from deerflow.persistence.agents import get_agent_store
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

_NULLISH_STRINGS = frozenset({"null", "none", "undefined"})

# inbound 메시지가 신뢰할 수 없는 외부 작성자(GitHub 저장소의 아무나 등)에게서 오는 channel들.
# lead-agent factory가 이미 이 channel의 run에서는 이 tool을 제외한다
# (``deerflow.agents.lead_agent.agent``의 ``_WEBHOOK_CHANNELS`` 참고). 이 집합은 tool 내부의
# 사본으로, ``update_agent``를 다시 붙이는 custom factory가 webhook을 통한 자기 변경을 조용히
# 열어주지 못하게 한다.
_UNTRUSTED_CHANNELS: frozenset[str] = frozenset({"github"})

_MODEL_BEHAVIOR_FIELDS: tuple[str, ...] = (
    "model_settings",
    "thinking_enabled",
    "reasoning_effort",
)


def _is_nullish_string(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NULLISH_STRINGS


def _normalize_nullish_string(value: object) -> object:
    return None if _is_nullish_string(value) else value


OptionalText = Annotated[str | None, BeforeValidator(_normalize_nullish_string)]
OptionalStringList = Annotated[list[str] | None, BeforeValidator(_normalize_nullish_string)]


@tool(parse_docstring=True)
def update_agent(
    runtime: Runtime,
    soul: OptionalText = None,
    description: OptionalText = None,
    skills: OptionalStringList = None,
    tool_groups: OptionalStringList = None,
    model: OptionalText = None,
) -> Command:
    """현재 custom agent의 SOUL.md와 config.yaml 변경 사항을 저장한다.

    사용자가 agent의 정체성, description, skill whitelist, tool-group whitelist,
    기본 model을 다듬어 달라고 할 때 사용하라. 명시적으로 전달한 필드만 갱신되고,
    생략한 필드는 기존 값을 유지한다.

    ``soul``은 SOUL.md 전체를 대체할 내용으로 전달하라 — patch 방식이 아니므로
    항상 현재 SOUL에서 시작해 편집한 결과를 넘겨야 한다.

    이 agent의 모든 skill을 비활성화하려면 ``skills=[]``를 전달하라. 기존 whitelist를
    유지하려면 ``skills``를 아예 생략하라. 변경하지 않을 필드에 ``"null"`` /
    ``"none"`` / ``"undefined"`` 같은 리터럴 문자열을 전달하지 마라. 그 필드는
    생략하라.

    Args:
        soul: 선택적. SOUL.md 전체를 대체할 내용.
        description: 선택적. 새 한 줄 description.
        skills: 선택적 skill whitelist. ``[]`` = skill 없음, 생략 = 변경 없음.
        tool_groups: 선택적 tool-group whitelist. ``[]`` = 비움, 생략 = 변경 없음.
        model: 선택적 model 오버라이드(설정된 model 이름과 일치해야 한다).

    Returns:
        결과를 설명하는 ToolMessage가 담긴 Command. 변경 사항은 다음 사용자 턴에
        (새 SOUL.md와 config.yaml로 lead agent가 다시 빌드될 때) 적용된다.
    """
    tool_call_id = runtime.tool_call_id
    agent_name_raw: str | None = runtime.context.get("agent_name") if runtime.context else None
    channel_name: str | None = runtime.context.get("channel_name") if runtime.context else None

    def _err(message: str) -> Command:
        return Command(update={"messages": [ToolMessage(content=f"Error: {message}", tool_call_id=tool_call_id, status="error")]})

    # 심층 방어 — lead-agent factory가 이미 webhook channel run에서는 이 tool을 제공하지
    # 않는다(``deerflow.agents.lead_agent.agent``의 ``_WEBHOOK_CHANNELS`` 참고). 같은 channel
    # 집합을 여기서도 확인해, ``_make_lead_agent``를 거치지 않고 tool을 다시 붙이는 향후 경로
    # (custom factory, 테스트 등)가 webhook에서 온 신뢰할 수 없는 자기 변경 요청을 조용히
    # 받아들이지 않게 한다.
    if channel_name in _UNTRUSTED_CHANNELS:
        return _err(f"update_agent is disabled on the {channel_name!r} channel. Self-mutation requests must come from an operator-trusted surface (chat UI or the HTTP API), not a webhook fan-out.")

    if soul is None and description is None and skills is None and tool_groups is None and model is None:
        return _err('No fields provided. Pass at least one of: soul, description, skills, tool_groups, model. Omit unchanged fields instead of passing null-like strings such as "null", "none", or "undefined".')

    # 파일시스템을 건드리기 전에 비어 있거나 공백뿐인 soul을 거부한다. setup_agent도 이미
    # 거부한다(#3553 / #3549). update_agent도 그래야 하며, 그러지 않으면 custom agent가 멀쩡한
    # SOUL.md를 지우고 다음 turn을 빈 personality로 남기면서도 성공을 보고할 수 있다.
    if soul is not None and not soul.strip():
        return _err("soul content is empty; refusing to update agent with an empty SOUL.md. Omit the soul field if you do not want to change it.")

    try:
        agent_name = validate_agent_name(agent_name_raw)
    except ValueError as e:
        return _err(str(e))

    if not agent_name:
        return _err("update_agent is only available inside a custom agent's chat. There is no agent_name in the current runtime context, so there is nothing to update. If you are inside the bootstrap flow, use setup_agent instead.")

    # 갱신이 이 사용자의 agent에만 영향을 주도록 활성 사용자를 확정한다.
    # ``resolve_runtime_user_id``는 ``runtime.context["user_id"]``(auth 검증된 요청에서
    # gateway가 설정)를 우선하고, 없으면 contextvar, 그다음 DEFAULT_USER_ID로 내려간다.
    # setup_agent와 동일하므로, agent를 만들고 나중에 다듬는 사용자는 async/thread 경계에서
    # contextvar가 유실되더라도(issue #2782 / #2862 부류의 버그) 항상 같은 파일을 건드린다.
    user_id = resolve_runtime_user_id(runtime)

    # 파일시스템을 건드리기 *전에* 알 수 없는 ``model``을 거부한다. 그러지 않으면
    # ``_resolve_model_name``이 런타임에 조용히 기본값으로 되돌아가고, 사용자는 이후 모든
    # turn에서 혼란스러운 경고를 반복해서 보게 된다.
    if model is not None and get_app_config().get_model_config(model) is None:
        return _err(f"Unknown model '{model}'. Pass a model name that exists in config.yaml's models section.")

    paths = get_paths()
    agent_dir = paths.user_agent_dir(user_id, agent_name)
    legacy_dir = paths.agent_dir(agent_name)
    # 디렉터리 존재 여부가 아니라 config.yaml을 요구한다. 사용자별 agent 디렉터리에 memory.json만
    # 들어 있을 수 있기 때문이다(이 사용자가 legacy 공유 agent와 처음 대화할 때, update_agent가
    # 호출되기도 전에 기록된다). 단순 .exists()는 그 경우를 놓치고 load_agent_config로 흘려보내며,
    # 그러면 load_agent_config는 resolve_agent_dir를 통해 legacy 공유 config로 올바르게
    # 해석하면서도, 차단하는 대신 memory.json만 있는 디렉터리에 새 config.yaml/SOUL.md를 조용히
    # 분기시킨다(resolve_agent_dir의 가드와 동일, #3390 참고).
    if not (agent_dir / "config.yaml").exists() and (legacy_dir / "config.yaml").exists():
        return _err(f"Agent '{agent_name}' only exists in the legacy shared layout and is not scoped to a user. Run scripts/migrate_user_isolation.py to move legacy agents into the per-user layout before updating.")

    try:
        existing_cfg = load_agent_config(agent_name, user_id=user_id)
    except FileNotFoundError:
        return _err(f"Agent '{agent_name}' does not exist for the current user. Use setup_agent to create a new agent first.")
    except ValueError as e:
        return _err(f"Agent '{agent_name}' has an unreadable config: {e}")

    if existing_cfg is None:
        return _err(f"Agent '{agent_name}' could not be loaded.")

    updated_fields: list[str] = []

    # ``existing_cfg.name``이 어긋나 있더라도(예: 수동 yaml 편집) 디스크의 ``name``을 지금
    # 기록하는 디렉터리와 강제로 일치시킨다.
    config_data: dict[str, Any] = {"name": agent_name}
    new_description = description if description is not None else existing_cfg.description
    config_data["description"] = new_description
    if description is not None and description != existing_cfg.description:
        updated_fields.append("description")

    new_model = model if model is not None else existing_cfg.model
    if new_model is not None:
        config_data["model"] = new_model
    if model is not None and model != existing_cfg.model:
        updated_fields.append("model")

    new_tool_groups = tool_groups if tool_groups is not None else existing_cfg.tool_groups
    if new_tool_groups is not None:
        config_data["tool_groups"] = new_tool_groups
    if tool_groups is not None and tool_groups != existing_cfg.tool_groups:
        updated_fields.append("tool_groups")

    new_skills = skills if skills is not None else existing_cfg.skills
    if new_skills is not None:
        config_data["skills"] = new_skills
    if skills is not None and skills != existing_cfg.skills:
        updated_fields.append("skills")

    # 이 tool은 #4336의 model-behavior 필드를 아직 LLM 호출 인자로 노출하지 않지만, 지원하는
    # 필드가 바뀌면 여전히 config.yaml을 다시 쓴다. 그 값들을 명시적으로 이어받아,
    # description/model/skills를 다듬는 agent가 temperature나 reasoning effort 같은 UI/API
    # 소유 기본값을 지우지 못하게 한다.
    for key in _MODEL_BEHAVIOR_FIELDS:
        value = getattr(existing_cfg, key, None)
        if value is None:
            continue
        if isinstance(value, BaseModel):
            dumped = value.model_dump(exclude_none=True)
            if dumped:
                config_data[key] = dumped
        else:
            config_data[key] = value

    # 이 tool이 인자로 노출하지 않는 최상위 AgentConfig 필드를 전부 보존한다(현재는
    # ``github:``와 앞으로 :class:`AgentConfig`에 추가될 모든 필드). HTTP
    # ``PATCH /api/agents/{name}`` route도 같은 헬퍼를 쓰므로 두 표면이 함께 움직인다. 이것이
    # 없으면 custom agent에 ``github:`` 블록을 직접 작성한 운영자는 agent가 ``update_agent``로
    # 자기 갱신하는 순간 그 설정을 조용히 잃는다.
    preserved = preserve_non_managed_fields(existing_cfg)
    for key, value in preserved.items():
        config_data.setdefault(key, value)

    config_changed = bool({"description", "model", "tool_groups", "skills"} & set(updated_fields))
    if soul is not None:
        updated_fields.append("soul")

    # 관리 대상 필드가 바뀌었으면 config를, soul이 주어졌으면 soul을 store를 통해 영속화한다.
    # db backend는 둘을 한 트랜잭션으로 commit하고, file backend는 각각 원자적이되 순차적으로
    # commit한다(이 모듈의 docstring 참고). 주어진 값이 모두 기존 config와 같고 soul도 없으면
    # 기록할 것이 없다.
    if config_changed or soul is not None:
        try:
            get_agent_store().update(agent_name, config_data if config_changed else None, soul, user_id=user_id)
        except Exception as e:
            logger.error("[update_agent] Failed to update agent '%s' (user=%s): %s", agent_name, user_id, e, exc_info=True)
            return _err(f"Failed to update agent '{agent_name}': {e}")

    if not updated_fields:
        return Command(update={"messages": [ToolMessage(content=f"No changes applied to agent '{agent_name}'. The provided values matched the existing config.", tool_call_id=tool_call_id)]})

    logger.info("[update_agent] Updated agent '%s' (user=%s) fields: %s", agent_name, user_id, updated_fields)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(f"Agent '{agent_name}' updated successfully. Changed: {', '.join(updated_fields)}. The new configuration takes effect on the next user turn."),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )
