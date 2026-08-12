import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deerflow.config.agents_config import SOUL_FILENAME, validate_agent_name
from deerflow.config.paths import get_paths
from deerflow.persistence.agents import get_agent_store
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)


@tool(parse_docstring=True)
def setup_agent(
    soul: str,
    description: str,
    runtime: Runtime,
    skills: list[str] | None = None,
) -> Command:
    """custom DeerFlow agent를 설정한다.

    Args:
        soul: agent의 성격과 동작을 정의하는 전체 SOUL.md 내용.
        description: 이 agent가 무엇을 하는지에 대한 한 줄 설명.
        skills: 이 agent가 사용할 skill 이름 목록(선택). None이면 활성화된 모든 skill을 사용하고, 빈 목록이면 skill을 쓰지 않는다.
    """

    # 파일시스템에 손대기 전에 비어 있거나 공백뿐인 soul을 거부한다. 이 가드가 없으면 tool이 빈
    # SOUL.md를 그대로 저장하고 성공을 보고해, 쓸 수 없는 agent에 대해 frontend가 "agent
    # created" 상태로 들어갔다(issue #3549). 시끄럽게 실패시켜야 모델이 깨진 artifact를 조용히
    # 만들어내는 대신 재시도한다. 앞단의 agent_name 수정과 함께, 전역 기본 SOUL.md가 빈 내용으로
    # 덮어써지는 것도 막는다.
    if not soul or not soul.strip():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: soul content is empty; refusing to create agent with an empty SOUL.md",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    agent_name: str | None = runtime.context.get("agent_name") if runtime.context else None

    try:
        agent_name = validate_agent_name(agent_name)
        if agent_name:
            # custom agent는 설정된 store(file 또는 db)를 통해 현재 user의 버킷 아래에 저장한다.
            # 그래야 서로 다른 user와 서로 다른 노드가 같은 agent를 찾는다. setup은 멱등하므로
            # 이 동작은 upsert다.
            user_id = resolve_runtime_user_id(runtime)
            config_data: dict = {"name": agent_name}
            if description:
                config_data["description"] = description
            if skills is not None:
                config_data["skills"] = skills
            get_agent_store().update(agent_name, config_data, soul, user_id=user_id)
        else:
            # 기본 agent(agent_name 없음)의 SOUL.md는 전역 base dir에 있다. custom agent
            # 레코드가 아니므로 agent storage backend와 무관하게 파일 기반으로 남는다.
            paths = get_paths()
            paths.base_dir.mkdir(parents=True, exist_ok=True)
            (paths.base_dir / SOUL_FILENAME).write_text(soul, encoding="utf-8")

        logger.info(f"[agent_creator] Created agent '{agent_name}'")
        return Command(
            update={
                "created_agent_name": agent_name,
                "messages": [ToolMessage(content=f"Agent '{agent_name}' created successfully!", tool_call_id=runtime.tool_call_id)],
            }
        )

    except Exception as e:
        logger.error(f"[agent_creator] Failed to create agent '{agent_name}': {e}", exc_info=True)
        return Command(update={"messages": [ToolMessage(content=f"Error: {e}", tool_call_id=runtime.tool_call_id)]})
