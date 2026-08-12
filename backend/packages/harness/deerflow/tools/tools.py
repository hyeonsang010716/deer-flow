import logging

from langchain.tools import BaseTool

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.reflection import resolve_variable
from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.tools.builtins import ask_clarification_tool, list_uploaded_files, present_file_tool, review_skill_package, task_tool, view_image_tool
from deerflow.tools.mcp_metadata import tag_mcp_tool
from deerflow.tools.sync import make_sync_tool_wrapper

logger = logging.getLogger(__name__)

BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    review_skill_package,
]

SUBAGENT_TOOLS = [
    task_tool,
    # task_status_tool은 더 이상 LLM에 노출하지 않는다(backend가 내부적으로 polling한다)
]


def _is_host_bash_tool(tool: object) -> bool:
    """tool config가 host bash 실행 표면을 나타내면 True를 반환한다."""
    group = getattr(tool, "group", None)
    use = getattr(tool, "use", None)
    if group == "bash":
        return True
    if use == "deerflow.sandbox.tools:bash_tool":
        return True
    return False


def _ensure_sync_invocable_tool(tool: BaseTool) -> BaseTool:
    """sync agent 호출자가 쓰는 async 전용 tool에 sync wrapper를 붙인다."""
    if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
        tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)
    return tool


def get_available_tools(
    groups: list[str] | None = None,
    include_mcp: bool = True,
    model_name: str | None = None,
    subagent_enabled: bool = False,
    *,
    include_upload_tool: bool = True,
    app_config: AppConfig | None = None,
) -> list[BaseTool]:
    """config에서 사용 가능한 모든 tool을 가져온다.

    참고: MCP tool은 애플리케이션 시작 시 deerflow.mcp 모듈의 `initialize_mcp_tools()`로
    초기화해야 한다.

    Args:
        groups: 필터링할 tool group 목록(선택).
        include_mcp: MCP server의 tool을 포함할지 여부(기본값: True).
        model_name: vision tool 포함 여부를 판단할 model 이름(선택).
        subagent_enabled: subagent tool(task, task_status)을 포함할지 여부.
        include_upload_tool: ``list_uploaded_files``를 포함할지 여부(기본값: True).
            subagent tool 조립 시에는 False로 준다. subagent는 독립된 ThreadState를 가져
            현재 run의 파일을 제외할 수 없기 때문이다.

    Returns:
        사용 가능한 tool 목록.
    """
    config = app_config or get_app_config()
    tool_configs = [tool for tool in config.tools if groups is None or tool.group in groups]

    # LocalSandboxProvider가 활성일 때는 기본적으로 host bash를 노출하지 않는다.
    if not is_host_bash_allowed(config):
        tool_configs = [tool for tool in tool_configs if not _is_host_bash_tool(tool)]

    loaded_tools_raw = [(cfg, resolve_variable(cfg.use, BaseTool)) for cfg in tool_configs]

    # config의 ``name`` 필드와 tool 객체의 ``.name`` 속성이 어긋나면 경고한다. 이 불일치가
    # issue #1803의 근본 원인이다. LLM은 tool schema에서 한 이름을 받는데 runtime router는
    # 다른 이름을 인식해 "not a valid tool" 오류가 발생한다.
    for cfg, loaded in loaded_tools_raw:
        if cfg.name != loaded.name:
            logger.warning(
                "Tool name mismatch: config name %r does not match tool .name %r (use: %s). The tool's own .name will be used for binding.",
                cfg.name,
                loaded.name,
                cfg.use,
            )

    loaded_tools = [_ensure_sync_invocable_tool(t) for _, t in loaded_tools_raw]

    # config에 따라 조건부로 tool을 추가한다
    builtin_tools = BUILTIN_TOOLS.copy()
    if include_upload_tool:
        builtin_tools.append(list_uploaded_files)
    skill_evolution_config = getattr(config, "skill_evolution", None)
    if getattr(skill_evolution_config, "enabled", False):
        from deerflow.tools.skill_manage_tool import skill_manage_tool

        builtin_tools.append(skill_manage_tool)

    # runtime 파라미터로 활성화된 경우에만 subagent tool을 추가한다
    if subagent_enabled:
        builtin_tools.extend(SUBAGENT_TOOLS)
        logger.info("Including subagent tools (task)")

    # model_name이 없으면 첫 번째 model(기본값)을 쓴다
    if model_name is None and config.models:
        model_name = config.models[0].name

    # model이 vision을 지원할 때만 view_image_tool을 추가한다
    model_config = config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        builtin_tools.append(view_image_tool)
        logger.info(f"Including view_image_tool for model '{model_name}' (supports_vision=True)")

    # 활성화되어 있으면 cache된 MCP tool을 가져온다.
    # NOTE: config.extensions 대신 ExtensionsConfig.from_file()을 써서 항상 디스크의 최신
    # 설정을 읽는다. 그래야 (별도 프로세스에서 도는) Gateway API로 변경한 내용이 MCP tool을
    # 로드할 때 즉시 반영된다.
    mcp_tools = []
    if include_mcp:
        try:
            from deerflow.config.extensions_config import ExtensionsConfig
            from deerflow.mcp.cache import get_cached_mcp_tools

            extensions_config = ExtensionsConfig.from_file()
            if extensions_config.get_enabled_mcp_servers():
                mcp_tools = get_cached_mcp_tools()
                if mcp_tools:
                    logger.info(f"Using {len(mcp_tools)} cached MCP tool(s)")

                    # MCP에서 온 tool에 태그를 달아, 각 agent 생성 지점의 deferred tool 조립이
                    # 이를 식별할 수 있게 한다. lead agent는 설정된 MCP catalog 전체를 조립하고
                    # runtime에 active skill policy를 적용한다. subagent는 skill이 시작 시점에
                    # 로드되므로 이미 policy로 필터링된 목록을 넘길 수 있다.
                    for t in mcp_tools:
                        tag_mcp_tool(t)
        except ImportError:
            logger.warning("MCP module not available. Install 'langchain-mcp-adapters' package to enable MCP tools.")
        except Exception as e:
            logger.error(f"Failed to get cached MCP tools: {e}")

    # ACP agent가 하나라도 설정되어 있으면 invoke_acp_agent tool을 추가한다
    acp_tools: list[BaseTool] = []
    try:
        from deerflow.tools.builtins.invoke_acp_agent_tool import build_invoke_acp_agent_tool

        if app_config is None:
            from deerflow.config.acp_config import get_acp_agents

            acp_agents = get_acp_agents()
        else:
            acp_agents = getattr(config, "acp_agents", {}) or {}
        if acp_agents:
            acp_tools.append(build_invoke_acp_agent_tool(acp_agents))
            logger.info(f"Including invoke_acp_agent tool ({len(acp_agents)} agent(s): {list(acp_agents.keys())})")
    except Exception as e:
        logger.warning(f"Failed to load ACP tool: {e}")

    logger.info(f"Total tools loaded: {len(loaded_tools)}, built-in tools: {len(builtin_tools)}, MCP tools: {len(mcp_tools)}, ACP tools: {len(acp_tools)}")

    # tool 이름으로 중복을 제거한다. config에서 로드한 tool이 우선이고 그다음이 built-in,
    # MCP tool, ACP tool 순이다. 이름이 중복되면 LLM이 모호하거나 이어붙은 function schema를
    # 받게 된다(issue #1803).
    all_tools = [_ensure_sync_invocable_tool(t) for t in loaded_tools + builtin_tools + mcp_tools + acp_tools]
    seen_names: set[str] = set()
    unique_tools: list[BaseTool] = []
    for t in all_tools:
        if t.name not in seen_names:
            unique_tools.append(t)
            seen_names.add(t.name)
        else:
            logger.warning(
                "Duplicate tool name %r detected and skipped — check your config.yaml and MCP server registrations (issue #1803).",
                t.name,
            )
    return unique_tools
