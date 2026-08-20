"""DeerFlowClient — DeerFlow agent 시스템의 임베디드 Python 클라이언트.

LangGraph Server나 Gateway API 프로세스 없이 DeerFlow agent 기능에 직접 접근한다.

Usage:
    from deerflow.client import DeerFlowClient

    client = DeerFlowClient()
    response = client.chat("Analyze this paper for me", thread_id="my-thread")
    print(response)

    # Streaming
    for event in client.stream("hello"):
        print(event)
"""

import asyncio
import concurrent.futures
import copy
import logging
import mimetypes
import os
import shutil
import uuid
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.agent import _authorize_model_name, build_middlewares
from deerflow.agents.lead_agent.prompt import apply_prompt_template, get_enabled_skills_for_config
from deerflow.agents.thread_state import ThreadState
from deerflow.authz.principal import build_principal_from_context
from deerflow.config.agents_config import AGENT_NAME_PATTERN
from deerflow.config.app_config import get_app_config, is_trace_correlation_enabled, reload_app_config
from deerflow.config.extensions_config import (
    ExtensionsConfig,
    SkillStateConfig,
    atomic_write_extensions_config,
    get_extensions_config,
    reload_extensions_config,
)
from deerflow.config.paths import get_paths
from deerflow.models import create_chat_model
from deerflow.runtime import CheckpointStateAccessor
from deerflow.runtime.goal import DEFAULT_MAX_GOAL_CONTINUATIONS, build_goal_state, goal_thread_lock, read_thread_goal, write_thread_goal
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.skills.describe import build_skill_search_setup
from deerflow.skills.storage import get_or_new_user_skill_storage
from deerflow.tools.builtins.tool_search import assemble_deferred_tools, build_mcp_routing_middleware, get_mcp_routing_hints_prompt_section
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY, generate_trace_id, get_current_trace_id, reset_current_trace_id, set_current_trace_id
from deerflow.tracing import build_tracing_callbacks, inject_langfuse_metadata
from deerflow.uploads.manager import (
    claim_unique_filename,
    delete_file_safe,
    enrich_file_listing,
    ensure_uploads_dir,
    get_uploads_dir,
    list_files_in_dir,
    upload_artifact_url,
    upload_virtual_path,
)
from deerflow.utils.thread_id import resolve_thread_id, validate_thread_id

logger = logging.getLogger(__name__)

_EMBEDDED_AUTHORIZATION_CONTEXT_KEYS = frozenset(
    {
        "user_id",
        "user_role",
        "oauth_provider",
        "oauth_id",
        "channel_user_id",
        "is_internal",
        "authz_attributes",
    }
)


def _run_async_from_sync(coro):
    """동기 클라이언트 API에서 async 헬퍼를 실행한다."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


StreamEventType = Literal["values", "messages-tuple", "custom", "end"]


@dataclass
class StreamEvent:
    """스트리밍 agent 응답의 단일 이벤트.

    이벤트 타입은 LangGraph SSE protocol을 따른다.

        - ``"values"``: 전체 state 스냅샷(title, messages, artifacts).
        - ``"messages-tuple"``: 메시지 단위 갱신(AI 텍스트, tool call, tool 결과).
        - ``"end"``: 스트림 종료.

    Attributes:
        type: 이벤트 타입.
        data: 이벤트 payload. 내용은 타입에 따라 다르다.
    """

    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)


class DeerFlowClient:
    """DeerFlow agent 시스템의 임베디드 Python 클라이언트.

    LangGraph Server나 Gateway API 프로세스 없이 DeerFlow agent 기능에 직접 접근한다.

    Note:
        멀티턴 대화에는 ``checkpointer``가 필요하다. 없으면 각 ``stream()`` / ``chat()`` 호출은
        stateless이고 ``thread_id``는 파일 격리(uploads / artifacts)에만 쓰인다.

        system prompt(날짜, memory, skill context 포함)는 내부 agent가 처음 생성될 때 만들어져
        설정 키가 바뀔 때까지 캐싱된다. 오래 사는 프로세스에서 갱신을 강제하려면
        :meth:`reset_agent`를 호출한다.

    Example::

        from deerflow.client import DeerFlowClient

        client = DeerFlowClient()

        # Simple one-shot
        print(client.chat("hello"))

        # Streaming
        for event in client.stream("hello"):
            print(event.type, event.data)

        # Configuration queries
        print(client.list_models())
        print(client.list_skills())
    """

    def __init__(
        self,
        config_path: str | None = None,
        checkpointer=None,
        *,
        model_name: str | None = None,
        thinking_enabled: bool = True,
        subagent_enabled: bool = False,
        plan_mode: bool = False,
        agent_name: str | None = None,
        available_skills: set[str] | None = None,
        middlewares: Sequence[AgentMiddleware] | None = None,
        environment: str | None = None,
    ):
        """클라이언트를 초기화한다.

        설정만 로드하고 agent 생성은 첫 사용까지 미룬다.

        Args:
            config_path: config.yaml 경로. None이면 기본 해석 규칙을 쓴다.
            checkpointer: state 영속화를 위한 LangGraph checkpointer 인스턴스.
                같은 thread_id에서 멀티턴 대화를 하려면 필요하다.
                없으면 각 호출이 stateless가 된다.
            model_name: config의 기본 model 이름을 덮어쓴다.
            thinking_enabled: 모델의 extended thinking을 켠다.
            subagent_enabled: subagent 위임을 켠다.
            plan_mode: plan mode용 TodoList middleware를 켠다.
            agent_name: 사용할 agent 이름.
            available_skills: 사용 가능하게 할 skill 이름 집합. None(기본값)이면 스캔된 모든 skill을 쓴다.
            middlewares: agent에 주입할 커스텀 middleware 목록.
            environment: ``langfuse_tags``에 실리는 배포 환경 라벨
                (예: ``"production"`` / ``"staging"``). ``None``이면 worker/client가
                ``DEER_FLOW_ENV`` 또는 ``ENVIRONMENT`` 환경 변수로 폴백한다.
                환경 변수에 결합되고 싶지 않은 프로그램적 호출자는 명시적으로 값을 넘긴다.
        """
        if config_path is not None:
            reload_app_config(config_path)
        self._app_config = get_app_config()

        if agent_name is not None and not AGENT_NAME_PATTERN.match(agent_name):
            raise ValueError(f"Invalid agent name '{agent_name}'. Must match pattern: {AGENT_NAME_PATTERN.pattern}")

        self._checkpointer = checkpointer
        self._model_name = model_name
        self._thinking_enabled = thinking_enabled
        self._subagent_enabled = subagent_enabled
        self._plan_mode = plan_mode
        self._agent_name = agent_name
        self._available_skills = set(available_skills) if available_skills is not None else None
        self._middlewares = list(middlewares) if middlewares else []
        self._environment = environment

        # 지연 생성 agent. 첫 호출에서 만들고 설정이 바뀌면 다시 만든다.
        self._agent = None
        self._agent_config_key: tuple | None = None

    def reset_agent(self) -> None:
        """다음 호출에서 내부 agent를 다시 만들도록 강제한다.

        system prompt나 도구 집합에 반영되어야 하는 외부 변경
        (예: memory 갱신, skill 설치) 후에 사용한다.
        """
        self._agent = None
        self._agent_config_key = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """JSON을 *path*에 원자적으로 쓴다(임시 파일 + replace)."""
        atomic_write_extensions_config(path, data)

    def _get_runnable_config(self, thread_id: str, **overrides) -> RunnableConfig:
        """agent 호출용 RunnableConfig를 만든다."""
        configurable = {
            "thread_id": thread_id,
            "model_name": overrides.get("model_name", self._model_name),
            "thinking_enabled": overrides.get("thinking_enabled", self._thinking_enabled),
            "is_plan_mode": overrides.get("plan_mode", self._plan_mode),
            "subagent_enabled": overrides.get("subagent_enabled", self._subagent_enabled),
        }
        return RunnableConfig(
            configurable=configurable,
            recursion_limit=overrides.get("recursion_limit", 100),
        )

    def _ensure_agent(self, config: RunnableConfig, *, context: Mapping[str, Any] | None = None):
        """설정에 의존하는 파라미터가 바뀌면 agent를 생성하거나 재생성한다."""
        cfg = dict(config.get("configurable", {}) or {})
        if context is not None:
            cfg.update(context)

        authorization_identity = None
        if self._app_config.authorization.enabled:
            principal = build_principal_from_context(
                cfg,
                default_role=self._app_config.authorization.default_role,
            )
            authorization_identity = (
                principal.user_id,
                principal.role,
                principal.oauth_provider,
                principal.oauth_id,
                principal.channel_user_id,
                principal.is_internal,
                copy.deepcopy(principal.attributes),
            )
        key = (
            cfg.get("model_name"),
            cfg.get("thinking_enabled"),
            cfg.get("is_plan_mode"),
            cfg.get("subagent_enabled"),
            cfg.get("max_concurrent_subagents"),
            cfg.get("max_total_subagents"),
            self._agent_name,
            frozenset(self._available_skills) if self._available_skills is not None else None,
            authorization_identity,
        )

        if self._agent is not None and self._agent_config_key == key:
            return

        thinking_enabled = cfg.get("thinking_enabled", True)
        model_name = cfg.get("model_name")
        # Phase 3: ``_make_lead_agent``의 Gateway runtime 경로와 동일하게 embedded/library
        # 경로에서도 model:use authorization을 강제한다. ``DeerFlowClient``로 agent를 만들어
        # role 범위 model 정책을 우회하지 못하게 하기 위해서다. ``None`` 기본값을 먼저 구체적인
        # 이름으로 해석해서(``create_chat_model(name=None)``이 고를 값) 암묵적 기본 model까지
        # 정책이 덮게 한다. ``cfg``에는 아래 ``apply_tool_authorization``이 읽는 identity가 이미 들어 있다.
        if model_name is None and self._app_config.models:
            model_name = self._app_config.models[0].name
        model_name = _authorize_model_name(model_name, context=cfg, app_config=self._app_config)
        subagent_enabled = cfg.get("subagent_enabled", False)
        max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
        max_total_subagents = cfg.get("max_total_subagents", self._app_config.subagents.max_total_per_run)

        tools = self._get_tools(model_name=model_name, subagent_enabled=subagent_enabled)

        # authorization 전에 framework 제공 도구를 추가해서, Layer 1이 모델에게 보일 수 있는
        # 모든 기능을 보게 한다.
        skills_list = get_enabled_skills_for_config(self._app_config)
        if self._available_skills is not None:
            skills_list = [s for s in skills_list if s.name in self._available_skills]
        skill_setup = build_skill_search_setup(
            skills_list,
            enabled=self._app_config.skills.deferred_discovery,
            container_base_path=self._app_config.skills.container_path,
        )
        late_tools = []
        if skill_setup.describe_skill_tool:
            late_tools.append(skill_setup.describe_skill_tool)

        # deferred assembly 전에 authorization Layer 1을 적용한다.
        from deerflow.authz.tool_filter import apply_tool_authorization

        configured_tool_ids = {id(tool) for tool in tools}
        authorized_tools, _authz_provider = apply_tool_authorization(
            [*tools, *late_tools],
            context=cfg,
            app_config=self._app_config,
        )
        tools = [tool for tool in authorized_tools if id(tool) in configured_tool_ids]
        late_tools = [tool for tool in authorized_tools if id(tool) not in configured_tool_ids]
        final_tools, deferred_setup = assemble_deferred_tools(tools, enabled=self._app_config.tool_search.enabled)
        final_tools.extend(late_tools)
        mcp_routing_middleware = build_mcp_routing_middleware(
            final_tools,
            deferred_setup,
            top_k=self._app_config.tool_search.auto_promote_top_k,
        )
        mcp_routing_hints_section = get_mcp_routing_hints_prompt_section(authorized_tools, deferred_names=deferred_setup.deferred_names)

        effective_user_id = cfg.get("user_id") or get_effective_user_id()

        kwargs: dict[str, Any] = {
            # attach_tracing=False인 이유: ``stream()``이 graph 호출 루트에 tracing callback을
            # 주입해서 embedded run 하나가 session_id / user_id가 올바르게 전파된 trace 하나를
            # 만들기 때문이다. 모델에 다시 붙이면 span이 중복된다.
            "model": create_chat_model(name=model_name, thinking_enabled=thinking_enabled, attach_tracing=False),
            "tools": final_tools,
            "middleware": build_middlewares(
                config,
                model_name=model_name,
                agent_name=self._agent_name,
                available_skills=self._available_skills,
                custom_middlewares=self._middlewares,
                app_config=self._app_config,
                deferred_setup=deferred_setup,
                mcp_routing_middleware=mcp_routing_middleware,
                user_id=effective_user_id,
                authorization_provider=_authz_provider,
            ),
            "system_prompt": apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                max_total_subagents=max_total_subagents,
                agent_name=self._agent_name,
                available_skills=self._available_skills,
                app_config=self._app_config,
                deferred_names=deferred_setup.deferred_names,
                mcp_routing_hints_section=mcp_routing_hints_section,
                user_id=effective_user_id,
                skill_names=skill_setup.skill_names or None,
            ),
            "state_schema": ThreadState,
        }
        checkpointer = self._checkpointer
        if checkpointer is None:
            from deerflow.runtime.checkpointer import get_checkpointer

            checkpointer = get_checkpointer()
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer

        self._agent = create_agent(**kwargs)
        self._agent_config_key = key
        logger.info("Agent created: agent_name=%s, model=%s, thinking=%s", self._agent_name, model_name, thinking_enabled)

    @staticmethod
    def _get_tools(*, model_name: str | None, subagent_enabled: bool):
        """모듈 수준 순환 의존을 피하기 위한 지연 import."""
        from deerflow.tools import get_available_tools

        return get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled)

    @staticmethod
    def _serialize_tool_calls(tool_calls) -> list[dict]:
        """LangChain tool_calls를 이벤트에서 쓰는 wire 형식으로 바꾼다."""
        return [{"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in tool_calls]

    @staticmethod
    def _serialize_additional_kwargs(msg) -> dict[str, Any] | None:
        """메시지에 additional_kwargs가 있으면 복사한다."""
        additional_kwargs = getattr(msg, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and additional_kwargs:
            return dict(additional_kwargs)
        return None

    @staticmethod
    def _ai_text_event(msg_id: str | None, text: str, usage: dict | None, additional_kwargs: dict[str, Any] | None = None) -> "StreamEvent":
        """``messages-tuple`` AI 텍스트 이벤트를 만든다."""
        data: dict[str, Any] = {"type": "ai", "content": text, "id": msg_id}
        if usage:
            data["usage_metadata"] = usage
        if additional_kwargs:
            data["additional_kwargs"] = additional_kwargs
        return StreamEvent(type="messages-tuple", data=data)

    @staticmethod
    def _ai_tool_calls_event(msg_id: str | None, tool_calls, additional_kwargs: dict[str, Any] | None = None) -> "StreamEvent":
        """``messages-tuple`` AI tool-call 이벤트를 만든다."""
        data: dict[str, Any] = {
            "type": "ai",
            "content": "",
            "id": msg_id,
            "tool_calls": DeerFlowClient._serialize_tool_calls(tool_calls),
        }
        if additional_kwargs:
            data["additional_kwargs"] = additional_kwargs
        return StreamEvent(type="messages-tuple", data=data)

    @staticmethod
    def _tool_message_event(msg: ToolMessage) -> "StreamEvent":
        """ToolMessage로부터 ``messages-tuple`` tool 결과 이벤트를 만든다."""
        data: dict[str, Any] = {
            "type": "tool",
            "content": DeerFlowClient._extract_text(msg.content),
            "name": msg.name,
            "tool_call_id": msg.tool_call_id,
            "id": msg.id,
        }
        if (artifact := getattr(msg, "artifact", None)) is not None:
            data["artifact"] = artifact
        return StreamEvent(type="messages-tuple", data=data)

    @staticmethod
    def _serialize_message(msg) -> dict:
        """LangChain 메시지를 values 이벤트용 평범한 dict로 직렬화한다."""
        if isinstance(msg, AIMessage):
            d: dict[str, Any] = {"type": "ai", "content": msg.content, "id": getattr(msg, "id", None)}
            if msg.tool_calls:
                d["tool_calls"] = DeerFlowClient._serialize_tool_calls(msg.tool_calls)
            if getattr(msg, "usage_metadata", None):
                d["usage_metadata"] = msg.usage_metadata
            if additional_kwargs := DeerFlowClient._serialize_additional_kwargs(msg):
                d["additional_kwargs"] = additional_kwargs
            return d
        if isinstance(msg, ToolMessage):
            d = {
                "type": "tool",
                "content": DeerFlowClient._extract_text(msg.content),
                "name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "id": getattr(msg, "id", None),
            }
            if additional_kwargs := DeerFlowClient._serialize_additional_kwargs(msg):
                d["additional_kwargs"] = additional_kwargs
            if (artifact := getattr(msg, "artifact", None)) is not None:
                d["artifact"] = artifact
            return d
        if isinstance(msg, HumanMessage):
            d = {"type": "human", "content": msg.content, "id": getattr(msg, "id", None)}
            if additional_kwargs := DeerFlowClient._serialize_additional_kwargs(msg):
                d["additional_kwargs"] = additional_kwargs
            return d
        if isinstance(msg, SystemMessage):
            d = {"type": "system", "content": msg.content, "id": getattr(msg, "id", None)}
            if additional_kwargs := DeerFlowClient._serialize_additional_kwargs(msg):
                d["additional_kwargs"] = additional_kwargs
            return d
        return {"type": "unknown", "content": str(msg), "id": getattr(msg, "id", None)}

    @staticmethod
    def _extract_text(content) -> str:
        """AIMessage content(str 또는 block 리스트)에서 순수 텍스트를 뽑아낸다.

        문자열 chunk는 구분자 없이 이어 붙인다. token/문자 delta나 조각난 JSON payload를
        망가뜨리지 않기 위해서다. dict 기반 텍스트 block은 완전한 텍스트 block으로 보고
        가독성을 위해 줄바꿈으로 연결한다.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if content and all(isinstance(block, str) for block in content):
                chunk_like = len(content) > 1 and all(isinstance(block, str) and len(block) <= 20 and any(ch in block for ch in '{}[]":,') for block in content)
                return "".join(content) if chunk_like else "\n".join(content)

            pieces: list[str] = []
            pending_str_parts: list[str] = []

            def flush_pending_str_parts() -> None:
                if pending_str_parts:
                    pieces.append("".join(pending_str_parts))
                    pending_str_parts.clear()

            for block in content:
                if isinstance(block, str):
                    pending_str_parts.append(block)
                elif isinstance(block, dict):
                    flush_pending_str_parts()
                    text_val = block.get("text")
                    if isinstance(text_val, str):
                        pieces.append(text_val)

            flush_pending_str_parts()
            return "\n".join(pieces) if pieces else ""
        return str(content)

    # ------------------------------------------------------------------
    # 공개 API — threads
    # ------------------------------------------------------------------

    def _get_thread_checkpointer(self):
        checkpointer = self._checkpointer
        if checkpointer is None:
            from deerflow.runtime.checkpointer.provider import get_checkpointer

            checkpointer = get_checkpointer()
        return checkpointer

    def get_goal(self, thread_id: str) -> dict:
        """thread의 활성 goal이 있으면 반환한다."""
        validate_thread_id(thread_id)
        checkpointer = self._get_thread_checkpointer()
        goal = _run_async_from_sync(read_thread_goal(checkpointer, thread_id))
        return {"goal": goal}

    def set_goal(
        self,
        thread_id: str,
        objective: str,
        *,
        max_continuations: int = DEFAULT_MAX_GOAL_CONTINUATIONS,
    ) -> dict:
        """thread 범위 goal을 설정하거나 교체한다."""
        validate_thread_id(thread_id)
        checkpointer = self._get_thread_checkpointer()
        goal = build_goal_state(objective, max_continuations=max_continuations)

        async def _set_goal() -> None:
            async with goal_thread_lock(thread_id):
                await write_thread_goal(checkpointer, thread_id, goal, create_if_missing=True)

        _run_async_from_sync(_set_goal())
        return {"goal": goal}

    def clear_goal(self, thread_id: str) -> dict:
        """thread의 활성 goal을 지운다."""
        validate_thread_id(thread_id)
        checkpointer = self._get_thread_checkpointer()

        async def _clear_goal() -> None:
            async with goal_thread_lock(thread_id):
                await write_thread_goal(checkpointer, thread_id, None)

        try:
            _run_async_from_sync(_clear_goal())
        except LookupError:
            pass
        return {"goal": None}

    def list_threads(self, limit: int = 10) -> dict:
        """최근 N개 thread를 나열한다.

        Args:
            limit: 반환할 thread 최대 개수. 기본값은 10.

        Returns:
            thread 정보 dict 목록을 담은 "thread_list" 키를 가진 dict.
            thread 생성 시각 내림차순으로 정렬된다.
        """
        checkpointer = self._get_thread_checkpointer()

        thread_info_map = {}

        for cp in checkpointer.list(config=None, limit=limit):
            cfg = cp.config.get("configurable", {})
            thread_id = cfg.get("thread_id")
            if not thread_id:
                continue

            ts = cp.checkpoint.get("ts")
            checkpoint_id = cfg.get("checkpoint_id")

            if thread_id not in thread_info_map:
                channel_values = cp.checkpoint.get("channel_values", {})
                thread_info_map[thread_id] = {
                    "thread_id": thread_id,
                    "created_at": ts,
                    "updated_at": ts,
                    "latest_checkpoint_id": checkpoint_id,
                    "title": channel_values.get("title"),
                }
            else:
                # 순서가 없는 namespace를 순회할 때 정확성을 위해 timestamp를 명시적으로 비교한다.
                # None은 "없음"으로 보고 기존 값이 None이 아닐 때만 비교한다.
                if ts is not None:
                    current_created = thread_info_map[thread_id]["created_at"]
                    if current_created is None or ts < current_created:
                        thread_info_map[thread_id]["created_at"] = ts

                    current_updated = thread_info_map[thread_id]["updated_at"]
                    if current_updated is None or ts > current_updated:
                        thread_info_map[thread_id]["updated_at"] = ts
                        thread_info_map[thread_id]["latest_checkpoint_id"] = checkpoint_id
                        channel_values = cp.checkpoint.get("channel_values", {})
                        thread_info_map[thread_id]["title"] = channel_values.get("title")

        threads = list(thread_info_map.values())
        threads.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return {"thread_list": threads[:limit]}

    def get_thread(self, thread_id: str) -> dict:
        """thread의 완전히 materialize된 checkpoint 히스토리를 가져온다."""
        checkpointer = self._get_thread_checkpointer()
        config = self._get_runnable_config(thread_id)
        self._ensure_agent(config)
        if self._agent is None:
            raise RuntimeError("Agent was not initialized")

        accessor = CheckpointStateAccessor.bind(self._agent, checkpointer)
        # 한 번의 streaming walk로 checkpoint id별 pending_writes를 모은다.
        # 스냅샷마다 get_tuple을 부르면 checkpoint당 round-trip이 한 번씩 든다.
        pending_writes_by_checkpoint: dict[str, list] = {}
        for raw_tuple in checkpointer.list(config):
            raw_checkpoint_id = raw_tuple.config.get("configurable", {}).get("checkpoint_id")
            if raw_checkpoint_id:
                pending_writes_by_checkpoint[raw_checkpoint_id] = list(getattr(raw_tuple, "pending_writes", ()) or ())

        checkpoints = []
        for snapshot in accessor.history(config):
            values = dict(snapshot.values or {})
            if "messages" in values:
                values["messages"] = [self._serialize_message(message) if hasattr(message, "content") else message for message in values["messages"]]

            snapshot_config = snapshot.config or {}
            configurable = snapshot_config.get("configurable", {})
            parent_config = snapshot.parent_config or {}
            parent_configurable = parent_config.get("configurable", {})
            pending_writes = pending_writes_by_checkpoint.get(configurable.get("checkpoint_id"), [])

            checkpoints.append(
                {
                    "checkpoint_id": configurable.get("checkpoint_id"),
                    "parent_checkpoint_id": parent_configurable.get("checkpoint_id"),
                    "ts": snapshot.created_at,
                    "metadata": snapshot.metadata,
                    "values": values,
                    "pending_writes": [{"task_id": write[0], "channel": write[1], "value": write[2]} for write in pending_writes],
                }
            )

        checkpoints.sort(key=lambda checkpoint: checkpoint["ts"] or "")
        return {"thread_id": thread_id, "checkpoints": checkpoints}

    # ------------------------------------------------------------------
    # 공개 API — conversation
    # ------------------------------------------------------------------

    def stream(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        **kwargs,
    ) -> Generator[StreamEvent, None, None]:
        """DeerFlow request trace context와 함께 대화 턴을 스트리밍한다.

        Gateway ``TraceMiddleware``의 게이트를 그대로 따른다. ``logging.enhance.enabled``가
        꺼져 있으면 embedded client는 새 request 수준 trace id를 만들지 **않는다**.
        그래서 embedded / TUI / CLI 호출자의 Langfuse trace는 enhancement 이전 스키마를 유지하고
        기본적으로 ``metadata.deerflow_trace_id`` 키를 갖지 않는다.
        :func:`deerflow.trace_context.request_trace_context`로 직접 trace를 바인딩한 호출자는
        여전히 참여한다. 내부의 ``get_current_trace_id()`` 읽기가 플래그와 무관하게
        그 값을 Langfuse metadata로 전파한다.
        """
        if not is_trace_correlation_enabled(self._app_config):
            yield from self._stream_without_trace_context(message, thread_id=thread_id, **kwargs)
            return

        # 호출자의 context를 건드리지 않고 trace id를 한 번만 해석한다.
        # 호출자가 ``request_trace_context``로 참여했다면 주변 id를 상속하고,
        # 아니면 새로 발급한다.
        trace_id = get_current_trace_id() or generate_trace_id()

        # trace id는 각 ``next()`` 단계 주위에만 바인딩하고 ``yield``를 가로질러 유지하지 않는다.
        # ``stream()``은 sync generator라서 호출자의 context를 공유한다.
        # ``with ensure_trace_context(): yield from ...``으로 감싸면 (1) yield 사이에 id가
        # 호출자 context로 새고, (2) 버려진 generator를 GC가 다른 context에서 정리할 때
        # ``ValueError: Token was created in a different Context`` 위험이 생긴다.
        # 단계마다 set/reset하면 LangGraph 노드 실행과 그 로그 레코드는 바인딩 안에 두면서,
        # 호출자에게 제어를 돌려줄 때는 ContextVar가 복원된 상태가 된다.
        inner = self._stream_without_trace_context(message, thread_id=thread_id, **kwargs)
        _EXHAUSTED = object()
        try:
            while True:
                token = set_current_trace_id(trace_id)
                try:
                    try:
                        event = next(inner)
                    except StopIteration:
                        event = _EXHAUSTED
                finally:
                    reset_current_trace_id(token)
                if event is _EXHAUSTED:
                    break
                yield event
        finally:
            inner.close()

    def _stream_without_trace_context(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        **kwargs,
    ) -> Generator[StreamEvent, None, None]:
        """대화 턴을 스트리밍하며 이벤트를 점진적으로 내보낸다.

        호출 한 번은 사용자 메시지 하나를 보내고 agent가 턴을 마칠 때까지 이벤트를 내보낸다.
        호출 간에 멀티턴 context를 유지하려면 init 시점에 ``checkpointer``를 넘겨야 한다.

        이벤트 타입은 LangGraph SSE protocol을 따르므로, 소비자는 이벤트 처리 로직을 바꾸지 않고
        HTTP streaming과 embedded 모드를 오갈 수 있다.

        토큰 수준 스트리밍
        ~~~~~~~~~~~~~~~~~~~~~
        이 메서드는 LangGraph의 ``messages`` stream mode를 구독한다. 따라서 AI 텍스트의
        ``messages-tuple`` 이벤트는 노드 완료 시 누적 덤프 하나가 아니라, 모델이 토큰을 생성하는
        대로 **delta**로 나온다. 각 delta는 안정적인 ``id``를 갖는다. 전체 텍스트가 필요한 소비자는
        ``id``별로 ``content``를 누적해야 한다. ``chat()``은 이미 이 처리를 해준다.

        tool call과 tool 결과는 여전히 논리적 메시지당 한 번만 나온다. ``values`` 이벤트는 각 graph
        노드가 끝날 때마다 전체 state 스냅샷을 계속 실어 나른다. 다만 ``messages`` 스트림으로 이미
        전달된 AI 텍스트는 중복 전달을 피하려고 스냅샷에서 다시 만들지 **않는다**.

        Gateway의 ``run_agent``를 재사용하지 않는 이유
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Gateway(``runtime/runs/worker.py``)에는 ``run_agent`` → ``StreamBridge`` →
        ``sse_consumer``로 이어지는 완전한 streaming 파이프라인이 있다. 이 클라이언트가 그 작업을
        중복하는 것처럼 보이지만, 두 경로는 대상이 다르고 실행을 공유할 **수 없다**.

        * ``run_agent``는 ``async def``이고 ``agent.astream()``을 쓴다. 이 메서드는
          ``agent.stream()``을 쓰는 sync generator라서 호출자가 asyncio를 건드리지 않고
          ``for event in client.stream(...)``을 쓸 수 있다. 둘을 잇자면 호출마다
          event loop와 thread를 띄워야 한다.
        * Gateway 이벤트는 SSE 전송을 위해 ``serialize()``로 JSON 직렬화된다. 이 클라이언트는
          HTTP 전달용 JSON/SSE 직렬화 계층 없이 in-process stream 이벤트 payload를 Python 데이터
          구조 그대로(``data``가 평범한 ``dict``인 ``StreamEvent``) 내보낸다.
        * ``StreamBridge``는 HTTP 경계를 사이에 두고 생산자와 소비자를 분리하는 asyncio queue다
          (``Last-Event-ID`` replay, heartbeat, 다중 구독자 fan-out). 직접 iterator를 쓰는
          단일 in-process 호출자에게는 그중 무엇도 필요 없다.

        즉 ``DeerFlowClient.stream()``은 Gateway를 감싼 wrapper가 아니라, 같은
        ``create_agent()`` factory를 쓰는 병렬적·동기적 in-process 소비자다. 두 경로는 어떤
        LangGraph stream mode를 구독하는지에 대해 계속 동기화되어야 **한다**. 이 불변식은 공유 상수가
        아니라 ``tests/test_client.py::test_messages_mode_emits_token_deltas``로 강제된다.
        세 계층(Graph, Platform SDK, HTTP)이 각자 다른 이름(``messages`` vs ``messages-tuple``)을
        쓰기 때문에 문자열을 문자 그대로 공유할 수 없기 때문이다.

        Args:
            message: 사용자 메시지 텍스트.
            thread_id: 대화 context용 Thread ID. None이면 자동 생성한다.
            **kwargs: 클라이언트 기본값을 덮어쓴다(model_name, thinking_enabled,
                plan_mode, subagent_enabled, recursion_limit). 신뢰된 embedded 호출자는
                user_id, user_role, oauth_provider, oauth_id, channel_user_id,
                is_internal, authz_attributes도 넘길 수 있다.

        Yields:
            다음 중 하나에 해당하는 StreamEvent:
            - type="values"          data={"title": str|None, "messages": [...], "artifacts": [...]}
            - type="custom"          data={...}
            - type="messages-tuple"  data={"type": "ai", "content": <delta>, "id": str}
            - type="messages-tuple"  data={"type": "ai", "content": <delta>, "id": str, "usage_metadata": {...}}
            - type="messages-tuple"  data={"type": "ai", "content": "", "id": str, "tool_calls": [...]}
            - type="messages-tuple"  data={"type": "ai", "content": "", "id": str, "additional_kwargs": {...}}
            - type="messages-tuple"  data={"type": "tool", "content": str, "name": str, "tool_call_id": str, "id": str}
              원본 ToolMessage에 None이 아닌 artifact가 있으면 tool 결과에 ``"artifact"``도 포함된다.
            - type="end"             data={"usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}}
        """
        thread_id = resolve_thread_id(thread_id)

        config = self._get_runnable_config(thread_id, **kwargs)

        # tracing callback과 Langfuse trace metadata를 graph 호출 루트에 주입해서 embedded
        # client가 gateway worker와 같은 동작을 하게 한다. ``stream()`` 한 번이 모든 노드 /
        # LLM / tool 호출을 하위에 중첩한 trace 하나를 만들고, 그 trace는 Langfuse
        # CallbackHandler가 루트 trace의 ``sessionId`` / ``userId``로 끌어올리는 예약 키
        # ``langfuse_session_id`` / ``langfuse_user_id``를 갖는다.
        tracing_callbacks = build_tracing_callbacks()
        if tracing_callbacks:
            existing_callbacks = list(config.get("callbacks") or [])
            config["callbacks"] = [*existing_callbacks, *tracing_callbacks]

        run_id = str(uuid.uuid4())
        context: dict[str, Any] = {"thread_id": thread_id, "run_id": run_id}
        for key in _EMBEDDED_AUTHORIZATION_CONTEXT_KEYS:
            if key in kwargs:
                context[key] = kwargs[key]

        configurable = config.get("configurable") or {}
        deerflow_trace_id = get_current_trace_id()
        effective_user_id = context.get("user_id") or get_effective_user_id()
        if self._app_config.authorization.enabled:
            # embedded 호출자가 명시적 user_id 대신 CurrentUser에 의존할 때, 기존의 user 범위
            # storage/tracing identity와 일치시킨다. Layer 1, Layer 2, agent 캐시가
            # 같은 행위 주체를 봐야 한다.
            context["user_id"] = effective_user_id
        inject_langfuse_metadata(
            config,
            thread_id=thread_id,
            user_id=effective_user_id,
            assistant_id=self._agent_name or "lead-agent",
            model_name=configurable.get("model_name") or self._model_name,
            environment=self._environment or os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT"),
            deerflow_trace_id=deerflow_trace_id,
        )

        self._ensure_agent(config, context=context)

        state: dict[str, Any] = {"messages": [HumanMessage(content=message, additional_kwargs={"run_id": run_id})]}
        if deerflow_trace_id:
            context[DEERFLOW_TRACE_METADATA_KEY] = deerflow_trace_id
        if self._agent_name:
            context["agent_name"] = self._agent_name

        seen_ids: set[str] = set()
        # 모드 간 인계: LangGraph ``messages`` 모드로 이미 스트리밍된 id들.
        # ``values`` 경로가 같은 메시지를 다시 만들지 않게 한다.
        streamed_ids: set[str] = set()
        # 같은 메시지 id는 마지막 ``messages`` chunk와 values 스냅샷 양쪽에 동일한 누적
        # ``usage_metadata``를 싣는다. 먼저 도착한 쪽에서만 집계한다.
        counted_usage_ids: set[str] = set()
        sent_additional_kwargs_by_id: dict[str, dict[str, Any]] = {}
        cumulative_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def _account_usage(msg_id: str | None, usage: Any) -> dict | None:
            """이 id가 아직 집계되지 않았다면 *usage*를 누적 합계에 더한다.

            ``usage``는 ``langchain_core.messages.UsageMetadata`` TypedDict이거나 ``None``이다.
            엄격한 타입 검사에서 TypedDict가 평범한 ``dict``에 구조적으로 대입되지 않으므로
            ``Any``로 표기했다. 값을 받아들였으면 (이벤트에 붙일) 정규화된 usage dict를,
            아니면 ``None``을 반환한다.
            """
            if not usage:
                return None
            if msg_id and msg_id in counted_usage_ids:
                return None
            if msg_id:
                counted_usage_ids.add(msg_id)
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
            cumulative_usage["input_tokens"] += input_tokens
            cumulative_usage["output_tokens"] += output_tokens
            cumulative_usage["total_tokens"] += total_tokens
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        def _unsent_additional_kwargs(msg_id: str | None, additional_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
            if not additional_kwargs:
                return None
            if not msg_id:
                return additional_kwargs

            sent = sent_additional_kwargs_by_id.setdefault(msg_id, {})
            delta = {key: value for key, value in additional_kwargs.items() if sent.get(key) != value}
            if not delta:
                return None

            sent.update(delta)
            return delta

        for item in self._agent.stream(
            state,
            config=config,
            context=context,
            stream_mode=["values", "messages", "custom"],
        ):
            if isinstance(item, tuple) and len(item) == 2:
                mode, chunk = item
                mode = str(mode)
            else:
                mode, chunk = "values", item

            if mode == "custom":
                yield StreamEvent(type="custom", data=chunk)
                continue

            if mode == "messages":
                # LangGraph ``messages`` 모드는 ``(message_chunk, metadata)``를 내보낸다.
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    msg_chunk, _metadata = chunk
                else:
                    msg_chunk = chunk

                msg_id = getattr(msg_chunk, "id", None)

                if isinstance(msg_chunk, AIMessage):
                    text = self._extract_text(msg_chunk.content)
                    additional_kwargs = self._serialize_additional_kwargs(msg_chunk)
                    counted_usage = _account_usage(msg_id, msg_chunk.usage_metadata)
                    sent_additional_kwargs = False

                    if text:
                        if msg_id:
                            streamed_ids.add(msg_id)
                        additional_kwargs_delta = _unsent_additional_kwargs(msg_id, additional_kwargs)
                        yield self._ai_text_event(
                            msg_id,
                            text,
                            counted_usage,
                            additional_kwargs_delta,
                        )
                        sent_additional_kwargs = bool(additional_kwargs_delta)

                    if msg_chunk.tool_calls:
                        if msg_id:
                            streamed_ids.add(msg_id)
                        additional_kwargs_delta = None if sent_additional_kwargs else _unsent_additional_kwargs(msg_id, additional_kwargs)
                        yield self._ai_tool_calls_event(
                            msg_id,
                            msg_chunk.tool_calls,
                            additional_kwargs_delta,
                        )

                elif isinstance(msg_chunk, ToolMessage):
                    if msg_id:
                        streamed_ids.add(msg_id)
                    yield self._tool_message_event(msg_chunk)
                continue

            # mode == "values"
            messages = chunk.get("messages", [])

            for msg in messages:
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id in seen_ids:
                    continue
                if msg_id:
                    seen_ids.add(msg_id)

                # ``messages`` 모드로 이미 스트리밍됐다. 여기서는 방어적으로 usage만 잡고
                # 이벤트를 다시 만들지 않는다.
                if msg_id and msg_id in streamed_ids:
                    if isinstance(msg, AIMessage):
                        _account_usage(msg_id, getattr(msg, "usage_metadata", None))
                        additional_kwargs = self._serialize_additional_kwargs(msg)
                        additional_kwargs_delta = _unsent_additional_kwargs(msg_id, additional_kwargs)
                        if additional_kwargs_delta:
                            # metadata만 담은 후속 이벤트. ``messages-tuple``에는 전용
                            # attribution 이벤트가 없으므로, 클라이언트는 이 빈 content AI
                            # 이벤트를 메시지 id로 병합하고 텍스트 렌더링에서는 무시해야 한다.
                            yield self._ai_text_event(msg_id, "", None, additional_kwargs_delta)
                    continue

                if isinstance(msg, AIMessage):
                    counted_usage = _account_usage(msg_id, msg.usage_metadata)
                    additional_kwargs = self._serialize_additional_kwargs(msg)
                    sent_additional_kwargs = False

                    if msg.tool_calls:
                        additional_kwargs_delta = _unsent_additional_kwargs(msg_id, additional_kwargs)
                        yield self._ai_tool_calls_event(
                            msg_id,
                            msg.tool_calls,
                            additional_kwargs_delta,
                        )
                        sent_additional_kwargs = bool(additional_kwargs_delta)

                    text = self._extract_text(msg.content)
                    if text:
                        additional_kwargs_delta = None if sent_additional_kwargs else _unsent_additional_kwargs(msg_id, additional_kwargs)
                        yield self._ai_text_event(
                            msg_id,
                            text,
                            counted_usage,
                            additional_kwargs_delta,
                        )
                    elif msg_id:
                        additional_kwargs_delta = None if sent_additional_kwargs else _unsent_additional_kwargs(msg_id, additional_kwargs)
                        if not additional_kwargs_delta:
                            continue
                        # 위의 metadata 전용 후속 이벤트 관례를 참고한다.
                        yield self._ai_text_event(msg_id, "", None, additional_kwargs_delta)

                elif isinstance(msg, ToolMessage):
                    yield self._tool_message_event(msg)

            # state 스냅샷마다 values 이벤트를 내보낸다
            yield StreamEvent(
                type="values",
                data={
                    "title": chunk.get("title"),
                    "messages": [self._serialize_message(m) for m in messages],
                    "artifacts": chunk.get("artifacts", []),
                },
            )

        yield StreamEvent(type="end", data={"usage": cumulative_usage})

    def chat(self, message: str, *, thread_id: str | None = None, **kwargs) -> str:
        """메시지를 보내고 최종 텍스트 응답을 반환한다.

        :meth:`stream`을 감싼 편의 wrapper다. delta ``messages-tuple`` 이벤트를 ``id``별로
        누적하고 **마지막**으로 완료된 AI 메시지의 텍스트를 반환한다. 중간 AI 메시지
        (예: planner 초안)는 버리고 마지막 id의 누적 텍스트만 반환한다.
        도착하는 모든 delta가 필요하면 :meth:`stream`을 직접 쓴다.

        Args:
            message: 사용자 메시지 텍스트.
            thread_id: 대화 context용 Thread ID. None이면 자동 생성한다.
            **kwargs: 클라이언트 기본값을 덮어쓴다(stream()과 동일).

        Returns:
            마지막 AI 메시지의 누적 텍스트. AI 텍스트가 없었다면 빈 문자열.
        """
        # id별 delta 리스트를 끝에서 한 번만 join한다. 긴 응답에서 버퍼가 커지며
        # ``str + str``을 반복할 때 드는 O(n²) 비용을 피한다.
        chunks: dict[str, list[str]] = {}
        last_id: str = ""
        for event in self.stream(message, thread_id=thread_id, **kwargs):
            if event.type == "messages-tuple" and event.data.get("type") == "ai":
                msg_id = event.data.get("id") or ""
                delta = event.data.get("content", "")
                if delta:
                    chunks.setdefault(msg_id, []).append(delta)
                    last_id = msg_id
        return "".join(chunks.get(last_id, ()))

    # ------------------------------------------------------------------
    # 공개 API — 설정 조회
    # ------------------------------------------------------------------

    def list_models(self) -> dict:
        """설정에 있는 사용 가능한 model을 나열한다.

        Returns:
            model 정보 dict 목록을 담은 "models" 키를 가진 dict.
            Gateway API ``ModelsListResponse`` 스키마와 일치한다.
        """
        token_usage_enabled = getattr(getattr(self._app_config, "token_usage", None), "enabled", False)
        if not isinstance(token_usage_enabled, bool):
            token_usage_enabled = False

        return {
            "models": [
                {
                    "name": model.name,
                    "model": getattr(model, "model", None),
                    "display_name": getattr(model, "display_name", None),
                    "description": getattr(model, "description", None),
                    "supports_thinking": getattr(model, "supports_thinking", False),
                    "supports_reasoning_effort": getattr(model, "supports_reasoning_effort", False),
                }
                for model in self._app_config.models
            ],
            "token_usage": {"enabled": token_usage_enabled},
        }

    def list_skills(self, enabled_only: bool = False) -> dict:
        """사용 가능한 skill을 나열한다.

        Args:
            enabled_only: True면 활성화된 skill만 반환한다.

        Returns:
            skill 정보 dict 목록을 담은 "skills" 키를 가진 dict.
            Gateway API ``SkillsListResponse`` 스키마와 일치한다.
        """
        storage = get_or_new_user_skill_storage(get_effective_user_id(), app_config=self._app_config)
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "license": s.license,
                    "category": s.category,
                    "enabled": s.enabled,
                }
                for s in storage.load_skills(enabled_only=enabled_only)
            ]
        }

    def get_memory(self) -> dict:
        """현재 memory 데이터를 가져온다.

        Returns:
            memory 데이터 dict. 구조는 src/agents/memory/updater.py를 참고한다.
        """
        from deerflow.agents.memory import get_memory_manager

        return get_memory_manager().get_memory(user_id=get_effective_user_id())

    def export_memory(self) -> dict:
        """백업이나 이전을 위해 현재 memory 데이터를 내보낸다."""
        from deerflow.agents.memory import get_memory_manager

        return get_memory_manager().get_memory(user_id=get_effective_user_id())

    def import_memory(self, memory_data: dict) -> dict:
        """전체 memory 데이터를 가져와 저장한다."""
        from deerflow.agents.memory import get_memory_manager

        return get_memory_manager().import_memory(memory_data, user_id=get_effective_user_id())

    def get_model(self, name: str) -> dict | None:
        """이름으로 특정 model의 설정을 가져온다.

        Args:
            name: model 이름.

        Returns:
            Gateway API ``ModelResponse`` 스키마와 일치하는 model 정보 dict.
            없으면 None.
        """
        model = self._app_config.get_model_config(name)
        if model is None:
            return None
        return {
            "name": model.name,
            "model": getattr(model, "model", None),
            "display_name": getattr(model, "display_name", None),
            "description": getattr(model, "description", None),
            "supports_thinking": getattr(model, "supports_thinking", False),
            "supports_reasoning_effort": getattr(model, "supports_reasoning_effort", False),
        }

    # ------------------------------------------------------------------
    # 공개 API — MCP 설정
    # ------------------------------------------------------------------

    def get_mcp_config(self) -> dict:
        """MCP 서버 설정을 가져온다.

        Returns:
            서버 이름을 설정에 매핑한 "mcp_servers" 키를 가진 dict.
            Gateway API ``McpConfigResponse`` 스키마와 일치한다.
        """
        config = get_extensions_config()
        return {"mcp_servers": {name: server.model_dump() for name, server in config.mcp_servers.items()}}

    def update_mcp_config(self, mcp_servers: dict[str, dict]) -> dict:
        """MCP 서버 설정을 갱신한다.

        extensions_config.json에 쓰고 캐시를 다시 로드한다.

        Args:
            mcp_servers: 서버 이름을 설정 dict에 매핑한 dict.
                각 값은 enabled, type, command, args, env, url 등의 키를 담는다.

        Returns:
            "mcp_servers" 키를 가진 dict. Gateway API ``McpConfigResponse``
            스키마와 일치한다.

        Raises:
            OSError: 설정 파일을 쓸 수 없을 때.
        """
        config_path = ExtensionsConfig.resolve_config_path()
        if config_path is None:
            raise FileNotFoundError("Cannot locate extensions_config.json. Set DEER_FLOW_EXTENSIONS_CONFIG_PATH or ensure it exists in the project root.")

        current_config = get_extensions_config()

        config_data = current_config.to_file_dict()
        config_data["mcpServers"] = mcp_servers

        self._atomic_write_json(config_path, config_data)

        self._agent = None
        self._agent_config_key = None
        reloaded = reload_extensions_config()
        return {"mcp_servers": {name: server.model_dump() for name, server in reloaded.mcp_servers.items()}}

    # ------------------------------------------------------------------
    # 공개 API — skill 관리
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> dict | None:
        """이름으로 특정 skill을 가져온다.

        Args:
            name: skill 이름.

        Returns:
            skill 정보 dict. 없으면 None.
        """
        storage = get_or_new_user_skill_storage(get_effective_user_id(), app_config=self._app_config)
        skill = next((s for s in storage.load_skills(enabled_only=False) if s.name == name), None)
        if skill is None:
            return None
        return {
            "name": skill.name,
            "description": skill.description,
            "license": skill.license,
            "category": skill.category,
            "enabled": skill.enabled,
        }

    def update_skill(self, name: str, *, enabled: bool) -> dict:
        """skill의 enabled 상태를 갱신한다.

        Args:
            name: skill 이름.
            enabled: 새 enabled 상태.

        Returns:
            갱신된 skill 정보 dict.

        Raises:
            ValueError: skill을 찾지 못했을 때.
            OSError: 설정 파일을 쓸 수 없을 때.
        """
        storage = get_or_new_user_skill_storage(get_effective_user_id(), app_config=self._app_config)
        skills = storage.load_skills(enabled_only=False)
        skill = next((s for s in skills if s.name == name), None)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found")

        # PUBLIC skill → 전역 extensions_config.json(공유 상태).
        # CUSTOM / LEGACY skill → 사용자별 _skill_states.json(격리된 상태).
        from deerflow.skills.types import SkillCategory

        if skill.category == SkillCategory.PUBLIC:
            config_path = ExtensionsConfig.resolve_config_path()
            if config_path is None:
                raise FileNotFoundError("Cannot locate extensions_config.json. Set DEER_FLOW_EXTENSIONS_CONFIG_PATH or ensure it exists in the project root.")

            from deerflow.skills.projection import skill_projection_mutation

            removal_names = (name,) if not enabled else ()
            with skill_projection_mutation(storage, "public", remove_names=removal_names):
                # projection lock은 프로세스 간에 걸리지만 singleton 캐시는 그렇지 않다.
                # 이 read-modify-write 전에 lock 안에서 디스크로부터 다시 읽는다.
                extensions_config = ExtensionsConfig.from_file(config_path)
                extensions_config.skills[name] = SkillStateConfig(enabled=enabled)

                config_data = extensions_config.to_file_dict()

                self._atomic_write_json(config_path, config_data)
                reload_extensions_config()
        else:
            # CUSTOM / LEGACY: 사용자별 상태를 쓴다
            from deerflow.skills.storage.user_scoped_skill_storage import UserScopedSkillStorage

            if isinstance(storage, UserScopedSkillStorage):
                storage.set_skill_enabled_state(name, enabled)
            else:
                # 사용자 범위가 아닌 storage용 fallback(실제로는 거의 없다)
                config_path = ExtensionsConfig.resolve_config_path()
                if config_path is None:
                    raise FileNotFoundError("Cannot locate extensions_config.json. Set DEER_FLOW_EXTENSIONS_CONFIG_PATH or ensure it exists in the project root.")
                extensions_config = get_extensions_config()
                extensions_config.skills[name] = SkillStateConfig(enabled=enabled)
                config_data = extensions_config.to_file_dict()
                self._atomic_write_json(config_path, config_data)
                reload_extensions_config()

        # 이 호출자의 prompt 캐시를 무효화한다. 바뀐 skill이 PUBLIC이면 상태가 공유되므로
        # 모든 사용자에 대해 무효화한다. ``routers/skills.py::update_skill``과 동일한 처리이며,
        # 이것이 없으면 캐시된 enabled 상태가 프로세스 재시작 전까지 낡은 채로 남는다.
        # PR #3889의 리뷰 피드백을 참고한다.
        try:
            from deerflow.agents.lead_agent.prompt import clear_skills_system_prompt_cache, invalidate_user_skill_cache

            skill_category_value = skill.category.value if hasattr(skill.category, "value") else skill.category
            if skill_category_value == SkillCategory.PUBLIC.value:
                clear_skills_system_prompt_cache()
            else:
                invalidate_user_skill_cache(get_effective_user_id())
        except Exception as exc:
            # 캐시 무효화 실패가 실제 쓰기 성공을 가리지 않게 한다. 로그만 남기고 계속 진행한다.
            # 낡은 캐시가 유지되는 구간은 다음 config reload로 제한된다.
            import logging

            logging.getLogger(__name__).warning("Failed to invalidate skills prompt cache after update_skill: %s", exc)

        self._agent = None
        self._agent_config_key = None

        updated = next((s for s in storage.load_skills(enabled_only=False) if s.name == name), None)
        if updated is None:
            raise RuntimeError(f"Skill '{name}' disappeared after update")
        return {
            "name": updated.name,
            "description": updated.description,
            "license": updated.license,
            "category": updated.category,
            "enabled": updated.enabled,
        }

    def install_skill(self, skill_path: str | Path) -> dict:
        """.skill 아카이브(ZIP)로부터 skill을 설치한다.

        Args:
            skill_path: .skill 파일 경로.

        Returns:
            success, skill_name, message를 담은 dict.

        Raises:
            FileNotFoundError: 파일이 없을 때.
            ValueError: 파일이 유효하지 않을 때.
        """
        return get_or_new_user_skill_storage(get_effective_user_id(), app_config=self._app_config).install_skill_from_archive(skill_path)

    # ------------------------------------------------------------------
    # 공개 API — memory 관리
    # ------------------------------------------------------------------

    def reload_memory(self) -> dict:
        """캐시를 무효화하며 파일에서 memory 데이터를 다시 로드한다.

        Returns:
            다시 로드한 memory 데이터 dict.

        reload 개념이 없는 backend(예: noop)는 ``get_memory``로 폴백한다. 둘 다 없는 backend
        (최소한의 ``add`` + ``get_context`` backend)는 ``NotImplementedError``를 던져서,
        호출자가 예외가 그대로 새어 나오는 대신 명확한 미지원 오류를 보게 한다.
        """
        from deerflow.agents.memory import get_memory_manager

        manager = get_memory_manager()
        user_id = get_effective_user_id()
        try:
            return manager.reload_memory(user_id=user_id)
        except NotImplementedError:
            pass  # reload 개념이 없다. 아래에서 현재 memory로 폴백한다
        try:
            return manager.get_memory(user_id=user_id)
        except NotImplementedError:
            raise NotImplementedError(f"reload_memory not supported by memory backend {type(manager).__name__}: implements neither reload_memory nor get_memory") from None

    def clear_memory(self) -> dict:
        """저장된 모든 memory 데이터를 지운다."""
        from deerflow.agents.memory import get_memory_manager

        return get_memory_manager().clear_memory(user_id=get_effective_user_id())

    def create_memory_fact(self, content: str, category: str = "context", confidence: float = 0.5) -> dict:
        """fact 하나를 수동으로 생성한다."""
        from deerflow.agents.memory import get_memory_manager

        manager = get_memory_manager()
        memory_data, fact_id = manager.create_fact(content=content, category=category, confidence=confidence, user_id=get_effective_user_id())
        if fact_id is None:
            raise ValueError("Fact was not stored because memory.max_facts kept higher-confidence facts")
        return memory_data

    def delete_memory_fact(self, fact_id: str) -> dict:
        """fact id로 memory에서 fact 하나를 삭제한다."""
        from deerflow.agents.memory import get_memory_manager

        manager = get_memory_manager()
        return manager.delete_fact(fact_id, user_id=get_effective_user_id())

    def update_memory_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        """fact 하나를 수동으로 갱신한다. 생략한 필드는 그대로 둔다."""
        from deerflow.agents.memory import get_memory_manager

        manager = get_memory_manager()
        return manager.update_fact(
            fact_id=fact_id,
            content=content,
            category=category,
            confidence=confidence,
            user_id=get_effective_user_id(),
        )

    def get_memory_config(self) -> dict:
        """memory 시스템 설정을 가져온다.

        Returns:
            memory 설정 dict.
        """
        from deerflow.config.memory_config import get_memory_config

        config = get_memory_config()
        return {
            "enabled": config.enabled,
            "mode": config.mode,
            "injection_enabled": config.injection_enabled,
            "shutdown_flush_timeout_seconds": config.shutdown_flush_timeout_seconds,
            "manager_class": config.manager_class,
            "backend_config": config.backend_config,
        }

    def get_memory_status(self) -> dict:
        """memory 상태(설정 + 현재 데이터)를 가져온다.

        Returns:
            "config"와 "data" 키를 가진 dict.
        """
        return {
            "config": self.get_memory_config(),
            "data": self.get_memory(),
        }

    # ------------------------------------------------------------------
    # 공개 API — 파일 업로드
    # ------------------------------------------------------------------

    def upload_files(self, thread_id: str, files: list[str | Path]) -> dict:
        """로컬 파일을 thread의 uploads 디렉터리로 업로드한다.

        PDF, PPT, Excel, Word 파일은 Markdown으로도 변환한다.

        Args:
            thread_id: 대상 thread ID.
            files: 업로드할 로컬 파일 경로 목록.

        Returns:
            success, files, message를 담은 dict. Gateway API ``UploadResponse``
            스키마와 일치한다.

        Raises:
            FileNotFoundError: 파일 중 하나라도 없을 때.
            ValueError: 주어진 경로가 존재하지만 일반 파일이 아닐 때.
        """
        validate_thread_id(thread_id)
        from deerflow.utils.file_conversion import CONVERTIBLE_EXTENSIONS, convert_file_to_markdown

        # 부분 업로드를 피하려고 모든 파일을 먼저 검증한다.
        resolved_files = []
        seen_names: set[str] = set()
        has_convertible_file = False
        for f in files:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {f}")
            if not p.is_file():
                raise ValueError(f"Path is not a file: {f}")
            dest_name = claim_unique_filename(p.name, seen_names)
            resolved_files.append((p, dest_name))
            if not has_convertible_file and p.suffix.lower() in CONVERTIBLE_EXTENSIONS:
                has_convertible_file = True

        uploads_dir = ensure_uploads_dir(thread_id)
        uploaded_files: list[dict] = []

        conversion_pool = None
        if has_convertible_file:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                conversion_pool = None
            else:
                import concurrent.futures

                # 이미 event loop 안이라면 worker 하나를 재사용해서, 변환 파일마다
                # 새 ThreadPoolExecutor를 만들지 않는다.
                conversion_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _convert_in_thread(path: Path, output_path: Path | None = None):
            return asyncio.run(convert_file_to_markdown(path, output_path=output_path))

        try:
            for src_path, dest_name in resolved_files:
                dest = uploads_dir / dest_name
                shutil.copy2(src_path, dest)

                info: dict[str, Any] = {
                    "filename": dest_name,
                    "size": dest.stat().st_size,
                    "path": str(dest),
                    "virtual_path": upload_virtual_path(dest_name),
                    "artifact_url": upload_artifact_url(thread_id, dest_name),
                }
                if dest_name != src_path.name:
                    info["original_filename"] = src_path.name

                if src_path.suffix.lower() in CONVERTIBLE_EXTENSIONS:
                    # 변환 전에 짝이 되는 .md 이름을 선점한다. 같은 .md로 합쳐지는 두 stem
                    # (또는 앞서 업로드된 .md)이 서로를 조용히 덮어쓰지 못하게 한다.
                    provisional_md_name = Path(dest_name).with_suffix(".md").name
                    unique_md_name = claim_unique_filename(provisional_md_name, seen_names)
                    md_output = dest.with_name(unique_md_name)
                    try:
                        if conversion_pool is not None:
                            md_path = conversion_pool.submit(_convert_in_thread, dest, md_output).result()
                        else:
                            md_path = asyncio.run(convert_file_to_markdown(dest, output_path=md_output))
                    except Exception:
                        logger.warning(
                            "Failed to convert %s to markdown",
                            src_path.name,
                            exc_info=True,
                        )
                        md_path = None

                    if md_path is not None:
                        info["markdown_file"] = md_path.name
                        info["markdown_path"] = str(uploads_dir / md_path.name)
                        info["markdown_virtual_path"] = upload_virtual_path(md_path.name)
                        info["markdown_artifact_url"] = upload_artifact_url(thread_id, md_path.name)
                    else:
                        # 변환이 실패해 아무것도 쓰지 않았으므로 선점을 푼다. 계속 붙들고 있으면
                        # 이후 같은 stem의 업로드가 아무도 쓰지 않는 이름을 피해 개명된다.
                        seen_names.discard(unique_md_name)

                uploaded_files.append(info)
        finally:
            if conversion_pool is not None:
                conversion_pool.shutdown(wait=True)

        return {
            "success": True,
            "files": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
        }

    def list_uploads(self, thread_id: str) -> dict:
        """thread의 uploads 디렉터리에 있는 파일을 나열한다.

        Args:
            thread_id: Thread ID.

        Returns:
            "files"와 "count" 키를 가진 dict. Gateway API ``list_uploaded_files``
            응답과 일치한다.
        """
        validate_thread_id(thread_id)
        uploads_dir = get_uploads_dir(thread_id)
        result = list_files_in_dir(uploads_dir)
        return enrich_file_listing(result, thread_id)

    def delete_upload(self, thread_id: str, filename: str) -> dict:
        """thread의 uploads 디렉터리에서 파일을 삭제한다.

        Args:
            thread_id: Thread ID.
            filename: 삭제할 파일 이름.

        Returns:
            success와 message를 담은 dict. Gateway API ``delete_uploaded_file``
            응답과 일치한다.

        Raises:
            FileNotFoundError: 파일이 없을 때.
            PermissionError: path traversal이 감지됐을 때.
        """
        validate_thread_id(thread_id)
        from deerflow.utils.file_conversion import CONVERTIBLE_EXTENSIONS

        uploads_dir = get_uploads_dir(thread_id)
        return delete_file_safe(uploads_dir, filename, convertible_extensions=CONVERTIBLE_EXTENSIONS)

    # ------------------------------------------------------------------
    # 공개 API — artifacts
    # ------------------------------------------------------------------

    def get_artifact(self, thread_id: str, path: str) -> tuple[bytes, str]:
        """agent가 만든 artifact 파일을 읽는다.

        Args:
            thread_id: Thread ID.
            path: 가상 경로(예: "mnt/user-data/outputs/file.txt").

        Returns:
            (file_bytes, mime_type) 튜플.

        Raises:
            FileNotFoundError: artifact가 없을 때.
            ValueError: 경로가 유효하지 않을 때.
        """
        validate_thread_id(thread_id)
        try:
            actual = get_paths().resolve_virtual_path(thread_id, path, user_id=get_effective_user_id())
        except ValueError as exc:
            if "traversal" in str(exc):
                from deerflow.uploads.manager import PathTraversalError

                raise PathTraversalError("Path traversal detected") from exc
            raise
        if not actual.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        if not actual.is_file():
            raise ValueError(f"Path is not a file: {path}")

        mime_type, _ = mimetypes.guess_type(actual)
        return actual.read_bytes(), mime_type or "application/octet-stream"
