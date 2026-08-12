"""skill의 ``allowed-tools``를 lead-agent context에서 활성화된 skill에만 적용한다."""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.runtime.secret_context import SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY, read_slash_skill_source_path
from deerflow.skills.storage import get_or_new_skill_storage, get_or_new_user_skill_storage
from deerflow.skills.tool_policy import ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES, allowed_tool_names_for_skills
from deerflow.skills.types import Skill

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.skills.storage.skill_storage import SkillStorage

logger = logging.getLogger(__name__)

_POLICY_DECISION_VERSION = 2
_POLICY_SOURCE_PASSIVE = "passive"
_POLICY_SOURCE_SLASH = "slash"
_POLICY_SOURCE_SKILL_CONTEXT = "skill_context"
_POLICY_SOURCES = frozenset({_POLICY_SOURCE_PASSIVE, _POLICY_SOURCE_SLASH, _POLICY_SOURCE_SKILL_CONTEXT})
_MISSING_POLICY_DECISION = object()
_TOOL_SEARCH_NAME = "tool_search"

type _PolicySignature = tuple[str, tuple[str, ...]]


class SkillToolPolicyMiddleware(AgentMiddleware[AgentState]):
    """agent의 도구를 slash 또는 in-context skill이 선언한 범위로 제한한다.

    skill을 enabled로 두는 것만으로는 탐색 가능해질 뿐 권한 정책이 활성화되지는 않는다. skill은 사용자가
    run에서 slash로 활성화하거나 모델이 ``skill_context``에 로드했을 때 정책상 활성 상태가 된다.
    명시적 slash 활성화는 남은 run 동안 우선한다. 두 번째 skill을 수동적으로 읽어도 slash skill의 권한을
    넓힐 수 없다.
    """

    def __init__(
        self,
        *,
        available_skills: set[str] | None = None,
        app_config: AppConfig | None = None,
        user_id: str | None = None,
        slash_source_owner_token: str,
    ) -> None:
        super().__init__()
        if not isinstance(slash_source_owner_token, str) or not slash_source_owner_token:
            raise ValueError("slash_source_owner_token must be a non-empty string")
        self._available_skills = set(available_skills) if available_skills is not None else None
        self._app_config = app_config
        self._user_id = user_id
        self._slash_source_owner_token = slash_source_owner_token
        self._decision_owner_token = secrets.token_urlsafe(24)

    def _storage(self) -> SkillStorage:
        if self._user_id is not None:
            return get_or_new_user_skill_storage(self._user_id, app_config=self._app_config)
        if self._app_config is not None:
            return get_or_new_skill_storage(app_config=self._app_config)
        return get_or_new_skill_storage()

    def _active_policy(self, request: ModelRequest | ToolCallRequest) -> _PolicySignature:
        context = getattr(getattr(request, "runtime", None), "context", None)
        slash_path = read_slash_skill_source_path(context, owner_token=self._slash_source_owner_token)
        if slash_path is not None:
            return _POLICY_SOURCE_SLASH, (slash_path,)

        paths: list[str] = []
        state = getattr(request, "state", None)
        if state is None:
            state = {}
        if isinstance(state, Mapping):
            entries = state.get("skill_context") or []
        elif hasattr(state, "skill_context"):
            entries = getattr(state, "skill_context") or []
        else:
            logger.warning("Unsupported agent state shape for skill tool policy: %s", type(state).__name__)
            entries = []
        if not isinstance(entries, (list, tuple)):
            logger.warning("Invalid skill_context shape for skill tool policy: %s", type(entries).__name__)
            entries = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
        if paths:
            return _POLICY_SOURCE_SKILL_CONTEXT, tuple(paths)
        return _POLICY_SOURCE_PASSIVE, ()

    def _active_skills_for_paths(self, paths: tuple[str, ...]) -> tuple[list[Skill], bool]:
        if not paths:
            return [], False

        try:
            storage = self._storage()
            skills = storage.load_skills(enabled_only=False)
            container_root = storage.get_container_root()
        except Exception:
            logger.exception("Failed to load active skills for allowed-tools policy")
            # 실제 활성 참조가 있는데 인가할 수 없는 상태다. 정책 실패를 알려 호출자가
            # framework-safe 도구만 유지하도록 한다.
            return [], True

        registry = {posixpath.normpath(skill.get_container_file_path(container_root)): skill for skill in skills}
        active: list[Skill] = []
        seen: set[str] = set()
        for path in paths:
            skill = registry.get(posixpath.normpath(path))
            if skill is None:
                logger.warning("Active skill path could not be resolved for allowed-tools policy: %s", path)
                continue
            if not skill.enabled:
                logger.warning("Active skill is disabled for allowed-tools policy: %s", path)
                continue
            if self._available_skills is not None and skill.name not in self._available_skills:
                logger.warning("Active skill is outside the agent allowlist for allowed-tools policy: %s", path)
                continue
            if skill.name in seen:
                continue
            seen.add(skill.name)
            active.append(skill)
        if not active:
            logger.warning("No active skill references could be authorized for allowed-tools policy; failing closed")
            return [], True
        return active, False

    def _allowed_names_for_paths(self, paths: tuple[str, ...]) -> set[str] | None:
        active_skills, policy_failed = self._active_skills_for_paths(paths)
        if policy_failed:
            return set(ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES)
        allowed = allowed_tool_names_for_skills(active_skills)
        if allowed is None:
            return None
        return allowed | set(ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES)

    @staticmethod
    def _runtime_context(request: ModelRequest | ToolCallRequest) -> dict | None:
        context = getattr(getattr(request, "runtime", None), "context", None)
        return context if isinstance(context, dict) else None

    def _store_policy_decision(self, request: ModelRequest, policy: _PolicySignature, allowed: set[str] | None) -> None:
        context = self._runtime_context(request)
        if context is not None:
            source, paths = policy
            context[SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY] = {
                "version": _POLICY_DECISION_VERSION,
                "owner_token": self._decision_owner_token,
                "source": source,
                "active_paths": list(paths),
                "allowed_names": None if allowed is None else sorted(allowed),
            }

    def _read_policy_decision(self, context: dict | None, policy: _PolicySignature) -> set[str] | None | object:
        if context is None:
            return _MISSING_POLICY_DECISION
        decision = context.get(SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY)
        if not isinstance(decision, dict):
            return _MISSING_POLICY_DECISION
        if type(decision.get("version")) is not int or decision["version"] != _POLICY_DECISION_VERSION:
            return _MISSING_POLICY_DECISION
        if not isinstance(decision.get("owner_token"), str) or decision["owner_token"] != self._decision_owner_token:
            return _MISSING_POLICY_DECISION
        source, paths = policy
        stored_source = decision.get("source")
        if not isinstance(stored_source, str) or stored_source not in _POLICY_SOURCES or stored_source != source:
            return _MISSING_POLICY_DECISION
        stored_paths = decision.get("active_paths")
        if not isinstance(stored_paths, list) or not all(isinstance(path, str) for path in stored_paths) or tuple(stored_paths) != paths:
            return _MISSING_POLICY_DECISION
        allowed = decision.get("allowed_names")
        if allowed is None:
            return None
        if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
            return _MISSING_POLICY_DECISION
        return set(allowed)

    def _allowed_names(
        self,
        request: ModelRequest | ToolCallRequest,
        *,
        policy: _PolicySignature | None = None,
    ) -> set[str] | None:
        resolved_policy = self._active_policy(request) if policy is None else policy
        _, paths = resolved_policy
        context = self._runtime_context(request)
        decision = self._read_policy_decision(context, resolved_policy)
        if decision is not _MISSING_POLICY_DECISION:
            return decision
        return self._allowed_names_for_paths(paths)

    def _filter_model_request(
        self,
        request: ModelRequest,
        *,
        policy: _PolicySignature | None = None,
        refresh_decision: bool = False,
    ) -> ModelRequest:
        resolved_policy = self._active_policy(request) if policy is None else policy
        _, paths = resolved_policy
        allowed = self._allowed_names_for_paths(paths) if refresh_decision else self._allowed_names(request, policy=resolved_policy)
        if refresh_decision:
            self._store_policy_decision(request, resolved_policy, allowed)
        if allowed is None:
            return request
        tools = [tool for tool in request.tools if getattr(tool, "name", None) in allowed]
        if len(tools) < len(request.tools):
            logger.debug("Skill policy filtered %d lead tool schema(s)", len(request.tools) - len(tools))
        return request.override(tools=tools)

    def _blocked_tool_message(
        self,
        request: ToolCallRequest,
        *,
        allowed: set[str] | None,
    ) -> ToolMessage | None:
        name = str(request.tool_call.get("name") or "")
        if allowed is None or not name or name in allowed:
            return None
        return ToolMessage(
            content=f"Error: Tool '{name}' is not allowed by the active skill policy.",
            tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
            name=name,
            status="error",
        )

    @staticmethod
    def _tool_search_policy_error(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content="Error: tool_search returned a result that could not be validated against the active skill policy.",
            tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
            name=_TOOL_SEARCH_NAME,
            status="error",
        )

    def _filter_tool_search_result(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
        *,
        allowed: set[str] | None,
    ) -> ToolMessage | Command:
        """tool_search 출력에서 거부된 schema와 promotion을 제거한다.

        tool_search를 계속 노출해도 안전한 것은, 활성 정책이 제거한 도구의 전체 schema를 반환할 수 없을
        때뿐이다. 허용된 schema가 언제 모델에 보이게 되는지는 여전히 deferred filtering이 통제하며,
        이 메서드는 탐색 결과 자체를 같은 인가 경계 안에 묶어 둔다.
        """
        name = str(request.tool_call.get("name") or "")
        if name != _TOOL_SEARCH_NAME or allowed is None:
            return result
        if not isinstance(result, Command) or not isinstance(result.update, dict):
            logger.warning("Active-policy tool_search returned an unsupported result shape")
            return self._tool_search_policy_error(request)

        promoted = result.update.get("promoted")
        messages = result.update.get("messages")
        if not isinstance(promoted, dict) or not isinstance(messages, list) or len(messages) != 1:
            logger.warning("Active-policy tool_search command omitted promoted/messages updates")
            return self._tool_search_policy_error(request)
        raw_names = promoted.get("names")
        if not isinstance(raw_names, list) or not all(isinstance(item, str) for item in raw_names):
            logger.warning("Active-policy tool_search returned malformed promoted names")
            return self._tool_search_policy_error(request)

        permitted_names = [item for item in raw_names if item in allowed]
        sanitized_messages: list[ToolMessage] = []
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name != _TOOL_SEARCH_NAME:
                logger.warning("Active-policy tool_search returned an unexpected message shape")
                return self._tool_search_policy_error(request)
            content = message.content
            if raw_names:
                try:
                    schemas = json.loads(content) if isinstance(content, str) else None
                except json.JSONDecodeError:
                    schemas = None
                if not isinstance(schemas, list):
                    logger.warning("Active-policy tool_search returned schemas that could not be filtered")
                    return self._tool_search_policy_error(request)
                filtered_schemas = [schema for schema in schemas if isinstance(schema, dict) and (schema.get("name") in permitted_names or (isinstance(schema.get("function"), dict) and schema["function"].get("name") in permitted_names))]
                content = json.dumps(filtered_schemas, indent=2, ensure_ascii=False) if filtered_schemas else "No tools found matching the active skill policy."
            sanitized_messages.append(message.model_copy(update={"content": content}))

        sanitized_update = dict(result.update)
        sanitized_update["promoted"] = {**promoted, "names": permitted_names}
        sanitized_update["messages"] = sanitized_messages
        return Command(
            graph=result.graph,
            update=sanitized_update,
            resume=result.resume,
            goto=result.goto,
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        policy = self._active_policy(request)
        return handler(self._filter_model_request(request, policy=policy, refresh_decision=True))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        policy = self._active_policy(request)
        _, paths = policy
        if not paths:
            self._store_policy_decision(request, policy, None)
            return await handler(request)
        filtered = await asyncio.to_thread(
            self._filter_model_request,
            request,
            policy=policy,
            refresh_decision=True,
        )
        return await handler(filtered)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        policy = self._active_policy(request)
        if not policy[1]:
            return handler(request)
        allowed = self._allowed_names(request, policy=policy)
        blocked = self._blocked_tool_message(request, allowed=allowed)
        if blocked is not None:
            return blocked
        return self._filter_tool_search_result(request, handler(request), allowed=allowed)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        policy = self._active_policy(request)
        if not policy[1]:
            return await handler(request)
        allowed = await asyncio.to_thread(self._allowed_names, request, policy=policy)
        blocked = self._blocked_tool_message(request, allowed=allowed)
        if blocked is not None:
            return blocked
        return self._filter_tool_search_result(request, await handler(request), allowed=allowed)
