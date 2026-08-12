"""도구 오류 처리 middleware와 공용 runtime middleware 빌더."""

import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.skill_context import (
    SKILL_CONTEXT_ENTRY_KEY,
    _tool_call_path,
    build_skill_entry_metadata_from_read,
)
from deerflow.agents.middlewares.tool_result_meta import (
    normalize_tool_result,
    stamp_exception_meta,
)
from deerflow.config.app_config import AppConfig
from deerflow.config.summarization_config import DEFAULT_SKILL_FILE_READ_TOOL_NAMES
from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.subagents.status_contract import (
    format_subagent_result_message,
    make_subagent_additional_kwargs,
)

if TYPE_CHECKING:
    from deerflow.tools.builtins.tool_search import DeferredToolSetup

logger = logging.getLogger(__name__)

_MISSING_TOOL_CALL_ID = "missing_tool_call_id"
_TASK_TOOL_NAME = "task"
_RECOVERY_HINT = "Continue with available context, or choose an alternative tool."


def _stamp_task_exception_status(message: ToolMessage, *, tool_name: str, error: str) -> ToolMessage:
    """여기서 만든 task 예외 wrapper에 failed 메타데이터를 찍는다."""
    if tool_name != _TASK_TOOL_NAME:
        return message
    content, metadata_error = format_subagent_result_message("failed", error=error)
    if not content.endswith((".", "!", "?")):
        content += "."
    message.content = f"{content} {_RECOVERY_HINT}"
    existing = dict(message.additional_kwargs or {})
    existing.update(make_subagent_additional_kwargs("failed", error=metadata_error))
    message.additional_kwargs = existing
    return message


class ToolErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """도구 예외를 error ToolMessage로 변환해 run이 계속 진행되게 한다."""

    def __init__(self, *, app_config: AppConfig | None = None) -> None:
        super().__init__()
        self._app_config = app_config
        if app_config is None:
            self._skill_read_tool_names = frozenset(DEFAULT_SKILL_FILE_READ_TOOL_NAMES)
            self._skills_root = DEFAULT_SKILLS_CONTAINER_PATH
        else:
            self._skill_read_tool_names = frozenset(app_config.summarization.skill_file_read_tool_names)
            self._skills_root = app_config.skills.container_path

    def _build_error_message(self, request: ToolCallRequest, exc: Exception) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "unknown_tool")
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
        detail = str(exc).strip() or exc.__class__.__name__
        if len(detail) > 500:
            detail = detail[:497] + "..."

        content = f"Error: Tool '{tool_name}' failed with {exc.__class__.__name__}: {detail}. {_RECOVERY_HINT}"
        message = ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )
        # 예외 wrapper는 이 middleware가 생산자다. 그래서 task_tool이 자체 Command를
        # 만들기 전에 발생한 task 실패도 동일한 구조화 메타데이터를 갖는다.
        structured_error = f"{exc.__class__.__name__}: {detail}"
        message = _stamp_task_exception_status(message, tool_name=tool_name, error=structured_error)
        return stamp_exception_meta(message, structured_error)

    def _stamp_skill_read_metadata(
        self,
        message: ToolMessage,
        request: ToolCallRequest,
        *,
        tool_name: str,
    ) -> ToolMessage:
        if tool_name not in self._skill_read_tool_names:
            return message
        if getattr(message, "status", "success") == "error":
            return message
        content = message.content if isinstance(message.content, str) else None
        if content is None:
            return message
        path = _tool_call_path(request.tool_call)
        if path is None:
            return message
        entry = build_skill_entry_metadata_from_read(path, content, skills_root=self._skills_root)
        if entry is None:
            return message
        existing = dict(message.additional_kwargs or {})
        existing[SKILL_CONTEXT_ENTRY_KEY] = dict(entry)
        message.additional_kwargs = existing
        return message

    def _maybe_stamp(self, result: ToolMessage | Command, request: ToolCallRequest) -> ToolMessage | Command:
        """생산자가 붙여야 하는 메타데이터를 해당 도구 결과에 적용한다."""
        if not isinstance(result, ToolMessage):
            return result
        tool_name = str(request.tool_call.get("name") or "")
        return self._stamp_skill_read_metadata(result, request, tool_name=tool_name)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            result = handler(request)
        except GraphBubbleUp:
            # LangGraph 제어 흐름 시그널(interrupt/pause/resume)은 그대로 전파한다.
            raise
        except Exception as exc:
            logger.exception("Tool execution failed (sync): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))
            return self._build_error_message(request, exc)
        return normalize_tool_result(self._maybe_stamp(result, request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            result = await handler(request)
        except GraphBubbleUp:
            # LangGraph 제어 흐름 시그널(interrupt/pause/resume)은 그대로 전파한다.
            raise
        except Exception as exc:
            logger.exception("Tool execution failed (async): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))
            return self._build_error_message(request, exc)
        return normalize_tool_result(self._maybe_stamp(result, request))


def _build_runtime_middlewares(
    *,
    app_config: AppConfig,
    include_uploads: bool,
    include_dangling_tool_call_patch: bool,
    lazy_init: bool = True,
    authorization_provider=None,
    authorization_infrastructure_tool_names: frozenset[str] = frozenset(),
) -> list[AgentMiddleware]:
    """에이전트 실행에 공통으로 쓰이는 base middleware를 구성한다."""
    from deerflow.agents.middlewares.input_sanitization_middleware import InputSanitizationMiddleware
    from deerflow.agents.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
    from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
    from deerflow.agents.middlewares.tool_output_budget_middleware import ToolOutputBudgetMiddleware
    from deerflow.agents.middlewares.tool_result_sanitization_middleware import ToolResultSanitizationMiddleware
    from deerflow.sandbox.middleware import SandboxMiddleware

    # Layer 1 — 최외곽 wrap_model_call wrapper(바깥→안쪽 순서로 나열).
    # InputSanitizationMiddleware를 맨 앞에 둬서 최외곽 wrapper가 되게 한다.
    # 안쪽 middleware는 모두 sanitize된 메시지만 본다.
    # ToolResultSanitizationMiddleware는 신뢰할 수 없는 콘텐츠의 다른 유입 지점,
    # 즉 원격 도구 결과(web_fetch / web_search)에 같은 framework/injection 태그
    # 무력화를 적용한다. ToolOutputBudgetMiddleware보다 안쪽(뒤에 나열)이라
    # 원본 도구 출력을 먼저 무력화하고, budget wrapper가 무력화된 텍스트를 자른다.
    outer_wrappers: list[AgentMiddleware] = [
        InputSanitizationMiddleware(),
        ToolOutputBudgetMiddleware.from_app_config(app_config),
        ToolResultSanitizationMiddleware(),
    ]

    # Layer 2 — thread 범위 데이터를 읽거나 덧붙이는 before_agent hook.
    thread_hooks: list[AgentMiddleware] = [
        ThreadDataMiddleware(lazy_init=lazy_init),
    ]
    if include_uploads:
        from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware

        thread_hooks.append(UploadsMiddleware())
    thread_hooks.append(SandboxMiddleware(lazy_init=lazy_init))

    # Layer 3 — 뒤에 덧붙이기만 하는 후처리 middleware.
    tail: list[AgentMiddleware] = []
    if include_dangling_tool_call_patch:
        from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware

        tail.append(DanglingToolCallMiddleware())
    tail.append(LLMErrorHandlingMiddleware(app_config=app_config))

    # Authorization은 기존 GuardrailMiddleware를 재사용한다. 실행 시점 deny, audit,
    # fail-closed 처리를 검증된 구현 한 곳에 모아 두기 위해서다.
    # 명시적 guardrail provider보다 앞에 붙이므로 authorization이 바깥쪽 guard가 되고,
    # 이미 거부된 도구에 대해 불필요한 외부 정책 호출을 하지 않는다.
    authorization_config = app_config.authorization
    if authorization_config.enabled is True:
        if authorization_provider is None:
            from deerflow.authz.runtime import resolve_authorization_provider

            authorization_provider = resolve_authorization_provider(authorization_config)
        if authorization_provider is not None:
            from deerflow.authz.adapter import GuardrailAuthorizationAdapter
            from deerflow.guardrails.middleware import GuardrailMiddleware

            tail.append(
                GuardrailMiddleware(
                    GuardrailAuthorizationAdapter(
                        authorization_provider,
                        default_role=authorization_config.default_role,
                        infrastructure_tool_names=authorization_infrastructure_tool_names,
                    ),
                    fail_closed=authorization_config.fail_closed,
                )
            )

    # 명시적 guardrail middleware는 설정되어 있으면 독립적으로 계속 동작한다.
    guardrails_config = app_config.guardrails
    if guardrails_config.enabled and guardrails_config.provider:
        import inspect

        from deerflow.guardrails.middleware import GuardrailMiddleware
        from deerflow.reflection import resolve_variable

        provider_cls = resolve_variable(guardrails_config.provider.use)
        provider_kwargs = dict(guardrails_config.provider.config) if guardrails_config.provider.config else {}
        # provider가 받아들이면 framework 힌트를 넘긴다(예: config 탐색 용도).
        # AllowlistProvider 같은 내장 provider는 필요 없으므로 생성자가 'framework' 또는
        # '**kwargs'를 받을 때만 주입한다.
        if "framework" not in provider_kwargs:
            try:
                sig = inspect.signature(provider_cls.__init__)
                if "framework" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    provider_kwargs["framework"] = "deerflow"
            except (ValueError, TypeError):
                pass
        provider = provider_cls(**provider_kwargs)
        tail.append(GuardrailMiddleware(provider, fail_closed=guardrails_config.fail_closed, passport=guardrails_config.passport))

    from deerflow.agents.middlewares.sandbox_audit_middleware import SandboxAuditMiddleware

    tail.append(SandboxAuditMiddleware())

    # ReadBeforeWriteMiddleware는 최외곽 쓰기 gate다. 모델이 현재 버전을 읽지 않은
    # 파일에 대한 쓰기를 막는다. ToolProgress와 ToolErrorHandling보다 바깥에 있어야
    # 차단된 쓰기가 ToolProgress 슬롯을 소모하지 않고 즉시 반환된다. 차단된
    # ToolMessage에 직접 deerflow_tool_meta를 찍어 하위 호출자가 형식이 온전한
    # 결과를 받게 한다.
    if app_config.read_before_write.enabled:
        from deerflow.agents.middlewares.read_before_write_middleware import ReadBeforeWriteMiddleware

        tail.append(ReadBeforeWriteMiddleware())

    # ToolProgressMiddleware는 바깥쪽(더 앞 index)이어야 한다. 그래야 wrap_tool_call
    # handler 체인에 안쪽의 ToolErrorHandlingMiddleware가 포함되고, 이쪽이 모든 결과에
    # deerflow_tool_meta를 찍은 뒤 ToolProgressMiddleware가 _update_state_from_result에서
    # 그것을 읽는다.
    # 프레임워크 규칙: 리스트의 첫 항목이 최외곽이다(types.py: "compose with first in list as outermost layer").
    tool_progress_config = app_config.tool_progress
    if tool_progress_config.enabled:
        from deerflow.agents.middlewares.tool_progress_middleware import ToolProgressMiddleware

        tail.append(ToolProgressMiddleware.from_config(tool_progress_config))

    tail.append(ToolErrorHandlingMiddleware(app_config=app_config))

    middlewares = [*outer_wrappers, *thread_hooks, *tail]

    # 순서 불변식은 deerflow.extensions.ordering에 선언되어 있고, extension 기여분이
    # 병합된 뒤 조립 빌더 끝에서 한 번 검증된다. 그러지 않으면 어떤 기여분이 이 빌더가
    # 이미 확인한 불변식을 조용히 뒤집을 수 있다.
    return middlewares


def build_lead_runtime_middlewares(
    *,
    app_config: AppConfig,
    lazy_init: bool = True,
    authorization_provider=None,
    deferred_setup: "DeferredToolSetup | None" = None,
) -> list[AgentMiddleware]:
    """lead 전용 middleware 앞에 오는, lead agent runtime 공용 middleware."""
    return _build_runtime_middlewares(
        app_config=app_config,
        include_uploads=True,
        include_dangling_tool_call_patch=True,
        lazy_init=lazy_init,
        authorization_provider=authorization_provider,
        authorization_infrastructure_tool_names=(frozenset({deferred_setup.tool_search_tool.name}) if authorization_provider is not None and deferred_setup is not None and deferred_setup.tool_search_tool is not None else frozenset()),
    )


def build_subagent_runtime_middlewares(
    *,
    app_config: AppConfig | None = None,
    model_name: str | None = None,
    lazy_init: bool = True,
    deferred_setup: "DeferredToolSetup | None" = None,
    mcp_routing_middleware: AgentMiddleware | None = None,
    agent_name: str | None = None,
    available_skills: set[str] | None = None,
    user_id: str | None = None,
    authorization_provider=None,
    extensions=None,
) -> list[AgentMiddleware]:
    """subagent 전용 middleware 앞에 오는, subagent runtime 공용 middleware."""
    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()

    middlewares = _build_runtime_middlewares(
        app_config=app_config,
        include_uploads=False,
        include_dangling_tool_call_patch=True,
        lazy_init=lazy_init,
        authorization_provider=authorization_provider,
        authorization_infrastructure_tool_names=(frozenset({deferred_setup.tool_search_tool.name}) if authorization_provider is not None and deferred_setup is not None and deferred_setup.tool_search_tool is not None else frozenset()),
    )

    # 활성화/설정된 skill은 탐색용 메타데이터일 뿐 자동으로 부여되는 권한이 아니다.
    # lead agent의 activation + policy 쌍을 그대로 두어, slash 명령이나 SKILL.md 읽기
    # 완료로 해당 allowed-tools 선언이 활성화되기 전까지 subagent가 평소 도구 집합을
    # 유지하게 한다.
    from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware
    from deerflow.agents.middlewares.skill_tool_policy_middleware import SkillToolPolicyMiddleware

    slash_source_owner_token = secrets.token_urlsafe(24)
    middlewares.append(
        SkillActivationMiddleware(
            available_skills=available_skills,
            app_config=app_config,
            user_id=user_id,
            slash_source_owner_token=slash_source_owner_token,
        )
    )
    middlewares.append(
        SkillToolPolicyMiddleware(
            available_skills=available_skills,
            app_config=app_config,
            user_id=user_id,
            slash_source_owner_token=slash_source_owner_token,
        )
    )

    if model_name is None and app_config.models:
        model_name = app_config.models[0].name

    model_config = app_config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

        middlewares.append(ViewImageMiddleware())

    if mcp_routing_middleware is not None:
        middlewares.append(mcp_routing_middleware)

    # tool_search가 승격하기 전까지 deferred(MCP) 도구 스키마를 subagent 모델 바인딩에서
    # 숨긴다. lead agent와 동일한 배선이다. deferred 집합과 catalog hash는 빌드 시점
    # setup(도구 정책 필터링 이후 조립)에서 오고, 승격 여부는 graph state에서 읽는다.
    # setup이 비어 있거나 None이면(deferral 비활성 또는 살아남은 MCP 도구 없음) 순수 no-op이다.
    if deferred_setup is not None and deferred_setup.deferred_names:
        from deerflow.agents.middlewares.deferred_tool_filter_middleware import DeferredToolFilterMiddleware

        middlewares.append(DeferredToolFilterMiddleware(deferred_setup.deferred_names, deferred_setup.catalog_hash))
        from deerflow.agents.middlewares.mcp_routing_middleware import assert_mcp_routing_before_deferred_filter

        assert_mcp_routing_before_deferred_filter(middlewares)

    # LoopDetectionMiddleware — 현재 subagent는 lead의 폭주 방지 guard를 하나도
    # 물려받지 않는다(#3875 참고). loop 감지가 없으면 망가진 subagent 도구 루프가
    # ``max_turns``까지 그대로 돌면서 매 턴 커지는 context를 재전송한다(보고된 4.4M 토큰 소모).
    # lead 체인을 그대로 따라 루프를 감지하고 끊는다. subagent는 ``task``를 쓸 수 없으므로
    # 여기서는 tool-loop 휴리스틱만 발동한다. 재귀 위임 경로가 없어 오탐 여지도 없다.
    # SafetyFinishReasonMiddleware보다 먼저 등록한다(리스트에서 더 앞).
    # LangChain은 after_model hook을 등록 역순으로 실행하므로 아래에서 추가되는
    # SafetyFinishReasonMiddleware가 먼저 돌아 safety로 종료된 tool_calls를 제거하고,
    # 그다음 LoopDetectionMiddleware가 정리된 메시지를 집계한다. 이것이
    # SafetyFinishReasonMiddleware docstring이 요구하는 배치("register after LoopDetection")이며
    # lead 체인(``lead_agent/agent.py``)과 같다. #3875 Phase 1이고, lead가 볼 수 있는
    # stop reason을 갖는 결정적 turn/token 예산은 Phase 2다.
    loop_detection_config = app_config.loop_detection
    if loop_detection_config.enabled:
        from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware

        middlewares.append(LoopDetectionMiddleware.from_config(loop_detection_config))

    # TokenBudgetMiddleware — 현재 subagent는 lead의 비용 방어선도 물려받지 않는다
    # (#3875 Phase 2). 망가진 subagent가 max_turns/timeout이 걸리기 전에 병적인 양의
    # 토큰을 태울 수 있다(보고된 4.4M run). lead 체인을 따라 run 단위 예산 hard-stop을
    # 적용한다. ``subagents.token_budget``은 기본 활성이고,
    # ``subagents.agents.<name>.token_budget``으로 에이전트별 override가 가능하다.
    # hard-stop은 예외를 던지지 않고 tool_calls만 제거해 run이 최종 답변으로 끝나게 하며,
    # executor가 ``consume_stop_reason``을 읽어 완료된 결과를 lead에게 ``token_capped``로
    # 표시한다. 상태는 run_id로 구분되고 task run마다 새 middleware 인스턴스를 만들므로
    # (``executor._create_agent`` 참고), context에서 부모 thread_id/run_id를 공유해도
    # 병렬 subagent끼리 서로 오염시키지 않는다.
    #
    # 기본 상한 연동(#3875 Phase 3 리뷰): 기본 ``max_tokens``는 ``summarization.enabled``와
    # 다시 연동된다. compaction이 켜져 있으면 1M, 꺼져 있으면 2M이다. 이는 기본값에만
    # 적용되고 사용자가 지정한 예산(전역이든 에이전트별이든)이 항상 이긴다. 따라서 값을
    # 고정해 둔 배포는 summarization 스위치를 바꿔도 조용히 바뀌지 않는다.
    summarization_enabled = app_config.summarization.enabled
    if agent_name is not None:
        token_budget_config = app_config.subagents.get_token_budget_for(agent_name, summarization_enabled=summarization_enabled)
    else:
        token_budget_config = app_config.subagents.token_budget
    if token_budget_config.enabled:
        from deerflow.agents.middlewares.token_budget_middleware import TokenBudgetMiddleware

        middlewares.append(TokenBudgetMiddleware.from_config(token_budget_config))

    from deerflow.agents.middlewares.configured_extensions import load_configured_extension_middlewares

    middlewares.extend(load_configured_extension_middlewares(app_config))

    # lead agent가 쓰는 것과 동일한 provider safety 종료 guard다. subagent도
    # finish_reason=content_filter(및 유사 사유)와 함께 잘린 tool_calls에 똑같이 노출되고,
    # 그 잘못된 호출은 task 도구 결과를 통해 lead agent로 되돌아간다.
    safety_config = app_config.safety_finish_reason
    if safety_config.enabled:
        from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware

        middlewares.append(SafetyFinishReasonMiddleware.from_config(safety_config))

    # DurableContextMiddleware (#4039) — summarization은 압축된 히스토리를 ``messages``에
    # summary 메시지로 되쓰지 않고 ``summary_text`` state 채널에 저장한다. lead 체인을 따라
    # subagent도 그 summary를 이후 모델 요청에 투영한다. 그러지 않으면 메시지 개수 기반 keep
    # 정책이 앞선 사용자 context 없이 assistant tool-call + tool-result 꼬리만 남길 수 있고,
    # 엄격한 provider는 이를 거부한다. 같은 middleware가 원본 읽기 결과가 압축돼도 skill
    # 참조를 지속시킨다.
    from deerflow.agents.middlewares.durable_context_middleware import DurableContextMiddleware

    middlewares.append(
        DurableContextMiddleware(
            skills_container_path=app_config.skills.container_path,
            skill_file_read_tool_names=app_config.summarization.skill_file_read_tool_names,
        )
    )

    # DeerFlowSummarizationMiddleware — 현재 subagent는 lead의 context compaction도
    # 물려받지 않는다(#3875 Phase 3). deep-research subagent(``max_turns`` 최대 150)는
    # Phase 2 예산이 병적인 꼬리를 막더라도 max_turns/timeout/token_budget이 걸리기 전에
    # 누적 입력 1M을 넘길 수 있다. lead가 읽는 것과 동일한
    # ``app_config.summarization.enabled`` 스위치로 제어해(#3875의 메인테이너 지침) 설정
    # 하나가 두 체인을 모두 덮게 한다. 별도 ``subagents.summarization`` 필드는 없다.
    # 공용 factory는 summarization이 꺼져 있으면 ``None``을 반환하므로 스위치가 꺼진
    # 상태에서는 순수 no-op이다. trigger/keep/model/prompt 모두 lead가 읽는 동일한
    # ``summarization`` 설정에서 오므로 두 체인이 어긋날 수 없다.
    #
    # 배치는 lead 체인과 다르다. lead는 guard 3종(loop/token/safety) 앞에 summarization을
    # 붙이지만 여기서는 뒤에 붙인다. compaction은 상대 위치와 무관하게 ``before_model``에서
    # 돌고 guard middleware는 ``after_model``에서 집계하므로 무해하지만, 정확한 대칭은
    # 아니라는 점을 남긴다.
    #
    # ``skip_memory_flush=True``: 그러지 않으면 factory가 ``memory.enabled``일 때
    # ``memory_flush_hook``을 붙이는데, 이 hook은 compaction 이전 메시지를 ``thread_id``
    # 기준 durable memory 큐로 flush한다. subagent는 context에서 부모의 ``thread_id``를
    # 공유하므로 hook을 건너뛰지 않으면 subagent 내부 턴이 부모 thread의 durable memory에
    # 기록된다(#3875 Phase 3 리뷰).
    #
    # 이 middleware는 ``RemoveMessage(id=REMOVE_ALL_MESSAGES)``로 히스토리를 재작성해
    # run 도중 messages 채널을 줄인다. ``capture_new_step_messages``가 그 축소를 견뎌야
    # 하며(``step_events.py`` 참고), 그러지 않으면 compaction 지점 이후 캡처된 step을
    # 잃는다. ``consume_stop_reason``은 구현하지 않으므로 Phase 2의 guard-cap stop-reason
    # 채널을 방해하지 않는다.
    from deerflow.agents.middlewares.summarization_middleware import create_summarization_middleware

    summarization_middleware = create_summarization_middleware(
        app_config=app_config,
        skip_memory_flush=True,
        # null-model summarization의 기준은 subagent가 확정한 모델이다. subagent의
        # context/configurable에는 자식 모델이 실려 있지 않고 부모 것을 물려받으므로,
        # 직접 넘겨야 다른 모델을 쓰는 subagent가 부모 모델이 아닌 자기 모델로 요약한다.
        run_model_name=model_name,
    )
    if summarization_middleware is not None:
        middlewares.append(summarization_middleware)

    # SystemMessageCoalescingMiddleware (#4040) — 위의 DurableContextMiddleware가
    # 선두 system prompt 뒤에 두 번째 ``SystemMessage(authority_contract)``를 삽입한다
    # (subagent는 prompt를 ``create_agent(system_prompt=...)``가 아니라 ``messages``의
    # 선두 ``SystemMessage``로 들고 다닌다). system 메시지가 둘이거나 선두가 아니면
    # 대상 backend(vLLM/SGLang/Qwen/Anthropic)가 정확히 그걸 거부하므로, durable 수정이
    # #4039의 assistant-first 400을 duplicate-system 400으로 바꿔치기할 뿐이다. lead 체인을
    # 따라 coalescer를 가장 안쪽에 붙여 나가는 요청의 모든 SystemMessage를 선두
    # ``system_message`` 하나로 병합한다. 요청 단위 payload만 재작성하고
    # ``after_model``/``consume_stop_reason``이 없으므로 Phase 2 guard-cap 채널에 영향이
    # 없으며, 주입된 system 메시지를 보려면 DurableContextMiddleware보다 안쪽이어야 한다.
    from deerflow.agents.middlewares.system_message_coalescing_middleware import SystemMessageCoalescingMiddleware

    middlewares.append(SystemMessageCoalescingMiddleware())

    from deerflow_extension_api import AgentScope

    from deerflow.extensions import get_agent_build_extensions
    from deerflow.extensions.stack import compose_with_extensions

    resolved_extensions = extensions if extensions is not None else get_agent_build_extensions()
    if not resolved_extensions.has_middleware_contributors:
        return compose_with_extensions(middlewares, AgentScope.SUBAGENT, None, resolved_extensions)

    from deerflow_extension_api import AgentBuildContext

    from deerflow.extensions.policy import project_host_policy

    return compose_with_extensions(
        middlewares,
        AgentScope.SUBAGENT,
        AgentBuildContext(
            scope=AgentScope.SUBAGENT,
            agent_name=agent_name,
            model_name=model_name,
            policy=project_host_policy(
                app_config,
                token_budget_config=token_budget_config,
                max_subagents_per_run=None,
            ),
        ),
        resolved_extensions,
    )
