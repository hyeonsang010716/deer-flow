"""DeerFlow용 summarization 미들웨어 확장."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Any, Protocol, override, runtime_checkable

from langchain.agents import AgentState
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, get_buffer_string, trim_messages
from langgraph.config import get_config
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder
from deerflow.config.app_config import get_app_config
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)
_SUMMARY_TRIGGER_MESSAGE_NAME = "summary"
_UNSET = object()
# 비어 있거나 요약하기엔 너무 긴 경계 상황을 위한, 생성하지 않은 유효한 요약 문구.
# model 호출을 건너뛰게 하며 생성 실패로 취급하면 안 된다.
_CANNED_SUMMARIES = frozenset(
    {
        "No previous conversation history.",
        "Previous conversation was too long to summarize.",
    }
)


class SummaryGenerationError(RuntimeError):
    """run model fallback까지 모두 소진한 뒤에도 요약 생성이 실패했음을 나타낸다.

    호출자가 ``raise_on_failure``로 명시적으로 선택한 경우(수동 ``/compact`` 경로)에만 발생시켜,
    실제 실패를 "압축할 것이 없음"과 구분해 보고한다. 자동 경로는 ``raise_on_failure``를 False로 두고
    실패를 삼켜, 그 턴의 compaction 상태를 그대로 둔다.
    """


@dataclass(frozen=True)
class SummarizationEvent:
    """대화 이력이 요약으로 대체되기 직전에 전달되는 context."""

    messages_to_summarize: tuple[AnyMessage, ...]
    preserved_messages: tuple[AnyMessage, ...]
    thread_id: str | None
    agent_name: str | None
    runtime: Runtime


@dataclass(frozen=True)
class ContextCompactionResult:
    """오래된 context를 요약하고 활성 tail을 남긴 결과."""

    summary_text: str
    messages_to_summarize: tuple[AnyMessage, ...]
    preserved_messages: tuple[AnyMessage, ...]
    total_tokens: int


@runtime_checkable
class BeforeSummarizationHook(Protocol):
    """summarization이 state에서 메시지를 제거하기 전에 호출되는 hook."""

    def __call__(self, event: SummarizationEvent) -> None: ...


def _resolve_thread_id(runtime: Runtime) -> str | None:
    """runtime context 또는 LangGraph config에서 현재 thread ID를 해석한다."""
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id is None:
        try:
            config_data = get_config()
        except RuntimeError:
            return None
        thread_id = config_data.get("configurable", {}).get("thread_id")
    return thread_id


def _resolve_agent_name(runtime: Runtime) -> str | None:
    """runtime context 또는 LangGraph config에서 현재 agent 이름을 해석한다."""
    agent_name = runtime.context.get("agent_name") if runtime.context else None
    if agent_name is None:
        try:
            config_data = get_config()
        except RuntimeError:
            return None
        agent_name = config_data.get("configurable", {}).get("agent_name")
    return agent_name


class DeerFlowSummarizationMiddleware(SummarizationMiddleware):
    """압축 직전 hook 디스패치를 지원하는 summarization 미들웨어."""

    def __init__(
        self,
        *args,
        before_summarization: list[BeforeSummarizationHook] | None = None,
        app_config: Any | None = None,
        configured_model_name: str | None = None,
        run_model_name: str | None = None,
        anchor_model_name: str | None = _UNSET,  # type: ignore[assignment]
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._before_summarization_hooks = before_summarization or []
        # model 소유권 상태. run을 실제로 실행하는 model은 run마다 선택되며 그것이 유일한 진실이다.
        # 그래서 호출자(lead / subagent / 수동 빌더)가 ``run_model_name``으로 직접 넘기고, 미들웨어가
        # ``runtime.context``나 ``get_config()``에서 다시 유도하지 않는다. 그 필드들은 custom agent나
        # subagent가 해석한 model을 담지 않기 때문이다.
        #
        # ``configured_model_name``은 명시적으로 설정된 요약 model이다(``None``이면 run 자신의 model로
        # 요약한다). ``run_model_name``은 run이 실행에 쓰는 model이다. 둘이 다르고 요약 provider가
        # 망가진 경우(키 만료, quota, 장애) run 자신의 정상 model로 compaction을 이어갈 수 있다.
        self._app_config = app_config
        self._configured_summary_model_name = configured_model_name
        self._run_model_name = run_model_name
        # 요약 LLM 호출은 LangGraph 미들웨어 hook 안에서 실행되므로, 그대로 두면 token 스트림이
        # messages-tuple 스트림 콜백에 잡혀 프론트엔드에 유령 AI 메시지로 방송된다. 별도 model 복사본에
        # TAG_NOSTREAM을 달아 streaming handler가 건너뛰게 한다.
        # 부모의 profile / ls_params 검사가 계속 동작하도록 self.model 자체에는 태그를 달지 않는다.
        self._summary_model = self._tag_nostream(self.model)
        # ``self.model``은 미리 만들어 둔 *anchor* model이다. 부모의 token counter / profile 검사를
        # 담당하고, 후보 이름이 일치하면 생성 단계에서 그대로 재사용된다. 팩토리는 이를 예외로 보호하며
        # 만들고 이름을 명시적으로 넘긴다. 직접 생성하는 경우(테스트)에는 기존 팩토리 선택
        # (설정된 model, 없으면 기본값)을 따르므로 넘겨받은 ``model``이 primary가 된다.
        if anchor_model_name is _UNSET:
            self._anchor_model_name = configured_model_name or self._default_model_name()
        else:
            self._anchor_model_name = anchor_model_name
        # 이름별로 지연 생성해 캐시하는 nostream 생성 model. None은 생성 실패를 뜻하며, 잘못된 후보
        # 설정을 매 턴 재시도하지 않고 fail-open 경계를 벗어나지도 않게 한다.
        self._model_cache: dict[str | None, Any] = {}

    def _tag_nostream(self, model: Any) -> Any:
        """기존 tag를 뭉개지 않으면서 TAG_NOSTREAM을 붙인 ``model`` 복사본을 반환한다.

        lead_agent/agent.py는 RunJournal 귀속을 위해 "middleware:summarize"를 바인딩한다.
        RunnableBinding.with_config는 config를 얕게 병합하므로, [TAG_NOSTREAM]만으로 덮어쓰지 말고
        기존 tag를 명시적으로 보존해야 한다.
        """
        existing_tags = list((getattr(model, "config", None) or {}).get("tags") or [])
        merged_tags = [*existing_tags, TAG_NOSTREAM] if TAG_NOSTREAM not in existing_tags else existing_tags
        return model.with_config(tags=merged_tags)

    def _default_model_name(self) -> str | None:
        if self._app_config is None:
            return None
        models = getattr(self._app_config, "models", None)
        return models[0].name if models else None

    def _generation_candidate_names(self) -> list[str | None]:
        """요약 생성 후보를 이름 기준 순서대로(중복 제거해) 반환한다.

        요약 model이 명시된 경우: 설정된 model이 먼저이고, run 자신의 model이 별도 fallback이 된다.
        ``model_name: null``인 경우: run 자신의 model만 쓴다. 그 생성이 primary이므로
        ``config.models[0]``에 미리 의존하지 않는다(run model을 해석하지 못했을 때만 기본값을 쓴다).
        ``None`` 항목은 "``create_chat_model``이 기본값을 고르게 하라"는 뜻이며, 어떤 이름도 해석되지
        않을 때만 나타난다.
        """
        default = self._default_model_name()
        if self._configured_summary_model_name is not None:
            names = [self._configured_summary_model_name, self._run_model_name or default]
        else:
            names = [self._run_model_name or default]
        deduped: list[str | None] = []
        seen: set[str | None] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _model_for(self, name: str | None) -> Any | None:
        """``name``에 해당하는 nostream 요약 model을 예외로 보호하며 지연 생성해 반환한다.

        ``name``이 anchor와 같으면 미리 만든 anchor를 그대로 반환하고(재생성 없음), 아니면 새로 만들어
        캐시한다. 생성 실패는 잡아서 ``None``으로 캐시하므로, 잘못된 후보 설정이 fail-open 경계를
        벗어나지 않고 이번 턴에 재시도되지도 않으며 다음 후보는 정상적으로 시도된다.
        """
        if name == self._anchor_model_name:
            return self._summary_model
        if name in self._model_cache:
            return self._model_cache[name]
        try:
            model = create_chat_model(
                name=name,
                thinking_enabled=False,
                app_config=self._app_config,
                attach_tracing=False,
            )
            built = self._tag_nostream(model.with_config(tags=["middleware:summarize"]))
        except Exception:
            logger.exception("Failed to build summary model %r; trying the next candidate", name)
            built = None
        self._model_cache[name] = built
        return built

    @override
    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str | None:
        return self._summarize_with(messages_to_summarize)

    @override
    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str | None:
        return await self._asummarize_with(messages_to_summarize)

    def _prepare_summary_prompt(self, messages_to_summarize: list[AnyMessage], previous_summary: str | None) -> str | None:
        """포맷된 prompt를 반환하거나, 비었거나 너무 긴 경계 상황에서는 정해진 문구를 반환한다.

        ``None``이 아니면서 실제 prompt가 아닌 반환값(두 개의 정해진 문구)은 유효한 요약이며 생성을
        건너뛴다. ``None``은 "prompt를 만들라"는 뜻이다.
        """
        if not messages_to_summarize:
            return "No previous conversation history."
        prompt = self._build_summary_prompt(messages_to_summarize, previous_summary=previous_summary)
        if prompt is None:
            return "Previous conversation was too long to summarize."
        return prompt

    @staticmethod
    def _nonempty_summary(text: Any) -> str | None:
        """model 응답 텍스트를 정규화한다. 비었거나 공백뿐인 본문은 실패로 본다.

        ``""``를 요약으로 확정하면 before_summarization hook이 실행되고 빈 대체물을 위해 이전 이력이
        전부 제거된다. 그래서 빈 본문은 유효한 요약이 아니라 생성 실패로 취급한다(fallback을 시도하거나
        state를 그대로 둔다).
        """
        stripped = text.strip() if isinstance(text, str) else ""
        return stripped or None

    def _summarize_with(self, messages_to_summarize: list[AnyMessage], previous_summary: str | None = None) -> str | None:
        """부모의 ``_create_summary``와 동일하게 동작하되 nostream 태그가 붙은 model을 호출한다.

        ``self.model``을 인스턴스 수준에서 바꿔치기하지 않는다. agent와 미들웨어는 캐시되어 동시 run들이
        공유하므로, 일시적 교체는 ``await`` 도중 ``RunnableBinding``을 다른 coroutine에 노출시키고
        raw model을 검사하는 부모 로직(``profile`` / ``_get_ls_params``)을 깨뜨린다.

        생성에는 run 자신의 model(``model_name: null``) 또는 명시적으로 설정된 요약 model을 쓰고,
        실패하면 run model로 fallback한다. 덕분에 요약 provider가 망가져도 정상 model이 있는 한
        compaction이 멈추지 않는다.
        """
        prompt = self._prepare_summary_prompt(messages_to_summarize, previous_summary)
        if prompt is None or prompt in _CANNED_SUMMARIES:
            return prompt
        # 후보를 순서대로 시도한다. 각 시도는 전체 생명주기(보호된 지연 생성 -> invoke -> 텍스트 추출 ->
        # 비어 있지 않은지 검증)를 스스로 담당하며, 어느 단계에서 실패하든 다음 후보로 넘어간다.
        # 모든 후보가 실패하면 호출자는 compaction 상태를 그대로 둔다.
        names = self._generation_candidate_names()
        for index, name in enumerate(names):
            text = self._invoke_summary(self._model_for(name), prompt, last=index == len(names) - 1)
            if text is not None:
                return text
        return None

    async def _asummarize_with(self, messages_to_summarize: list[AnyMessage], previous_summary: str | None = None) -> str | None:
        """nostream model을 쓰는 :meth:`_summarize_with`의 async 버전."""
        prompt = self._prepare_summary_prompt(messages_to_summarize, previous_summary)
        if prompt is None or prompt in _CANNED_SUMMARIES:
            return prompt
        names = self._generation_candidate_names()
        for index, name in enumerate(names):
            text = await self._ainvoke_summary(self._model_for(name), prompt, last=index == len(names) - 1)
            if text is not None:
                return text
        return None

    def _invoke_summary(self, model: Any | None, prompt: str, *, last: bool = False) -> str | None:
        """``model``을 호출해 요약을 얻는다. 오류이거나 응답이 비었으면 ``None``을 반환한다.

        텍스트 추출과 비어 있지 않은지 검증은 try *안에서* 수행한다. 응답의 ``.text``를 읽는 것도
        provider 결과를 소비하는 과정의 일부이므로, 접근자 실패는 fail-open 경계를 벗어나지 않고
        후보 실패(다음 후보로 진행)로 변환되어야 한다.
        """
        if model is None:
            return None
        try:
            response = model.invoke(prompt, config={"metadata": {"lc_source": "summarization"}})
            return self._checked_summary(response, last)
        except Exception:
            self._log_summary_error(last)
            return None

    async def _ainvoke_summary(self, model: Any | None, prompt: str, *, last: bool = False) -> str | None:
        """:meth:`_invoke_summary`의 async 버전."""
        if model is None:
            return None
        try:
            response = await model.ainvoke(prompt, config={"metadata": {"lc_source": "summarization"}})
            return self._checked_summary(response, last)
        except Exception:
            self._log_summary_error(last)
            return None

    def _checked_summary(self, response: Any, last: bool) -> str | None:
        summary = self._nonempty_summary(getattr(response, "text", None))
        if summary is None:
            self._log_summary_empty(last)
        return summary

    @staticmethod
    def _log_summary_error(last: bool) -> None:
        if last:
            logger.exception("Summary generation failed; skipping compaction this turn")
        else:
            logger.warning("Summary generation failed; falling back to the run model", exc_info=True)

    @staticmethod
    def _log_summary_empty(last: bool) -> None:
        if last:
            logger.warning("Summary model returned empty text; skipping compaction this turn")
        else:
            logger.warning("Summary model returned empty text; falling back to the run model")

    @staticmethod
    def _summary_count_message(summary_text: str) -> HumanMessage:
        return HumanMessage(content=summary_text, name=_SUMMARY_TRIGGER_MESSAGE_NAME)

    def _messages_for_trigger_count(self, messages: list[AnyMessage], summary_text: str | None) -> list[AnyMessage]:
        if not summary_text:
            return messages
        return [*messages, self._summary_count_message(summary_text)]

    @staticmethod
    def _bound_text(text: str, cap: int) -> str:
        if len(text) <= cap:
            return text
        if cap <= 0:
            return ""
        head = cap * 2 // 3
        omitted_marker = "\n...\n"
        if cap <= len(omitted_marker):
            return text[:cap]
        tail = max(0, cap - head - len(omitted_marker))
        if tail == 0:
            return text[:cap]
        return f"{text[:head]}{omitted_marker}{text[-tail:]}"

    def _trim_summary_section_text(self, text: str, max_tokens: int, *, strategy: str) -> str:
        if not text.strip():
            return ""
        max_tokens = max(1, max_tokens)
        try:
            trimmed = trim_messages(
                [HumanMessage(content=text)],
                max_tokens=max_tokens,
                token_counter=self.token_counter,
                strategy=strategy,
                allow_partial=True,
                text_splitter=list,
            )
            if trimmed:
                content = trimmed[-1].content
                if isinstance(content, str) and content.strip():
                    return content
        except Exception:
            logger.debug("Failed to trim summary prompt section with token counter; falling back to deterministic text cap", exc_info=True)
        return self._bound_text(text, max_tokens)

    def _build_summary_input_text(self, formatted_messages: str, previous_summary: str | None = None) -> str | None:
        if self.trim_tokens_to_summarize is None:
            trimmed_new_messages = formatted_messages
            trimmed_previous_summary = previous_summary.strip() if previous_summary else ""
        else:
            max_tokens = max(1, self.trim_tokens_to_summarize)
            if previous_summary:
                new_message_tokens = max(1, max_tokens // 2)
                previous_summary_tokens = max(1, max_tokens - new_message_tokens)
                trimmed_previous_summary = self._trim_summary_section_text(
                    previous_summary.strip(),
                    previous_summary_tokens,
                    strategy="last",
                )
                trimmed_new_messages = self._trim_summary_section_text(
                    formatted_messages,
                    new_message_tokens,
                    strategy="first",
                )
            else:
                trimmed_previous_summary = ""
                trimmed_new_messages = self._trim_summary_section_text(
                    formatted_messages,
                    max_tokens,
                    strategy="first",
                )

        # <existing_summary>/<new_messages> 블록에 넣기 전에 < > &를 이스케이프한다. new_messages는
        # raw state["messages"] tail에 get_buffer_string을 적용한 것이고(InputSanitizationMiddleware는
        # ModelRequest만 덮어쓰고 state는 건드리지 않으므로 summarizer는 실제 사용자 텍스트를 본다),
        # existing_summary는 이전 턴의 summary_text다. "</new_messages>..." 같은 값이 이스케이프되지 않으면
        # 블록을 닫고 추출용 LLM에 권위 있는 섹션을 위조할 수 있다. #4162가 <conversation> 블록에,
        # #4097이 <memory> 블록에 적용한 것과 같은 block-breakout 방어다. 끝의 "..."가 엔티티를 쪼개지
        # 않도록 trimming 이후에 이스케이프하며, 내용이 속성값이 아니라 요소 텍스트 위치에 들어가므로
        # quote=False를 쓴다.
        parts: list[str] = []
        if trimmed_previous_summary:
            parts.extend(
                [
                    "<existing_summary>",
                    html.escape(trimmed_previous_summary, quote=False),
                    "</existing_summary>",
                    "",
                ]
            )
        if trimmed_new_messages:
            parts.extend(
                [
                    "<new_messages>",
                    html.escape(trimmed_new_messages, quote=False),
                    "</new_messages>",
                ]
            )
        if not parts:
            return None
        return "\n".join(parts)

    def _build_summary_prompt(self, messages_to_summarize: list[AnyMessage], previous_summary: str | None = None) -> str | None:
        """요약 prompt를 만든다. trimming 후 남는 것이 없으면 ``None``을 반환한다."""
        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            trimmed_messages = messages_to_summarize[-1:]
        if not trimmed_messages:
            return None
        # 메시지 객체에 str()을 호출할 때 metadata 때문에 token이 부풀지 않도록 별도로 포맷한다.
        formatted_messages = get_buffer_string(trimmed_messages)
        formatted_messages = self._build_summary_input_text(formatted_messages, previous_summary=previous_summary)
        if not formatted_messages:
            return None
        return self.summary_prompt.format(messages=formatted_messages).rstrip()

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._maybe_summarize(state, runtime)

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return await self._amaybe_summarize(state, runtime)

    def _prepare_compaction(
        self,
        state: AgentState,
        *,
        force: bool = False,
    ) -> tuple[list[AnyMessage], list[AnyMessage], str | None, int] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        previous_summary = state.get("summary_text") if isinstance(state.get("summary_text"), str) else None
        trigger_messages = self._messages_for_trigger_count(messages, previous_summary)
        total_tokens = self.token_counter(trigger_messages)
        if not force and not self._should_summarize(trigger_messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)
        messages_to_summarize, preserved_messages = self._preserve_dynamic_context_reminders(messages_to_summarize, preserved_messages)
        if not messages_to_summarize:
            return None
        return messages_to_summarize, preserved_messages, previous_summary, total_tokens

    def compact_state(
        self,
        state: AgentState,
        runtime: Runtime,
        *,
        force: bool = False,
        raise_on_failure: bool = False,
    ) -> ContextCompactionResult | None:
        """오래된 context를 요약하고 활성 tail을 남긴다.

        ``force``는 자동 트리거 임계값을 건너뛴다(수동 호출자는 항상 압축을 원한다).
        ``raise_on_failure``는 *별개* 관심사다. 설정하면(수동 ``/compact`` 경로) 생성 실패 시
        ``SummaryGenerationError``를 발생시켜 "압축할 것이 없음"과 구분해 보고할 수 있다. 자동 경로는
        False로 두고 실패를 삼킨 뒤 나중에 트리거되는 턴에서 재시도한다.
        """
        prepared = self._prepare_compaction(state, force=force)
        if prepared is None:
            return None
        messages_to_summarize, preserved_messages, previous_summary, total_tokens = prepared
        summary = self._summarize_with(messages_to_summarize, previous_summary=previous_summary)
        if summary is None:
            if raise_on_failure:
                raise SummaryGenerationError("summary generation failed")
            return None
        # 대체 요약이 만들어진 뒤에만 hook을 실행한다. 끝내 생성되지 않을 요약을 위해 압축 이전 메시지를
        # durable memory로 flush하면 다음 시도에서 같은 작업을 중복하게 된다. 메시지 제거는 이 함수가
        # 반환된 뒤(_maybe_summarize에서) 일어나므로 hook은 메시지가 사라지기 전에 실행된다.
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)
        return ContextCompactionResult(
            summary_text=summary,
            messages_to_summarize=tuple(messages_to_summarize),
            preserved_messages=tuple(preserved_messages),
            total_tokens=total_tokens,
        )

    async def acompact_state(
        self,
        state: AgentState,
        runtime: Runtime,
        *,
        force: bool = False,
        raise_on_failure: bool = False,
    ) -> ContextCompactionResult | None:
        """:meth:`compact_state`의 async 버전(``raise_on_failure`` 설명은 그쪽 참고)."""
        prepared = self._prepare_compaction(state, force=force)
        if prepared is None:
            return None
        messages_to_summarize, preserved_messages, previous_summary, total_tokens = prepared
        summary = await self._asummarize_with(messages_to_summarize, previous_summary=previous_summary)
        if summary is None:
            if raise_on_failure:
                raise SummaryGenerationError("summary generation failed")
            return None
        # 대체 요약이 만들어진 뒤에만 hook을 실행한다(compact_state 참고).
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)
        return ContextCompactionResult(
            summary_text=summary,
            messages_to_summarize=tuple(messages_to_summarize),
            preserved_messages=tuple(preserved_messages),
            total_tokens=total_tokens,
        )

    def _maybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        result = self.compact_state(state, runtime, force=False)
        if result is None:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *result.preserved_messages,
            ],
            "summary_text": result.summary_text,
        }

    async def _amaybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        result = await self.acompact_state(state, runtime, force=False)
        if result is None:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *result.preserved_messages,
            ],
            "summary_text": result.summary_text,
        }

    def _preserve_dynamic_context_reminders(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """숨겨진 dynamic-context reminder와 ID-swap 짝을 요약 압축 대상에서 제외한다.

        이 reminder들은 현재 날짜와 선택적 memory를 담는다. summarization이 이를 제거하면
        DynamicContextMiddleware는 이미 주입한 reminder를 잃고 대체본을 대화의 엉뚱한 지점에 주입할 수 있다.

        ``_make_reminder_and_user_messages``가 만드는 ID-swap 삼중 항목은 메시지 세 개로 구성된다.
        ``SystemMessage(id=X)``와 ``HumanMessage(id=X__memory)``는 ``dynamic_context_reminder=True``로
        태깅되지만, 원본 사용자 내용을 담은 ``HumanMessage(id=X__user)``는 태깅되지 **않는다**.
        짝을 함께 구제하지 않으면 ``__user``가 ``to_summarize``에 남아 산문으로 압축되고, 태깅된 메시지들이
        고아가 되며 사용자 질문이 모델의 직접 context에서 사라진다.

        이 메서드는 태깅된 reminder를 구제하고, ``id``가 같은 ``stable_id`` prefix를 공유하는 태깅되지 않은
        메시지(``X__user``, ``X__memory``)도 함께 구제한다.
        """
        reminders = [msg for msg in messages_to_summarize if is_dynamic_context_reminder(msg)]
        if not reminders:
            return messages_to_summarize, preserved_messages

        # 태깅된 reminder에서 base ID(stable_id prefix)를 모은다.
        # id="ctx-001__memory"인 reminder의 base는 "ctx-001"이다.
        # id="ctx-001"인 reminder(SystemMessage)의 base도 "ctx-001"이다.
        # removesuffix는 접미사만 제거하므로 stable_id 중간의 "__"는 건드리지 않는다
        # (예: "ctx__001"은 그대로 남는다. rsplit이었다면 "ctx"로 잘못 유도했을 것이다).
        # 알려진 ID-swap 접미사(__memory, __user)만 제거한다. __user는 태깅되지 않아 reminders에
        # 나타나지 않지만 방어적으로 포함한다.
        reminder_base_ids: set[str] = set()
        for msg in reminders:
            if msg.id:
                base = msg.id.removesuffix("__memory").removesuffix("__user")
                reminder_base_ids.add(base)

        # 한 번의 순회로 분할한다. messages_to_summarize를 시간 순서대로 훑으며 태깅된 reminder와
        # 태깅되지 않은 ID-swap 짝(알려진 base + "__"로 시작하는 id)을 모두 구제한다. 이렇게 하면 구제된
        # 메시지들 사이의 원래 순서가 보존되고(한 summarization 구간에 여러 삼중 항목이 들어올 때 중요하다),
        # 기존의 reminders+peers 연결 방식이 필요로 하던 id(m) 기반 중복 제거도 없어진다.
        rescued: list[AnyMessage] = []
        remaining: list[AnyMessage] = []
        for msg in messages_to_summarize:
            if is_dynamic_context_reminder(msg) or (msg.id and any(msg.id.startswith(b + "__") for b in reminder_base_ids)):
                rescued.append(msg)
            else:
                remaining.append(msg)
        return remaining, rescued + preserved_messages

    def _fire_hooks(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
        runtime: Runtime,
    ) -> None:
        if not self._before_summarization_hooks:
            return

        event = SummarizationEvent(
            messages_to_summarize=tuple(messages_to_summarize),
            preserved_messages=tuple(preserved_messages),
            thread_id=_resolve_thread_id(runtime),
            agent_name=_resolve_agent_name(runtime),
            runtime=runtime,
        )

        for hook in self._before_summarization_hooks:
            try:
                hook(event)
            except Exception:
                hook_name = getattr(hook, "__name__", None) or type(hook).__name__
                logger.exception("before_summarization hook %s failed", hook_name)


def _build_summary_anchor(candidate_names: list[str | None], app_config: Any) -> tuple[Any | None, str | None]:
    """``candidate_names`` 중 가장 먼저 생성 가능한 model을 예외로 보호하며 만든다.

    반환되는 model에는 RunJournal 귀속용 tag가 붙지만 TAG_NOSTREAM은 붙지 *않는다*(미들웨어가 nostream
    복사본을 따로 감싼다). 이 model이 부모의 token counter / profile anchor가 되고, 후보 이름이 일치하면
    생성 단계에서 재사용된다. 이름별 생성 실패는 삼키고 다음 후보를 시도하므로, primary 생성자가 망가져도
    agent 생성이 깨지지 않고 정상적인 run model을 건너뛰지도 않는다. 마지막의 ``None`` 이름은
    ``create_chat_model``에 자체 기본값을 요청한다는 뜻이다. 아무것도 만들 수 없으면 ``(None, None)``을
    반환한다.
    """
    tried: set[str | None] = set()
    for name in candidate_names:
        if name in tried:
            continue
        tried.add(name)
        try:
            model = create_chat_model(name=name, thinking_enabled=False, app_config=app_config, attach_tracing=False)
        except Exception:
            logger.exception("Failed to build summary anchor model %r; trying the next candidate", name)
            continue
        return model.with_config(tags=["middleware:summarize"]), name
    return None, None


def create_summarization_middleware(
    *,
    app_config: Any | None = None,
    keep: tuple[str, int | float] | None = None,
    skip_memory_flush: bool = False,
    run_model_name: str | None = None,
) -> DeerFlowSummarizationMiddleware | None:
    """설정에 따른 summarization 미들웨어를 생성한다.

    lead-agent 자동 경로와 수동 context compaction 경로가 모두 이 팩토리를 쓰므로 model 해석, hook,
    prompt 설정, 보존 기본값이 서로 어긋날 수 없다.

    ``run_model_name``은 run이 실제로 실행에 쓰는 model이며, 호출자(lead / subagent / 수동 빌더 각각이
    이미 해석한다)가 넘긴다. ``model_name: null`` 요약과 명시적 요약 model의 fallback에서 이것이 유일한
    진실이다. 미들웨어는 ``runtime.context``나 ``get_config()``에서 다시 유도하지 않는다. 그 값들은
    custom agent나 subagent가 해석한 model을 담지 않기 때문이다.

    ``skip_memory_flush``는 압축 이전 메시지를 durable memory 큐로 flush하는 ``memory_flush_hook``을
    생략한다. lead chain은 이를 유지하고(리서치 결과는 남아야 한다), subagent chain은 이를 설정한다.
    subagent의 내부 턴("Task" human 메시지와 중간 AI/tool 턴)이 부모 thread의 durable memory에 기록되지
    않게 하기 위함이다. hook은 ``thread_id``를 키로 쓰고 subagent는 부모의 ``thread_id``를 공유한다
    (#3875 Phase 3 리뷰).
    """
    resolved_app_config = app_config or get_app_config()
    config = resolved_app_config.summarization

    if not config.enabled:
        return None

    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [item.to_tuple() for item in config.trigger]
        else:
            trigger = config.trigger.to_tuple()

    default_name = resolved_app_config.models[0].name if getattr(resolved_app_config, "models", None) else None
    # anchor(token counter / profile용이며 생성에도 재사용되는 model)를 예외로 보호하며 만든다.
    # 설정된 model이나 기본 model을 그냥 만들었다가 망가진 생성자가 밖으로 새어 나가게 두지 않는다.
    # 후보 순서: primary 생성 model(설정된 요약 model, 없으면 run 자신의 model), run model, 기본값,
    # 마지막으로 ``None``(create_chat_model의 기본값). 따라서 null인 경우 ``config.models[0]``이 아니라
    # ``run_model_name``에서 생성되며, primary가 망가져도 agent 생성이 실패하는 대신 정상적인 run model로
    # 넘어간다.
    primary_name = config.model_name or run_model_name or default_name
    anchor_model, anchor_name = _build_summary_anchor(
        [primary_name, run_model_name or default_name, default_name, None],
        resolved_app_config,
    )
    if anchor_model is None:
        logger.warning("Summarization is enabled but no summary model could be constructed; compaction is unavailable for this build")
        return None

    kwargs: dict[str, Any] = {
        "model": anchor_model,
        "trigger": trigger,
        "keep": keep or config.keep.to_tuple(),
    }
    if config.trim_tokens_to_summarize is not None:
        kwargs["trim_tokens_to_summarize"] = config.trim_tokens_to_summarize
    if config.summary_prompt is not None:
        kwargs["summary_prompt"] = config.summary_prompt

    hooks: list[BeforeSummarizationHook] = []
    if resolved_app_config.memory.enabled and not skip_memory_flush:
        from deerflow.agents.memory.summarization_hook import memory_flush_hook

        hooks.append(memory_flush_hook)

    return DeerFlowSummarizationMiddleware(
        **kwargs,
        before_summarization=hooks,
        app_config=resolved_app_config,
        configured_model_name=config.model_name,
        run_model_name=run_model_name,
        anchor_model_name=anchor_name,
    )
