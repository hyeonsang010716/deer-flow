"""run 단위 token 예산 상한을 강제하는 미들웨어.

한 agent run 안의 model 호출들에 걸쳐 누적 token 사용량(input, output, total)을 추적하고, 설정 가능한
소프트 경고와 하드 정지 임계값을 적용한다.

감지 전략:

  1. 매 model 응답 후 현재 thread 이력의 모든 `AIMessage`의 `usage_metadata`를 합산한다.
     `TokenUsageMiddleware`가 subagent의 token을 이력에 소급 반영하므로 자동으로 함께 잡힌다.
  2. 가장 높은 비율(input, output, total 중)이 warn_threshold 이상이면 경고를 큐에 넣는다.
  3. 가장 높은 비율이 hard_stop_threshold 이상이면 tool_calls를 제거한다.

경고 주입은 지연 패턴을 쓴다.

  - after_model이 경고를 큐에 넣는다(state를 변경하지 않는다).
  - wrap_model_call이 다음 model 호출에서 HumanMessage로 주입한다.

이렇게 하면 AIMessage(tool_calls) → ToolMessage 짝이 보존된다.

stop reason 노출(#3875 Phase 2):

  하드 정지는 예외를 던지지 않는다. tool_calls를 제거해 agent 루프가 자연스럽게 끝나고 최종 답변을 내게
  한다. 호출자(예: subagent executor)가 예산으로 잘린 완료와 정상 완료를 구분할 수 있도록, 하드 정지가
  발생한 run을 ``_stop_reason``에 기록하고 :meth:`consume_stop_reason`으로 노출한다. 이 dict은
  ``after_agent``/``_clear_run_state``에서 의도적으로 지우지 않는다. run이 반환된 뒤 executor가 읽어야
  하기 때문이다. bounded dict이라 버려진 run 때문에 무한히 커지지 않으며, subagent run마다 새 미들웨어
  인스턴스를 만들므로 run 간 오염도 없다.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares._bounded_dict import BoundedDict
from deerflow.config.token_budget_config import TokenBudgetConfig

logger = logging.getLogger(__name__)

_BUDGET_WARNING_MSG = (
    "[TOKEN BUDGET WARNING] You have used {used:,} of your {budget:,} {reason} token budget ({percent:.0f}%). Wrap up your current work and produce a final answer. Avoid starting new tool calls unless absolutely necessary."
)
_BUDGET_EXCEEDED_MSG = "[TOKEN BUDGET EXCEEDED] The {reason} token usage ({used:,}) has exceeded the safety limit ({budget:,}). Producing final answer with results collected so far."


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    total: int = 0


class TokenBudgetMiddleware(AgentMiddleware[AgentState]):
    """run 단위 token 예산 상한을 강제한다."""

    def __init__(self, config: TokenBudgetConfig) -> None:
        super().__init__()
        self._config = config
        self._lock = threading.Lock()

        # run_id만을 키로 쓰고(덮어쓰기 안전) 크기를 제한한다(누수 안전).
        self._warned: BoundedDict[str, bool] = BoundedDict(1000)
        self._pending_warnings: BoundedDict[str, list[str]] = BoundedDict(1000)
        self._seen_messages: BoundedDict[str, dict[str, tuple[int, int]]] = BoundedDict(1000)
        self._cumulative_usage: BoundedDict[str, TokenUsage] = BoundedDict(1000)
        # 하드 정지가 발생할 때 기록되는 stop reason. run이 반환된 뒤 executor가 소비할 수 있도록
        # ``_clear_run_state``/``after_agent``에서 지우지 않으며, 버려진 run이 누수되지 않게 크기를 제한한다.
        self._stop_reason: BoundedDict[str, str] = BoundedDict(1000)

    @classmethod
    def from_config(cls, config: TokenBudgetConfig) -> TokenBudgetMiddleware:
        return cls(config=config)

    def reset(self) -> None:
        with self._lock:
            self._warned.clear()
            self._pending_warnings.clear()
            self._seen_messages.clear()
            self._cumulative_usage.clear()
            self._stop_reason.clear()

    def consume_stop_reason(self, run_id: str | None) -> str | None:
        """이 run에 대해 하드 정지가 기록한 stop reason을 꺼내 반환한다.

        run 중 예산 하드 정지가 발생했으면 ``"token_capped"``를, 아니면 ``None``을 반환한다. executor는
        run이 반환된 뒤 이를 호출해, 완료된 subagent가 실제로는 예산으로 잘린 것인지(그래서 lead에게
        ``stop_reason=token_capped``를 전달해야 하는지) 판단한다. 꺼내면서 제거하므로 재사용되는
        인스턴스에서 run이 쌓이지 않는다.
        """
        with self._lock:
            return self._stop_reason.pop(run_id, None)

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict) and "run_id" in ctx:
            return ctx["run_id"]
        # embedded client run 간 충돌을 막기 위해 runtime 객체 ID로 대체한다.
        return str(id(runtime))

    def _clear_run_state(self, run_id: str) -> None:
        with self._lock:
            self._warned.pop(run_id, None)
            self._pending_warnings.pop(run_id, None)
            self._seen_messages.pop(run_id, None)
            self._cumulative_usage.pop(run_id, None)

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> None:
        if not self._config.enabled:
            return

        # 이전 run의 메시지를 모두 'seen'으로 표시해 이번 run의 예산에 계산되지 않게 한다.
        messages = state.get("messages", [])
        if not messages:
            return

        run_id = self._get_run_id(runtime)
        with self._lock:
            seen = self._seen_messages.setdefault(run_id, {})
            self._cumulative_usage.setdefault(run_id, TokenUsage())

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.id and hasattr(msg, "usage_metadata"):
                    usage = msg.usage_metadata or {}
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    seen[msg.id] = (input_tokens, output_tokens)

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.before_agent(state, runtime)

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> None:
        if not self._config.enabled:
            return
        self._clear_run_state(self._get_run_id(runtime))

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.after_agent(state, runtime)

    @staticmethod
    def _append_text(content: str | list[dict | None] | None, stop_msg: str) -> str | list[dict | str]:
        """AIMessage.content 필드에 정지 메시지를 덧붙인다."""
        if content is None:
            return stop_msg
        if isinstance(content, str):
            if content:
                return f"{content}\n\n{stop_msg}"
            return f"\n\n{stop_msg}"
        if isinstance(content, list):
            new_content = list(content)
            new_content.append({"type": "text", "text": f"\n\n{stop_msg}"})
            return new_content
        return f"{content}\n\n{stop_msg}"

    def _build_hard_stop_update(self, msg: AIMessage, stop_msg: str) -> dict[str, Any]:
        """하드 정지를 위한 state 갱신 dict을 만든다."""
        updated_content = self._append_text(msg.content, stop_msg)
        kwargs = dict(msg.additional_kwargs) if msg.additional_kwargs else {}
        if "tool_calls" in kwargs:
            del kwargs["tool_calls"]
        if "function_call" in kwargs:
            del kwargs["function_call"]

        response_metadata = dict(getattr(msg, "response_metadata", {}) or {})

        if response_metadata.get("finish_reason") == "tool_calls":
            response_metadata["finish_reason"] = "stop"

        stopped_msg = msg.model_copy(update={"content": updated_content, "tool_calls": [], "additional_kwargs": kwargs, "response_metadata": response_metadata})
        return {"messages": [stopped_msg]}

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        if not self._config.enabled:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None

        run_id = self._get_run_id(runtime)

        with self._lock:
            seen = self._seen_messages.setdefault(run_id, {})
            usage_accum = self._cumulative_usage.setdefault(run_id, TokenUsage())

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.id and hasattr(msg, "usage_metadata"):
                    usage = msg.usage_metadata or {}

                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                    # 이 메시지에 대해 이전에 기록한 값을 확인한다.
                    prev_input, prev_output = seen.get(msg.id, (0, 0))

                    # 새로 추가된 token이 있는지 계산한다(소급 반영되는 subagent token도 처리한다).
                    diff_input = max(0, input_tokens - prev_input)
                    diff_output = max(0, output_tokens - prev_output)

                    if diff_input > 0 or diff_output > 0:
                        usage_accum.input += diff_input
                        usage_accum.output += diff_output
                        usage_accum.total += diff_input + diff_output
                        seen[msg.id] = (input_tokens, output_tokens)

            if usage_accum.total <= 0:
                return None

            fractions = [("total", usage_accum.total, self._config.max_tokens)]
            if self._config.max_input_tokens:
                fractions.append(("input", usage_accum.input, self._config.max_input_tokens))
            if self._config.max_output_tokens:
                fractions.append(("output", usage_accum.output, self._config.max_output_tokens))

            highest_fraction = 0.0
            trigger_reason = ""
            trigger_used = 0
            trigger_budget = 0

            for reason, used, limit in fractions:
                frac = used / limit
                if frac > highest_fraction:
                    highest_fraction = frac
                    trigger_reason = reason
                    trigger_used = used
                    trigger_budget = limit

            if highest_fraction >= self._config.hard_stop_threshold:
                logger.warning("Token budget hard stop triggered for run %s: %s limit exceeded", run_id, trigger_reason)
                # run이 반환된 뒤 executor가 lead에게 ``stop_reason=token_capped``를 알릴 수 있도록
                # stop reason을 기록한다(하드 정지 자체는 예외를 던지지 않는다). ``consume_stop_reason`` 참고.
                self._stop_reason[run_id] = "token_capped"
                # lead worker가 이 미들웨어 인스턴스 참조 없이도 읽을 수 있게 runtime.context에도 쓴다(#4176).
                ctx = getattr(runtime, "context", None)
                if isinstance(ctx, dict):
                    ctx["stop_reason"] = "token_capped"
                stop_text = _BUDGET_EXCEEDED_MSG.format(reason=trigger_reason, used=trigger_used, budget=trigger_budget)
                return self._build_hard_stop_update(last_msg, stop_text)

            if highest_fraction >= self._config.warn_threshold and not self._warned.get(run_id, False):
                self._warned[run_id] = True
                percent = highest_fraction * 100
                warn_text = _BUDGET_WARNING_MSG.format(reason=trigger_reason, used=trigger_used, budget=trigger_budget, percent=percent)
                logger.info("Token budget warning triggered for run %s: %s limit at %.1f%%", run_id, trigger_reason, percent)
                # wrap_model_call이 주입하도록 경고를 큐에 넣는다.
                warnings = self._pending_warnings.setdefault(run_id, [])
                warnings.append(warn_text)
                return None

            return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    def _drain_pending_warnings(self, runtime: Runtime) -> list[str]:
        if not self._config.enabled:
            return []

        run_id = self._get_run_id(runtime)
        with self._lock:
            warnings = self._pending_warnings.pop(run_id, None)
        return warnings or []

    def _inject_warnings(self, request: ModelRequest, warnings: list[str]) -> ModelRequest:
        if not warnings:
            return request

        merged_text = "\n\n".join(warnings)
        warning_msg = HumanMessage(content=merged_text, name="budget_warning")

        messages = getattr(request, "messages", [])
        new_messages = list(messages) + [warning_msg]
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:

        warnings = self._drain_pending_warnings(request.runtime)
        request = self._inject_warnings(request, warnings)

        return handler(request)

    @override
    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        warnings = self._drain_pending_warnings(request.runtime)
        request = self._inject_warnings(request, warnings)
        return await handler(request)
