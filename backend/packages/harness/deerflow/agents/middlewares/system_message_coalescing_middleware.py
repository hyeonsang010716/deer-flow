"""여러 SystemMessage를 맨 앞의 하나로 합치는 미들웨어.

엄격한 OpenAI 호환 백엔드(vLLM, SGLang, Qwen)와 Anthropic은 맨 앞이 아닌 SystemMessage를
"System message must be at the beginning"이나 "Received multiple non-consecutive system messages"
같은 오류로 거부한다. 공식 OpenAI API는 대화 중간의 system 메시지를 허용하므로 이 문제는 엄격한
백엔드에서만 드러난다.

DeerFlow의 lead agent에 SystemMessage가 여러 개 쌓이는 이유는 DynamicContextMiddleware가 ID-swap
기법으로 첫 번째 또는 마지막 HumanMessage를 삼중 항목으로 바꾸고, 그 첫 요소가 SystemMessage
reminder이기 때문이다(OWASP LLM01에 따라 framework 소유의 날짜/metadata가 사용자 입력인 척해서는
안 된다). 자정을 넘기면 날짜 갱신용 SystemMessage가 하나 더 주입된다. create_agent는 정적
system_prompt를 별도 필드 ``request.system_message``에 두고, model 호출 handler 안에서만 메시지
목록으로 펼친다(``[request.system_message, *messages]``).

이 미들웨어는 handler가 둘을 펼치기 전인 wrap_model_call에서 동작하며, ``request.system_message``와
``request.messages``의 모든 SystemMessage를 합쳐 ``system_message`` 필드로 하나의 선두
SystemMessage를 내보낸다. request payload만 건드리고 영속 대화 state(checkpoint)는 바꾸지 않으므로,
marker로 이력을 훑는 미들웨어(예: is_dynamic_context_reminder)는 계속 정상 동작한다.

Note: claude_provider._coalesce_system_messages가 Claude에 대해 이미 하던 요청별 병합과 같은 일을
provider 비의존 계층에서 수행한다. provider마다 패치하는 대신 한 번의 수정으로 모든 백엔드가 혜택을 본다.
"""

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder


def _flatten_content(content) -> str:
    """메시지 content를 평범한 문자열로 변환한다. str과 list 타입을 모두 처리한다.

    langchain 메시지는 멀티모달을 위해 list 타입 content를 지원한다
    (예: ``[{"type": "text", "text": "..."}]``). DeerFlow의 SystemMessage는 항상 문자열이지만,
    이 헬퍼는 어떤 content 형태에도 견디도록 한다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _coalesce_request(request: ModelRequest) -> ModelRequest | None:
    """``request.system_message``와 ``messages`` 안의 SystemMessage를 하나로 합친다.

    langchain 1.2.15 이상에서는 정적 system prompt가 ``request.messages``가 아니라 별도 필드
    ``request.system_message``에 있다. model 호출 handler는 마지막 순간에야 둘을 펼치므로
    (``[system_message, *messages]``), ``messages``만 훑는 미들웨어는 prompt를 보지 못하고 아무 일도
    하지 못한다. 이 헬퍼는 두 소스를 모두 살펴 모든 SystemMessage를 하나로 합치고 결과를
    ``system_message``로 내보내, handler가 여전히 올바르게 앞에 붙이도록 한다.

    ``messages`` 안에 SystemMessage가 없으면 None을 반환한다. 그 경우 ``system_message``가 있다면 이미
    유일한 선두 system 블록이므로, 요청을 전혀 변형하지 않고 통과시켜 prefix cache 적중을 보존한다.
    """
    in_msg_systems = [m for m in request.messages if isinstance(m, SystemMessage)]
    if not in_msg_systems:
        return None

    # system_message(있다면)와 messages 안의 모든 SystemMessage를 합친다.
    parts: list[SystemMessage] = []
    if request.system_message is not None:
        parts.append(request.system_message)
    parts.extend(in_msg_systems)

    # dynamic_context_reminder SystemMessage는 중복을 제거한다. 마지막 것(가장 최근 날짜)만 남기고
    # 이전 reminder는 버린다. 그러지 않으면 자정을 넘길 때 병합된 내용에 서로 모순되는 <current_date>
    # 블록 두 개가 시간 기준 없이 나란히 놓인다. 원래 둘을 갈라놓던 중간 턴들이 병합 후 사라지기 때문이다.
    # 모델은 무시해야 할 낡은 날짜가 아니라 최신 날짜만 봐야 한다.
    reminder_indices = [i for i, p in enumerate(parts) if is_dynamic_context_reminder(p)]
    if len(reminder_indices) > 1:
        keep_last = reminder_indices[-1]
        parts = [p for i, p in enumerate(parts) if i not in reminder_indices[:-1] or i == keep_last]

    # 첫 SystemMessage(대개 정적 system_prompt)의 id를 보존해, 선두 system 메시지 id를 키로 쓰는 하위
    # 소비자들이 영향을 받지 않게 한다. 모든 조각의 additional_kwargs를 병합해 reminder의
    # hide_from_ui / dynamic_context_reminder 같은 marker가 합쳐진 블록에 남도록 한다.
    first = parts[0]
    merged_kwargs: dict = {}
    for p in parts:
        merged_kwargs.update(p.additional_kwargs or {})
    merged = SystemMessage(
        content="\n\n".join(_flatten_content(p.content) for p in parts),
        id=first.id,
        additional_kwargs=merged_kwargs,
    )

    non_system = [m for m in request.messages if not isinstance(m, SystemMessage)]
    return request.override(system_message=merged, messages=non_system)


class SystemMessageCoalescingMiddleware(AgentMiddleware[AgentState]):
    """모든 SystemMessage를 맨 앞의 단일 SystemMessage로 병합한다.

    before_agent가 아니라 wrap_model_call을 쓴다. 그래야 ``system_message``와 ``messages``가 아직 별도
    필드인 최종 request payload에서 병합이 이뤄지고, 영속화된 state["messages"]는 건드리지 않는다.
    덕분에 이력을 훑는 모든 소비자(memory builder, journal, summarization, dynamic-context 탐지)에게
    checkpoint 구조가 그대로 유지된다.
    """

    @staticmethod
    def _maybe_coalesce(request: ModelRequest) -> ModelRequest:
        coalesced = _coalesce_request(request)
        if coalesced is None:
            return request
        return coalesced

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._maybe_coalesce(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._maybe_coalesce(request))
