"""graph를 거치지 않는 일회성 LLM 텍스트 요청을 위한 공용 헬퍼.

여러 Gateway route(입력 다듬기, 후속 질문 제안, 제목류 재작성)가 같은 일을 한다. config로부터 chat
model을 만들고, Langfuse trace metadata를 붙이고, system + user 메시지 쌍으로 한 번 호출한 뒤
응답에서 평문을 꺼낸다. 이 절차를 한곳에 모아 두면 tracing metadata 필드와 호출 형태가 router마다
어긋나지 않는다. 한쪽을 고치면(예: Langfuse 필드 추가) 잊힌 사본에서 조용히 퇴행하는 대신 모든
호출자에 적용된다.

응답 텍스트의 *정리*(think block / code fence 제거, JSON 파싱)는 호출자마다 후처리가 다르므로
의도적으로 각 호출자에게 맡긴다. 이 헬퍼는 원문 텍스트를 추출하는 데서 멈춘다.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tracing import inject_langfuse_metadata
from deerflow.utils.llm_text import extract_response_text


def _resolve_environment() -> str | None:
    return os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT")


async def run_oneshot_llm(
    *,
    system_instruction: str,
    user_content: str,
    run_name: str,
    app_config: AppConfig,
    model_name: str | None = None,
    thread_id: str | None = None,
) -> str:
    """graph를 거치지 않는 system+user LLM turn을 한 번 실행하고 원문 텍스트를 반환한다.

    Args:
        system_instruction: System message 내용.
        user_content: Human message 내용.
        run_name: 이 호출의 LangChain ``run_name`` 및 Langfuse ``assistant_id``.
        app_config: model을 만드는 데 쓰는 애플리케이션 config.
        model_name: model 재정의(선택). ``None``이면 기본 model을 쓴다.
        thread_id: thread id(선택). tracing 목적으로만 Langfuse에 전달한다.

    Returns:
        model 응답에서 추출한 평문 내용(정리 전 원문).
    """
    model = create_chat_model(name=model_name, thinking_enabled=False, app_config=app_config)
    invoke_config: dict = {"run_name": run_name}
    inject_langfuse_metadata(
        invoke_config,
        thread_id=thread_id,
        user_id=get_effective_user_id(),
        assistant_id=run_name,
        model_name=model_name,
        environment=_resolve_environment(),
    )
    response = await model.ainvoke(
        [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content),
        ],
        config=invoke_config,
    )
    return extract_response_text(response.content)
