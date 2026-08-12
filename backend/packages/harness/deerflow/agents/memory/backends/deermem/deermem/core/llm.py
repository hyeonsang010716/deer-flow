"""DeerMem 자체 LLM 생성. deer-flow의 ``create_chat_model``을 쓰지 않는다.

``build_llm(model_config)``은 DeerMem의 model 하위 설정(provider/model/api_key/base_url/
temperature)으로 ``langchain.chat_models.init_chat_model``을 호출해 langchain ``ChatModel``을
만든다. 만들어진 인스턴스는 DeerMem이 소유하며(``self._llm``) 의존성 주입으로
``MemoryUpdater``에 넘긴다.

``DeerMem.__init__``은 host가 주입한 ``host_llm``을 우선한다. ``model``이 비어 있으면
deer-flow factory가 앱 기본 model을 거기에 주입하며, 이는 추상화 이전 ``model_name: null``과
같다. 이 ``build_llm``은 ``model`` 하위 설정으로 만드는 fallback이다. ``model``이 비어 있으면
``None``을 반환한다. 단독 DeerMem은 그때 LLM이 없다(LLM이 필요 없는 작업은 되지만 업데이트는
예외를 던진다). factory를 거치면 ``host_llm``이 무설정 상황을 덮는다. langchain의
``init_chat_model``이 지원하는 provider는 모두 쓸 수 있다(OpenAI, Anthropic, DeepSeek 같은
OpenAI 호환 gateway 등).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeerMemModelConfig

logger = logging.getLogger(__name__)


def build_llm(model_config: DeerMemModelConfig | None) -> Any:
    """DeerMem의 model 설정으로 langchain ChatModel을 만들어 주입한다.

    다음 경우에 ``None``을 반환한다. ``model_config``가 None이거나 ``model``이 설정되지
    않았을 때(무설정 상태라 LLM이 없다. LLM이 필요 없는 작업은 되지만 업데이트는 예외를
    던진다), 또는 ``init_chat_model``이 실패했을 때(provider/api_key/base_url 설정 오류).
    실패 경로는 :func:`_host_default_llm`과 마찬가지로 WARNING을 남기고 ``None``으로
    떨어진다. 그래야 잘못 지정한 ``model`` 때문에 앱 startup이 죽지 않는다. memory
    CRUD/읽기/검색은 계속 동작하고, 추출만 비활성화되며, 업데이트는 runtime에 원인 에러가
    로그로 남은 채 예외를 던진다.
    """
    if model_config is None or not model_config.model:
        return None
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {}
    if model_config.api_key is not None:
        kwargs["api_key"] = model_config.api_key
    if model_config.base_url is not None:
        kwargs["base_url"] = model_config.base_url
    if model_config.temperature is not None:
        kwargs["temperature"] = model_config.temperature
    try:
        return init_chat_model(
            model=model_config.model,
            model_provider=model_config.provider or "openai",
            **kwargs,
        )
    except Exception as e:  # noqa: BLE001 - _host_default_llm처럼 완만히 실패시켜 startup을 죽이지 않는다
        logger.warning(
            "build_llm failed for model=%r (provider=%r): %s; memory extraction disabled (non-LLM ops still work; an update will raise).",
            model_config.model,
            model_config.provider or "openai",
            e,
        )
        return None
