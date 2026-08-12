import logging

from langchain.chat_models import BaseChatModel
from langchain_openai.chat_models.base import BaseChatOpenAI

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.reflection import resolve_class
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)


def _deep_merge_dicts(base: dict | None, override: dict) -> dict:
    """입력을 변형하지 않고 두 dict를 재귀적으로 병합한다."""
    merged = dict(base or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _vllm_disable_chat_template_kwargs(chat_template_kwargs: dict) -> dict:
    """vLLM/Qwen chat template kwargs용 비활성화 payload를 만든다."""
    disable_kwargs: dict[str, bool] = {}
    if "thinking" in chat_template_kwargs:
        disable_kwargs["thinking"] = False
    if "enable_thinking" in chat_template_kwargs:
        disable_kwargs["enable_thinking"] = False
    return disable_kwargs


def _declares_api_base(model_class: type) -> bool:
    """*model_class*가 ``api_base``를 자체 생성자 필드로 선언하는지 여부.

    ``langchain_deepseek:ChatDeepSeek``(따라서 ``PatchedChatDeepSeek``)가 그렇다. 이 경우
    ``api_base``가 정식 endpoint 키이므로 손대지 않고 그대로 전달해야 한다. 그 외의
    ``BaseChatOpenAI`` 하위 클래스는 ``openai_api_base``(별칭 ``base_url``)만 물려받는다.
    """
    return "api_base" in getattr(model_class, "model_fields", {})


def _normalize_openai_base_url(model_class: type, model_settings_from_config: dict) -> None:
    """OpenAI 호환 client를 위해 흔한 ``api_base`` 별칭을 ``base_url``로 매핑한다.

    ``BaseChatOpenAI`` 하위 클래스는 OpenAI endpoint override를 ``base_url``로 받는다
    (``openai_api_base``는 legacy 별칭). ``config.example.yaml``의 여러 provider가 *다른* model
    클래스에 ``api_base``를 쓰기 때문에, 사용자가 실수로 그 키를 이런 model에 복사해 오는 일이
    잦다. ``ModelConfig``가 ``extra="allow"``라서 잘못된 키는 config 로드 시점에 걸리지 않고
    생성자로 전달되는데, 생성자는 이를 거부하지 않고 ``model_kwargs``로 옮긴다. 그 값은 모든
    ``Completions.create()`` 호출에 펼쳐지고, OpenAI SDK가 *요청* 시점에 알아보기 힘든
    ``unexpected keyword argument 'api_base'`` 에러로 거부한다(그리고 endpoint override는 조용히
    사라진다). 사용자의 의도대로 동작하도록 여기서 이름을 바꾼다.

    클래스 경로 allowlist가 아니라 ``issubclass(model_class, BaseChatOpenAI)``로 판단하므로
    모든 OpenAI 호환 하위 클래스가 자동으로 적용된다 — 이 divert-and-crash 동작은 base class의
    성질이지 예전에 나열되던 두 경로만의 문제가 아니다. ``api_base``를 직접 선언하는 클래스는
    건너뛴다. 거기서는 그 키가 오타가 아니라 정식 키이기 때문이다.
    """
    if not issubclass(model_class, BaseChatOpenAI) or _declares_api_base(model_class):
        return
    if "api_base" not in model_settings_from_config:
        return
    if "base_url" in model_settings_from_config or "openai_api_base" in model_settings_from_config:
        # 정식 키가 이미 있다. 의도가 중복되는 kwarg를 피하려고 별칭을 버린다.
        model_settings_from_config.pop("api_base", None)
        logger.warning("Model config sets both an endpoint key (base_url/openai_api_base) and 'api_base'; using the former and ignoring 'api_base'.")
        return
    model_settings_from_config["base_url"] = model_settings_from_config.pop("api_base")
    logger.debug("Normalized model config key 'api_base' -> 'base_url' for OpenAI-compatible client.")


def _warn_unknown_model_settings(model_class, model_name: str, model_settings_from_config: dict) -> None:
    """OpenAI client가 조용히 ``model_kwargs``로 흘려보낼 config 키에 대해 경고한다.

    ``ModelConfig``는 ``extra="allow"``라서 오타 난 키(예: ``maxx_tokens``)가 config 로드
    시점에 걸리지 않는다. LangChain의 OpenAI client는 알 수 없는 생성자 kwarg를 거부하지
    않고 ``UserWarning``만 낸 뒤 그 키를 ``model_kwargs``로 옮긴다. 그 값은 모든
    ``Completions.create()`` 호출에 펼쳐지고, OpenAI SDK가 *요청* 시점에 config 오타까지
    거슬러 올라가기 매우 어려운 ``unexpected keyword argument`` 에러로 거부한다.

    이 함수는 잠복해 있던 그 실패를 model 생성 시점의 명시적이고 조치 가능한 로그로 바꾼다.
    적용 범위는 **OpenAI 호환 계열로 한정**된다 — ``model_kwargs`` divert-and-crash 동작이
    거기서 일어나고, 알려진 field/alias 집합도 그 계열에서만 정확하기 때문이다. 계열 판정은
    ``issubclass(model_class, BaseChatOpenAI)``다. divert가 그 base class에 구현되어 있어
    모든 하위 클래스가 물려받는다. 다른 provider(예: ``ChatAnthropic``)는 추가 kwarg를 다르게
    라우팅해서 이 allow-list에 대해 false positive를 낼 것이므로 의도적으로 건드리지 않는다.
    best-effort이고 치명적이지 않다. 클래스가 pydantic ``model_fields`` 스키마를 노출할 때만
    동작하고, field 이름과 별칭을 모두 유효한 것으로 취급하며, factory가 주입하고 OpenAI
    client가 받아들이는 표준 passthrough kwarg를 allow-list에 넣는다.
    """
    if not issubclass(model_class, BaseChatOpenAI):
        return
    known = getattr(model_class, "model_fields", None)
    if not known:
        return
    valid_names = set(known.keys())
    for field in known.values():
        alias = getattr(field, "alias", None)
        if alias:
            valid_names.add(alias)
    # 선언된 field 외에 factory가 주입하거나 OpenAI client가 받아들이는 표준 kwarg.
    valid_names |= {
        "model",
        "model_kwargs",
        "extra_body",
        "default_headers",
        "default_query",
        "stream_usage",
        "stream_chunk_timeout",
        "reasoning_effort",
    }
    unknown = sorted(k for k in model_settings_from_config if k not in valid_names)
    if unknown:
        logger.warning(
            "Model '%s' (%s): config key(s) %s are not recognized parameters of the model class and will be forwarded as-is; this may raise at request time. Check for typos (e.g. 'maxx_tokens' -> 'max_tokens').",
            model_name,
            getattr(model_class, "__name__", "?"),
            unknown,
        )


# OpenAI 호환 streaming 응답의 기본 chunk 간격 예산.
#
# langchain-openai는 이 초 수만큼 chunk를 받지 못하면 ``StreamChunkTimeoutError``를 던진다.
# 자체 기본값은 120초인데, 첫 chunk가 정상적으로 90~150초 걸릴 수 있는 reasoning
# model(DeepSeek-R1, Doubao-thinking, GPT-5)에는 너무 공격적이다. 그래서 240초를 기본값으로
# 두어 streaming 레이어가 긴 thinking 정지에 잘 걸리지 않게 한다. 실제로 멈추면
# LLMErrorHandlingMiddleware가 여전히 재시도한다(budget=2). 사용자는 config.yaml에서
# model별로 override할 수 있다.
_DEFAULT_STREAM_CHUNK_TIMEOUT_SECONDS: float = 240.0


def _apply_stream_chunk_timeout_default(model_class: type, model_settings_from_config: dict) -> None:
    """OpenAI 호환 client에 넉넉한 ``stream_chunk_timeout``을 주입한다.

    ``stream_chunk_timeout``은 langchain-openai ``BaseChatOpenAI``의 field이므로
    ``ChatOpenAI``와 이를 상속하는 모든 DeerFlow provider가 받아들인다: ``PatchedChatOpenAI``와
    self-hosted / reasoning 어댑터인 ``VllmChatModel``, ``MindIEChatModel``,
    ``PatchedChatDeepSeek``, ``PatchedChatMiMo``, ``PatchedChatStepFun``,
    ``PatchedChatMiniMax``. 명시적 클래스 경로 allowlist 대신
    ``issubclass(model_class, BaseChatOpenAI)``로 판단해서, 모든 OpenAI 호환 하위 클래스가
    기본값을 자동으로 물려받고 명시적 override도 존중하게 한다. 이슈 #3189는 ``mimo-v2.5``
    (``PatchedChatMiMo``)에서 보고되었는데, 최초 수정(#3195)은 ``ChatOpenAI`` /
    ``PatchedChatOpenAI``만 매칭해서 그 하위 클래스들은 langchain-openai의 공격적인 내장 chunk
    간격 timeout을 그대로 쓰고, 더 나쁘게는 사용자가 명시한 ``stream_chunk_timeout``을 조용히
    버렸다.

    동작:

    * ``BaseChatOpenAI`` 하위 클래스: ``config.yaml``의 명시적 값은 보존한다. 명시적
      ``null``은 상위 단계의 ``model_dump(exclude_none=True)``에서 제거되어 "미설정"으로
      취급되므로 기본값이 주입된다.
    * 그 외의 client(예: ``ChatAnthropic``): 해당 키를 선언하지 않는 생성자로 전달되지
      않도록 키를 버린다. 이 client들에서 이 kwarg는 선언된 field가 아니어서, client에 따라
      조용히 버려지거나(``ChatAnthropic``은 ``extra="ignore"``를 선언한다) 다른 OpenAI 계열
      client에서는 ``model_kwargs``로 흘러가 요청 시점에 거부된다. 어느 쪽이든 사용자의
      의도는 사라지므로 선제적으로 버린다.
    """
    if not issubclass(model_class, BaseChatOpenAI):
        model_settings_from_config.pop("stream_chunk_timeout", None)
        return
    if "stream_chunk_timeout" in model_settings_from_config:
        return
    model_settings_from_config["stream_chunk_timeout"] = _DEFAULT_STREAM_CHUNK_TIMEOUT_SECONDS


def create_chat_model(name: str | None = None, thinking_enabled: bool = False, *, app_config: AppConfig | None = None, attach_tracing: bool = True, model_overrides: dict | None = None, **kwargs) -> BaseChatModel:
    """config로부터 chat model 인스턴스를 생성한다.

    Args:
        name: 생성할 model 이름. None이면 config의 첫 번째 model을 사용한다.
        thinking_enabled: 지원되는 경우 model의 extended-thinking 모드를 켠다.
        app_config: 명시적 application config. 생략하면 캐시된 전역 config로 대체한다.
        model_overrides: 호출자별 sampling override(예: custom agent의 ``temperature`` /
            ``max_tokens``)를 model 프로필 위에 얹는다. ``None`` 값은 무시되므로 설정되지
            않은 override가 프로필 값을 덮어쓰지 않는다. thinking / Codex 변환보다 먼저
            적용되므로 provider별 정규화(예: Codex의 ``max_tokens`` 제거)가 override된
            값에도 그대로 적용된다.
        attach_tracing: True(기본값)이면 tracing callback(Langfuse, LangSmith)을 model
            인스턴스에 직접 붙인다. standalone 호출자 — 이미 invocation root에서 tracing을
            연결한 LangGraph 실행 밖에서 model을 호출하는 모든 것(``MemoryUpdater``,
            임시 유틸리티 등) — 는 model 단위 callback으로 trace가 남도록 이 기본값을
            유지한다. 이미 graph root에서 tracing을 붙이는 호출자(``make_lead_agent``,
            graph 내부의 ``TitleMiddleware``)는 반드시 ``attach_tracing=False``를 넘겨야
            한다. 그러지 않으면 같은 LLM 호출이 span을 중복 생성하고(graph 기준 하나,
            model 기준 하나), model이 nested observation이 되면서 ``langfuse_*`` 키가
            제거되어 ``session_id`` / ``user_id`` metadata가 trace에 도달하지 못한다.

    Returns:
        chat model 인스턴스.
    """
    config = app_config or get_app_config()
    if name is None:
        name = config.models[0].name
    model_config = config.get_model_config(name)
    if model_config is None:
        raise ValueError(f"Model {name} not found in config") from None
    model_class = resolve_class(model_config.use, BaseChatModel)
    model_settings_from_config = model_config.model_dump(
        exclude_none=True,
        exclude={
            "use",
            "name",
            "display_name",
            "description",
            "supports_thinking",
            "supports_reasoning_effort",
            "when_thinking_enabled",
            "when_thinking_disabled",
            "thinking",
            "supports_vision",
            # context 표시기 크기를 잡는 데 쓰는 runtime/UI metadata. provider client는
            # 이를 model 생성자 인자로 받지 않는다.
            "context_window",
            # 표시 전용 metadata(console의 비용 표시가 사용한다) — provider client에는
            # 절대 도달하면 안 된다. 알 수 없는 kwarg를 completion 요청 payload로
            # 넘겨버리기 때문이다.
            "pricing",
        },
    )
    # 호출자별 sampling override(예: custom agent의 temperature / max_tokens)를 프로필
    # 위에 얹는다. None은 무시해서 설정되지 않은 override가 구성된 프로필 값을 덮어쓰지
    # 않게 한다. 아래의 thinking/Codex 변환보다 먼저 적용하므로, provider별 정규화(Codex의
    # max_tokens 제거, thinking 비활성화 경로)가 프로필 원래 값과 똑같이 병합된 값에도
    # 적용된다.
    if model_overrides:
        model_settings_from_config.update({key: value for key, value in model_overrides.items() if value is not None})
    # `thinking` 단축 field를 병합해 실효 when_thinking_enabled를 계산한다.
    # `thinking` 단축은 when_thinking_enabled["thinking"]을 설정하는 것과 같다.
    has_thinking_settings = (model_config.when_thinking_enabled is not None) or (model_config.thinking is not None)
    effective_wte: dict = dict(model_config.when_thinking_enabled) if model_config.when_thinking_enabled else {}
    if model_config.thinking is not None:
        merged_thinking = {**(effective_wte.get("thinking") or {}), **model_config.thinking}
        effective_wte = {**effective_wte, "thinking": merged_thinking}
    if thinking_enabled and has_thinking_settings:
        if not model_config.supports_thinking:
            raise ValueError(f"Model {name} does not support thinking. Set `supports_thinking` to true in the `config.yaml` to enable thinking.") from None
        if effective_wte:
            model_settings_from_config.update(effective_wte)
    if not thinking_enabled:
        if model_config.when_thinking_disabled is not None:
            # 사용자가 준 비활성화 설정이 전적으로 우선한다
            model_settings_from_config.update(model_config.when_thinking_disabled)
        elif has_thinking_settings and effective_wte.get("extra_body", {}).get("thinking", {}).get("type"):
            # OpenAI 호환 gateway: thinking이 extra_body 아래에 중첩되어 있다
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"thinking": {"type": "disabled"}},
            )
            model_settings_from_config["reasoning_effort"] = "minimal"
        elif has_thinking_settings and (disable_chat_template_kwargs := _vllm_disable_chat_template_kwargs(effective_wte.get("extra_body", {}).get("chat_template_kwargs") or {})):
            # vLLM은 chat template kwargs로 thinking을 켜고 끈다.
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"chat_template_kwargs": disable_chat_template_kwargs},
            )
        elif has_thinking_settings and effective_wte.get("thinking", {}).get("type"):
            # 네이티브 langchain_anthropic: thinking이 생성자 파라미터 그 자체다
            model_settings_from_config["thinking"] = {"type": "disabled"}
    if not model_config.supports_reasoning_effort:
        kwargs.pop("reasoning_effort", None)
        model_settings_from_config.pop("reasoning_effort", None)

    # api_base -> base_url 별칭을 먼저 정규화해서, 이후의 OpenAI 호환 휴리스틱(아래
    # stream_usage 기본값 / stream_chunk_timeout)이 정식 endpoint 키를 보게 한다.
    _normalize_openai_base_url(model_class, model_settings_from_config)
    _apply_stream_chunk_timeout_default(model_class, model_settings_from_config)

    # Codex Responses API model: thinking 모드를 reasoning_effort로 매핑한다
    from deerflow.models.openai_codex_provider import CodexChatModel

    if issubclass(model_class, CodexChatModel):
        # 현재 ChatGPT Codex endpoint는 max_tokens/max_output_tokens를 거부한다.
        model_settings_from_config.pop("max_tokens", None)

        # frontend가 명시적 reasoning_effort를 주면 그것을 사용한다(low/medium/high)
        explicit_effort = kwargs.pop("reasoning_effort", None)
        if not thinking_enabled:
            model_settings_from_config["reasoning_effort"] = "none"
        elif explicit_effort and explicit_effort in ("low", "medium", "high", "xhigh"):
            model_settings_from_config["reasoning_effort"] = explicit_effort
        elif "reasoning_effort" not in model_settings_from_config:
            model_settings_from_config["reasoning_effort"] = "medium"

    # MindIE model: 보수적인 retry 기본값을 강제한다.
    # timeout 정규화는 MindIEChatModel 내부에서 처리한다.
    if getattr(model_class, "__name__", "") == "MindIEChatModel":
        # timeout이 연쇄적으로 번지지 않도록 max_retries 제약을 강제한다.
        model_settings_from_config["max_retries"] = model_settings_from_config.get("max_retries", 1)

    # streaming 응답에서 token usage metadata를 얻을 수 있도록 stream_usage를 켠다.
    # LangChain의 BaseChatOpenAI는 custom base_url/api_base가 없을 때만 stream_usage=True를
    # 기본값으로 삼기 때문에, 서드파티 endpoint(예: doubao, deepseek)를 쓰는 model은 usage
    # 데이터를 조용히 잃는다. 명시적으로 설정하지 않았다면 True를 기본값으로 둔다.
    if "stream_usage" not in model_settings_from_config and "stream_usage" not in kwargs:
        if "stream_usage" in getattr(model_class, "model_fields", {}):
            model_settings_from_config["stream_usage"] = True

    _warn_unknown_model_settings(model_class, name, model_settings_from_config)

    model_instance = model_class(**kwargs, **model_settings_from_config)

    if attach_tracing:
        callbacks = build_tracing_callbacks()
        if callbacks:
            existing_callbacks = model_instance.callbacks or []
            model_instance.callbacks = [*existing_callbacks, *callbacks]
            logger.debug(f"Tracing attached to model '{name}' with providers={len(callbacks)}")
    return model_instance
