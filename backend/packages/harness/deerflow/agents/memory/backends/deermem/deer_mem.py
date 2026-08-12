"""DeerMem — 기본 :class:`MemoryManager` backend이며 독립적이다.

DeerMem은 DeerFlow의 memory 기계(``core/``의 다섯 모듈: storage/queue/updater/prompt/
message_processing)를 backend 중립적인
:class:`~deerflow.agents.memory.manager.MemoryManager` contract 뒤에 감싼다. storage/queue/
updater를 모듈 수준 singleton이 아니라 ``PrivateAttr`` 의존성으로 소유한다. factory가
``backend_config``를 BaseModel 필드로 넘기고, ``model_post_init``이 이를 :class:`DeerMemConfig`
로 파싱해 의존성을 구성한다. 동작은 추상화 이전 코드와 같다. 같은 filter와 human/ai 검증,
correction/reinforcement 탐지가 같은 debounce 큐로 들어가고, 같은
``format_memory_for_injection``이 주입 텍스트를 만들며, 같은 CRUD가 관리 엔드포인트를 받친다.

DeerMem 전용 관심사(filter/detect, ``<memory>`` 감싸기, ``enabled`` gating, facts 모델)는
의도적으로 ABC 밖에 두고 여기에 둔다. ``warm``/``reload_memory``/fact CRUD는 ABC의 tier-3
선택 hook이다(기본값은 ``warm``=True, 나머지는 ``NotImplementedError``). DeerMem은 지원하는
것만 override한다. 호출자(gateway/client/tools)는 직접 호출하고 미지원 backend에 대해
``NotImplementedError``를 잡는다. ``hasattr`` 탐색은 더 이상 하지 않는다.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

from deerflow.agents.memory.manager import MemoryConflictError, MemoryCorruptionError, MemoryManager

from .deermem.config import DeerMemConfig
from .deermem.core.llm import build_llm
from .deermem.core.message_processing import (
    SIGNAL_NAMES,
    detect_signals,
    filter_messages_for_memory,
    filter_trivial,
    load_patterns,
)
from .deermem.core.paths import DEFAULT_AGENT_BUCKET
from .deermem.core.prompt import format_memory_for_injection, load_prompt, load_prompt_messages, warm_tiktoken_cache
from .deermem.core.queue import MemoryUpdateQueue, QueueFull
from .deermem.core.storage import MemoryRevisionConflict, MemoryStorageCorruption, create_storage
from .deermem.core.updater import MemoryUpdater, _coerce_source_confidence

logger = logging.getLogger(__name__)


def _resolve_agent_name(agent_name: str | None) -> str:
    """대소문자를 구분하지 않는 DeerFlow 표준 agent 식별자를 반환한다."""
    return agent_name.lower() if agent_name is not None else DEFAULT_AGENT_BUCKET


def _call_backend(operation):
    """DeerMem 내부 storage 에러를 공개 manager contract로 변환한다."""
    try:
        return operation()
    except MemoryRevisionConflict as exc:
        raise MemoryConflictError(str(exc)) from exc
    except MemoryStorageCorruption as exc:
        raise MemoryCorruptionError(str(exc)) from exc


def _legacy_source_value(source: Any) -> str:
    """구조화된 source 메타데이터를 레거시 공개 문자열로 되돌린다."""
    if isinstance(source, str):
        return source
    if not isinstance(source, dict):
        return "unknown"
    source_type = source.get("type")
    thread_id = source.get("threadId")
    if source_type == "conversation" and isinstance(thread_id, str) and thread_id:
        return thread_id
    if isinstance(source_type, str) and source_type:
        return source_type
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return "unknown"


def _compat_document(memory_data: dict[str, Any]) -> dict[str, Any]:
    """저장 형식을 바꾸지 않고 기존 Manager/API 형태로 반환한다."""
    result = copy.deepcopy(memory_data)
    for fact in result.get("facts", []):
        if isinstance(fact, dict):
            fact["source"] = _legacy_source_value(fact.get("source"))
    return result


class DeerMem(MemoryManager):
    """기본 memory backend. 파일 기반 fact와 debounce된 LLM 추출을 쓴다."""

    # backend 전용 의존성은 pydantic 필드가 아니라 PrivateAttr다. storage/llm/queue는
    # pydantic 객체가 아니며 검증/직렬화에 참여하면 안 된다.
    # self.backend_config -> DeerMemConfig를 거쳐 model_post_init에서 한 번 만든다.
    _config: Any = PrivateAttr(default=None)
    _storage: Any = PrivateAttr(default=None)
    _llm: Any = PrivateAttr(default=None)
    _updater: Any = PrivateAttr(default=None)
    _queue: Any = PrivateAttr(default=None)
    _trivial_patterns: Any = PrivateAttr(default=None)

    # DeerMem은 search()를 구현하므로(저장된 fact에 대한 대소문자 무시 부분 문자열 검색)
    # mode="tool"에서 유효하다. base의 불변식 검증기가 tool 모드에 이를 요구한다.
    # 실제 검색이 없는 backend는 False 기본값을 물려받아 mode="tool"로 쓸 수 없다.
    supports_search: ClassVar[bool] = True

    def model_post_init(self, __context: Any) -> None:
        """``self.backend_config``로부터 DeerMem의 의존성을 구성한다.

        pydantic의 ``__init__``이 필드를 검증한 뒤에 실행된다. ``backend_config``를
        :class:`DeerMemConfig`로 파싱하고(비었거나 None이면 기본값 적용) storage/patterns/
        llm/updater/queue를 DI로 연결한다.
        """
        self._config = DeerMemConfig.from_backend_config(self.backend_config)
        self._storage = create_storage(self._config)
        # signal 탐지 패턴(외부 YAML. ``patterns_dir`` override 또는 번들 기본값이
        # 외부화 이전 동작과 같다). 생성 시 한 번 로드해 ``_prepare_update``의 detect_*
        # 호출이 재사용한다. trivial과 signal 패턴을 생성 시점에 미리 로드해, 잘못 설정된
        # patterns_dir(파일 없음/잘못된 yaml)이 첫 업데이트가 아니라 startup에서 드러나게
        # 한다. 컴파일된 패턴은 load_patterns가 캐시한다.
        self._trivial_patterns = load_patterns("trivial", patterns_dir=self._config.patterns_dir)
        for _signal_name in SIGNAL_NAMES:
            load_patterns(_signal_name, patterns_dir=self._config.patterns_dir)
        # host_llm(host가 주입한 기본 model)이 build_llm(model)보다 우선한다. 그래야 설정이
        # 없는 DeerMem(`model`이 비어 있음)도 앱 기본값으로 추출하며, 추상화 이전의
        # `model_name: null`과 동작이 같다. factory 없이 단독으로 쓰면 None이다.
        self._llm = self._config.host_llm if self._config.host_llm is not None else build_llm(self._config.model)
        self._updater = MemoryUpdater(self._config, self._storage, self._llm, prompts_dir=self._config.prompts_dir, callbacks=self.callbacks)
        # retrieval은 파생 데이터다. 어떤 scope의 첫 검색이 lazy하게 다시 만들고,
        # Gateway warm-up이 event loop 밖에서 전체 rebuild를 수행한다.
        self._retrieval_lock = threading.RLock()
        self._retrieval_warmed_scopes: set[tuple[str | None, str | None]] = set()
        self._retrieval_fully_warmed = False
        # 명시적으로 지정된 *전역* prompt 템플릿을 생성 시점에 검증한다. 그래야 잘못 설정된
        # prompts_dir이 조용히 업데이트가 버려지는 형태가 아니라 startup에서 드러난다.
        # agent별 override({prompts_dir}/{agent}/*.yaml)는 여기서 알 수 없다. 첫 사용 시
        # lazy하게 검증되고 updater의 예외 핸들러가 ERROR로 남긴다.
        # fact_extraction은 휴면 상태다(runtime 호출자가 없다). 제외한다.
        if self._config.prompts_dir is not None:
            _dummy_vars = {
                "current_memory": "{}",
                "conversation": "(validation)",
                "correction_hint": "",
                "staleness_review_section": "",
                "consolidation_section": "",
            }
            load_prompt("staleness_review", prompts_dir=self._config.prompts_dir).format(stale_facts="")
            load_prompt("consolidation", prompts_dir=self._config.prompts_dir).format(consolidation_groups="", max_groups=1)
            load_prompt_messages("memory_update", _dummy_vars, prompts_dir=self._config.prompts_dir)
        self._queue = MemoryUpdateQueue(self._config, self._updater)

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> DeerMem:
        """host hook을 소비하면서 의존성이 연결된 DeerMem을 만든다.

        factory는 host hook(tracing, hidden-message filter, trace-context manager, host-llm
        factory)을 ``backend_config``에 주입하는 대신 kwargs로 넘긴다. DeerMem은 자기가 쓰는
        것(DeerMemConfig 필드)만 여기서 병합하며, 명시된 ``backend_config`` 값을 존중한다.
        ``host_llm``은 model이 설정되지 않은 경우에만 host factory로 만든다(host_llm이
        ``build_llm(model)``보다 우선하며, model이 있는데 쓰이지 않을 host 기본값을 만들면
        startup 시간만 낭비한다). 실제 의존성 연결은 직접 생성 경로와 공유하는
        ``model_post_init``에서 이뤄진다.
        """
        config_dict = dict(backend_config or {})
        for key in ("should_keep_hidden_message", "trace_context_manager", "extraction_callback"):
            if key not in config_dict and key in host_hooks:
                config_dict[key] = host_hooks[key]
        if "host_llm" not in config_dict:
            model_cfg = config_dict.get("model")
            if not (isinstance(model_cfg, dict) and model_cfg.get("model")):
                host_llm_factory = host_hooks.get("host_llm_factory")
                if host_llm_factory is not None:
                    config_dict["host_llm"] = host_llm_factory()
        # callbacks는 DeerMemConfig가 아니라 base MemoryManager의 필드다. 그대로 넘긴다.
        # config_dict는 위에서 병합한 host hook을 담아 model_post_init이 이를
        # DeerMemConfig(self._config, PrivateAttr)로 파싱하게 한다. 연결이 끝나면
        # backend_config를 host가 넘긴 순수 데이터로 되돌려(주입된 hook 없이) 필드가
        # 직렬화 가능하게 유지하고 README contract("host hook은 from_config kwargs로 오며
        # backend_config에는 들어가지 않는다")와 맞춘다. hook은 backend_config 필드가 아니라
        # self._config에 있다.
        instance = cls(backend_config=config_dict, mode=mode, callbacks=host_hooks.get("callbacks"))
        instance.backend_config = dict(backend_config or {})
        return instance

    # ── 쓰기 ─────────────────────────────────────────────────────────────
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """필터링, 검증, signal 탐지를 거쳐 debounce 큐에 넣는다.

        추상화 이전에 ``MemoryMiddleware.after_agent``에 있던 전처리를 그대로 옮긴 것이다.
        ``enabled`` gate와 ``thread_id``/``user_id``/``trace_id`` 결정은 호출 지점에 남는다.
        """
        prepared = self._prepare_update(messages)
        if prepared is None:
            return
        filtered, signals = prepared
        # 큐를 DeerMem이 소유하므로 backpressure 시 성능 저하 처리도 DeerMem이 맡는다.
        # 여기서의 QueueFull은 로그를 남기고 버린다. 그래야 memory backpressure가
        # MemoryMiddleware.after_agent로 전파돼 agent run을 깨뜨리지 않고 "업데이트 건너뜀"
        # 으로 그친다(동료 middleware도 같은 방식으로 자기 방어를 한다). 버려진 업데이트는
        # 다음 턴에 다시 들어온다. middleware가 매 사이클 전체 대화를 넘기고, 큐에 들어가지
        # 않은 턴에서는 watermark가 전진하지 않기 때문이다.
        try:
            self._queue.add(
                thread_id=thread_id,
                messages=filtered,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
                trace_id=trace_id,
                signals=signals,
            )
        except QueueFull as e:
            logger.warning("Memory update rejected under backpressure (thread=%s): %s", thread_id, e)

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """필터링, 검증, signal 탐지를 거쳐 즉시 flush하도록 큐에 넣는다.

        추상화 이전에 ``memory_flush_hook``에 있던 전처리를 그대로 옮긴 것이다.
        summarization이 메시지를 제거하기 직전에 쓴다.
        """
        prepared = self._prepare_update(messages)
        if prepared is None:
            return
        filtered, signals = prepared
        # 다층 방어. 긴급 경로는 backpressure에서도 항상 받아들이므로(_enqueue_locked 참고)
        # 여기서 QueueFull은 예상되지 않는다. 다만 긴급 flush는 summarization_hook에서
        # 호출되므로 예외가 전파되면 summarization이 깨진다. 안전하게 잡아서 로그만 남긴다.
        try:
            self._queue.add_nowait(
                thread_id=thread_id,
                messages=filtered,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
                signals=signals,
            )
        except QueueFull as e:
            logger.warning("Memory emergency flush rejected under backpressure (thread=%s): %s", thread_id, e)

    def _prepare_update(
        self,
        messages: list[Any],
    ) -> tuple[list[Any], frozenset[str]] | None:
        """user와 최종 AI 메시지만 남기고 둘 다 있는지 확인한 뒤 signal을 탐지한다.

        ``(filtered, signals)``를 반환한다. ``signals``는 최근 턴에서 탐지된 signal 클래스
        집합이다. 의미 있는 대화가 없으면(user 턴이나 assistant 턴이 없거나, 모든 턴이 단순
        확인 응답으로 걸러졌으면) ``None``을 반환한다.
        """
        filtered = filter_messages_for_memory(
            messages,
            should_keep_hidden_message=self._config.should_keep_hidden_message,
        )
        filtered = filter_trivial(filtered, patterns=self._trivial_patterns)
        user_messages = [m for m in filtered if getattr(m, "type", None) == "human"]
        assistant_messages = [m for m in filtered if getattr(m, "type", None) == "ai"]
        if not user_messages or not assistant_messages:
            return None
        signals = detect_signals(filtered, patterns_dir=self._config.patterns_dir)
        return filtered, frozenset(signals)

    # ── 읽기 ─────────────────────────────────────────────────────────────
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """memory를 로드해 주입용으로 포맷한다(감싸지 않은 평문).

        middleware 모드는 선택된 agent의 fact를 사용자 전역 요약과 함께 주입한다. tool
        모드는 전역 요약만 주입하고 fact는 ``memory_search`` 뒤에 남겨, prompt와 이후 검색
        결과에 중복되지 않게 한다.

        포맷 파라미터는 DeerMem 자신의 ``DeerMemConfig``에서 온다(생성 시 ``backend_config``
        로 설정된다). ``enabled``/``injection_enabled`` gate와 ``<memory>`` 감싸기는 호출
        지점(``_get_memory_context``)에 남으며, 여기서는 본문만 반환한다.
        """
        injection_agent = None if self.mode == "tool" else _resolve_agent_name(agent_name)
        memory_data = _call_backend(lambda: self._updater.get_memory_data(agent_name=injection_agent, user_id=user_id))
        return format_memory_for_injection(
            memory_data,
            max_tokens=self._config.max_injection_tokens,
            use_tiktoken=(self._config.token_counting == "tiktoken"),
            guaranteed_categories=self._config.guaranteed_categories,
            guaranteed_token_budget=self._config.guaranteed_token_budget,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """설정된 retrieval adapter로 검색한다.

        retrieval 에러가 정본 memory를 못 쓰게 만들지는 않는다. 기존의 대소문자 무시 부분
        문자열 경로가 최후의 fallback으로 남는다.
        """
        if not query or not query.strip() or top_k <= 0:
            return []
        resolved_agent_name = _resolve_agent_name(agent_name)
        indexed = self._fts5_search(query, top_k=top_k, user_id=user_id, agent_name=resolved_agent_name, category=category)
        if indexed:
            return indexed
        return self._substring_search(query, top_k=top_k, user_id=user_id, agent_name=resolved_agent_name, category=category)

    def _fts5_search(
        self,
        query: str,
        *,
        top_k: int,
        user_id: str | None,
        agent_name: str | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        """adapter 결과를 공개 fact 형태로 반환하는 호환용 helper."""
        agent_name = _resolve_agent_name(agent_name)
        search_facts = getattr(self._storage, "search_facts", None)
        scopes = [{"userId": user_id, "agentName": agent_name}]
        try:
            self._ensure_retrieval_scopes(scopes)
            indexed = (
                search_facts(
                    query,
                    scopes=scopes,
                    top_k=top_k,
                    mode="hybrid",
                    filters={"category": category} if category else None,
                )
                if callable(search_facts)
                else []
            )
        except Exception:
            logger.exception("Memory retrieval adapter failed; using substring fallback")
            indexed = []
        if indexed:
            return [_compat_document({"facts": [result.get("fact", result)]})["facts"][0] for result in indexed]

        return []

    def _substring_search(
        self,
        query: str,
        *,
        top_k: int,
        user_id: str | None,
        agent_name: str | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        query_lower = query.strip().lower()
        memory_data = _call_backend(lambda: self._updater.get_memory_data(agent_name=agent_name, user_id=user_id))
        matched = [fact for fact in memory_data.get("facts", []) if isinstance(fact.get("content"), str) and query_lower in fact["content"].lower() and (category is None or fact.get("category") == category)]
        matched.sort(key=_coerce_source_confidence, reverse=True)
        return _compat_document({"facts": matched[:top_k]})["facts"]

    def _ensure_retrieval_scopes(self, scopes: list[dict[str, str | None]]) -> None:
        """warm-up을 건너뛴 경우 요청된 모든 scope를 lazy하게 다시 만든다."""
        if not hasattr(self, "_retrieval_lock"):
            self._retrieval_lock = threading.RLock()
        if not hasattr(self, "_retrieval_warmed_scopes"):
            self._retrieval_warmed_scopes = set()
        if not hasattr(self, "_retrieval_fully_warmed"):
            self._retrieval_fully_warmed = False
        rebuild = getattr(self._storage, "rebuild_index", None)
        if not callable(rebuild):
            return
        with self._retrieval_lock:
            if self._retrieval_fully_warmed:
                return
            status = getattr(self._storage, "retrieval_status", lambda: {"configured": True})()
            if not status.get("configured", True):
                self._retrieval_warmed_scopes.update((scope.get("userId"), scope.get("agentName")) for scope in scopes)
                return
            for scope in scopes:
                key = (scope.get("userId"), scope.get("agentName"))
                if key in self._retrieval_warmed_scopes:
                    continue
                try:
                    result = rebuild([scope])
                except Exception:
                    logger.exception("Failed to lazily rebuild memory retrieval index for scope %r", key)
                    continue
                if result.get("supported") and not result.get("fatal"):
                    self._retrieval_warmed_scopes.add(key)

    # ── 관리 ─────────────────────────────────────────────────────────────
    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        memory_data = _call_backend(lambda: self._updater.get_memory_data(agent_name=_resolve_agent_name(agent_name), user_id=user_id))
        return _compat_document(memory_data)

    # delete_memory/export_memory는 base의 tier-2 기본값(NotImplementedError)을 그대로
    # 물려받는다. 호출자가 없는 죽은 contract이며(/memory/export는 get_memory를 거친다)
    # DeerMem이 raise를 다시 적지 않는다.

    def clear_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        if agent_name is None:
            memory_data = _call_backend(lambda: self._updater.clear_all_memory_data(user_id=user_id))
        else:
            memory_data = _call_backend(lambda: self._updater.clear_memory_data(agent_name=_resolve_agent_name(agent_name), user_id=user_id))
        return _compat_document(memory_data)

    def import_memory(
        self,
        memory_data: dict[str, Any],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        imported = _call_backend(
            lambda: self._updater.import_memory_data(
                memory_data,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
            )
        )
        return _compat_document(imported)

    # ── 생명주기 ─────────────────────────────────────────────────────────
    def shutdown_flush(self, timeout: float) -> bool:
        """graceful shutdown 시 ``timeout`` 안에 debounce 큐를 비운다.

        큐의 시간 제한이 있는 동기 flush에 위임한다. 그 flush는 먼저 실행 중인 worker를
        join해(debounce Timer가 이미 큐에서 꺼낸 context가 종료 시 유실되지 않도록) 처리하고,
        나머지는 실제 hard timeout을 둔 daemon 스레드에서 비운다. memory 업데이트 LLM 호출은
        동기라 중단할 수 없기 때문이다. ``timeout`` 안에 실제로 다 비운 경우에만 ``True``를
        반환한다.
        """
        return self._queue.flush_sync(timeout)

    def close(self) -> None:
        """대기 중인 업데이트가 모두 빠진 뒤 파생 retrieval 자원을 닫는다."""
        self._storage.close()

    # ── tier-3 hook (base 기본값을 override. warm/reload/fact CRUD) ──
    def warm(self) -> bool:
        """DeerMem의 토큰 계산 자원을 미리 워밍한다.

        base의 tier-3 hook을 override한다(기본값 None은 워밍할 것이 없다는 뜻). Gateway
        lifespan이 event loop 밖에서 ``manager.warm()``을 직접 호출한다. 무거운 초기화가 없는
        backend는 None 기본값을 물려받고 host가 "skipping"을 로그로 남긴다. encoding이
        로드됐거나 이미 캐시돼 있거나 워밍이 불필요하면 True, tiktoken을 쓸 수 없거나
        다운로드가 실패하면 False를 반환한다.
        """
        if self._config.token_counting == "char":
            logger.info("token_counting='char'; tiktoken not used, skipping warm-up")
            return True
        return warm_tiktoken_cache()

    def warm_retrieval(self) -> bool:
        """트래픽을 받기 전에 파생 retrieval 인덱스 전체를 다시 만든다."""
        rebuild = getattr(self._storage, "rebuild_index", None)
        if not callable(rebuild):
            return True
        try:
            result = rebuild()
            index_ok = not bool(result.get("fatal"))
            failed = int(result.get("failed") or 0)
            if failed and index_ok:
                logger.warning(
                    "Memory retrieval index rebuilt with %d fact(s) skipped",
                    failed,
                )
            if index_ok:
                with self._retrieval_lock:
                    self._retrieval_fully_warmed = True
                    self._retrieval_warmed_scopes.clear()
            return index_ok
        except Exception:
            logger.exception("Failed to rebuild memory retrieval index during warm-up")
            return False

    def reload_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """캐시된 memory 문서를 버리고 디스크에서 다시 읽는다."""
        memory_data = _call_backend(
            lambda: self._updater.reload_memory_data(
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
            )
        )
        return _compat_document(memory_data)

    def create_fact(
        self,
        content: str,
        category: str = "context",
        confidence: float = 0.5,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        memory_data, fact_id = _call_backend(
            lambda: self._updater.create_memory_fact(
                content,
                category=category,
                confidence=confidence,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
            )
        )
        return _compat_document(memory_data), fact_id

    def delete_fact(
        self,
        fact_id: str,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        memory_data = _call_backend(
            lambda: self._updater.delete_memory_fact(
                fact_id,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
            )
        )
        return _compat_document(memory_data)

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        memory_data = _call_backend(
            lambda: self._updater.update_memory_fact(
                fact_id,
                content=content,
                category=category,
                confidence=confidence,
                agent_name=_resolve_agent_name(agent_name),
                user_id=user_id,
            )
        )
        return _compat_document(memory_data)
