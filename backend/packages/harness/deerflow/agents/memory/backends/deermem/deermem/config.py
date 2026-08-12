"""DeerMem backend 설정. ``MemoryConfig.backend_config``에서 파싱한다.

DeerMem 전용 설정은 여기에 있고 공유 ``MemoryConfig``에는 두지 않는다. 공유 쪽은 host가
함께 쓰는 필드(``enabled``/``injection_enabled``/``manager_class``/``backend_config``)만 담는다.
factory가 dict인 ``backend_config``를 ``DeerMem.__init__``에 넘기고, 그것이 ``DeerMemConfig``로
파싱된다. 기본값 덕분에 ``backend_config`` 없이도 DeerMem이 동작한다.

필드 이름은 추상화 이전 ``MemoryConfig``의 비공개 필드를 그대로 따르므로 마이그레이션은 단순
이동이다(config.yaml ``memory.<field>`` -> ``memory.backend_config.<field>``). ``model``은
``core/llm.py``가 쓰는 중첩 ``DeerMemModelConfig``다(provider/model/api_key/base_url/temperature).
``should_keep_hidden_message``는 host가 주입하는 선택적 hook이다(None이면 DeerMem 기본값).
tracing은 DeerMemConfig 슬롯이 아니라 base의 ``MemoryManager.callbacks`` 필드로 처리한다
(LLM 호출 전 ``on_memory_llm_call``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class DeerMemModelConfig(BaseModel):
    """DeerMem의 memory 업데이트용 LLM 설정. langchain ``init_chat_model`` 인자다."""

    provider: str | None = Field(
        default=None,
        description="langchain model_provider, e.g. 'openai' (default when None). DeepSeek/other OpenAI-compatible gateways use 'openai' + base_url.",
    )
    model: str | None = Field(
        default=None,
        description="Model name. None = no LLM configured (non-LLM ops still work; an update raises).",
    )
    api_key: str | None = Field(default=None, description="API key (or rely on the provider's env var).")
    base_url: str | None = Field(default=None, description="Override base URL (e.g. an OpenAI-compatible gateway).")
    temperature: float | None = Field(default=None, description="Sampling temperature.")


class DeerMemConfig(BaseModel):
    """DeerMem 전용 설정. 독립적이며 host에 무관하다."""

    # ── 저장소 ───────────────────────────────────────────────────────────
    storage_path: str = Field(
        default="",
        description=("DeerMem data root. Empty = default (``$DEERMEM_DATA_DIR`` or ``~/.deermem/``); per-user memory at ``{root}/users/{user_id}/memory.json``. Any value (absolute or relative) is used as the root directory."),
    )
    storage_class: str = Field(
        default="",
        description="Dotted class path for an alternative storage provider; empty (default) = FileMemoryStorage (no importlib, portable).",
    )
    strict_user_scope: bool = Field(
        default=False,
        description="Require user_id for every storage scope. False preserves no-auth and legacy callers.",
    )
    manifest_filename: str = Field(
        default="memory.json",
        description="User-global summary JSON filename. Kept under this name for config compatibility; must be a plain .json filename.",
    )
    file_lock_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Maximum wait for the per-scope cross-process advisory file lock.",
    )
    retrieval_adapter: str = Field(
        default="fts5",
        description="Retrieval adapter factory: 'fts5' (default), an empty string to disable, or a dotted factory receiving DeerMemConfig and implementing RetrievalPort.",
    )
    # ── 큐 ───────────────────────────────────────────────────────────────
    debounce_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds to wait before processing queued updates (debounce).",
    )
    queue_max_depth: int = Field(
        default=1000,
        ge=0,
        description=("Backpressure cap on pending items. 0 = unlimited. When the cap is reached, new non-signal updates are rejected (QueueFull); signal updates are always admitted so important memories are never shed."),
    )
    # ── fact ─────────────────────────────────────────────────────────────
    max_facts: int = Field(default=100, ge=10, le=500, description="Maximum number of facts to store.")
    fact_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for storing facts.",
    )
    # ── 주입 ─────────────────────────────────────────────────────────────
    max_injection_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens to use for memory injection.",
    )
    token_counting: Literal["tiktoken", "char"] = Field(
        default="tiktoken",
        description=("Token counting strategy for memory-injection budgeting. 'tiktoken' is accurate but may download BPE data on first use; 'char' is network-free CJK-aware estimation."),
    )
    guaranteed_categories: list[str] = Field(
        default_factory=lambda: ["correction"],
        description="Fact categories always injected regardless of the regular token budget.",
    )
    guaranteed_token_budget: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Token ceiling for guaranteed-category facts.",
    )
    # ── staleness 검토 ───────────────────────────────────────────────────
    staleness_review_enabled: bool = Field(
        default=True,
        description="Enable staleness review for aged facts.",
    )
    staleness_age_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Facts older than this become staleness-review candidates.",
    )
    staleness_min_candidates: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Minimum stale facts required to trigger a review cycle.",
    )
    staleness_max_removals_per_cycle: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum facts the staleness review can remove per cycle.",
    )
    staleness_protected_categories: list[str] = Field(
        default_factory=lambda: ["correction"],
        description="Fact categories exempt from staleness review.",
    )
    staleness_max_lifetime_multiplier: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description=(
            "Creation-time cap multiplier for a fact's LLM-assigned "
            "expected_valid_days. When a new fact is stored, its "
            "expected_valid_days is clamped to "
            "staleness_age_days * staleness_max_lifetime_multiplier so the "
            "model cannot set an initial lifetime so long that the fact is "
            "never re-evaluated. Default 20.0 (90 x 20 = 1800 d ~= 5 years) "
            "is generous enough to support the 'very stable' prompt tier "
            "(core skills, native language) without needing multiple review "
            "cycles to escape the cap. Lifetime extensions (staleFactsToExtend) "
            "are subject to staleness_max_extension_days instead."
        ),
    )
    staleness_max_extension_days: int = Field(
        default=3650,
        ge=90,
        le=36500,
        description=(
            "Absolute upper bound (in days) on expected_valid_days after a "
            "lifetime extension (staleFactsToExtend). Applied at write time "
            "during staleness review: new_evd = min(days_since + extend_by, "
            "staleness_max_extension_days). Separate from the creation-time "
            "multiplier cap because extensions are deliberate recalibration "
            "decisions and are not subject to the staleness_age_days scale. "
            "The ceiling prevents a single LLM misfire from permanently "
            "deferring a fact or causing timedelta overflow on the next "
            "candidate-selection pass. Default 3650 (10 years)."
        ),
    )
    # ── memory 통합 ──────────────────────────────────────────────────────
    consolidation_enabled: bool = Field(
        default=False,
        description=(
            "Enable memory consolidation. When enabled, the LLM reviews "
            "fragmented fact categories during the normal memory-update call "
            "(same invocation - no extra API call) and decides whether groups "
            "of related facts can be synthesized into a single richer fact. "
            "Defaults to False because consolidation is lossy (source content "
            "is not preserved, only consolidatedFrom IDs). Opt in explicitly "
            "once the memory-file backup / audit story is in place."
        ),
    )
    consolidation_min_facts: int = Field(
        default=8,
        ge=3,
        le=30,
        description=("Minimum number of facts in a single category to trigger consolidation review. Below this threshold the overhead of surfacing the group is not justified."),
    )
    consolidation_max_groups_per_cycle: int = Field(
        default=3,
        ge=1,
        le=10,
        description=("Maximum number of consolidation groups the LLM can merge in a single update cycle. Prevents over-consolidation."),
    )
    consolidation_max_sources: int = Field(
        default=8,
        ge=2,
        le=20,
        description=("Maximum number of source facts per consolidation group. Prevents the LLM from merging too many facts into one and losing important details."),
    )
    # ── 추출 품질 callback (호출 이후 관측) ──────────────────────────────
    extraction_callback: Any = Field(
        default=None,
        description=(
            "Optional ``callback(metrics)`` invoked AFTER the extraction LLM "
            "call (token usage, facts passing/rejected by the confidence "
            "filter, rejection rate, prompt version). The host injects a "
            "Langfuse-based callback to emit an extraction span; None = no "
            "post-invoke observability. Set programmatically (not from YAML)."
        ),
    )
    # ── watermark 캐시 (인메모리, 크기 제한 LRU) ─────────────────────────
    watermark_max_keys: int = Field(
        default=4096,
        ge=0,
        description=(
            "Soft cap on the in-memory conversation-watermark cache (one entry "
            "per distinct thread/user/agent). The cache is a bounded LRU: when "
            "over capacity the least-recently-used entry is dropped, and a "
            "dropped key re-extracts one batch on that thread's next turn (the "
            "same as a restart). 0 = unbounded."
        ),
    )
    # ── 메시지 처리 (외부화된 pattern/prompt) ──
    patterns_dir: str | None = Field(
        default=None,
        description=("Directory with correction.yaml / reinforcement.yaml overriding the bundled signal-detection patterns. None (default) = bundled core/message_patterns/. When set explicitly, both files must exist."),
    )
    prompts_dir: str | None = Field(
        default=None,
        description=("Directory with custom memory-extraction prompt templates (memory_update.chat.yaml, staleness_review.yaml, consolidation.yaml, fact_extraction.yaml). None (default) = bundled core/prompts/."),
    )
    # ── LLM (core/llm.py의 build_llm이 쓰는 구조화된 model 하위 설정) ──
    model: DeerMemModelConfig = Field(
        default_factory=DeerMemModelConfig,
        description=(
            "Memory-update LLM config (provider/model/api_key/base_url/temperature). "
            "Empty = the host factory injects its default chat model as ``host_llm`` "
            "(zero-config UX, mirrors pre-abstraction ``model_name: null``); "
            "when ``host_llm`` is also absent (standalone DeerMem) an update raises "
            "but non-LLM ops still work."
        ),
    )
    # ── hook (host가 주입하는 선택적 callable. None이면 DeerMem 기본값) ──
    # tracing은 DeerMemConfig 슬롯이 아니라 base의 ``MemoryManager.callbacks`` 필드로
    # 처리한다(LLM 호출 전 ``on_memory_llm_call``).
    should_keep_hidden_message: Any = Field(
        default=None,
        description=("Optional ``hook(additional_kwargs) -> bool``; when set, ``hide_from_ui`` messages are kept if it returns True. None = skip all ``hide_from_ui`` (host-agnostic safe default). Set programmatically."),
    )
    host_llm: Any = Field(
        default=None,
        description=(
            "Host-injected pre-built chat model for memory extraction (zero-config "
            "UX). The deer-flow factory injects its default model here when "
            "``model`` is empty, mirroring pre-abstraction ``model_name: null`` -> "
            "app default. Takes precedence over ``build_llm(model)``. None = build "
            "from ``model`` (or no LLM when ``model`` is also empty). Set "
            "programmatically (an instance cannot come from YAML)."
        ),
    )
    trace_context_manager: Any = Field(
        default=None,
        description=(
            "Host-injected context-manager callable ``cm(trace_id)`` that binds "
            "``trace_id`` into the host request-trace ContextVar for the memory-"
            "update worker thread (Timer / executor), restoring structured-log "
            "trace correlation. None = no binding (DeerMem standalone; trace_id "
            "still reaches ``on_memory_llm_call`` and the log message text). Set "
            "programmatically."
        ),
    )

    @model_validator(mode="after")
    def _check_storage_path_is_directory(self) -> DeerMemConfig:
        """DeerMem은 ``storage_path``를 루트 디렉터리로 다룬다.

        사용자별 memory는 ``{storage_path}/users/{uid}/memory.json``에 있다. 추상화 이전
        파일 경로 의미에서 남은 ``.json`` 같은 파일 형태 값이 들어오면
        ``FileMemoryStorage.save``의 ``mkdir(parents=True)``가 ``NotADirectoryError``를
        던지고, 이는 OSError로 잡혀 조용한 쓰기 실패가 된다. 그래서 생성 시점에 크게 실패시킨다.
        memory는 영속 상태라 잘못된 루트는 데이터 무결성 사고로 이어진다. host factory가 아니라
        여기에 두어 DeerMem을 factory 없이 단독 생성해도 검증이 동작한다.
        빈 storage_path는 허용한다(설정 없이 host가 디렉터리를 주입하는 경우).
        """
        if self.storage_path:
            resolved = Path(self.storage_path)
            if resolved.is_file():
                raise ValueError(
                    f"memory.backend_config.storage_path={self.storage_path!r} "
                    f"resolves to an existing file {resolved}; DeerMem treats "
                    f"storage_path as a root DIRECTORY (per-user memory under "
                    f"{{storage_path}}/users/{{uid}}/memory.json). Point it at a directory."
                )
        return self

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any] | None) -> DeerMemConfig:
        """``backend_config`` dict를 파싱한다.

        알 수 없는 key는 상위 호환을 위해 무시하되 WARNING으로 남긴다. 그래야 오타
        (예: ``h``가 빠진 ``storage_pat``)가 조용히 기본값으로 넘어가 의도치 않은 위치에
        memory를 쓰는 일이 없다. host 계층의 ``load_memory_config_from_dict`` 경고와 같다.

        ``None`` 값은 버려서 필드 기본값으로 넘어가게 한다. YAML은 빈 key
        (``config.example.yaml``에 있는, 자식이 전부 주석인 ``model:``)를 ``None``으로 만드는데,
        key를 아예 생략하는 것은 유효한데도 ``model`` 같은 non-Optional 필드는 이를 거부하기
        때문이다.
        """
        if not backend_config:
            return cls()
        backend_config = dict(backend_config)
        known = {k: v for k, v in backend_config.items() if k in cls.model_fields and v is not None}
        unknown = sorted(k for k in backend_config if k not in cls.model_fields)
        if unknown:
            logger.warning(
                "Unknown backend_config keys ignored by DeerMem; check for typos: %s",
                unknown,
            )
        return cls(**known)
