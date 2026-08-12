"""mem0 백엔드 config — ``backend_config``를 파싱하고 검증한다.

noop 템플릿 패턴을 따른다: 평범한 dataclass + ``from_backend_config``.
host는 모든 백엔드의 config dict에 ``storage_path``(그리고 경우에 따라
``should_keep_hidden_message``)를 주입하므로 그 키들은 받아들이되 무시한다.
그 외의 알 수 없는 키는 거부한다. 영속 상태 설정의 오타는 조용히 기본값으로
떨어지지 말고 즉시 실패해야 한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

#: host factory가 backend_config에 주입하는 키. 받아들이되 무시한다.
_HOST_INJECTED_KEYS = frozenset({"storage_path", "should_keep_hidden_message"})

_STARTUP_POLICIES = frozenset({"fail_fast", "tolerate"})
_READ_POLICIES = frozenset({"fail_open", "fail_closed"})
_WRITE_POLICIES = frozenset({"log_and_drop", "raise"})


@dataclass(frozen=True)
class Mem0Config:
    """mem0 HTTP 백엔드의 검증된 설정값."""

    #: mem0 API key를 담은 환경 변수 이름. key 자체는 config.yaml에 절대 넣지 않는다.
    api_key_env: str = "MEM0_API_KEY"
    #: mem0 Platform API 루트. on-prem이면 self-hosted 서버를 가리킨다.
    base_url: str = "https://api.mem0.ai"
    #: 평문 HTTP로 API token 전송을 허용한다. 신뢰된 로컬 개발망 전용이다.
    allow_insecure_http: bool = False
    #: get_context가 주입하는 최대 memory 수이자 기본 search 폭(1-1000).
    top_k: int = 8
    #: search() 결과의 최소 관련도 점수(mem0 `threshold`, 0-1).
    score_threshold: float = 0.1
    #: get_context가 반환하는 주입 텍스트의 상한. 통째로 들어가지 않는 memory는
    #: 건너뛴다(잘림은 항상 항목 경계에서 일어난다).
    max_injection_chars: int = 12000
    #: 요청당 HTTP timeout(초).
    timeout_seconds: float = 10.0
    #: "fail_fast" = from_config에서 인증 확인, "tolerate" = 첫 사용 시점으로 미룬다.
    startup_policy: str = "fail_fast"
    #: "fail_open" = recall 오류 시 아무것도 주입하지 않고 계속 진행한다.
    #: "fail_closed" = recall 오류 시 MemoryManagerError를 던진다.
    read_policy: str = "fail_open"
    #: "log_and_drop" = 쓰기 오류를 로그로 남기고 버린다(at-most-once).
    #: "raise" = 쓰기 오류 시 MemoryManagerError를 던진다.
    write_policy: str = "log_and_drop"

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any] | None) -> Mem0Config:
        cfg = dict(backend_config or {})
        failure_policy = cfg.pop("failure_policy", {}) or {}
        unknown = (
            set(cfg)
            - {
                "api_key_env",
                "base_url",
                "allow_insecure_http",
                "top_k",
                "score_threshold",
                "max_injection_chars",
                "timeout_seconds",
                "startup_policy",
            }
            - _HOST_INJECTED_KEYS
        )
        if unknown:
            raise ValueError(f"mem0 backend_config has unknown keys: {sorted(unknown)}")
        if not isinstance(failure_policy, dict):
            raise ValueError("mem0 failure_policy must be a mapping {read, write}")
        unknown_fp = set(failure_policy) - {"read", "write"}
        if unknown_fp:
            raise ValueError(f"mem0 failure_policy has unknown keys: {sorted(unknown_fp)}")
        allow_insecure_http = cfg.get("allow_insecure_http", False)
        if not isinstance(allow_insecure_http, bool):
            raise ValueError("mem0 allow_insecure_http must be a boolean")

        config = cls(
            api_key_env=str(cfg.get("api_key_env", "MEM0_API_KEY")),
            base_url=str(cfg.get("base_url", "https://api.mem0.ai")).rstrip("/"),
            allow_insecure_http=allow_insecure_http,
            top_k=int(cfg.get("top_k", 8)),
            score_threshold=float(cfg.get("score_threshold", 0.1)),
            max_injection_chars=int(cfg.get("max_injection_chars", 12000)),
            timeout_seconds=float(cfg.get("timeout_seconds", 10.0)),
            startup_policy=str(cfg.get("startup_policy", "fail_fast")),
            read_policy=str(failure_policy.get("read", "fail_open")),
            write_policy=str(failure_policy.get("write", "log_and_drop")),
        )
        if config.startup_policy not in _STARTUP_POLICIES:
            raise ValueError(f"mem0 startup_policy must be one of {sorted(_STARTUP_POLICIES)}")
        if config.read_policy not in _READ_POLICIES:
            raise ValueError(f"mem0 failure_policy.read must be one of {sorted(_READ_POLICIES)}")
        if config.write_policy not in _WRITE_POLICIES:
            raise ValueError(f"mem0 failure_policy.write must be one of {sorted(_WRITE_POLICIES)}")
        if not 1 <= config.top_k <= 1000:
            raise ValueError("mem0 top_k must be in [1, 1000]")
        if not 0.0 <= config.score_threshold <= 1.0:
            raise ValueError("mem0 score_threshold must be in [0, 1]")
        if config.max_injection_chars <= 0:
            raise ValueError("mem0 max_injection_chars must be positive")
        if config.timeout_seconds <= 0:
            raise ValueError("mem0 timeout_seconds must be positive")
        if not config.api_key_env.strip():
            raise ValueError("mem0 api_key_env must be a non-empty env var name")
        parsed_base_url = urlsplit(config.base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("mem0 base_url must be an absolute http:// or https:// URL")
        if parsed_base_url.scheme == "http" and not config.allow_insecure_http:
            raise ValueError("mem0 base_url must use https:// because it carries the API key; set allow_insecure_http: true only for trusted local development")
        return config

    def resolve_api_key(self) -> str:
        """설정된 환경 변수에서 API key를 읽는다."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ValueError(f"mem0 API key missing: environment variable {self.api_key_env} is unset or empty")
        return key
