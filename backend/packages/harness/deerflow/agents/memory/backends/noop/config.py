"""Noop 백엔드 config — ``backend_config`` 파싱 템플릿.

새 memory 백엔드가 자기 설정을 어떻게 읽는지 보여주는 레퍼런스다.
**이식성 원칙**(백엔드를 작성하기 전에 반드시 읽는다):

    백엔드는 host가 주는 모든 정보를 정확히 두 경로로만 받는다.
      1. :class:`MemoryManager` ABC 메서드 인자(``manager.py``) —
         ``user_id`` / ``agent_name`` / ``thread_id`` / ``messages`` 등.
      2. ``__init__``에 전달되는 ``backend_config`` dict.
    deer-flow 모듈을 import하거나 deer-flow 경로를 하드코딩하면 안 된다. 백엔드 폴더
    전체에서 허용되는 유일한 ``from deerflow`` 줄은 ``<name>_manager.py``의 ABC 계약
    import뿐이다::

        from deerflow.agents.memory.manager import MemoryManager

    이 한 줄이 백엔드를 host에 묶는다. 다른 에이전트로 이식할 때는 이 줄만 바꾸면
    된다. storage 루트, model, 훅 등 나머지는 전부 ``backend_config``로 들어온다.

factory(``manager.py::get_memory_manager``)가 각 백엔드에 제공하는 것:
  - ``backend_config["storage_path"]`` (str): 쓰기 가능한 상태 디렉터리(host 기본값
    또는 사용자가 config.yaml에 지정한 값). **이것을 storage 루트로 쓴다.**
    deer-flow 경로 헬퍼를 직접 호출하면 안 된다.
  - host 훅(``from_config``의 kwargs로 전달되며 backend_config에는 없다):
    ``callbacks``(``on_memory_llm_call``로 tracing하는 ``MemoryCallbacks``),
    ``should_keep_hidden_message``, ``trace_context_manager``,
    ``host_llm_factory``. ``from_config``에서 필요한 것만 쓰고 나머지는 무시한다.
  - 여기에 사용자의 ``config.yaml::memory.backend_config`` 키들(백엔드 고유 설정:
    ``model``, ``vector_store``, ``embedder``, 각종 임계값 등)이 더해진다.

아래 ``NoopConfig``가 그 표면을 그대로 반영한다. noop은 아무것도 저장하지 않아 모든
필드를 무시하지만, 이 구조를 복사해 이름을 바꾸고 자기 설정을 채워 넣으면 된다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class NoopConfig:
    """noop 백엔드의 파싱된 config(템플릿이며 noop은 모든 필드를 무시한다).

    실제 백엔드는 여기에 자기 설정(예: ``model``, ``vector_store``, ``max_facts``)을
    선언하고 :meth:`from_backend_config`에서 파싱한다.
    """

    #: host가 주입하는 쓰기 가능한 상태 디렉터리. 실제 백엔드는 이 아래에 저장소
    #: (DB / vector store / JSON)를 둔다. noop은 무시한다.
    storage_path: str = ""

    #: 백엔드 전용 설정 예시(config.yaml의
    #: ``memory.backend_config.example_option``에서 온다). 자기 것으로 교체한다.
    example_option: str = "default"

    #: host가 주입하는 선택적 훅. ``hide_from_ui`` 메시지를 거르는 백엔드는
    #: ``self._config.should_keep_hidden_message(additional_kwargs)``를 호출한다
    #: (True면 hide_from_ui여도 유지). ``None``이면 hidden 메시지를 전부 건너뛴다.
    should_keep_hidden_message: Callable[[Any], bool] | None = None

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any] | None) -> NoopConfig:
        """``backend_config`` dict로부터 config를 만든다.

        manager의 ``model_post_init``에서 이렇게 쓴다::

            self._config = YourConfig.from_backend_config(self.backend_config)

        알려진 키만 읽고 나머지는 무시한다. 덕분에 host가 모든 백엔드의
        ``backend_config``에 ``storage_path``를 주입해도 그것을 쓰지 않는 백엔드가
        깨지지 않는다. (tracing 같은 host 훅은 ``backend_config``가 아니라
        ``from_config``의 kwargs로 들어온다.)
        """
        cfg = dict(backend_config or {})
        return cls(
            storage_path=str(cfg.get("storage_path") or ""),
            example_option=str(cfg.get("example_option", "default")),
            should_keep_hidden_message=cfg.get("should_keep_hidden_message"),
        )
