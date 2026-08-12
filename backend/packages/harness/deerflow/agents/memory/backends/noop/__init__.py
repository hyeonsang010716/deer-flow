"""Noop memory 백엔드 — 동작하는 빈 adapter(플러그인 구조 검증 + 템플릿)."""

from .noop_manager import NoopMemoryManager

#: 이 백엔드가 노출하는
#: :class:`~deerflow.agents.memory.manager.MemoryManager` 하위 클래스.
#: 폴더명 ``noop``으로 factory의 ``_scan_backends`` drop-in 메커니즘이 발견한다.
MANAGER_CLASS = NoopMemoryManager
