"""DeerFlow의 교체 가능한 memory.

backend에 무관한 공유 코어다. :class:`MemoryManager` contract, :func:`get_memory_manager`
singleton factory, :func:`reset_memory_manager`를 담는다. backend는 :mod:`backends` 아래에
있고 각각 독립적으로 ``MANAGER_CLASS``를 노출한다. 기본 DeerMem backend의 기능 모듈은
``backends/deermem/core/``에 있다. backend 교체는 ``backends/<name>/`` 폴더를 넣고
``MemoryConfig.manager_class``를 설정하면 끝이며, deer-flow의 다른 곳은 바뀌지 않는다.

DeerMem 전용 심볼(``format_memory_for_injection``, ``get_memory_data``, ``MemoryUpdater``,
``FileMemoryStorage`` 등)은 여기서 다시 export하지 않는다.
``deerflow.agents.memory.backends.deermem.deermem.core.*``에서 직접 import한다.
"""

from deerflow.agents.memory.manager import (
    MemoryConflictError,
    MemoryCorruptionError,
    MemoryManager,
    MemoryManagerError,
    get_memory_manager,
    reset_memory_manager,
)

__all__ = [
    "MemoryManager",
    "MemoryManagerError",
    "MemoryConflictError",
    "MemoryCorruptionError",
    "get_memory_manager",
    "reset_memory_manager",
]
