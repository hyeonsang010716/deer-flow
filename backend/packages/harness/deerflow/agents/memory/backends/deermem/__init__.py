"""DeerMem backend — 기본 memory manager이며 독립적이다.

자체 manager 클래스(:mod:`deer_mem`)와 기능 모듈 다섯 개(storage/queue/updater/prompt/
message_processing)를 담은 ``core/`` 폴더를 가진다. DeerMem 전용 로직은 전부 여기에 있고,
공유 패키지 최상단은 contract와 factory, 얇은 진입점만 담는다.
"""

from .deer_mem import DeerMem

#: 이 backend가 노출하는 :class:`~deerflow.agents.memory.manager.MemoryManager` 서브클래스.
#: factory의 ``_scan_backends`` drop-in 메커니즘이 폴더 이름 ``deermem``으로 찾아낸다.
MANAGER_CLASS = DeerMem
