"""mem0 memory backend — mem0 Platform API를 호출하는 HTTP client.

Drop-in 규약: 폴더명 == 백엔드명 == ``manager_class: mem0``.
"""

from .mem0_manager import Mem0Manager

#: 폴더명 ``mem0``으로 factory의 ``_scan_backends``가 발견한다.
MANAGER_CLASS = Mem0Manager
