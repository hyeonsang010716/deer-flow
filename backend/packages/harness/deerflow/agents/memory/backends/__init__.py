"""교체 가능한 memory backend.

각 하위 패키지는 자기 ``__init__``에 ``MANAGER_CLASS``
(:class:`~deerflow.agents.memory.manager.MemoryManager` 서브클래스)를 노출하는 독립적인
backend다. drop-in contract는 폴더 이름 == backend 이름 == ``MemoryConfig.manager_class`` 값이다.

새 backend는 여기에 폴더를 하나 넣고 ``manager_class: <name>``만 설정하면 된다.
deer-flow의 다른 코드는 바뀌지 않는다.
"""
