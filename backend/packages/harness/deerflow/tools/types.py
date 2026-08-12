from typing import Any

from langchain.tools import ToolRuntime

from deerflow.agents.thread_state import ThreadState

# 모든 DeerFlow tool이 쓰는 구체 runtime 타입.
# context 파라미터에 bound되지 않은 ContextT TypeVar 대신 dict[str, Any]를 쓰면, LangChain이
# tool의 자동 생성 args_schema에 model_dump()를 호출할 때 나오는
# PydanticSerializationUnexpectedValue 경고를 막을 수 있다.
Runtime = ToolRuntime[dict[str, Any], ThreadState]
