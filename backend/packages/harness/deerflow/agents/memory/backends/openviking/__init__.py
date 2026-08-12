"""공식 LangChain 통합을 사용하는 OpenViking memory 백엔드."""

from .openviking_manager import OpenVikingMemoryManager

MANAGER_CLASS = OpenVikingMemoryManager

__all__ = ["OpenVikingMemoryManager"]
