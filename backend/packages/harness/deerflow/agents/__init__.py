from .features import Next, Prev, RuntimeFeatures

__all__ = [
    "create_deerflow_agent",
    "RuntimeFeatures",
    "Next",
    "Prev",
    "make_lead_agent",
    "SandboxState",
    "ThreadState",
]


def __getattr__(name: str):
    if name == "create_deerflow_agent":
        from .factory import create_deerflow_agent

        globals()[name] = create_deerflow_agent
        return create_deerflow_agent
    if name == "make_lead_agent":
        from .lead_agent import make_lead_agent
        from .lead_agent.prompt import prime_enabled_skills_cache

        # LangGraph는 graph를 등록할 때 deerflow.agents:make_lead_agent를 해석한다.
        # 패키지 import 시점이 아니라 이 명시적 진입점에서 prime해야 가벼운 하위 모듈을
        # tool/subagent graph 전체를 끌어들이지 않고 import할 수 있다.
        prime_enabled_skills_cache()
        globals()[name] = make_lead_agent
        return make_lead_agent
    if name in {"SandboxState", "ThreadState"}:
        from .thread_state import SandboxState, ThreadState

        exports = {
            "SandboxState": SandboxState,
            "ThreadState": ThreadState,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
