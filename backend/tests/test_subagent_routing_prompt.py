"""Prompt-contract tests for benefit-based subagent routing."""

from deerflow.agents.lead_agent import prompt as prompt_module
from deerflow.subagents.builtins.bash_agent import BASH_AGENT_CONFIG
from deerflow.subagents.builtins.general_purpose import GENERAL_PURPOSE_CONFIG
from deerflow.tools.builtins.task_tool import task_tool


def _build_section(monkeypatch, names: list[str] | None = None, max_concurrent: int = 3) -> str:
    monkeypatch.setattr(prompt_module, "get_available_subagent_names", lambda: names or ["general-purpose"])
    return prompt_module._build_subagent_section(max_concurrent)


def test_routing_requires_clear_net_benefit(monkeypatch) -> None:
    section = _build_section(monkeypatch)

    assert "기본은 직접 수행이다" in section
    assert "단지 작업이 복잡하거나, 여러 단계이거나, 장황한 출력을 내거나, 큰 repository를 다룬다는 이유만으로 위임하지 마라" in section
    assert "기대 이득이 기대 비용보다 명확히 클 때만 위임하라" in section
    assert "병렬 wall-clock 시간 절감" in section
    assert "전문성" in section
    assert "context 격리" in section
    assert "중복된 context 및 repository 탐색" in section
    assert "조율과 종합" in section
    assert "상태 충돌 위험" in section
    assert "side effect 위험" in section


def test_hard_vetoes_apply_to_parallel_dispatch_not_single_agent_chains(monkeypatch) -> None:
    section = _build_section(monkeypatch)

    assert "병렬 dispatch 절대 금지 조건" in section
    assert "agent 간 의존" in section
    assert "파일이 겹치거나, 가변 상태를 공유하거나, 담당이 분리되지 않은 외부 side effect" in section
    assert "한정된 순차 체인을 subagent 하나에 위임할 수는 있다" in section
    assert "위임 비용과 부정 신호" in section
    assert "중복 탐색" in section
    assert "저렴한 직접 경로" in section


def test_later_batches_retain_within_batch_parallel_benefit(monkeypatch) -> None:
    section = _build_section(monkeypatch)

    assert "매 batch가 끝날 때마다 남은 작업을 다시 판단하라" in section
    assert "이후 batch는 앞선 batch와 겹칠 수 없지만" in section
    assert "batch 내부의 실질적인 병렬 절감" in section
    assert "이득을 얻는 데 필요한 최소 수의 subagent만 사용하라" in section


def test_hard_limit_warning_is_emphatic_and_explains_lost_work(monkeypatch) -> None:
    section = _build_section(monkeypatch)

    assert "HARD LIMITS - 절대 협상 불가" in section
    assert "응답당 `task` 호출은 최대 3개 - 절대 더 내보내지 마라(NEVER)" in section
    assert "위반은 HARD ERROR다" in section
    assert "초과한 호출은 폐기되고 그 작업은 사라진다" in section


def test_multi_batch_example_preserves_reassessment_and_synthesis(monkeypatch) -> None:
    section = _build_section(monkeypatch)

    assert "다중 batch 예시(한도 3)" in section
    assert "Batch 1: 독립적인 범위를 최대 3개까지 실행한다" in section
    assert "해당 batch를 기다린 뒤 남은 작업과 순이득을 다시 판단한다" in section
    assert "Batch 2" in section
    assert "남긴 모든 결과를 종합한다" in section


def test_single_subagent_limit_omits_parallel_batch_guidance(monkeypatch) -> None:
    section = _build_section(monkeypatch, max_concurrent=1)

    assert "기대 이득 = 전문성 + context 격리" in section
    assert "실질적인 전문성 이득이나 context 격리 이득이 있을 때만 위임하라" in section
    assert "병렬 dispatch로 wall-clock 지연을 줄일 수 없다" in section
    assert "병렬 wall-clock 시간 절감" not in section
    assert "batch 내부의 실질적인 병렬 절감" not in section
    assert "다중 batch 예시" not in section
    assert "독립적인 provider 비교" not in section


def test_general_purpose_and_task_descriptions_match_routing_policy() -> None:
    tool_description = task_tool.description
    role_description = GENERAL_PURPOSE_CONFIG.description
    # 두 표면이 병렬 위임 범위를 같은 문구로 규정해야 한다.
    shared_parallel_policy = "독립적이고 겹치지 않는"

    assert "기대 이득" in tool_description
    assert "의존 관계가 있는 단계들을 여러 병렬 subagent로 쪼개지 마라" in tool_description
    assert shared_parallel_policy in tool_description
    assert shared_parallel_policy in role_description
    assert "명확한 위임 이득" in role_description
    assert "단지 순차적이라는 이유만으로" in role_description
    assert "한정된 의존 체인도 위임할 수 있다" in role_description


def test_bash_descriptions_require_benefit_beyond_routine_commands(monkeypatch) -> None:
    section = _build_section(monkeypatch, ["general-purpose", "bash"])
    policy = "일상적인 git, build, test, deploy 작업은 위임 사유로 충분하지 않다"

    assert policy in section
    assert policy in task_tool.description
    assert policy in BASH_AGENT_CONFIG.description
    assert "서로 의존하는 명령은 한 번에 하나씩 실행하라" in BASH_AGENT_CONFIG.system_prompt
