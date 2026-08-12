"""커스텀 agent 설정과 로더.

커스텀 agent는 ``{base_dir}/users/{user_id}/agents/{name}/`` 아래에 사용자별로 저장한다.
사용자 격리 이전에 설치된 환경이 ``scripts/migrate_user_isolation.py`` 마이그레이션을 돌리기
전까지 계속 동작하도록 레거시 공유 레이아웃(``{base_dir}/agents/{name}/``)도 읽을 수 있다.
새 쓰기는 항상 사용자별 레이아웃을 대상으로 한다.
"""

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)

SOUL_FILENAME = "SOUL.md"
AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
MAX_AGENT_OUTPUT_TOKENS = 200_000


def _blank_to_none(value: str | None) -> str | None:
    """공백뿐인 문자열을 ``None``으로 정규화한다. 실제 값은 그대로 둔다.

    공백만 있는 문자열(예: ``"   "``)도 Python에서는 truthy라, strip하지 않은
    ``value or fallback`` 식은 절대 fallback으로 넘어가지 않는다. ``require_mention`` 의
    우선순위 체인(``trigger.mention_login`` -> ``github.bot_login`` ->
    ``channels.github.default_mention_login`` -> ``agent.name``, AGENTS.md 참고)이 바로 그
    fallthrough에 의존하므로, 설정에서 오는 두 링크를 모델 계층에서 한 번에 정규화한다.
    그러면 이후 모든 소비자(현재와 미래)가 실제 ``@mention`` 과 결코 일치할 수 없는 공백 문자열
    대신 정직한 "unset"을 보게 된다.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class GitHubTriggerConfig(BaseModel):
    """:class:`GitHubBinding` 안의 이벤트별 trigger 필터."""

    # 지정하면 이 GitHub action 값들만 agent를 발동시킨다. None이면 "모든 action 허용"이다.
    # 예: pull_request에 ["opened"]를 주면 새로 열린 PR에만 반응한다.
    actions: list[str] | None = None
    # True면 comment 본문에 bot login이 @-mention된 경우에만 comment 이벤트가 발동한다.
    # comment가 아닌 이벤트에서는 무시된다.
    require_mention: bool = False
    # 이 GitHub login들의 이벤트는 require_mention을 우회한다.
    # repo 소유자가 매번 핸들을 치지 않고도 bot과 대화할 수 있게 한다.
    allow_authors: list[str] = Field(default_factory=list)
    # 이 trigger에 한해 전역 기본 bot mention login을 덮어쓴다.
    # 한 agent는 @bot-a로, 다른 agent는 @bot-b로 답할 때 유용하다. 공백뿐인 값은 None으로
    # 정규화되므로(``_blank_to_none`` 참고) 문자 그대로 비교되지 않고 unset으로 취급되어
    # ``github.bot_login`` 으로 넘어간다.
    mention_login: str | None = None

    @field_validator("mention_login")
    @classmethod
    def _normalize_mention_login(cls, value: str | None) -> str | None:
        return _blank_to_none(value)


class GitHubBinding(BaseModel):
    """이벤트별 trigger 오버라이드를 갖는 (agent, repo) 바인딩 하나."""

    # GitHub "owner/name" 문자열.
    repo: str
    # 이벤트 이름 → trigger 오버라이드. 키가 없으면 해당 이벤트에 대한
    # dispatcher 기본 trigger를 쓴다.
    triggers: dict[str, GitHubTriggerConfig] = Field(default_factory=dict)


class GitHubAgentConfig(BaseModel):
    """커스텀 agent ``config.yaml`` 최상위의 ``github:`` 블록."""

    # repo별 access token을 발급할 때 쓰는 GitHub App installation id.
    # ``ChannelManager`` 가 이 값으로 1시간짜리 installation token을 만들어
    # ``run_context["github_token"]`` 에 주입하고, ``bash`` 도구가 이를 agent sandbox에
    # ``GH_TOKEN`` / ``GITHUB_TOKEN`` 으로 노출한다. 그러면 agent가 직접 ``gh`` 로 repo 상태를 읽고
    # branch를 push하고 comment를 남긴다. None이면 token을 발급하지 않는다. agent는 여전히 돌지만
    # push나 comment를 할 수 없다(공개 repo는 인증 없는 ``gh`` 로 읽기만, 비공개 repo는 아예 접근 불가).
    installation_id: int | None = None
    # 이 agent가 글을 쓸 때 사용하는 GitHub App login
    # (예: ``llm-gateway-ai[bot]`` App 신원이면 ``[bot]`` 접미사를 뺀 ``llm-gateway-ai``).
    # dispatcher의 self-event 게이트가 이 값으로, agent가 trigger 매칭에 어떤 ``mention_login`` 을
    # 쓰든 상관없이 자기 활동이 유발한 webhook 전달을 알아본다. None이면
    # "mention_login / agent 이름으로 fallback"이며, 그 값들이 bot 신원과 같으면 괜찮지만
    # 다르면 명시적으로 지정해야 한다. 공백뿐인 값은 None으로 정규화되어(``_blank_to_none`` 참고)
    # unset으로 취급되고 나머지 체인으로 넘어간다.
    bot_login: str | None = None
    # github 채널 기본 ``recursion_limit``(250)을 덮어쓴다. GitHub 실행은 본래 자율적이고
    # 오래 걸린다(clone, 탐색, 편집, 테스트, push, comment). 하지만 적절한 상한은 작업 성격에 따라
    # 크게 다르다. 리뷰 전용 agent는 50이면 충분하고, 다중 파일 리팩터링 agent는 500 이상이 필요할 수 있다.
    # None이면 "채널 기본값(250)"을 쓴다. 양의 정수는 채널 기본값이나 전역 100 스텝 하한보다
    # 낮더라도 그대로 존중하므로, ``recursion_limit: 50`` 같은 명시적 안전 설정은 실제로 50
    # super-step에서 agent를 멈춘다. 0 이하 값은 무시한다(None으로 취급).
    # 음수나 0이면 첫 스텝도 밟기 전에 agent가 멈춰 버리기 때문이다.
    recursion_limit: int | None = None
    # 이 agent가 바인딩된 repo들. 빈 리스트면 아무것도 바인딩되지 않은 것이라,
    # ``github:`` 블록이 있어도 webhook으로 발동하지 않는다.
    bindings: list[GitHubBinding] = Field(default_factory=list)

    @field_validator("bot_login")
    @classmethod
    def _normalize_bot_login(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @model_validator(mode="after")
    def _unique_binding_repos(self) -> "GitHubAgentConfig":
        """``bindings`` 안에 중복된 ``repo`` 값이 있으면 거부한다.

        repo당 binding은 최대 하나다. 하나의 binding에 있는 이벤트별 ``triggers`` 맵이 이미
        "이 agent는 이 repo에서 N개 이벤트를 듣는다"를 표현하므로, 같은 repo에 binding이 여러 개면
        이벤트가 중복되거나(조용한 first-wins / 이중 등록. PR 피드백 R3 참고) 아무 이득 없이
        여러 row로 쪼개질 뿐이다. 초기 구현이고 중복 repo binding에 의존하는 기존 운영 설정도 없으므로,
        dispatch 시점에 모호함을 덮는 대신 설정 로드 시점에 크게 실패시킨다.
        """
        seen: set[str] = set()
        dupes: set[str] = set()
        for binding in self.bindings:
            if binding.repo in seen:
                dupes.add(binding.repo)
            seen.add(binding.repo)
        if dupes:
            raise ValueError(f"Agent github.bindings has duplicate repos {sorted(dupes)}. Each repo must appear at most once — merge their `triggers:` maps into a single binding.")
        return self


def validate_agent_name(name: str | None) -> str | None:
    """파일시스템 경로에 쓰기 전에 커스텀 agent 이름을 검증한다."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("Invalid agent name. Expected a string or None.")
    if not AGENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid agent name '{name}'. Must match pattern: {AGENT_NAME_PATTERN.pattern}")
    return name


class AgentModelSettings(BaseModel):
    """model profile 위에 얹는 agent별 LLM sampling 오버라이드.

    DeerFlow 런타임 스위치(``thinking_enabled`` 같은 것)가 아니라 provider의 sampling 옵션이다.
    *같은* ``models:`` profile을 참조하는 두 agent가 서로 다른 temperature / 출력 길이로 돌 수 있게 한다.
    "agent마다 능력이 다르므로 temperature를 공유하는 건 맞지 않다"는 issue #4336의 핵심 요구다.

    ``extra="forbid"``: sampling 표면을 명시적 allowlist로 둬서, 엉뚱한 키가 provider 요청 본문까지
    흘러가 요청 시점에 알아보기 힘든 에러로 실패하는 일을 막는다. 넓히려면 model config를 느슨하게
    하지 말고 필드(예: ``top_p``)를 선언해서 추가한다. 모든 필드는 선택이며 ``None``은
    "profile 값을 덮어쓰지 않는다"는 뜻이다.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature override (0.0-2.0). None = inherit the model profile's value.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAX_AGENT_OUTPUT_TOKENS,
        description=f"Max output tokens override (1-{MAX_AGENT_OUTPUT_TOKENS}). None = inherit the model profile's value.",
    )


class AgentConfig(BaseModel):
    """커스텀 agent 설정."""

    name: str
    description: str = ""
    model: str | None = None
    tool_groups: list[str] | None = None
    # skills는 agent가 발견하고 활성화할 수 있는 skill을 제어한다.
    # 생성 시점에 해당 skill의 allowed-tools 정책을 활성화하지는 않는다.
    # - None(또는 생략): 활성화된 모든 skill을 로드한다(기본 fallback 동작)
    # - [](명시적 빈 리스트): 모든 skill을 끈다
    # - ["skill1", "skill2"]: 지정한 skill만 로드한다
    skills: list[str] | None = None
    # 참조하는 model profile 위에 얹는 agent별 LLM sampling 오버라이드(temperature / max_tokens).
    # None이면 오버라이드 없음(issue #4336).
    model_settings: AgentModelSettings | None = None
    # agent별 thinking 모드 기본값. None이면 런타임 기본값을 덮어쓰지 않는다
    # (요청에 담긴 thinking 플래그는 여전히 이 값보다 우선한다).
    thinking_enabled: bool | None = None
    # 지원 모델을 위한 agent별 reasoning-effort 기본값. None이면 덮어쓰지 않는다
    # (요청에 담긴 reasoning_effort가 여전히 우선한다).
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    # 이 agent가 gateway dispatcher의 webhook 이벤트에 반응할 수 있도록 GitHub repo에 거는
    # 선택적 바인딩. None이면 "GitHub 연동 없음"이며, 기존 agent는 모두 여기에 해당한다.
    github: GitHubAgentConfig | None = None


# agent 갱신 표면이 명시적으로 관리하는 필드들. :class:`AgentConfig` 에 선언된 나머지
# (현재는 ``github``, 그리고 앞으로 추가될 필드)는 :func:`preserve_non_managed_fields` 가 그대로
# 보존해서, 갱신 표면이 손으로 작성한 설정을 조용히 날리지 않게 한다. 일부 표면은 이 관리 필드 중
# 일부만 노출하므로(예: harness ``update_agent`` 도구는 모델 동작 인자를 받지 않는다),
# config.yaml을 다시 쓸 때 지원하지 않는 관리 필드를 명시적으로 이어 넘겨야 한다.
# ``name`` 이 포함된 이유는 갱신 로직이 항상 디렉터리 이름에서 다시 만들어 내보내기 때문이다
# (요청 본문에서 와서는 안 된다).
MANAGED_AGENT_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "model",
        "tool_groups",
        "skills",
        "model_settings",
        "thinking_enabled",
        "reasoning_effort",
    }
)


def preserve_non_managed_fields(existing_cfg: AgentConfig) -> dict[str, object]:
    """``existing_cfg`` 에서 :data:`MANAGED_AGENT_CONFIG_FIELDS` 에 없는 최상위 필드를 모두 반환한다.

    커스텀 agent의 ``config.yaml`` 을 다시 쓰는 두 표면(``update_agent`` harness 도구와 HTTP
    ``PATCH /api/agents/{name}`` route)이, 갱신 API가 인자로 노출하지 않는 손으로 작성한 필드
    (현재는 ``github``, 앞으로 :class:`AgentConfig` 에 추가될 필드 포함)를 이어 넘기는 데 쓴다.
    이게 없으면 커스텀 agent에 ``github:`` 블록을 손으로 작성한 운영자가, 다음번에 agent나 UI 편집기가
    ``description`` / ``model`` / ``tool_groups`` / ``skills`` 를 건드리는 순간 그 설정을 조용히 잃는다.

    Pydantic v2에서 ``exclude_unset=True`` 는 재귀적으로 동작하므로, 사용자가 쓰지 않아
    Pydantic 기본값이 들어간 하위 필드는 dict에 나타나지 않는다. 즉 파일이 보기에도 그대로 왕복한다.
    """
    return existing_cfg.model_dump(exclude_unset=True, exclude=MANAGED_AGENT_CONFIG_FIELDS)


def resolve_agent_dir(name: str, *, user_id: str | None = None) -> Path:
    """agent의 디스크 디렉터리를 반환한다. 사용자별 레이아웃을 우선한다.

    해석 순서:
    1. ``{base_dir}/users/{user_id}/agents/{name}/`` (사용자별, 현재 레이아웃).
    2. ``{base_dir}/agents/{name}/`` (레거시 공유 레이아웃, 읽기 전용 fallback).

    둘 다 없으면 사용자별 경로를 반환해서, agent를 만들려는 호출자가 새 레이아웃에 쓰게 한다.

    Args:
        name: 검증된 agent 이름.
        user_id: agent 소유자. 기본값은 요청 컨텍스트의 유효 사용자
            (인증이 없는 모드에서는 ``"default"``).
    """
    paths = get_paths()
    effective_user = user_id or get_effective_user_id()
    user_path = paths.user_agent_dir(effective_user, name)
    # config.yaml이 있어야 진짜 agent 디렉터리로 인정한다.
    # memory/storage 쓰기가 남긴 잔재가 아님을 확인하기 위해서다(#3390 참고).
    if user_path.exists() and (user_path / "config.yaml").exists():
        return user_path

    legacy_path = paths.agent_dir(name)
    if legacy_path.exists() and (legacy_path / "config.yaml").exists():
        return legacy_path

    return user_path


def load_agent_config(name: str | None, *, user_id: str | None = None) -> AgentConfig | None:
    """커스텀 agent 또는 기본 agent의 설정을 로드한다.

    설정된 agent store(``agent_storage.backend``)로 위임한다. ``file`` 백엔드는 사용자별 레이아웃을
    먼저 읽고 레거시 공유 레이아웃으로 fallback하며, ``db`` 백엔드는 공유 ``agents`` 테이블을 읽는다.
    동작과 에러 의미는 기존 파일 전용 로더와 동일하다.

    Args:
        name: agent 이름.
        user_id: agent 소유자. 기본값은 현재 요청 컨텍스트의 유효 사용자.

    Returns:
        AgentConfig 인스턴스. ``name`` 이 ``None``이면 ``None``.

    Raises:
        FileNotFoundError: agent가 존재하지 않을 때.
        ValueError: 저장된 설정을 파싱할 수 없을 때.
    """
    if name is None:
        return None
    # 지연 import. store 패키지가 이 모듈을 다시 import하기 때문이다.
    from deerflow.persistence.agents import get_agent_store

    return get_agent_store().get(name, user_id=user_id)


def load_agent_soul(agent_name: str | None, *, user_id: str | None = None) -> str | None:
    """agent의 SOUL.md 내용을 읽는다. 없으면 None을 반환한다.

    SOUL.md는 agent의 성격, 가치관, 행동 가드레일을 정의하며 lead agent의 system prompt에
    추가 컨텍스트로 주입된다. 기본 agent(``agent_name`` 이 falsy)는 커스텀 agent 레코드가 아니므로
    백엔드와 무관하게 항상 ``{base_dir}/SOUL.md`` 를 직접 읽는다. 이름이 있는 agent는
    설정된 store로 위임한다.

    Args:
        agent_name: agent 이름. 기본 agent면 None.
        user_id: agent 소유자. 기본값은 현재 요청 컨텍스트의 유효 사용자.

    Returns:
        SOUL.md 내용 문자열. 설정되지 않았으면 None.
    """
    if not agent_name:
        soul_path = get_paths().base_dir / SOUL_FILENAME
        if not soul_path.exists():
            return None
        content = soul_path.read_text(encoding="utf-8").strip()
        return content or None
    from deerflow.persistence.agents import get_agent_store

    return get_agent_store().get_soul(agent_name, user_id=user_id)


def list_custom_agents(*, user_id: str | None = None) -> list[AgentConfig]:
    """``user_id`` 의 유효한 커스텀 agent를 모두 반환한다.

    설정된 agent store로 위임한다. ``file`` 백엔드는 사용자별 레이아웃과 레거시 공유 레이아웃의
    합집합을 반환하며, 이름이 같으면 사용자별 항목이 레거시 항목을 가린다. ``db`` 백엔드는 해당
    사용자의 row를 반환한다. 결과는 이름순으로 정렬된다.

    Args:
        user_id: 나열할 agent의 소유자. 기본값은 현재 요청 컨텍스트의 유효 사용자.

    Returns:
        발견된 유효한 agent 각각의 AgentConfig 리스트.
    """
    from deerflow.persistence.agents import get_agent_store

    return get_agent_store().list(user_id=user_id)
