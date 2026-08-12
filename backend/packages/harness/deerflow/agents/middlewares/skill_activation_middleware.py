"""skill 활성화 미들웨어 — 명시적 slash 활성화와 in-context secret 바인딩을 담당한다."""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import posixpath
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.runtime.events.catalog import (
    MIDDLEWARE_SKILL_ACTIVATION_TAG,
    MIDDLEWARE_SKILL_SECRETS_TAG,
)
from deerflow.runtime.secret_context import (
    _SECRETS_BINDING_AUDIT_KEY,
    _SLASH_SKILL_ACTIVATION_RUN_KEY,
    ACTIVE_SECRETS_CONTEXT_KEY,
    extract_request_secrets,
    read_slash_skill_source_path,
    write_slash_skill_source_path,
)
from deerflow.skills.slash import parse_slash_skill_reference, resolve_slash_skill
from deerflow.skills.storage import get_or_new_skill_storage, get_or_new_user_skill_storage
from deerflow.skills.storage.skill_storage import SkillStorage
from deerflow.skills.types import SKILL_MD_FILE, SecretRequirement, Skill, SkillCategory
from deerflow.utils.messages import get_original_user_content_text, is_real_user_message

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

_SLASH_SKILL_ACTIVATION_KEY = "slash_skill_activation"
_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY = "slash_skill_activation_target_id"

# _SECRETS_BINDING_AUDIT_KEY: 마지막으로 감사한 바인딩(skill명과 secret명만, 값은 절대 저장하지 않는다).
# 바뀌지 않은 바인딩을 매 호출마다 다시 기록하지 않기 위한 것이다.
# 공유 slash-source context 계약은 최신 slash 활성화 정보를 담되, 활성화된 skill의 canonical container
# path만 저장한다(선언된 secret은 절대 저장하지 않는다 — 매 호출마다 live registry에서 읽는다, #3938).
# 주입 집합은 매 model 호출마다 다시 계산되지만, slash로 활성화된 skill은 run이 끝날 때까지 바인딩을
# 유지해야 한다. 단 한 번의 활성화 호출 뒤에도 model의 tool loop가 여러 번 model을 호출하기 때문이다
# (#3861 의미론).
# _SLASH_SKILL_ACTIVATION_RUN_KEY: 이 run에서 이미 활성화한 slash 메시지의 식별자. 덕분에 reminder 주입,
# skill 디스크 읽기, "activate" 감사 이벤트가 매 model 호출이 아니라 사용자 slash 커맨드당 한 번만 발생한다.
# reminder는 단일 model 호출용으로 request.override(messages=...)를 통해 추가될 뿐 graph state에는
# 저장되지 않는다. 따라서 한 턴의 2번째 이후 model 호출은 reminder 없이 state에서 request.messages를
# 재구성하며, run context만이 tool loop를 넘어 살아남는 유일한 신호다.
# 세 키 모두 secret_context에 두어 REDACTED_CONTEXT_KEYS가 한곳에서 처리하게 한다.


@dataclass(frozen=True, slots=True)
class _Activation:
    skill_name: str
    category: str
    container_file_path: str
    skill_content: str
    content_hash: str
    remaining_text: str
    editable: bool
    required_secrets: tuple[SecretRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class _ActivationResolution:
    activation: _Activation | None = None
    failure_message: str | None = None


def is_slash_skill_activation_reminder(message: object) -> bool:
    """메시지가 숨겨진 slash-skill 활성화 context인지 반환한다."""
    return isinstance(message, HumanMessage) and bool(message.additional_kwargs.get(_SLASH_SKILL_ACTIVATION_KEY))


def _is_user_activation_target(message: object) -> bool:
    return is_real_user_message(message)


class SkillActivationMiddleware(AgentMiddleware):
    """사용자가 /skill-name을 명시적으로 입력하면 SKILL.md 전체 내용을 주입한다."""

    def __init__(
        self,
        *,
        available_skills: set[str] | None = None,
        app_config: AppConfig | None = None,
        user_id: str | None = None,
        slash_source_owner_token: str,
    ) -> None:
        super().__init__()
        if not isinstance(slash_source_owner_token, str) or not slash_source_owner_token:
            raise ValueError("slash_source_owner_token must be a non-empty string")
        self._available_skills = set(available_skills) if available_skills is not None else None
        self._app_config = app_config
        self._user_id = user_id
        self._slash_source_owner_token = slash_source_owner_token

    def _storage(self) -> SkillStorage:
        if self._user_id is not None:
            return get_or_new_user_skill_storage(self._user_id, app_config=self._app_config)
        if self._app_config is not None:
            return get_or_new_skill_storage(app_config=self._app_config)
        return get_or_new_skill_storage()

    @staticmethod
    def _read_skill_content(skill_file: Path, skills_root: Path, *, storage: SkillStorage | None = None) -> str:
        if skill_file.name != SKILL_MD_FILE:
            raise ValueError(f"Expected {SKILL_MD_FILE}, got {skill_file.name}")
        # 가능하면 storage의 경로 검증을 쓴다. UserScopedSkillStorage는 custom skill을 전역 skills root의
        # 하위 경로가 아닌 사용자별 디렉터리에 저장하므로, 단순 relative_to 검사로는 거부되기 때문이다.
        # storage가 validate_skill_file_path를 구현하지 않은 mock(예: 테스트)이면 relative_to 검사로 돌아간다.
        if storage is not None and hasattr(storage, "validate_skill_file_path"):
            resolved_file = storage.validate_skill_file_path(skill_file)
        else:
            resolved_file = skill_file.resolve()
            resolved_root = skills_root.resolve()
            try:
                resolved_file.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError("Resolved skill file must stay within the configured skills root.") from exc
        if not resolved_file.is_file():
            raise FileNotFoundError(resolved_file)
        return resolved_file.read_text(encoding="utf-8")

    def _resolve_activation(self, text: str) -> _ActivationResolution | None:
        reference = parse_slash_skill_reference(text)
        if reference is None:
            return None

        storage = self._storage()
        skills = storage.load_skills(enabled_only=False)
        skill = next((candidate for candidate in skills if candidate.name == reference.name), None)
        if skill is None:
            return _ActivationResolution(failure_message=f"Skill `/{reference.name}` is not installed.")
        if not skill.enabled:
            return _ActivationResolution(failure_message=f"Skill `/{reference.name}` is installed but disabled. Enable it before using slash activation.")
        if self._available_skills is not None and reference.name not in self._available_skills:
            return _ActivationResolution(failure_message=f"Skill `/{reference.name}` is not available for this agent.")

        resolved = resolve_slash_skill(
            text,
            skills,
            available_skills=self._available_skills,
            container_base_path=storage.get_container_root(),
        )
        if resolved is None:
            return _ActivationResolution(failure_message=f"Skill `/{reference.name}` could not be resolved.")

        try:
            skill_content = self._read_skill_content(resolved.skill.skill_file, storage.get_skills_root_path(), storage=storage)
        except (OSError, ValueError):
            logger.exception("Failed to read slash-activated skill %s", resolved.skill.name)
            return _ActivationResolution(failure_message=f"Skill `/{reference.name}` could not be loaded safely. Please check the skill installation.")

        content_hash = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()
        # CUSTOM skill은 편집 가능하고 PUBLIC과 LEGACY는 읽기 전용이다.
        editable = resolved.skill.category == SkillCategory.CUSTOM
        return _ActivationResolution(
            activation=_Activation(
                skill_name=resolved.skill.name,
                category=str(resolved.skill.category),
                container_file_path=resolved.container_file_path,
                skill_content=skill_content,
                content_hash=content_hash,
                remaining_text=resolved.remaining_text,
                editable=editable,
                required_secrets=tuple(resolved.skill.required_secrets or ()),
            )
        )

    @staticmethod
    def _build_activation_reminder(activation: _Activation) -> str:
        user_request = activation.remaining_text or ("No additional task text was provided after the slash skill command. Ask the user what they want to do with this skill if the next step is unclear.")
        escaped_user_request = html.escape(user_request, quote=False)
        escaped_skill_content = html.escape(activation.skill_content, quote=False)
        escaped_skill_name = html.escape(activation.skill_name, quote=True)
        escaped_category = html.escape(activation.category, quote=True)
        escaped_path = html.escape(activation.container_file_path, quote=True)
        escaped_content_hash = html.escape(activation.content_hash, quote=True)
        editable_str = "true" if activation.editable else "false"
        return f"""<slash_skill_activation>
The user explicitly activated the `{escaped_skill_name}` skill for this turn.
Treat the task text as:
<user_request>
{escaped_user_request}
</user_request>

Follow this skill before choosing a general workflow. Load supporting resources from the same skill directory only when needed.

<skill name="{escaped_skill_name}" category="{escaped_category}" path="{escaped_path}" sha256="{escaped_content_hash}" editable="{editable_str}">
<skill_content encoding="xml-escaped">
{escaped_skill_content}
</skill_content>
</skill>
</slash_skill_activation>"""

    @staticmethod
    def _has_existing_activation_for_target(messages: list, target_index: int, target: HumanMessage) -> bool:
        if target_index <= 0:
            return False

        if target.id:
            for previous in messages[:target_index]:
                if not is_slash_skill_activation_reminder(previous):
                    continue
                target_id = previous.additional_kwargs.get(_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY)
                if target_id == target.id or previous.id == f"{target.id}__slash_activation":
                    return True

        previous = messages[target_index - 1]
        return is_slash_skill_activation_reminder(previous)

    @staticmethod
    def _activation_run_key(target: HumanMessage) -> str:
        """run당 한 번만 활성화하기 위한 사용자 slash 메시지의 안정적 식별자를 만든다.

        메시지 id를 우선 쓴다(LangGraph는 메시지가 graph state에 들어가면 안정적인 id를 부여하고 유지한다).
        id가 없으면 실제 사용자 텍스트의 digest로 대체해 run 안에서 중복을 제거한다.
        새 slash 메시지(새 id 또는 새 텍스트)는 새 키를 만들므로 억제되지 않는다.
        """
        if target.id:
            return target.id
        content = get_original_user_content_text(target.content, target.additional_kwargs)
        return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _run_context(request: ModelRequest) -> dict | None:
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        return context if isinstance(context, dict) else None

    @staticmethod
    def _already_activated(run_context: dict | None, run_key: str) -> bool:
        """이 run에서 ``run_key``가 이미 활성화된 것으로 기록되었는지 반환한다.

        ``_has_existing_activation_for_target``의 형제 함수다. 그쪽은 스캔한 ``messages`` 구간에 아직 남아
        있는 활성화 reminder를 잡고, 이쪽은 reminder가 이미 그 구간 밖으로 밀려난 상태에서
        ``run_context``에 기록된 이전 활성화를 잡는다(tool loop 상황 — ``_SLASH_SKILL_ACTIVATION_RUN_KEY``
        참고). ``run_key``는 호출자(``_find_activation_target``)가 한 번 계산해 ``_prepare_model_request``의
        기록 지점까지 그대로 전달하므로, 검사와 기록에 항상 같은 키가 쓰인다. 이 헬퍼는 키를 계산하지 않고
        포함 여부만 확인한다.
        """
        return isinstance(run_context, dict) and run_context.get(_SLASH_SKILL_ACTIVATION_RUN_KEY) == run_key

    def _find_activation_target(self, messages: list, *, run_context: dict | None = None) -> tuple[int, HumanMessage, _ActivationResolution, str] | None:
        if not messages:
            return None

        target_index = next((idx for idx in range(len(messages) - 1, -1, -1) if _is_user_activation_target(messages[idx])), None)
        if target_index is None:
            return None

        target = messages[target_index]
        if target is None:
            return None
        if self._has_existing_activation_for_target(messages, target_index, target):
            return None
        # 이 slash 메시지가 run 앞부분에서 이미 활성화됐을 수 있다. reminder는 호출별 request override에만
        # 존재하고 state에는 없으므로 위의 메시지 스캔으로는 잡지 못하며, run context가 유일한 durable 신호다
        # (_already_activated / _SLASH_SKILL_ACTIVATION_RUN_KEY 참고).
        # 여기서 건너뛰면 불필요한 skill 디스크 읽기, reminder 재주입, "activate" 감사 중복을 피할 수 있다.
        # run_key는 여기서 한 번 계산해 _prepare_model_request의 기록 지점까지 전달한다.
        run_key = self._activation_run_key(target)
        if self._already_activated(run_context, run_key):
            return None

        content = get_original_user_content_text(target.content, target.additional_kwargs)
        resolution = self._resolve_activation(content)
        if resolution is None:
            return None
        return target_index, target, resolution, run_key

    @staticmethod
    def _record_activation(request: ModelRequest, activation: _Activation, *, hook: str) -> None:
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        journal = context.get("__run_journal") if isinstance(context, dict) else None
        if journal is None:
            return
        try:
            journal.record_middleware(
                MIDDLEWARE_SKILL_ACTIVATION_TAG,
                name="SkillActivationMiddleware",
                hook=hook,
                action="activate",
                changes={
                    "skill_name": activation.skill_name,
                    "category": activation.category,
                    "path": activation.container_file_path,
                    "content_hash": activation.content_hash,
                },
            )
        except Exception:
            logger.warning("Failed to record slash skill activation audit event", exc_info=True)

    def _prepare_model_request(self, request: ModelRequest, *, hook: str) -> tuple[ModelRequest | AIMessage | None, _Activation | None]:
        run_context = self._run_context(request)
        target_and_resolution = self._find_activation_target(list(request.messages), run_context=run_context)
        if target_and_resolution is None:
            return None, None

        target_index, target, resolution, run_key = target_and_resolution
        if resolution.failure_message:
            return AIMessage(content=resolution.failure_message), None

        activation = resolution.activation
        if activation is None:
            return None, None

        logger.info(
            "SkillActivationMiddleware: activating slash skill %s category=%s path=%s hash=%s",
            activation.skill_name,
            activation.category,
            activation.container_file_path,
            activation.content_hash,
        )
        self._record_activation(request, activation, hook=hook)
        # 이 slash 메시지를 run 기준으로 활성화됨으로 표시해, tool loop의 이후 model 호출이 중복 재활성화를
        # 건너뛰게 한다(#3861: 활성화 호출 한 번, 후속 model 호출 다수). 새 slash 메시지는 키가 달라 여전히
        # 활성화된다. 누적이 아니라 덮어쓰기(`=`)인 것은 의도적이다. _find_activation_target은 항상 가장
        # 최근의 실제 사용자 메시지만 활성화 대상으로 보므로, 새 활성화가 덮어쓴 뒤에는 run 앞부분에
        # 기억할 것이 없다. 이를 set으로 "고치지" 말 것. run_key는 _find_activation_target에서 이미 검사한
        # 값을 그대로 넘겨받은 것이며 다시 계산하지 않는다.
        if run_context is not None:
            run_context[_SLASH_SKILL_ACTIVATION_RUN_KEY] = run_key
        activation_msg = self._make_activation_message(target, self._build_activation_reminder(activation))
        messages = list(request.messages)
        messages.insert(target_index, activation_msg)
        return request.override(messages=messages), activation

    def _handle_model_request(self, request: ModelRequest, *, hook: str) -> ModelRequest | AIMessage:
        prepared, activation = self._prepare_model_request(request, hook=hook)
        if isinstance(prepared, AIMessage):
            return prepared
        effective = prepared if prepared is not None else request
        self._resolve_secret_bindings(effective, activation, hook=hook)
        return effective

    def _resolve_secret_bindings(self, request: ModelRequest, activation: _Activation | None, *, hook: str) -> None:
        """run 단위 secret 주입 집합을 다시 계산한다(binding point A+, #3861/#3914).

        매 model 호출마다 다음 두 source를 합집합으로 사용한다.

        - 이 run의 가장 최근 slash 활성화. 활성화 호출 이후의 tool loop 전체가 바인딩을 유지하도록
          run context에 source로 저장하며, 새 slash 활성화가 이를 대체한다. slash source는 활성화 시점에
          한 번만 검증하고(``_resolve_activation``의 enabled + allowlist 검사) 호출마다 재검증하지
          않는다. slash는 사용자가 run 범위로 내린 명시적 결정이고 run과 함께 사라지기 때문이다.
        - 모델이 이 thread에서 앞서 로드한 skill(``ThreadState.skill_context``). 매 호출마다 live registry로
          재검증한다: enabled 여부, 이 agent에 대한 runtime 허용 여부, ``secrets-autonomous: false``로
          opt-out하지 않았는지. slash 활성화는 이 opt-out에서 면제된다. 명시적 절차를 거친 경로이기 때문이다.

        집합은 매 호출마다 다시 계산되어 통째로 교체된다. 따라서 skill_context에서 밀려난 skill이나
        값 공급을 중단한 호출자는 다음 호출에서 자동으로 주입을 잃는다. 주입되는 값은 항상 호출자의
        요청(``context.secrets``)에서 오며 호스트 환경에서 오지 않는다. 호스트 환경은
        ``env_policy.build_sandbox_env``가 주입 전에 제거하므로 skill이 호스트 플랫폼 자격 증명을 가져갈 수 없다.
        secret *값*은 절대 로그에 남기지 않으며, 감사 journal에는 이름만 기록한다.
        """
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        if not isinstance(context, dict):
            return

        # slash source에는 canonical container path와 middleware chain 내부용 owner token만 기록하고
        # 선언된 secret은 절대 기록하지 않는다. 두 소비자 모두 source를 인증한 뒤 path로 live registry의
        # skill을 조회하므로, 호출자가 병합할 수 있는 context로는 활성화를 위조할 수 없다.
        if activation is not None:
            write_slash_skill_source_path(
                context,
                activation.container_file_path,
                owner_token=self._slash_source_owner_token,
            )

        request_secrets = extract_request_secrets(context)
        sources: list[tuple[str, tuple[SecretRequirement, ...]]] = []
        if request_secrets:
            registry = self._load_skill_registry_by_path()
            if registry is not None:
                # slash source는 명시적 절차이므로 ``secrets-autonomous`` opt-out에서 면제되지만,
                # enabled와 allowlist 검사는 그대로 받는다.
                slash_path = read_slash_skill_source_path(context, owner_token=self._slash_source_owner_token)
                slash_skill = self._resolve_registry_skill(registry, slash_path, require_autonomous=False)
                if slash_skill is not None:
                    sources.append((slash_skill.name, tuple(slash_skill.required_secrets)))
                sources.extend(self._in_context_secret_sources(request, registry))

        injected: dict[str, str] = {}
        bound_skills: set[str] = set()
        missing: dict[str, list[str]] = {}
        for skill_name, requirements in sources:
            for req in requirements:
                if req.name in request_secrets:
                    injected[req.name] = request_secrets[req.name]
                    bound_skills.add(skill_name)
                elif not req.optional:
                    missing.setdefault(skill_name, []).append(req.name)

        if injected:
            context[ACTIVE_SECRETS_CONTEXT_KEY] = injected
        else:
            context.pop(ACTIVE_SECRETS_CONTEXT_KEY, None)

        audit_state = {
            "skills": sorted(bound_skills),
            "secrets": sorted(injected),
            "missing": {name: sorted(values) for name, values in sorted(missing.items())},
        }
        previous = context.get(_SECRETS_BINDING_AUDIT_KEY)
        if previous == audit_state:
            return
        if previous is None and not injected and not missing:
            return
        context[_SECRETS_BINDING_AUDIT_KEY] = audit_state
        for skill_name, names in sorted(missing.items()):
            logger.warning(
                "Skill %s is active but required secrets are missing from the request context: %s",
                skill_name,
                ", ".join(names),
            )
        self._record_secret_binding(context, audit_state, hook=hook)

    def _load_skill_registry_by_path(self) -> dict[str, Skill] | None:
        """정규화된 container file path를 키로 하는 live skill registry를 로드한다.

        캐시하지 않고 매 호출마다 다시 읽는 것은 의도적이다. load_skills가 extensions_config에서 enabled
        상태를 다시 읽으므로, 운영자가 skill을 비활성화하면 바로 다음 model 호출에서 secret 바인딩이
        해제된다. 파일 mtime 기반 캐시는 SKILL.md를 건드리지 않는 enable/disable 토글을 놓쳐 비활성화 후에도
        주입을 계속하게 되며, 즉시 해제라는 보안 속성을 속도와 맞바꾸는 셈이다. 비용은 제한적이다.
        유일한 호출자가 호출자로부터 secret이 제공된 경우에만 이 함수를 실행한다.

        path를 정규화하므로 canonical하지 않은 ``container_path`` 설정(예: 끝에 슬래시)도
        ``skill_context``에 담긴 canonical path와 매칭된다(#3938). registry를 로드하지 못하면 ``None``을
        반환하고, 그 호출에서는 slash와 in-context source 모두 아무것도 바인딩하지 않는다(fail closed).
        이는 의도적인 가용성-보안 트레이드오프다. run 도중 일시적 registry 읽기 실패는 오래된 호출자 제공
        데이터를 신뢰하는 대신 그 호출의 주입을 포기한다.
        """
        try:
            storage = self._storage()
            skills = storage.load_skills(enabled_only=False)
            container_root = storage.get_container_root()
        except Exception:
            logger.exception("Failed to load skills while resolving secret bindings")
            return None
        return {posixpath.normpath(skill.get_container_file_path(container_root)): skill for skill in skills}

    def _resolve_registry_skill(self, registry: dict[str, Skill], path: object, *, require_autonomous: bool) -> Skill | None:
        """container path를 secret 바인딩 자격이 있는 live registry skill로 해석하거나 ``None``을 반환한다.

        정규화된 container file path로만 매칭하고 이름으로는 절대 매칭하지 않는다. 이름 fallback은
        confused deputy가 된다. DeerFlow는 custom skill이 같은 이름의 public/legacy skill을 가리도록
        허용하므로(load_skills가 이름으로 중복 제거하며 custom이 이긴다), public/foo 참조가 custom foo의
        secret을 바인딩할 수 있다. 해석되지 않는 path는 아무것도 바인딩하지 않으며(안전한 방향),
        호출자가 위조한 path에도 fail closed로 동작한다(#3938).

        조건: skill이 enabled여야 하고, secret을 선언해야 하며, 이 agent의 allowlist에 있어야 한다.
        ``require_autonomous``는 in-context 경로에 ``secrets-autonomous`` opt-out을 추가로 강제한다.
        slash 경로는 ``False``를 넘기는데, opt-out이 지키려는 것이 바로 명시적 활성화라는 절차이기 때문이다.
        """
        if not isinstance(path, str) or not path:
            return None
        skill = registry.get(posixpath.normpath(path))
        if skill is None or not skill.enabled or not skill.required_secrets:
            return None
        if require_autonomous and not skill.secrets_autonomous:
            return None
        if self._available_skills is not None and skill.name not in self._available_skills:
            return None
        return skill

    def _in_context_secret_sources(self, request: ModelRequest, registry: dict[str, Skill]) -> list[tuple[str, tuple[SecretRequirement, ...]]]:
        """``ThreadState.skill_context`` 항목을 선언된 secret source로 매핑한다.

        각 항목은 모델이 이 thread에서 실제로 로드한 skill에 대한 참조다. 매번 live registry로 재검증하므로,
        읽힌 뒤 비활성화·삭제·opt-out되었거나 agent allowlist에서 빠진 skill은 즉시 바인딩을 멈춘다.
        """
        state = getattr(request, "state", None) or {}
        try:
            entries = state.get("skill_context") or []
        except AttributeError:
            return []

        sources: list[tuple[str, tuple[SecretRequirement, ...]]] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            skill = self._resolve_registry_skill(registry, entry.get("path"), require_autonomous=True)
            if skill is None or skill.name in seen:
                continue
            seen.add(skill.name)
            sources.append((skill.name, tuple(skill.required_secrets)))
        return sources

    @staticmethod
    def _record_secret_binding(context: dict, audit_state: dict, *, hook: str) -> None:
        journal = context.get("__run_journal")
        if journal is None:
            return
        try:
            journal.record_middleware(
                MIDDLEWARE_SKILL_SECRETS_TAG,
                name="SkillActivationMiddleware",
                hook=hook,
                action="bind_secrets",
                changes=audit_state,
            )
        except Exception:
            logger.warning("Failed to record skill secret binding audit event", exc_info=True)

    @staticmethod
    def _make_activation_message(target: HumanMessage, activation_content: str) -> HumanMessage:
        stable_id = target.id or str(uuid.uuid4())
        additional_kwargs = {
            "hide_from_ui": True,
            _SLASH_SKILL_ACTIVATION_KEY: True,
        }
        if target.id:
            additional_kwargs[_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY] = target.id
        return HumanMessage(
            content=activation_content,
            id=f"{stable_id}__slash_activation",
            additional_kwargs=additional_kwargs,
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        prepared = self._handle_model_request(request, hook="wrap_model_call")
        if isinstance(prepared, AIMessage):
            return prepared
        return handler(prepared)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        prepared = await asyncio.to_thread(self._handle_model_request, request, hook="awrap_model_call")
        if isinstance(prepared, AIMessage):
            return prepared
        return await handler(prepared)
