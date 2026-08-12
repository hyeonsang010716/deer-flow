"""run context의 request 범위 secret carrier (issue #3861).

caller는 요청별 secret을 ``config.context.secrets``(name -> value 매핑)에 out-of-band로
전달한다. 그 값은 prompt, tool 인자, 실행되는 command 문자열에 절대 들어가지 않으며,
활성화된 skill이 ``required-secrets`` frontmatter 필드로 선언한 경우에만 그 skill의 sandbox
subprocess에 환경 변수로 주입된다.

이 모듈은 예약 키 이름과 안전한 추출을 한곳에 모아 carrier 계약을 한 군데서 관리한다.
skill-activation middleware(턴별 주입 집합 구성)와 tracing redactor(trace payload에서 제거)가
이를 사용한다.
"""

from __future__ import annotations

from typing import Any

# caller가 제공한 request 범위 secret을 담는 run context의 예약 하위 키.
# skill이 *받을 수 있는* 것에 대한 단일 진실 원천이다.
SECRETS_CONTEXT_KEY = "secrets"

# 현재 활성화된 skill에 대해 해석된 secret을 담는 예약 하위 키(binding point A).
# skill-activation middleware가 쓰고 bash tool이 읽는다. 두 예약 키 모두 trace payload에서
# 제거된다(tracing redactor 참고).
ACTIVE_SECRETS_CONTEXT_KEY = "__active_skill_secrets"

# 한 model step 동안의 활성 skill tool-policy 결정을 담는 예약 하위 키. 이 결정에는
# middleware 인스턴스 owner token이 포함되어, caller가 병합 가능한 run context에 allow-all
# 결정을 위조하는 것을 막는다. 따라서 값 전체를 관찰 가능한 모든 직렬화 지점에서 제거해야 한다.
SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY = "__skill_tool_policy_decision"

LEGACY_AUTH_TOKEN_METADATA_KEY = "auth_token"


class LegacyRunMetadataSecretError(ValueError):
    """run이 request credential을 영속 metadata에 넣을 때 raise된다."""


def validate_run_metadata_secrets(metadata: Any) -> None:
    """run 승인 시점에 legacy credential 필드를 거부한다."""
    if isinstance(metadata, dict) and LEGACY_AUTH_TOKEN_METADATA_KEY in metadata:
        raise LegacyRunMetadataSecretError("Run metadata key 'auth_token' is not allowed; pass request-scoped credentials via config.context.secrets instead.")


def redact_metadata_secrets(metadata: Any) -> Any:
    """기존 저장 객체를 변경하지 않고 API에 안전한 metadata를 반환한다."""
    if not isinstance(metadata, dict):
        return metadata
    return {key: value for key, value in metadata.items() if key != LEGACY_AUTH_TOKEN_METADATA_KEY}


def _string_pairs(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, str)}


def extract_request_secrets(context: Any) -> dict[str, str]:
    """caller가 제공한 request 범위 secret 매핑을 반환한다. 없으면 ``{}``.

    키와 값이 모두 문자열인 항목만 남기고 나머지는 무시하므로, 잘못된 형태의 carrier가
    secret 해석이나 주입을 망가뜨릴 수 없다.
    """
    if not isinstance(context, dict):
        return {}
    return _string_pairs(context.get(SECRETS_CONTEXT_KEY))


def read_active_secrets(context: Any) -> dict[str, str]:
    """활성 skill에 대해 해석된 secret(run별 주입 집합)을 반환한다. 없으면 ``{}``.
    bash tool이 subprocess env를 구성할 때 읽는다."""
    if not isinstance(context, dict):
        return {}
    return _string_pairs(context.get(ACTIVE_SECRETS_CONTEXT_KEY))


def write_slash_skill_source_path(context: Any, path: str, *, owner_token: str) -> None:
    """인증된 slash 활성화 skill 경로를 run context에 저장한다.

    이 source는 경로 참조와 middleware chain 내부용 token을 담는다. 소비자는 skill metadata를
    신뢰하기 전에 token을 인증하고 경로를 live skill registry에 대해 해석해야 한다.
    """
    if isinstance(context, dict) and isinstance(path, str) and path and isinstance(owner_token, str) and owner_token:
        context[_SLASH_SECRET_SOURCE_KEY] = {"path": path, "owner_token": owner_token}


def read_slash_skill_source_path(context: Any, *, owner_token: str) -> str | None:
    """형식이 올바르면 인증된 slash 활성화 skill 경로를 반환한다."""
    if not isinstance(context, dict):
        return None
    source = context.get(_SLASH_SECRET_SOURCE_KEY)
    if not isinstance(source, dict):
        return None
    path = source.get("path")
    source_owner_token = source.get("owner_token")
    if not isinstance(owner_token, str) or not owner_token or source_owner_token != owner_token:
        return None
    return path if isinstance(path, str) and path else None


# skill-activation middleware가 run 전체에 걸쳐 secret 바인딩을 나르기 위해 쓰는 비공개
# run-context 키들. secret 값을 담는 것은 ``secrets`` / ``__active_skill_secrets``뿐이다.
# slash source는 middleware chain owner token을, audit 키는 이름만 담는다. redaction
# allowlist가 완전한 방어선이 되도록 전부 나열한다.
_SLASH_SECRET_SOURCE_KEY = "__slash_skill_secret_source"
_SECRETS_BINDING_AUDIT_KEY = "__skill_secrets_binding_audit"

# 이 run에서 이미 발동한 가장 최근 slash 활성화의 identity. 덕분에 reminder 주입, skill
# 디스크 읽기, ``activate`` audit 이벤트가 tool 루프의 모든 model 호출이 아니라 사용자
# slash 명령 하나당 한 번만 일어난다.
# reminder는 호출별 model request에만 주입되고 graph state로 되돌아 쓰이지 않으므로,
# 2번째~N번째 model 호출에서는 ``request.messages``를 훑어도 이전 활성화를 감지할 수 없다.
# run context만이 살아남는 유일한 신호다(``_SLASH_SECRET_SOURCE_KEY``와 같은 방식).
# message id / content digest만 담고 secret 값은 담지 않는다. redaction 방어선을 완전하게
# 유지하려고 아래에 함께 나열한다.
_SLASH_SKILL_ACTIVATION_RUN_KEY = "__slash_skill_activation_run"

# 값이 request 범위 secret이라서, context 매핑이 관찰 가능한 곳(trace, log)으로 직렬화되기
# 전에 반드시 제거해야 하는 run-context 키들.
REDACTED_CONTEXT_KEYS = frozenset(
    {
        SECRETS_CONTEXT_KEY,
        ACTIVE_SECRETS_CONTEXT_KEY,
        _SLASH_SECRET_SOURCE_KEY,
        _SECRETS_BINDING_AUDIT_KEY,
        _SLASH_SKILL_ACTIVATION_RUN_KEY,
        SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY,
    }
)


def redact_secret_context_keys(context: Any) -> Any:
    """secret을 담은 키를 제거한 ``context``의 shallow copy를 반환한다.

    run context를 관찰 가능한 곳으로 직렬화하는 모든 코드 경로를 위한 방어용 헬퍼다.
    DeerFlow 자체 trace-metadata builder는 context를 복사하지 않으므로, 이것은 앞으로 생길
    호출 지점과 커스텀 tracer 설정을 위한 이중 안전장치다.
    """
    if not isinstance(context, dict):
        return context
    return {key: value for key, value in context.items() if key not in REDACTED_CONTEXT_KEYS}


def redact_config_secrets(config: Any) -> Any:
    """저장하거나 client에 되돌려줘도 안전한 run config 사본을 반환한다.

    그대로 두면 request config(``body.config``)가 run record(``runs.kwargs_json``)에 원문
    그대로 저장되고 run API가 그것을 돌려준다. ``context``에서 secret을 담은 키를,
    ``metadata``에서 legacy credential을 제거해 보호 대상 config 표면이 저장되거나 반환되지
    않게 한다. 실제 run을 구동하는 live config(별도로 만들어진다)는 그대로 유지한다.
    일반 metadata는 보존되며, dict가 아닌 config는 그대로 통과한다.
    """
    if not isinstance(config, dict):
        return config

    redacted = dict(config)
    context = config.get("context")
    if isinstance(context, dict):
        redacted["context"] = redact_secret_context_keys(context)

    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        redacted["metadata"] = redact_metadata_secrets(metadata)

    return redacted
