import logging
import re
from pathlib import Path

import yaml

from .types import SKILL_MD_FILE, SecretRequirement, Skill, SkillCategory

logger = logging.getLogger(__name__)

# 유효한 POSIX 환경 변수 이름.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _format_yaml_error(skill_file: Path, exc: yaml.YAMLError, source: str) -> str:
    """YAML front-matter 오류를 개발자가 이해하기 쉬운 형태로 렌더링한다."""

    lines = [f"Invalid YAML front-matter in {skill_file}: {exc}"]

    mark = getattr(exc, "problem_mark", None)
    source_lines = source.splitlines()
    if mark is not None and 0 <= mark.line < len(source_lines):
        offending = source_lines[mark.line]

        # mark.line은 front-matter 본문 기준 0-based다. +1로 1-based가 되고, front-matter
        # regex가 yaml.safe_load 전에 제거하는 선행 `---` fence를 위해 +1을 더한다.
        file_line_number = mark.line + 2
        lines.append(f"  line {file_line_number}: {offending}")

        if getattr(exc, "problem", "") == "mapping values are not allowed here" and ":" in offending:
            key, _, value = offending.partition(":")
            value = value.strip()
            if value and value[0] not in {'"', "'", "|", ">", "[", "{"}:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'  hint: values containing ":" must be quoted, e.g. {key}: "{escaped}"')

    return "\n".join(lines)


def parse_allowed_tools(raw: object, skill_file: Path) -> tuple[str, ...] | None:
    """선택 항목인 allowed-tools frontmatter 필드를 파싱한다.

    필드가 없으면 None을 반환한다. 필드가 문자열 YAML sequence면 tuple을 반환하며, tool을
    명시적으로 하나도 쓰지 않는 skill은 빈 tuple이 된다. 값이 잘못되면 ValueError를 던진다.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"allowed-tools in {skill_file} must be a list of strings")

    allowed_tools: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"allowed-tools in {skill_file} must contain only strings")
        tool_name = item.strip()
        if not tool_name:
            raise ValueError(f"allowed-tools in {skill_file} cannot contain empty tool names")
        allowed_tools.append(tool_name)
    return tuple(allowed_tools)


def parse_required_secrets(raw: object, skill_file: Path) -> tuple[SecretRequirement, ...]:
    """선택 항목인 required-secrets frontmatter 필드를 파싱한다(issue #3861).

    항목이 문자열(secret / 환경 변수 이름)이거나 mapping(``{name, optional}``)인 YAML sequence를
    받는다. 필드가 없으면 빈 tuple을 반환한다. 이름이 없거나 유효한 환경 변수 이름이 아닌 항목은
    경고와 함께 버려서, 잘못된 선언 하나가 skill 전체를 무효화하지 않게 한다. 필드가 있지만
    list가 아닐 때만 ValueError를 던진다.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"required-secrets in {skill_file} must be a list")

    secrets: list[SecretRequirement] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            name, optional = item.strip(), False
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            optional = bool(item.get("optional", False))
        else:
            logger.warning("Ignoring malformed required-secrets entry in %s: %r", skill_file, item)
            continue

        if not _ENV_VAR_NAME_RE.match(name):
            logger.warning("Ignoring required-secrets entry with invalid env var name in %s: %r", skill_file, name)
            continue
        if name in seen:
            continue
        seen.add(name)
        secrets.append(SecretRequirement(name=name, optional=optional))
    return tuple(secrets)


def parse_secrets_autonomous(raw: object, skill_file: Path) -> bool:
    """선택 항목인 ``secrets-autonomous`` frontmatter 필드를 파싱한다(issue #3914).

    ``True``(기본값)면 모델이 자율적으로 로드해 skill이 context에 들어와 있는 동안에도 선언된
    secret이 바인딩된다. ``False``면 명시적인 ``/slash`` activation으로만 바인딩된다. 형식이
    잘못된(boolean이 아닌) 값은 더 안전하고 injection 위험이 낮은 ``False``로 fail-closed된다.
    """
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    logger.warning("Ignoring malformed secrets-autonomous value in %s: %r (autonomous binding disabled)", skill_file, raw)
    return False


def parse_skill_file(skill_file: Path, category: SkillCategory, relative_path: Path | None = None) -> Skill | None:
    """SKILL.md 파일을 파싱해 metadata를 추출한다.

    Args:
        skill_file: SKILL.md 파일 경로.
        category: skill의 category.
        relative_path: category root에서 skill 디렉터리까지의 상대 경로. 생략하면 skill
            디렉터리 이름을 쓴다.

    Returns:
        파싱에 성공하면 Skill 객체, 아니면 None.
    """
    if not skill_file.exists() or skill_file.name != SKILL_MD_FILE:
        return None

    try:
        content = skill_file.read_text(encoding="utf-8")

        # parser 진단은 host 경로가 없는 순수 helper의 오류 문자열보다 풍부하게 유지한다.
        # 테스트와 저작 UX가 줄 단위 힌트에 의존한다.
        front_matter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
        if not front_matter_match:
            return None
        front_matter_text = front_matter_match.group(1)
        try:
            metadata = yaml.safe_load(front_matter_text)
        except yaml.YAMLError as exc:
            logger.error("%s", _format_yaml_error(skill_file, exc, front_matter_text))
            return None
        if not isinstance(metadata, dict):
            logger.error("Invalid SKILL.md front-matter in %s: Frontmatter must be a YAML dictionary", skill_file)
            return None

        # 필수 필드를 추출한다. 둘 다 비어 있지 않은 문자열이어야 한다.
        name = metadata.get("name")
        description = metadata.get("description")

        if not name or not isinstance(name, str):
            return None
        if not description or not isinstance(description, str):
            return None

        # 정규화: YAML이 남길 수 있는 앞뒤 공백을 제거한다.
        name = name.strip()
        description = description.strip()

        if not name or not description:
            return None

        license_text = metadata.get("license")
        if license_text is not None:
            license_text = str(license_text).strip() or None

        try:
            allowed_tools = parse_allowed_tools(metadata.get("allowed-tools"), skill_file)
        except ValueError as exc:
            logger.error("Invalid allowed-tools in %s: %s", skill_file, exc)
            return None

        try:
            required_secrets = parse_required_secrets(metadata.get("required-secrets"), skill_file)
        except ValueError as exc:
            logger.error("Invalid required-secrets in %s: %s", skill_file, exc)
            return None

        secrets_autonomous = parse_secrets_autonomous(metadata.get("secrets-autonomous"), skill_file)

        return Skill(
            name=name,
            description=description,
            license=license_text,
            skill_dir=skill_file.parent,
            skill_file=skill_file,
            relative_path=relative_path or Path(skill_file.parent.name),
            category=category,
            allowed_tools=allowed_tools,
            enabled=True,  # 실제 상태는 extensions config 파일에서 온다.
            required_secrets=required_secrets,
            secrets_autonomous=secrets_autonomous,
        )

    except Exception:
        logger.exception("Unexpected error parsing skill file %s", skill_file)
        return None
