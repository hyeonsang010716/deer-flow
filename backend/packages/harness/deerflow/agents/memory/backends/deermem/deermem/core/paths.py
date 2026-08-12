"""DeerMem 자체 저장 경로 해석(deer-flow의 ``get_paths`` / ``AGENT_NAME_PATTERN``을 쓰지 않는다).

DeerMem이 데이터를 어디에 저장할지는 더 이상 host가 정하지 않는다. 루트는
``config.storage_path``(설정된 경우, 절대/상대 모두 가능), ``$DEERMEM_DATA_DIR``,
``~/.deermem/`` 순으로 정해진다. 사용자마다 프로젝트와 무관한 요약을 담는 전역
``memory.json``이 하나씩 있다. agent별 fact는 ``agents/{agent_name}/facts`` 아래에
두며, 그 JSON 문서에는 fact 색인을 절대 추가하지 않는다.

user_id는 프로세스 안에서 정규화하고(``[A-Za-z0-9_-]``, 손실이 생기는 id에는 SHA-256
digest를 덧붙인다), agent_name은 인라인 패턴으로 검증한다. DeerMem은 host의
``make_safe_user_id`` / ``_validate_user_id`` / ``AGENT_NAME_PATTERN``을 import하지 않는다.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeerMemConfig

# user_id 문자 집합과 정규화 규칙. host의 make_safe_user_id와 동일하게 맞춰
# 마이그레이션 후에도 기존 사용자별 bucket이 그대로 이어지게 한다.
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_UNSAFE_USER_ID_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]")
_SAFE_USER_ID_DIGEST_HEX_LEN = 16

# agent_name 검증용 패턴(deer-flow의 AGENT_NAME_PATTERN을 인라인으로 가져왔다).
AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
# 호출자가 ``agent_name``을 생략했을 때 쓰는 내부 bucket. 밑줄 덕분에
# AGENT_NAME_PATTERN이 허용하는 공개 custom-agent 이름 공간과 겹치지 않는다.
DEFAULT_AGENT_BUCKET = "__default__"


def safe_user_id(raw: str) -> str:
    """외부 identity를 user-id 문자 집합(``[A-Za-z0-9_-]``)으로 정규화한다.

    멱등하다. 이미 안전한 id는 그대로 통과하고, 변환에서 정보가 손실되는 id에는
    짧은 SHA-256 digest를 접미사로 붙여 서로 다른 입력이 같은 bucket을 쓰지 않게 한다.
    host의 ``make_safe_user_id``와 동일한 규칙이라 마이그레이션 후에도 기존
    사용자별 bucket이 그대로 이어진다.
    """
    if not raw:
        raise ValueError("user_id must be a non-empty string.")
    sanitized = _UNSAFE_USER_ID_CHAR_RE.sub("-", raw)
    if sanitized == raw:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_SAFE_USER_ID_DIGEST_HEX_LEN]
    return f"{sanitized}-{digest}"


def validate_agent_name(name: str) -> None:
    """agent 이름이 파일 시스템 경로에 써도 안전한지 검증한다."""
    if not name:
        raise ValueError("Agent name must be a non-empty string.")
    if name != DEFAULT_AGENT_BUCKET and not AGENT_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid agent name {name!r}: names must match {AGENT_NAME_PATTERN.pattern}")


def _default_root() -> Path:
    """DeerMem 기본 데이터 루트를 반환한다: ``$DEERMEM_DATA_DIR`` 또는 ``~/.deermem/``."""
    env = os.environ.get("DEERMEM_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".deermem"


def memory_file_path(
    config: DeerMemConfig,
    agent_name: str | None = None,
    *,
    user_id: str | None = None,
) -> Path:
    """DeerMem 데이터 루트 아래의 memory 파일 경로를 해석한다.

    루트는 ``config.storage_path``(절대/상대 모두 가능)이고, 비어 있으면 기본 루트
    (``$DEERMEM_DATA_DIR`` / ``~/.deermem/``)를 쓴다. host(deer-flow factory)는
    ``storage_path``로 절대 base_dir를 주입하므로 memory는 CWD와 무관하게
    ``{base_dir}/users/{user_id}/memory.json``에 놓인다.
    """
    root = Path(config.storage_path) if config.storage_path else _default_root()
    if config.strict_user_scope and user_id is None:
        raise ValueError("user_id is required when strict_user_scope is enabled.")
    manifest_filename = config.manifest_filename
    if Path(manifest_filename).name != manifest_filename or not manifest_filename.endswith(".json"):
        raise ValueError("manifest_filename must be a plain .json filename.")

    if user_id is not None:
        uid = safe_user_id(user_id)
        if agent_name is not None:
            validate_agent_name(agent_name)
        bucket = root / "users" / uid
        return bucket / manifest_filename
    # 레거시 경로: user_id가 없는 경우
    if agent_name is not None:
        validate_agent_name(agent_name)
    bucket = root
    return bucket / manifest_filename


def agent_facts_directory(memory_path: Path, agent_name: str) -> Path:
    """사용자 memory 파일 아래에서 지정한 agent의 fact 루트를 반환한다."""
    validate_agent_name(agent_name)
    return memory_path.parent / "agents" / agent_name.lower() / "facts"


def fact_file_path(memory_path: Path, fact_id: str, *, agent_name: str) -> Path:
    """agent가 소유한 fact 하나의 샤딩된 Markdown 경로를 반환한다."""
    if not fact_id or not re.fullmatch(r"[A-Za-z0-9_-]+", fact_id):
        raise ValueError("Fact id may contain only letters, numbers, '_' and '-'.")
    prefix = hashlib.sha256(fact_id.encode("utf-8")).hexdigest()[:2]
    return agent_facts_directory(memory_path, agent_name) / prefix / f"{fact_id}.md"
