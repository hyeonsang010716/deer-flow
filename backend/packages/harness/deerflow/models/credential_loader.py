"""Claude Code CLI와 Codex CLI에서 credential을 자동으로 로드한다.

두 가지 credential 전략을 구현한다:
  1. 명시적 환경 변수 또는 내보낸 credential 파일에서 얻는 Claude Code OAuth 토큰
     - Authorization: Bearer header를 쓴다(x-api-key가 아니다)
     - anthropic-beta: oauth-2025-04-20,claude-code-20250219이 필요하다
     - $CLAUDE_CODE_OAUTH_TOKEN, $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR,
       $ANTHROPIC_AUTH_TOKEN을 지원한다
     - $CLAUDE_CODE_CREDENTIALS_PATH로 경로를 덮어쓸 수 있다
  2. ~/.codex/auth.json에서 얻는 Codex CLI 토큰
     - chatgpt.com/backend-api/codex/responses endpoint를 쓴다
     - legacy 최상위 토큰 형태와 현재의 중첩 토큰 형태를 모두 지원한다
     - $CODEX_AUTH_PATH로 경로를 덮어쓸 수 있다
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Claude Code OAuth 토큰에 필요한 beta header
OAUTH_ANTHROPIC_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"


def is_oauth_token(token: str) -> bool:
    """토큰이 (표준 API key가 아닌) Claude Code OAuth 토큰인지 확인한다."""
    return isinstance(token, str) and "sk-ant-oat" in token


@dataclass
class ClaudeCodeCredential:
    """Claude Code CLI의 OAuth credential."""

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0
    source: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000  # 1분 여유


@dataclass
class CodexCliCredential:
    """Codex CLI의 credential."""

    access_token: str
    account_id: str = ""
    source: str = ""


def _resolve_credential_path(env_var: str, default_relative_path: str) -> Path:
    configured_path = os.getenv(env_var)
    if configured_path:
        return Path(configured_path).expanduser()
    return _home_dir() / default_relative_path


def _home_dir() -> Path:
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    return Path.home()


def _load_json_file(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        logger.debug(f"{label} not found: {path}")
        return None
    if path.is_dir():
        logger.warning(f"{label} path is a directory, expected a file: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {label}: {e}")
        return None


def _read_secret_from_file_descriptor(env_var: str) -> str | None:
    fd_value = os.getenv(env_var)
    if not fd_value:
        return None

    try:
        fd = int(fd_value)
    except ValueError:
        logger.warning(f"{env_var} must be an integer file descriptor, got: {fd_value}")
        return None

    try:
        secret = os.read(fd, 1024 * 1024).decode().strip()
    except OSError as e:
        logger.warning(f"Failed to read {env_var}: {e}")
        return None

    return secret or None


def _credential_from_direct_token(access_token: str, source: str) -> ClaudeCodeCredential | None:
    token = access_token.strip()
    if not token:
        return None
    return ClaudeCodeCredential(access_token=token, source=source)


def _iter_claude_code_credential_paths() -> list[Path]:
    paths: list[Path] = []
    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    if override_path:
        paths.append(Path(override_path).expanduser())

    default_path = _home_dir() / ".claude/.credentials.json"
    if not paths or paths[-1] != default_path:
        paths.append(default_path)

    return paths


def _extract_claude_code_credential(data: dict[str, Any], source: str) -> ClaudeCodeCredential | None:
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    if not access_token:
        logger.debug("Claude Code credentials container exists but no accessToken found")
        return None

    cred = ClaudeCodeCredential(
        access_token=access_token,
        refresh_token=oauth.get("refreshToken", ""),
        expires_at=oauth.get("expiresAt", 0),
        source=source,
    )

    if cred.is_expired:
        logger.warning("Claude Code OAuth token is expired. Run 'claude' to refresh.")
        return None

    return cred


def load_claude_code_credential() -> ClaudeCodeCredential | None:
    """명시적인 Claude Code 인계 소스에서 OAuth credential을 로드한다.

    조회 순서:
      1. $CLAUDE_CODE_OAUTH_TOKEN 또는 $ANTHROPIC_AUTH_TOKEN
      2. $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
      3. $CLAUDE_CODE_CREDENTIALS_PATH
      4. ~/.claude/.credentials.json

    내보낸 credential 파일의 구조:
    {
      "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt": 1773430695128,
        "scopes": ["user:inference", ...],
        ...
      }
    }
    """
    direct_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if direct_token:
        cred = _credential_from_direct_token(direct_token, "claude-cli-env")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from environment")
        return cred

    fd_token = _read_secret_from_file_descriptor("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if fd_token:
        cred = _credential_from_direct_token(fd_token, "claude-cli-fd")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from file descriptor")
        return cred

    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    override_path_obj = Path(override_path).expanduser() if override_path else None
    for cred_path in _iter_claude_code_credential_paths():
        data = _load_json_file(cred_path, "Claude Code credentials")
        if data is None:
            continue
        cred = _extract_claude_code_credential(data, "claude-cli-file")
        if cred:
            source_label = "override path" if override_path_obj is not None and cred_path == override_path_obj else "plaintext file"
            logger.info(f"Loaded Claude Code OAuth credential from {source_label} (expires_at={cred.expires_at})")
            return cred

    return None


def load_codex_cli_credential() -> CodexCliCredential | None:
    """Codex CLI(~/.codex/auth.json)에서 credential을 로드한다."""
    cred_path = _resolve_credential_path("CODEX_AUTH_PATH", ".codex/auth.json")
    data = _load_json_file(cred_path, "Codex CLI credentials")
    if data is None:
        return None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    access_token = data.get("access_token") or data.get("token") or tokens.get("access_token", "")
    account_id = data.get("account_id") or tokens.get("account_id", "")
    if not access_token:
        logger.debug("Codex CLI credentials file exists but no token found")
        return None

    logger.info("Loaded Codex CLI credential")
    return CodexCliCredential(
        access_token=access_token,
        account_id=account_id,
        source="codex-cli",
    )
