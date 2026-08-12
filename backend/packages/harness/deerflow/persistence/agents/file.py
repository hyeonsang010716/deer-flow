"""파일시스템 기반 agent store — 현재의 per-user 레이아웃을 동작 변경 없이 유지한다.

읽기 메서드는 리팩터링 이전의 ``load_agent_config`` / ``load_agent_soul`` /
``list_custom_agents`` 본문 그대로다(:mod:`deerflow.config.agents_config`의 free function이
동작 변경 없이 여기로 dispatch한다). 쓰기는 임시 파일에 staging한 뒤 원자적 ``os.replace``로
commit한다 — ``update_agent`` 도구가 이미 갖고 있던 crash-safety를 create/update에 동일하게
적용한 것이다.

경로/사용자 해석은 직접 import 대신 :mod:`deerflow.config.agents_config` 모듈 객체
(``_ac.get_paths`` / ``_ac.get_effective_user_id``)를 통해 수행한다. 기존 agent 테스트가
노리는 monkeypatch 지점을 그대로 존중하기 위해서다.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml

from deerflow.config import agents_config as _ac
from deerflow.config.agents_config import (
    SOUL_FILENAME,
    AgentConfig,
    resolve_agent_dir,
    validate_agent_name,
)
from deerflow.persistence.agents.base import (
    AgentDeleteOutcome,
    AgentExistsError,
    AgentStore,
    parse_agent_config,
)
from deerflow.runtime.user_context import DEFAULT_USER_ID

logger = logging.getLogger(__name__)


class FileAgentStore(AgentStore):
    def get(self, name: str, *, user_id: str | None = None) -> AgentConfig:
        name = validate_agent_name(name)
        agent_dir = resolve_agent_dir(name, user_id=user_id)
        config_file = agent_dir / "config.yaml"
        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {agent_dir}")
        if not config_file.exists():
            raise FileNotFoundError(f"Agent config not found: {config_file}")
        try:
            with open(config_file, encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse agent config {config_file}: {e}") from e
        return parse_agent_config(data, name)

    def exists(self, name: str, *, user_id: str | None = None) -> bool:
        name = validate_agent_name(name)
        paths = _ac.get_paths()
        effective_user = user_id or _ac.get_effective_user_id()
        return paths.user_agent_dir(effective_user, name).exists() or paths.agent_dir(name).exists()

    def get_soul(self, name: str, *, user_id: str | None = None) -> str | None:
        agent_dir = resolve_agent_dir(name, user_id=user_id)
        soul_path = agent_dir / SOUL_FILENAME
        # resolve_agent_dir는 config.yaml을 요구하지만 SOUL.md 로딩은 그렇지 않다. 따라서
        # resolver가 조건을 만족하는 디렉터리를 못 찾아 기본 경로로 fallback한 경우 per-user와
        # legacy 디렉터리를 직접 확인한다(#4135). config.yaml 가드는 SOUL.md만 없는 정상
        # per-user agent에서 이 fallback이 발동하지 않게 해서 per-user가 legacy를 가리는
        # 동작을 유지한다.
        if not soul_path.exists() and not (agent_dir / "config.yaml").exists():
            paths = _ac.get_paths()
            effective_user = user_id or _ac.get_effective_user_id()
            for candidate in (
                paths.user_agent_dir(effective_user, name),
                paths.agent_dir(name),
            ):
                if (candidate / SOUL_FILENAME).exists():
                    soul_path = candidate / SOUL_FILENAME
                    break
        if not soul_path.exists():
            return None
        content = soul_path.read_text(encoding="utf-8").strip()
        return content or None

    def list(self, *, user_id: str | None = None) -> list[AgentConfig]:
        paths = _ac.get_paths()
        effective_user = user_id or _ac.get_effective_user_id()
        seen: set[str] = set()
        agents: list[AgentConfig] = []
        for root in (paths.user_agents_dir(effective_user), paths.agents_dir):
            if not root.exists():
                continue
            for entry in sorted(root.iterdir()):
                if not entry.is_dir() or entry.name in seen:
                    continue
                if not (entry / "config.yaml").exists():
                    logger.debug("Skipping %s: no config.yaml", entry.name)
                    continue
                try:
                    agents.append(self.get(entry.name, user_id=effective_user))
                    seen.add(entry.name)
                except Exception as e:  # noqa: BLE001 — agent 하나가 잘못돼도 나머지를 가리면 안 된다
                    logger.warning("Skipping agent '%s': %s", entry.name, e)
        agents.sort(key=lambda a: a.name)
        return agents

    def list_all(self) -> list[tuple[str, AgentConfig]]:
        result: list[tuple[str, AgentConfig]] = []
        for user_id, name in self._discover():
            try:
                result.append((user_id, self.get(name, user_id=user_id)))
            except Exception as e:  # noqa: BLE001
                logger.warning("list_all: skipping agent %s/%s: %s", user_id, name, e)
        return result

    def create(self, name: str, config: dict, soul: str, *, user_id: str | None = None) -> None:
        name = validate_agent_name(name)
        paths = _ac.get_paths()
        effective_user = user_id or _ac.get_effective_user_id()
        agent_dir = paths.user_agent_dir(effective_user, name)
        # per-user 디렉터리나 legacy 공유 디렉터리가 이미 그 이름을 갖고 있으면 거부한다 —
        # agents router의 409 의미(legacy agent는 가려지면 안 되고, per-user 디렉터리는
        # memory 전용이더라도 여전히 이름을 막는다).
        if agent_dir.exists() or paths.agent_dir(name).exists():
            raise AgentExistsError(f"Agent '{name}' already exists for user '{effective_user}'")
        try:
            agent_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            # 동시에 실행된 create가 위 존재 확인을 통과하고 mkdir에 먼저 도달했다. 일반적인
            # 500 대신 router의 409를 노출한다(AgentExistsError 경유) — SqlAgentStore의
            # IntegrityError 경로와 동일하다.
            raise AgentExistsError(f"Agent '{name}' already exists for user '{effective_user}'") from e
        try:
            self._write(agent_dir, config, soul)
        except Exception:
            # 이 호출에서 새로 만든 디렉터리이므로, 쓰기가 실패하면 비어 있거나 일부만 쓰인
            # agent 디렉터리를 남기면 안 된다.
            shutil.rmtree(agent_dir, ignore_errors=True)
            raise

    def update(self, name: str, config: dict | None, soul: str | None, *, user_id: str | None = None) -> None:
        name = validate_agent_name(name)
        effective_user = user_id or _ac.get_effective_user_id()
        agent_dir = _ac.get_paths().user_agent_dir(effective_user, name)
        pre_existing = agent_dir.exists()
        agent_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._write(agent_dir, config, soul)
        except Exception:
            # 이 호출이 만든 디렉터리만 정리한다 — 쓰기 실패로 기존 agent를 삭제하지 않는다.
            if not pre_existing:
                shutil.rmtree(agent_dir, ignore_errors=True)
            raise

    def delete(self, name: str, *, user_id: str | None = None) -> AgentDeleteOutcome:
        name = validate_agent_name(name)
        paths = _ac.get_paths()
        effective_user = user_id or _ac.get_effective_user_id()
        agent_dir = paths.user_agent_dir(effective_user, name)
        if not agent_dir.exists():
            # legacy 공유 레이아웃 agent는 의도적으로 그대로 둔다(쓰기 경로가 그것을 대상으로
            # 삼지 않는다). 별도 상태로 보고한다.
            return "legacy" if paths.agent_dir(name).exists() else "missing"
        if not (agent_dir / "config.yaml").is_file():
            # 이 디렉터리는 memory/facts 데이터를 담고 있을 뿐 custom agent가 아니다
            # (config.yaml 없음). 사용자의 memory를 지우는 대신 보존한다(#4279). 그대로 두면
            # 아래 rmtree가 트리 전체를 날린다.
            return "not-custom-agent"
        # rmtree가 config.yaml, SOUL.md, 같은 위치의 memory.json을 한 번에 제거한다 —
        # 기존 동작 그대로다.
        shutil.rmtree(agent_dir)
        return "deleted"

    def signature(self) -> Hashable:
        sig: list[tuple[str, str, float]] = []
        for user_id, name in self._discover():
            config = resolve_agent_dir(name, user_id=user_id) / "config.yaml"
            try:
                sig.append((user_id, name, config.stat().st_mtime))
            except OSError:
                continue
        return tuple(sig)

    # -- internals --

    def _discover(self) -> list[tuple[str, str]]:
        """per-user와 legacy 레이아웃 전체에서 ``(user_id, name)``을 열거한다.

        legacy 공유 레이아웃 agent는 ``DEFAULT_USER_ID`` 소유로 간주하며, 같은 이름의
        ``users/default/`` agent에만 가려진다 — 다른 사용자가 소유한 agent에는 가려지지 않는다.
        GitHub registry의 기존 discovery 동작과 일치한다(``load_agent_config(name)``은 legacy
        agent를 ``DEFAULT_USER_ID`` 아래에서 해석한다).
        """
        paths = _ac.get_paths()
        discovered: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        users_root = paths.base_dir / "users"
        if users_root.exists():
            for user_entry in sorted(users_root.iterdir()):
                if not user_entry.is_dir():
                    continue
                agents_root = paths.user_agents_dir(user_entry.name)
                if not agents_root.exists():
                    continue
                for entry in sorted(agents_root.iterdir()):
                    if entry.is_dir() and (entry / "config.yaml").exists():
                        key = (user_entry.name, entry.name)
                        discovered.append(key)
                        seen.add(key)
        legacy_root = paths.agents_dir
        if legacy_root.exists():
            for entry in sorted(legacy_root.iterdir()):
                if entry.is_dir() and (entry / "config.yaml").exists() and (DEFAULT_USER_ID, entry.name) not in seen:
                    discovered.append((DEFAULT_USER_ID, entry.name))
        return discovered

    @staticmethod
    def _write(agent_dir: Path, config: dict | None, soul: str | None) -> None:
        """config.yaml과 SOUL.md를 각각 원자적 ``os.replace``로 기록한다.

        각 부분은 값이 주어졌을 때만(``config``/``soul``이 None이 아닐 때) 임시 파일에 staging된
        뒤 ``os.replace``로 commit되므로, 어느 파일도 절반만 쓰인 상태로 관찰되지 않는다. 두
        commit은 순차적이며 **단일 트랜잭션이 아니다**: 그 사이에 crash가 나면 방금 교체된
        config.yaml 옆에 오래된 SOUL.md가 남을 수 있다(단일 노드, 밀리초 미만의 구간).
        ``db`` backend는 두 필드를 한 트랜잭션으로 commit한다. 여기서 파일 간 원자성이 필요해지면
        ``update_agent``의 부분 쓰기 보고를 복원한다.
        """
        pending: list[tuple[Path, Path]] = []
        staged: list[Path] = []
        try:
            if config is not None:
                config_text = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
                config_tmp = _stage_temp(agent_dir / "config.yaml", config_text)
                staged.append(config_tmp)
                pending.append((config_tmp, agent_dir / "config.yaml"))
            if soul is not None:
                soul_tmp = _stage_temp(agent_dir / SOUL_FILENAME, soul)
                staged.append(soul_tmp)
                pending.append((soul_tmp, agent_dir / SOUL_FILENAME))
            for tmp, target in pending:
                tmp.replace(target)
                staged.remove(tmp)
        finally:
            for tmp in staged:
                tmp.unlink(missing_ok=True)


def _stage_temp(target: Path, text: str) -> Path:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as f:
        f.write(text)
        return Path(f.name)
