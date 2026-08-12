"""harness를 단독으로 쓸 때의 런타임 경로 해석."""

import os
from pathlib import Path


def project_root() -> Path:
    """런타임이 소유하는 파일들의 기준이 되는 호출자 프로젝트 루트를 반환한다."""
    if env_root := os.getenv("DEER_FLOW_PROJECT_ROOT"):
        root = Path(env_root).resolve()
        if not root.exists():
            raise ValueError(f"DEER_FLOW_PROJECT_ROOT is set to '{env_root}', but the resolved path '{root}' does not exist.")
        if not root.is_dir():
            raise ValueError(f"DEER_FLOW_PROJECT_ROOT is set to '{env_root}', but the resolved path '{root}' is not a directory.")
        return root
    return Path.cwd().resolve()


def runtime_home() -> Path:
    """쓰기 가능한 DeerFlow 상태 디렉터리를 반환한다."""
    if env_home := os.getenv("DEER_FLOW_HOME"):
        return Path(env_home).resolve()
    return project_root() / ".deer-flow"


def resolve_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """절대 경로는 그대로, 상대 경로는 프로젝트 루트 기준으로 해석한다."""
    path = Path(value)
    if not path.is_absolute():
        path = (base or project_root()) / path
    return path.resolve()


def existing_project_file(names: tuple[str, ...]) -> Path | None:
    """프로젝트 루트에서 주어진 이름 중 가장 먼저 존재하는 파일을 반환한다."""
    root = project_root()
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None
