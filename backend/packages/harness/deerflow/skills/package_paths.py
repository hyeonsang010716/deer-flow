"""skill package 상대 경로 처리를 위한 공용 헬퍼."""

from __future__ import annotations

from pathlib import PurePosixPath


def _parts(path: str | PurePosixPath) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def is_eval_fixture_path(path: str | PurePosixPath) -> bool:
    """경로가 eval fixture 디렉터리 아래에 있는지 반환한다."""
    parts = _parts(path)
    for index, part in enumerate(parts[:-1]):
        if part == "evals" and len(parts) > index + 2:
            return parts[index + 1] == "fixtures"
    return False


def is_eval_fixture_skill_md(path: str | PurePosixPath) -> bool:
    """경로가 eval fixture 안에 중첩된 SKILL.md 파일인지 반환한다."""
    parts = _parts(path)
    return bool(parts) and parts[-1] == "SKILL.md" and is_eval_fixture_path(PurePosixPath(*parts[:-1]))
