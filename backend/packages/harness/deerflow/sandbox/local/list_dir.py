from pathlib import Path

from deerflow.sandbox.search import should_ignore_name


def list_dir(path: str, max_depth: int = 2) -> list[str]:
    """
    max_depth 단계까지 파일과 디렉터리를 나열한다.

    Args:
        path: 나열할 루트 디렉터리 경로.
        max_depth: 순회할 최대 깊이(기본값: 2).
                   1 = 직계 자식만, 2 = 자식 + 손자, 이런 식이다.

    Returns:
        파일과 디렉터리의 절대 경로 리스트. IGNORE_PATTERNS에 걸리는 항목은 제외한다.
    """
    result: list[str] = []
    root_path = Path(path).resolve()

    if not root_path.is_dir():
        return result

    def _is_within_root(candidate: Path) -> bool:
        try:
            candidate.relative_to(root_path)
            return True
        except ValueError:
            return False

    def _traverse(current_path: Path, current_depth: int) -> None:
        """max_depth까지 디렉터리를 재귀적으로 순회한다."""
        if current_depth > max_depth:
            return

        try:
            for item in current_path.iterdir():
                if should_ignore_name(item.name):
                    continue

                if item.is_symlink():
                    try:
                        item_resolved = item.resolve()
                        if not _is_within_root(item_resolved):
                            continue
                    except OSError:
                        continue
                    post_fix = "/" if item_resolved.is_dir() else ""
                    result.append(str(item_resolved) + post_fix)
                    continue

                item_resolved = item.resolve()
                if not _is_within_root(item_resolved):
                    continue

                post_fix = "/" if item.is_dir() else ""
                result.append(str(item_resolved) + post_fix)

                # 최대 깊이가 아니면 하위 디렉터리로 재귀한다
                if item.is_dir() and current_depth < max_depth:
                    _traverse(item, current_depth + 1)
        except PermissionError:
            pass

    _traverse(root_path, 1)

    return sorted(result)
