from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from deerflow.config import get_paths

from .diff import compare_snapshots, get_changed_paths
from .scanner import scan_workspace_roots
from .types import (
    WORKSPACE_CHANGES_EVENT_CATEGORY,
    WORKSPACE_CHANGES_EVENT_TYPE,
    WORKSPACE_CHANGES_METADATA_KEY,
    WorkspaceChangeLimits,
    WorkspaceRoot,
    WorkspaceSnapshot,
)

logger = logging.getLogger(__name__)


def build_thread_workspace_roots(thread_id: str, *, user_id: str | None = None) -> list[WorkspaceRoot]:
    paths = get_paths()
    return [
        WorkspaceRoot(
            name="workspace",
            host_path=paths.sandbox_work_dir(thread_id, user_id=user_id),
            virtual_prefix="/mnt/user-data/workspace",
        ),
        WorkspaceRoot(
            name="outputs",
            host_path=paths.sandbox_outputs_dir(thread_id, user_id=user_id),
            virtual_prefix="/mnt/user-data/outputs",
        ),
    ]


def _prepare_capture(thread_id: str, *, user_id: str | None, include_text: bool) -> tuple[list[WorkspaceRoot], Path | None]:
    # worker thread에서 실행한다. sandbox root 해석은 파일시스템에 접근하고 mkdtemp는 text
    # cache 디렉터리를 만든다. 둘 다 event loop 밖에 있어야 하는 blocking IO다.
    roots = build_thread_workspace_roots(thread_id, user_id=user_id)
    text_cache_dir = Path(tempfile.mkdtemp(prefix="deerflow-workspace-changes-")) if include_text else None
    return roots, text_cache_dir


async def _remove_text_cache_dir(text_cache_dir: str | Path) -> None:
    """snapshot의 text cache를 event loop 밖에서 제거한다.

    계약상 best-effort다. 모든 호출자가 실패 또는 teardown 경로이므로, 정리 중 발생한 에러가 이미
    진행 중인 예외나 결과를 대체해서는 안 된다.
    """
    try:
        await asyncio.to_thread(shutil.rmtree, text_cache_dir, ignore_errors=True)
    except Exception:
        logger.warning("Failed to remove workspace text cache %s", text_cache_dir, exc_info=True)


async def _reclaim_prepare_and_cleanup(prepare: asyncio.Future[tuple[list[WorkspaceRoot], Path | None]]) -> None:
    """취소된 prepare 인계를 기다린 뒤 그것이 만든 디렉터리를 제거한다.

    자체 task로 소유한다. 회수 중 호출자가 취소되면 *await*는 끊기지만 방금 만들어진 text cache
    디렉터리는 절대 방치되지 않는다. `_remove_text_cache_dir`와 마찬가지로 best-effort다.
    """
    try:
        _, orphaned = await prepare
    except Exception:
        return  # prepare가 디렉터리를 만들기 전에 실패했다. 회수할 것이 없다
    if orphaned is not None:
        await _remove_text_cache_dir(orphaned)


async def capture_workspace_snapshot(
    thread_id: str,
    *,
    user_id: str | None = None,
    limits: WorkspaceChangeLimits | None = None,
    include_text: bool = True,
    extra_excluded_dir_names: frozenset[str] | None = None,
) -> WorkspaceSnapshot:
    # `_prepare_capture`는 worker 안에서 text cache 디렉터리를 만들기 때문에 인계가 취소에
    # 안전해야 한다. mkdtemp 이후 경로를 받기 전에 run이 취소되면, shield된 worker는 그대로 끝나고
    # 그 결과를 회수해 방치된 디렉터리를 제거한 뒤 예외를 다시 던진다.
    prepare = asyncio.ensure_future(asyncio.to_thread(_prepare_capture, thread_id, user_id=user_id, include_text=include_text))
    try:
        roots, text_cache_dir = await asyncio.shield(prepare)
    except asyncio.CancelledError:
        # `prepare`는 shield되어 있으므로 이 취소 이후에도 계속 실행되어 디렉터리를 만들 수 있다.
        # 회수와 제거는 호출자가 버릴 수 없는 task가 소유한다. 재취소는 우리 await만 끊고 task는
        # 끊지 못하므로, cleanup이 끝날 때까지 반복 취소를 흡수한 뒤 취소를 복원한다. await에
        # `shield()`를 한 번 더 거는 것만으로는 재취소가 회수를 건너뛰어 디렉터리가 방치된다.
        cleanup = asyncio.ensure_future(_reclaim_prepare_and_cleanup(prepare))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass
        raise
    try:
        return await asyncio.to_thread(
            scan_workspace_roots,
            roots,
            limits=limits,
            include_text=include_text,
            text_cache_dir=text_cache_dir,
            extra_excluded_dir_names=extra_excluded_dir_names,
        )
    except Exception:
        if text_cache_dir is not None:
            await _remove_text_cache_dir(text_cache_dir)
        raise


async def record_workspace_changes(
    event_store: Any,
    thread_id: str,
    run_id: str,
    before: WorkspaceSnapshot,
    *,
    user_id: str | None = None,
    limits: WorkspaceChangeLimits | None = None,
    extra_excluded_dir_names: frozenset[str] | None = None,
) -> dict | None:
    try:
        roots = await asyncio.to_thread(build_thread_workspace_roots, thread_id, user_id=user_id)
        after_metadata = await asyncio.to_thread(
            scan_workspace_roots,
            roots,
            limits=limits,
            include_text=False,
            extra_excluded_dir_names=extra_excluded_dir_names,
        )
        changed_paths = get_changed_paths(before, after_metadata)
        after = await asyncio.to_thread(
            scan_workspace_roots,
            roots,
            limits=limits,
            include_text=True,
            text_paths=changed_paths,
            extra_excluded_dir_names=extra_excluded_dir_names,
        )
        result = compare_snapshots(before, after, limits=limits)
        if not result.has_changes():
            return None

        payload = result.to_dict()
        summary = result.summary
        changed_file_count = summary.created + summary.modified + summary.deleted + summary.symlink_created
        content = f"{changed_file_count} file{'s' if changed_file_count != 1 else ''} changed +{summary.additions} -{summary.deletions}"
        return await event_store.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type=WORKSPACE_CHANGES_EVENT_TYPE,
            category=WORKSPACE_CHANGES_EVENT_CATEGORY,
            content=content,
            metadata={WORKSPACE_CHANGES_METADATA_KEY: payload},
        )
    finally:
        await _cleanup_snapshot_text_cache(before)


async def _cleanup_snapshot_text_cache(snapshot: WorkspaceSnapshot) -> None:
    if snapshot.text_cache_dir:
        await _remove_text_cache_dir(snapshot.text_cache_dir)
