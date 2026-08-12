"""임의의 BaseCheckpointSaver를 감싸는 read-through delta-history cache wrapper.

정확성 논거(spec §3): checkpoint의 delta history는 봉인된 ancestor chain의 순수 함수다.
LangGraph 계약상 target 자신의 pending write는 제외되고, parent link는 생성 시점에 고정되며,
ancestor의 write는 자식이 생기는 순간 봉인된다. 따라서 (thread, ns, checkpoint_id, channel)로
키가 잡힌 항목은 불변이다. invalidation이 필요 없고, 공유 backend는 프로세스 간에도 일관된다.

wrapper는 "최신 checkpoint" 해석 결과는 절대 캐시하지 않는다. 이미 해석된 불변 checkpoint_id로
키가 잡힌 history만 캐시한다.

데이터 lifecycle: thread 삭제와 prune은 해당 thread의 캐시 항목을 비운다(원본이 지워졌는데
cache에 history payload가 남으면 안 된다). run 단위 삭제는 thread로 싸게 매핑할 수 없으므로
LRU/TTL 한도에 맡긴다.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, PendingWrite

from deerflow.runtime.checkpoint_cache.base import make_history_key

logger = logging.getLogger(__name__)

# chain을 데우는 walk로 폴백하기 전까지 재귀 compose에 허용하는 depth 예산. 정상 상태의 run은
# ~2면 충분하다(step당 중간 checkpoint 하나). 더 깊은 cold chain은 단일 tuple을 여러 번 재귀
# fetch 하는 것보다 warming walk 한 번이 빠르다.
_COMPOSE_MAX_DEPTH = 8


def _checkpoint_ref(tup: CheckpointTuple) -> tuple[str, str, str]:
    configurable = tup.config["configurable"]
    return (
        str(configurable["thread_id"]),
        str(configurable.get("checkpoint_ns", "")),
        str(configurable["checkpoint_id"]),
    )


def _channel_writes(tup: CheckpointTuple, channel: str) -> list[PendingWrite]:
    """한 channel의 write를 오래된 것→최신 순(tuple 저장 순서)으로 반환한다."""
    return [w for w in (tup.pending_writes or []) if w[1] == channel]


class CachedHistorySaver(BaseCheckpointSaver):
    def __init__(self, inner: BaseCheckpointSaver, cache: Any, *, key_prefix: str) -> None:
        # 인스턴스 속성이 base class의 JsonPlusSerializer 기본값을 가린다.
        self.serde = inner.serde
        self._inner = inner
        self._cache = cache
        self._key_prefix = key_prefix
        self._compose_hits = 0
        self._full_walks = 0

    def __getattr__(self, name: str) -> Any:
        # saver 고유 확장(예: AsyncSqliteSaver.setup)을 위한 안전망. base class 메서드는 아래에서
        # 명시적으로 위임하므로, 여기는 BaseCheckpointSaver가 정의하지 않은 속성에만 걸린다.
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # ------------------------------------------------------------------
    # 키 구성
    # ------------------------------------------------------------------

    def _key(self, tup: CheckpointTuple, channel: str) -> str:
        thread_id, ns, checkpoint_id = _checkpoint_ref(tup)
        return make_history_key(self._key_prefix, thread_id, ns, checkpoint_id, channel)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        backend = self._cache.stats().as_dict()
        return {**backend, "compose_hits": self._compose_hits, "full_walks": self._full_walks}

    # ------------------------------------------------------------------
    # delta history: 유일하게 재정의하는 동작
    # ------------------------------------------------------------------

    async def aget_delta_channel_history(self, *, config: RunnableConfig, channels: Sequence[str]) -> dict[str, Any]:
        if not channels:
            return {}
        if not getattr(self._cache, "enabled", True):
            # cache가 꺼져 있으면 그대로 통과시킨다. 전부 miss인 항목을 compose 하는 것은
            # raw saver의 walk보다 무조건 더 많은 일이다.
            return await self._walk_inner(config, channels)
        target = await self._inner.aget_tuple(config)
        if target is None:
            return await self._walk_inner(config, channels)

        keys = {ch: self._key(target, ch) for ch in channels}
        hits = await self._cache.aget_many(list(keys.values()))
        found: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for ch in channels:
            entry = hits.get(keys[ch])
            if entry is None:
                missing.append(ch)
            else:
                found[ch] = entry

        computed: dict[str, dict[str, Any]] = {}
        if missing:
            computed = await self._compose_or_walk(config, target, missing)

        new_entries = {keys[ch]: computed[ch] for ch in missing if ch in computed}
        if new_entries:
            await self._cache.aset_many(new_entries)

        return {ch: found.get(ch) or computed.get(ch) or {"writes": []} for ch in channels}

    async def _compose_or_walk(self, config: RunnableConfig, target: CheckpointTuple, missing: list[str]) -> dict[str, dict[str, Any]]:
        return {ch: await self._aresolve(target, ch, _COMPOSE_MAX_DEPTH) for ch in missing}

    async def _aresolve(self, tup: CheckpointTuple, channel: str, depth: int) -> dict[str, Any]:
        """가장 가까운 warm ancestor에서 history(tup)를 재귀적으로 합성한다.

        실제 run은 super-step마다 checkpoint를 여러 개 만들지만 그중 일부만 target으로
        materialize 되므로, parent는 보통 데워지지 않은 중간 checkpoint다(측정: 500-step sqlite
        run에서 단일 레벨 compose는 cache hit이 0이었다). 중간 checkpoint마다 한 레벨씩 재귀하면
        정상 상태에서는 ~2레벨 안에 warm ancestor에 닿는다. 합성한 레벨은 매번 캐시되므로 warm
        경계가 run을 따라 이동한다. cold chain에서 depth가 0이 되면 ancestor tuple을 하나씩 훑는
        대신 내부 fast-path walk 한 번(SQL 2회)에 위임한다.
        """
        parent_config = tup.parent_config
        if parent_config is None:
            return {"writes": []}
        parent = await self._inner.aget_tuple(parent_config)
        if parent is None:
            return {"writes": []}

        channel_values = parent.checkpoint.get("channel_values") or {}
        writes = _channel_writes(parent, channel)
        if channel in channel_values:
            self._compose_hits += 1
            return {"writes": writes, "seed": channel_values[channel]}

        key = self._key(parent, channel)
        hits = await self._cache.aget_many([key])
        parent_history = hits.get(key)
        if parent_history is None:
            if depth > 0:
                parent_history = await self._aresolve(parent, channel, depth - 1)
            else:
                # cold chain에서 depth 예산이 바닥났다. ancestor tuple을 하나씩 가져오는 대신
                # 이 레벨에 대해 내부 fast-path walk를 딱 한 번(총 SQL 2회) 위임한다. 아래쪽
                # ancestor는 cold로 남지만, 나중에 그것을 해석할 때 가장 가까운 warm 레벨까지
                # 재귀하므로 warm 경계는 여전히 run을 따라간다.
                self._full_walks += 1
                walked = await self._inner.aget_delta_channel_history(config=parent.config, channels=[channel])
                parent_history = walked.get(channel) or {"writes": []}
            if parent_history is not None:
                await self._cache.aset_many({key: parent_history})

        self._compose_hits += 1
        entry: dict[str, Any] = {"writes": list(parent_history["writes"]) + writes}
        if "seed" in parent_history:
            entry["seed"] = parent_history["seed"]
        return entry

    async def _walk_inner(self, config: RunnableConfig, channels: Sequence[str]) -> dict[str, Any]:
        self._full_walks += 1
        return dict(await self._inner.aget_delta_channel_history(config=config, channels=channels))

    def get_delta_channel_history(self, *, config: RunnableConfig, channels: Sequence[str]) -> dict[str, Any]:
        if not channels:
            return {}
        if not getattr(self._cache, "enabled", True):
            return self._walk_inner_sync(config, channels)
        get_many = getattr(self._cache, "get_many", None)
        set_many = getattr(self._cache, "set_many", None)
        if get_many is None or set_many is None:
            raise TypeError("sync get_delta_channel_history requires a SyncCheckpointHistoryCache (memory backend)")
        target = self._inner.get_tuple(config)
        if target is None:
            return self._walk_inner_sync(config, channels)

        keys = {ch: self._key(target, ch) for ch in channels}
        hits = get_many(list(keys.values()))
        found: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for ch in channels:
            entry = hits.get(keys[ch])
            if entry is None:
                missing.append(ch)
            else:
                found[ch] = entry

        computed: dict[str, dict[str, Any]] = {}
        if missing:
            computed = {ch: self._resolve_sync(target, ch, _COMPOSE_MAX_DEPTH) for ch in missing}

        new_entries = {keys[ch]: computed[ch] for ch in missing if ch in computed}
        if new_entries:
            set_many(new_entries)
        return {ch: found.get(ch) or computed.get(ch) or {"writes": []} for ch in channels}

    def _resolve_sync(self, tup: CheckpointTuple, channel: str, depth: int) -> dict[str, Any]:
        """_aresolve의 동기 버전(재귀 compose, 자세한 내용은 그쪽 docstring 참고)."""
        parent_config = tup.parent_config
        if parent_config is None:
            return {"writes": []}
        parent = self._inner.get_tuple(parent_config)
        if parent is None:
            return {"writes": []}

        channel_values = parent.checkpoint.get("channel_values") or {}
        writes = _channel_writes(parent, channel)
        if channel in channel_values:
            self._compose_hits += 1
            return {"writes": writes, "seed": channel_values[channel]}

        key = self._key(parent, channel)
        parent_history = self._cache.get_many([key]).get(key)
        if parent_history is None:
            if depth > 0:
                parent_history = self._resolve_sync(parent, channel, depth - 1)
            else:
                # _aresolve 참고: tuple 단위로 훑지 않고 내부 fast-path walk를 한 번만 쓴다.
                self._full_walks += 1
                parent_history = self._inner.get_delta_channel_history(config=parent.config, channels=[channel]).get(channel) or {"writes": []}
            if parent_history is not None:
                self._cache.set_many({key: parent_history})

        self._compose_hits += 1
        entry: dict[str, Any] = {"writes": list(parent_history["writes"]) + writes}
        if "seed" in parent_history:
            entry["seed"] = parent_history["seed"]
        return entry

    def _walk_inner_sync(self, config: RunnableConfig, channels: Sequence[str]) -> dict[str, Any]:
        self._full_walks += 1
        return dict(self._inner.get_delta_channel_history(config=config, channels=channels))

    # ------------------------------------------------------------------
    # 명시적 위임(BaseCheckpointSaver가 이들을 구체적으로 정의하므로
    # __getattr__은 절대 걸리지 않는다)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._inner.get_tuple(config)

    def list(self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None, before: RunnableConfig | None = None, limit: int | None = None) -> Iterator[CheckpointTuple]:
        return self._inner.list(config, filter=filter, before=before, limit=limit)

    def put(self, config: RunnableConfig, checkpoint: dict[str, Any], metadata: dict[str, Any], new_versions: dict[str, Any]) -> RunnableConfig:
        return self._inner.put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, str, Any]], task_id: str, task_path: str = "") -> None:
        self._inner.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._inner.delete_thread(thread_id)
        self._purge_thread_sync(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        # run 단위 삭제는 추가 질의 없이 thread로 되돌려 매핑할 수 없으므로 캐시 항목을 그대로
        # 둔다. 다른 chain은 봉인 논거에 영향을 주지 않으므로 항목은 여전히 *정확*하고, 남는
        # 데이터는 LRU/TTL로 제한된다. 현재 트리 안에는 호출자가 없다.
        self._inner.delete_for_runs(run_ids)

    def _purge_thread_sync(self, thread_id: str) -> None:
        delete = getattr(self._cache, "delete_thread", None)
        if delete is not None:
            delete(self._key_prefix, thread_id)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._inner.copy_thread(source_thread_id, target_thread_id)

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        self._inner.prune(thread_ids, strategy=strategy)
        # prune은 해당 thread의 chain을 다시 쓰므로, 캐시된 history가 삭제된 ancestor나
        # prune 이전 chain을 참조하지 않도록 비운다.
        for thread_id in thread_ids:
            self._purge_thread_sync(thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._inner.aget_tuple(config)

    def alist(self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None, before: RunnableConfig | None = None, limit: int | None = None) -> Any:
        return self._inner.alist(config, filter=filter, before=before, limit=limit)

    async def aput(self, config: RunnableConfig, checkpoint: dict[str, Any], metadata: dict[str, Any], new_versions: dict[str, Any]) -> RunnableConfig:
        return await self._inner.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, str, Any]], task_id: str, task_path: str = "") -> None:
        await self._inner.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await self._inner.adelete_thread(thread_id)
        await self._apurge_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        # delete_for_runs 참고: 범위를 좁히지 못한 잔여 데이터는 LRU/TTL로 제한된다.
        await self._inner.adelete_for_runs(run_ids)

    async def _apurge_thread(self, thread_id: str) -> None:
        delete = getattr(self._cache, "adelete_thread", None)
        if delete is not None:
            await delete(self._key_prefix, thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self._inner.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await self._inner.aprune(thread_ids, strategy=strategy)
        # prune 참고: 다시 쓰인 chain이 prune 이전 캐시 history를 들고 있으면 안 된다.
        for thread_id in thread_ids:
            await self._apurge_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self._inner.get_next_version(current, channel)
