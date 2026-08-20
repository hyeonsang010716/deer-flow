from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from deerflow.runtime import CheckpointStateAccessor


class FakeCheckpointer:
    def __init__(self) -> None:
        self.sync_configs: list[dict[str, Any]] = []
        self.async_configs: list[dict[str, Any]] = []

    def get_tuple(self, config: dict[str, Any]) -> None:
        self.sync_configs.append(config)
        return None

    async def aget_tuple(self, config: dict[str, Any]) -> None:
        self.async_configs.append(config)
        return None


class FakeGraph:
    def __init__(self) -> None:
        self.checkpointer: Any = None
        self.store: Any = None
        self.calls: list[tuple[Any, ...]] = []
        self.sync_history_yields = 0
        self.async_history_yields = 0

    def get_state(self, config: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(("get", config))
        return SimpleNamespace(values={"messages": ["sync"]})

    def get_state_history(self, config: dict[str, Any], *, limit: int | None = None):
        self.calls.append(("history", config, limit))
        for index in range(4):
            if limit is not None and self.sync_history_yields >= limit:
                return
            self.sync_history_yields += 1
            yield SimpleNamespace(values={"index": index})

    def update_state(self, config: dict[str, Any], values: dict[str, Any], *, as_node: str | None = None) -> dict[str, Any]:
        self.calls.append(("update", config, values, as_node))
        return {"updated": values, "as_node": as_node}

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(("aget", config))
        return SimpleNamespace(values={"messages": ["async"]})

    async def aget_state_history(self, config: dict[str, Any], *, limit: int | None = None):
        self.calls.append(("ahistory", config, limit))
        for index in range(4):
            if limit is not None and self.async_history_yields >= limit:
                return
            self.async_history_yields += 1
            yield SimpleNamespace(values={"index": index})

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("aupdate", config, values, as_node))
        return {"updated": values, "as_node": as_node}


def _assert_config_is_copied(original: dict[str, Any], forwarded: dict[str, Any]) -> None:
    assert forwarded is not original
    assert forwarded["configurable"] is not original["configurable"]
    assert forwarded["metadata"] is not original["metadata"]
    assert forwarded["configurable"] == original["configurable"]
    assert forwarded["metadata"] == original["metadata"]


def test_mutation_graph_falls_back_to_the_thread_state_schema() -> None:
    from deerflow.agents.thread_state import ThreadState
    from deerflow.runtime.checkpoint_state import build_state_mutation_graph, graph_state_schema

    graph = build_state_mutation_graph("compact")
    assert graph_state_schema(graph) is ThreadState
    assert "messages" in graph.channels


def test_sync_accessor_binds_persistence_guards_operations_and_preserves_input() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    store = object()
    accessor = CheckpointStateAccessor.bind(graph, saver, store=store)
    config = {
        "configurable": {"thread_id": "thread-sync", "checkpoint_ns": ""},
        "metadata": {"caller": "test"},
        "tags": ["preserved"],
    }
    original = deepcopy(config)

    snapshot = accessor.get(config)
    history = accessor.history(config, limit=2)
    update = accessor.update(config, {"messages": ["changed"]}, as_node="tools")

    assert snapshot.values == {"messages": ["sync"]}
    assert [item.values for item in history] == [{"index": 0}, {"index": 1}]
    assert graph.sync_history_yields == 2
    assert update == {"updated": {"messages": ["changed"]}, "as_node": "tools"}
    assert graph.checkpointer is saver
    assert graph.store is store
    assert config == original
    for call in graph.calls:
        _assert_config_is_copied(config, call[1])
    assert graph.calls[-1][2:] == ({"messages": ["changed"]}, "tools")


@pytest.mark.anyio
async def test_async_accessor_binds_persistence_guards_operations_and_preserves_input() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    store = object()
    accessor = CheckpointStateAccessor.bind(graph, saver, store=store)
    config = {
        "configurable": {"thread_id": "thread-async", "checkpoint_ns": ""},
        "metadata": {"caller": "test"},
        "tags": ["preserved"],
    }
    original = deepcopy(config)

    snapshot = await accessor.aget(config)
    history = await accessor.ahistory(config, limit=2)
    update = await accessor.aupdate(config, {"messages": ["changed"]}, as_node="agent")

    assert snapshot.values == {"messages": ["async"]}
    assert [item.values for item in history] == [{"index": 0}, {"index": 1}]
    assert graph.async_history_yields == 2
    assert update == {"updated": {"messages": ["changed"]}, "as_node": "agent"}
    assert graph.checkpointer is saver
    assert graph.store is store
    assert config == original
    for call in graph.calls:
        _assert_config_is_copied(config, call[1])
    assert graph.calls[-1][2:] == ({"messages": ["changed"]}, "agent")


def test_sync_history_zero_limit_guards_without_consuming_a_snapshot() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-sync-zero"}}

    assert accessor.history(config, limit=0) == []
    assert len(saver.sync_configs) == 0
    assert graph.sync_history_yields == 0


@pytest.mark.anyio
async def test_async_history_zero_limit_guards_without_consuming_a_snapshot() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-async-zero"}}

    assert await accessor.ahistory(config, limit=0) == []
    assert len(saver.async_configs) == 0
    assert graph.async_history_yields == 0


class _CountingSaver(InMemorySaver):
    """InMemorySaver that counts checkpoint round-trips."""

    def __init__(self) -> None:
        super().__init__()
        self.aget_tuple_calls = 0
        self.alist_limits: list[int | None] = []

    async def aget_tuple(self, config):
        self.aget_tuple_calls += 1
        return await super().aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        self.alist_limits.append(limit)
        async for item in super().alist(config, filter=filter, before=before, limit=limit):
            yield item


def _build_counting_graph(saver):
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph

    class _State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    async def _append(state):
        return {"messages": [HumanMessage(content=f"turn-{len(state.get('messages') or [])}")]}

    builder = StateGraph(_State)
    builder.add_node("append", _append)
    builder.set_entry_point("append")
    builder.set_finish_point("append")
    return builder.compile(checkpointer=saver)


@pytest.mark.anyio
async def test_ahistory_pushes_limit_into_alist_and_reads_each_snapshot_once() -> None:
    """The history limit must reach ``checkpointer.alist`` (SQL LIMIT), and the
    read path must not add a standalone tuple fetch per call."""
    saver = _CountingSaver()
    graph = _build_counting_graph(saver)
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-counted"}}
    for _ in range(4):
        await graph.ainvoke({}, config)

    saver.aget_tuple_calls = 0
    history = await accessor.ahistory(config, limit=2)

    assert len(history) == 2
    assert saver.alist_limits[-1] == 2
    assert saver.aget_tuple_calls == 0


@pytest.mark.anyio
async def test_aget_fetches_the_checkpoint_exactly_once() -> None:
    """Reads perform exactly one fetch inside aget_state."""
    saver = _CountingSaver()
    graph = _build_counting_graph(saver)
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-counted-get"}}
    await graph.ainvoke({}, config)

    saver.aget_tuple_calls = 0
    snapshot = await accessor.aget(config)

    assert [message.content for message in snapshot.values["messages"]] == ["turn-0"]
    assert saver.aget_tuple_calls == 1
