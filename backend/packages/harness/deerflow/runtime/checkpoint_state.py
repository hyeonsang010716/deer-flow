"""materialize된 checkpoint state 접근과 state 전용 mutation graph.

:class:`CheckpointStateAccessor`는 thread checkpoint state 읽기/쓰기의 단일 choke point다.
compiled graph(channel schema를 담고 있다)와 checkpointer를 함께 묶어두므로, state를 만지는
모든 경로가 같은 channel table을 거친다.

:func:`build_state_mutation_graph`는 rollback 복원이나 context compaction처럼 state를
통째로 교체할 때 쓰는 state 전용 graph(no-op node 하나, entry = finish)를 컴파일한다.
agent graph의 checkpoint machinery를 공유하지만 pending node를 스케줄하지 않으므로,
기록된 head는 idle 상태로 남는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deerflow.agents.thread_state import ThreadState


def _finish_state_mutation(_state: dict[str, Any]) -> dict[str, Any]:
    return {}


def build_state_mutation_graph(as_node: str, state_schema: Any | None = None) -> Any:
    """writer node 하나가 즉시 종료하는 state 전용 graph를 컴파일한다.

    ``update_state(..., as_node=...)``는 해당 node가 graph에 등록되어 있어야 한다. 전용
    single-node graph는 reducer write를 적용하고 바로 끝나므로, mutation checkpoint는 agent
    node를 스케줄하지 않고 pending ``next`` node도 남기지 않는다.

    write가 materialize된 state를 담는 경우 ``state_schema``는 반드시 thread의 *실제* schema
    (assistant graph가 컴파일될 때 쓴 클래스)여야 한다. 기본 ThreadState fallback은 custom
    middleware가 기여한 channel을 모르고, 알 수 없는 channel에 대한 write는 조용히 버려진다.
    """
    if not as_node:
        raise ValueError("as_node is required for checkpoint state mutation")
    from langgraph.graph import StateGraph

    builder = StateGraph(state_schema if state_schema is not None else ThreadState)
    builder.add_node(as_node, _finish_state_mutation)
    builder.set_entry_point(as_node)
    builder.set_finish_point(as_node)
    return builder.compile()


def graph_state_schema(graph: Any) -> Any | None:
    """compiled graph가 어떤 state schema 클래스로 만들어졌는지 반환한다.

    schema는 ``StateGraph.schemas``의 첫 항목이다(state schema가 input/output schema보다 먼저
    등록된다). 실제 compiled graph를 감싸지 않는 테스트용 stub accessor에는 ``None``을 반환한다.
    """
    schemas = getattr(getattr(graph, "builder", None), "schemas", None)
    if not schemas:
        return None
    return next(iter(schemas))


def graph_writable_channels(graph: Any) -> frozenset[str] | None:
    """compiled graph에서 사용자에게 보이는 state channel 이름들을 반환한다.

    Pregel 내부 channel(``__*``)과 branch fan-in channel(``branch:*``)은 제외한다. graph가
    channel을 노출하지 않으면(stub accessor) ``None``을 반환해, 호출자가 기본 ThreadState
    집합으로 fallback할 수 있게 한다.
    """
    channels = getattr(graph, "channels", None)
    if not channels:
        return None
    return frozenset(name for name in channels if not name.startswith("__") and not name.startswith("branch:"))


def graph_reducer_channels(graph: Any) -> frozenset[str] | None:
    """write가 reducer를 거쳐 병합되는 channel 이름들을 반환한다.

    ``BinaryOperatorAggregate`` channel이 여기 해당하며, replace 방식 write에는 ``Overwrite``
    래핑이 필요하다. graph가 channel을 노출하지 않으면(stub accessor) ``None``을 반환해,
    호출자가 기본 ThreadState 집합으로 fallback할 수 있게 한다.
    """
    from langgraph.channels import BinaryOperatorAggregate

    channels = getattr(graph, "channels", None)
    if channels is None:
        return None
    return frozenset(name for name, channel in channels.items() if isinstance(channel, BinaryOperatorAggregate))


@dataclass
class CheckpointStateAccessor:
    graph: Any
    checkpointer: Any

    @classmethod
    def bind(
        cls,
        graph: Any,
        checkpointer: Any,
        *,
        store: Any | None = None,
    ) -> CheckpointStateAccessor:
        graph.checkpointer = checkpointer
        if store is not None:
            graph.store = store
        return cls(graph=graph, checkpointer=checkpointer)

    def _prepare_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **config,
            "configurable": dict(config.get("configurable", {})),
            "metadata": dict(config.get("metadata", {})),
        }

    def get(self, config: dict[str, Any]) -> Any:
        prepared = self._prepare_config(config)
        return self.graph.get_state(prepared)

    async def aget(self, config: dict[str, Any]) -> Any:
        prepared = self._prepare_config(config)
        return await self.graph.aget_state(prepared)

    def history(self, config: dict[str, Any], *, limit: int | None = None) -> list[Any]:
        prepared = self._prepare_config(config)
        if limit is not None and limit <= 0:
            return []
        result = []
        for snapshot in self.graph.get_state_history(prepared, limit=limit):
            result.append(snapshot)
            if limit is not None and len(result) >= limit:
                break
        return result

    async def ahistory(self, config: dict[str, Any], *, limit: int | None = None) -> list[Any]:
        prepared = self._prepare_config(config)
        if limit is not None and limit <= 0:
            return []
        result = []
        async for snapshot in self.graph.aget_state_history(prepared, limit=limit):
            result.append(snapshot)
            if limit is not None and len(result) >= limit:
                break
        return result

    def update(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_config(config)
        return self.graph.update_state(prepared, values, as_node=as_node)

    async def aupdate(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_config(config)
        return await self.graph.aupdate_state(prepared, values, as_node=as_node)
