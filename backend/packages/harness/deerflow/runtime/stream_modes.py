"""LangGraph 호환 runtime 경계가 지원하는 stream mode."""

from __future__ import annotations

from typing import Literal, get_args

type RunStreamMode = Literal[
    "values",
    "messages-tuple",
    "updates",
    "debug",
    "tasks",
    "checkpoints",
    "custom",
]

SUPPORTED_RUN_STREAM_MODES: frozenset[str] = frozenset(get_args(RunStreamMode.__value__))


class UnsupportedStreamModeError(ValueError):
    """DeerFlow가 지원하지 않는 stream mode를 호출자가 요청하면 발생한다."""

    def __init__(self, modes: list[str]) -> None:
        self.modes = tuple(dict.fromkeys(modes))
        super().__init__(f"Unsupported stream mode(s): {', '.join(self.modes)}")


def normalize_stream_modes(raw: list[str] | str | None) -> list[str]:
    """공개 run stream mode를 정규화하고 검증한다."""
    if raw is None:
        modes = ["values"]
    elif isinstance(raw, str):
        modes = [raw]
    else:
        modes = raw or ["values"]

    unsupported = [mode if isinstance(mode, str) else type(mode).__name__ for mode in modes if not isinstance(mode, str) or mode not in SUPPORTED_RUN_STREAM_MODES]
    if unsupported:
        raise UnsupportedStreamModeError(unsupported)
    return modes


def to_langgraph_stream_modes(raw: list[str] | str | None) -> list[str]:
    """공개 run mode를 ``graph.astream`` mode로 매핑한다. 조용한 fallback은 하지 않는다."""
    modes = normalize_stream_modes(raw)
    mapped = ["messages" if mode == "messages-tuple" else mode for mode in modes]
    return list(dict.fromkeys(mapped))
