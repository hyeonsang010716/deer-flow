"""tool 출력 budget 보호 설정."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator

from deerflow.constants import TOOL_RESULTS_DIRNAME


class ToolOutputConfig(BaseModel):
    """tool 결과 출력 budget 강제를 위한 config 섹션.

    tool이 ``externalize_min_chars``보다 긴 출력을 반환하면 전체 출력을 디스크에 저장하고 간결한
    preview + 파일 참조로 대체한다. 디스크 저장이 불가능하면 앞뒤를 남기는 truncation으로
    대체한다.
    """

    enabled: bool = Field(
        default=True,
        description="Enable the tool output budget middleware.",
    )
    externalize_min_chars: int = Field(
        default=12_000,
        ge=0,
        description="Character threshold to trigger disk externalization. Outputs below this pass through unchanged. Set to 0 to disable externalization (fallback truncation still applies when output exceeds fallback_max_chars).",
    )
    preview_head_chars: int = Field(
        default=2_000,
        ge=0,
        description="Sampling budget retained for compatibility. Typed previews use this with preview_tail_chars only for fallback samples inside the structured synopsis.",
    )
    preview_tail_chars: int = Field(
        default=1_000,
        ge=0,
        description="Sampling budget retained for compatibility. Typed previews use this with preview_head_chars only for fallback samples inside the structured synopsis.",
    )
    fallback_max_chars: int = Field(
        default=30_000,
        ge=0,
        description="Maximum characters when disk persistence is unavailable. 0 disables fallback truncation.",
    )
    fallback_head_chars: int = Field(
        default=8_000,
        ge=0,
        description="Head characters for fallback truncation.",
    )
    fallback_tail_chars: int = Field(
        default=3_000,
        ge=0,
        description="Tail characters for fallback truncation.",
    )
    storage_subdir: str = Field(
        default=TOOL_RESULTS_DIRNAME,
        description=(
            "Single-segment directory name under the thread outputs path for persisted tool results. "
            "TOOL_RESULTS_DIRNAME is always excluded by the workspace-changes scanner; other custom values are "
            "excluded from workspace snapshots and run delivery verification at capture time."
        ),
    )

    @field_validator("storage_subdir")
    @classmethod
    def _storage_subdir_is_single_segment(cls, value: str) -> str:
        """디렉터리 이름 한 조각만 허용한다(경로 구분자 불가).

        workspace-changes 스캐너는 ``os.walk`` 중 디렉터리 이름으로 가지치기하며, 그때 얻는
        dirname은 한 조각이다. ``cache/tool-results`` 같은 중첩 값은 제외 조건에 절대 걸리지
        않아 그 파일들이 다시 조용히 산출 artifact로 집계된다. 조용한 무효 제외보다 요란한
        config 오류가 낫다.
        """
        if value == "" or value in {".", ".."} or os.path.isabs(value):
            raise ValueError("storage_subdir must be a single non-empty directory name")
        if "/" in value or "\\" in value:
            raise ValueError(f"storage_subdir must be a single directory name without path separators (got {value!r})")
        return value

    exempt_tools: list[str] = Field(
        default_factory=lambda: ["read_file", "read_file_tool"],
        description="Tool names exempt from budget enforcement (prevents persist→read→persist loops).",
    )
    tool_overrides: dict[str, int] = Field(
        default_factory=dict,
        description="Per-tool externalize_min_chars overrides. Keys are tool names, values are char thresholds. Use 0 to disable externalization for a specific tool.",
    )
