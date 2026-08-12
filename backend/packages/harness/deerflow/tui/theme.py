"""TUI용 절제된 색상 + 기호 팔레트.

Tokyo Night 계열 팔레트다. 차분하고 어두운 터미널에서 읽기 좋으며, 화자와 tool 상태를 구분할
accent 색조가 몇 개 있다. Rich 호환 hex 색상이라 같은 상수로 Rich renderable과 Textual CSS
변수를 함께 굴린다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    bg: str = "#1a1b26"
    panel: str = "#1f2335"
    border: str = "#2f334d"
    text: str = "#c0caf5"
    dim: str = "#565f89"
    muted: str = "#737aa2"

    primary: str = "#7dcfff"  # 제목 / 앱 accent
    user: str = "#7aa2f7"  # 사용자 화자
    assistant: str = "#c0caf5"  # assistant 화자
    tool: str = "#bb9af7"  # tool 활동
    accent: str = "#9ece6a"  # 성공 / 정상
    warning: str = "#e0af68"  # 실행 중 / 주의
    error: str = "#f7768e"  # 오류


THEME = Theme()

SYMBOLS = {
    "user": "›",
    "assistant": "●",
    "tool": "⚙",
    "running": "◐",
    "ok": "✓",
    "error": "✗",
    "system": "·",
    "spinner": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
}
