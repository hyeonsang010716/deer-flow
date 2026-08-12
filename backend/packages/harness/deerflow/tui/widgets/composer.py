"""전각(CJK) 문자 커서 문제를 고친 Composer 입력 위젯.

Textual의 ``Input._cursor_offset``은 커서가 값의 끝에 있을 때 무조건 ``+1``을 더한다. 전각(CJK)
문자 뒤에서는 이것이 한 셀만큼 넘쳐 *하드웨어 / IME* 커서 위치를 어긋나게 하며, iTerm2 같은
터미널에서 중국어를 입력할 때 눈에 띄는 밀림으로 나타난다. (화면의 블록 커서는 ``render_line``에서
문자 인덱스 스타일링으로 따로 그려지므로 영향이 없다. 여기서 고치는 것은 IME 후보 창이 따라가는
터미널 커서 기준점뿐이다.)

영어 입력은 IME를 거치지 않아 이 경로를 타지 않으며, 그래서 밀림이 CJK에서만 보인다.
"""

from __future__ import annotations

from textual.widgets import Input


class ComposerInput(Input):
    @property
    def _cursor_offset(self) -> int:
        # Textual의 값 끝 +1 없이 계산한 커서의 실제 셀 offset.
        return self._position_to_cell(self.cursor_position)
