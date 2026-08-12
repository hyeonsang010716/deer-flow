"""host→virtual 출력 마스킹 정규식을 한곳에서 구성한다.

boundary와 tail은 의도적으로 비공개다. ``build_output_mask_pattern``이 이 규칙을 표현하는 유일한
방법이므로, 세 번째 호출처가 조각을 import해 나머지 둘과 어긋나는 변형을 직접 만들 수 없다.

두 개의 독립적인 호출처가 모델로 흘러가는 텍스트에서 host 경로를 virtual 형태로 되돌린다:
``LocalSandbox._reverse_output_patterns``(bash 출력)와
``sandbox.tools._compiled_mask_patterns``(glob/grep/ls 결과). 둘은 같은 하위 계약을 따르므로
host base가 어디서 끝날 수 있는지에 대해 합의해야 한다 — 실제 segment 경계에 못 미쳐 끝나는
매치는 container 경로로 바뀌고, 이후 정방향 해석이 그것을 되돌려 매핑하기를 거부한다.

파일마다 그 규칙을 하나씩 갖고 있던 것이 drift의 원인이었다. #4035가 reverse 패턴에 segment
경계를 추가하면서 masking 패턴을 빠뜨렸고, #4053이 다른 사본에 같은 경계를 다시 넣어야 했다.
이 모듈이 규칙을 한 번만 담고 있어 세 번째 사본이 조용히 어긋날 수 없다.

두 호출처는 *동일하지 않으며*, 그 차이는 의도적이다 — ``separator_agnostic`` 참고.
"""

from __future__ import annotations

import re

# host base가 실제 path segment 경계에서 끝나는 경우에만 매치한다. 그래야 mount root가 접두사만
# 같은 형제 경로 안에서 매치되지 않는다(``.../skills-extra`` 안의 ``.../skills``).
#
# 이 문자 클래스는 shell이 아니라 텍스트 기준이다(``LocalSandbox._command_pattern``과 대비).
# 두 호출자 모두 임의의 명령 출력이나 파일 목록을 다루는데, 거기서는 root 뒤에 ``,`` ``:``
# ``\``가 정당하게 올 수 있으며 shell 기준 클래스라면 이를 거부한다.
#
# ``$``는 필수다. 없으면 정확히 mount root에서 끝나는 출력이 lookahead에 걸려 raw host 경로로
# 그대로 나간다.
_SEGMENT_BOUNDARY = r"(?=/|$|[^\w./-])"

# base 뒤에 붙는 경로 꼬리. ``[/\\]``는 Windows 구분자 경로도 매치되게 하고, 부정 문자 클래스는
# 공백과 shell 구두점에서 멈춰 더 긴 줄에 박힌 경로를 과하게 삼키지 않게 한다.
_PATH_TAIL = r"(?:[/\\][^\s\"';&|<>()]*)?"


def build_output_mask_pattern(base: str, *, separator_agnostic: bool = False) -> re.Pattern[str]:
    """모델에 보이는 출력에서 host ``base`` 하나를 매치할 matcher를 컴파일한다.

    Args:
        base: 매치할 host 경로 root(호출자가 이미 해석해 둔 값).
        separator_agnostic: base *내부*의 구분자를 둘 다 허용한다. ``\\``로 잡힌 base가 같은
            경로를 ``/``로 표기한 출력에도 매치되게 한다. ``sandbox.tools``는 base를
            ``_path_variants``(Windows 형식 표기를 낸다)에서 얻고 구분자를 통제할 수 없는 출력에
            매치하므로 이 옵션이 필요하다. ``LocalSandbox``는 필요 없다. base가
            ``Path.resolve()``에서 오므로 이미 실행 플랫폼의 구분자를 갖고 있고, 완화하면 마스킹
            범위만 넓어진다.

    Returns:
        segment 경계에서 ``base``와 선택적 경로 꼬리를 매치하는 컴파일된 패턴.
    """
    escaped = re.escape(base)
    if separator_agnostic:
        escaped = escaped.replace(r"\\", r"[/\\]")
    return re.compile(escaped + _SEGMENT_BOUNDARY + _PATH_TAIL)
