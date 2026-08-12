"""SandboxAuditMiddleware - bash 명령 보안 감사."""

import json
import logging
import re
import shlex
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 명령 분류 규칙
# ---------------------------------------------------------------------------

# 출력을 *실행*하면 위험한 실행 파일들. 아래 command substitution 규칙에서 쓴다.
# ``\b``는 이 단어들로 시작하기만 하는 무관한 이름(``shellcheck``, ``shasum``,
# ``pythonic-tool``)이 매칭되는 것을 막는다.
_RISKY_SUBSTITUTION_EXECUTABLES = r"(?:curl|wget|bash|sh|python[\d.]*|ruby|perl|base64)\b"

# 위 실행 파일을 여는 substitution의 모든 표기: ``$(cmd``, ``<(cmd``, 그리고 괄호가
# 없는 backtick 형태. 하나의 opener를 공유하기 때문에 ``eval $(curl u)``만 겨냥한
# 규칙을 ``eval `curl u` ``가 빠져나가지 못한다.
_RISKY_SUBSTITUTION = rf"(?:[$<]\(\s*|`\s*){_RISKY_SUBSTITUTION_EXECUTABLES}"

# 인자로 받은 *코드 문자열*을 실행하는 interpreter들과 그것을 받는 flag:
# ``-c``(shell, python), ``-e``(perl/ruby/node), ``-p``(perl/node print loop),
# ``-r``(php). flag가 받은 내용은 그대로 실행되므로 거기 있는 위험한 substitution도
# 실행된다 — ``eval``/``source``와 같은 부류를 flag로 표기한 것일 뿐이다.
# here-string(``<<<``)은 stdin을 통해 같은 지점에 도달한다.
#
# 이 규칙들은 의도적으로 위치를 가리지 않는다: ``bash -c``는 어디에 나타나든 실행
# context이며, 다른 명령의 인자로 들어간 경우(``xargs sh -c "$(curl url)"``)도 포함한다.
# 선행 flag 반복은 긴 입력에서 alternation이 backtracking하지 않도록 상한을 둔다.
_CODE_STRING_INTERPRETERS = r"(?:(?:ba|da|k|z)?sh|python[\d.]*|perl|ruby|node|php)"
_LEADING_FLAGS = r"(?:-\w+\s+){0,4}"

# 각 패턴은 import 시점에 한 번 컴파일된다.
_HIGH_RISK_PATTERNS: list[re.Pattern[str]] = [
    # --- 기존 규칙(유지) ---
    re.compile(r"rm\s+-[^\s]*r[^\s]*\s+(/\*?|~/?\*?|/home\b|/root\b)\s*$"),
    re.compile(r"dd\s+if="),
    re.compile(r"mkfs"),
    re.compile(r"cat\s+/etc/shadow"),
    re.compile(r">+\s*/etc/"),
    # --- sh/bash로의 pipe (일반화, 기존 curl|sh 규칙 대체) ---
    re.compile(r"\|\s*(ba)?sh\b"),
    # --- eval/source는 위치와 무관하게 substitution을 실행한다 ---
    re.compile(rf"\b(eval|source)\s+[\"']?{_RISKY_SUBSTITUTION}"),
    # --- interpreter의 code-string flag도 실행 context다 ---
    re.compile(rf"\b{_CODE_STRING_INTERPRETERS}\s+{_LEADING_FLAGS}-[cepr]\s+[\"']?{_RISKY_SUBSTITUTION}"),
    re.compile(rf"\b{_CODE_STRING_INTERPRETERS}\s+{_LEADING_FLAGS}<<<\s*[\"']?{_RISKY_SUBSTITUTION}"),
    # --- base64 디코드를 실행으로 pipe ---
    re.compile(r"base64\s+.*-d.*\|"),
    # --- 시스템 바이너리 덮어쓰기 ---
    re.compile(r">+\s*(/usr/bin/|/bin/|/sbin/)"),
    # --- shell 시작 파일 덮어쓰기 ---
    re.compile(r">+\s*~/?\.(bashrc|profile|zshrc|bash_profile)"),
    # --- 프로세스 환경 변수 유출 ---
    re.compile(r"/proc/[^/]+/environ"),
    # --- dynamic linker hijack (한 단계 권한 상승) ---
    re.compile(r"\b(LD_PRELOAD|LD_LIBRARY_PATH)\s*="),
    # --- bash 내장 네트워킹 (tool allowlist 우회) ---
    re.compile(r"/dev/tcp/"),
    # --- fork bomb ---
    re.compile(r"\S+\(\)\s*\{[^}]*\|\s*\S+\s*&"),  # :(){ :|:& };:
    re.compile(r"while\s+true.*&\s*done"),  # while true; do bash & done
]

# *command position*의 command substitution: substitution 결과가 실행될 명령이 되므로,
# 받아온 내용이나 해석된 내용이 그대로 실행된다.
#
# 이 규칙들은 전체 복합 문자열이 아니라 개별 sub-command에 anchor를 걸어 매칭한다.
# 두 형태를 구분하는 것이 바로 위치이기 때문이다.
#
#   $(curl url)          → 다운로드한 것을 실행한다        → block
#   x=$(curl url)        → 출력을 변수에 담는다            → pass
#   echo $(curl url)     → 출력을 인자로 넘긴다            → pass
#
# 이전의 anchor 없는 규칙은 둘을 구분하지 못해 일상적인 출력 캡처까지 거부했다
# (issue #4611).
#
# command position이 항상 첫 글자인 것은 아니다. POSIX shell은 선행 변수 할당을
# 허용하고, exec wrapper는 뒤따르는 것을 command position에 남긴다
# (``FOO=1 $(curl url)``, ``env FOO=1 $(curl url)``, ``nohup $(curl url)``).
# 할당 분기는 할당과 substitution 사이에 공백을 요구하므로 ``x=$(curl url)``에는
# 매칭될 수 없고, 따라서 value position은 계속 허용된다. 긴 입력에서 alternation이
# backtracking하지 않도록 반복에는 상한을 둔다.
_COMMAND_POSITION_PREFIX = r"(?:(?:env|command|builtin|exec|nohup|time|sudo|doas)\s+|\w+=\S*\s+){0,8}"

_HIGH_RISK_COMMAND_POSITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"^{_COMMAND_POSITION_PREFIX}[\"']?\$\(\s*{_RISKY_SUBSTITUTION_EXECUTABLES}"),
    re.compile(rf"^{_COMMAND_POSITION_PREFIX}[\"']?`\s*{_RISKY_SUBSTITUTION_EXECUTABLES}"),
]

_MEDIUM_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"chmod\s+777"),
    re.compile(r"pip3?\s+install"),
    re.compile(r"apt(-get)?\s+install"),
    # sudo/su: Docker root 환경에서는 no-op이지만, LLM이 인지하도록 warn한다
    re.compile(r"\b(sudo|su)\b"),
    # PATH 변경: 공격 사슬이 길므로 block 대신 warn
    re.compile(r"\bPATH\s*="),
]


# heredoc 헤더와 그 delimiter: ``<<EOF``, ``<< EOF``, ``<<-EOF``, ``<<\EOF``,
# ``<<'EOF'``, ``<<"EOF"``. ``<<<``(본문이 없는 here-string)가 heredoc을 열지 못하게
# 하려면 두 guard가 모두 필요하다. lookahead가 첫 ``<``에서 거부하고, lookbehind는
# 뒤쪽 ``<<``가 한 글자 뒤에서 매칭되는 것을 막는다. 그렇지 않으면 ``<<< "text"``가
# delimiter ``text``인 heredoc으로 읽힌다.
_HEREDOC_HEADER = re.compile(r"(?<!<)<<(?!<)-?[ \t]*(?:\\?([A-Za-z_][\w.-]*)|'([^'\n]*)'|\"([^\"\n]*)\")")


def _consume_heredoc_bodies(command: str, pos: int, delimiters: list[str]) -> int:
    """지금까지 열린 *delimiters*의 본문 바로 다음 인덱스를 반환한다.

    본문은 헤더가 나타난 순서대로 소비되며, 각 본문은 strip한 내용이 자기 delimiter와
    같은 줄까지 이어진다(``<<-``는 선행 탭을 제거하는데 ``strip()``이 이를 포함한다).
    종료되지 않은 본문은 문자열의 나머지를 전부 소비한다. 헤더 뒤는 실제로 전부 본문이며
    찾을 수 있는 이후 statement가 없기 때문이다.
    """
    for delimiter in delimiters:
        while pos < len(command):
            newline = command.find("\n", pos)
            if newline == -1:
                return len(command)
            line = command[pos:newline]
            pos = newline + 1
            if line.strip() == delimiter:
                break
        else:
            return len(command)
    return pos


def _split_compound_command(command: str, *, split_pipes: bool = False) -> list[str]:
    """복합 명령을 sub-command로 분리한다(따옴표 인식).

    원본 명령 문자열을 스캔하므로 따옴표 밖의 shell 제어 연산자가 공백으로 둘러싸이지
    않아도 인식된다(예: ``safe;rm -rf /``, ``rm -rf /&&echo ok``). 따옴표 안의 연산자는
    무시한다. 명령이 닫히지 않은 따옴표나 매달린 escape로 끝나면 명령 전체를 그대로
    반환한다(fail-closed — 일부를 조용히 버리는 것보다 분리하지 않은 문자열을 분류하는
    편이 안전하다).

    순차 연산자(``&&``, ``||``, ``;``)가 분리 기준이며, 따옴표 밖의 개행도 마찬가지다.
    개행은 ``;``와 정확히 같은 방식으로 statement를 나누므로, 붙여 두면 shell 의미가
    동일한데도 ``echo hi; $(curl url)``이 걸리는 anchor된 command-position 규칙을
    ``echo hi\\n$(curl url)``이 빠져나갔다.

    heredoc 본문은 statement가 아니라 데이터다. 그 안의 개행과 연산자는 파일 내용이다.
    따라서 헤더(``<<EOF``, ``<<-EOF``, ``<<'EOF'``)는 읽는 대로 기록하고, 본문은 헤더를
    끝내는 개행 지점에서 그대로 소비한다. 덕분에 ``$(curl url)``로 시작하는 본문 줄이
    command position으로 승격되지 않는다. ``<<<``는 heredoc이 아니라 here-string이므로
    heredoc을 열지 않고, ``$(( ... ))``나 ``(( ... ))`` 안의 ``<<``도 마찬가지다. 그것은
    bit shift이며, 그렇지 않으면 오른쪽 피연산자가 끝내 나타나지 않는 delimiter로 읽혀
    명령의 나머지를 통째로 삼킨다. 이것은 shell 파싱이 아니라 휴리스틱이다. 목표는 shell이
    결코 만들지 않을 command position을 만들어내지 않는 것과, 진짜 command position을
    없애지 않는 것뿐이다.

    pipeline은 논리적으로 하나의 명령이므로 pipe는 기본적으로 분리하지 않는다.
    ``split_pipes=True``를 주면 ``|``에서도 분리하는데, command-position 탐지에 필요하다 —
    pipe 뒤의 단어는 새 명령을 시작하기 때문이다. pipe를 가로지르는 규칙(``| sh``,
    ``base64 -d | ...``)은 :func:`_classify_command`의 전체 명령 스캔에서 매칭되므로 추가
    분리의 영향을 받지 않는다.
    """
    parts: list[str] = []
    current: list[str] = []
    pending_heredocs: list[str] = []
    in_single_quote = False
    in_double_quote = False
    arithmetic_depth = 0
    escaping = False
    index = 0

    while index < len(command):
        char = command[index]

        if escaping:
            current.append(char)
            escaping = False
            index += 1
            continue

        if char == "\\" and not in_single_quote:
            current.append(char)
            escaping = True
            index += 1
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
            index += 1
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
            index += 1
            continue

        if not in_single_quote and not in_double_quote:
            # 산술식 안의 ``<<``는 redirection이 아니라 bit shift이며, delimiter가 끝내
            # 나타나지 않는 유령 헤더는 명령의 나머지를 삼켜 버린다. ``$(( ... ))``와
            # 순수 산술 명령 ``(( ... ))``를 모두 추적한다. 닫히지 않은 ``((``는 depth를
            # 양수로 남기지만 그것은 heredoc 탐지만 끄고 개행 분리는 계속되므로, 실패
            # 방향은 command position을 덜 보는 쪽이 아니라 더 보는 쪽으로 유지된다.
            if char == "(" and command.startswith("((", index):
                arithmetic_depth += 1
                current.append("((")
                index += 2
                continue
            if arithmetic_depth and char == ")" and command.startswith("))", index):
                arithmetic_depth -= 1
                current.append("))")
                index += 2
                continue
            # 헤더는 ``<``에서만 시작할 수 있다. 이를 먼저 검사하면 긴 명령의 나머지
            # 모든 문자에 regex를 돌리지 않아도 된다.
            if char == "<" and not arithmetic_depth:
                heredoc = _HEREDOC_HEADER.match(command, index)
                if heredoc:
                    pending_heredocs.append(next(group for group in heredoc.groups() if group is not None))
                    current.append(heredoc.group(0))
                    index = heredoc.end()
                    continue
            if char == "\n":
                # heredoc 헤더 뒤의 개행이 statement 구분자이며, 그 본문은 닫히는
                # statement에 속한다.
                if pending_heredocs:
                    body_end = _consume_heredoc_bodies(command, index + 1, pending_heredocs)
                    pending_heredocs = []
                    current.append(command[index:body_end])
                    index = body_end
                else:
                    index += 1
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            if command.startswith("&&", index) or command.startswith("||", index):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += 2
                continue
            # "||" 뒤에 검사해 단일 "|"가 그 연산자를 가로채지 못하게 한다.
            if split_pipes and char == "|":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += 1
                continue
            if char == ";":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += 1
                continue

        current.append(char)
        index += 1

    # 닫히지 않은 따옴표나 매달린 escape → fail-closed, 명령 전체를 반환
    if in_single_quote or in_double_quote or escaping:
        return [command]

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts if parts else [command]


def _matches_high_risk(candidate: str) -> bool:
    """*candidate*(sub-command 하나)가 high-risk 규칙 중 하나에 매칭되면 True를 반환한다."""
    if any(pattern.search(candidate) for pattern in _HIGH_RISK_PATTERNS):
        return True
    # anchor된 규칙: 복합 문자열이 아니라 단일 sub-command에 대해서만 의미가 있다.
    return any(pattern.match(candidate) for pattern in _HIGH_RISK_COMMAND_POSITION_PATTERNS)


def _classify_single_command(command: str) -> str:
    """단일(복합이 아닌) 명령을 분류한다. 'block', 'warn', 'pass' 중 하나를 반환한다."""
    normalized = " ".join(command.split())

    if _matches_high_risk(normalized):
        return "block"

    # high-risk 탐지를 위해 shlex로 파싱한 토큰도 시도한다
    try:
        tokens = shlex.split(command)
        joined = " ".join(tokens)
        if _matches_high_risk(joined):
            return "block"
    except ValueError:
        # heredoc이나 다른 여러 줄 shell 형태는 유효한 bash지만 shlex로는 파싱되지 않을
        # 수 있다. 원본에 대한 high-risk 패턴은 이미 검사했다.
        pass

    for pattern in _MEDIUM_RISK_PATTERNS:
        if pattern.search(normalized):
            return "warn"

    return "pass"


def _classify_command(command: str) -> str:
    """'block', 'warn', 'pass' 중 하나를 반환한다.

    전략:
    1. 먼저 원본 명령 *전체*를 high-risk 패턴으로 스캔한다. ``while true; do bash & done``
       이나 ``:(){ :|:& };:``처럼 여러 shell statement에 걸친 구조적 공격을 잡기 위해서다 —
       ``;``로 분리하면 패턴의 context가 파괴된다.
    2. 그다음 복합 명령(예: ``cmd1 && cmd2 ; cmd3``)을 분리해 각 sub-command를 독립적으로
       분류한다. 가장 심각한 판정이 이긴다.
    """
    # Pass 1: 명령 전체 high-risk 스캔 (여러 statement에 걸친 패턴을 잡는다)
    normalized = " ".join(command.split())
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(normalized):
            return "block"

    # Pass 2: sub-command 단위 분류. pipe 뒤의 단어가 새 command position을 시작하므로
    # (``echo hi | $(curl ...)``) 여기서는 pipe도 분리한다.
    sub_commands = _split_compound_command(command, split_pipes=True)
    worst = "pass"
    for sub in sub_commands:
        verdict = _classify_single_command(sub)
        if verdict == "block":
            return "block"  # short-circuit: 더 나빠질 수 없다
        if verdict == "warn":
            worst = "warn"
    return worst


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class SandboxAuditMiddleware(AgentMiddleware[ThreadState]):
    """bash 명령 보안 감사 middleware.

    모든 ``bash`` tool call에 대해:
    1. **명령 분류**: regex + shlex 분석으로 명령을 high-risk(block), medium-risk(warn),
       safe(pass)로 등급을 매긴다.
    2. **감사 로그**: 모든 bash 호출을 표준 logger로 구조화된 JSON 항목으로 기록한다
       (gateway.log에서 확인 가능).

    high-risk 명령(예: ``rm -rf /``, ``curl url | bash``)은 차단된다. handler를 호출하지
    않고 error ``ToolMessage``를 반환해 agent loop가 무리 없이 이어지게 한다.

    medium-risk 명령(예: ``pip install``, ``chmod 777``)은 정상 실행되며, LLM이 인지하도록
    tool 결과에 경고를 덧붙인다.
    """

    state_schema = ThreadState

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _get_thread_id(self, request: ToolCallRequest) -> str | None:
        runtime = request.runtime  # ToolRuntime. 테스트에서는 None에 가까울 수 있다
        if runtime is None:
            return None
        ctx = getattr(runtime, "context", None) or {}
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        if thread_id is None:
            cfg = getattr(runtime, "config", None) or {}
            thread_id = cfg.get("configurable", {}).get("thread_id")
        return thread_id

    _AUDIT_COMMAND_LIMIT = 200

    def _write_audit(self, thread_id: str | None, command: str, verdict: str, *, truncate: bool = False) -> None:
        audited_command = command
        if truncate and len(command) > self._AUDIT_COMMAND_LIMIT:
            audited_command = f"{command[: self._AUDIT_COMMAND_LIMIT]}... ({len(command)} chars)"
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "thread_id": thread_id or "unknown",
            "command": audited_command,
            "verdict": verdict,
        }
        logger.info("[SandboxAudit] %s", json.dumps(record, ensure_ascii=False))

    def _build_block_message(self, request: ToolCallRequest, reason: str) -> ToolMessage:
        tool_call_id = str(request.tool_call.get("id") or "missing_id")
        return ToolMessage(
            content=f"Command blocked: {reason}. Please use a safer alternative approach.",
            tool_call_id=tool_call_id,
            name="bash",
            status="error",
        )

    def _append_warn_to_result(self, result: ToolMessage | Command, command: str) -> ToolMessage | Command:
        """medium-risk 명령에 대해 tool 결과에 경고 문구를 덧붙인다."""
        if not isinstance(result, ToolMessage):
            return result
        warning = f"\n\n⚠️ Warning: `{command}` is a medium-risk command that may modify the runtime environment."
        if isinstance(result.content, list):
            new_content = list(result.content) + [{"type": "text", "text": warning}]
        else:
            new_content = str(result.content) + warning
        return ToolMessage(
            content=new_content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
        )

    # ------------------------------------------------------------------
    # 입력 sanitisation
    # ------------------------------------------------------------------

    # 정상적인 bash 명령이 수백 자를 넘는 경우는 드물다. 10 000은 정당한 사용 사례를
    # 훨씬 웃돌면서도 Linux ARG_MAX에 비하면 아주 작은 값이다. 그보다 긴 것은 거의
    # 확실히 payload injection이거나 base64로 인코딩된 공격 문자열이다.
    _MAX_COMMAND_LENGTH = 10_000

    def _validate_input(self, command: str) -> str | None:
        """*command*가 허용 가능하면 ``None``을, 아니면 거부 사유를 반환한다."""
        if not command.strip():
            return "empty command"
        if len(command) > self._MAX_COMMAND_LENGTH:
            return "command too long"
        if "\x00" in command:
            return "null byte detected"
        return None

    # ------------------------------------------------------------------
    # 핵심 로직 (sync/async 경로 공용)
    # ------------------------------------------------------------------

    def _pre_process(self, request: ToolCallRequest) -> tuple[str, str | None, str, str | None]:
        """
        (command, thread_id, verdict, reject_reason)를 반환한다.
        verdict는 'block', 'warn', 'pass' 중 하나다.
        reject_reason은 입력 sanitisation 거부일 때만 None이 아니다.
        """
        args = request.tool_call.get("args", {})
        raw_command = args.get("command")
        command = raw_command if isinstance(raw_command, str) else ""
        thread_id = self._get_thread_id(request)

        # ① 입력 sanitisation — regex 분석 전에 잘못된 입력을 거부한다
        reject_reason = self._validate_input(command)
        if reject_reason:
            self._write_audit(thread_id, command, "block", truncate=True)
            logger.warning("[SandboxAudit] INVALID INPUT thread=%s reason=%s", thread_id, reject_reason)
            return command, thread_id, "block", reject_reason

        # ② 명령 분류
        verdict = _classify_command(command)

        # ③ 감사 로그
        self._write_audit(thread_id, command, verdict)

        if verdict == "block":
            logger.warning("[SandboxAudit] BLOCKED thread=%s cmd=%r", thread_id, command)
        elif verdict == "warn":
            logger.warning("[SandboxAudit] WARN (medium-risk) thread=%s cmd=%r", thread_id, command)

        return command, thread_id, verdict, None

    # ------------------------------------------------------------------
    # wrap_tool_call hooks
    # ------------------------------------------------------------------

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "bash":
            return handler(request)

        command, _, verdict, reject_reason = self._pre_process(request)
        if verdict == "block":
            reason = reject_reason or "security violation detected"
            return self._build_block_message(request, reason)
        result = handler(request)
        if verdict == "warn":
            result = self._append_warn_to_result(result, command)
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "bash":
            return await handler(request)

        command, _, verdict, reject_reason = self._pre_process(request)
        if verdict == "block":
            reason = reject_reason or "security violation detected"
            return self._build_block_message(request, reason)
        result = await handler(request)
        if verdict == "warn":
            result = self._append_warn_to_result(result, command)
        return result
