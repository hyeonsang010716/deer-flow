"""DeerFlow TUI의 command-line 진입점과 실행 모드 결정.

``plan_launch``는 순수 결정 함수다(단위 테스트로 완전히 덮여 있다). argv, TTY 상태, 환경을 받아
터미널 UI를 열지 headless 일회성 실행을 할지 결정한다. ``main``은 그 결정을 embedded
``DeerFlowClient``에 연결하고, 실제로 UI를 띄울 때만 Textual app을 지연 import한다. 덕분에
Textual이 없어도 ``deerflow`` 콘솔 스크립트로 headless command를 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_UNSET = object()

Mode = Literal["tui", "print", "json", "headless-help"]


@dataclass
class LaunchPlan:
    mode: Mode
    message: str | None = None
    read_stdin: bool = False
    thread_id: str | None = None
    continue_recent: bool = False
    forced_tui: bool = False
    transparent: bool = False
    recursion_limit: int | None = None
    reason: str = ""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deerflow",
        description="DeerFlow terminal workbench — a TUI over the embedded DeerFlow harness.",
        add_help=True,
    )
    parser.add_argument("message", nargs="*", help="initial prompt for the TUI, or message in --cli mode")
    parser.add_argument(
        "--print",
        dest="print",
        nargs="?",
        const=None,
        default=_UNSET,
        metavar="MESSAGE",
        help="headless one-shot: print the final answer and exit (reads stdin if no MESSAGE)",
    )
    parser.add_argument(
        "--json",
        dest="json",
        nargs="?",
        const=None,
        default=_UNSET,
        metavar="MESSAGE",
        help="headless streaming: emit newline-delimited JSON StreamEvents and exit",
    )
    parser.add_argument("--tui", action="store_true", help="force the terminal UI (error if unavailable)")
    parser.add_argument(
        "--tui-transparent",
        action="store_true",
        help="use the terminal's default background in the TUI",
    )
    parser.add_argument("--cli", action="store_true", help="force headless/classic mode for one invocation")
    parser.add_argument("--continue", dest="continue_recent", action="store_true", help="resume the most recent thread")
    parser.add_argument("--resume", dest="resume", metavar="THREAD", default=None, help="resume a thread by id or title")
    parser.add_argument(
        "--recursion-limit",
        type=_positive_int,
        metavar="N",
        help="headless agent-loop super-step limit (default: 100)",
    )
    return parser


def _strip_chat(argv: Sequence[str]) -> list[str]:
    """맨 앞의 선택적 ``chat`` 하위 command를 기본 화면의 별칭으로 받아들인다."""
    argv = list(argv)
    if argv and argv[0] == "chat":
        return argv[1:]
    return argv


def _truthy(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


def plan_launch(
    argv: Sequence[str],
    *,
    stdin_isatty: bool,
    stdout_isatty: bool,
    env: dict[str, str],
) -> LaunchPlan:
    """어떤 화면을 띄울지 결정한다. 순수 함수라서 I/O도 client 생성도 하지 않는다."""
    parser = build_parser()
    args = parser.parse_args(_strip_chat(argv))
    positional = " ".join(args.message).strip() or None
    resume = args.resume
    continue_recent = bool(args.continue_recent)
    headless_requested = args.print is not _UNSET or args.json is not _UNSET or args.cli
    if args.recursion_limit is not None and not headless_requested:
        parser.error("--recursion-limit requires --print, --json, or --cli")

    if args.print is not _UNSET:
        message = args.print if isinstance(args.print, str) else None
        if message is None and stdin_isatty:
            return LaunchPlan(mode="headless-help", reason="--print needs a MESSAGE argument or piped stdin.")
        return LaunchPlan(
            mode="print",
            message=message,
            read_stdin=message is None,
            thread_id=resume,
            continue_recent=continue_recent,
            recursion_limit=args.recursion_limit,
        )

    if args.json is not _UNSET:
        message = args.json if isinstance(args.json, str) else None
        if message is None and stdin_isatty:
            return LaunchPlan(mode="headless-help", reason="--json needs a MESSAGE argument or piped stdin.")
        return LaunchPlan(
            mode="json",
            message=message,
            read_stdin=message is None,
            thread_id=resume,
            continue_recent=continue_recent,
            recursion_limit=args.recursion_limit,
        )

    if args.cli:
        if positional:
            return LaunchPlan(
                mode="print",
                message=positional,
                thread_id=resume,
                continue_recent=continue_recent,
                recursion_limit=args.recursion_limit,
            )
        # --print와 동일하게 동작한다. 파이프로 들어온 메시지나 --continue만 있어도 headless로 돌린다.
        if continue_recent or not stdin_isatty:
            return LaunchPlan(
                mode="print",
                message=None,
                read_stdin=True,
                thread_id=resume,
                continue_recent=continue_recent,
                recursion_limit=args.recursion_limit,
            )
        return LaunchPlan(
            mode="headless-help",
            reason='--cli needs a message. Try: deerflow --print "your question".',
        )

    forced_tui = bool(args.tui)
    transparent = bool(args.tui_transparent) or _truthy(env.get("DEER_FLOW_TUI_TRANSPARENT"))
    if forced_tui or _truthy(env.get("DEER_FLOW_TUI")) or (stdin_isatty and stdout_isatty):
        return LaunchPlan(
            mode="tui",
            message=positional,
            thread_id=resume,
            continue_recent=continue_recent,
            forced_tui=forced_tui,
            transparent=transparent,
        )

    return LaunchPlan(
        mode="headless-help",
        message=positional,
        thread_id=resume,
        continue_recent=continue_recent,
        reason="No interactive terminal detected. Use --print MESSAGE for one-shot output, or --tui to force the UI.",
    )


# --------------------------------------------------------------------------- #
# runtime dispatch. 여기서는 단위 테스트하지 않고 smoke + 통합 테스트로 덮는다.
# --------------------------------------------------------------------------- #

_HEADLESS_HELP = """\
deerflow — DeerFlow terminal workbench

  deerflow                      launch the terminal UI (TTY required)
  deerflow --tui                force the terminal UI
  deerflow --tui-transparent    use the terminal's default background
  deerflow --continue           resume the most recent thread in the UI
  deerflow --resume THREAD      resume a thread by id or title
  deerflow --print "question"   one-shot answer to stdout
  deerflow --json "question"    stream newline-delimited JSON events
  deerflow --recursion-limit N --print "question"
                              set the headless agent-loop super-step limit
  echo "question" | deerflow --print
"""


def _resolve_message(plan: LaunchPlan) -> str:
    if plan.read_stdin:
        return sys.stdin.read().strip()
    return plan.message or ""


def _run_overrides(plan: LaunchPlan) -> dict[str, int]:
    if plan.recursion_limit is None:
        return {}
    return {"recursion_limit": plan.recursion_limit}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    plan = plan_launch(
        argv,
        stdin_isatty=sys.stdin.isatty(),
        stdout_isatty=sys.stdout.isatty(),
        env=dict(os.environ),
    )

    if plan.mode == "headless-help":
        if plan.reason:
            print(plan.reason, file=sys.stderr)
        print(_HEADLESS_HELP, file=sys.stderr)
        return 0 if not plan.reason else 2

    if plan.mode == "print":
        return _run_print(plan)

    if plan.mode == "json":
        return _run_json(plan)

    return _run_tui(plan)


def _make_session():
    # 순수 planning 경로가 무거운 harness를 import하지 않도록 지연 import한다. headless 일회성
    # 실행은 threads_meta writer를 쓰지 않으므로 persistence를 건너뛴다(버릴 background loop /
    # engine / connection pool을 굳이 세우지 않는다).
    from .session import open_session

    return open_session(persistence=False)


def _run_print(plan: LaunchPlan) -> int:
    message = _resolve_message(plan)
    if not message:
        print("No message provided.", file=sys.stderr)
        return 2
    session = _make_session()
    thread_id = session.resolve_thread(plan)
    answer = session.client.chat(message, thread_id=thread_id, **_run_overrides(plan))
    print(answer)
    return 0


def _run_json(plan: LaunchPlan) -> int:
    message = _resolve_message(plan)
    if not message:
        print("No message provided.", file=sys.stderr)
        return 2
    session = _make_session()
    thread_id = session.resolve_thread(plan)
    for event in session.client.stream(message, thread_id=thread_id, **_run_overrides(plan)):
        payload = {"type": event.type, "data": event.data}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    return 0


def _run_tui(plan: LaunchPlan) -> int:
    try:
        # `from .app`이 아니라 절대 import를 쓴다. harness import 경계 검사는 상대 module
        # 이름을 그대로 기록하므로, 형제 module인 `deerflow.tui.app`을 금지된 최상위 `app`
        # 패키지로 오인하지 않게 하기 위함이다.
        from deerflow.tui.app import run_tui
    except ModuleNotFoundError as exc:  # textual이 없는 경우
        if getattr(exc, "name", "") == "textual" or "textual" in str(exc):
            msg = "The terminal UI needs the optional 'textual' dependency.\nInstall it with:  uv pip install 'deerflow-harness[tui]'   (or: pip install textual)\n"
            if plan.forced_tui:
                print(msg, file=sys.stderr)
                return 1
            print(msg + "\nFalling back to headless help:\n", file=sys.stderr)
            print(_HEADLESS_HELP, file=sys.stderr)
            return 0
        raise
    return run_tui(plan)


if __name__ == "__main__":
    raise SystemExit(main())
