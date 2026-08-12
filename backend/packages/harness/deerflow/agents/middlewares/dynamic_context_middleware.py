"""동적 context(memory, 현재 날짜)를 system-reminder로 주입하는 middleware.

system prompt는 사용자와 session 전반에서 prefix-cache 재사용을 극대화하기 위해 완전히
정적으로 유지한다. 현재 날짜는 항상 주입되고, app config의 ``memory.injection_enabled``가
True이면 사용자별 memory도 함께 주입된다. 둘 다 첫 user 메시지 앞에 삽입되는 전용
<system-reminder> SystemMessage로 대화당 한 번만 전달된다(frozen-snapshot 패턴).

대화가 자정을 넘기면 middleware가 날짜 변경을 감지해 현재 turn 앞에 별도 SystemMessage로
가벼운 날짜 갱신 reminder를 주입한다. 이 보정은 영속화되므로 새 날짜의 이후 turn들은 일관된
history를 보고 다시 주입하지 않는다.

Reminder 형식:

    <system-reminder>
    <memory>...</memory>

    <current_date>2026-05-08, Friday</current_date>
    </system-reminder>

날짜 갱신 형식:

    <system-reminder>
    <current_date>2026-05-09, Saturday</current_date>
    </system-reminder>
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from deerflow.runtime.context_keys import CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY
from deerflow.runtime.user_context import resolve_runtime_user_id

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# 단일 _inject() offload의 상한(초). gateway 시작 시 warm-up이 조용히 실패했다면 첫 요청이
# 여전히 cold tiktoken BPE 다운로드에 걸려 OS TCP timeout(~26분)까지 블로킹될 수 있다.
# 이 상한 덕분에 요청은 멈추지 않고 우아하게 성능을 낮춘다.
_INJECT_TIMEOUT_SECONDS = 5.0

_DATE_RE = re.compile(r"<current_date>([^<]+)</current_date>")
_DYNAMIC_CONTEXT_REMINDER_KEY = "dynamic_context_reminder"
# 주입된 날짜의 authoritative 값. 날짜 SystemMessage의 additional_kwargs에 실린다. 탐지는
# 메시지 content를 regex로 파싱하는 대신 이 값을 읽으므로, 사용자가 영향을 줄 수 있는 memory
# 내용에 절대 노출되지 않는다.
_REMINDER_DATE_KEY = "reminder_date"
_SUMMARY_MESSAGE_NAME = "summary"
# ID-swap이 실제 user 메시지에 붙이는 suffix. reminder SystemMessage가 원래 id를 가져가므로
# ``add_messages``가 제자리에서 교체할 수 있다.
INJECTED_USER_MESSAGE_ID_SUFFIX = "__user"


def strip_injected_user_message_id_suffix(message_id: str | None) -> str | None:
    """reminder ID-swap 이전에 *message_id*가 가지고 있던 id를 반환한다.

    영속화된 user turn을 replay할 때는 client가 원래 보낸 id를 graph에 넣어야 한다.
    ``{id}__user`` 메시지는 주입 대상에서 제외되므로, reminder가 아직 없는 state에 그대로
    replay하면 해당 turn의 날짜와 memory 블록이 조용히 사라진다.
    """

    if isinstance(message_id, str) and message_id.endswith(INJECTED_USER_MESSAGE_ID_SUFFIX):
        return message_id[: -len(INJECTED_USER_MESSAGE_ID_SUFFIX)] or message_id
    return message_id


def _extract_date(content: str) -> str | None:
    """*content*에서 처음 발견된 <current_date> 값을 반환한다. 없으면 None."""
    m = _DATE_RE.search(content)
    return m.group(1) if m else None


def is_dynamic_context_reminder(message: object) -> bool:
    """*message*가 숨겨진 dynamic-context reminder인지 반환한다."""
    # DEPRECATED: HumanMessage reminder는 이 PR 이전 checkpoint에만 존재한다. 활성 checkpoint가
    # 모두 마이그레이션되면 HumanMessage 분기를 제거하고 SystemMessage만 검사하면 된다.
    return isinstance(message, (HumanMessage, SystemMessage)) and bool(message.additional_kwargs.get(_DYNAMIC_CONTEXT_REMINDER_KEY))


def _last_injected_date(messages: list) -> str | None:
    """메시지를 역순으로 훑어 가장 최근에 주입된 날짜를 반환한다.

    탐지는 content substring 매칭이 아니라 ``dynamic_context_reminder``
    additional_kwargs 플래그를 쓰므로, ``<system-reminder>``를 포함한 user 메시지가 주입된
    reminder로 오인되지 않는다.

    authoritative 날짜는 날짜 SystemMessage의 additional_kwargs에 있는 ``reminder_date``
    값이다. 이 값이 없는 reminder(별도의 ``<memory>`` HumanMessage나 향후의 날짜 없는
    reminder)는 날짜를 담지 않으므로 건너뛰고, 실제 날짜 reminder를 가리지 못한다.
    """
    for msg in reversed(messages):
        if not is_dynamic_context_reminder(msg):
            continue
        structured = msg.additional_kwargs.get(_REMINDER_DATE_KEY)
        if isinstance(structured, str) and structured:
            return structured
        # reminder_date가 생기기 전에 기록된 checkpoint를 위한 하위 호환: 그때는 날짜가
        # content에 있었다. regex를 SystemMessage로 한정해 사용자가 영향을 줄 수 있는 memory
        # HumanMessage에서는 절대 돌지 않게 한다(#3630의 OWASP role 분리를 유지하고 memory를
        # 통한 날짜 위조 구멍을 막는다).
        if isinstance(msg, SystemMessage):
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            date = _extract_date(content_str)
            if date is not None:
                return date
    return None


def _is_user_injection_target(message: object) -> bool:
    """*message*가 dynamic-context reminder를 받을 수 있는지 반환한다."""
    if not isinstance(message, HumanMessage):
        return False
    if is_dynamic_context_reminder(message):
        return False
    if message.name == _SUMMARY_MESSAGE_NAME:
        return False
    # 재귀적 ID-swap 방지: ID가 "__user"로 끝나는 메시지는 이전
    # _make_reminder_and_user_messages 호출이 만든 것이므로 다시 처리하면 안 된다. 그러면
    # suffix가 무한히 늘어나고(id__user__user__user...) ghost 메시지가 재실행된다.
    # substring "in"이 아니라 endswith를 써서 중간에 "__user"가 들어간 ID의 오탐을 막는다.
    if message.id and str(message.id).endswith(INJECTED_USER_MESSAGE_ID_SUFFIX):
        return False
    return True


class DynamicContextMiddleware(AgentMiddleware):
    """memory와 현재 날짜를 SystemMessage <system-reminder>로 주입한다.

    첫 turn
    -------
    첫 HumanMessage 앞에 전체 system-reminder(memory + 날짜)를 붙이고 같은 메시지 ID로
    영속화한다. 이후 첫 메시지는 session 내내 고정되어 content가 다시 바뀌지 않으므로, 뒤따르는
    모든 turn에서 prefix cache가 적중할 수 있다.

    자정 통과
    ---------
    대화가 자정을 넘기면 현재 날짜가 앞서 주입된 날짜와 달라진다. 이 경우 **현재**(마지막)
    HumanMessage 앞에 가벼운 날짜 갱신 reminder를 붙이고 영속화한다. 새 날짜의 이후 turn들은
    history에서 보정된 날짜를 보고 재주입을 건너뛴다.
    """

    def __init__(self, agent_name: str | None = None, *, app_config: AppConfig | None = None):
        super().__init__()
        self._agent_name = agent_name
        self._app_config = app_config

    def _build_full_reminder(self, runtime: Runtime | None = None) -> tuple[str, str | None]:
        """(date_reminder, memory_block | None)을 반환한다.

        framework 소유 데이터(날짜)를 사용자 소유 데이터(memory)와 분리해, 하위의
        SystemMessage는 framework 권한만 담고 memory는 role:user에 머무르게 한다. 신뢰할 수
        없는 내용이 system 권한을 얻는 것을 막는다(OWASP LLM01).
        """
        from deerflow.agents.lead_agent.prompt import _get_memory_context

        injection_enabled = self._app_config.memory.injection_enabled if self._app_config else True
        memory_context = (
            _get_memory_context(
                self._agent_name,
                app_config=self._app_config,
                user_id=resolve_runtime_user_id(runtime),
            )
            if injection_enabled
            else ""
        )
        current_date = datetime.now().strftime("%Y-%m-%d, %A")

        date_reminder = "\n".join(
            [
                "<system-reminder>",
                f"<current_date>{current_date}</current_date>",
                "</system-reminder>",
            ]
        )

        memory_block = memory_context.strip() if memory_context else None

        return date_reminder, memory_block

    def _build_date_update_reminder(self) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        return "\n".join(
            [
                "<system-reminder>",
                f"<current_date>{current_date}</current_date>",
                "</system-reminder>",
            ]
        )

    @staticmethod
    def _make_reminder_and_user_messages(
        original: HumanMessage,
        reminder_content: str,
        memory_content: str | None = None,
        *,
        reminder_date: str | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """ID-swap 기법으로 메시지들을 반환한다.

        SystemMessage는 framework 소유 데이터(날짜, metadata)를 담고 원래 ID를 가져가므로
        add_messages가 제자리에서 교체한다. *reminder_date*는 주입된 날짜의 authoritative
        값으로 그 additional_kwargs에 기록된다(``_last_injected_date``가 content를 파싱하는
        대신 이 값을 읽는다). 선택적인 HumanMessage는 사용자 소유 memory 내용을
        ``{id}__memory``로 담는다. 실제 user 메시지는 ``{id}__user``를 받는다.

        SystemMessage를 쓰는 이유는 system context가 user 입력으로 위장해서는 안 되기
        때문이다(#3630). memory는 의도적으로 HumanMessage로 남겨서 사용자가 영향을 줄 수 있는
        내용이 system 권한을 얻지 않게 하고(OWASP LLM01), 의도적으로 ``reminder_date``를 절대
        싣지 않는다.
        """
        stable_id = original.id or str(uuid.uuid4())
        messages: list[SystemMessage | HumanMessage] = []

        reminder_kwargs = {"hide_from_ui": True, _DYNAMIC_CONTEXT_REMINDER_KEY: True}
        if reminder_date is not None:
            reminder_kwargs[_REMINDER_DATE_KEY] = reminder_date
        messages.append(
            SystemMessage(
                content=reminder_content,
                id=stable_id,
                additional_kwargs=reminder_kwargs,
            )
        )

        if memory_content:
            messages.append(
                HumanMessage(
                    content=memory_content,
                    id=f"{stable_id}__memory",
                    additional_kwargs={"hide_from_ui": True, _DYNAMIC_CONTEXT_REMINDER_KEY: True},
                )
            )

        messages.append(
            HumanMessage(
                content=original.content,
                id=f"{stable_id}{INJECTED_USER_MESSAGE_ID_SUFFIX}",
                name=original.name,
                additional_kwargs=original.additional_kwargs,
            )
        )
        return messages

    def _inject(self, state, runtime: Runtime | None = None) -> dict | None:
        messages = list(state.get("messages", []))
        if not messages:
            return None

        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        last_date = _last_injected_date(messages)
        logger.debug(
            "DynamicContextMiddleware._inject: msg_count=%d last_date=%r current_date=%r",
            len(messages),
            last_date,
            current_date,
        )

        if last_date is None:
            # ── 첫 turn: 전체 reminder를 SystemMessage로 주입 ─────
            first_idx = next((i for i, m in enumerate(messages) if _is_user_injection_target(m)), None)
            if first_idx is None:
                return None
            date_reminder, memory_block = self._build_full_reminder(runtime)
            logger.info(
                "DynamicContextMiddleware: injecting full reminder (has_memory=%s) into first HumanMessage id=%r",
                memory_block is not None,
                messages[first_idx].id,
            )
            result_msgs = self._make_reminder_and_user_messages(messages[first_idx], date_reminder, memory_block, reminder_date=current_date)
            return {"messages": result_msgs}

        if last_date == current_date:
            # ── 같은 날: 할 일 없음 ──────────────────────────────────────────
            return None

        # ── 자정 통과: 날짜 갱신 reminder를 SystemMessage로 주입 ──
        last_human_idx = next((i for i in reversed(range(len(messages))) if _is_user_injection_target(messages[i])), None)
        if last_human_idx is None:
            return None

        result_msgs = self._make_reminder_and_user_messages(messages[last_human_idx], self._build_date_update_reminder(), reminder_date=current_date)
        logger.info("DynamicContextMiddleware: midnight crossing detected — injected date update before current turn")
        return {"messages": result_msgs}

    @override
    def before_agent(self, state, runtime: Runtime) -> dict | None:
        result = self._inject(state, runtime)
        self._record_effective_memory(state, result, runtime)
        return result

    @override
    async def abefore_agent(self, state, runtime: Runtime) -> dict | None:
        # _inject()는 동기 파일 I/O(memory JSON 로딩)와 잠재적으로 블로킹되는 네트워크 호출
        # (최초 사용 시 tiktoken encoding 다운로드)을 수행한다. event loop가 절대 막히지 않도록
        # thread로 offload한다. 여기서 블로킹되면 동시에 처리 중인 모든 HTTP handler(auth,
        # SSE heartbeat 등)가 굶는다. issue #3402 참고.
        #
        # 제한된 timeout: 시작 시 warm-up이 조용히 실패했다면(예: 배포 중 네트워크 순단) 첫
        # 요청의 cold tiktoken 다운로드가 수십 분(OS TCP timeout) 동안 블로킹될 수 있다. 주입에
        # 시간 제한을 둬서 요청이 멈추는 대신 우아하게 성능을 낮춘다(새 dynamic-context 갱신
        # 없음). 이미 state에 고정된 context는 그대로 유효하다.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._inject, state, runtime),
                timeout=_INJECT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "DynamicContextMiddleware: injection timed out (%.1fs); skipping new memory/date injection for this turn",
                _INJECT_TIMEOUT_SECONDS,
            )
            self._record_effective_memory(state, None, runtime)
            return None
        self._record_effective_memory(state, result, runtime)
        return result

    @staticmethod
    def _effective_memory_message(state, update: dict | None, runtime: Runtime) -> HumanMessage | None:
        """이번 run에 유효한, 서버가 생성한 memory를 찾는다.

        첫 run의 블록은 이 middleware의 update에서 와야 한다. 재사용된 블록은 run 이전에
        checkpoint에 존재했어야 한다. Gateway가 신뢰할 수 없는 입력에서 reminder 마커를
        제거하므로, 호출자가 알려진 checkpoint ID를 위조된 출처로 바꿔치기할 수 없다.
        """
        if isinstance(update, dict):
            update_messages = update.get("messages")
            if isinstance(update_messages, list):
                for message in update_messages:
                    if not isinstance(message, HumanMessage):
                        continue
                    message_id = str(message.id or "")
                    if message_id.endswith("__memory") and is_dynamic_context_reminder(message) and isinstance(message.content, str):
                        return message

        context = getattr(runtime, "context", None)
        raw_pre_existing_ids = context.get(CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY) if isinstance(context, dict) else None
        if not isinstance(raw_pre_existing_ids, (frozenset, set, list, tuple)):
            return None
        pre_existing_ids = {str(message_id) for message_id in raw_pre_existing_ids if message_id}
        for message in state.get("messages", []):
            if not isinstance(message, HumanMessage):
                continue
            message_id = str(message.id or "")
            if message_id in pre_existing_ids and message_id.endswith("__memory") and is_dynamic_context_reminder(message) and isinstance(message.content, str):
                return message
        return None

    def _record_effective_memory(self, state, update: dict | None, runtime: Runtime) -> None:
        """유효한 숨은 memory 블록을 현재 run ledger에 기록한다."""
        context = getattr(runtime, "context", None)
        journal = context.get("__run_journal") if isinstance(context, dict) else None
        if journal is None:
            return

        message = self._effective_memory_message(state, update, runtime)
        if message is None:
            return

        try:
            journal.record_memory_context(
                content_sha256=hashlib.sha256(message.content.encode("utf-8")).hexdigest(),
            )
        except Exception:
            logger.debug("Failed to record effective memory context", exc_info=True)
