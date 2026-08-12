"""LLM 호출 전에 이미지 상세 정보를 대화에 주입하는 middleware."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import override
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# tool 쪽 크기 상한을 심층 방어 차원에서 그대로 둔다. tool은 기록 시점에 이를 강제하고,
# middleware는 view와 주입 사이에 디스크에서 파일이 커졌을 경우에 대비해 읽기 시점에 다시
# 검사한다.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_CONTEXT_MESSAGE_ID_PREFIX = "view-image-context:"
_IMAGE_CONTEXT_MESSAGE_MARKER_KEY = "deerflow_view_image_context"


class ViewImageMiddlewareState(ThreadState):
    """reducer가 붙은 key들이 annotation을 유지하도록 thread state를 재사용한다."""


class ViewImageMiddleware(AgentMiddleware[ViewImageMiddlewareState]):
    """view_image tool이 완료되면 LLM 호출 전에 이미지 상세 정보를 human message로 주입한다.

    이 middleware는:
    1. 매 LLM 호출 전에 실행된다
    2. 마지막 assistant 메시지에 view_image tool call이 있는지 확인한다
    3. 그 메시지의 모든 tool call이 완료됐는지(대응하는 ToolMessage가 있는지) 검증한다
    4. 조건이 맞으면 조회된 모든 이미지 상세 정보(base64 데이터 포함)를 담은 human message를 만든다
    5. 그 메시지를 state에 추가해 LLM이 이미지를 보고 분석할 수 있게 한다
    6. LLM 호출 이후 임시 메시지를 제거해 이후 checkpoint에 base64 데이터가 남지 않게 한다

    덕분에 LLM은 사용자가 이미지 설명을 명시적으로 요청하지 않아도 view_image tool로 로드된
    이미지를 자동으로 받아 분석할 수 있다.
    """

    state_schema = ViewImageMiddlewareState

    @staticmethod
    def _is_image_context_message(message: object) -> bool:
        """메시지가 신뢰할 수 있는 임시 이미지 context인지 반환한다."""
        return isinstance(message, HumanMessage) and bool(message.id) and message.id.startswith(_IMAGE_CONTEXT_MESSAGE_ID_PREFIX) and message.additional_kwargs.get(_IMAGE_CONTEXT_MESSAGE_MARKER_KEY) is True

    def _get_last_assistant_message(self, messages: list) -> AIMessage | None:
        """메시지 목록에서 마지막 assistant 메시지를 가져온다.

        Args:
            messages: 메시지 목록

        Returns:
            마지막 AIMessage, 없으면 None
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def _has_view_image_tool(self, message: AIMessage) -> bool:
        """assistant 메시지에 view_image tool call이 있는지 확인한다.

        Args:
            message: 검사할 assistant 메시지

        Returns:
            view_image tool call이 있으면 True
        """
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return False

        return any(tool_call.get("name") == "view_image" for tool_call in message.tool_calls)

    def _all_tools_completed(self, messages: list, assistant_msg: AIMessage) -> bool:
        """assistant 메시지의 모든 tool call이 완료됐는지 확인한다.

        Args:
            messages: 전체 메시지 목록
            assistant_msg: tool call을 담은 assistant 메시지

        Returns:
            모든 tool call에 대응하는 ToolMessage가 있으면 True
        """
        if not hasattr(assistant_msg, "tool_calls") or not assistant_msg.tool_calls:
            return False

        # assistant 메시지의 모든 tool call ID를 모은다
        tool_call_ids = {tool_call.get("id") for tool_call in assistant_msg.tool_calls if tool_call.get("id")}

        # assistant 메시지의 인덱스를 찾는다
        try:
            assistant_idx = messages.index(assistant_msg)
        except ValueError:
            return False

        # assistant 메시지 이후의 모든 ToolMessage를 모은다
        completed_tool_ids = set()
        for msg in messages[assistant_idx + 1 :]:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                completed_tool_ids.add(msg.tool_call_id)

        # 모든 tool call이 완료됐는지 확인한다
        return tool_call_ids.issubset(completed_tool_ids)

    @staticmethod
    def _read_image_as_data_url(actual_path: str, mime_type: str, expected_size: int) -> str | None:
        """이미지 파일을 읽어 `data:` URL을 반환한다. 실패하면 None.

        신뢰 전제: ``actual_path``는 ``view_image_tool``이 설정하고(서버 측에서, 기록 시점에
        허용된 virtual root를 기준으로 검증됨) LangGraph가 관리하는 state에 보관된다. client
        입력은 이 필드에 도달할 수 없으므로 읽기 범위는 신뢰할 수 있다. 그래도 TOCTOU로 인한
        크기 증가를 막기 위해 읽기 시점에 크기를 다시 확인하고, ``_MAX_IMAGE_BYTES``를 넘는
        파일은 건너뛴다.
        """
        try:
            file_path = Path(actual_path)
            if not file_path.exists() or not file_path.is_file():
                return None
            current_size = file_path.stat().st_size
            if current_size != expected_size:
                # view와 주입 사이에 파일이 바뀌었으므로 건너뛴다.
                return None
            if current_size > _MAX_IMAGE_BYTES:
                return None
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"
        except OSError:
            return None

    def _create_image_details_message(self, state: ViewImageMiddlewareState) -> list[str | dict]:
        """조회된 모든 이미지 상세 정보를 담은 포맷된 메시지를 만든다.

        필요할 때 디스크에서 이미지 파일을 읽어 모델용 base64로 인코딩한다. base64 데이터는
        state에 영속화하지 **않는다**. ``viewed_images``에는 가벼운 metadata(path, mime_type,
        size)만 저장해 모든 checkpoint에 큰 payload가 중복되는 것을 피한다(#4138 참고).

        Args:
            state: viewed_images를 담은 현재 state

        Returns:
            HumanMessage에 넣을 content block(텍스트와 이미지) 목록
        """
        viewed_images = state.get("viewed_images", {})
        if not viewed_images:
            # 단순 문자열 배열이 아니라 올바르게 포맷된 텍스트 블록을 반환한다
            return [{"type": "text", "text": "No images have been viewed."}]

        # 이미지 정보를 담은 메시지를 구성한다
        content_blocks: list[str | dict] = [{"type": "text", "text": "Here are the images you've viewed:"}]

        for image_path, image_data in viewed_images.items():
            mime_type = image_data.get("mime_type", "unknown")
            actual_path = image_data.get("actual_path", "")
            expected_size = image_data.get("size", 0)

            # 텍스트 설명을 추가한다
            content_blocks.append({"type": "text", "text": f"\n- **{image_path}** ({mime_type})"})

            # 필요할 때 이미지 파일을 읽어 모델용 base64로 인코딩한다
            if actual_path:
                data_url = self._read_image_as_data_url(actual_path, mime_type, expected_size)
                if data_url:
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                else:
                    content_blocks.append({"type": "text", "text": f"  (file unavailable or changed on disk: {actual_path})"})

        return content_blocks

    def _should_inject_image_message(self, state: ViewImageMiddlewareState) -> bool:
        """이미지 상세 정보 메시지를 주입해야 하는지 판단한다.

        Args:
            state: 현재 state

        Returns:
            메시지를 주입해야 하면 True
        """
        messages = state.get("messages", [])
        if not messages:
            return False

        # 마지막 assistant 메시지를 가져온다
        last_assistant_msg = self._get_last_assistant_message(messages)
        if not last_assistant_msg:
            return False

        # view_image tool call이 있는지 확인한다
        if not self._has_view_image_tool(last_assistant_msg):
            return False

        # 모든 tool이 완료됐는지 확인한다
        if not self._all_tools_completed(messages, last_assistant_msg):
            return False

        # 이미 이미지 상세 정보 메시지를 추가했는지 확인한다.
        # 마지막 assistant 메시지 이후에 이미지 상세 정보를 담은 human message가 있는지 본다.
        assistant_idx = messages.index(last_assistant_msg)
        for msg in messages[assistant_idx + 1 :]:
            if isinstance(msg, HumanMessage):
                if self._is_image_context_message(msg):
                    return False
                content_str = str(msg.content)
                if "Here are the images you've viewed" in content_str or "Here are the details of the images you've viewed" in content_str:
                    # 이미 추가됐으므로 다시 추가하지 않는다
                    return False

        return True

    @staticmethod
    def _create_image_context_message(content: list[str | dict]) -> HumanMessage:
        """식별 가능한, 모델 전용 이미지 context 메시지를 만든다."""
        return HumanMessage(
            id=f"{_IMAGE_CONTEXT_MESSAGE_ID_PREFIX}{uuid4().hex}",
            content=content,
            additional_kwargs={
                "hide_from_ui": True,
                _IMAGE_CONTEXT_MESSAGE_MARKER_KEY: True,
            },
        )

    @staticmethod
    def _remove_image_context_messages(state: ViewImageMiddlewareState) -> dict | None:
        """모델이 소비한 뒤 임시 이미지 context 메시지를 제거한다."""
        removals = [RemoveMessage(id=msg.id) for msg in state.get("messages", []) if ViewImageMiddleware._is_image_context_message(msg)]
        if not removals:
            return None
        return {"messages": removals}

    def _inject_image_message(self, state: ViewImageMiddlewareState) -> dict | None:
        """이미지 상세 정보 메시지를 주입하는 내부 헬퍼.

        Args:
            state: 현재 state

        Returns:
            human message가 추가된 state update, 갱신이 필요 없으면 None
        """
        if not self._should_inject_image_message(state):
            return None

        # 텍스트와 이미지 content를 담은 이미지 상세 정보 메시지를 만든다
        image_content = self._create_image_details_message(state)

        # 혼합 content(텍스트 + 이미지)를 담은 새 human message를 만든다. 이는 모델 전용 내부
        # context이므로 chat UI와 IM channel에서 숨긴다(다른 middleware 주입 context 메시지와
        # 동일하다).
        human_msg = self._create_image_context_message(image_content)

        logger.debug("Injecting image details message with images before LLM call")

        # 새 메시지를 담은 state update를 반환한다
        return {"messages": [human_msg]}

    @override
    def before_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """view_image tool이 완료됐다면 LLM 호출 전에 이미지 상세 정보 메시지를 주입한다(동기 버전).

        매 LLM 호출 전에 실행되며, 직전 turn에 모두 완료된 view_image tool call이 있었는지
        확인한다. 있으면 LLM이 이미지를 보고 분석할 수 있도록 이미지 상세 정보를 담은 human
        message를 주입한다.

        Args:
            state: 현재 state
            runtime: Runtime context(사용하지 않지만 인터페이스상 필요하다)

        Returns:
            human message가 추가된 state update, 갱신이 필요 없으면 None
        """
        return self._inject_image_message(state)

    @override
    async def abefore_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """view_image tool이 완료됐다면 LLM 호출 전에 이미지 상세 정보 메시지를 주입한다(비동기 버전).

        매 LLM 호출 전에 실행되며, 직전 turn에 모두 완료된 view_image tool call이 있었는지
        확인한다. 있으면 LLM이 이미지를 보고 분석할 수 있도록 이미지 상세 정보를 담은 human
        message를 주입한다.

        Args:
            state: 현재 state
            runtime: Runtime context(사용하지 않지만 인터페이스상 필요하다)

        Returns:
            human message가 추가된 state update, 갱신이 필요 없으면 None
        """
        if not self._should_inject_image_message(state):
            return None
        # 이미지 읽기와 base64 인코딩은 느릴 수 있으므로(최대 20MB), event loop를 멈추는 대신
        # 블로킹 작업을 thread로 offload한다.
        image_content = await asyncio.to_thread(self._create_image_details_message, state)
        human_msg = self._create_image_context_message(image_content)
        logger.debug("Injecting image details message with images before LLM call")
        return {"messages": [human_msg]}

    @override
    def after_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """이후 checkpoint 전에 모델 전용 이미지 데이터를 제거한다(동기 버전)."""
        return self._remove_image_context_messages(state)

    @override
    async def aafter_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """이후 checkpoint 전에 모델 전용 이미지 데이터를 제거한다(비동기 버전)."""
        return self._remove_image_context_messages(state)
