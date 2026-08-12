import ast
import html
import json
import re
import uuid
from collections.abc import Iterator

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


def _fix_messages(messages: list) -> list:
    """MindIE 호환을 위해 들어오는 메시지를 정리한다.

    MindIE의 chat template은 LangChain의 네이티브 tool_calls나 ToolMessage role을 파싱하지 못해
    0-token 생성 오류를 낼 수 있다. 이 함수는 multi-modal list 내용을 문자열로 펼치고, tool 관련
    메시지를 바탕 모델이 기대하는 XML 태그가 붙은 raw 텍스트로 변환한다.
    """
    fixed = []
    for msg in messages:
        # 내용이 block list면 펼친다
        if isinstance(msg.content, list):
            parts = []
            for block in msg.content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "".join(parts)
        else:
            text = msg.content or ""

        # tool_calls가 있는 AIMessage를 raw XML 텍스트 형식으로 변환한다
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            xml_parts = []
            for tool in msg.tool_calls:
                args_xml = " ".join(f"<parameter={html.escape(str(k), quote=False)}>{html.escape(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), quote=False)}</parameter>" for k, v in tool.get("args", {}).items())
                xml_parts.append(f"<tool_call> <function={html.escape(str(tool['name']), quote=False)}> {args_xml} </function> </tool_call>")
            full_text = f"{text}\n" + "\n".join(xml_parts) if text else "\n".join(xml_parts)
            fixed.append(AIMessage(content=full_text.strip() or " "))
            continue

        # tool 실행 결과를 XML 태그로 감싸 HumanMessage로 변환한다.
        # tool 출력을 escape해서, "</tool_response>" 문자열이 그대로 들어 있는 결과(예: 신뢰할 수
        # 없는 파일에 대한 read_file, bash 출력, ToolResultSanitizationMiddleware allowlist가
        # 다루지 않는 MCP 도구)가 framing을 조기에 닫고 뒤에 텍스트를 주입하지 못하게 한다 —
        # 위에서 tool-call 이름/인자에 이미 적용한 escape와 동일한 이유다.
        if isinstance(msg, ToolMessage):
            tool_result_text = f"<tool_response>\n{html.escape(text, quote=False)}\n</tool_response>"
            fixed.append(HumanMessage(content=tool_result_text))
            continue

        # 메시지 내용이 완전히 비지 않도록 하는 fallback
        if not text.strip():
            text = " "

        fixed.append(msg.model_copy(update={"content": text}))

    return fixed


def _parse_xml_tool_call_to_dict(content: str) -> tuple[str, list[dict]]:
    """모델 출력의 XML 형식 tool call을 LangChain dict로 파싱한다.

    Args:
        content: 모델이 낸 raw 텍스트 출력.

    Returns:
        XML block을 제거한 정리된 텍스트와, LangChain 형식의 tool call dict list로 이루어진
        tuple.
    """
    if not isinstance(content, str) or "<tool_call>" not in content:
        return content, []

    tool_calls = []
    clean_parts: list[str] = []
    cursor = 0
    for start, end, inner_content in _iter_tool_call_blocks(content):
        clean_parts.append(content[cursor:start])
        cursor = end

        func_match = re.search(r"<function=([^>]+)>", inner_content)
        if not func_match:
            continue
        function_name = html.unescape(func_match.group(1).strip())

        # 이 호출의 파라미터를 뽑을 때 중첩된 tool block은 무시한다. 중첩된 `<tool_call>`
        # 구간은 별개의 호출이므로 그 `<parameter>` 태그가 현재 호출 인자로 새면 안 된다.
        param_source_parts: list[str] = []
        nested_cursor = 0
        for nested_start, nested_end, _ in _iter_tool_call_blocks(inner_content):
            param_source_parts.append(inner_content[nested_cursor:nested_start])
            nested_cursor = nested_end
        param_source_parts.append(inner_content[nested_cursor:])
        param_source = "".join(param_source_parts)

        args = {}
        param_pattern = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
        for param_match in param_pattern.finditer(param_source):
            key = html.unescape(param_match.group(1).strip())
            raw_value = html.unescape(param_match.group(2).strip())

            # 이후 Pydantic 검증을 통과하도록 문자열 값을 네이티브 Python 타입으로
            # 역직렬화해 본다.
            parsed_value = raw_value
            if raw_value.startswith(("[", "{")) or raw_value in ("true", "false", "null") or raw_value.isdigit():
                try:
                    parsed_value = json.loads(raw_value)
                except json.JSONDecodeError:
                    try:
                        parsed_value = ast.literal_eval(raw_value)
                    except (ValueError, SyntaxError):
                        pass

            args[key] = parsed_value

        tool_calls.append({"name": function_name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}"})
    clean_parts.append(content[cursor:])

    return "".join(clean_parts).strip(), tool_calls


def _iter_tool_call_blocks(content: str) -> Iterator[tuple[int, int, str]]:
    """`<tool_call>...</tool_call>` block을 순회하며 중첩도 허용한다."""
    token_pattern = re.compile(r"</?tool_call>")
    depth = 0
    block_start = -1

    for match in token_pattern.finditer(content):
        token = match.group(0)
        if token == "<tool_call>":
            if depth == 0:
                block_start = match.start()
            depth += 1
            continue

        if depth == 0:
            continue

        depth -= 1
        if depth == 0 and block_start != -1:
            block_end = match.end()
            inner_start = block_start + len("<tool_call>")
            inner_end = match.start()
            yield block_start, block_end, content[inner_start:inner_end]
            block_start = -1


def _decode_escaped_newlines_outside_fences(content: str) -> str:
    """fenced code block 바깥의 리터럴 `\\n`을 디코딩한다."""
    if "\\n" not in content:
        return content

    parts = re.split(r"(```[\s\S]*?```)", content)
    for idx, part in enumerate(parts):
        if part.startswith("```"):
            continue
        parts[idx] = part.replace("\\n", "\n")
    return "".join(parts)


class MindIEChatModel(ChatOpenAI):
    """MindIE 엔진용 chat model adapter.

    다음 호환성 문제를 처리한다:
    - multimodal list 내용을 문자열로 펼친다.
    - 하드코딩된 XML tool call을 가로채 LangChain 표준으로 파싱한다.
    - tool이 있을 때 stream=True가 choices를 누락하는 문제를 비streaming 생성으로 fallback한 뒤
      가짜 chunk를 내보내 처리한다.
    - gateway 응답의 과도하게 escape된 개행 문자를 고친다.
    """

    def __init__(self, **kwargs):
        """오래 사는 client를 만들지 않고 timeout kwargs를 정규화한다."""
        connect_timeout = kwargs.pop("connect_timeout", 30.0)
        read_timeout = kwargs.pop("read_timeout", 900.0)
        write_timeout = kwargs.pop("write_timeout", 60.0)
        pool_timeout = kwargs.pop("pool_timeout", 30.0)

        kwargs.setdefault(
            "timeout",
            httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )
        super().__init__(**kwargs)

    def _patch_result_with_tools(self, result: ChatResult) -> ChatResult:
        """모델 결과에 생성 후 보정을 적용한다."""
        for gen in result.generations:
            msg = gen.message

            if isinstance(msg.content, str):
                # fenced code block 안의 escape된 개행은 건드리지 않는다.
                msg.content = _decode_escaped_newlines_outside_fences(msg.content)

                if "<tool_call>" in msg.content:
                    clean_content, extracted_tools = _parse_xml_tool_call_to_dict(msg.content)

                    if extracted_tools:
                        msg.content = clean_content
                        if getattr(msg, "tool_calls", None) is None:
                            msg.tool_calls = []
                        msg.tool_calls.extend(extracted_tools)
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(_fix_messages(messages), stop=stop, run_manager=run_manager, **kwargs)
        return self._patch_result_with_tools(result)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(_fix_messages(messages), stop=stop, run_manager=run_manager, **kwargs)
        return self._patch_result_with_tools(result)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # 일반 질의는 TTFB를 낮추기 위해 네이티브 streaming으로 보낸다
        if not kwargs.get("tools"):
            async for chunk in super()._astream(_fix_messages(messages), stop=stop, run_manager=run_manager, **kwargs):
                if isinstance(chunk.message.content, str):
                    chunk.message.content = _decode_escaped_newlines_outside_fences(chunk.message.content)
                yield chunk
            return

        # tool을 쓰는 요청의 fallback:
        # 현재 MindIE는 stream=True이면서 tool이 있으면 choices를 누락한다. 생성이 끝날 때까지
        # 기다린 뒤 chunk로 쪼개 내보내 streaming을 흉내 낸다.
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        for gen in result.generations:
            msg = gen.message
            content = msg.content
            standard_tool_calls = getattr(msg, "tool_calls", [])

            # 하위 UI/Markdown 파서가 매끄럽게 렌더링하도록 텍스트를 chunk 단위로 내보낸다
            if isinstance(content, str) and content:
                chunk_size = 15
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i : i + chunk_size]
                    chunk_msg = AIMessageChunk(content=chunk_text, id=msg.id, response_metadata=msg.response_metadata if i == 0 else {})
                    yield ChatGenerationChunk(message=chunk_msg, generation_info=gen.generation_info if i == 0 else None)

                if standard_tool_calls:
                    yield ChatGenerationChunk(message=AIMessageChunk(content="", id=msg.id, tool_calls=standard_tool_calls, invalid_tool_calls=getattr(msg, "invalid_tool_calls", [])))
            else:
                chunk_msg = AIMessageChunk(content=content, id=msg.id, tool_calls=standard_tool_calls, invalid_tool_calls=getattr(msg, "invalid_tool_calls", []))
                yield ChatGenerationChunk(message=chunk_msg, generation_info=gen.generation_info)
