"""tool 주도 memory 모드용 memory tool 모음.

memory_search, memory_add, memory_update, memory_delete를 모델이 직접 호출할 수 있는
LangChain @tool 함수로 노출한다.

memory.mode == "tool"이면 이 tool들이 에이전트에 등록된다. 대부분의 backend는
MemoryMiddleware를 빼고 모델이 저장을 주도하게 하지만, ``requires_passive_writes_in_tool_mode``
를 설정한 backend는 대화 쓰기를 유지하면서 tool로 query 기반 recall을 제공한다.

backend 중립적이다. 모든 tool이 ``MemoryManager`` ABC(:func:`get_memory_manager`)를 거친다.
``search``/``get_memory``는 tier-2 메서드이고, ``create_fact``/``update_fact``/``delete_fact``는
기본 구현이 ``NotImplementedError``를 던지는 tier-3 hook이다(미지원이면 tool이 예외를 잡아
크래시 대신 JSON ``error``를 반환한다). 따라서 해당 연산을 override한 backend라면 tool 모드가
동작한다(DeerMem은 지원하고, noop은 기본 raise를 물려받아 error가 된다).
"""

import json
import logging

from langchain.tools import tool

from deerflow.agents.memory.manager import get_memory_manager
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)


def _resolve_scope(runtime: Runtime | None = None) -> tuple[str | None, str]:
    """tool 핸들러 범위에 쓸 agent_name과 user_id를 해석한다.

    tool 실행은 user·agent metadata를 LangGraph runtime context로 받는다. request/task
    경계를 넘어도 저장 범위가 어긋나지 않도록 ContextVar fallback보다 이 경로를 우선한다.
    """
    context = getattr(runtime, "context", None)
    agent_name = None
    if isinstance(context, dict) and context.get("agent_name"):
        agent_name = str(context["agent_name"])
    return agent_name, resolve_runtime_user_id(runtime)


def _memory_content_key(content: str) -> str:
    return content.strip().casefold()


@tool("memory_search", parse_docstring=True)
def memory_search_tool(
    runtime: Runtime,
    query: str,
    category: str | None = None,
    limit: int = 10,
) -> str:
    """자연어 query로 저장된 fact를 검색한다.

    사용자에 대해 이미 알고 있는 것 - 선호, 과거 정정 사항, 맥락, 저장된 모든 fact -
    을 확인해야 할 때 사용하라.

    Args:
        query: fact 내용과 매칭할 자연어 query. 대소문자를 구분하지 않는 부분 문자열
            매칭이다.
        category: 선택적 category 필터(예: "preference", "correction",
            "context"). 이 category와 정확히 일치하는 fact만 반환된다.
        limit: 반환할 최대 결과 수(기본값 10).

    Returns:
        "results"(fact 객체 리스트)와 "count"를 담은 JSON 문자열.
        각 fact는 id, content, category, confidence, createdAt, source를 가진다.
    """
    agent_name, user_id = _resolve_scope(runtime)
    try:
        results = get_memory_manager().search(
            query,
            top_k=limit,
            user_id=user_id,
            agent_name=agent_name,
            category=category,
        )
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_search_tool failed")
        return json.dumps({"error": str(exc)})


@tool("memory_add", parse_docstring=True)
def memory_add_tool(
    runtime: Runtime,
    content: str,
    category: str = "context",
    confidence: float = 0.7,
) -> str:
    """사용자 또는 대화 맥락에 대한 새 fact를 저장한다.

    사용자가 이후 대화에서 기억할 만한 것 - 선호, 정정 사항, 개인 정보, 업무 맥락 -
    을 알려줬을 때 사용하라. 저장된 fact는 세션을 넘어 유지되며 memory_search와
    자동 context 주입으로 사용할 수 있다.

    Args:
        content: 기억할 fact 텍스트. 구체적이고 사실에 기반해 작성하라.
        category: 정리를 위한 category 라벨(기본값 "context").
            예: "preference", "correction", "behavior", "personal".
        confidence: 이 fact에 대한 확신 정도, 0.0-1.0
            (기본값 0.7). 사용자가 명시적으로 말한 내용은 높게, 추론한 내용은
            낮게 설정하라.

    Returns:
        "fact_id"와 "status": "added"를 담은 JSON 문자열.
        내용이 중복이면 설명이 담긴 "error"를 반환한다.
    """
    agent_name, user_id = _resolve_scope(runtime)
    try:
        normalized_content = content.strip()
        if not normalized_content:
            return json.dumps({"error": "empty content"})
        content_key = _memory_content_key(normalized_content)
        manager = get_memory_manager()
        existing_facts = manager.get_memory(agent_name=agent_name, user_id=user_id).get("facts", [])
        # 흔한 경우의 쓰기 시도를 아끼기 위한 빠른 중복 거부다. 최종 판정은 backend의 생성
        # 임계 구역에 있으므로(DeerMem은 create_memory_fact의 revision 충돌 재시도마다 새
        # 스냅샷으로 다시 검사한다) 같은 user에 대한 동시 tool 호출이 같은 내용을 둘 다 저장할
        # 수는 없다.
        if any(_memory_content_key(str(fact.get("content", ""))) == content_key for fact in existing_facts):
            return json.dumps({"error": "Duplicate fact"})

        # create_fact는 (memory_data, fact_id)를 반환한다. 내용 매칭으로 id를 다시 유추하지
        # 않고 그대로 쓴다. 재유추는 tool을 backend의 내용 정규화 방식에 묶고 storage 상한
        # 상황을 잘못 보고할 수 있다. 미지원 backend는 NotImplementedError(tier-3 기본)를
        # 던지므로 JSON error가 된다.
        try:
            _memory_data, fact_id = manager.create_fact(
                normalized_content,
                category=category,
                confidence=confidence,
                agent_name=agent_name,
                user_id=user_id,
            )
        except NotImplementedError:
            return json.dumps({"error": f"memory backend {type(manager).__name__} does not support create_fact"})
        if fact_id is None:
            # max_facts 상한이 confidence가 더 높은 fact를 남기고 새 fact를 밀어냈다.
            # 저장되지 않았으므로 붕 뜬 id 대신 사실대로 보고한다.
            return json.dumps({"error": "Fact was not stored because memory.max_facts kept higher-confidence facts"})
        return json.dumps({"fact_id": fact_id, "status": "added"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_add_tool failed")
        return json.dumps({"error": str(exc)})


# tool 모드는 수동적인 staleness-review 경로가 아니라 명시적 CRUD를 노출한다.
# staleness의 나이/카테고리/제거 개수 guardrail은 자동 middleware 정리를 보호하는 장치이고,
# tool 모드 operator는 모델 주도 갱신·삭제를 스스로 선택한 것이다. 설정 검토 시 참고하도록
# 문서에도 이 차이를 명시해 두었다.


@tool("memory_update", parse_docstring=True)
def memory_update_tool(
    runtime: Runtime,
    fact_id: str,
    content: str | None = None,
    category: str | None = None,
    confidence: float | None = None,
) -> str:
    """기존 fact를 수정한다. 전달한 필드만 변경되고, 생략한 필드는 그대로
    유지된다.

    저장된 fact가 오래됐거나 틀렸거나 다듬어야 할 때 사용하라.
    먼저 memory_search로 fact_id를 찾은 뒤 수정하라.

    Args:
        fact_id: memory_search 결과에서 얻은 fact ID(필수).
        content: 새 fact 텍스트(생략하면 변경되지 않는다).
        category: 새 category(생략하면 변경되지 않는다).
        confidence: 새 confidence 점수 0.0-1.0(생략하면 변경되지 않는다).

    Returns:
        "fact_id"와 "status": "updated"를 담은 JSON 문자열.
        fact_id가 유효하지 않으면 설명이 담긴 "error"를 반환한다.
    """
    agent_name, user_id = _resolve_scope(runtime)
    try:
        manager = get_memory_manager()
        try:
            manager.update_fact(
                fact_id,
                content=content,
                category=category,
                confidence=confidence,
                agent_name=agent_name,
                user_id=user_id,
            )
        except NotImplementedError:
            return json.dumps({"error": f"memory backend {type(manager).__name__} does not support update_fact"})
        return json.dumps({"fact_id": fact_id, "status": "updated"})
    except KeyError:
        return json.dumps({"error": f"Fact not found: {fact_id}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_update_tool failed")
        return json.dumps({"error": str(exc)})


@tool("memory_delete", parse_docstring=True)
def memory_delete_tool(runtime: Runtime, fact_id: str) -> str:
    """ID로 fact를 삭제한다.

    fact가 더 이상 정확하지 않거나 관련이 없을 때 사용하라. 먼저
    memory_search로 fact_id를 찾은 뒤 삭제하라.

    Args:
        fact_id: 삭제할 fact ID(memory_search 결과에서 얻는다).

    Returns:
        "fact_id"와 "status": "deleted"를 담은 JSON 문자열.
        fact_id가 유효하지 않으면 설명이 담긴 "error"를 반환한다.
    """
    agent_name, user_id = _resolve_scope(runtime)
    try:
        manager = get_memory_manager()
        try:
            manager.delete_fact(fact_id, agent_name=agent_name, user_id=user_id)
        except NotImplementedError:
            return json.dumps({"error": f"memory backend {type(manager).__name__} does not support delete_fact"})
        return json.dumps({"fact_id": fact_id, "status": "deleted"})
    except KeyError:
        return json.dumps({"error": f"Fact not found: {fact_id}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_delete_tool failed")
        return json.dumps({"error": str(exc)})


def get_memory_tools() -> list:
    """에이전트 등록용 memory tool 전체를 반환한다.

    memory.mode == "tool"일 때 agent factory가 호출한다.
    """
    return [
        memory_search_tool,
        memory_add_tool,
        memory_update_tool,
        memory_delete_tool,
    ]
