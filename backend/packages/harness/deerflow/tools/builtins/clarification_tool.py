from typing import Literal, Required, TypedDict

from langchain.tools import tool


class ClarificationFormField(TypedDict, total=False):
    """구조화된 clarification 카드의 form field 하나에 대한 정의.

    모델에 노출되는 schema는 항목의 모양을 문서화할 뿐이다. middleware가 tool 실행 전에 호출을
    가로채므로, runtime 검증은 여전히 ``ClarificationMiddleware``에서 방어적으로 수행된다.
    """

    name: Required[str]
    label: str
    type: Literal["text", "textarea", "number", "select", "multi_select", "checkbox", "date"]
    required: bool
    options: list[str]
    placeholder: str


@tool("ask_clarification", parse_docstring=True, return_direct=True)
def ask_clarification_tool(
    question: str,
    clarification_type: Literal[
        "missing_info",
        "ambiguous_requirement",
        "approach_choice",
        "risk_confirmation",
        "suggestion",
    ],
    context: str | None = None,
    options: list[str] | None = None,
    fields: list[ClarificationFormField] | None = None,
) -> str:
    """진행하는 데 정보가 더 필요할 때 사용자에게 확인을 요청한다.

    사용자 입력 없이는 진행할 수 없는 상황에서 이 tool을 사용하라:

    - **정보 누락**: 필요한 세부 정보가 제공되지 않음 (예: 파일 경로, URL, 구체적인 요구사항)
    - **모호한 요구사항**: 유효한 해석이 여러 개 존재함
    - **접근 방식 선택**: 유효한 접근 방식이 여러 개라 사용자 선호가 필요함
    - **위험한 작업**: 명시적 확인이 필요한 파괴적 동작 (예: 파일 삭제, production 수정)
    - **제안**: 권장안이 있지만 진행 전 사용자 승인을 받고 싶음

    실행이 중단되고 질문이 사용자에게 표시된다.
    계속 진행하기 전에 사용자의 응답을 기다려라.

    ask_clarification을 사용해야 할 때:
    - 사용자 요청에 포함되지 않은 정보가 필요할 때
    - 요구사항이 여러 방식으로 해석될 수 있을 때
    - 유효한 구현 방식이 여러 개 존재할 때
    - 잠재적으로 위험한 작업을 수행하려 할 때
    - 권장안이 있지만 사용자 승인이 필요할 때

    상호작용 형태 선택:
    - 열린 질문 하나 -> `question`만 사용 (자유 텍스트 입력)
    - 정확히 하나만 고르게 할 때 -> `options`
    - 여러 개를 고르게 할 때 -> `multi_select` 타입의 `fields` 항목 하나
    - 여러 값을 한 번에 수집할 때 (예: 한 동작에 필요한 파라미터 묶음) ->
      `fields`. 순차적인 여러 질문 대신 구조화된 form 하나를 렌더링한다.
      항목별로 나눠 묻지 말고 form 하나를 우선하라.

    모범 사례:
    - 명확성을 위해 한 번에 ONE clarification만 요청하라. 여러 field를 가진 form도
      하나의 clarification으로 센다
    - 질문은 구체적이고 명확하게 하라
    - clarification이 필요한 상황에서 임의로 가정하지 마라
    - 위험한 작업에는 ALWAYS 확인을 요청하라
    - skill이 미리 정의된 field 템플릿을 제공하면 재설계하지 말고 `fields`로 그대로
      전달하라
    - 이 tool을 호출하면 실행이 자동으로 중단된다

    Args:
        question: 사용자에게 물어볼 clarification 질문. 구체적이고 명확하게 작성하라.
        clarification_type: 필요한 clarification의 종류 (missing_info, ambiguous_requirement, approach_choice, risk_confirmation, suggestion).
        context: clarification이 필요한 이유를 설명하는 선택적 context. 사용자가 상황을 이해하는 데 도움이 된다.
        options: 선택지 목록 (선택, approach_choice 또는 suggestion 타입용). 사용자가 고를 수 있도록 명확한 선택지를 제시하라.
        fields: 한 카드에서 여러 값을 수집하기 위한 form field 정의 (선택). `options`보다 우선한다.
            각 field는 다음 키를 가진 object다: `name` (고유 식별자, 필수. `constructor`나 `toString` 같은
            JavaScript prototype 이름은 피하라), `label` (표시 텍스트, 기본값은 name), `type` (다음 중 하나: text, textarea,
            number, select, multi_select, checkbox, date. 기본값은 text), `required` (boolean, 기본값 false),
            `options` (문자열 목록, select/multi_select 타입에는 필수), `placeholder` (선택적 힌트 텍스트).
            `checkbox` field는 기본값이 "no"인 boolean이다. checkbox에 `required`를 지정하는 것은 반드시 동의해야
            하는 의미(사용자가 체크해야 제출 가능)일 때만 사용하라. form 크기는 제한을 지켜라: 최대 16 fields,
            field당 24 options, name/label/option/placeholder당 200 characters. 한도를 초과하면 요청 전체가
            평문 질문으로 격하된다.
    """
    # 자리표시자 구현이다.
    # 실제 로직은 ClarificationMiddleware가 담당한다. 이 tool 호출을 가로채고 실행을 중단해서
    # 사용자에게 질문을 보여준다.
    return "Clarification request processed by middleware"
