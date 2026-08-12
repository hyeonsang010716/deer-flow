"""Principal builder — Principal을 생성하는 유일한 공식 경로.

Layer 1(tool assembly)과 Layer 2(GuardrailAuthorizationAdapter)가 모두 이 builder를
써야 identity 의미가 일관되게 유지된다. 순수 함수라서 전역 config를 읽지 않고,
캐싱하지 않고, 입력을 변경하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deerflow.authz.provider import Principal


def normalize_authz_attributes(raw: Any) -> dict[str, Any]:
    """``authz_attributes``를 검증한 뒤 새 dict로 복사한다.

    Principal builder와 모든 전파 지점(middleware, executor, task_tool)이 공유하는
    단일 정규화 지점이다. 한 곳에 모아둬야 모든 in-process 소비 경계가 Mapping이
    아닌 값에 대해 조용히 변환하지 않고 ``TypeError``를 던진다.

    Raises:
        TypeError: *raw*가 ``None``도 ``Mapping``도 아닐 때.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(f"authz_attributes must be a Mapping, got {type(raw).__name__}")


def build_principal_from_context(
    context: Mapping[str, Any],
    *,
    default_role: str,
) -> Principal:
    """runtime context mapping으로부터 :class:`Principal`을 만든다.

    Args:
        context: runtime context(``config["context"]`` 또는
            :class:`~deerflow.guardrails.provider.GuardrailRequest`에서 조립한 dict).
        default_role: ``user_role``이 ``None``이거나 빈 문자열일 때 쓰는 role.
            비어 있지 않다면 알 수 없는 role이어도 **교체하지 않는다**. 없는 경우만 채운다.

    Raises:
        TypeError: ``authz_attributes``가 있으나 ``Mapping``이 아닐 때.
    """
    resolved_role = context.get("user_role")
    if resolved_role is None or resolved_role == "":
        resolved_role = default_role

    return Principal(
        user_id=context.get("user_id"),
        role=resolved_role,
        oauth_provider=context.get("oauth_provider"),
        oauth_id=context.get("oauth_id"),
        channel_user_id=context.get("channel_user_id"),
        is_internal=context.get("is_internal") is True,
        attributes=normalize_authz_attributes(context.get("authz_attributes")),
    )
