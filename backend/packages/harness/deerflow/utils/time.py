"""Gateway와 embedded runtime을 위한 ISO 8601 timestamp 헬퍼.

DeerFlow는 LangGraph Platform schema에 맞추기 위해 thread/run timestamp를 ISO 8601
UTC 문자열로 저장하고 직렬화한다(``langgraph_sdk.schema.Thread``에서 ``created_at`` /
``updated_at``은 ``datetime``이고 JSON 인코딩 시 ISO 8601이 된다). 모든 timestamp 생성은
:func:`now_iso`를 거쳐야 endpoint, embedded ``RunManager``, Gateway가 쓰는 checkpoint
metadata 전반에서 wire format이 일관되게 유지된다.

:func:`coerce_iso`는 과거에 ``str(time.time())`` float를 저장하던 레코드를 위한
전방 호환 읽기 경로를 제공한다.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

__all__ = ["coerce_iso", "is_lease_expired", "now_iso"]


def is_lease_expired(lease_expires_at: str | None, *, grace_seconds: int) -> bool:
    """*lease_expires_at*이 grace를 넘겨 지났으면 ``True``를 반환한다.

    NULL lease(ownership 도입 이전 데이터)는 항상 만료된 것으로 간주해서, reconciliation과
    같은 방식으로 take-over(소유하지 않은 worker의 cancel)가 회수할 수 있게 한다. 파싱할 수
    없는 timestamp도 만료로 취급한다(defence in depth).
    """
    if lease_expires_at is None:
        return True
    try:
        dt = datetime.fromisoformat(lease_expires_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return True
    return dt < datetime.now(UTC) - timedelta(seconds=grace_seconds)


_UNIX_TIMESTAMP_PATTERN = re.compile(r"^\d{10}(?:\.\d+)?$")
"""과거에 ``str(time.time())``이 쓰던 unix timestamp 문자열 형태(10자리 초와 선택적
소수부)에 매칭된다. 10자리 고정 덕분에 ``"2026"`` 같은 ISO 연도를 실수로 변환하지 않으며
2286년까지 유효하다.
"""


def now_iso() -> str:
    """현재 UTC 시각을 ISO 8601 문자열로 반환한다.

    예: ``"2026-04-27T03:19:46.511479+00:00"``.
    """
    return datetime.now(UTC).isoformat()


def coerce_iso(value: object) -> str:
    """저장된 timestamp를 best-effort로 ISO 8601 문자열로 변환한다.

    구버전 DeerFlow가 쓰던 legacy unix timestamp float/문자열을 일괄 migration 없이 ISO로
    옮긴다. ISO 문자열은 그대로 통과하고, ``datetime``은 UTC로 정규화한 뒤(tz 정보가 없으면
    UTC로 가정) ``isoformat()``으로 내보내 wire format이 항상 ``T`` 구분자를 쓰게 한다.
    빈 값은 ``""``가 되고, 인식할 수 없는 값은 최후 수단으로 문자열화한다.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        # ``bool``은 ``int``의 서브클래스다. 0/1이 아니라 잘못된 값으로 취급한다.
        return str(value)
    if isinstance(value, datetime):
        # ``datetime``은 ``int``/``float`` 검사보다 먼저 처리해야 한다. str(datetime)은
        # ``"YYYY-MM-DD HH:MM:SS+00:00"``(공백 구분자)을 만들어 엄격한 ISO 8601 소비자를
        # 깨뜨린다.
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return value.isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            return str(value)
    if isinstance(value, str):
        if _UNIX_TIMESTAMP_PATTERN.match(value):
            try:
                return datetime.fromtimestamp(float(value), UTC).isoformat()
            except (ValueError, OverflowError, OSError):
                return value
        return value
    return str(value)
