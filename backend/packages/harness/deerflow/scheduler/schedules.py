from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter


def validate_timezone(timezone_name: str) -> str:
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
    return timezone_name


def normalize_cron_expression(expr: str) -> str:
    parts = [part for part in expr.split() if part]
    if len(parts) != 5:
        raise ValueError("Cron expression must contain exactly 5 fields")
    return " ".join(parts)


def next_run_at(
    schedule_type: str,
    schedule_spec: dict[str, object],
    timezone_name: str,
    *,
    now: datetime,
) -> datetime | None:
    validate_timezone(timezone_name)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    if schedule_type == "once":
        run_at_raw = schedule_spec.get("run_at")
        if not isinstance(run_at_raw, str):
            raise ValueError("once schedule requires run_at")
        run_at = datetime.fromisoformat(run_at_raw)
        if run_at.tzinfo is None:
            # naive한 run_at은 "task가 선언한 timezone의 벽시계 시각"을 뜻하며, cron
            # schedule이 해석하는 방식과 같다.
            run_at = run_at.replace(tzinfo=ZoneInfo(timezone_name))
        # cron 분기와 마찬가지로 UTC로 정규화한다. next_run_at은 timezone을 버리는
        # 컬럼(SQLite)에 저장되므로, UTC가 아닌 offset은 실제 발화 시각을 그 offset만큼
        # 통째로 밀어 버린다.
        run_at = run_at.astimezone(UTC)
        return run_at if run_at > now else None

    if schedule_type == "cron":
        cron_expr = normalize_cron_expression(str(schedule_spec.get("cron", "")))
        zone = ZoneInfo(timezone_name)
        local_now = now.astimezone(zone)
        next_local = croniter(cron_expr, local_now).get_next(datetime)
        if next_local.tzinfo is None:
            next_local = next_local.replace(tzinfo=zone)
        return next_local.astimezone(UTC)

    raise ValueError(f"Unsupported schedule_type: {schedule_type}")
