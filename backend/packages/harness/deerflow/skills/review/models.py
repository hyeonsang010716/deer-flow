"""skill review의 공용 contract와 결정적 헬퍼."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

PACKAGE_SNAPSHOT_SCHEMA_VERSION = "deerflow.skill-package-snapshot.v1"
FACTS_SCHEMA_VERSION = "deerflow.skill-review.facts.v1"
REPORT_SCHEMA_VERSION = "deerflow.skill-review.report.v1"

Severity = Literal["blocker", "error", "warning", "info"]
ProfileName = Literal["deerflow", "agentskills"]

SEVERITY_RANK: dict[str, int] = {
    "blocker": 0,
    "error": 1,
    "warning": 2,
    "info": 3,
}

SKILLSCAN_SEVERITY_MAP: dict[str, Severity] = {
    "CRITICAL": "blocker",
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "info",
}


@dataclass(frozen=True)
class PackageLimits:
    max_files: int = 4096
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024

    def to_dict(self) -> dict[str, int]:
        return {
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


DEFAULT_PACKAGE_LIMITS = PackageLimits()


def stable_json_dumps(data: Any) -> str:
    """review 데이터를 바이트 단위로 안정적이고 경로에 무관한 형태로 직렬화한다."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_relative_path(path: str) -> str:
    """package 상대 경로를 정규화하고, 바깥으로 벗어나려는 시도는 거부한다."""
    raw = path.replace("\\", "/").strip()
    if not raw:
        raise ValueError("path must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError("absolute paths are not allowed")
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."}:
        raise ValueError("path must not resolve to package root")
    parts = PurePosixPath(normalized).parts
    if any(part in {"..", ""} for part in parts):
        raise ValueError("path must not contain parent-directory traversal")
    return normalized


def make_finding(
    rule_id: str,
    *,
    severity: Severity,
    message: str,
    remediation: str,
    source: str = "review-core",
    profile: str = "deerflow",
    path: str | None = None,
    line: int | None = None,
    evidence: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finding = {
        "rule_id": rule_id,
        "source": source,
        "profile": profile,
        "severity": severity,
        "path": path,
        "line": line,
        "message": message,
        "remediation": remediation,
        "evidence": evidence,
    }
    if extra:
        finding.update(extra)
    return finding


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda item: (
            SEVERITY_RANK.get(str(item.get("severity")), 99),
            str(item.get("path") or ""),
            item.get("line") if item.get("line") is not None else 10**9,
            str(item.get("rule_id") or ""),
            str(item.get("message") or ""),
        ),
    )


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"blockers": 0, "errors": 0, "warnings": 0, "infos": 0}
    for finding in findings:
        severity = finding.get("severity")
        if severity == "blocker":
            summary["blockers"] += 1
        elif severity == "error":
            summary["errors"] += 1
        elif severity == "warning":
            summary["warnings"] += 1
        else:
            summary["infos"] += 1
    return summary
