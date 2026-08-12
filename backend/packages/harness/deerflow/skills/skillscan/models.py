"""DeerFlow SkillScan의 데이터 계약.

``SecurityFinding``의 모든 필드에는 Phase 1 소비자가 있다. 차단 정책은 ``severity``를 읽고,
Gateway 거부 응답과 agent tool 오류, LLM 스캐너 context가 나머지를 읽는다. rule 카테고리와
담당 analyzer는 ``rule_id`` 접두사(``package-``, ``secret-``, ``declaration-``, ``python-``,
``shell-``, ``network-``/``resource-``)에 인코딩되며, 별도 필드로 중복하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

FindingSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class SecurityFinding(TypedDict):
    rule_id: str
    severity: FindingSeverity
    file: str | None
    line: int | None
    message: str
    remediation: str
    evidence: str | None


class ScanResult(TypedDict):
    findings: list[SecurityFinding]
    blocked: bool
    scanner_errors: list[str]


@dataclass(frozen=True)
class RuleSpec:
    """SkillScan rule 하나의 정적 정의. ``remediation``은 여기서 한 번 작성해 finding으로 복사한다."""

    rule_id: str
    severity: FindingSeverity
    message: str
    remediation: str


class StaticScannerError(RuntimeError):
    """SkillScan이 패키지 경계에서 입력을 평가할 수 없을 때 발생한다."""


class StaticScanBlockedError(ValueError):
    """결정적 finding이 skill 쓰기나 설치를 차단할 때 발생한다."""

    findings: list[SecurityFinding]
    skill_name: str | None

    def __init__(self, findings: list[SecurityFinding], *, skill_name: str | None = None, message: str | None = None) -> None:
        self.findings = [dict(finding) for finding in findings]  # type: ignore[list-item]
        self.skill_name = skill_name
        subject = f"skill '{skill_name}'" if skill_name else "skill content"
        super().__init__(message or f"Static security scan blocked {subject}")
