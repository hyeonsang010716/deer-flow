"""DeerFlow skill을 위한 네이티브 결정적(deterministic) 스캔.

``scan_archive_preflight()``와 ``scan_skill_dir()``는 입력에 대해 순수한 동기 함수이므로,
async 호출자는 event loop 밖으로 dispatch해야 한다. 정책은 코드 상수 하나다 — ``CRITICAL``은
차단, 나머지는 경고 — 이를 ``enforce_static_scan()``이 적용하며, ``skill_scan.enabled``
kill switch도 함께 존중한다. rule spec은 이를 매칭하는 analyzer 옆에 두어 rule을 한곳에서
작성하고 읽고 테스트할 수 있게 한다.
"""

from __future__ import annotations

import ast
import io
import logging
import posixpath
import re
import stat
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from deerflow.skills.package_paths import is_eval_fixture_skill_md
from deerflow.skills.skillscan.models import (
    FindingSeverity,
    RuleSpec,
    ScanResult,
    SecurityFinding,
    StaticScanBlockedError,
    StaticScannerError,
)

logger = logging.getLogger(__name__)

MAX_TOTAL_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024

_BLOCK_SEVERITY = "CRITICAL"
_NESTED_ZIP_PEEK_MEMBER_LIMIT = 256
_MAX_ARCHIVE_MEMBERS = 4096

_SPECS = [
    RuleSpec("package-path-traversal", "CRITICAL", "Archive member path traverses outside the skill root.", "Remove parent-directory traversal from the package path."),
    RuleSpec("package-absolute-path", "CRITICAL", "Archive member path is absolute.", "Use relative paths inside the skill archive."),
    RuleSpec(
        "package-ads-stream-name",
        "CRITICAL",
        "Archive member path contains a colon, which on Windows/NTFS addresses an alternate data stream hidden from directory listing.",
        "Remove colons from archive member paths.",
    ),
    RuleSpec("package-symlink", "HIGH", "Package contains a symlink entry.", "Remove symlinks from the skill package."),
    RuleSpec("package-nested-skill-md", "CRITICAL", "Package contains a nested SKILL.md file.", "Keep exactly one SKILL.md at the skill root."),
    RuleSpec("package-oversized-total", "CRITICAL", "Package total uncompressed size exceeds the limit.", "Remove large files or split assets out of the skill package."),
    RuleSpec("package-too-many-members", "CRITICAL", "Package contains more members than the allowed limit.", "Reduce the number of files in the skill package."),
    RuleSpec("package-oversized-file", "CRITICAL", "Package contains a file that exceeds the per-file size limit.", "Remove or shrink the oversized file."),
    RuleSpec("package-executable-binary", "CRITICAL", "Package contains an executable binary.", "Remove binary executables from the skill package."),
    RuleSpec("package-nested-archive", "HIGH", "Package contains a nested archive file.", "Unpack and review nested archives before packaging the skill."),
    RuleSpec("package-hidden-sensitive-file", "HIGH", "Package contains a hidden sensitive file.", "Remove hidden credential or package-manager config files."),
    RuleSpec("package-git-directory", "MEDIUM", "Package contains a .git directory.", "Package only source files needed by the skill, excluding repository metadata."),
    RuleSpec("secret-private-key", "CRITICAL", "Private key material is embedded in skill content.", "Move private keys to a managed secret store and remove them from the skill."),
    RuleSpec("secret-cloud-token", "CRITICAL", "High-confidence cloud or API token is embedded in skill content.", "Move tokens to environment variables or a secret store."),
    RuleSpec("secret-env-assignment", "HIGH", "Secret-like assignment contains a non-placeholder value.", "Replace hardcoded credentials with documented runtime configuration."),
    RuleSpec("declaration-prompt-override", "HIGH", "SKILL.md contains a prompt override phrase.", "Rephrase examples so they describe unsafe text instead of instructing the agent to follow it."),
    RuleSpec("declaration-sensitive-capability", "HIGH", "SKILL.md declares a sensitive capability.", "Make the capability explicit, narrow, and justified, or remove it."),
    RuleSpec("declaration-sensitive-path", "HIGH", "SKILL.md references sensitive host or credential paths.", "Remove references to sensitive host paths unless they are harmless documentation."),
    RuleSpec("declaration-external-endpoint", "MEDIUM", "SKILL.md declares an external network endpoint.", "Document why the endpoint is needed and prefer HTTPS."),
    RuleSpec("python-dynamic-exec", "CRITICAL", "Python dynamic code execution primitive is used in a skill file.", "Remove dynamic execution and replace it with explicit typed logic."),
    RuleSpec("python-shell-exec", "CRITICAL", "Python shell execution primitive is used in a skill file.", "Use subprocess with a fixed argument list and shell=False, or remove shell execution."),
    RuleSpec("python-sensitive-exfil", "CRITICAL", "Python code reads a sensitive path and uses an outbound network sink in the same file.", "Remove the sensitive read or network sink, and keep credential access outside skills."),
    RuleSpec("python-env-dump-exfil", "CRITICAL", "Python code reads the process environment in bulk and uses an outbound network sink in the same file.", "Avoid bulk environment reads and never send environment data over the network."),
    RuleSpec("python-reverse-shell", "CRITICAL", "Python code matches a reverse-shell shape.", "Remove reverse-shell behavior from the skill."),
    RuleSpec("python-dynamic-import", "HIGH", "Python dynamically imports a non-literal module.", "Use explicit imports or a constrained allowlist."),
    RuleSpec("python-subprocess", "HIGH", "Python invokes subprocess without shell=True.", "Review subprocess usage and keep arguments fixed and minimal."),
    RuleSpec("python-sensitive-path-read", "HIGH", "Python reads a sensitive path.", "Remove sensitive host-path access from the skill."),
    RuleSpec("python-unsafe-deserialization", "MEDIUM", "Python uses unsafe deserialization.", "Use safe loaders or trusted typed formats."),
    RuleSpec("shell-reverse-shell", "CRITICAL", "Shell script contains a reverse-shell idiom.", "Remove reverse-shell behavior from the skill."),
    RuleSpec("shell-reverse-shell-heuristic", "HIGH", "Shell script resembles a reverse-shell idiom.", "Confirm this is not reverse-shell behavior; unmistakable reverse-shell signals are blocked outright."),
    RuleSpec("shell-sensitive-exfil", "CRITICAL", "Shell script reads sensitive paths and sends data over the network.", "Remove sensitive reads or outbound transfer commands."),
    RuleSpec("shell-curl-pipe-shell", "HIGH", "Shell script pipes remote content into a shell.", "Download, verify, and execute reviewed code explicitly instead."),
    RuleSpec("shell-destructive-command", "HIGH", "Shell script contains an unmistakably destructive command.", "Remove destructive commands from skill scripts."),
    RuleSpec("shell-env-dump", "MEDIUM", "Shell script dumps the environment.", "Avoid bulk environment dumps in skills."),
    RuleSpec("network-cloud-metadata", "CRITICAL", "Skill content references a cloud metadata service.", "Remove cloud metadata access from the skill."),
    RuleSpec("resource-fork-bomb", "CRITICAL", "Skill content contains a fork-bomb pattern.", "Remove resource-exhaustion payloads."),
    RuleSpec("network-cleartext-http", "MEDIUM", "Skill content references a non-local cleartext HTTP endpoint.", "Use HTTPS or document why cleartext local development is required."),
    RuleSpec("network-local-http", "LOW", "Skill content references a local HTTP endpoint.", "Confirm the local endpoint is expected for this skill."),
]

RULES: dict[str, RuleSpec] = {spec.rule_id: spec for spec in _SPECS}

_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".7z",
    ".rar",
    ".whl",
)
_HIDDEN_SENSITIVE_FILES = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "config",
}
_PLACEHOLDER_VALUES = {"", "x", "xx", "xxx", "xxxx", "changeme", "change-me", "example", "placeholder", "test", "dummy", "your-key", "<your-key>"}
_SENSITIVE_PATH_RE = re.compile(r"(~/.ssh|/etc/passwd|/etc/shadow|/var/run/docker\.sock|docker\.sock|169\.254\.169\.254)")
_EXTERNAL_HTTP_RE = re.compile(r"http://([A-Za-z0-9.-]+)(?::\d+)?(?:/|\b)")
_URL_RE = re.compile(r"https?://[^\s)'\"<>]+")
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
# 재귀 플래그가 붙은 `rm`(순서/조합 무관, --no-preserve-root 선택)이 파일시스템 root,
# 와일드카드, 또는 시스템 root 디렉터리 전체를 대상으로 하는 경우.
# ``/tmp/scratch``나 ``/home/user/project`` 같은 하위 경로는 플래그하지 않는다.
_DESTRUCTIVE_RM_RE = (
    r"\brm\s+(?:-\S+\s+|--no-preserve-root\s+)*-\S*[rR]\S*\s+"
    r"(?:-\S+\s+|--no-preserve-root\s+)*"
    r"/(?:\*|\s|$|(?:bin|boot|dev|etc|home|lib|lib64|opt|proc|root|run|sbin|srv|sys|usr|var)(?:/\*?)?(?:\s|$))"
)


def skill_scan_enabled(app_config: Any | None = None) -> bool:
    if app_config is None:
        try:
            from deerflow.config import get_app_config

            app_config = get_app_config()
        except Exception:
            app_config = None
    skill_scan_config = getattr(app_config, "skill_scan", None)
    if skill_scan_config is not None and hasattr(skill_scan_config, "enabled"):
        return bool(skill_scan_config.enabled)
    return True


def format_static_findings(findings: list[SecurityFinding]) -> str:
    parts = []
    for finding in findings:
        location = finding["file"] or "<archive>"
        if finding["line"] is not None:
            location = f"{location}:{finding['line']}"
        parts.append(f"{finding['rule_id']} ({finding['severity']}) at {location}: {finding['message']} Remediation: {finding['remediation']}")
    return "; ".join(parts)


def enforce_static_scan(
    skill_dir: Path,
    *,
    skill_name: str | None = None,
    app_config: Any | None = None,
) -> list[SecurityFinding]:
    if not skill_scan_enabled(app_config):
        return []

    result = scan_skill_dir(Path(skill_dir))
    blocked = [finding for finding in result["findings"] if finding["severity"] == _BLOCK_SEVERITY]
    if blocked:
        raise StaticScanBlockedError(
            blocked,
            skill_name=skill_name,
            message=f"Static security scan blocked skill '{skill_name}': {format_static_findings(blocked)}" if skill_name else f"Static security scan blocked skill content: {format_static_findings(blocked)}",
        )
    if result["scanner_errors"]:
        logger.warning("SkillScan analyzer errors for %s: %s", skill_name or skill_dir, "; ".join(result["scanner_errors"]))
    warnings = [finding for finding in result["findings"] if finding["severity"] != _BLOCK_SEVERITY]
    if warnings:
        logger.warning("SkillScan warning findings for %s: %s", skill_name or skill_dir, format_static_findings(warnings))
    return [dict(finding) for finding in result["findings"]]  # type: ignore[misc]


def scan_archive_preflight(archive_path: Path) -> ScanResult:
    findings: list[SecurityFinding] = []
    scanner_errors: list[str] = []
    total_size = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                # 아래의 member별 읽기 전에 조기 중단한다. 총 크기가 작더라도 member 수가
                # 지나치게 많으면 그 자체로 DoS 벡터가 된다.
                finding = _finding("package-too-many-members", file=None, evidence=f"{len(members)} members")
                return _scan_result([finding], scanner_errors)
            for info in members:
                normalized = _normalize_archive_name(info.filename)
                findings.extend(_scan_archive_member_metadata(info, normalized))
                if info.is_dir():
                    continue
                total_size += max(info.file_size, 0)
                if info.file_size > MAX_FILE_BYTES:
                    findings.append(_finding("package-oversized-file", file=normalized, evidence=f"{info.file_size} bytes"))
                if _is_hidden_sensitive_path(normalized):
                    findings.append(_finding("package-hidden-sensitive-file", file=normalized, evidence=Path(normalized).name))
                if ".git" in PurePosixPath(normalized).parts:
                    findings.append(_finding("package-git-directory", file=normalized, evidence=".git"))
                if _is_symlink_member(info):
                    continue
                try:
                    with zf.open(info) as member:
                        prefix = member.read(8)
                except Exception as e:
                    scanner_errors.append(f"{normalized}: failed to read archive member prefix: {e}")
                    continue
                if _is_executable_binary(prefix):
                    findings.append(_finding("package-executable-binary", file=normalized, evidence=_binary_magic_evidence(prefix)))
                if _is_nested_archive_name(normalized) or _looks_like_archive(prefix):
                    findings.append(_nested_archive_finding(normalized, prefix, lambda: _read_archive_member(zf, info), scanner_errors))
            if total_size > MAX_TOTAL_ARCHIVE_BYTES:
                findings.append(_finding("package-oversized-total", file=None, evidence=f"{total_size} bytes"))
    except (zipfile.BadZipFile, OSError) as e:
        raise StaticScannerError(f"failed to read skill archive: {e}") from e

    return _scan_result(_dedupe(findings), scanner_errors)


def scan_skill_dir(skill_dir: Path) -> ScanResult:
    root = Path(skill_dir)
    if not root.is_dir():
        raise StaticScannerError(f"skill_dir is not a directory: {root}")

    findings: list[SecurityFinding] = []
    scanner_errors: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        rel_path = _relative_file(path, root)
        try:
            file_bytes = path.read_bytes()
        except OSError as e:
            scanner_errors.append(f"{rel_path}: failed to read file: {e}")
            continue

        findings.extend(_scan_file_package_properties(rel_path, file_bytes, path.stat().st_size))
        text = _decode_text_for_analysis(file_bytes)
        if text is None:
            continue

        try:
            findings.extend(_scan_text_file(rel_path, text))
        except Exception as e:
            scanner_errors.append(f"{rel_path}: analyzer failed: {e}")
            logger.warning("SkillScan analyzer failed for %s", rel_path, exc_info=True)

    return _scan_result(_dedupe(findings), scanner_errors)


def _scan_archive_member_metadata(info: zipfile.ZipInfo, normalized: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    if _archive_member_is_absolute(info.filename):
        findings.append(_finding("package-absolute-path", file=normalized, evidence=info.filename))
    elif _archive_member_traverses(info.filename):
        findings.append(_finding("package-path-traversal", file=normalized, evidence=info.filename))
    elif _archive_member_has_colon(info.filename):
        findings.append(_finding("package-ads-stream-name", file=normalized, evidence=info.filename))
    if _is_symlink_member(info):
        findings.append(_finding("package-symlink", file=normalized, evidence=info.filename))
    parts = PurePosixPath(normalized).parts
    if parts and parts[-1] == "SKILL.md" and len(parts) > 2 and not is_eval_fixture_skill_md(PurePosixPath(normalized)):
        findings.append(_finding("package-nested-skill-md", file=normalized, evidence=normalized))
    return findings


def _scan_file_package_properties(rel_path: str, file_bytes: bytes, file_size: int) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    path = PurePosixPath(rel_path)
    if path.name == "SKILL.md" and len(path.parts) > 1 and not is_eval_fixture_skill_md(path):
        findings.append(_finding("package-nested-skill-md", file=rel_path, evidence=rel_path))
    if file_size > MAX_FILE_BYTES:
        findings.append(_finding("package-oversized-file", file=rel_path, evidence=f"{file_size} bytes"))
    if _is_hidden_sensitive_path(rel_path):
        findings.append(_finding("package-hidden-sensitive-file", file=rel_path, evidence=path.name))
    if ".git" in path.parts:
        findings.append(_finding("package-git-directory", file=rel_path, evidence=".git"))
    if _is_nested_archive_name(rel_path) or _looks_like_archive(file_bytes):
        findings.append(_nested_archive_finding(rel_path, file_bytes[:8], lambda: file_bytes, []))
    if _is_executable_binary(file_bytes[:8]):
        findings.append(_finding("package-executable-binary", file=rel_path, evidence=_binary_magic_evidence(file_bytes[:8])))
    return findings


def _scan_text_file(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    findings.extend(_scan_secrets(rel_path, text))
    if PurePosixPath(rel_path).name == "SKILL.md":
        findings.extend(_scan_declaration(rel_path, text))
    if _is_python_path(rel_path, text):
        findings.extend(_scan_python(rel_path, text))
    if _is_shell_path(rel_path, text):
        findings.extend(_scan_shell(rel_path, text))
    findings.extend(_scan_network_and_resource(rel_path, text))
    return findings


def _scan_secrets(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    private_key = re.search(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", text)
    if private_key:
        findings.append(_finding_from_match("secret-private-key", rel_path, text, private_key))

    token_patterns = [
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
        r"\bsk-[A-Za-z0-9]{20,}\b",
    ]
    for pattern in token_patterns:
        match = re.search(pattern, text)
        if match and not _looks_like_placeholder(match.group(0)):
            findings.append(_finding_from_match("secret-cloud-token", rel_path, text, match))
            break

    assignment_re = re.compile(r"(?im)\b(token|password|passwd|api[_-]?key|secret|credential)s?\b\s*[:=]\s*[\"']?([^\"'\s#]+)")
    for match in assignment_re.finditer(text):
        value = match.group(2).strip()
        if not _looks_like_placeholder(value):
            findings.append(_finding_from_match("secret-env-assignment", rel_path, text, match))
            break
    return findings


def _scan_declaration(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    prompt_re = re.compile(r"(?i)\b(ignore|disregard)\s+(all\s+)?(previous|prior)\s+instructions\b|\boverride\s+(the\s+)?(system|developer)\s+instructions\b")
    if match := prompt_re.search(text):
        findings.append(_finding_from_match("declaration-prompt-override", rel_path, text, match))

    capability_re = re.compile(r"(?i)(execute\s+(arbitrary\s+)?commands?|shell\s+commands?|credential\s+access|read\s+secrets?|arbitrary\s+network|network\s+egress)")
    if match := capability_re.search(text):
        findings.append(_finding_from_match("declaration-sensitive-capability", rel_path, text, match))

    if match := _SENSITIVE_PATH_RE.search(text):
        findings.append(_finding_from_match("declaration-sensitive-path", rel_path, text, match))

    for match in _URL_RE.finditer(text):
        if match.group(0).startswith("http://"):
            host = _http_host(match.group(0))
            if host and host not in _LOCAL_HTTP_HOSTS:
                findings.append(_finding_from_match("declaration-external-endpoint", rel_path, text, match))
                break
    return findings


def _scan_python(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return findings

    aliases = _collect_python_aliases(tree)
    has_sensitive_read = False
    has_env_dump = False
    has_network_sink = False
    sensitive_node: ast.AST | None = None
    env_node: ast.AST | None = None
    network_node: ast.AST | None = None
    reverse_shell_parts: set[str] = set()
    reverse_shell_node: ast.AST | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _SENSITIVE_PATH_RE.search(node.value):
                has_sensitive_read = True
                sensitive_node = sensitive_node or node
            if _is_outbound_url(node.value):
                has_network_sink = True
                network_node = network_node or node

        if isinstance(node, (ast.Attribute, ast.Name)) and _python_name(node, aliases) == "os.environ":
            has_env_dump = True
            env_node = env_node or node

        if not isinstance(node, ast.Call):
            continue

        call_name = _python_call_name(node, aliases)
        if call_name in {"eval", "exec"} or (call_name == "compile" and _compile_mode_is_exec(node)):
            findings.append(_finding_for_node("python-dynamic-exec", rel_path, node, call_name))
        elif call_name in {"os.system", "os.popen"} or (call_name.startswith("subprocess.") and _call_shell_may_be_true(node)):
            findings.append(_finding_for_node("python-shell-exec", rel_path, node, call_name))
        elif call_name.startswith("subprocess."):
            findings.append(_finding_for_node("python-subprocess", rel_path, node, call_name))
        elif call_name == "__import__" or call_name == "importlib.import_module":
            if not node.args or not isinstance(node.args[0], ast.Constant):
                findings.append(_finding_for_node("python-dynamic-import", rel_path, node, call_name))
        elif call_name in {"pickle.load", "pickle.loads"} or (call_name == "yaml.load" and not _yaml_load_uses_safe_loader(node)):
            findings.append(_finding_for_node("python-unsafe-deserialization", rel_path, node, call_name))

        if _call_is_network_sink(call_name):
            has_network_sink = True
            network_node = network_node or node

        if call_name == "os.dup2":
            reverse_shell_parts.add("dup2")
            reverse_shell_node = reverse_shell_node or node
        elif call_name in {"socket.socket", "socket.create_connection"}:
            reverse_shell_parts.add("socket")
        elif call_name.startswith("subprocess.") or call_name in {"os.system", "os.popen"}:
            reverse_shell_parts.add("subprocess")

    if not has_network_sink:
        try:
            if handle_sink := _find_client_handle_sink(tree, rel_path):
                has_network_sink = True
                network_node = network_node or handle_sink
        except RecursionError:
            # AST는 신뢰할 수 없는 입력이다. 악의적으로 깊은 트리가 재귀적 handle 분석
            # 예산을 초과하더라도 위에서 수집한 결정적 finding은 보존한다.
            logger.warning("SkillScan client-handle analysis hit recursion limit for %s", rel_path)

    if {"dup2", "socket", "subprocess"} <= reverse_shell_parts:
        findings.append(_finding_for_node("python-reverse-shell", rel_path, reverse_shell_node, "socket + dup2 + subprocess"))

    if has_sensitive_read and has_network_sink:
        findings.append(_finding_for_node("python-sensitive-exfil", rel_path, sensitive_node or network_node, "sensitive read + network sink"))
    elif has_sensitive_read:
        findings.append(_finding_for_node("python-sensitive-path-read", rel_path, sensitive_node, "sensitive path read"))
    if has_env_dump and has_network_sink:
        findings.append(_finding_for_node("python-env-dump-exfil", rel_path, env_node or network_node, "environment dump + network sink"))
    return findings


def _scan_shell(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    # 명백한 reverse-shell 신호는 hard-block한다. 더 약한 관용구(bash -i, mkfifo)는
    # 정상적인 스크립트에도 등장하므로 경고만 내고 LLM으로 넘긴다.
    if match := re.search(r"(/dev/tcp/|nc\s+-e\b)", text):
        findings.append(_finding_from_match("shell-reverse-shell", rel_path, text, match))
    if match := re.search(r"(bash\s+-i\b|mkfifo\s+)", text):
        findings.append(_finding_from_match("shell-reverse-shell-heuristic", rel_path, text, match))
    if re.search(r"(/etc/shadow|/etc/passwd)", text) and re.search(r"\b(curl|wget|nc|scp)\b", text):
        findings.append(_finding_for_text("shell-sensitive-exfil", rel_path, text, "/etc"))
    if match := re.search(r"\b(curl|wget)\b[^\n|;]*\|\s*(?:sh|bash)\b", text):
        findings.append(_finding_from_match("shell-curl-pipe-shell", rel_path, text, match))
    if match := re.search(_DESTRUCTIVE_RM_RE + r"|:\(\)\{\s*:\|:&\s*\};:|dd\s+[^#\n]*\bof=/dev/", text):
        findings.append(_finding_from_match("shell-destructive-command", rel_path, text, match))
    if match := re.search(r"\b(env|printenv|export\s+-p)\b", text):
        findings.append(_finding_from_match("shell-env-dump", rel_path, text, match))
    return findings


def _scan_network_and_resource(rel_path: str, text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    if match := re.search(r"(169\.254\.169\.254|metadata\.google\.internal)", text):
        findings.append(_finding_from_match("network-cloud-metadata", rel_path, text, match))
    if match := re.search(r":\(\)\{\s*:\|:&\s*\};:", text):
        findings.append(_finding_from_match("resource-fork-bomb", rel_path, text, match))
    for match in _EXTERNAL_HTTP_RE.finditer(text):
        host = match.group(1)
        if host in _LOCAL_HTTP_HOSTS or host.startswith("10.") or host.startswith("192.168.") or re.match(r"172\.(1[6-9]|2\d|3[01])\.", host):
            findings.append(_finding_from_match("network-local-http", rel_path, text, match))
        else:
            findings.append(_finding_from_match("network-cleartext-http", rel_path, text, match))
        break
    return findings


def _finding(rule_id: str, *, file: str | None, evidence: str | None, line: int | None = None, severity: FindingSeverity | None = None) -> SecurityFinding:
    spec = RULES[rule_id]
    if evidence is not None and rule_id.startswith("secret-"):
        evidence = _redact_secret_evidence(evidence)
    return {
        "rule_id": rule_id,
        "severity": severity or spec.severity,
        "file": file,
        "line": line,
        "message": spec.message,
        "remediation": spec.remediation,
        "evidence": evidence,
    }


def _finding_from_match(rule_id: str, rel_path: str, text: str, match: re.Match[str]) -> SecurityFinding:
    return _finding(rule_id, file=rel_path, line=_line_number(text, match.start()), evidence=match.group(0))


def _finding_for_text(rule_id: str, rel_path: str, text: str, evidence: str) -> SecurityFinding:
    index = text.find(evidence)
    return _finding(rule_id, file=rel_path, line=_line_number(text, index if index >= 0 else 0), evidence=evidence)


def _finding_for_node(rule_id: str, rel_path: str, node: ast.AST | None, evidence: str) -> SecurityFinding:
    return _finding(rule_id, file=rel_path, line=getattr(node, "lineno", 1), evidence=evidence)


def _nested_archive_finding(rel_path: str, prefix: bytes, read_data, scanner_errors: list[str]) -> SecurityFinding:
    name = PurePosixPath(rel_path).name
    if prefix.startswith(b"PK\x03\x04"):
        try:
            data = read_data()
        except Exception as e:
            scanner_errors.append(f"{rel_path}: failed to read nested archive for inspection: {e}")
        else:
            if data is not None and _nested_zip_contains_executable(data):
                return _finding("package-nested-archive", file=rel_path, evidence=f"{name}: contains an executable binary member", severity="CRITICAL")
    return _finding("package-nested-archive", file=rel_path, evidence=name)


def _nested_zip_contains_executable(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as nested:
            for info in nested.infolist()[:_NESTED_ZIP_PEEK_MEMBER_LIMIT]:
                if info.is_dir():
                    continue
                try:
                    with nested.open(info) as member:
                        if _is_executable_binary(member.read(8)):
                            return True
                except Exception:
                    continue
    except (zipfile.BadZipFile, OSError):
        return False
    return False


def _read_archive_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes | None:
    if info.file_size > MAX_FILE_BYTES:
        return None
    with zf.open(info) as member:
        return member.read(MAX_FILE_BYTES + 1)


def _redact_secret_evidence(value: str) -> str:
    # 값을 통째로 버린다. rule_id가 이미 secret 범주를 알려주고, 접두사를 조금이라도
    # 남기면(예: value[:6]) 실제 token 바이트가 Gateway 응답과 LLM context로 흘러가는
    # finding에 새어 나간다.
    return "[redacted]"


def _scan_result(findings: list[SecurityFinding], scanner_errors: list[str]) -> ScanResult:
    blocked = any(finding["severity"] == _BLOCK_SEVERITY for finding in findings)
    return {"findings": findings, "blocked": blocked, "scanner_errors": scanner_errors}


def _dedupe(findings: Iterable[SecurityFinding]) -> list[SecurityFinding]:
    seen: set[tuple[str, str | None, int | None]] = set()
    deduped: list[SecurityFinding] = []
    for finding in findings:
        key = (finding["rule_id"], finding["file"], finding["line"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _line_number(text: str, index: int) -> int:
    return text[: max(index, 0)].count("\n") + 1


def _normalize_archive_name(name: str) -> str:
    return posixpath.normpath(name.replace("\\", "/")).removeprefix("./")


def _archive_member_is_absolute(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return normalized.startswith("/") or PurePosixPath(normalized).is_absolute() or PureWindowsPath(name).is_absolute()


def _archive_member_traverses(name: str) -> bool:
    return ".." in PurePosixPath(name.replace("\\", "/")).parts


def _archive_member_has_colon(name: str) -> bool:
    # 상대 archive member 경로에서 콜론은 정당하게 쓰일 일이 없다(zip 엔트리는 ``/``
    # 구분자를 쓰고, 실제 Windows 드라이브 접두사는 이미 ``_archive_member_is_absolute``가
    # 잡는다). Windows/NTFS에서 경로 중간의 콜론은 바로 앞 경로 구성요소의 Alternate Data
    # Stream을 가리키며(예: ``scripts/run.sh:hidden.txt``는 형제 파일을 만드는 대신
    # ``run.sh``에 숨겨진 내용을 붙인다), 그 스트림은 디렉터리 목록 기반 스캔에 보이지
    # 않는다. "안전한" 콜론 위치를 allow-list하려 하지 말고 통째로 거부한다.
    return ":" in name


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _relative_file(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_hidden_sensitive_path(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    if ".aws" in parts and parts[-1] == "credentials":
        return True
    if ".git" in parts and parts[-1] == "config":
        return True
    return parts[-1] in _HIDDEN_SENSITIVE_FILES and (parts[-1].startswith(".") or any(token in parts[-1].lower() for token in ("credential", "npmrc", "pypirc", "netrc")))


def _is_nested_archive_name(rel_path: str) -> bool:
    lower = rel_path.lower()
    return any(lower.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _looks_like_archive(file_bytes: bytes) -> bool:
    return file_bytes.startswith(b"PK\x03\x04") or file_bytes.startswith(b"\x1f\x8b") or file_bytes.startswith(b"7z\xbc\xaf\x27\x1c")


def _is_executable_binary(prefix: bytes) -> bool:
    return prefix.startswith(b"\x7fELF") or prefix.startswith(b"MZ") or prefix.startswith((b"\xfe\xed\xfa", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"))


def _binary_magic_evidence(prefix: bytes) -> str:
    if prefix.startswith(b"\x7fELF"):
        return "ELF"
    if prefix.startswith(b"MZ"):
        return "PE"
    return "Mach-O"


def _decode_text_for_analysis(file_bytes: bytes) -> str | None:
    # 바이너리는 아래의 NUL 검사와 디코드 실패로 걸러지므로, NUL이 없고 UTF-8로 디코드되는
    # 파일은 확장자와 무관하게 모두 분석한다.
    if b"\x00" in file_bytes[:4096]:
        return None
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_python_path(rel_path: str, text: str) -> bool:
    return PurePosixPath(rel_path).suffix.lower() == ".py" or text.startswith("#!") and "python" in text.splitlines()[0].lower()


def _is_shell_path(rel_path: str, text: str) -> bool:
    suffix = PurePosixPath(rel_path).suffix.lower()
    return suffix in {".sh", ".bash"} or text.startswith("#!") and any(shell in text.splitlines()[0].lower() for shell in ("sh", "bash", "zsh"))


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    if normalized in _PLACEHOLDER_VALUES:
        return True
    return normalized.startswith("<") or normalized.startswith("${") or "your" in normalized or "example" in normalized


def _http_host(url: str) -> str | None:
    match = re.match(r"https?://\[?([^]/:]+)", url)
    return match.group(1) if match else None


def _is_outbound_url(value: str) -> bool:
    return bool(value.startswith(("http://", "https://")) and (_http_host(value) or "") not in _LOCAL_HTTP_HOSTS)


def _collect_python_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _python_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _python_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _python_import_name(node: ast.AST, aliases: dict[str, str]) -> str:
    """scope 내 import map으로 증명된 이름만 해석한다."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, "")
    if isinstance(node, ast.Attribute):
        base = _python_import_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _python_call_name(node: ast.Call, aliases: dict[str, str]) -> str:
    return _python_name(node.func, aliases)


def _compile_mode_is_exec(node: ast.Call) -> bool:
    if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
        return node.args[2].value == "exec"
    return any(keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "exec" for keyword in node.keywords)


def _call_shell_may_be_true(node: ast.Call) -> bool:
    # 모호하면 fail closed한다. ``shell=`` 키워드는 정적으로 증명 가능한 리터럴 ``False``일
    # 때만 안전하다. 그 키워드에 붙은 다른 값, 리터럴 ``True``, 다른 리터럴, 혹은 변수나
    # 함수 호출 같은 비리터럴 표현식(``shell=shell_flag``, ``shell=bool(1)``)은 정적 분석으로
    # 안전을 증명할 수 없으므로, 차단하지 않는 ``python-subprocess`` 분류로 조용히 흘려보내지
    # 않고 명시적 ``shell=True``와 동일하게 취급한다.
    for keyword in node.keywords:
        if keyword.arg is None:
            # ``**mapping`` / ``**kwargs`` 언패킹은 AST에서 ``arg is None``인 keyword로
            # 표현된다. 그 mapping의 내용(그리고 ``shell`` 키가 있는지 여부)은 정적
            # 분석으로 알 수 없으므로, 변수/표현식 shell= 값과 마찬가지로 fail closed한다.
            # 의도적이고 문서화된 과차단이다. ``shell`` 키가 없는 무해한 kwargs 언패킹도
            # 차단되는데, 대안(언패킹된 mapping의 내용을 들여다보는 것)이 정적으로는 일반적
            # 으로 불가능하기 때문이다.
            return True
        if keyword.arg == "shell":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False)
    return False


def _call_is_network_sink(call_name: str) -> bool:
    return call_name in {
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "requests.head",
        "requests.options",
        "requests.request",
        "httpx.get",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.delete",
        "httpx.head",
        "httpx.options",
        "httpx.request",
        "httpx.stream",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
        "socket.socket",
        "socket.create_connection",
    }


# 인스턴스 client는 생성과 송신을 분리한다. 생성자는 I/O를 하지 않고 outbound 호출은 변수에
# 대한 attribute 호출이라, 어느 한 문장만으로는 call-name sink가 되지 않는다. 그래서 이 신호는
# 확신도가 높은 최소 연결고리만 따라간다: 알려진 생성자 -> 단순 이름/alias -> 같은 lexical
# scope에서 그 생성자가 지원하는 직접 method 호출. 중첩 scope는 안정적인 import alias만
# 물려받고 client handle은 절대 물려받지 않는다. comprehension, walrus를 포함한 문장, 주석
# (annotation), 복잡한 binding target 안의 실행 표현식은 의도적으로 이 신호에서 아무 finding도
# 만들지 않는다. 그렇게 건너뛴 구조가 바인딩할 수 있는 이름은 모두 무효화해서, 낡은 상태가
# finding을 만들지 못하게 한다.
#
# 이름이 아니라 값으로 도달하는 handle — attribute, 컨테이너 항목, factory 반환값, 지역적으로
# alias된 생성자, 동적 `getattr` — 과 `name.method(...)` 이외의 형태로 호출되는 sink는 구조상
# 이 연결고리 밖이다. 이들을 따라가는 것은 값 추적이고, RFC #2634은 그것을 Phase 5 범위 밖으로
# 둔다. 이는 누락이 아니라 범위 결정이다. issue #4296이 이들을 열거하고
# `test_python_declared_false_negatives_stay_unreported`가 각각을 고정하므로, 모델을 넓히거나
# 좁히려면 동작을 조용히 바꾸는 대신 그 테스트를 바꿔야 한다.
#
# `if True:`가 만능 우회가 되지 않도록 compound body는 여전히 격리된 진입 상태 복사본에서
# 순회하되, 모호한 binding은 병합하지 않고 버린다. 모든 AST 방문과 복사된 scope 진입은 결정적
# 작업 예산을 소모하며, sink를 하나 찾는 즉시 순회를 멈춘다. 신뢰할 수 없는 소스에 대한
# 분기 복사 비용을 이렇게 제한한다.


@dataclass(frozen=True)
class _ClientSpec:
    methods: frozenset[str]
    sync_context: bool = False
    async_context: bool = False


# `getresponse()` 같은 응답 전용 연산은 제외한다. 이 신호는 outbound I/O를 필요로 한다.
_PYTHON_CLIENT_SPECS = {
    "http.client.HTTPConnection": _ClientSpec(frozenset({"request", "connect", "send"})),
    "http.client.HTTPSConnection": _ClientSpec(frozenset({"request", "connect", "send"})),
    "requests.Session": _ClientSpec(frozenset({"request", "get", "post", "put", "patch", "delete", "head", "options", "send"}), sync_context=True),
    "urllib3.PoolManager": _ClientSpec(frozenset({"request", "urlopen"}), sync_context=True),
    "aiohttp.ClientSession": _ClientSpec(frozenset({"request", "get", "post", "put", "patch", "delete", "head", "options"}), async_context=True),
}
_PYTHON_CLIENT_CONSTRUCTORS = frozenset(_PYTHON_CLIENT_SPECS)
_PYTHON_CLIENT_SINK_METHODS = frozenset().union(*(spec.methods for spec in _PYTHON_CLIENT_SPECS.values()))
_PYTHON_CLIENT_ANALYSIS_BUDGET = 100_000
_PYTHON_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_PYTHON_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_PYTHON_MATCH_CAPTURE_NODES = (ast.MatchAs, ast.MatchStar, ast.MatchMapping)
# 모든 부분이 실행되지는 않거나, 실행 횟수를 알 수 없는 문장들. 이들의 body는 복사본에서
# 분석하고, 이들이 바인딩한 이름은 이후 모두 버린다.
_PYTHON_BRANCHING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.Match)


@dataclass
class _ClientScope:
    handles: dict[str, str]
    aliases: dict[str, str]
    unstable_aliases: frozenset[str] = frozenset()

    def copy_without(self, analysis: _ClientAnalysis, names: set[str] | None = None) -> _ClientScope:
        names = names or set()
        analysis.charge(len(self.handles) + len(self.aliases) + 1)
        return _ClientScope(
            handles={name: constructor for name, constructor in self.handles.items() if name not in names},
            aliases={name: target for name, target in self.aliases.items() if name not in names},
            unstable_aliases=self.unstable_aliases,
        )

    def aliases_only(self, analysis: _ClientAnalysis, names: set[str] | None = None, unstable_aliases: frozenset[str] = frozenset()) -> _ClientScope:
        names = names or set()
        analysis.charge(len(self.aliases) + 1)
        return _ClientScope(
            handles={},
            aliases={name: target for name, target in self.aliases.items() if name not in names and name not in self.unstable_aliases},
            unstable_aliases=unstable_aliases,
        )


class _ClientAnalysisBudgetExceeded(Exception):
    pass


@dataclass
class _ClientAnalysis:
    remaining: int
    found: ast.AST | None = None

    def charge(self, cost: int = 1) -> None:
        if cost > self.remaining:
            raise _ClientAnalysisBudgetExceeded
        self.remaining -= cost


def _find_client_handle_sink(tree: ast.AST, rel_path: str) -> ast.AST | None:
    analysis = _ClientAnalysis(remaining=_PYTHON_CLIENT_ANALYSIS_BUDGET)
    module = _ClientScope(handles={}, aliases={})
    try:
        if isinstance(tree, ast.Module):
            module.unstable_aliases = _client_unstable_aliases(tree.body, analysis)
        _walk_client_scope(tree, module, module, analysis)
    except _ClientAnalysisBudgetExceeded:
        logger.warning("SkillScan client-handle analysis exhausted work budget for %s", rel_path)
    return analysis.found


def _walk_client_statements(body: list[ast.AST], scope: _ClientScope, inherited: _ClientScope, analysis: _ClientAnalysis) -> None:
    """일반 statement를 순회한다. walrus를 포함한 statement는 명시적인 false negative다."""
    for statement in body:
        if analysis.found is not None:
            return
        walrus_names = set() if isinstance(statement, _PYTHON_SCOPE_NODES) else _walrus_target_names(statement, analysis)
        if walrus_names:
            bound_names: set[str] = set()
            declared_names: set[str] = set()
            _collect_client_scope_bindings(statement, bound_names, declared_names, analysis)
            _drop_client_bindings(scope, walrus_names | (bound_names - declared_names))
            continue
        _walk_client_scope(statement, scope, inherited, analysis)


def _walk_client_scope(node: ast.AST, scope: _ClientScope, inherited: _ClientScope, analysis: _ClientAnalysis) -> None:
    """실행 가능한 AST를 문장 순서대로 순회하면서 한 단계짜리 client-handle map을 들고 다닌다."""
    if analysis.found is not None:
        return
    analysis.charge()
    if isinstance(node, ast.Module):
        _walk_client_statements(node.body, scope, inherited, analysis)
        return
    if isinstance(node, _PYTHON_SCOPE_NODES):
        _walk_client_nested_scope(node, scope, inherited, analysis)
        return
    if isinstance(node, _PYTHON_COMPREHENSION_NODES):
        return
    if isinstance(node, ast.NamedExpr):
        _drop_client_bindings(scope, set(_client_assignment_target_names(node.target)))
        return
    if isinstance(node, ast.TypeAlias):
        # PEP 695: `type X = ...`의 값은 lazy하게 평가된다 — import 시점에는 전혀 실행되지
        # 않고, 나중에 `X.__value__`에 접근할 때만 실행된다. 이를 sink 탐색으로 순회하면
        # 일어날 수 없는 egress를 보고하고 무해한 파일을 hard-block하게 되는데, 이는
        # annotation을 순회하지 않는 이유와 같다(_client_scope_prelude 참고). 값 안으로
        # 내려가는 대신 여기서 바인딩되는 이름을 무효화하므로, 값을 건너뛰어도 낡은 handle이
        # 남지 않는다.
        _drop_client_bindings(scope, set(_client_assignment_target_names(node.name)))
        return
    if isinstance(node, _PYTHON_BRANCHING_NODES):
        _walk_client_branching(node, scope, inherited, analysis)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        _bind_client_import(node, scope)
        return
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        if node.value is not None:
            _walk_client_scope(node.value, scope, inherited, analysis)
        if analysis.found is not None:
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        _rebind_client_scope(targets, node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None, scope)
        return
    if isinstance(node, (ast.With, ast.AsyncWith)):
        bound_names = {name for item in node.items if item.optional_vars is not None for name in _client_assignment_target_names(item.optional_vars)}
        for item in node.items:
            _walk_client_scope(item.context_expr, scope, inherited, analysis)
            if analysis.found is not None:
                return
            constructor = _client_constructor_from_value(item.context_expr, scope)
            if item.optional_vars is not None:
                _drop_client_bindings(scope, set(_client_assignment_target_names(item.optional_vars)))
            if constructor:
                spec = _PYTHON_CLIENT_SPECS[constructor]
                supported = spec.async_context if isinstance(node, ast.AsyncWith) else spec.sync_context
                if not supported:
                    _drop_client_bindings(scope, bound_names)
                    return
                if isinstance(item.optional_vars, ast.Name):
                    scope.handles[item.optional_vars.id] = constructor
        _walk_client_statements(node.body, scope, inherited, analysis)
        return
    if isinstance(node, ast.Delete):
        _drop_client_bindings(scope, {name for target in node.targets for name in _client_assignment_target_names(target)})
        return
    if isinstance(node, ast.Call):
        if _call_is_client_handle_sink(node, scope.handles):
            analysis.found = node
            return
    for child in ast.iter_child_nodes(node):
        _walk_client_scope(child, scope, inherited, analysis)
        if analysis.found is not None:
            return


def _walk_client_branching(node: ast.AST, scope: _ClientScope, inherited: _ClientScope, analysis: _ClientAnalysis) -> None:
    """어느 부분이 어떤 순서로 실행되는지 판단하지 않고 compound 문장을 분석한다.

    이 문장이 바인딩할 수 있는 모든 이름은 body를 순회하기 *전에* 버린다(뒤가 아니다). 뒤에
    버리면, 형제 부분이 이미 이름을 재바인딩한 뒤에 실행되는 부분 — handler가 handle을 교체한
    뒤의 `finally`, 나중의 `except*` 절, 두 번째 루프 반복 — 에서 잘못된 결과가 나오고, 그런
    불일치는 각각 무해한 파일을 `CRITICAL`로 hard-block하는 결과가 된다. 버리는 대신 부분들의
    순서를 따지는 것은 이 신호가 하지 않는 control-flow 인터프리터의 일이다.

    body는 각각 자기 복사본에서 여전히 순회하므로, 한 분기 안의 생성이 형제 분기로 새지
    않으면서도 분기 안의 sink는 여전히 발견된다. 잃는 것은 같은 문장이 호출하면서 동시에
    재바인딩하는 이름이며, 이는 문서화된 false negative다.
    """
    _drop_client_bindings(scope, _branching_bound_names(node, analysis))
    for header in _branching_header_exprs(node):
        walrus_names = _walrus_target_names(header, analysis)
        if walrus_names:
            _drop_client_bindings(scope, walrus_names)
            continue
        _walk_client_scope(header, scope, inherited, analysis)
        if analysis.found is not None:
            return
    for body in _branching_bodies(node):
        branch_scope = scope.copy_without(analysis)
        _walk_client_statements(body, branch_scope, inherited, analysis)
        if analysis.found is not None:
            return


def _branching_header_exprs(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.If):
        return [node.test]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.iter]
    if isinstance(node, ast.While):
        return [node.test]
    if isinstance(node, ast.Match):
        return [node.subject]
    return []  # `try`에는 header가 없다. handler의 타입은 예외가 발생했을 때만 실행된다


def _branching_bodies(node: ast.AST) -> list[list[ast.AST]]:
    if isinstance(node, (ast.Try, ast.TryStar)):
        # handler의 `type` 표현식과 body는 같은 경로에서 실행되므로 복사본 하나를 공유한다.
        handlers = [[*([handler.type] if handler.type is not None else []), *handler.body] for handler in node.handlers]
        return [node.body, *handlers, node.orelse, node.finalbody]
    if isinstance(node, ast.Match):
        return [[*([case.guard] if case.guard is not None else []), *case.body] for case in node.cases]
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.body, node.orelse]
    if isinstance(node, ast.While):
        return [node.body, node.orelse]
    return []


def _branching_bound_names(node: ast.AST, analysis: _ClientAnalysis) -> set[str]:
    """이 문장이 바인딩할 수 있는 모든 이름. loop/handler/capture 대상 자체도 포함한다."""
    names: set[str] = set()
    declared: set[str] = set()
    if isinstance(node, (ast.For, ast.AsyncFor)):
        names.update(_client_assignment_target_names(node.target))
    for body in _branching_bodies(node):
        for statement in body:
            _collect_client_scope_bindings(statement, names, declared, analysis)
    if isinstance(node, (ast.Try, ast.TryStar)):
        names.update(handler.name for handler in node.handlers if handler.name)
    if isinstance(node, ast.Match):
        for case in node.cases:
            _collect_client_scope_bindings(case.pattern, names, declared, analysis)
    return names - declared


def _walrus_target_names(node: ast.AST, analysis: _ClientAnalysis) -> set[str]:
    """모호한 문장 전체를 건너뛸 수 있도록 이 scope의 walrus 대상들을 반환한다."""
    if isinstance(node, _PYTHON_SCOPE_NODES):
        return set()
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        analysis.charge()
        if isinstance(current, ast.NamedExpr):
            found.update(_client_assignment_target_names(current.target))
        for child in ast.iter_child_nodes(current):
            if isinstance(child, _PYTHON_SCOPE_NODES):
                continue
            stack.append(child)
    return found


def _walk_client_nested_scope(node: ast.AST, scope: _ClientScope, inherited: _ClientScope, analysis: _ClientAnalysis) -> None:
    annotation_bindings = {name for annotation in _client_scope_annotations(node) for name in _walrus_target_names(annotation, analysis)}
    _drop_client_bindings(scope, annotation_bindings)
    for expr in _client_scope_prelude(node):
        _walk_client_scope(expr, scope, inherited, analysis)
        if analysis.found is not None:
            return
    body = node.body if isinstance(node.body, list) else [node.body]
    unstable_aliases = _client_unstable_aliases(body, analysis)
    if isinstance(node, ast.ClassDef):
        inner, nested = inherited.aliases_only(analysis, unstable_aliases=unstable_aliases), inherited
    else:
        inner = inherited.aliases_only(analysis, _client_scope_bindings(node, analysis), unstable_aliases)
        nested = inner
    _walk_client_statements(body, inner, nested, analysis)
    if not isinstance(node, ast.Lambda):
        _drop_client_bindings(scope, {node.name})


def _match_capture_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    return [node.name] if node.name else []


def _client_scope_prelude(node: ast.AST) -> list[ast.AST]:
    """scope를 정의하는 문장이 새 scope가 아니라 *바깥* scope에서 평가하는 표현식들:
    decorator, 인자/키워드 기본값, class base/keyword. annotation은 sink 탐색으로 순회하지
    않는다 — runtime이 annotation을 평가하는지는 scope, `from __future__ import annotations`
    여부, Python 버전에 따라 달라지기 때문이다. annotation이 만들 수 있는 binding 효과는 별도로
    무효화하므로, annotation을 건너뛰어도 낡은 handle이 남지 않는다.
    """
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *(keyword.value for keyword in node.keywords)]
    defaults = [default for default in [*node.args.defaults, *node.args.kw_defaults] if default is not None]
    if isinstance(node, ast.Lambda):
        return defaults
    return [*node.decorator_list, *defaults]


def _client_scope_annotations(node: ast.AST) -> list[ast.AST]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    annotations = [arg.annotation for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs] if arg.annotation is not None]
    for extra in (args.vararg, args.kwarg):
        if extra is not None and extra.annotation is not None:
            annotations.append(extra.annotation)
    if node.returns is not None:
        annotations.append(node.returns)
    return annotations


def _client_unstable_aliases(body: list[ast.AST], analysis: _ClientAnalysis) -> frozenset[str]:
    """중첩 scope에 대해 import alias 값이 안정적이지 않은 이름들.

    binding만 보는 prepass다. 표현식의 값이나 경로를 해석하지 않는다. 평범한 binding 하나만
    있어도 같은 이름의 import alias는 상속 불가가 되며, 중복/star import도 마찬가지다. scope의
    body는 건너뛰고, 바깥 scope prelude에 있는 walrus 대상은 무효화한다.
    """
    imported: set[str] = set()
    unstable: set[str] = set()
    saw_star_import = False
    stack = list(reversed(body))
    while stack:
        current = stack.pop()
        analysis.charge()
        if isinstance(current, (ast.Import, ast.ImportFrom)):
            for alias in current.names:
                if alias.name == "*":
                    saw_star_import = True
                    continue
                name = alias.asname or alias.name.split(".")[0]
                if name in imported:
                    unstable.add(name)
                imported.add(name)
            continue
        if isinstance(current, _PYTHON_SCOPE_NODES):
            if not isinstance(current, ast.Lambda):
                unstable.add(current.name)
            for expr in [*_client_scope_prelude(current), *_client_scope_annotations(current)]:
                unstable.update(_walrus_target_names(expr, analysis))
            continue
        if isinstance(current, _PYTHON_COMPREHENSION_NODES):
            unstable.update(_walrus_target_names(current, analysis))
            continue
        if isinstance(current, ast.Global | ast.Nonlocal):
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, (ast.Store, ast.Del)):
            unstable.add(current.id)
        elif isinstance(current, ast.ExceptHandler) and current.name:
            unstable.add(current.name)
        elif isinstance(current, _PYTHON_MATCH_CAPTURE_NODES):
            unstable.update(_match_capture_names(current))
        stack.extend(reversed(list(ast.iter_child_nodes(current))))
    if saw_star_import:
        unstable.update(imported)
    return frozenset(unstable)


def _client_scope_bindings(node: ast.AST, analysis: _ClientAnalysis) -> set[str]:
    """함수 scope 전체에서 상속된 생성자 alias를 가리는 이름들."""
    args = node.args
    names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    declared: set[str] = set()
    for statement in node.body if isinstance(node.body, list) else [node.body]:
        _collect_client_scope_bindings(statement, names, declared, analysis)
    return names - declared


def _collect_client_scope_bindings(node: ast.AST, names: set[str], declared: set[str], analysis: _ClientAnalysis) -> None:
    analysis.charge()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)  # 이 문장은 여기서 자기 이름을 바인딩한다. body는 별도 scope다.
        return
    if isinstance(node, ast.Lambda):
        return
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        declared.update(node.names)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        names.add(node.id)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        names.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, _PYTHON_MATCH_CAPTURE_NODES):
        names.update(_match_capture_names(node))
    if isinstance(node, _PYTHON_COMPREHENSION_NODES):
        # 이 표현식은 sink 탐색 대상이 아니지만, walrus 대상은 그것을 포함한 함수의 지역
        # 이름이다. 같은 이름의 상속된 생성자 alias를 제거하면 정의 순서에서 오는 false
        # positive를 막을 수 있다. comprehension이 실제로 실행되는지 여부는 설계상 false
        # negative로 남는다.
        names.update(_walrus_target_names(node, analysis))
        return
    for child in ast.iter_child_nodes(node):
        _collect_client_scope_bindings(child, names, declared, analysis)


def _client_constructor_from_value(value: ast.AST | None, scope: _ClientScope) -> str:
    if isinstance(value, ast.Call):
        called = _python_import_name(value.func, scope.aliases)
        return called if called in _PYTHON_CLIENT_CONSTRUCTORS else ""
    if isinstance(value, ast.Name):
        return scope.handles.get(value.id, "")
    return ""


def _rebind_client_scope(targets: list[ast.AST], value: ast.AST | None, scope: _ClientScope) -> None:
    """binding 하나를 적용한다: 대상을 버린 뒤, 값이 client handle이면 다시 추가한다.

    값은 대상을 버리기 전에 해석하므로 `session = session`과 `s = session`은 handle을 유지한다.
    이 이름-대-이름 전파가 있어야 두 글자짜리 rename만으로 handle이 사라지지 않는다. 전파는 한
    단계에 머무르므로, attribute나 항목을 거쳐 도달하는 handle은 추적하지 않는다.
    """
    constructor = _client_constructor_from_value(value, scope)
    names = {name for target in targets for name in _client_assignment_target_names(target)}
    _drop_client_bindings(scope, names)
    if constructor:
        for target in targets:
            if isinstance(target, ast.Name):
                scope.handles[target.id] = constructor


def _client_assignment_target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _client_assignment_target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return [name for element in target.elts for name in _client_assignment_target_names(element)]
    return []


def _drop_client_bindings(scope: _ClientScope, names: set[str]) -> None:
    for name in names:
        scope.handles.pop(name, None)
        scope.aliases.pop(name, None)


def _bind_client_import(node: ast.Import | ast.ImportFrom, scope: _ClientScope) -> None:
    for alias in node.names:
        if alias.name == "*":
            continue
        name = alias.asname or alias.name.split(".")[0]
        _drop_client_bindings(scope, {name})
        if isinstance(node, ast.Import):
            scope.aliases[name] = alias.name if alias.asname else name
        elif node.module:
            scope.aliases[name] = f"{node.module}.{alias.name}"


def _call_is_client_handle_sink(node: ast.Call, handles: dict[str, str]) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    constructor = handles.get(func.value.id)
    return bool(constructor and func.attr in _PYTHON_CLIENT_SPECS[constructor].methods)


def _yaml_load_uses_safe_loader(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg in {"Loader", "loader"}:
            name = _python_name(keyword.value, {})
            if "SafeLoader" in name:
                return True
    return False
