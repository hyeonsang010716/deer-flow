"""큰 도구 출력 preview를 위한 결정적 요약."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

try:
    from defusedxml import ElementTree as SafeET  # type: ignore[import-not-found]
except ImportError:
    SafeET = None  # stdlib로 fallback한다. 에이전트가 요청한 출력이라 위험이 제한적이다.

import yaml

ToolOutputKind = Literal["json", "csv", "tsv", "yaml", "xml", "code", "text", "unknown"]

_KEY_LIMIT = 12
_SCALAR_LIMIT = 6
_TABLE_SAMPLE_ROWS = 50
_TABLE_COLUMN_LIMIT = 18
_TEXT_HEADER_LIMIT = 16
_TEXT_EXCERPT_CHARS = 420
_CODE_IMPORT_LIMIT = 12
_CODE_SYMBOL_LIMIT = 24
_JSON_SHAPE_MAX_DEPTH = 2
_JSON_STRUCTURE_LIMIT = 24
_JSON_STRUCTURE_DEPTH = 4

# synopsis 입력 크기의 상한. 이 임계값을 넘으면 전체 파싱을 건너뛰고 raw head/tail
# 샘플만 낸다. 외부화된 도구 출력이 병적으로 큰 경우(예: 50MB 이상 로그 덤프) 최악의
# 메모리/CPU를 제한하고 XML/YAML entity-expansion을 통한 DoS를 막는다.
_MAX_SYNOPSIS_INPUT_BYTES = 5_000_000

_CODE_HINTS = (
    re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", re.MULTILINE),
    re.compile(r"^\s*(?:class|def|async\s+def|function|export\s+function)\s+[A-Za-z_]\w*", re.MULTILINE),
    # Rust/Java는 더 강한 신호를 요구한다: `use ...;`(끝에 세미콜론),
    # `fn ...(`(괄호), `pub fn ...(`, `public class ...`. 맨 `use <word>`나
    # `fn <word>`는 일반 산문을 오분류한다(예: "use the following …").
    re.compile(r"^\s*(?:package\s+[A-Za-z_][\w.]*|use\s+[A-Za-z_][\w:]*\s*;|pub\s+fn\s+[A-Za-z_]\w*\s*\(|fn\s+[A-Za-z_]\w*\s*\(|public\s+class\s+[A-Za-z_]\w*)", re.MULTILINE),
)


@dataclass(frozen=True)
class ToolOutputSynopsis:
    """큰 도구 출력의 구조화된 preview 데이터."""

    kind: ToolOutputKind
    title: str
    summary: list[str]
    structure: list[str]
    notable_items: list[str]
    sample: str = ""


def build_tool_output_synopsis(content: str, *, tool_name: str = "") -> ToolOutputSynopsis:
    """LLM 없이 *content*의 typed synopsis를 만든다."""
    if content == "":
        return ToolOutputSynopsis(
            kind="unknown",
            title="Empty output",
            summary=["The tool returned an empty string."],
            structure=[],
            notable_items=[],
        )

    # 크기 가드: 임계값을 넘는 콘텐츠를 전부 파싱하는 것은 DoS 위험이다
    # (XML entity expansion, YAML alias bomb, 원문 텍스트로 인한 메모리/CPU).
    # 최악의 경우를 제한하기 위해 raw head/tail 샘플로 fallback한다.
    if len(content.encode("utf-8")) > _MAX_SYNOPSIS_INPUT_BYTES:
        return ToolOutputSynopsis(
            kind="unknown",
            title="Oversized output",
            summary=[
                f"The output has {len(content)} characters ({len(content.encode('utf-8')) / 1024 / 1024:.1f} MB). Parsing skipped due to size limit.",
            ],
            structure=[],
            notable_items=[],
            sample=_head_tail_sample(content, _TEXT_EXCERPT_CHARS * 2),
        )

    if _looks_binary(content):
        return ToolOutputSynopsis(
            kind="unknown",
            title="Binary-like output",
            summary=[f"The output has {len(content)} characters and includes non-text control bytes."],
            structure=[],
            notable_items=[],
            sample=_head_tail_sample(content, _TEXT_EXCERPT_CHARS * 2),
        )

    stripped = content.strip()
    json_synopsis = _try_json(content)
    if json_synopsis is not None:
        return json_synopsis

    xml_synopsis = _try_xml(stripped)
    if xml_synopsis is not None:
        return xml_synopsis

    if "\t" in content:
        table = _try_table(content, delimiter="\t", kind="tsv")
        if table is not None:
            return table

    if "," in content:
        table = _try_table(content, delimiter=",", kind="csv")
        if table is not None:
            return table

    yaml_synopsis = _try_yaml(content)
    if yaml_synopsis is not None:
        return yaml_synopsis

    if _looks_code(content):
        return _summarize_code(content)

    return _summarize_text(content, tool_name=tool_name)


def render_tool_output_preview(
    content: str,
    *,
    tool_name: str,
    virtual_path: str,
    head_chars: int,
    tail_chars: int,
) -> str:
    """파일 기반 preview를 typed synopsis + raw head/tail 샘플로 렌더링한다.

    synopsis가 주된 신호다. raw 샘플은 synopsis 도입 전에 운영자가
    preview_head_chars / preview_tail_chars로 받던 inline head/tail 바이트를 복원한다.
    binary 성격의 출력은 synopsis가 이미 raw 샘플을 들고 있고, 그 외에는 *content*의
    앞에서 head_chars, 뒤에서 tail_chars만큼 잘라 쓴다.
    """
    total = len(content)
    synopsis = build_tool_output_synopsis(content, tool_name=tool_name)
    head_budget = max(0, head_chars)
    tail_budget = max(0, tail_chars)
    # text 종류는 raw 샘플이 덧붙을 예정이면 synopsis의 발췌를 생략한다
    # (head/tail 바이트가 두 곳에 중복되는 것을 피한다).
    if synopsis.kind == "text" and head_budget + tail_budget > 0 and len(content) > head_budget + tail_budget:
        synopsis = _summarize_text(content, tool_name=tool_name, include_excerpts=False)
    lines = [
        f"[Full {tool_name} output saved to {virtual_path} ({total} chars, ~{total // 4} tokens).]",
        f"[Preview kind: {synopsis.kind}. This is a structured synopsis, not a raw head/tail truncation.]",
        "",
        f"{synopsis.title}:",
    ]
    lines.extend(f"- {item}" for item in synopsis.summary)

    if synopsis.structure:
        lines.append("")
        lines.append("Structure:")
        lines.extend(f"- {item}" for item in synopsis.structure)

    if synopsis.notable_items:
        lines.append("")
        lines.append("Notable items:")
        lines.extend(f"- {item}" for item in synopsis.notable_items)

    raw_sample = _build_raw_sample(content, head_budget=head_budget, tail_budget=tail_budget, existing=synopsis.sample)
    if raw_sample:
        lines.append("")
        lines.append("Raw sample (head + tail, clipped to head_chars / tail_chars):")
        lines.append(raw_sample)

    lines.append("")
    lines.append("Access:")
    lines.append(f"- Use read_file on {virtual_path} with start_line and end_line to inspect the raw output.")
    return "\n".join(lines)


def _clip(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _build_raw_sample(content: str, *, head_budget: int, tail_budget: int, existing: str) -> str:
    """inline head/tail raw 샘플을 구성한다.

    synopsis가 이미 샘플을 제공하면(binary 성격 출력) 그대로 쓴다. 아니면 앞에서
    head_budget, 뒤에서 tail_budget만큼 잘라내되 줄 경계에 맞춰 preview가 깔끔한 줄바꿈에서
    끝나게 한다. 두 조각이 겹칠 경우 바이트가 중복되지 않게 한다.
    """
    if existing:
        return existing
    if head_budget <= 0 and tail_budget <= 0:
        return ""
    if len(content) <= head_budget + tail_budget:
        return content
    parts: list[str] = []
    if head_budget > 0:
        head = content[:head_budget]
        # 깔끔하게 자르기 위해 예산 안의 마지막 개행에 맞춘다.
        snap = head.rfind("\n")
        if snap > 0:
            head = head[:snap]
        parts.append(head)
    if tail_budget > 0 and head_budget + tail_budget < len(content):
        tail = content[-tail_budget:]
        # 깔끔하게 자르기 위해 tail 안의 첫 개행에 맞춘다.
        snap = tail.find("\n")
        if snap >= 0 and snap < len(tail) - 1:
            tail = tail[snap + 1 :]
        parts.append(tail)
    if len(parts) == 2:
        return f"{parts[0]}\n...\n{parts[1]}"
    return parts[0]


def _one_line(value: str, limit: int) -> str:
    return _clip(re.sub(r"\s+", " ", value).strip(), limit)


def _head_tail_sample(content: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(content) <= limit:
        return content
    half = max(1, limit // 2)
    return f"{content[:half]}\n...\n{content[-half:]}"


def _looks_binary(content: str) -> bool:
    if "\x00" in content:
        return True
    sample = content[:1000]
    controls = sum(1 for char in sample if ord(char) < 32 and char not in "\n\r\t")
    return controls / max(1, len(sample)) > 0.05


def _type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _short_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(_clip(value, 80), ensure_ascii=False)
    return _clip(repr(value), 80)


def _json_shape(value: Any, *, depth: int = 0) -> str:
    if depth >= _JSON_SHAPE_MAX_DEPTH:
        return "..."
    if isinstance(value, dict):
        keys = [str(key) for key in list(value.keys())[:_KEY_LIMIT]]
        suffix = f": {', '.join(keys)}" if keys else ""
        return f"object(keys={len(value)}{suffix})"
    if isinstance(value, list):
        samples = ", ".join(_json_shape(item, depth=depth + 1) for item in value[:3])
        suffix = f", first=[{samples}]" if samples else ""
        return f"array(len={len(value)}{suffix})"
    return _type_name(value)


def _json_path(parent: str, key: Any) -> str:
    key_text = str(key)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key_text):
        return f"{parent}.{key_text}"
    return f"{parent}[{json.dumps(key_text, ensure_ascii=False)}]"


def _json_container_description(value: Any) -> str:
    if isinstance(value, dict):
        keys = [str(key) for key in list(value.keys())[:_KEY_LIMIT]]
        suffix = f"; keys {', '.join(keys)}" if keys else ""
        return f"object keys {len(value)}{suffix}"
    if isinstance(value, list):
        detail = f"array length {len(value)}"
        if value:
            detail += f"; first item {_type_name(value[0])}"
        return detail
    return _type_name(value)


def _json_container_paths(value: Any, *, limit: int = _JSON_STRUCTURE_LIMIT) -> list[str]:
    """중첩된 JSON container 경로를 요약한다.

    위치 정보는 의도적으로 뺀다. 문자열 검색 기반의 '(line N, byte offset M)' 근사 앵커는
    키 문자열이 문서 앞쪽에 값으로도 등장하거나 같은 키가 여러 깊이에 나타나면 틀린다.
    경로 자체만으로도 탐색에 충분하고, 에이전트는 어느 구간이 필요한지 스스로 판단해
    read_file의 start_line을 정한다.
    """
    paths: list[str] = []

    def walk(node: Any, current_path: str, depth: int) -> None:
        if len(paths) >= limit or depth >= _JSON_STRUCTURE_DEPTH:
            return
        if isinstance(node, dict):
            for key, child in list(node.items())[:_KEY_LIMIT]:
                if len(paths) >= limit:
                    break
                next_path = _json_path(current_path, key)
                if isinstance(child, (dict, list)):
                    paths.append(f"{next_path}: {_json_container_description(child)}")
                    walk(child, next_path, depth + 1)
            return
        if isinstance(node, list) and node:
            first = node[0]
            if isinstance(first, (dict, list)):
                walk(first, f"{current_path}[]", depth + 1)

    walk(value, "$", 0)
    return paths


def _scalar_examples(value: Any, *, path: str = "$", limit: int = _SCALAR_LIMIT) -> list[str]:
    examples: list[str] = []

    def walk(node: Any, current: str, depth: int) -> None:
        if len(examples) >= limit or depth >= _JSON_STRUCTURE_DEPTH:
            return
        if isinstance(node, dict):
            for key, child in list(node.items())[:_KEY_LIMIT]:
                walk(child, f"{current}.{key}", depth + 1)
                if len(examples) >= limit:
                    break
            return
        if isinstance(node, list):
            for index, child in enumerate(node[:2]):
                walk(child, f"{current}[{index}]", depth + 1)
                if len(examples) >= limit:
                    break
            return
        examples.append(f"{current}: {_short_value(node)}")

    walk(value, path, 0)
    return examples


def _try_json(content: str) -> ToolOutputSynopsis | None:
    stripped = content.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(stripped)
    except Exception:
        return None

    trailing = len(stripped[end:].strip())
    summary: list[str] = []
    structure: list[str] = [f"shape: {_json_shape(value)}"]
    structure.extend(_json_container_paths(value))
    notable = _scalar_examples(value)
    # NOTE: scalar 예시는 파싱된 구조 어디서든 값을 노출할 수 있다(head/tail 바이트로
    # 한정되지 않는다). 의도된 동작이며, synopsis는 구조 요약이지 기밀 필터가 아니다.
    # 예전 preview가 head/tail 조각만 노출한다고 믿던 운영자는 문서 중간의 민감한 값을
    # 도구 출력에서 점검해야 한다.
    if isinstance(value, dict):
        keys = [str(key) for key in value.keys()]
        summary.append(f"JSON object with {len(keys)} top-level keys.")
        summary.append(f"Top-level keys: {', '.join(keys[:_KEY_LIMIT]) or '(none)'}")
    elif isinstance(value, list):
        summary.append(f"JSON array with {len(value)} items.")
        if value:
            structure.append(f"first item type: {_type_name(value[0])}")
    else:
        summary.append(f"JSON {_type_name(value)}.")

    if trailing:
        notable.append(f"Trailing non-JSON characters after first value: {trailing}")

    return ToolOutputSynopsis(
        kind="json",
        title="JSON output",
        summary=summary,
        structure=structure,
        notable_items=notable,
    )


def _try_xml(stripped: str) -> ToolOutputSynopsis | None:
    if not stripped.startswith("<"):
        return None
    if SafeET is None:  # defusedxml이 없으면 entity-expansion DoS를 피하려고 XML 파싱을 건너뛴다
        return None
    try:
        root = (SafeET or ET).fromstring(stripped)
    except Exception:
        return None

    child_counts = Counter(child.tag for child in list(root))
    structure = [f"root tag: {root.tag}", f"root attributes: {len(root.attrib)}"]
    structure.extend(f"{tag}: {count}" for tag, count in child_counts.most_common(_KEY_LIMIT))
    return ToolOutputSynopsis(
        kind="xml",
        title="XML output",
        summary=[f"XML document with root tag {root.tag}."],
        structure=structure,
        notable_items=[],
    )


_TABLE_MIN_DATA_ROWS = 5
_TABLE_HEADER_IDENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def _try_table(content: str, *, delimiter: str, kind: Literal["csv", "tsv"]) -> ToolOutputSynopsis | None:
    sample_text = "\n".join(content.splitlines()[:_TABLE_SAMPLE_ROWS])
    try:
        rows = list(csv.reader(io.StringIO(sample_text), delimiter=delimiter))
    except csv.Error:
        return None

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2 or len(rows[0]) < 2:
        return None

    width = len(rows[0])
    consistent = [row for row in rows[1:11] if len(row) == width]
    # TSV와 CSV 모두 같은 폭의 데이터 행이 _TABLE_MIN_DATA_ROWS개 이상이어야 한다.
    # TSV는 오탐이 많다(들여쓴 bash, ls -l 목록, tree 덤프). CSV는 드물지만 첫 줄에
    # 쉼표가 있는 산문도 이 gate가 없으면 통과한다.
    if len(consistent) < _TABLE_MIN_DATA_ROWS:
        return None

    # 헤더 행은 식별자처럼 보여야 한다(공백 없음, 선행 공백 없음).
    # 탭으로 들여쓴 bash 출력, ls -l 목록, 우연히 탭 구분인 tree 덤프를 거른다.
    raw_header = rows[0]
    if any(not _TABLE_HEADER_IDENT_RE.match(cell.strip()) for cell in raw_header):
        return None
    if any(cell.startswith((" ", "\t")) for cell in raw_header):
        return None

    columns = [cell.strip() or f"column_{idx + 1}" for idx, cell in enumerate(raw_header)]
    total_nonempty_lines = sum(1 for line in content.splitlines() if line.strip())
    data_rows = max(0, total_nonempty_lines - 1)
    # 첫 데이터 행을 key=value 목록으로 렌더링한다. 구분자를 포함한 인용 셀이 쉼표로
    # 다시 이어져 모델이 컬럼 수를 오해하는 일을 막는다.
    first_data_pairs: list[str] = []
    if len(rows) > 1:
        for col_name, cell in list(zip(columns, rows[1]))[:_TABLE_COLUMN_LIMIT]:
            first_data_pairs.append(f"{col_name}={_clip(cell, 80)}")
    title = "CSV table output" if kind == "csv" else "TSV table output"
    label = kind.upper()
    return ToolOutputSynopsis(
        kind=kind,
        title=title,
        summary=[f"{label} table with {data_rows} data rows and {width} columns."],
        structure=[
            f"columns: {', '.join(columns[:_TABLE_COLUMN_LIMIT])}",
            f"first data row: {' | '.join(first_data_pairs) or '(none)'}",
        ],
        notable_items=[],
    )


_YAML_KEY_LINE_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_.\-]*:\s*\S.*$")


def _looks_yaml(content: str) -> bool:
    """YAML 형태 콘텐츠를 판별하는 휴리스틱.

    구조적으로 YAML처럼 보일 때만 True를 반환한다(문서 시작 마커, 또는 값이 대문자
    로그 접두사가 아닌 중첩 key/value 줄이 여러 개). 평범한 로그, Python traceback,
    전부 대문자 태그로 된 `key: value` 줄(YAML은 문자열 키로 취급)은 거른다.
    """
    stripped = content.lstrip()
    if stripped.startswith("---"):
        return True
    if _looks_code(content):
        return False

    key_like = 0
    for line in content.splitlines()[:80]:
        if not _YAML_KEY_LINE_RE.match(line):
            continue
        # 키가 대문자 태그이고 값이 자유 형식 메시지인 로그 형태 줄은 거른다.
        # 예: "INFO: starting service".
        key = line.split(":", 1)[0].strip()
        if key.isupper() and "_" not in key:
            continue
        key_like += 1
        if key_like >= 3:
            return True
    return False


def _try_yaml(content: str) -> ToolOutputSynopsis | None:
    if not _looks_yaml(content):
        return None
    # alias-bomb DoS를 막기 위해 파싱 크기를 제한한다(yaml.safe_load는 지수적으로 커질 수
    # 있는 YAML alias를 해석한다). 휴리스틱이 대부분의 비YAML 콘텐츠를 거르지만, 의도적으로
    # 만든 alias bomb은 휴리스틱을 쉽게 통과한다.
    if len(content) > 500_000:
        return None
    try:
        value = yaml.safe_load(content)
    except Exception:
        return None
    if not isinstance(value, (dict, list)):
        return None
    # 값이 전부 문자열인 평평한 payload는 거른다. 로그 줄과 Python traceback이
    # safe_load 후 갖는 모양이며, 실제로는 자유 형식 텍스트인 출력에 대해 모델에게
    # "YAML with N keys"라는 오해를 부르는 요약을 주게 된다.
    if isinstance(value, dict):
        non_string_children = sum(1 for v in value.values() if not isinstance(v, str))
        if non_string_children == 0 and len(value) > 0:
            return None

    summary: list[str]
    structure: list[str] = []
    if isinstance(value, dict):
        keys = [str(key) for key in value.keys()]
        summary = [f"YAML object with {len(keys)} top-level keys.", f"Top-level keys: {', '.join(keys[:_KEY_LIMIT])}"]
        for key, child in list(value.items())[:_KEY_LIMIT]:
            structure.append(f"{key}: {_type_name(child)}")
    else:
        summary = [f"YAML array with {len(value)} items."]
        if value:
            structure.append(f"first item type: {_type_name(value[0])}")

    return ToolOutputSynopsis(
        kind="yaml",
        title="YAML output",
        summary=summary,
        structure=structure,
        notable_items=[],
    )


def _looks_code(content: str) -> bool:
    return any(pattern.search(content) for pattern in _CODE_HINTS)


def _summarize_code(content: str) -> ToolOutputSynopsis:
    imports: list[str] = []
    symbols: list[str] = []
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        import_match = re.match(r"^(?:from\s+(\S+)\s+import|import\s+(\S+))", stripped)
        if import_match:
            imports.append(_one_line(import_match.group(1) or import_match.group(2) or "", 160))
            continue
        symbol_match = re.match(
            r"^(class|def|async\s+def|function|export\s+function|pub\s+fn|fn)\s+([A-Za-z_]\w*)",
            stripped,
        )
        if symbol_match:
            symbols.append(_one_line(f"{symbol_match.group(1)} {symbol_match.group(2)}", 180))

    structure = [f"line count: {len(lines)}"]
    if imports:
        structure.append(f"imports: {', '.join(imports[:_CODE_IMPORT_LIMIT])}")

    return ToolOutputSynopsis(
        kind="code",
        title="Code-like output",
        summary=[f"Code-like text with {len(lines)} lines."],
        structure=structure,
        notable_items=symbols[:_CODE_SYMBOL_LIMIT],
    )


def _summarize_text(content: str, *, tool_name: str = "", include_excerpts: bool = True) -> ToolOutputSynopsis:
    lines = content.splitlines()
    normalized = re.sub(r"\s+", " ", content).strip()
    headers: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not (re.match(r"^#{1,6}\s+", stripped) or re.match(r"^[A-Z0-9][A-Z0-9\s:_-]{6,}$", stripped)):
            continue
        header = _one_line(stripped, 160)
        if header in seen:
            continue
        seen.add(header)
        headers.append(header)
        if len(headers) >= _TEXT_HEADER_LIMIT:
            break

    tool_hint = f" from {tool_name}" if tool_name else ""
    summary_lines = [
        f"Text output{tool_hint} with {len(content)} characters, {len(normalized.split()) if normalized else 0} words, and {len(lines)} lines.",
        f"Detected section headers: {' | '.join(headers) if headers else 'none detected'}.",
    ]
    # render_tool_output_preview가 raw head/tail 샘플을 덧붙이지 않을 때만 시작/끝 발췌를
    # 넣는다(synopsis 요약과 raw 샘플에 같은 head/tail 바이트가 중복되는 것을 피한다).
    if include_excerpts:
        opener = _one_line(content[:_TEXT_EXCERPT_CHARS], _TEXT_EXCERPT_CHARS)
        if len(content) <= _TEXT_EXCERPT_CHARS:
            closer = ""
        else:
            close_start = max(_TEXT_EXCERPT_CHARS, len(content) - _TEXT_EXCERPT_CHARS)
            closer = _one_line(content[close_start:], _TEXT_EXCERPT_CHARS) if close_start < len(content) else ""
        summary_lines.append(f"Opening excerpt: {opener or '(empty)'}")
        if closer:
            summary_lines.append(f"Closing excerpt: {closer}")
    return ToolOutputSynopsis(
        kind="text",
        title="Text output",
        summary=summary_lines,
        structure=[],
        notable_items=[],
    )
