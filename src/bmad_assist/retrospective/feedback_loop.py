"""Retrospective upstream document feedback-loop validation.

The BMAD retrospective skill requires discoveries to be carried back into
upstream planning, architecture, governance, or backlog documents when they
change project understanding. This module turns that process requirement into
an enforceable BMAD-ASSIST completion gate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT_EXTENSIONS = {
    ".cs",
    ".json",
    ".md",
    ".ps1",
    ".toml",
    ".yaml",
    ".yml",
}
SNAPSHOT_FILENAMES = {"AGENTS.md", "README.md"}
MAX_SNAPSHOT_FILE_BYTES = 5 * 1024 * 1024

VOLATILE_BMAD_ROOTS = {
    "_bmad",
    "_bmad_output",
    "_bmad-output",
    ".bmad_assist",
    ".bmad-assist",
}
SNAPSHOT_EXCLUDED_ROOTS = VOLATILE_BMAD_ROOTS | {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "bin",
    "node_modules",
    "obj",
}
GOVERNANCE_CANDIDATE_ROOT = ".output/adr-control-plane/bmad-governance-candidates"

DOCUMENT_FEEDBACK_HEADING = "document feedback matrix"
ALLOWED_DISPOSITIONS = {
    "updated": "updated",
    "document updated": "updated",
    "document-updated": "updated",
    "doc updated": "updated",
    "prd update": "updated",
    "architecture update": "updated",
    "epic story backlog update": "updated",
    "epic/story/backlog update": "updated",
    "governance candidate": "governance-candidate",
    "governance-candidate": "governance-candidate",
    "governance_candidate": "governance-candidate",
    "candidate": "governance-candidate",
    "human gated": "human-gated",
    "human-gated": "human-gated",
    "human_gate": "human-gated",
    "deferred": "human-gated",
    "blocked": "human-gated",
    "no change": "no-change",
    "no-change": "no-change",
    "no_change": "no-change",
    "not needed": "no-change",
    "not-needed": "no-change",
}


@dataclass(frozen=True)
class FeedbackSnapshot:
    """Content hashes captured before retrospective execution."""

    project_root: Path
    file_hashes: dict[str, str]


@dataclass(frozen=True)
class FeedbackLoopValidationResult:
    """Result of validating the retrospective feedback-loop evidence."""

    valid: bool
    errors: tuple[str, ...] = ()
    dispositions: tuple[str, ...] = ()


def capture_feedback_snapshot(project_root: Path) -> FeedbackSnapshot:
    """Capture before-state hashes for files that may be upstream feedback targets."""
    root = project_root.resolve()
    file_hashes: dict[str, str] = {}

    if not root.exists():
        return FeedbackSnapshot(project_root=root, file_hashes=file_hashes)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_snapshot_path(path, root):
            continue
        try:
            if path.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
                continue
            rel_path = _relative_posix(path, root)
            file_hashes[rel_path] = _hash_file(path)
        except OSError:
            continue

    return FeedbackSnapshot(project_root=root, file_hashes=file_hashes)


def validate_retrospective_feedback_loop(
    report_content: str,
    project_root: Path,
    snapshot: FeedbackSnapshot | None,
) -> FeedbackLoopValidationResult:
    """Validate that a retrospective completed its upstream feedback-loop contract."""
    root = project_root.resolve()
    section = _extract_feedback_matrix_section(report_content)
    if section is None:
        return FeedbackLoopValidationResult(
            valid=False,
            errors=(
                "retrospective report is missing required 'Document Feedback Matrix' section",
            ),
        )

    rows, table_error = _parse_feedback_table(section)
    if table_error is not None:
        return FeedbackLoopValidationResult(valid=False, errors=(table_error,))
    if not rows:
        return FeedbackLoopValidationResult(
            valid=False,
            errors=("Document Feedback Matrix must contain at least one disposition row",),
        )

    errors: list[str] = []
    dispositions: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        disposition = _normalize_disposition(_get_cell(row, "disposition"))
        if disposition is None:
            raw = _get_cell(row, "disposition") or "<blank>"
            errors.append(f"row {row_index}: unsupported feedback disposition '{raw}'")
            continue

        dispositions.append(disposition)
        if disposition == "updated":
            errors.extend(_validate_updated_row(row_index, row, root, snapshot))
        elif disposition == "governance-candidate":
            errors.extend(_validate_governance_candidate_row(row_index, row, root, snapshot))
        elif disposition == "human-gated":
            errors.extend(_validate_human_gated_row(row_index, row))
        elif disposition == "no-change":
            errors.extend(_validate_no_change_row(row_index, row))

    return FeedbackLoopValidationResult(
        valid=not errors,
        errors=tuple(errors),
        dispositions=tuple(dispositions),
    )


def _should_skip_snapshot_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True

    if any(part in SNAPSHOT_EXCLUDED_ROOTS for part in rel.parts):
        return True

    return path.suffix.lower() not in SNAPSHOT_EXTENSIONS and path.name not in SNAPSHOT_FILENAMES


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _extract_feedback_matrix_section(report_content: str) -> str | None:
    lines = report_content.splitlines()
    start_index: int | None = None

    for index, line in enumerate(lines):
        heading = _normalize_heading(line)
        if heading == DOCUMENT_FEEDBACK_HEADING:
            start_index = index + 1
            break

    if start_index is None:
        return None

    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if re.match(r"^#{1,6}\s+", lines[index]):
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip()


def _normalize_heading(line: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", line.strip())
    text = re.sub(r"[*`_]", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def _parse_feedback_table(section: str) -> tuple[list[dict[str, str]], str | None]:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        if index + 1 >= len(lines) or not _is_markdown_separator(lines[index + 1]):
            continue

        headers = [_normalize_column(cell) for cell in _split_table_row(line)]
        required_columns = [
            "finding",
            "disposition",
            "evidence",
            "owner",
            "blocked downstream work",
        ]
        missing = [column for column in required_columns if not _has_column(headers, column)]
        if missing:
            return [], f"Document Feedback Matrix is missing required columns: {', '.join(missing)}"

        rows: list[dict[str, str]] = []
        for data_line in lines[index + 2 :]:
            if not data_line.startswith("|"):
                break
            if _is_markdown_separator(data_line):
                continue
            cells = _split_table_row(data_line)
            row: dict[str, str] = {}
            for header_index, header in enumerate(headers):
                row[header] = cells[header_index].strip() if header_index < len(cells) else ""
            rows.append(row)
        return rows, None

    return [], "Document Feedback Matrix must be a markdown table with a header separator"


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _normalize_column(value: str) -> str:
    text = re.sub(r"[*`_]", "", value)
    return re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()


def _has_column(headers: list[str], column: str) -> bool:
    return any(header == column or header.startswith(column) for header in headers)


def _get_cell(row: dict[str, str], column: str) -> str:
    for header, value in row.items():
        if header == column or header.startswith(column):
            return value.strip()
    return ""


def _normalize_disposition(value: str) -> str | None:
    key = _normalize_column(value)
    return ALLOWED_DISPOSITIONS.get(key)


def _validate_updated_row(
    row_index: int,
    row: dict[str, str],
    root: Path,
    snapshot: FeedbackSnapshot | None,
) -> list[str]:
    evidence = _get_cell(row, "evidence")
    paths, path_errors = _extract_evidence_paths(evidence, root)
    errors = [f"row {row_index}: {error}" for error in path_errors]
    if not paths:
        errors.append(f"row {row_index}: updated disposition must list at least one updated path")
        return errors

    for rel_path, abs_path in paths:
        if _is_volatile_bmad_path(rel_path):
            errors.append(
                f"row {row_index}: updated path '{rel_path}' is inside volatile BMAD output/config"
            )
            continue
        if not abs_path.exists():
            errors.append(f"row {row_index}: updated path '{rel_path}' does not exist")
            continue
        if not _path_changed_since_snapshot(rel_path, abs_path, snapshot):
            errors.append(
                f"row {row_index}: updated path '{rel_path}' was not changed during retrospective"
            )

    return errors


def _validate_governance_candidate_row(
    row_index: int,
    row: dict[str, str],
    root: Path,
    snapshot: FeedbackSnapshot | None,
) -> list[str]:
    evidence = _get_cell(row, "evidence")
    paths, path_errors = _extract_evidence_paths(evidence, root)
    errors = [f"row {row_index}: {error}" for error in path_errors]
    if not paths:
        errors.append(
            f"row {row_index}: governance-candidate disposition must list a candidate handoff path"
        )
        return errors

    for rel_path, abs_path in paths:
        if not _is_governance_candidate_path(rel_path):
            errors.append(
                f"row {row_index}: governance candidate path '{rel_path}' must be under "
                f"{GOVERNANCE_CANDIDATE_ROOT}/"
            )
            continue
        if not abs_path.exists():
            errors.append(f"row {row_index}: governance candidate path '{rel_path}' does not exist")
            continue
        if not _path_changed_since_snapshot(rel_path, abs_path, snapshot):
            errors.append(
                f"row {row_index}: governance candidate path '{rel_path}' was not changed "
                "during retrospective"
            )

    return errors


def _validate_human_gated_row(row_index: int, row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    owner = _get_cell(row, "owner")
    evidence = _get_cell(row, "evidence")
    blocked = _get_cell(row, "blocked downstream work")
    if _is_blank_or_tbd(owner):
        errors.append(f"row {row_index}: human-gated disposition must record an owner")
    if _is_blank_or_tbd(evidence):
        errors.append(f"row {row_index}: human-gated disposition must record evidence")
    if _is_blank_or_tbd(blocked):
        errors.append(
            f"row {row_index}: human-gated disposition must record blocked downstream work"
        )
    return errors


def _validate_no_change_row(row_index: int, row: dict[str, str]) -> list[str]:
    evidence = _get_cell(row, "evidence")
    if len(_strip_markdown(evidence)) < 20:
        return [f"row {row_index}: no-change disposition must include a concrete rationale"]
    return []


def _extract_evidence_paths(evidence: str, root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", evidence))
    candidates.extend(match.group(1) for match in re.finditer(r"`([^`]+)`", evidence))
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"(?<![\w.-])(?:\.output|\.adr|\.rules|\.operations|docs|src|tests|infra|"
            r"\.scripts|AGENTS\.md|README\.md|_bmad|_bmad-output|_bmad_output|"
            r"\.bmad-assist|\.bmad_assist)/[^\s,;)\]]+",
            evidence,
        )
    )

    paths: list[tuple[str, Path]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_path_token(candidate)
        if not normalized:
            continue
        resolved = _resolve_project_path(normalized, root)
        if isinstance(resolved, str):
            errors.append(resolved)
            continue
        rel_path, abs_path = resolved
        if rel_path not in seen:
            seen.add(rel_path)
            paths.append((rel_path, abs_path))

    return paths, errors


def _normalize_path_token(token: str) -> str:
    clean = token.strip().strip("<>\"'")
    clean = clean.split("#", 1)[0]
    clean = re.sub(r":\d+(?::\d+)?$", "", clean)
    clean = clean.rstrip("`.,;:")
    return clean


def _resolve_project_path(token: str, root: Path) -> tuple[str, Path] | str:
    candidate = Path(token)
    try:
        if candidate.is_absolute():
            abs_path = candidate.resolve()
            rel = abs_path.relative_to(root)
        else:
            rel = candidate
            if any(part == ".." for part in rel.parts):
                return f"path '{token}' escapes the project root"
            abs_path = (root / rel).resolve()
            rel = abs_path.relative_to(root)
    except ValueError:
        return f"path '{token}' is outside the project root"

    return rel.as_posix(), abs_path


def _is_volatile_bmad_path(rel_path: str) -> bool:
    first_part = Path(rel_path).parts[0] if Path(rel_path).parts else ""
    return first_part in VOLATILE_BMAD_ROOTS


def _is_governance_candidate_path(rel_path: str) -> bool:
    return rel_path == GOVERNANCE_CANDIDATE_ROOT or rel_path.startswith(
        f"{GOVERNANCE_CANDIDATE_ROOT}/"
    )


def _path_changed_since_snapshot(
    rel_path: str,
    abs_path: Path,
    snapshot: FeedbackSnapshot | None,
) -> bool:
    if snapshot is None:
        return False
    before_hash = snapshot.file_hashes.get(rel_path)
    if before_hash is None:
        return abs_path.exists()
    try:
        return _hash_file(abs_path) != before_hash
    except OSError:
        return False


def _is_blank_or_tbd(value: str) -> bool:
    return _strip_markdown(value).lower() in {"", "n/a", "na", "none", "tbd", "todo"}


def _strip_markdown(value: str) -> str:
    text = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", value)
    text = re.sub(r"[*_`]", "", text)
    return text.strip()
