"""Source context collection service for workflow compilers.

This module provides centralized, configurable source file collection
with scoring, adaptive content extraction, and symbol-boundary truncation.

Usage:
    from bmad_assist.compiler.source_context import SourceContextService

    service = SourceContextService(context, "code_review")
    source_files = service.collect_files(file_list_paths, git_diff_files)
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bmad_assist.compiler.shared_utils import estimate_tokens
from bmad_assist.core.config import SourceContextConfig, get_config
from bmad_assist.core.exceptions import ConfigError

if TYPE_CHECKING:
    from bmad_assist.compiler.types import CompilerContext

logger = logging.getLogger(__name__)

# Minimum budget to be considered "enabled" (0-99 = disabled)
MIN_ENABLED_BUDGET = 100

# Max file size for AST parsing (100KB)
MAX_AST_PARSE_SIZE = 100 * 1024

# Binary file extensions to skip
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".bmp",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".webm",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyc",
        ".pyo",
        ".pyd",
        ".class",
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)

# Config file extensions (for scoring penalty)
CONFIG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
    }
)

SOURCE_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".cs",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".go",
        ".rb",
        ".rs",
        ".php",
        ".swift",
        ".scala",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".m",
        ".mm",
        ".fs",
        ".fsx",
        ".vb",
        ".sql",
        ".sh",
        ".ps1",
    }
)

DOCUMENTATION_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
    }
)

SOURCE_PREFIXES: tuple[str, ...] = (
    "src/",
    "app/",
    "lib/",
    "services/",
    "packages/",
)

GENERATED_ARTIFACT_PREFIXES: tuple[str, ...] = (
    "_bmad-output/",
    ".bmad-assist/",
)

SEMANTIC_PRIORITY_WORKFLOWS: frozenset[str] = frozenset(
    {
        "dev_story",
        "create_story",
    }
)

GENERATED_DOCUMENT_EXCLUSION_WORKFLOWS: frozenset[str] = frozenset(
    {
        "create_story",
        "dev_story",
    }
)

# Test file patterns
TEST_PATH_PATTERNS: tuple[str, ...] = (
    "tests/",
    "test/",
    "__tests__/",
    "spec/",
)
TEST_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_.*\.py$"),
    re.compile(r".*_test\.py$"),
    re.compile(r"^conftest\.py$"),
    re.compile(r".*_spec\.(py|js|ts)$"),
)

# Header: match any level 2+ heading containing "File List"
_FILE_LIST_HEADER = re.compile(
    r"^(#{2,})\s+File\s+List\s*$", re.MULTILINE | re.IGNORECASE
)

# File paths from bullet lists: - `path` or * `path`
_BULLET_PATH = re.compile(
    r"^\s*[-*]\s+`([^`\n]+)`", re.MULTILINE
)

# File paths from numbered lists: 1. `path` or 1) `path`
_NUMBERED_PATH = re.compile(
    r"^\s*\d+[.)]\s+`([^`\n]+)`", re.MULTILINE
)

# File paths from markdown tables: | `path` |
_TABLE_PATH = re.compile(
    r"\|\s*`([^`\n]+)`", re.MULTILINE
)

# Fallback: bullet/numbered with no backticks — first path-like token
_PLAIN_PATH = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+(\S+\.\w+)", re.MULTILINE
)


@dataclass
class ScoredFile:
    """A file with scoring information for prioritization.

    Attributes:
        path: File path relative to project root.
        score: Total score from all factors.
        tokens: Estimated tokens for full file.
        in_file_list: Whether file was in story's File List.
        in_git_diff: Whether file was in git diff.
        is_test: Whether detected as test file.
        is_config: Whether detected as config file.
        relevance_rank: Deterministic semantic tie-breaker. Higher is better.
        is_current_story_context: Whether the path appears to match the current story.
        change_lines: Lines changed (from git diff, 0 if not in diff).
        hunk_ranges: Line ranges of changes [(start, end), ...].

    """

    path: str
    score: int = 0
    tokens: int = 0
    in_file_list: bool = False
    in_git_diff: bool = False
    is_test: bool = False
    is_config: bool = False
    relevance_rank: int = 0
    is_current_story_context: bool = False
    change_lines: int = 0
    hunk_ranges: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class GitDiffFile:
    """Information about a file from git diff.

    Attributes:
        path: File path relative to project root.
        change_lines: Total lines changed.
        hunk_ranges: Line ranges of changes [(start_line, end_line), ...].

    """

    path: str
    change_lines: int
    hunk_ranges: list[tuple[int, int]] = field(default_factory=list)


def _extract_file_list_section(content: str) -> str | None:
    """Find File List heading and extract section content.

    Handles any heading level (##, ###, ####).
    Section ends at next heading of same-or-higher level.
    Sub-headers (deeper level) are included.

    Args:
        content: Full document content.

    Returns:
        Section content after the heading, or None if not found.

    """
    header_match = _FILE_LIST_HEADER.search(content)
    if not header_match:
        return None

    hash_count = len(header_match.group(1))  # number of #'s
    section_start = header_match.end()

    # Stop at next heading of same or higher level (≤ hash_count #'s)
    # (?!#) prevents ## from matching ###
    boundary = re.compile(
        rf"^#{{2,{hash_count}}}(?!#)\s", re.MULTILINE
    )
    next_section = boundary.search(content[section_start:])
    if next_section:
        return content[section_start:section_start + next_section.start()]
    return content[section_start:]


def extract_file_paths_from_section(section_content: str) -> list[str]:
    """Extract file paths from File List section content.

    Supports: bullet lists, numbered lists, markdown tables,
    with and without backticks.

    Args:
        section_content: Content of the File List section (after heading).

    Returns:
        List of file paths found.

    """
    paths: list[str] = []
    seen: set[str] = set()

    # Priority order: backtick patterns first (most precise)
    for pattern in (_BULLET_PATH, _NUMBERED_PATH, _TABLE_PATH):
        for m in pattern.finditer(section_content):
            p = m.group(1).strip()
            if p and p not in seen:
                paths.append(p)
                seen.add(p)

    # Fallback: plain paths (no backticks) — only if nothing found yet
    if not paths:
        for m in _PLAIN_PATH.finditer(section_content):
            p = m.group(1).strip()
            if p and p not in seen and not p.startswith("#") and not p.startswith("*"):
                paths.append(p)
                seen.add(p)

    return paths


def extract_file_paths_from_story(story_content: str) -> list[str]:
    """Extract file paths from File List section in story content.

    Parses the "## File List", "### File List", etc. section and extracts
    file paths from markdown list items, numbered lists, and tables.

    Args:
        story_content: Full story file content.

    Returns:
        List of file paths found in the File List section.

    """
    section = _extract_file_list_section(story_content)
    if not section:
        return []
    return extract_file_paths_from_section(section)


def is_binary_file(path: Path) -> bool:
    """Check if file is binary.

    Uses extension check first, then falls back to magic bytes detection.

    Args:
        path: Path to file.

    Returns:
        True if file is binary.

    """
    # Check by extension
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True

    # Check magic bytes (first 8KB for null bytes)
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
    except OSError:
        return True  # Treat unreadable as binary

    return False


def safe_read_file(path: Path) -> str | None:
    """Safely read file content with encoding handling.

    Args:
        path: Path to file.

    Returns:
        File content or None if unreadable.

    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Encoding issue in %s, using replacement chars", path)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    except OSError:
        return None


def _normalize_path(path: str) -> str:
    """Normalize path for deduplication.

    Handles ./src/../src/file.py → src/file.py.

    Args:
        path: File path (possibly with ./ or ..).

    Returns:
        Normalized path string.

    """
    # Use Path for normalization
    normalized = Path(path)
    # Resolve . and .. but keep relative
    parts: list[str] = []
    for part in normalized.parts:
        if part == ".":
            continue
        elif part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) if parts else "."


def _build_story_markers(context: CompilerContext) -> tuple[str, ...]:
    """Build lightweight story identifiers for path relevance checks.

    Args:
        context: Compiler context with resolved story variables.

    Returns:
        Lower-cased marker strings sorted longest-first for stable matching.

    """
    resolved = context.resolved_variables or {}
    markers: set[str] = set()

    epic_num = resolved.get("epic_num")
    story_num = resolved.get("story_num")
    story_id = resolved.get("story_id")
    story_key = resolved.get("story_key")

    for raw in (story_id, story_key):
        if raw is None:
            continue
        text = str(raw).strip().lower()
        if not text:
            continue
        markers.add(text)
        markers.add(text.replace(".", "-"))
        markers.add(text.replace("-", "."))
        markers.add(text.replace(".", "_"))
        markers.add(text.replace("-", "_"))

    if epic_num is not None and story_num is not None:
        epic_text = str(epic_num).strip().lower()
        story_text = str(story_num).strip().lower()
        markers.update(
            {
                f"{epic_text}.{story_text}",
                f"{epic_text}-{story_text}",
                f"{epic_text}_{story_text}",
                f"story-{epic_text}-{story_text}",
            }
        )

    return tuple(sorted((marker for marker in markers if marker), key=len, reverse=True))


def _is_source_code_file(path: str) -> bool:
    """Check if file is likely live implementation source.

    Args:
        path: Normalized file path.

    Returns:
        True if file appears to be source code rather than documentation.

    """
    normalized = _normalize_path(path)
    suffix = Path(normalized).suffix.lower()

    if suffix in SOURCE_CODE_EXTENSIONS:
        return True

    return any(normalized.startswith(prefix) for prefix in SOURCE_PREFIXES)


def _is_documentation_file(path: str) -> bool:
    """Check if file is documentation-like content.

    Args:
        path: Normalized file path.

    Returns:
        True if file appears to be documentation text.

    """
    suffix = Path(_normalize_path(path)).suffix.lower()
    return suffix in DOCUMENTATION_EXTENSIONS


def _is_generated_artifact(path: str) -> bool:
    """Check if file belongs to generated BMAD artifact folders.

    Args:
        path: Normalized file path.

    Returns:
        True if file is in a generated artifact folder.

    """
    normalized = _normalize_path(path)
    return any(normalized.startswith(prefix) for prefix in GENERATED_ARTIFACT_PREFIXES)


def _should_exclude_path_for_workflow(path: str, workflow_name: str) -> bool:
    """Check whether a candidate path is low-value noise for a workflow.

    Args:
        path: Normalized candidate path.
        workflow_name: Active compiler workflow.

    Returns:
        True when the candidate should be skipped before scoring.

    """
    normalized = _normalize_path(path)

    return (
        workflow_name in GENERATED_DOCUMENT_EXCLUSION_WORKFLOWS
        and _is_generated_artifact(normalized)
        and _is_documentation_file(normalized)
    )


def _matches_story_marker(path: str, story_markers: tuple[str, ...]) -> bool:
    """Check whether a path appears to reference the active story.

    Args:
        path: Normalized file path.
        story_markers: Known story identifier variants.

    Returns:
        True if the path includes a bounded story marker.

    """
    normalized = _normalize_path(path).lower()
    for marker in story_markers:
        pattern = rf"(^|[^a-z0-9]){re.escape(marker)}([^a-z0-9]|$)"
        if re.search(pattern, normalized):
            return True
    return False


def _relevance_rank(path: str, story_markers: tuple[str, ...]) -> tuple[int, bool]:
    """Assign a semantic relevance tier for deterministic tie-breaking.

    Args:
        path: File path to classify.
        story_markers: Known story identifier variants.

    Returns:
        Tuple of relevance rank and whether the file appears tied to the story.

    """
    normalized = _normalize_path(path)
    is_current_story_context = _matches_story_marker(normalized, story_markers)

    if _is_source_code_file(normalized) and not _is_test_file(normalized):
        return 40, is_current_story_context
    if _is_test_file(normalized):
        return 30, is_current_story_context
    if _is_generated_artifact(normalized) and is_current_story_context:
        return 20, True
    if is_current_story_context:
        return 18, True
    if _is_generated_artifact(normalized):
        return 10, False
    if _is_documentation_file(normalized):
        return 5, False
    return 0, False


def _is_test_file(path: str) -> bool:
    """Check if file is a test file.

    Args:
        path: File path.

    Returns:
        True if file appears to be a test file.

    """
    # Check path patterns
    for path_pattern in TEST_PATH_PATTERNS:
        if path_pattern in path:
            return True

    # Check filename patterns
    filename = Path(path).name
    return any(file_pattern.match(filename) for file_pattern in TEST_FILE_PATTERNS)


def _is_config_file(path: str) -> bool:
    """Check if file is a config file.

    Args:
        path: File path.

    Returns:
        True if file appears to be a config file.

    """
    suffix = Path(path).suffix.lower()
    return suffix in CONFIG_EXTENSIONS


class SourceContextService:
    """Service for collecting source files with configurable prioritization.

    Provides centralized source file collection with:
    - Configurable token budget per workflow
    - File List + git diff intersection scoring
    - Adaptive content mode (full files vs hunks)
    - Symbol-boundary truncation for Python files

    Usage:
        service = SourceContextService(context, "code_review")
        files = service.collect_files(file_list_paths, git_diff_files)

    """

    def __init__(self, context: CompilerContext, workflow_name: str) -> None:
        """Initialize service with compiler context and workflow name.

        Args:
            context: Compilation context with project_root.
            workflow_name: Name of workflow (e.g., 'code_review').

        """
        self.context = context
        self.project_root = context.project_root.resolve()
        self.workflow_name = workflow_name
        self.story_markers = _build_story_markers(context)

        # Get config, fallback to defaults if not loaded
        try:
            config = get_config()
            self.config: SourceContextConfig = config.compiler.source_context
        except ConfigError:
            # Config not loaded (e.g., in tests) - use defaults
            self.config = SourceContextConfig()
            logger.debug("Config not loaded, using default SourceContextConfig")

        self.budget = self.config.budgets.get_budget(workflow_name)

        logger.debug(
            "SourceContextService initialized for %s with budget %d",
            workflow_name,
            self.budget,
        )

    def is_enabled(self) -> bool:
        """Check if source context collection is enabled.

        Returns:
            True if budget >= MIN_ENABLED_BUDGET.

        """
        return self.budget >= MIN_ENABLED_BUDGET

    def collect_files(
        self,
        file_list_paths: list[str],
        git_diff_files: list[GitDiffFile] | None = None,
    ) -> dict[str, str]:
        """Collect source files with scoring and budget management.

        Args:
            file_list_paths: File paths from story's File List section.
            git_diff_files: Files from git diff with change info.

        Returns:
            Dictionary mapping absolute file paths to content.

        """
        if not self.is_enabled():
            logger.debug(
                "Source context disabled for %s (budget=%d)", self.workflow_name, self.budget
            )  # noqa: E501
            return {}

        git_diff_files = git_diff_files or []

        # Normalize paths for deduplication
        file_list_normalized = {_normalize_path(p) for p in file_list_paths}
        git_diff_map = {_normalize_path(f.path): f for f in git_diff_files}

        # Determine candidate files (intersection logic per ADR-2)
        if file_list_normalized and git_diff_map:
            # Both non-empty: use intersection
            candidate_paths = file_list_normalized & set(git_diff_map.keys())
            logger.debug(
                "Intersection mode: %d file_list, %d git_diff → %d candidates",
                len(file_list_normalized),
                len(git_diff_map),
                len(candidate_paths),
            )
        elif file_list_normalized:
            # File List only (create-story, dev-story)
            candidate_paths = file_list_normalized
            logger.debug("File List only mode: %d candidates", len(candidate_paths))
        elif git_diff_map:
            # Git diff only (no File List)
            candidate_paths = set(git_diff_map.keys())
            logger.debug("Git diff only mode: %d candidates", len(candidate_paths))
        else:
            logger.debug("No candidates: empty File List and git diff")
            return {}

        # Score files
        scored_files = self._score_files(
            list(candidate_paths),
            file_list_normalized,
            git_diff_map,
        )

        # Select files (apply max_files cap)
        selected_files = self._select_files(scored_files)

        # Extract content with adaptive mode
        return self._extract_all_content(selected_files)

    def _score_files(
        self,
        candidate_paths: list[str],
        file_list_set: set[str],
        git_diff_map: dict[str, GitDiffFile],
    ) -> list[ScoredFile]:
        """Score candidate files for prioritization.

        Args:
            candidate_paths: Normalized paths to score.
            file_list_set: Set of paths in File List.
            git_diff_map: Map of path to GitDiffFile.

        Returns:
            List of ScoredFile with scores calculated.

        """
        scoring = self.config.scoring
        scored: list[ScoredFile] = []

        for path in candidate_paths:
            abs_path = self.project_root / path

            if _should_exclude_path_for_workflow(path, self.workflow_name):
                logger.debug(
                    "Skipping low-value generated documentation for %s: %s",
                    self.workflow_name,
                    path,
                )
                continue

            # Skip if doesn't exist or is binary
            if not abs_path.exists():
                logger.debug("Skipping non-existent file: %s", path)
                continue
            if is_binary_file(abs_path):
                logger.debug("Skipping binary file: %s", path)
                continue

            # Security check
            try:
                if not abs_path.resolve().is_relative_to(self.project_root):
                    logger.debug("Skipping path outside project: %s", path)
                    continue
            except ValueError:
                continue

            # Read content for token estimation
            content = safe_read_file(abs_path)
            if content is None:
                logger.debug("Skipping unreadable file: %s", path)
                continue

            # Create scored file
            in_file_list = path in file_list_set
            git_diff_info = git_diff_map.get(path)
            in_git_diff = git_diff_info is not None
            is_test = _is_test_file(path)
            is_config = _is_config_file(path)
            relevance_rank, is_current_story_context = _relevance_rank(
                path, self.story_markers
            )
            change_lines = git_diff_info.change_lines if git_diff_info else 0
            hunk_ranges = git_diff_info.hunk_ranges if git_diff_info else []

            # Calculate score
            score = 0
            if in_file_list:
                score += scoring.in_file_list
            if in_git_diff:
                score += scoring.in_git_diff
            if is_test:
                score += scoring.is_test_file  # Usually negative
            if is_config:
                score += scoring.is_config_file  # Usually negative

            # Change lines contribution (capped)
            change_contribution = min(
                change_lines * scoring.change_lines_factor,
                scoring.change_lines_cap,
            )
            score += change_contribution

            sf = ScoredFile(
                path=path,
                score=score,
                tokens=estimate_tokens(content),
                in_file_list=in_file_list,
                in_git_diff=in_git_diff,
                is_test=is_test,
                is_config=is_config,
                relevance_rank=relevance_rank,
                is_current_story_context=is_current_story_context,
                change_lines=change_lines,
                hunk_ranges=list(hunk_ranges),
            )
            scored.append(sf)

            logger.debug(
                "Scored %s: score=%d rank=%d story=%s (file_list=%s, git_diff=%s, test=%s, config=%s, changes=%d)",
                path,
                score,
                relevance_rank,
                is_current_story_context,
                in_file_list,
                in_git_diff,
                is_test,
                is_config,
                change_lines,
            )

        return scored

    def _select_files(self, scored_files: list[ScoredFile]) -> list[ScoredFile]:
        """Select files to include based on score and max_files cap.

        Args:
            scored_files: List of scored files.

        Returns:
            Selected files sorted by score desc, then path asc.

        """
        if not scored_files:
            return []

        max_files = self.config.extraction.max_files

        # Story-driven workflows benefit from semantic ordering before score
        # because File List entries often share the same base score while stale
        # docs and artifact references are still present in the surrounding
        # planning context.
        def semantic_sort_key(file: ScoredFile) -> tuple[int, int, str]:
            return (-file.relevance_rank, -file.score, file.path)

        def score_sort_key(file: ScoredFile) -> tuple[int, int, str]:
            return (-file.score, -file.relevance_rank, file.path)

        sort_key = (
            semantic_sort_key
            if self.workflow_name in SEMANTIC_PRIORITY_WORKFLOWS
            else score_sort_key
        )

        sorted_files = sorted(scored_files, key=sort_key)

        # Apply max_files cap
        selected = sorted_files[:max_files]

        if len(sorted_files) > max_files:
            logger.debug(
                "Applied max_files cap: %d → %d files",
                len(sorted_files),
                len(selected),
            )

        return selected

    def _extract_all_content(
        self,
        selected_files: list[ScoredFile],
    ) -> dict[str, str]:
        """Extract content from selected files with adaptive mode.

        Args:
            selected_files: Files to extract content from.

        Returns:
            Dictionary mapping absolute paths to content.

        """
        if not selected_files:
            return {}

        extraction = self.config.extraction
        result: dict[str, str] = {}
        tokens_used = 0

        # Calculate threshold for adaptive mode
        effective_files = len(selected_files)
        threshold_tokens = (
            int((self.budget / effective_files) * extraction.adaptive_threshold)
            if effective_files > 0
            else self.budget
        )

        logger.debug(
            "Adaptive threshold: %d tokens (budget=%d, files=%d, threshold=%.2f)",
            threshold_tokens,
            self.budget,
            effective_files,
            extraction.adaptive_threshold,
        )

        for sf in selected_files:
            if tokens_used >= self.budget:
                logger.debug("Budget exhausted, stopping extraction")
                break

            abs_path = self.project_root / sf.path
            content = safe_read_file(abs_path)
            if content is None:
                continue

            # Decide: full file or hunks?
            # Single file edge case: always full if fits
            if len(selected_files) == 1 and sf.tokens <= self.budget:
                extracted = content
                mode = "full (single file)"
            elif sf.tokens <= threshold_tokens:
                extracted = content
                mode = "full"
            elif sf.hunk_ranges:
                extracted = self._extract_hunks(content, sf)
                mode = "hunks"
            else:
                # No hunk info, use full or truncate
                extracted = content
                mode = "full (no hunk info)"

            # Check if we need to truncate
            extracted_tokens = estimate_tokens(extracted)
            remaining_budget = self.budget - tokens_used

            if extracted_tokens > remaining_budget:
                extracted = self._truncate_at_symbol(
                    extracted,
                    remaining_budget,
                    abs_path.suffix,
                )
                extracted_tokens = estimate_tokens(extracted)
                mode += " + truncated"

            result[str(abs_path)] = extracted
            tokens_used += extracted_tokens

            logger.debug(
                "Extracted %s (%s): ~%d tokens, total: %d",
                sf.path,
                mode,
                extracted_tokens,
                tokens_used,
            )

        # Minimum guarantee: at least one file if available
        if not result and selected_files:
            sf = selected_files[0]
            abs_path = self.project_root / sf.path
            content = safe_read_file(abs_path)
            if content:
                truncated = self._truncate_at_symbol(
                    content,
                    self.budget,
                    abs_path.suffix,
                )
                result[str(abs_path)] = truncated
                logger.debug(
                    "Minimum guarantee: included %s with head truncation",
                    sf.path,
                )

        if result:
            logger.info(
                "Collected %d source files (~%d tokens) for %s",
                len(result),
                tokens_used,
                self.workflow_name,
            )

        return result

    def _extract_hunks(self, content: str, scored_file: ScoredFile) -> str:
        """Extract hunks (changed regions with context) from content.

        Args:
            content: Full file content.
            scored_file: Scored file with hunk_ranges.

        Returns:
            Extracted hunks with context.

        """
        if not scored_file.hunk_ranges:
            return content

        extraction = self.config.extraction
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Build context regions for each hunk
        regions: list[tuple[int, int, int]] = []  # (start, end, change_count)
        for start, end in scored_file.hunk_ranges:
            change_count = end - start + 1
            context_lines = max(
                extraction.hunk_context_lines,
                int(change_count * extraction.hunk_context_scale),
            )
            region_start = max(0, start - 1 - context_lines)  # Convert to 0-indexed
            region_end = min(total_lines, end + context_lines)
            regions.append((region_start, region_end, change_count))

        # Sort by change_count descending (largest hunks first per AC26)
        regions.sort(key=lambda r: -r[2])

        # Merge overlapping regions
        merged: list[tuple[int, int]] = []
        for start, end, _ in regions:
            if merged and start <= merged[-1][1]:
                # Overlapping - extend previous region
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Sort by position for output
        merged.sort(key=lambda r: r[0])

        # Build output
        output_parts: list[str] = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                output_parts.append("\n[... gap ...]\n\n")

            # Add line number prefix for first line of region
            output_parts.append(f"# Lines {start + 1}-{end}\n")
            output_parts.extend(lines[start:end])

        return "".join(output_parts)

    # Suffix-to-language mapping for shared context extractor
    _SUFFIX_TO_LANGUAGE: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
    }

    def _truncate_at_symbol(
        self,
        content: str,
        max_tokens: int,
        suffix: str,
    ) -> str:
        """Truncate content at symbol boundary.

        Delegates to shared context extractor for supported languages.
        Falls back to line-based truncation if extractor unavailable.

        Args:
            content: Full content.
            max_tokens: Maximum tokens to keep.
            suffix: File extension (e.g., '.py').

        Returns:
            Truncated content.

        """
        max_chars = max_tokens * 4  # 4 chars per token

        if len(content) <= max_chars:
            return content

        # Try shared context extractor for supported languages
        language = self._SUFFIX_TO_LANGUAGE.get(suffix)
        if language:
            try:
                from bmad_assist.context.extractor import extract_context
                from bmad_assist.context.formatter import format_for_source_context

                ctx = extract_context(content, f"file{suffix}", budget=max_chars, language=language)
                result = format_for_source_context(ctx, max_chars)
                if result:
                    return result
            except (ValueError, ImportError) as e:
                logger.debug("Shared extractor failed, using line-based: %s", e)

        # Line-based truncation fallback
        return self._truncate_at_line(content, max_chars)

    # TODO: remove after context/ module is validated
    def _truncate_python_at_symbol(self, content: str, max_chars: int) -> str:
        """Truncate Python code at function/class boundary.

        Args:
            content: Python source code.
            max_chars: Maximum characters to keep.

        Returns:
            Truncated content at symbol boundary.

        """
        tree = ast.parse(content)
        lines = content.splitlines(keepends=True)

        # Build symbol map: (start_line, end_line, name)
        symbols: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and hasattr(node, "lineno")
                and hasattr(node, "end_lineno")
            ):
                symbols.append((node.lineno, node.end_lineno or node.lineno, node.name))

        # Find the line where max_chars cuts
        char_count = 0
        cut_line = 0
        for i, line in enumerate(lines):
            char_count += len(line)
            if char_count >= max_chars:
                cut_line = i + 1  # 1-indexed
                break
        else:
            return content  # No truncation needed

        # Find innermost symbol containing cut_line
        containing_symbol: tuple[int, int, str] | None = None
        for start, end, name in symbols:
            if start <= cut_line <= end and (
                containing_symbol is None
                or (start >= containing_symbol[0] and end <= containing_symbol[1])
            ):
                containing_symbol = (start, end, name)

        if containing_symbol:
            start, end, name = containing_symbol
            # Calculate what fits
            chars_before_symbol = sum(len(lines[i]) for i in range(start - 1))
            chars_of_symbol = sum(len(lines[i]) for i in range(start - 1, end))

            # Symbol fits after symbol, otherwise truncate before
            truncate_line = end if chars_before_symbol + chars_of_symbol <= max_chars else start - 1
        else:
            # No symbol found - truncate at line
            truncate_line = cut_line

        if truncate_line <= 0:
            truncate_line = 1

        truncated_lines = lines[:truncate_line]
        truncated = "".join(truncated_lines)
        truncated += f"\n\n# ... truncated at line {truncate_line} (symbol boundary) ..."

        return truncated

    def _truncate_at_line(self, content: str, max_chars: int) -> str:
        """Truncate content at line boundary.

        Args:
            content: Content to truncate.
            max_chars: Maximum characters to keep.

        Returns:
            Truncated content at line boundary.

        """
        truncated = content[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
            line_count = truncated.count("\n") + 1
        else:
            line_count = 1

        truncated += f"\n\n[... TRUNCATED at line {line_count} due to token budget ...]"

        return truncated


def get_git_diff_files(
    project_root: Path,
    stat_output: str,
) -> list[GitDiffFile]:
    """Parse git diff stat output and get hunk ranges.

    Args:
        project_root: Project root for git commands.
        stat_output: Output from git diff --stat.

    Returns:
        List of GitDiffFile with change info.

    """
    # Extract files from stat
    files = _parse_git_stat(stat_output)

    # Get hunk ranges for each file
    result: list[GitDiffFile] = []
    for path, changes in files:
        hunk_ranges = get_hunk_ranges_for_file(project_root, path)
        result.append(
            GitDiffFile(
                path=path,
                change_lines=changes,
                hunk_ranges=hunk_ranges,
            )
        )

    return result


def _parse_git_stat(stat_output: str) -> list[tuple[str, int]]:
    """Parse git diff --stat output.

    Args:
        stat_output: Raw output from git diff --stat.

    Returns:
        List of (path, change_count) tuples.

    """
    # Pattern for standard lines: " src/file.py | 42 ++++++----"
    stat_pattern = re.compile(r"^\s*(.+?)\s*\|\s*(\d+)", re.MULTILINE)

    # Pattern for renamed files: " old.py => new.py | 5"
    rename_pattern = re.compile(r"^\s*(?:.+?)\s*=>\s*(.+?)\s*\|\s*(\d+)", re.MULTILINE)

    # Extract stat section (before summary line)
    stat_end = re.search(r"^\s*\d+\s+files?\s+changed", stat_output, re.MULTILINE | re.IGNORECASE)
    stat_section = stat_output[: stat_end.start()] if stat_end else stat_output

    result: list[tuple[str, int]] = []
    seen: set[str] = set()

    # Handle renames first
    for match in rename_pattern.finditer(stat_section):
        new_path = match.group(1).strip()
        changes = int(match.group(2))
        if new_path not in seen:
            result.append((new_path, changes))
            seen.add(new_path)

    # Handle standard entries
    for match in stat_pattern.finditer(stat_section):
        path = match.group(1).strip()
        changes = int(match.group(2))

        # Skip if already handled as rename or if it looks like binary
        if path in seen or "=>" in path or "Bin" in stat_output[match.start() : match.end() + 50]:
            continue

        result.append((path, changes))
        seen.add(path)

    return result


def get_hunk_ranges_for_file(project_root: Path, path: str) -> list[tuple[int, int]]:
    """Get hunk line ranges from git diff for a specific file.

    Args:
        project_root: Project root for git command.
        path: File path relative to project root.

    Returns:
        List of (start_line, end_line) tuples (1-indexed inclusive).

    """
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", "--no-color", "--", path],
            cwd=project_root,
            capture_output=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return []

        # Parse unified diff format
        # @@ -old_start,old_count +new_start,new_count @@
        hunk_pattern = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@", re.MULTILINE)

        ranges: list[tuple[int, int]] = []
        for match in hunk_pattern.finditer(result.stdout):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            if count > 0:
                ranges.append((start, start + count - 1))

        return ranges

    except (subprocess.TimeoutExpired, OSError):
        return []
