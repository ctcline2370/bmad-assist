"""Base resolver ABC for TEA artifact loading.

This module provides the abstract base class for all TEA context resolvers.
Each resolver handles a specific artifact type and implements pattern
matching and content loading.

Addresses:
- F3 Fix: Explicit estimate_tokens import from compiler/shared_utils.py
- F10 Fix: Shared _truncate_content() implementation
- F15 Fix: Exception handling with _safe_read()
- F17 Fix: Path traversal prevention via validate_artifact_path()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from bmad_assist.compiler.shared_utils import estimate_tokens
from bmad_assist.testarch.paths import (
    get_artifact_dir,
    get_artifact_search_dirs,
    validate_artifact_path,
)

if TYPE_CHECKING:
    from bmad_assist.core.types import EpicId

logger = logging.getLogger(__name__)


class BaseResolver(ABC):
    """Abstract base class for TEA artifact resolvers.

    Each resolver handles a specific type of TEA artifact (test-design,
    atdd, test-review, trace) and implements pattern matching and
    content loading with token budget management.

    Attributes:
        _base_path: Primary base directory for artifact search.
        _base_paths: Ordered base directories for artifact search.
        _max_tokens: Maximum tokens for this resolver's artifacts.

    """

    def __init__(self, base_path: Path | list[Path] | tuple[Path, ...], max_tokens: int) -> None:
        """Initialize resolver.

        Args:
            base_path: Base directory or ordered base directories for artifact search.
            max_tokens: Maximum tokens budget for this resolver.

        """
        if isinstance(base_path, Path):
            base_paths = [base_path]
        else:
            base_paths = [Path(path) for path in base_path]

        self._base_paths = self._dedupe_base_paths(base_paths)
        self._base_path = self._base_paths[0]
        self._max_tokens = max_tokens

    def _dedupe_base_paths(self, base_paths: list[Path]) -> tuple[Path, ...]:
        """Return base paths with duplicates removed while preserving order."""
        result: list[Path] = []
        seen: set[Path] = set()
        for base_path in base_paths:
            resolved = base_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(base_path)
        if not result:
            result.append(Path(".").resolve())
        return tuple(result)

    @property
    @abstractmethod
    def artifact_type(self) -> str:
        """Return artifact type identifier.

        Returns:
            Artifact type string (e.g., 'test-design', 'atdd').

        """
        ...

    @abstractmethod
    def resolve(
        self,
        epic_id: EpicId,
        story_id: str | None = None,
    ) -> dict[str, str]:
        """Resolve artifacts for the given context.

        Args:
            epic_id: Epic identifier (int or str).
            story_id: Optional story identifier.

        Returns:
            Dict mapping file paths to content.
            Empty dict if no artifacts found.

        """
        ...

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content at markdown boundaries.

        Finds a sensible truncation point near the token limit,
        preferring to break at headers or blank lines.

        Args:
            content: Content to potentially truncate.
            max_tokens: Maximum allowed tokens.

        Returns:
            Content truncated if needed, with marker appended.

        """
        tokens = estimate_tokens(content)
        if tokens <= max_tokens:
            return content

        # Find truncation point at markdown boundary
        lines = content.split("\n")
        truncated_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = estimate_tokens(line + "\n")
            if current_tokens + line_tokens > max_tokens:
                # Try to end at a sensible boundary
                break
            truncated_lines.append(line)
            current_tokens += line_tokens

        result = "\n".join(truncated_lines)
        result += "\n\n<!-- truncated: exceeded token budget -->"
        return result

    def _safe_read(self, path: Path) -> str | None:
        """Read file with path validation and exception handling.

        Args:
            path: File path to read.

        Returns:
            File content or None if read failed.

        Note:
            - Validates path is within base_path (F17 Fix)
            - Handles FileNotFoundError, PermissionError, UnicodeDecodeError (F15 Fix)
            - Skips empty files with warning

        """
        # F17: Path traversal protection across every configured search root.
        if not any(validate_artifact_path(path, base_path) for base_path in self._base_paths):
            logger.warning(
                "Path traversal attempt blocked: %s (outside TEA search roots: %s)",
                path,
                ", ".join(str(base_path) for base_path in self._base_paths),
            )
            return None

        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning("Empty artifact file: %s", path)
                return None
            return content
        except FileNotFoundError:
            logger.debug("Artifact not found: %s", path)
            return None
        except PermissionError as e:
            logger.error("Permission denied reading artifact: %s - %s", path, e)
            return None
        except UnicodeDecodeError as e:
            logger.error("Encoding error reading artifact: %s - %s", path, e)
            return None

    def _find_matching_files(self, pattern: str, subdir: str = "") -> list[Path]:
        """Find files matching glob pattern.

        Args:
            pattern: Glob pattern to match.
            subdir: Optional subdirectory within base_path to search in.

        Returns:
            List of matching paths, sorted by name.

        """
        search_subdirs = get_artifact_search_dirs(self.artifact_type) if subdir else [""]
        matches: list[Path] = []
        seen_paths: set[Path] = set()

        for base_path in self._base_paths:
            for search_subdir in search_subdirs:
                search_path = base_path / search_subdir if search_subdir else base_path
                if not search_path.exists():
                    logger.debug("Search path does not exist: %s", search_path)
                    continue

                for match in sorted(search_path.glob(pattern)):
                    resolved = match.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    matches.append(match)

        return matches

    def _get_artifact_dir(self) -> str:
        """Get subdirectory for this artifact type.

        Returns:
            Subdirectory relative to base_path.

        """
        return get_artifact_dir(self.artifact_type)
