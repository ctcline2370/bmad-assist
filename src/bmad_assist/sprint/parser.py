"""Schema-tolerant parser for sprint-status files.

This module implements a parser that can read multiple discovered sprint-status
format variants and normalize them to the canonical SprintStatus model.

Supported format variants:
- FULL: metadata + development_status dict (production format)
- HYBRID: epics list of dicts + development_status
- ARRAY: epics as int/string array + development_status
- MINIMAL: empty epics array, optional current_epic/story/phase
- UNKNOWN: fallback for unrecognized structures

The parser gracefully handles malformed files with warnings rather than crashes,
enabling continued operation with partial data where possible.

Public API:
    - FormatVariant: Enum for detected format types
    - detect_format: Function to determine format variant from data
    - parse_sprint_status: Main entry point returning canonical model
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from bmad_assist.core.exceptions import ParserError
from bmad_assist.sprint.classifier import EntryType, classify_entry
from bmad_assist.sprint.models import (
    SprintStatus,
    SprintStatusEntry,
    SprintStatusMetadata,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FormatVariant",
    "detect_format",
    "parse_sprint_status",
]


class FormatVariant(Enum):
    """Detected format variant of sprint-status file.

    Each variant represents a different structural pattern found in
    sprint-status files across different BMAD projects.

    Variants:
        FULL: No 'epics' key, has 'development_status' dict.
            Example: production _bmad-output/ files with comments.

        HYBRID: 'epics' is list of dicts with 'id' key + development_status.
            Example: cli-dashboard fixture with epic metadata.

        ARRAY: 'epics' is list of integers/strings + development_status.
            Example: test-data-gen fixture with epic IDs.

        MINIMAL: 'epics' is empty list, optional current_epic/story/phase.
            Example: webhook-relay, auth-service fixtures for new projects.

        UNKNOWN: Unrecognized structure, fallback for graceful degradation.
    """

    FULL = "full"
    HYBRID = "hybrid"
    ARRAY = "array"
    MINIMAL = "minimal"
    UNKNOWN = "unknown"


def detect_format(data: dict[str, Any]) -> FormatVariant:
    """Detect sprint-status format variant from parsed YAML data.

    Uses heuristic-based detection with strict priority ordering to prevent
    ambiguity between formats. Detection handles edge cases gracefully.

    Priority order:
    1. FULL: No 'epics' key but has 'development_status' dict
    2. MINIMAL: 'epics' exists and is empty list
    3. HYBRID: 'epics' is list of dicts with 'id' key
    4. ARRAY: 'epics' is list of non-dicts (integers/strings)
    5. UNKNOWN: Anything else (fallback for unrecognized patterns)

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        FormatVariant indicating detected format.

    Examples:
        >>> detect_format({"development_status": {"1-1-story": "done"}})
        FormatVariant.FULL
        >>> detect_format({"epics": [], "current_epic": 1})
        FormatVariant.MINIMAL
        >>> detect_format({"epics": [{"id": 1, "title": "Epic"}]})
        FormatVariant.HYBRID
        >>> detect_format({"epics": [1, 2, 3]})
        FormatVariant.ARRAY

    """
    # Edge case: empty or None data
    if not data:
        return FormatVariant.UNKNOWN

    # Priority 1: FULL format - no epics key but has development_status
    if "epics" not in data:
        if "development_status" in data and isinstance(data.get("development_status"), dict):
            return FormatVariant.FULL
        return FormatVariant.UNKNOWN

    epics = data.get("epics")

    # Edge case: epics is not a list (string, dict, int, None)
    if not isinstance(epics, list):
        return FormatVariant.UNKNOWN

    # Priority 2: MINIMAL - empty epics list
    if not epics:
        return FormatVariant.MINIMAL

    # Check first item to determine variant
    first_item = epics[0]

    # Priority 3: HYBRID - list of dicts with 'id' key
    if isinstance(first_item, dict) and "id" in first_item:
        return FormatVariant.HYBRID

    # Priority 4: ARRAY - list of integers or strings
    if isinstance(first_item, (int, str)):
        # Verify all items are same type (not mixed)
        if all(isinstance(item, (int, str)) for item in epics):
            return FormatVariant.ARRAY
        # Mixed types in list
        return FormatVariant.UNKNOWN

    # Default: unrecognized structure
    return FormatVariant.UNKNOWN


def _parse_generated(value: str | datetime | None) -> datetime:
    """Parse generated timestamp from various formats.

    Handles multiple date/datetime formats commonly found in sprint-status files.

    Args:
        value: Timestamp value from YAML (string, datetime, date, or None).

    Returns:
        Parsed datetime. Falls back to current UTC time if parsing fails.

    """
    if value is None:
        return datetime.now(UTC).replace(tzinfo=None)

    if isinstance(value, datetime):
        return value

    # YAML date objects come as datetime.date
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        # It's a date object
        from datetime import date

        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)

    # Try string parsing
    value_str = str(value)

    # Try ISO format first (most common)
    try:
        return datetime.fromisoformat(value_str)
    except ValueError:
        pass

    # Try date-only format (YYYY-MM-DD)
    try:
        return datetime.strptime(value_str, "%Y-%m-%d")
    except ValueError:
        pass

    # Fallback to current time with warning
    logger.warning(
        "Could not parse 'generated' date: %s, using current time",
        value,
    )
    return datetime.now(UTC).replace(tzinfo=None)


def _extract_metadata(data: dict[str, Any]) -> SprintStatusMetadata:
    """Extract metadata fields from sprint-status data.

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        SprintStatusMetadata with extracted fields.

    """
    return SprintStatusMetadata(
        generated=_parse_generated(data.get("generated")),
        last_updated=(
            _parse_generated(data.get("last_updated"))
            if data.get("last_updated") is not None
            else None
        ),
        project=data.get("project"),
        project_key=data.get("project_key"),
        tracking_system=data.get("tracking_system"),
        story_location=data.get("story_location"),
    )


def _parse_dev_status_entry(key: str, status: str) -> SprintStatusEntry:
    """Parse a single development_status entry.

    Args:
        key: Entry key (e.g., "12-3-story-name").
        status: Status value (e.g., "done", "in-progress").

    Returns:
        SprintStatusEntry with normalized status. Invalid statuses are
        normalized to "backlog" with a logged warning.

    """
    # Normalize status
    status_lower = str(status).lower().strip()

    # Map common variations to valid statuses
    status_mapping = {
        "backlog": "backlog",
        "ready-for-dev": "ready-for-dev",
        "ready_for_dev": "ready-for-dev",
        "readyfordev": "ready-for-dev",
        "in-progress": "in-progress",
        "in_progress": "in-progress",
        "inprogress": "in-progress",
        "review": "review",
        "done": "done",
        "blocked": "blocked",
        "deferred": "deferred",
        "optional": "optional",
        # Legacy status aliases
        "drafted": "backlog",  # treat drafted as backlog
        "completed": "done",  # treat completed as done
    }

    normalized_status = status_mapping.get(status_lower)
    if normalized_status is None:
        logger.warning(
            "Invalid status '%s' for key '%s', treating as 'backlog'",
            status,
            key,
        )
        normalized_status = "backlog"

    return SprintStatusEntry(
        key=key,
        status=normalized_status,  # type: ignore[arg-type]
        entry_type=classify_entry(key),
        source="sprint-status",
        comment=None,  # Comment extraction deferred to Story 20.8
    )


def _parse_development_status(data: dict[str, Any]) -> dict[str, SprintStatusEntry]:
    """Parse development_status section into entries dict.

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Dict of key -> SprintStatusEntry preserving insertion order.

    """
    dev_status = data.get("development_status")

    # Handle missing development_status
    if dev_status is None:
        return {}

    # Handle non-dict development_status
    if not isinstance(dev_status, dict):
        logger.warning(
            "development_status is %s, expected dict. Returning empty entries.",
            type(dev_status).__name__,
        )
        return {}

    entries: dict[str, SprintStatusEntry] = {}
    for key, status in dev_status.items():
        if not isinstance(key, str):
            key = str(key)

        entry = _parse_dev_status_entry(key, str(status))
        entries[key] = entry

    return entries


def _parse_full_format(data: dict[str, Any]) -> SprintStatus:
    """Parse FULL format sprint-status.

    FULL format has:
    - Metadata fields at top level (generated, project, etc.)
    - development_status dict with entries

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Normalized SprintStatus model.

    """
    metadata = _extract_metadata(data)
    entries = _parse_development_status(data)

    return SprintStatus(
        metadata=metadata,
        entries=entries,
    )


def _parse_hybrid_format(data: dict[str, Any]) -> SprintStatus:
    """Parse HYBRID format sprint-status.

    HYBRID format has:
    - epics: list of dicts with {id, title, status}
    - development_status dict with story entries
    - Optional metadata fields

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Normalized SprintStatus model.

    """
    metadata = _extract_metadata(data)
    entries: dict[str, SprintStatusEntry] = {}

    # First, extract epic metadata entries
    epics = data.get("epics", [])
    for epic_data in epics:
        if not isinstance(epic_data, dict):
            continue

        epic_id = epic_data.get("id")
        if epic_id is None:
            continue

        epic_key = f"epic-{epic_id}"
        epic_status = str(epic_data.get("status", "backlog")).lower()

        # Reuse centralized status parsing and override entry_type for epics
        epic_entry = _parse_dev_status_entry(epic_key, epic_status)
        entries[epic_key] = SprintStatusEntry(
            key=epic_key,
            status=epic_entry.status,
            entry_type=EntryType.EPIC_META,
            source="sprint-status",
            comment=None,
        )

    # Then add story entries from development_status
    story_entries = _parse_development_status(data)
    entries.update(story_entries)

    return SprintStatus(
        metadata=metadata,
        entries=entries,
    )


def _parse_array_format(data: dict[str, Any]) -> SprintStatus:
    """Parse ARRAY format sprint-status.

    ARRAY format has:
    - epics: list of integers or strings (epic IDs)
    - development_status dict with story entries
    - Optional current_epic, current_story, phase

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Normalized SprintStatus model.

    """
    metadata = _extract_metadata(data)
    entries: dict[str, SprintStatusEntry] = {}

    # Create placeholder epic entries from array
    epics = data.get("epics", [])
    for epic_id in epics:
        epic_key = f"epic-{epic_id}"
        entries[epic_key] = SprintStatusEntry(
            key=epic_key,
            status="backlog",  # Default status for array format
            entry_type=EntryType.EPIC_META,
            source="sprint-status",
            comment=None,
        )

    # Add story entries from development_status
    story_entries = _parse_development_status(data)
    entries.update(story_entries)

    return SprintStatus(
        metadata=metadata,
        entries=entries,
    )


def _parse_minimal_format(data: dict[str, Any]) -> SprintStatus:
    """Parse MINIMAL format sprint-status.

    MINIMAL format has:
    - epics: empty list []
    - Optional: current_epic, current_story, phase
    - No development_status (or empty)

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Normalized SprintStatus model with empty entries.

    """
    metadata = _extract_metadata(data)

    # Minimal format typically has no entries
    entries = _parse_development_status(data)

    return SprintStatus(
        metadata=metadata,
        entries=entries,
    )


def _parse_unknown_format(data: dict[str, Any]) -> SprintStatus:
    """Parse UNKNOWN format sprint-status.

    Attempts best-effort parsing of unrecognized format by:
    1. Extracting any metadata fields present
    2. Attempting to parse development_status if present

    Args:
        data: Parsed YAML data dictionary.

    Returns:
        Normalized SprintStatus with whatever was parseable.

    """
    metadata = _extract_metadata(data)
    entries = _parse_development_status(data)

    return SprintStatus(
        metadata=metadata,
        entries=entries,
    )


def parse_sprint_status(path: Path | str) -> SprintStatus:
    """Parse sprint-status file and normalize to canonical model.

    Main entry point for parsing sprint-status files. Handles multiple
    format variants and normalizes all to the canonical SprintStatus model.

    Error handling:
    - FileNotFoundError: Raises ParserError with file path
    - yaml.YAMLError: Logs WARNING, returns SprintStatus.empty()
    - Empty file: Returns SprintStatus.empty(), logs INFO
    - Type mismatches: Logs WARNING, returns partial result

    Args:
        path: Path to sprint-status.yaml file.

    Returns:
        Normalized SprintStatus model.

    Raises:
        ParserError: If file does not exist.

    Examples:
        >>> status = parse_sprint_status(Path("sprint-status.yaml"))
        >>> status.get_epic_status(12)
        'done'

    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        raise ParserError(f"Sprint status file not found: {path}")

    # Read and parse YAML
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning("Failed to parse sprint-status YAML at %s: %s", path, e)
        return SprintStatus.empty()

    # Handle empty file
    if data is None:
        logger.info("Sprint-status file is empty: %s", path)
        return SprintStatus.empty()

    # Handle non-dict at root
    if not isinstance(data, dict):
        logger.warning(
            "Sprint-status root is %s, expected dict: %s",
            type(data).__name__,
            path,
        )
        return SprintStatus.empty()

    # Detect format and parse
    variant = detect_format(data)
    logger.debug("Detected sprint-status format: %s for %s", variant.value, path)

    if variant == FormatVariant.FULL:
        return _parse_full_format(data)
    elif variant == FormatVariant.HYBRID:
        return _parse_hybrid_format(data)
    elif variant == FormatVariant.ARRAY:
        return _parse_array_format(data)
    elif variant == FormatVariant.MINIMAL:
        return _parse_minimal_format(data)
    else:
        return _parse_unknown_format(data)
