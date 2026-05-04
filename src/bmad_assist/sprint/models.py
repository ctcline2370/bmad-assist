"""Canonical Pydantic models for sprint-status representation.

This module defines the normalized data structures for sprint-status files.
All format variants are parsed and normalized to these canonical models,
enabling consistent querying, serialization, and comment preservation.

The models support:
- Entry ordering preservation (Python dict insertion order since 3.7+)
- Inline YAML comment extraction and storage
- Helper methods for epic/story queries
- Serialization to YAML format

Public API:
    - SprintStatusMetadata: Metadata header fields
    - SprintStatusEntry: Single status entry
    - SprintStatus: Container with entries and helpers
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from bmad_assist.core.types import EpicId
from bmad_assist.sprint.classifier import EntryType

logger = logging.getLogger(__name__)

# Valid status values for sprint entries
ValidStatus = Literal[
    "backlog",
    "ready-for-dev",
    "in-progress",
    "review",
    "done",
    "blocked",
    "deferred",
    "optional",
]


@lru_cache(maxsize=64)
def _get_epic_pattern(epic_str: str) -> re.Pattern[str]:
    """Get cached regex pattern for epic story matching.

    Pattern matches: exact epic_id followed by dash and story number start.
    This prevents prefix collision (e.g., epic "1" won't match "12-3-story").

    Args:
        epic_str: Epic identifier as string.

    Returns:
        Compiled regex pattern.

    """
    return re.compile(rf"^{re.escape(epic_str)}-\d")


if TYPE_CHECKING:
    pass

__all__ = [
    "ValidStatus",
    "SprintStatusMetadata",
    "SprintStatusEntry",
    "SprintStatus",
]


class SprintStatusMetadata(BaseModel):
    """Metadata header fields for sprint-status file.

    All fields are optional except `generated` to support legacy formats
    and minimal sprint-status files.

    Attributes:
        generated: Timestamp when status was generated (required).
        last_updated: Timestamp when status was last written.
        project: Project name.
        project_key: Short project identifier.
        tracking_system: Tracking system name (e.g., "BMAD").
        story_location: Path to story files.

    Example:
        >>> meta = SprintStatusMetadata(generated=datetime.now())
        >>> meta.project is None
        True

    """

    generated: datetime = Field(
        ...,
        description="Timestamp when sprint-status was generated (UTC)",
    )
    last_updated: datetime | None = Field(
        None,
        description="Timestamp when sprint-status was last written (UTC)",
    )
    project: str | None = Field(
        None,
        description="Project name",
    )
    project_key: str | None = Field(
        None,
        description="Short project identifier",
    )
    tracking_system: str | None = Field(
        None,
        description="Tracking system name (e.g., 'BMAD')",
    )
    story_location: str | None = Field(
        None,
        description="Path to story files directory",
    )


class SprintStatusEntry(BaseModel):
    """Single entry in sprint-status file.

    Represents one line in the development_status section with
    classification metadata for merge behavior control.

    Attributes:
        key: Sprint-status key (e.g., "12-3-story-name").
        status: Status value (backlog|ready-for-dev|in-progress|review|done|blocked|deferred).
        entry_type: Classification for merge behavior (from classifier.py).
        source: Origin of entry ("epic", "sprint-status", "artifact").
        comment: Inline YAML comment if present.

    Example:
        >>> entry = SprintStatusEntry(
        ...     key="12-3-auth-flow",
        ...     status="done",
        ...     entry_type=EntryType.EPIC_STORY,
        ... )
        >>> entry.key
        '12-3-auth-flow'

    """

    key: str = Field(
        ...,
        description="Sprint-status key (e.g., '12-3-story-name')",
    )
    status: ValidStatus = Field(
        ...,
        description="Status: backlog|ready-for-dev|in-progress|review|done|blocked|deferred|optional",
    )
    entry_type: EntryType = Field(
        default=EntryType.UNKNOWN,
        description="Entry classification for merge behavior",
    )
    source: str | None = Field(
        None,
        description="Origin: 'epic', 'sprint-status', 'artifact'",
    )
    comment: str | None = Field(
        None,
        description="Inline YAML comment if present",
    )

    def __repr__(self) -> str:
        """Return debug-friendly representation."""
        return (
            f"SprintStatusEntry(key={self.key!r}, status={self.status!r}, "
            f"entry_type={self.entry_type.value})"
        )


class SprintStatus(BaseModel):
    """Container for sprint-status file content.

    Holds metadata and all entries with helper methods for common queries.
    Entry ordering is preserved via Python dict (3.7+ guarantees insertion order).

    Attributes:
        metadata: Sprint-status metadata header.
        entries: Dict of entries keyed by sprint-status key (preserves insertion order).

    Example:
        >>> status = SprintStatus(
        ...     metadata=SprintStatusMetadata(generated=datetime.now()),
        ...     entries={},
        ... )
        >>> status.get_epic_status(12)
        None

    """

    metadata: SprintStatusMetadata = Field(
        ...,
        description="Sprint-status metadata header",
    )
    entries: dict[str, SprintStatusEntry] = Field(
        default_factory=dict,
        description="Dict of entries keyed by sprint-status key (preserves order)",
    )

    def get_stories_for_epic(self, epic_id: EpicId) -> list[SprintStatusEntry]:
        """Get all story entries belonging to an epic.

        Finds entries that match the epic ID followed by a story number.
        Uses strict prefix matching to prevent collisions (e.g., epic 1
        should not match "12-3-story").

        Args:
            epic_id: Epic identifier (int or str, e.g., 12 or "testarch").

        Returns:
            List of SprintStatusEntry for stories in this epic, preserving order.
            Includes EPIC_STORY and MODULE_STORY entries.
            Returns empty list if epic has no stories.

        Example:
            >>> status.get_stories_for_epic(12)
            [SprintStatusEntry(key='12-1-setup', ...), ...]
            >>> status.get_stories_for_epic("testarch")
            [SprintStatusEntry(key='testarch-1-config', ...), ...]

        """
        results: list[SprintStatusEntry] = []
        epic_str = str(epic_id)
        # Use cached pattern for performance
        pattern = _get_epic_pattern(epic_str)

        for entry in self.entries.values():
            if entry.entry_type in (EntryType.EPIC_STORY, EntryType.MODULE_STORY) and pattern.match(
                entry.key
            ):
                results.append(entry)

        return results

    def get_epic_status(self, epic_id: EpicId) -> str | None:
        """Get status of an epic entry.

        Looks up the epic-level status entry (e.g., "epic-12" or "epic-testarch").

        Args:
            epic_id: Epic identifier (int or str).

        Returns:
            Status string if epic entry exists, None otherwise.

        Example:
            >>> status.get_epic_status(12)
            'done'
            >>> status.get_epic_status(99)
            None

        """
        epic_key = f"epic-{epic_id}"
        entry = self.entries.get(epic_key)
        return entry.status if entry else None

    def to_yaml(self) -> str:
        """Serialize to YAML string.

        Produces YAML representation of sprint status in the standard format.

        Warning:
            Inline comments stored in entry metadata are NOT preserved in the
            YAML output. PyYAML does not support comment preservation. Use the
            AtomicWriter from Story 20.8 for comment-preserving writes.

        Returns:
            YAML representation of sprint status.

        Example:
            >>> yaml_str = status.to_yaml()
            >>> 'development_status:' in yaml_str
            True

        """
        # Build serializable structure
        data: dict[str, object] = {
            "generated": self.metadata.generated.isoformat(),
        }
        if self.metadata.last_updated:
            data["last_updated"] = self.metadata.last_updated.isoformat()

        # Add optional metadata fields if present
        if self.metadata.project:
            data["project"] = self.metadata.project
        if self.metadata.project_key:
            data["project_key"] = self.metadata.project_key
        if self.metadata.tracking_system:
            data["tracking_system"] = self.metadata.tracking_system
        if self.metadata.story_location:
            data["story_location"] = self.metadata.story_location

        # Add development_status section with entry statuses
        data["development_status"] = {key: entry.status for key, entry in self.entries.items()}

        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def empty(cls, project: str | None = None) -> SprintStatus:
        """Create empty SprintStatus with default metadata.

        Factory method for creating a new sprint-status with no entries.
        Useful for initialization before populating from epic files.

        Args:
            project: Optional project name for metadata.

        Returns:
            New SprintStatus instance with empty entries.

        Example:
            >>> status = SprintStatus.empty("bmad-assist")
            >>> len(status.entries)
            0
            >>> status.metadata.project
            'bmad-assist'

        """
        from datetime import UTC

        return cls(
            metadata=SprintStatusMetadata(
                generated=datetime.now(UTC).replace(tzinfo=None),
                project=project,
            ),
            entries={},
        )

    @classmethod
    def from_entries(
        cls,
        entries: list[SprintStatusEntry],
        metadata: SprintStatusMetadata | None = None,
    ) -> SprintStatus:
        """Build SprintStatus from entry list.

        Factory method for creating sprint-status from a list of entries.
        Entry ordering is preserved from the list order.

        Args:
            entries: List of SprintStatusEntry to include.
            metadata: Optional metadata. If None, generates default metadata.

        Returns:
            New SprintStatus instance with entries in list order.

        Example:
            >>> entries = [
            ...     SprintStatusEntry(key="1-1-setup", status="done"),
            ...     SprintStatusEntry(key="1-2-config", status="in-progress"),
            ... ]
            >>> status = SprintStatus.from_entries(entries)
            >>> list(status.entries.keys())
            ['1-1-setup', '1-2-config']

        """
        from datetime import UTC

        if metadata is None:
            metadata = SprintStatusMetadata(
                generated=datetime.now(UTC).replace(tzinfo=None),
            )

        # Check for duplicate keys and warn
        entries_dict: dict[str, SprintStatusEntry] = {}
        for entry in entries:
            if entry.key in entries_dict:
                logger.warning(
                    "Duplicate key '%s' in from_entries() - later entry overwrites earlier",
                    entry.key,
                )
            entries_dict[entry.key] = entry

        return cls(
            metadata=metadata,
            entries=entries_dict,
        )
