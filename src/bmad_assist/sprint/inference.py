"""Evidence-based status inference for sprint-status entries.

This module provides the status inference engine that determines story and epic
status from artifact evidence using a prioritized hierarchy. The inference
algorithm examines project artifacts (story files, code reviews, validations,
test reviews, retrospectives) to determine the most accurate status for each
entry.

Evidence Hierarchy (highest to lowest priority):
1. Story file Status: field → EXPLICIT (authoritative unless validated-done is required)
2. Master code review or synthesis exists → STRONG (story = done)
3. Any code review exists (validator reviews) → MEDIUM (story = review)
4. Validation report exists → MEDIUM (story = ready-for-dev)
5. Story file exists (without Status) → WEAK (story = in-progress)
6. No evidence found → NONE (preserve existing or backlog)

Public API:
    - InferenceConfidence: Enum for confidence levels
    - InferenceResult: Dataclass for inference results with evidence
    - infer_story_status: Infer story status from artifacts
    - infer_story_status_detailed: Infer with full evidence audit trail
    - infer_epic_status: Infer epic status from story statuses
    - infer_all_statuses: Batch inference for all stories
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from bmad_assist.core.types import EpicId
from bmad_assist.sprint.models import ValidStatus

if TYPE_CHECKING:
    from bmad_assist.sprint.scanner import ArtifactIndex

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceConfidence",
    "InferenceResult",
    "infer_story_status",
    "infer_story_status_detailed",
    "infer_epic_status",
    "infer_all_statuses",
]


# ============================================================================
# InferenceConfidence Enum
# ============================================================================


class InferenceConfidence(IntEnum):
    """Confidence level of status inference.

    Values ordered by priority (higher = more confident).
    Using IntEnum allows natural comparison operators.

    Attributes:
        NONE: No evidence found, using default.
        WEAK: Story file exists only (no Status field).
        MEDIUM: Validator review or validation exists.
        STRONG: Master review or synthesis exists.
        EXPLICIT: Status field in story file (authoritative).

    Examples:
        >>> InferenceConfidence.EXPLICIT > InferenceConfidence.STRONG
        True
        >>> InferenceConfidence.NONE < InferenceConfidence.WEAK
        True

    """

    NONE = 0  # No evidence
    WEAK = 1  # Story file exists only
    MEDIUM = 2  # Validator review or validation exists
    STRONG = 3  # Master review or synthesis exists
    EXPLICIT = 4  # Status field in story file

    def __str__(self) -> str:
        """Return lowercase name for display."""
        return self.name.lower()


# ============================================================================
# InferenceResult Dataclass
# ============================================================================


@dataclass(frozen=True)
class InferenceResult:
    """Result of status inference with evidence audit trail.

    Immutable dataclass containing the inferred status, confidence level,
    and list of artifact paths that contributed to the inference.

    Attributes:
        key: Story or epic key that was inferred.
        status: Inferred status value (ValidStatus).
        confidence: Confidence level of inference.
        evidence_sources: Tuple of artifact paths used in inference.

    Examples:
        >>> result = InferenceResult(
        ...     key="20-1-entry-classification",
        ...     status="done",
        ...     confidence=InferenceConfidence.STRONG,
        ...     evidence_sources=(Path("/project/code-reviews/synthesis-20-1.md"),),
        ... )
        >>> result.confidence > InferenceConfidence.MEDIUM
        True

    """

    key: str
    status: ValidStatus
    confidence: InferenceConfidence
    evidence_sources: tuple[Path, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        """Return debug-friendly representation."""
        return (
            f"InferenceResult(key={self.key!r}, status={self.status!r}, "
            f"confidence={self.confidence.name})"
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _normalize_status(value: str) -> ValidStatus | None:
    """Normalize status string to ValidStatus enum value.

    Case-insensitive matching. Handles common variations like underscores
    and spaces being converted to dashes.

    Args:
        value: Raw status string from story file.

    Returns:
        Normalized status value if valid, None otherwise.

    Examples:
        >>> _normalize_status("Done")
        'done'
        >>> _normalize_status("IN-PROGRESS")
        'in-progress'
        >>> _normalize_status("ready_for_dev")
        'ready-for-dev'
        >>> _normalize_status("invalid")
        None

    """
    if not value or not value.strip():
        return None

    # Normalize: lowercase, strip, replace underscores/spaces with dashes
    normalized = value.lower().strip().replace("_", "-").replace(" ", "-")

    # Check against valid status values
    valid_statuses = get_args(ValidStatus)
    if normalized in valid_statuses:
        # Cast to ValidStatus since we've verified it's a valid value
        return normalized  # type: ignore[return-value]
    return None


def _get_story_keys_for_epic(epic_id: EpicId, index: ArtifactIndex) -> list[str]:
    """Find all story keys belonging to an epic.

    Scans index.story_files for keys matching pattern {epic_id}-{story_num}-{slug}.
    Uses regex pattern matching to ensure exact epic ID prefix matching
    (prevents epic "1" from matching "12-3-story").

    Args:
        epic_id: Epic identifier (int or str like "testarch").
        index: ArtifactIndex with scanned artifacts.

    Returns:
        List of story keys belonging to the epic, sorted by story number.

    Examples:
        >>> _get_story_keys_for_epic(20, index)
        ['20-1-entry-classification', '20-2-models', '20-3-parser']
        >>> _get_story_keys_for_epic("testarch", index)
        ['testarch-1-config', 'testarch-2-framework']

    """
    # Pattern: exact epic_id followed by dash and story number
    pattern = re.compile(rf"^{re.escape(str(epic_id))}-(\d+)-")
    matches: list[tuple[int, str]] = []

    for key in index.story_files:
        match = pattern.match(key)
        if match:
            story_num = int(match.group(1))
            matches.append((story_num, key))

    # Sort by story number
    matches.sort(key=lambda x: x[0])
    return [key for _, key in matches]


# ============================================================================
# Story Status Inference
# ============================================================================


def infer_story_status(
    story_key: str,
    index: ArtifactIndex,
    *,
    require_test_review_for_done: bool = False,
) -> tuple[ValidStatus, InferenceConfidence]:
    """Infer story status from artifact evidence.

    Checks evidence in strict priority order and returns the first match.
    See module docstring for Evidence Hierarchy.

    This is a convenience wrapper around infer_story_status_detailed()
    that returns just the status and confidence tuple.

    Args:
        story_key: Story key (full or short format supported).
        index: ArtifactIndex with scanned artifacts.
        require_test_review_for_done: When true, story "done" inference
            requires both code-review synthesis/master evidence and a
            test-review artifact.

    Returns:
        Tuple of (inferred_status, confidence_level).

    Examples:
        >>> index = ArtifactIndex.scan(project_root)
        >>> status, confidence = infer_story_status("20-1", index)
        >>> status
        'done'
        >>> confidence
        InferenceConfidence.STRONG

    """
    result = infer_story_status_detailed(
        story_key,
        index,
        require_test_review_for_done=require_test_review_for_done,
    )
    return result.status, result.confidence


def infer_story_status_detailed(
    story_key: str,
    index: ArtifactIndex,
    *,
    require_test_review_for_done: bool = False,
) -> InferenceResult:
    """Infer story status with full evidence audit trail.

    Similar to infer_story_status but returns InferenceResult with all
    artifact paths that contributed to the inference decision.

    Args:
        story_key: Story key (full or short format supported).
        index: ArtifactIndex with scanned artifacts.
        require_test_review_for_done: When true, story "done" inference
            requires both code-review synthesis/master evidence and a
            test-review artifact.

    Returns:
        InferenceResult with status, confidence, and evidence sources.

    Examples:
        >>> result = infer_story_status_detailed("20-1", index)
        >>> result.confidence
        InferenceConfidence.STRONG
        >>> len(result.evidence_sources)
        1

    """
    evidence: list[Path] = []

    # Priority 1: Explicit Status field in story file
    raw_status = index.get_story_status(story_key)
    story_artifact = index.get_story_artifact(story_key)

    if raw_status is not None:
        normalized = _normalize_status(raw_status)
        if normalized is not None:
            if normalized == "done" and require_test_review_for_done:
                logger.warning(
                    "Story %s: Status: done ignored until durable completion evidence exists",
                    story_key,
                )
            else:
                if story_artifact:
                    evidence.append(story_artifact.path)
                logger.debug(
                    "Story %s: EXPLICIT status '%s' from story file",
                    story_key,
                    normalized,
                )
                return InferenceResult(
                    key=story_key,
                    status=normalized,
                    confidence=InferenceConfidence.EXPLICIT,
                    evidence_sources=tuple(evidence),
                )
            if story_artifact:
                evidence.append(story_artifact.path)
        else:
            logger.warning(
                "Story %s: Invalid status '%s' in story file, falling through",
                story_key,
                raw_status,
            )

    # Priority 2: Master code review or synthesis exists
    if index.has_master_review(story_key):
        # Collect master review paths
        reviews = index.get_code_reviews(story_key)
        for review in reviews:
            if review.is_master or review.is_synthesis:
                evidence.append(review.path)

        if require_test_review_for_done:
            test_reviews = index.get_test_reviews(story_key)
            if test_reviews:
                for test_review in test_reviews:
                    evidence.append(test_review.path)
            else:
                logger.debug(
                    "Story %s: STRONG confidence 'review' - master review exists "
                    "but test review is missing",
                    story_key,
                )
                return InferenceResult(
                    key=story_key,
                    status="review",
                    confidence=InferenceConfidence.STRONG,
                    evidence_sources=tuple(evidence),
                )

        logger.debug(
            "Story %s: STRONG confidence 'done' - master review exists",
            story_key,
        )
        return InferenceResult(
            key=story_key,
            status="done",
            confidence=InferenceConfidence.STRONG,
            evidence_sources=tuple(evidence),
        )

    # Priority 3: Any code review exists (validators)
    if index.has_any_review(story_key):
        reviews = index.get_code_reviews(story_key)
        for review in reviews:
            evidence.append(review.path)
        logger.debug(
            "Story %s: MEDIUM confidence 'review' - validator reviews exist",
            story_key,
        )
        return InferenceResult(
            key=story_key,
            status="review",
            confidence=InferenceConfidence.MEDIUM,
            evidence_sources=tuple(evidence),
        )

    # Priority 4: Validation report exists
    if index.has_validation(story_key):
        validations = index.get_validations(story_key)
        for validation in validations:
            evidence.append(validation.path)
        logger.debug(
            "Story %s: MEDIUM confidence 'ready-for-dev' - validation exists",
            story_key,
        )
        return InferenceResult(
            key=story_key,
            status="ready-for-dev",
            confidence=InferenceConfidence.MEDIUM,
            evidence_sources=tuple(evidence),
        )

    # Priority 5: Story file exists (without Status field)
    if index.has_story_file(story_key):
        if story_artifact:
            evidence.append(story_artifact.path)
        logger.debug(
            "Story %s: WEAK confidence 'in-progress' - story file exists",
            story_key,
        )
        return InferenceResult(
            key=story_key,
            status="in-progress",
            confidence=InferenceConfidence.WEAK,
            evidence_sources=tuple(evidence),
        )

    # Priority 6: No evidence
    logger.debug(
        "Story %s: NONE confidence 'backlog' - no evidence found",
        story_key,
    )
    return InferenceResult(
        key=story_key,
        status="backlog",
        confidence=InferenceConfidence.NONE,
        evidence_sources=(),
    )


# ============================================================================
# Epic Status Inference
# ============================================================================


def infer_epic_status(
    epic_id: EpicId,
    index: ArtifactIndex,
    story_statuses: dict[str, ValidStatus] | None = None,
    *,
    require_test_review_for_done: bool = False,
) -> tuple[ValidStatus, InferenceConfidence]:
    """Infer epic status from story statuses and retrospective.

    Examines the epic's retrospective (if exists) and the aggregate status
    of all stories in the epic to determine the epic's overall status.

    Args:
        epic_id: Epic identifier (int or str).
        index: ArtifactIndex with scanned artifacts.
        story_statuses: Optional pre-computed story statuses.
            If None, infers statuses for all stories in epic.
        require_test_review_for_done: Passed through to story status inference
            when story_statuses is not supplied.

    Returns:
        Tuple of (inferred_status, confidence_level).

    Epic Inference Rules (in priority order):
        1. Retrospective plus all known stories 'done' → 'done' (STRONG)
        2. Retrospective with no story evidence → 'done' (STRONG, legacy evidence)
        3. Empty story list → 'backlog' (NONE)
        4. All stories 'done' → 'done' (MEDIUM)
        5. Any story 'done' (partial completion) → 'in-progress' (MEDIUM)
        6. Any story 'in-progress', 'review', or 'blocked' → 'in-progress' (MEDIUM)
        7. Any story 'ready-for-dev' → 'backlog' (WEAK)
        8. Default → 'backlog' (NONE)

    Examples:
        >>> status, confidence = infer_epic_status(12, index)
        >>> status
        'done'
        >>> confidence
        InferenceConfidence.STRONG

    """
    has_retrospective = index.has_retrospective(epic_id)

    # Get or compute story statuses
    if story_statuses is None:
        # Discover story keys for this epic using pattern matching
        story_keys = _get_story_keys_for_epic(epic_id, index)
        story_statuses = {
            key: infer_story_status(
                key,
                index,
                require_test_review_for_done=require_test_review_for_done,
            )[0]
            for key in story_keys
        }

    # Priority 2: Empty story list → backlog unless retrospective is the only evidence.
    if not story_statuses:
        if has_retrospective:
            logger.debug(
                "Epic %s: STRONG confidence 'done' - retrospective exists with no story evidence",
                epic_id,
            )
            return "done", InferenceConfidence.STRONG
        logger.debug(
            "Epic %s: NONE confidence 'backlog' - no stories found",
            epic_id,
        )
        return "backlog", InferenceConfidence.NONE

    # Aggregate story statuses
    status_values = list(story_statuses.values())
    all_done = all(s == "done" for s in status_values)
    any_done = any(s == "done" for s in status_values)
    any_active = any(s in ("in-progress", "review", "blocked") for s in status_values)
    any_ready = any(s == "ready-for-dev" for s in status_values)

    # Priority 1: Retrospective closes the epic only when all known stories are done.
    if has_retrospective and all_done:
        logger.debug(
            "Epic %s: STRONG confidence 'done' - retrospective exists and all %d stories done",
            epic_id,
            len(story_statuses),
        )
        return "done", InferenceConfidence.STRONG

    # Priority 3: All stories done → epic done
    if all_done:
        logger.debug(
            "Epic %s: MEDIUM confidence 'done' - all %d stories done",
            epic_id,
            len(story_statuses),
        )
        return "done", InferenceConfidence.MEDIUM

    # Priority 4: Any story done (partial completion) → epic in-progress
    if any_done:
        done_count = sum(1 for s in status_values if s == "done")
        logger.debug(
            "Epic %s: MEDIUM confidence 'in-progress' - partial completion (%d/%d done)",
            epic_id,
            done_count,
            len(story_statuses),
        )
        return "in-progress", InferenceConfidence.MEDIUM

    # Priority 5: Any story in-progress or review → epic in-progress
    if any_active:
        logger.debug(
            "Epic %s: MEDIUM confidence 'in-progress' - active stories found",
            epic_id,
        )
        return "in-progress", InferenceConfidence.MEDIUM

    # Priority 6: Any story ready-for-dev → epic is ready (not started)
    if any_ready:
        logger.debug(
            "Epic %s: WEAK confidence 'backlog' - stories ready but not started",
            epic_id,
        )
        return "backlog", InferenceConfidence.WEAK

    # Default: epic in backlog
    logger.debug(
        "Epic %s: NONE confidence 'backlog' - no active progress",
        epic_id,
    )
    return "backlog", InferenceConfidence.NONE


# ============================================================================
# Batch Inference
# ============================================================================


def infer_all_statuses(
    index: ArtifactIndex,
    story_keys: list[str] | None = None,
    *,
    require_test_review_for_done: bool = False,
) -> dict[str, InferenceResult]:
    """Infer statuses for all stories in the index.

    Efficiently batch-processes stories with a single pass through the index.
    If story_keys is not provided, infers all stories found in the index.

    Args:
        index: ArtifactIndex with scanned artifacts.
        story_keys: Optional list of story keys to infer.
            If None, infers all stories in index.
        require_test_review_for_done: Passed through to story status inference.

    Returns:
        Dict mapping story_key to InferenceResult.

    Examples:
        >>> results = infer_all_statuses(index)
        >>> len(results)
        42
        >>> results["20-1-entry-classification"].status
        'done'

    """
    # If no specific keys provided, use all story keys from index
    if story_keys is None:
        story_keys = list(index.story_files.keys())

    results: dict[str, InferenceResult] = {}
    for key in story_keys:
        result = infer_story_status_detailed(
            key,
            index,
            require_test_review_for_done=require_test_review_for_done,
        )
        results[key] = result

    logger.debug(
        "Inferred statuses for %d stories",
        len(results),
    )

    return results
