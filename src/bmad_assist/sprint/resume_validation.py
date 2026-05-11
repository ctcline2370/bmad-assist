"""Resume validation against sprint-status.

This module implements validation of state.yaml against sprint-status.yaml
when resuming the development loop. It's the REVERSE of sync.py - it reads
sprint-status to detect if state.yaml is stale and needs advancement.

Use case:
- Loop crashes or is interrupted
- sprint-status.yaml is manually updated (marking stories/epics done)
- On resume, state.yaml still points to old position
- This module detects the discrepancy and advances state

Architecture:
- sprint-status.yaml is checked as SECONDARY source
- state.yaml remains the authoritative source for crash recovery
- Validation only ADVANCES state (never rolls back)

Public API:
    - ResumeValidationResult: Dataclass with validation outcome
    - validate_resume_state: Main function to check and fix stale state
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.retrospective_artifacts import has_durable_retrospective_artifact
from bmad_assist.core.state import Phase, State
from bmad_assist.core.types import EpicId
from bmad_assist.sprint.models import SprintStatus
from bmad_assist.sprint.scanner import ArtifactIndex

logger = logging.getLogger(__name__)

__all__ = [
    "ResumeValidationResult",
    "validate_resume_state",
]


@dataclass
class ResumeValidationResult:
    """Result of resume state validation.

    Attributes:
        state: Updated state (may be same as input if no changes).
        stories_skipped: List of story IDs that were skipped as already done.
        epics_skipped: List of epic IDs that were skipped as already done.
        advanced: True if state was modified (stories or epics skipped).
        project_complete: True if all epics are done.

    """

    state: State
    stories_skipped: list[str]
    epics_skipped: list[EpicId]
    advanced: bool
    project_complete: bool

    def summary(self) -> str:
        """Return human-readable summary."""
        if not self.advanced:
            return "Resume validation: no changes needed"
        parts = []
        if self.stories_skipped:
            parts.append(f"skipped {len(self.stories_skipped)} done stories")
        if self.epics_skipped:
            parts.append(f"skipped {len(self.epics_skipped)} done epics")
        if self.project_complete:
            parts.append("project complete")
        return "Resume validation: " + ", ".join(parts)


def _normalize_epic_teardown_phases(
    epic_teardown_phases: Sequence[str | Phase] | None,
) -> set[Phase]:
    """Return the epic teardown phases that resume validation must not skip."""
    phases = {Phase.RETROSPECTIVE}
    if not epic_teardown_phases:
        return phases

    for phase in epic_teardown_phases:
        if isinstance(phase, Phase):
            phases.add(phase)
            continue

        try:
            phases.add(Phase(phase))
        except ValueError:
            logger.warning(
                "Ignoring unknown epic teardown phase in resume validation: %s",
                phase,
            )

    return phases


def _get_story_status_from_sprint(
    story_id: str,
    sprint_status: SprintStatus,
) -> str | None:
    """Get story status from sprint-status by story ID.

    Converts state format (e.g., "20.9") to sprint-status key prefix (e.g., "20-9")
    and searches for matching entry.

    Args:
        story_id: Story ID from state (e.g., "20.9", "testarch.1").
        sprint_status: Parsed sprint-status.

    Returns:
        Status string if found ("done", "in-progress", etc.), None if not found.

    """
    prefix = story_id.replace(".", "-")
    for key, entry in sprint_status.entries.items():
        if key.startswith(f"{prefix}-") or key == prefix:
            return entry.status
    return None


def _is_story_done_in_sprint(
    story_id: str,
    sprint_status: SprintStatus,
) -> bool:
    """Check if story is marked as done in sprint-status.

    Args:
        story_id: Story ID from state.
        sprint_status: Parsed sprint-status.

    Returns:
        True if story has status "done", False otherwise.

    """
    status = _get_story_status_from_sprint(story_id, sprint_status)
    return status == "done"


def _get_story_completion_gaps(
    story_id: str,
    sprint_status: SprintStatus,
    project_path: Path | None,
    *,
    require_completion_artifacts_for_done: bool = False,
    require_test_review_for_done: bool = False,
    artifact_index: ArtifactIndex | None = None,
) -> list[str]:
    """Return concrete story-completion gaps that block validated done."""
    gaps: list[str] = []
    status = _get_story_status_from_sprint(story_id, sprint_status)
    if status != "done":
        status_text = status if status is not None else "missing"
        gaps.append(f"sprint-status story entry is {status_text!r}")
        return gaps

    require_code_review_evidence = (
        require_completion_artifacts_for_done or require_test_review_for_done
    )
    if not require_code_review_evidence:
        return gaps

    if artifact_index is None:
        if project_path is None:
            gaps.append("artifact index is unavailable")
            return gaps
        artifact_index = ArtifactIndex.scan(project_path)

    story_key = story_id.replace(".", "-")
    if not artifact_index.has_master_review(story_key):
        gaps.append("durable code-review synthesis artifact is missing")

    if require_test_review_for_done and not artifact_index.has_test_review(story_key):
        gaps.append("durable test-review artifact is missing")

    return gaps


def _is_story_durably_done(
    story_id: str,
    sprint_status: SprintStatus,
    project_path: Path | None = None,
    *,
    require_completion_artifacts_for_done: bool = False,
    require_test_review_for_done: bool = False,
    artifact_index: ArtifactIndex | None = None,
) -> bool:
    """Check if story completion is backed by required durable artifacts."""
    return not _get_story_completion_gaps(
        story_id,
        sprint_status,
        project_path,
        require_completion_artifacts_for_done=require_completion_artifacts_for_done,
        require_test_review_for_done=require_test_review_for_done,
        artifact_index=artifact_index,
    )


def _is_epic_done_in_sprint(
    epic_id: EpicId,
    sprint_status: SprintStatus,
) -> bool:
    """Check if epic is FULLY done in sprint-status (including retrospective).

    An epic is only considered done if BOTH:
    1. epic-X entry has status "done"
    2. epic-X-retrospective entry has status "done"

    If epic is "done" but retrospective is "backlog"/"in-progress", the epic
    is NOT considered done - it needs to run its retrospective phase.

    Args:
        epic_id: Epic ID.
        sprint_status: Parsed sprint-status.

    Returns:
        True if epic AND its retrospective are done, False otherwise.

    """
    # Check epic status
    epic_status = sprint_status.get_epic_status(epic_id)
    if epic_status != "done":
        return False

    # Check retrospective status
    retro_key = f"epic-{epic_id}-retrospective"
    retro_entry = sprint_status.entries.get(retro_key)
    return retro_entry is not None and retro_entry.status == "done"


def _get_epic_teardown_gaps(
    epic_id: EpicId,
    sprint_status: SprintStatus,
    project_path: Path,
) -> list[str]:
    """Return concrete teardown gaps that still block epic completion."""
    gaps: list[str] = []

    epic_status = sprint_status.get_epic_status(epic_id)
    if epic_status != "done":
        status = epic_status if epic_status is not None else "missing"
        gaps.append(f"sprint-status epic entry is {status!r}")

    retro_key = f"epic-{epic_id}-retrospective"
    retro_entry = sprint_status.entries.get(retro_key)
    if retro_entry is None:
        gaps.append("sprint-status retrospective entry is missing")
    elif retro_entry.status != "done":
        gaps.append(f"sprint-status retrospective entry is {retro_entry.status!r}")

    if not has_durable_retrospective_artifact(epic_id, project_path):
        gaps.append("durable retrospective artifact is missing")

    return gaps


def _is_epic_durably_done(
    epic_id: EpicId,
    sprint_status: SprintStatus,
    project_path: Path,
) -> bool:
    """Check if epic completion is durably backed by teardown artifacts."""
    return not _get_epic_teardown_gaps(epic_id, sprint_status, project_path)


def _ensure_prior_epics_durably_complete(
    current_epic: EpicId,
    epic_list: list[EpicId],
    sprint_status: SprintStatus,
    project_path: Path,
) -> None:
    """Fail closed if state advanced beyond an epic whose teardown is incomplete."""
    try:
        current_idx = epic_list.index(current_epic)
    except ValueError:
        return

    for prior_epic in epic_list[:current_idx]:
        gaps = _get_epic_teardown_gaps(prior_epic, sprint_status, project_path)
        if gaps:
            gap_text = "; ".join(gaps)
            raise StateError(
                f"Epic {prior_epic} teardown is incomplete ({gap_text}). "
                f"Finish epic {prior_epic} teardown before advancing to epic {current_epic}."
            )


def _ensure_prior_stories_durably_complete(
    current_story: str,
    epic_stories: list[str],
    sprint_status: SprintStatus,
    project_path: Path,
    *,
    require_completion_artifacts_for_done: bool = False,
    require_test_review_for_done: bool = False,
    artifact_index: ArtifactIndex | None = None,
) -> None:
    """Fail closed if state advanced beyond a story whose completion is incomplete."""
    try:
        current_idx = epic_stories.index(current_story)
    except ValueError:
        return

    for prior_story in epic_stories[:current_idx]:
        gaps = _get_story_completion_gaps(
            prior_story,
            sprint_status,
            project_path,
            require_completion_artifacts_for_done=require_completion_artifacts_for_done,
            require_test_review_for_done=require_test_review_for_done,
            artifact_index=artifact_index,
        )
        if gaps:
            gap_text = "; ".join(gaps)
            raise StateError(
                f"Story {prior_story} completion is incomplete ({gap_text}). "
                f"Finish story {prior_story} before advancing to story {current_story}."
            )


def validate_resume_state(
    state: State,
    project_path: Path,
    epic_list: list[EpicId],
    epic_stories_loader: Callable[[EpicId], list[str]],
    *,
    require_completion_artifacts_for_done: bool = False,
    require_test_review_for_done: bool = False,
    epic_teardown_phases: Sequence[str | Phase] | None = None,
    honor_current_story: bool = False,
) -> ResumeValidationResult:
    """Validate and advance state based on sprint-status.

    Checks sprint-status.yaml to see if current story/epic is already done.
    If so, advances state to the next incomplete story/epic.

    This handles the case where:
    - Loop was interrupted/crashed
    - sprint-status was manually updated to mark work as done
    - state.yaml is now stale and points to completed work

    Args:
        state: Current state from state.yaml.
        project_path: Project root directory.
        epic_list: Ordered list of epic IDs.
        epic_stories_loader: Function to get stories for an epic.
        require_completion_artifacts_for_done: Require durable code-review
            synthesis or master-review artifacts before skipping done stories.
        require_test_review_for_done: Require durable test-review artifacts
            before skipping done stories. Implies code-review completion evidence.
        epic_teardown_phases: Configured epic teardown phases that must execute
            even when sprint-status already marks every story in the epic done.
        honor_current_story: Preserve the selected story when an explicit
            story-level phase rerun has intentionally targeted a done story.

    Returns:
        ResumeValidationResult with potentially advanced state.

    """
    from datetime import UTC, datetime

    from bmad_assist.core.exceptions import ParserError
    from bmad_assist.core.paths import get_paths
    from bmad_assist.sprint.parser import parse_sprint_status

    stories_skipped: list[str] = []
    epics_skipped: list[EpicId] = []
    configured_teardown_phases = _normalize_epic_teardown_phases(epic_teardown_phases)
    artifact_index: ArtifactIndex | None = None
    if require_completion_artifacts_for_done or require_test_review_for_done:
        artifact_index = ArtifactIndex.scan(project_path)

    # Find sprint-status location (uses paths singleton for external paths support)
    try:
        sprint_path = get_paths().sprint_status_file
    except RuntimeError:
        # Fallback for tests or early startup when singleton not initialized
        # Check multiple locations for consistency with state_reader.py
        fallback_candidates = [
            project_path
            / "_bmad-output"
            / "implementation-artifacts"
            / "sprint-status.yaml",  # New # noqa: E501
            project_path / "docs" / "sprint-artifacts" / "sprint-status.yaml",  # Legacy
            project_path / "docs" / "sprint-status.yaml",  # Legacy (direct)
        ]
        # Use first existing, or default to new location
        sprint_path = next(
            (p for p in fallback_candidates if p.exists()),
            fallback_candidates[0],  # Default to new location
        )

    # If no sprint-status exists, nothing to validate against
    if not sprint_path.exists():
        logger.debug("No sprint-status.yaml found, skipping resume validation")
        return ResumeValidationResult(
            state=state,
            stories_skipped=[],
            epics_skipped=[],
            advanced=False,
            project_complete=False,
        )

    # Parse sprint-status
    try:
        sprint_status = parse_sprint_status(sprint_path)
    except ParserError as e:
        logger.warning("Failed to parse sprint-status for validation: %s", e)
        return ResumeValidationResult(
            state=state,
            stories_skipped=[],
            epics_skipped=[],
            advanced=False,
            project_complete=False,
        )

    # If sprint-status is empty, nothing to validate
    if not sprint_status.entries:
        logger.debug("Sprint-status is empty, skipping resume validation")
        return ResumeValidationResult(
            state=state,
            stories_skipped=[],
            epics_skipped=[],
            advanced=False,
            project_complete=False,
        )

    current_state = state
    now = datetime.now(UTC).replace(tzinfo=None)

    # Loop: Keep advancing while current position is "done" in sprint-status
    max_iterations = 1000  # Safety limit to prevent infinite loops
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Sanity check
        if current_state.current_epic is None:
            logger.debug("No current epic set, cannot validate")
            break

        # Type narrowing: current_epic is guaranteed non-None from here
        current_epic: EpicId = current_state.current_epic

        _ensure_prior_epics_durably_complete(
            current_epic,
            epic_list,
            sprint_status,
            project_path,
        )

        # CRITICAL: If we're in a configured epic teardown phase, don't skip anything.
        # The loop needs to execute teardown - we shouldn't try to advance past it
        # just because all stories in the epic are done.
        if current_state.current_phase in configured_teardown_phases:
            logger.debug(
                "Current phase is configured epic teardown phase %s - "
                "not skipping, let loop execute it",
                current_state.current_phase.name,
            )
            break

        if current_state.current_phase == Phase.TEST_REVIEW:
            logger.debug("Current phase is TEST_REVIEW - not skipping, let loop execute it")
            break

        # Check if current epic is durably done (sprint-status plus artifact)
        if not honor_current_story and _is_epic_durably_done(
            current_epic, sprint_status, project_path
        ):
            # Epic is done - add to completed_epics if not already there
            if current_epic not in current_state.completed_epics:
                logger.info(
                    "Sprint-status shows epic %s is done, adding to completed_epics",
                    current_epic,
                )
                epics_skipped.append(current_epic)
                current_state = current_state.model_copy(
                    update={
                        "completed_epics": [
                            *current_state.completed_epics,
                            current_epic,
                        ],
                        "updated_at": now,
                    }
                )

            # Find next epic that's not done
            next_epic = _find_next_incomplete_epic(
                current_epic,
                epic_list,
                current_state.completed_epics,
                sprint_status,
                project_path,
            )

            if next_epic is None:
                # All epics done
                logger.info("All epics are done according to sprint-status")
                return ResumeValidationResult(
                    state=current_state,
                    stories_skipped=stories_skipped,
                    epics_skipped=epics_skipped,
                    advanced=bool(stories_skipped or epics_skipped),
                    project_complete=True,
                )

            # Advance to next epic
            try:
                next_epic_stories = epic_stories_loader(next_epic)
            except Exception as e:
                raise StateError(f"Failed to load stories for epic {next_epic}: {e}") from e

            if not next_epic_stories:
                raise StateError(f"No stories found in epic {next_epic}")

            current_state = current_state.model_copy(
                update={
                    "current_epic": next_epic,
                    "current_story": next_epic_stories[0],
                    "current_phase": Phase.CREATE_STORY,
                    "updated_at": now,
                }
            )
            logger.info(
                "Advanced to epic %s, story %s",
                next_epic,
                next_epic_stories[0],
            )
            # Continue loop to check if new position is also done
            continue

        # Epic not done - check if current story is done
        if current_state.current_story is None:
            logger.debug("No current story set, cannot validate")
            break
        current_story = current_state.current_story

        try:
            current_epic_stories = epic_stories_loader(current_epic)
        except Exception as e:
            raise StateError(f"Failed to load stories for epic {current_epic}: {e}") from e

        _ensure_prior_stories_durably_complete(
            current_story,
            current_epic_stories,
            sprint_status,
            project_path,
            require_completion_artifacts_for_done=require_completion_artifacts_for_done,
            require_test_review_for_done=require_test_review_for_done,
            artifact_index=artifact_index,
        )

        story_completion_gaps = _get_story_completion_gaps(
            current_story,
            sprint_status,
            project_path,
            require_completion_artifacts_for_done=require_completion_artifacts_for_done,
            require_test_review_for_done=require_test_review_for_done,
            artifact_index=artifact_index,
        )
        if not story_completion_gaps:
            if honor_current_story:
                logger.info(
                    "Sprint-status shows story %s is done; preserving it for explicit rerun",
                    current_story,
                )
                break

            # Story is done but epic is not - need to advance to next story
            logger.info(
                "Sprint-status shows story %s is done, advancing",
                current_story,
            )
            stories_skipped.append(current_story)

            # Add to completed_stories if not there
            if current_story not in current_state.completed_stories:
                current_state = current_state.model_copy(
                    update={
                        "completed_stories": [
                            *current_state.completed_stories,
                            current_story,
                        ],
                        "updated_at": now,
                    }
                )

            next_story = _find_next_incomplete_story(
                current_story,
                current_epic_stories,
                current_state.completed_stories,
                sprint_status,
                project_path=project_path,
                require_completion_artifacts_for_done=require_completion_artifacts_for_done,
                require_test_review_for_done=require_test_review_for_done,
                artifact_index=artifact_index,
            )

            if next_story is None:
                gap_text = "; ".join(_get_epic_teardown_gaps(current_epic, sprint_status, project_path))
                logger.error(
                    "Epic %s has all stories marked done in sprint-status, but epic teardown is incomplete: %s",
                    current_epic,
                    gap_text,
                )
                raise StateError(
                    f"Epic {current_epic} has all stories marked done in sprint-status, "
                    f"but epic teardown is incomplete ({gap_text}). Finish the epic teardown "
                    "before advancing to the next epic."
                )

            # Advance to next story
            current_state = current_state.model_copy(
                update={
                    "current_story": next_story,
                    "current_phase": Phase.CREATE_STORY,
                    "updated_at": now,
                }
            )
            logger.info("Advanced to story %s", next_story)
            # Continue loop to check if new story is also done
            continue

        if _is_story_done_in_sprint(current_story, sprint_status):
            gap_text = "; ".join(story_completion_gaps)
            raise StateError(
                f"Story {current_story} completion is incomplete ({gap_text}). "
                f"Finish story {current_story} before advancing."
            )

        # Current position is not done - we're at the right place
        break

    if iterations >= max_iterations:
        logger.error("Resume validation hit iteration limit - possible infinite loop")

    return ResumeValidationResult(
        state=current_state,
        stories_skipped=stories_skipped,
        epics_skipped=epics_skipped,
        advanced=bool(stories_skipped or epics_skipped),
        project_complete=False,
    )


def _find_next_incomplete_epic(
    current_epic: EpicId,
    epic_list: list[EpicId],
    completed_epics: list[EpicId],
    sprint_status: SprintStatus,
    project_path: Path,
) -> EpicId | None:
    """Find the next epic that is not complete.

    Skips epics that are either:
    - In completed_epics list
    - Marked as "done" in sprint-status

    Args:
        current_epic: Current epic ID.
        epic_list: Ordered list of all epics.
        completed_epics: List of epics marked complete in state.
        sprint_status: Parsed sprint-status.

    Returns:
        Next incomplete epic ID, or None if all remaining epics are done.

    """
    try:
        current_idx = epic_list.index(current_epic)
    except ValueError:
        # Current epic not in list - start from beginning
        current_idx = -1

    for epic in epic_list[current_idx + 1 :]:
        # Skip if in completed_epics
        if epic in completed_epics:
            logger.debug("Skipping epic %s - in completed_epics", epic)
            continue
        # Skip only if durably complete
        if _is_epic_durably_done(epic, sprint_status, project_path):
            logger.debug("Skipping epic %s - durably done", epic)
            continue
        return epic

    return None


def _find_next_incomplete_story(
    current_story: str,
    epic_stories: list[str],
    completed_stories: list[str],
    sprint_status: SprintStatus,
    *,
    project_path: Path | None = None,
    require_completion_artifacts_for_done: bool = False,
    require_test_review_for_done: bool = False,
    artifact_index: ArtifactIndex | None = None,
) -> str | None:
    """Find the next story in the epic that is not complete.

    Skips stories that are either:
    - In completed_stories list
    - Marked as "done" in sprint-status

    Args:
        current_story: Current story ID.
        epic_stories: Ordered list of stories in the epic.
        completed_stories: List of stories marked complete in state.
        sprint_status: Parsed sprint-status.
        project_path: Project root used to scan durable completion artifacts.
        require_completion_artifacts_for_done: Require durable code-review
            completion artifacts before skipping done stories.
        require_test_review_for_done: Require durable test-review artifacts
            before skipping done stories.
        artifact_index: Optional pre-scanned artifact index.

    Returns:
        Next incomplete story ID, or None if all remaining stories are done.

    """
    try:
        current_idx = epic_stories.index(current_story)
    except ValueError:
        # Current story not in list - start from beginning
        current_idx = -1

    for story in epic_stories[current_idx + 1 :]:
        # Skip if in completed_stories
        if story in completed_stories:
            logger.debug("Skipping story %s - in completed_stories", story)
            continue
        # Skip if done in sprint-status and backed by required durable evidence
        if _is_story_durably_done(
            story,
            sprint_status,
            project_path,
            require_completion_artifacts_for_done=require_completion_artifacts_for_done,
            require_test_review_for_done=require_test_review_for_done,
            artifact_index=artifact_index,
        ):
            logger.debug("Skipping story %s - durably done", story)
            continue
        return story

    return None
