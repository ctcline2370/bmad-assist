"""Sprint status synchronization for the loop runner.

Story 20.10: Sprint-status sync and repair integration.
Extracted from runner.py as part of the runner refactoring.

"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.state import Phase, State, save_state
from bmad_assist.core.types import EpicId

logger = logging.getLogger(__name__)

__all__ = [
    "_validate_resume_against_sprint",
    "_invoke_sprint_sync",
    "_ensure_sprint_sync_callback",
    "_trigger_interactive_repair",
    "_run_archive_artifacts",
]

# Type alias for state parameter
LoopState = State


def _requires_test_review_for_done() -> bool:
    """Return true when the active story loop includes BMAD TEST_REVIEW."""
    try:
        from bmad_assist.core.config import get_loop_config

        loop_config = get_loop_config()
    except Exception:
        return False

    return Phase.TEST_REVIEW.value in loop_config.story


def _get_epic_teardown_phases() -> tuple[str, ...]:
    """Return configured epic teardown phases for resume validation."""
    try:
        from bmad_assist.core.config import get_loop_config

        loop_config = get_loop_config()
    except Exception:
        return (Phase.RETROSPECTIVE.value,)

    return tuple(loop_config.epic_teardown)


def _validate_resume_against_sprint(
    state: LoopState,
    project_path: Path,
    epic_list: list[EpicId],
    epic_stories_loader: Callable[[EpicId], list[str]],
    state_path: Path,
    *,
    honor_current_story: bool = False,
) -> tuple[LoopState, bool]:
    """Validate and advance state based on sprint-status on resume.

    Checks sprint-status.yaml to see if current story/epic is already done.
    If so, advances state to the next incomplete story/epic.

    This fixes the bug where:
    - Loop was interrupted after completing work
    - sprint-status.yaml reflects the completed work
    - But state.yaml is stale and points to the completed position
    - On resume, loop would re-execute completed work

    Args:
        state: Current state from state.yaml.
        project_path: Project root directory.
        epic_list: Ordered list of epic IDs.
        epic_stories_loader: Function to get stories for an epic.
        state_path: Path to state file for persistence.
        honor_current_story: Preserve the selected story when an explicit
            story-level phase rerun has intentionally targeted a done story.

    Returns:
        Tuple of (updated_state, is_project_complete).
        - updated_state: May be same as input if no changes needed.
        - is_project_complete: True if all epics are done.

    """
    try:
        from bmad_assist.sprint.resume_validation import validate_resume_state
    except ImportError:
        logger.debug("Sprint resume validation module not available")
        return state, False

    try:
        result = validate_resume_state(
            state,
            project_path,
            epic_list,
            epic_stories_loader,
            require_completion_artifacts_for_done=True,
            require_test_review_for_done=_requires_test_review_for_done(),
            epic_teardown_phases=_get_epic_teardown_phases(),
            honor_current_story=honor_current_story,
        )

        if result.project_complete:
            # All epics done - save state if changed and signal completion
            logger.info("Resume validation: project is complete")
            if result.advanced:
                save_state(result.state, state_path)
            return result.state, True

        if result.advanced:
            logger.info("Resume validation: %s", result.summary())
            # Persist the advanced state
            save_state(result.state, state_path)
            # Trigger sprint sync to update sprint-status with new state
            _invoke_sprint_sync(result.state, project_path)
            return result.state, False

        logger.debug("Resume validation: no changes needed")
        return state, False

    except StateError:
        # Invalid lifecycle state must stop the loop so the operator sees the
        # concrete teardown gap instead of silently continuing with drift.
        raise
    except Exception as e:
        # Unexpected validation failures stay non-fatal until their root cause is
        # diagnosed; state-integrity failures are handled above.
        logger.warning("Resume validation failed (continuing): %s", e)
        return state, False


def _invoke_sprint_sync(state: LoopState, project_path: Path) -> None:
    """Invoke sprint sync callbacks after state save.

    Fire-and-forget invocation of registered sync callbacks. All errors are
    caught and logged at WARNING level, never propagating to the caller.

    Args:
        state: Current State instance.
        project_path: Project root directory.

    """
    try:
        from bmad_assist.sprint.sync import invoke_sync_callbacks

        invoke_sync_callbacks(state, project_path)
    except ImportError:
        logger.debug("Sprint module not available for sync")
    except Exception as e:
        logger.warning("Sprint sync failed (ignored): %s", e)


def _ensure_sprint_sync_callback() -> None:
    """Ensure default sprint sync callback is registered at loop startup.

    Idempotent registration - safe to call multiple times. Uses lazy import
    with ImportError guard for sprint module availability.

    """
    try:
        from bmad_assist.sprint.repair import ensure_sprint_sync_callback

        ensure_sprint_sync_callback()
    except ImportError:
        logger.debug("Sprint module not available for callback registration")
    except Exception as e:
        logger.warning("Sprint callback registration failed (ignored): %s", e)


def _trigger_interactive_repair(project_path: Path, state: LoopState) -> None:
    """Trigger interactive repair on loop initialization.

    Catches all exceptions including ImportError - NEVER crashes the loop.
    Called only on fresh start to perform full artifact-based repair.

    Args:
        project_path: Project root directory.
        state: Current State instance.

    """
    try:
        from bmad_assist.sprint.repair import RepairMode, repair_sprint_status
    except ImportError:
        logger.debug("Sprint module not available for repair")
        return

    try:
        result = repair_sprint_status(project_path, RepairMode.INTERACTIVE, state)
        if result.user_cancelled:
            logger.warning("Sprint repair cancelled, continuing without repair")
        elif result.errors:
            logger.warning("Sprint repair encountered errors: %s", result.errors)
        else:
            logger.info("Sprint repair complete: %s", result.summary())
    except Exception as e:
        logger.warning("Sprint repair failed (ignored): %s", e)


def _run_archive_artifacts(project_path: Path) -> None:
    """Run archive-artifacts.sh to archive multi-LLM validation and review reports.

    Archives non-master/non-synthesis .md files from:
    - _bmad-output/implementation-artifacts/code-reviews/
    - _bmad-output/implementation-artifacts/story-validations/

    Called after CODE_REVIEW_SYNTHESIS to clean up multi-reviewer artifacts.
    Script is idempotent - safe to call multiple times.

    Args:
        project_path: Project root directory.

    """
    script_path = project_path / "scripts" / "archive-artifacts.sh"

    if not script_path.exists():
        logger.debug("archive-artifacts.sh not found at %s, skipping", script_path)
        return

    try:
        result = subprocess.run(
            [str(script_path), "-s"],  # -s for silent mode
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Archived multi-LLM artifacts after code review synthesis")
        else:
            logger.warning(
                "archive-artifacts.sh failed (returncode=%d): %s",
                result.returncode,
                result.stderr,
            )
    except subprocess.TimeoutExpired:
        logger.warning("archive-artifacts.sh timed out after 30s")
    except Exception as e:
        logger.warning("archive-artifacts.sh execution failed: %s", e)
