"""Focused tests for loop sprint synchronization boundaries."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.loop.sprint_sync import _validate_resume_against_sprint
from bmad_assist.core.state import State
from bmad_assist.sprint.resume_validation import ResumeValidationResult


def test_validate_resume_against_sprint_propagates_state_errors(tmp_path: Path) -> None:
    """Lifecycle drift must fail closed instead of being downgraded to a warning."""
    state = State(current_epic=8, current_story="8.1")
    state_path = tmp_path / "state.yaml"

    with (
        patch(
            "bmad_assist.sprint.resume_validation.validate_resume_state",
            side_effect=StateError("Epic 7 teardown is incomplete"),
        ),
        pytest.raises(StateError, match="Epic 7 teardown is incomplete"),
    ):
        _validate_resume_against_sprint(
            state,
            tmp_path,
            [7, 8],
            lambda _epic_id: [],
            state_path,
        )


def test_validate_resume_against_sprint_warns_and_ignores_unexpected_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected validator faults remain visible without crashing the loop."""
    state = State(current_epic=8, current_story="8.1")
    state_path = tmp_path / "state.yaml"

    with (
        patch(
            "bmad_assist.sprint.resume_validation.validate_resume_state",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        result_state, project_complete = _validate_resume_against_sprint(
            state,
            tmp_path,
            [7, 8],
            lambda _epic_id: [],
            state_path,
        )

    assert result_state is state
    assert project_complete is False
    assert "Resume validation failed (continuing): boom" in caplog.text


def test_validate_resume_against_sprint_passes_configured_epic_teardown_phases(
    tmp_path: Path,
) -> None:
    """Resume validation uses the active loop config instead of a hardcoded teardown set."""
    from bmad_assist.core.config import LoopConfig

    state = State(current_epic=1, current_story="1.5")
    state_path = tmp_path / "state.yaml"
    loop_config = LoopConfig(
        epic_setup=[],
        story=["create_story", "dev_story", "test_review"],
        epic_teardown=["trace", "tea_nfr_assess", "retrospective"],
    )
    result = ResumeValidationResult(
        state=state,
        stories_skipped=[],
        epics_skipped=[],
        advanced=False,
        project_complete=False,
    )

    with (
        patch("bmad_assist.core.config.get_loop_config", return_value=loop_config),
        patch(
            "bmad_assist.sprint.resume_validation.validate_resume_state",
            return_value=result,
        ) as validate_resume_state,
    ):
        _validate_resume_against_sprint(
            state,
            tmp_path,
            [1],
            lambda _epic_id: ["1.5"],
            state_path,
        )

    assert validate_resume_state.call_args.kwargs["epic_teardown_phases"] == (
        "trace",
        "tea_nfr_assess",
        "retrospective",
    )
