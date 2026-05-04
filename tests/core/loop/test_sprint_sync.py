"""Focused tests for loop sprint synchronization boundaries."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.loop.sprint_sync import _validate_resume_against_sprint
from bmad_assist.core.state import State


def test_validate_resume_against_sprint_propagates_state_errors(tmp_path: Path) -> None:
    """Lifecycle drift must fail closed instead of being downgraded to a warning."""
    state = State(current_epic=8, current_story="8.1")
    state_path = tmp_path / "state.yaml"

    with patch(
        "bmad_assist.sprint.resume_validation.validate_resume_state",
        side_effect=StateError("Epic 7 teardown is incomplete"),
    ):
        with pytest.raises(StateError, match="Epic 7 teardown is incomplete"):
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

    with patch(
        "bmad_assist.sprint.resume_validation.validate_resume_state",
        side_effect=RuntimeError("boom"),
    ):
        with caplog.at_level(logging.WARNING):
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
