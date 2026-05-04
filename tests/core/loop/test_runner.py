"""Tests for runner module.

Story 6.5: Main Loop Runner
- run_loop() function with all its scenarios
"""

from __future__ import annotations

import contextlib
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.core.config import Config, load_config
from bmad_assist.core.exceptions import StateError

if TYPE_CHECKING:
    pass


class TestRunLoopStub:
    """Tests for run_loop() stub function."""

    @pytest.fixture
    def valid_config(self) -> Config:
        """Create a valid Config object for testing."""
        config_data = {
            "providers": {
                "master": {
                    "provider": "claude",
                    "model": "opus_4",
                }
            }
        }
        return load_config(config_data)

    def test_run_loop_is_importable(self) -> None:
        """run_loop can be imported from bmad_assist.core.loop."""
        from bmad_assist.core.loop import run_loop

        assert callable(run_loop)

    def test_run_loop_is_exported_from_core(self) -> None:
        """run_loop is exported from bmad_assist.core."""
        from bmad_assist.core import run_loop

        assert callable(run_loop)

    def test_run_loop_accepts_config_and_path(self, valid_config: Config, tmp_path: Path) -> None:
        """AC5: run_loop accepts Config and Path arguments (with epic_list, loader)."""
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import State

        # Story 6.5 update: run_loop now requires epic_list and epic_stories_loader
        with patch("bmad_assist.core.loop.runner.load_state", return_value=State()):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()):
                    with patch("bmad_assist.core.loop.runner.save_state"):
                        with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                            mock_epic.return_value = (State(), True)

                            run_loop(valid_config, tmp_path, [1], lambda x: ["1.1"])

    def test_run_loop_returns_completed(self, valid_config: Config, tmp_path: Path) -> None:
        """AC5: run_loop returns LoopExitReason.COMPLETED (blocking call that completes)."""
        from bmad_assist.core.loop import LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import State

        # Story 6.5 update: run_loop now requires epic_list and epic_stories_loader
        # Story 6.6 update: run_loop returns LoopExitReason instead of None
        with patch("bmad_assist.core.loop.runner.load_state", return_value=State()):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()):
                    with patch("bmad_assist.core.loop.runner.save_state"):
                        with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                            mock_epic.return_value = (State(), True)

                            result = run_loop(valid_config, tmp_path, [1], lambda x: ["1.1"])

        assert result == LoopExitReason.COMPLETED

    def test_run_loop_logs_debug_info(
        self, valid_config: Config, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """run_loop logs debug information about config and path."""
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import State

        # Story 6.5 update: run_loop now requires epic_list and epic_stories_loader
        with patch("bmad_assist.core.loop.runner.load_state", return_value=State()):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()):
                    with patch("bmad_assist.core.loop.runner.save_state"):
                        with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                            mock_epic.return_value = (State(), True)

                            with caplog.at_level(logging.DEBUG):
                                run_loop(valid_config, tmp_path, [1], lambda x: ["1.1"])

        # Should log provider info
        assert "claude" in caplog.text.lower()
        # Should log path
        assert str(tmp_path) in caplog.text


class TestRunLoopIntegration:
    """Integration tests for run_loop with CLI."""

    def test_run_loop_callable_from_cli_context(self, tmp_path: Path) -> None:
        """run_loop works when called via CLI context."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import State

        config_data = {
            "providers": {
                "master": {
                    "provider": "gemini",
                    "model": "flash",
                }
            },
            "state_path": str(tmp_path / "state.yaml"),
        }
        config = load_config(config_data)
        project_path = tmp_path / "project"
        project_path.mkdir()

        # Must patch execute_phase in BOTH modules because each imports it separately:
        # - runner.py: from dispatch import execute_phase (for main loop)
        # - epic_phases.py: from dispatch import execute_phase (for teardown phases)
        # Python imports create separate references, so we must patch both
        with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
            with patch("bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                    mock_epic.return_value = (State(), True)

                    run_loop(config, project_path, [1], lambda x: ["1.1"])

    def test_run_loop_with_different_providers(self, tmp_path: Path) -> None:
        """run_loop handles different provider configurations."""
        from bmad_assist.core.config import _reset_config, load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import State

        providers = [
            ("claude", "opus_4"),
            ("codex", "gpt-4"),
            ("gemini", "flash"),
        ]

        for provider, model in providers:
            _reset_config()
            config_data = {
                "providers": {
                    "master": {
                        "provider": provider,
                        "model": model,
                    }
                },
                "state_path": str(tmp_path / f"state-{provider}.yaml"),
            }
            config = load_config(config_data)

            # Must patch execute_phase in BOTH modules (each imports it separately)
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (State(), True)

                        run_loop(config, tmp_path, [1], lambda x: ["1.1"])


class TestRunLoopFreshStart:
    """AC1: run_loop() creates fresh state when no state file."""

    def test_run_loop_empty_epic_list_raises(self, tmp_path: Path) -> None:
        """AC1: Raises StateError when epic_list is empty."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import run_loop

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        with pytest.raises(StateError, match="No epics found"):
            run_loop(config, tmp_path, [], lambda x: [])

    def test_run_loop_empty_stories_raises(self, tmp_path: Path) -> None:
        """AC1: Raises StateError when first epic has no stories."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import run_loop

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        with pytest.raises(StateError, match="No stories found in epic 1"):
            run_loop(config, tmp_path, [1], lambda x: [])

    def test_run_loop_creates_fresh_state(self, tmp_path: Path) -> None:
        """AC1: Creates fresh state with first epic/story when no state file."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase

        state_file = tmp_path / "state.yaml"
        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(state_file),
            }
        )

        epic_list = [1, 2, 3]
        loader = lambda x: [f"{x}.1", f"{x}.2"]

        saved_states = []

        with patch("bmad_assist.core.loop.runner.execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.ok()
            with patch("bmad_assist.core.loop.runner.save_state") as mock_save:
                mock_save.side_effect = lambda state, path: saved_states.append(state)
                with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                    mock_epic.return_value = (MagicMock(), True)  # Project complete

                    run_loop(config, tmp_path, epic_list, loader)

        # First saved state should be fresh start
        assert len(saved_states) >= 1
        initial_state = saved_states[0]
        assert initial_state.current_epic == 1
        assert initial_state.current_story == "1.1"
        assert initial_state.current_phase == Phase.CREATE_STORY


class TestRunLoopResume:
    """AC1: run_loop() loads existing state on resume."""

    def test_run_loop_loads_existing_state(self, tmp_path: Path) -> None:
        """AC1: Loads state from file if exists."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        existing_state = State(
            current_epic=2,
            current_story="2.1",  # Changed to valid story ID
            current_phase=Phase.DEV_STORY,
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=existing_state):
            with patch("bmad_assist.core.loop.runner.execute_phase") as mock_exec:
                mock_exec.return_value = PhaseResult.ok()
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (existing_state, True)

                        run_loop(config, tmp_path, [1, 2], lambda x: [f"{x}.1", f"{x}.2"])

        # Should have executed at least one phase
        assert mock_exec.called

    def test_run_loop_propagates_state_error(self, tmp_path: Path) -> None:
        """AC1: Propagates StateError from load_state for corrupted files."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import run_loop

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        with patch("bmad_assist.core.loop.runner.load_state") as mock_load:
            mock_load.side_effect = StateError("Corrupted state file")

            with pytest.raises(StateError, match="Corrupted state file"):
                run_loop(config, tmp_path, [1], lambda x: ["1.1"])


class TestRunLoopPhaseExecution:
    """AC2: run_loop() executes phases in sequence."""

    def test_run_loop_executes_phase(self, tmp_path: Path) -> None:
        """AC2: execute_phase is called."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(current_epic=1, current_story="1.1", current_phase=Phase.CREATE_STORY)

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase") as mock_exec:
                mock_exec.return_value = PhaseResult.ok()
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (state, True)

                        run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        assert mock_exec.called

    def test_run_loop_advances_phases(self, tmp_path: Path) -> None:
        """AC2/AC7: Phases advance after successful execution."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(current_epic=1, current_story="1.1", current_phase=Phase.CREATE_STORY)
        executed_phases = []

        def track_execute(s):
            executed_phases.append(s.current_phase)
            if len(executed_phases) >= 3:
                # Stop after a few phases by returning to epic completion
                return PhaseResult.ok()
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", side_effect=track_execute):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (state, True)

                        run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # Should have executed CREATE_STORY first
        assert Phase.CREATE_STORY in executed_phases


class TestRunLoopStoryCompletion:
    """AC3: run_loop() handles story completion."""

    def test_run_loop_handles_story_completion_not_last(self, tmp_path: Path) -> None:
        """AC3: Story completion advances to next story when not last."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        # Start at CODE_REVIEW_SYNTHESIS (triggers story completion)
        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
        )

        next_state = State(
            current_epic=1,
            current_story="1.2",
            current_phase=Phase.CREATE_STORY,
        )

        call_count = [0]

        def controlled_execute(s):
            call_count[0] += 1
            if call_count[0] > 3:
                # Prevent infinite loop
                raise StateError("Breaking loop for test")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch(
                "bmad_assist.core.loop.runner.execute_phase", side_effect=controlled_execute
            ):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch(
                        "bmad_assist.core.loop.runner.handle_story_completion"
                    ) as mock_story:
                        mock_story.return_value = (next_state, False)
                        with patch(
                            "bmad_assist.core.loop.runner.handle_epic_completion"
                        ) as mock_epic:
                            mock_epic.return_value = (next_state, True)

                            try:
                                run_loop(config, tmp_path, [1], lambda x: ["1.1", "1.2"])
                            except StateError:
                                pass  # Expected when breaking loop

        mock_story.assert_called()

    def test_run_loop_sets_retrospective_when_epic_complete(self, tmp_path: Path) -> None:
        """AC3: run_loop sets phase to RETROSPECTIVE when is_epic_complete=True."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.2",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            completed_stories=["1.1"],
        )

        completed_state = State(
            current_epic=1,
            current_story="1.2",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            completed_stories=["1.1", "1.2"],
        )

        saved_states = []

        def track_save(s, path):
            saved_states.append(s)

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch(
                    "bmad_assist.core.loop.epic_phases.execute_phase", return_value=PhaseResult.ok()
                ):
                    # Both runner and epic_phases import save_state separately
                    with patch("bmad_assist.core.loop.runner.save_state", side_effect=track_save):
                        with patch(
                            "bmad_assist.core.loop.epic_phases.save_state", side_effect=track_save
                        ):
                            with patch(
                                "bmad_assist.core.loop.runner.handle_story_completion"
                            ) as mock_story:
                                mock_story.return_value = (
                                    completed_state,
                                    True,
                                )  # is_epic_complete=True
                                with patch(
                                    "bmad_assist.core.loop.runner.handle_epic_completion"
                                ) as mock_epic:
                                    mock_epic.return_value = (completed_state, True)

                                    run_loop(config, tmp_path, [1], lambda x: ["1.1", "1.1"])

        # Should have saved state with RETROSPECTIVE phase
        retrospective_saves = [s for s in saved_states if s.current_phase == Phase.RETROSPECTIVE]
        assert len(retrospective_saves) >= 1


class TestRunLoopEpicCompletion:
    """AC4: run_loop() handles epic completion."""

    def test_run_loop_handles_epic_completion_not_last(self, tmp_path: Path) -> None:
        """AC4: Epic completion advances to next epic when not last."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[],
        )

        next_state = State(
            current_epic=2,
            current_story="2.1",
            current_phase=Phase.CREATE_STORY,
            completed_epics=[1],
        )

        call_count = [0]

        def controlled_execute(s):
            call_count[0] += 1
            if call_count[0] > 3:
                raise StateError("Breaking loop for test")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch(
                "bmad_assist.core.loop.runner.execute_phase", side_effect=controlled_execute
            ):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        # First call: not complete, second: complete to break loop
                        mock_epic.side_effect = [
                            (next_state, False),
                            (next_state, True),
                        ]

                        try:
                            run_loop(config, tmp_path, [1, 2], lambda x: [f"{x}.1"])
                        except StateError:
                            pass

        assert mock_epic.called

    def test_run_loop_terminates_on_project_complete(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC4: Loop terminates gracefully when project is complete."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[],
        )

        final_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[1],
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (final_state, True)

                        with caplog.at_level(logging.INFO):
                            run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        assert "project complete" in caplog.text.lower()


class TestRunLoopFailureHandling:
    """AC5: run_loop() handles phase failures."""

    def test_run_loop_handles_phase_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC5: Loop logs warning and calls guardian on failure."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import GuardianDecision, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
        )

        call_count = [0]

        def failing_execute(s):
            call_count[0] += 1
            if call_count[0] == 1:
                return PhaseResult.fail("Test failure")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", side_effect=failing_execute):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch(
                        "bmad_assist.core.loop.runner.guardian_check_anomaly",
                        return_value=GuardianDecision.CONTINUE,
                    ) as mock_guard:
                        with patch(
                            "bmad_assist.core.loop.runner.handle_epic_completion"
                        ) as mock_epic:
                            mock_epic.return_value = (state, True)

                            with caplog.at_level(logging.WARNING):
                                run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        mock_guard.assert_called()
        assert "fail" in caplog.text.lower()

    def test_run_loop_saves_state_before_guardian(self, tmp_path: Path) -> None:
        """AC5: State is saved BEFORE guardian call on failures."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import GuardianDecision, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
        )

        call_order = []

        def track_save(s, path):
            call_order.append("save")

        def track_guardian(result, s):
            call_order.append("guardian")
            return GuardianDecision.CONTINUE

        call_count = [0]

        def failing_then_success(s):
            call_count[0] += 1
            if call_count[0] == 1:
                return PhaseResult.fail("Error")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch(
                "bmad_assist.core.loop.runner.execute_phase", side_effect=failing_then_success
            ):
                with patch("bmad_assist.core.loop.runner.save_state", side_effect=track_save):
                    with patch(
                        "bmad_assist.core.loop.runner.guardian_check_anomaly",
                        side_effect=track_guardian,
                    ):
                        with patch(
                            "bmad_assist.core.loop.runner.handle_epic_completion"
                        ) as mock_epic:
                            mock_epic.return_value = (state, True)

                            run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # Verify save happens before guardian
        save_idx = call_order.index("save")
        guardian_idx = call_order.index("guardian")
        assert save_idx < guardian_idx

    def test_run_loop_guardian_halt_terminates(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC5: Guardian 'halt' response terminates loop gracefully."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import GuardianDecision, LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch(
                "bmad_assist.core.loop.runner.execute_phase",
                return_value=PhaseResult.fail("Error"),
            ):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch(
                        "bmad_assist.core.loop.runner.guardian_check_anomaly",
                        return_value=GuardianDecision.HALT,
                    ):
                        with caplog.at_level(logging.DEBUG):
                            # Should return normally (not exception)
                            result = run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # Guardian halt returns GUARDIAN_HALT exit reason
        assert result == LoopExitReason.GUARDIAN_HALT

    def test_run_loop_retrospective_failure_halts_without_advancing_epic(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resumed teardown failure should halt without calling epic completion."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        # Start with an epic in RETROSPECTIVE phase to simulate resume during teardown
        initial_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[],
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=initial_state):
            with patch(
                "bmad_assist.core.loop.runner.execute_phase",
                return_value=PhaseResult.fail("Retrospective analysis failed"),
            ):
                with patch("bmad_assist.core.loop.runner.save_state") as mock_save_state:
                    with patch(
                        "bmad_assist.core.loop.runner.handle_epic_completion"
                    ) as mock_handle_epic_completion:
                        with caplog.at_level(logging.WARNING):
                            result = run_loop(config, tmp_path, [1, 2], lambda x: [f"{x}.1"])

        assert "RETROSPECTIVE" in caplog.text and "failed" in caplog.text
        mock_handle_epic_completion.assert_not_called()
        mock_save_state.assert_called()
        assert result == LoopExitReason.GUARDIAN_HALT


class TestRunLoopRunTracking:
    """Persist terminal run status truthfully for diagnostics and resume."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create a minimal config suitable for runner-level persistence tests."""
        return load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

    @pytest.mark.parametrize(
        ("exit_reason", "expected_status"),
        [
            ("completed", "completed"),
            ("cancelled", "cancelled"),
            ("guardian_halt", "halted"),
            ("interrupted_sigint", "interrupted"),
            ("interrupted_sigterm", "interrupted"),
        ],
    )
    def test_run_loop_persists_terminal_status(
        self,
        tmp_path: Path,
        config: Config,
        exit_reason: str,
        expected_status: str,
    ) -> None:
        """run_loop should persist the true terminal status for non-error exits."""
        from copy import deepcopy

        from bmad_assist.core.loop.runner import run_loop
        from bmad_assist.core.loop.types import LoopExitReason

        saved_logs = []

        def capture_run_log(run_log, *_args, **_kwargs):
            saved_logs.append(deepcopy(run_log))
            return tmp_path / ".bmad-assist" / "runs" / "dummy.yaml"

        with patch(
            "bmad_assist.core.config.load_loop_config",
            return_value=MagicMock(epic_setup=[], story=[], epic_teardown=[]),
        ):
            with patch("bmad_assist.core.config.set_loop_config"):
                with patch("bmad_assist.core.loop.runner.init_handlers"):
                    with patch("bmad_assist.core.loop.runner._ensure_sprint_sync_callback"):
                        with patch(
                            "bmad_assist.core.loop.runner._running_lock",
                            return_value=contextlib.nullcontext(),
                        ):
                            with patch(
                                "bmad_assist.core.loop.runner._run_loop_body",
                                return_value=LoopExitReason(exit_reason),
                            ):
                                with patch(
                                    "bmad_assist.core.loop.runner.save_run_log",
                                    side_effect=capture_run_log,
                                ):
                                    result = run_loop(
                                        config,
                                        tmp_path,
                                        [1],
                                        lambda _epic: ["1.1"],
                                        skip_signal_handlers=True,
                                        ipc_enabled=False,
                                    )

        assert result == LoopExitReason(exit_reason)
        assert len(saved_logs) == 2
        assert saved_logs[0].status.value == "running"
        assert saved_logs[0].exit_reason is None
        assert saved_logs[-1].status.value == expected_status
        assert saved_logs[-1].exit_reason == exit_reason
        assert saved_logs[-1].ended_at is not None

    def test_run_loop_persists_crash_status_on_unhandled_exception(
        self, tmp_path: Path, config: Config
    ) -> None:
        """run_loop should persist crash metadata when _run_loop_body raises."""
        from copy import deepcopy

        from bmad_assist.core.loop.runner import run_loop

        saved_logs = []

        def capture_run_log(run_log, *_args, **_kwargs):
            saved_logs.append(deepcopy(run_log))
            return tmp_path / ".bmad-assist" / "runs" / "dummy.yaml"

        with pytest.raises(RuntimeError, match="boom"):
            with patch(
                "bmad_assist.core.config.load_loop_config",
                return_value=MagicMock(epic_setup=[], story=[], epic_teardown=[]),
            ):
                with patch("bmad_assist.core.config.set_loop_config"):
                    with patch("bmad_assist.core.loop.runner.init_handlers"):
                        with patch("bmad_assist.core.loop.runner._ensure_sprint_sync_callback"):
                            with patch(
                                "bmad_assist.core.loop.runner._running_lock",
                                return_value=contextlib.nullcontext(),
                            ):
                                with patch(
                                    "bmad_assist.core.loop.runner._run_loop_body",
                                    side_effect=RuntimeError("boom"),
                                ):
                                    with patch(
                                        "bmad_assist.core.loop.runner.save_run_log",
                                        side_effect=capture_run_log,
                                    ):
                                        run_loop(
                                            config,
                                            tmp_path,
                                            [1],
                                            lambda _epic: ["1.1"],
                                            skip_signal_handlers=True,
                                            ipc_enabled=False,
                                        )

        assert len(saved_logs) == 2
        assert saved_logs[-1].status.value == "crashed"
        assert saved_logs[-1].exit_reason == "error"
        assert saved_logs[-1].ended_at is not None

    def test_run_loop_does_not_persist_running_log_when_lock_acquisition_fails(
        self, tmp_path: Path, config: Config
    ) -> None:
        """run_loop should not create a RUNNING log if the process lock is denied."""
        from bmad_assist.core.loop.runner import run_loop

        with pytest.raises(StateError, match="already running"):
            with patch(
                "bmad_assist.core.config.load_loop_config",
                return_value=MagicMock(epic_setup=[], story=[], epic_teardown=[]),
            ):
                with patch("bmad_assist.core.config.set_loop_config"):
                    with patch("bmad_assist.core.loop.runner.init_handlers"):
                        with patch("bmad_assist.core.loop.runner._ensure_sprint_sync_callback"):
                            with patch(
                                "bmad_assist.core.loop.runner._running_lock",
                                side_effect=StateError("already running"),
                            ):
                                with patch("bmad_assist.core.loop.runner.save_run_log") as mock_save:
                                    run_loop(
                                        config,
                                        tmp_path,
                                        [1],
                                        lambda _epic: ["1.1"],
                                        skip_signal_handlers=True,
                                        ipc_enabled=False,
                                    )

        mock_save.assert_not_called()

    def test_run_loop_persists_running_log_only_after_lock_acquired(
        self, tmp_path: Path, config: Config
    ) -> None:
        """run_loop should enter the process lock before persisting RUNNING state."""
        from bmad_assist.core.loop.runner import run_loop
        from bmad_assist.core.loop.types import LoopExitReason

        events: list[str] = []

        @contextlib.contextmanager
        def tracking_lock(_project_path: Path):
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")

        def capture_run_log(run_log, *_args, **_kwargs):
            events.append(f"save:{run_log.status.value}")
            return tmp_path / ".bmad-assist" / "runs" / "dummy.yaml"

        with patch(
            "bmad_assist.core.config.load_loop_config",
            return_value=MagicMock(epic_setup=[], story=[], epic_teardown=[]),
        ):
            with patch("bmad_assist.core.config.set_loop_config"):
                with patch("bmad_assist.core.loop.runner.init_handlers"):
                    with patch("bmad_assist.core.loop.runner._ensure_sprint_sync_callback"):
                        with patch(
                            "bmad_assist.core.loop.runner._running_lock",
                            side_effect=tracking_lock,
                        ):
                            with patch(
                                "bmad_assist.core.loop.runner._run_loop_body",
                                return_value=LoopExitReason.COMPLETED,
                            ):
                                with patch(
                                    "bmad_assist.core.loop.runner.save_run_log",
                                    side_effect=capture_run_log,
                                ):
                                    result = run_loop(
                                        config,
                                        tmp_path,
                                        [1],
                                        lambda _epic: ["1.1"],
                                        skip_signal_handlers=True,
                                        ipc_enabled=False,
                                    )

        assert result == LoopExitReason.COMPLETED
        assert events.index("lock_enter") < events.index("save:running")
        assert events.count("save:running") == 1
        assert events[-1] == "lock_exit"

    def test_run_loop_pre_exit_finalizer_marks_run_interrupted_and_removes_lock(
        self, tmp_path: Path, config: Config
    ) -> None:
        """Hard signal cleanup should finalize the active run before os._exit."""
        from copy import deepcopy

        from bmad_assist.core.loop.runner import run_loop

        lock_path = tmp_path / ".bmad-assist" / "running.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("1234\n2026-01-01T00:00:00+00:00\n", encoding="utf-8")

        saved_logs = []
        captured_callback = {}
        unregistered_callbacks = []

        def capture_run_log(run_log, *_args, **_kwargs):
            saved_logs.append(deepcopy(run_log))
            return tmp_path / ".bmad-assist" / "runs" / "dummy.yaml"

        def register_callback(callback):
            captured_callback["fn"] = callback

        def unregister_callback(callback):
            unregistered_callbacks.append(callback)

        def stop_with_signal(*_args, **_kwargs):
            captured_callback["fn"](signal.SIGTERM)
            raise SystemExit(143)

        with pytest.raises(SystemExit) as exc_info:
            with patch(
                "bmad_assist.core.config.load_loop_config",
                return_value=MagicMock(epic_setup=[], story=[], epic_teardown=[]),
            ):
                with patch("bmad_assist.core.config.set_loop_config"):
                    with patch("bmad_assist.core.loop.runner.init_handlers"):
                        with patch("bmad_assist.core.loop.runner._ensure_sprint_sync_callback"):
                            with patch(
                                "bmad_assist.core.loop.runner._running_lock",
                                return_value=contextlib.nullcontext(),
                            ):
                                with patch(
                                    "bmad_assist.core.loop.runner.register_pre_exit_cleanup",
                                    side_effect=register_callback,
                                ):
                                    with patch(
                                        "bmad_assist.core.loop.runner.unregister_pre_exit_cleanup",
                                        side_effect=unregister_callback,
                                    ):
                                        with patch(
                                            "bmad_assist.core.loop.runner._run_loop_body",
                                            side_effect=stop_with_signal,
                                        ):
                                            with patch(
                                                "bmad_assist.core.loop.runner.save_run_log",
                                                side_effect=capture_run_log,
                                            ):
                                                run_loop(
                                                    config,
                                                    tmp_path,
                                                    [1],
                                                    lambda _epic: ["1.1"],
                                                    skip_signal_handlers=True,
                                                    ipc_enabled=False,
                                                )

        assert exc_info.value.code == 143
        assert lock_path.exists() is False
        assert captured_callback["fn"] in unregistered_callbacks
        assert any(
            log.status.value == "interrupted" and log.exit_reason == "interrupted_sigterm"
            for log in saved_logs
        )


class TestRunLoopStatePersistence:
    """AC6: run_loop() saves state after each phase."""

    def test_run_loop_saves_state_after_each_phase(self, tmp_path: Path) -> None:
        """AC6: State is saved after every phase completion."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
        )

        save_count = [0]

        def track_save(s, path):
            save_count[0] += 1

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", return_value=PhaseResult.ok()):
                with patch("bmad_assist.core.loop.runner.save_state", side_effect=track_save):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (state, True)

                        run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # Should have saved at least once
        assert save_count[0] >= 1


class TestRunLoopPhaseAdvancement:
    """AC7: run_loop() advances phases correctly."""

    def test_run_loop_uses_get_next_phase(self, tmp_path: Path) -> None:
        """AC7: Normal phases advance via get_next_phase()."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
        )

        phases_seen = []

        def track_execute(s):
            phases_seen.append(s.current_phase)
            if len(phases_seen) >= 4:
                raise StateError("Breaking loop")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.loop.runner.execute_phase", side_effect=track_execute):
                with patch("bmad_assist.core.loop.runner.save_state"):
                    with patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic:
                        mock_epic.return_value = (state, True)

                        try:
                            run_loop(config, tmp_path, [1], lambda x: ["1.1"])
                        except StateError:
                            pass

        # Phases should advance in order
        assert phases_seen[0] == Phase.CREATE_STORY
        if len(phases_seen) > 1:
            assert phases_seen[1] == Phase.VALIDATE_STORY


class TestRunLoopIntegrationFull:
    """Integration test for full loop cycle."""

    def test_run_loop_full_single_story_cycle(self, tmp_path: Path) -> None:
        """Integration: Full loop for single story, single epic."""
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        phases_executed = []

        def track_execute(s):
            phases_executed.append(s.current_phase)
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=State()):
            with patch("bmad_assist.core.loop.runner.execute_phase", side_effect=track_execute):
                with patch(
                    "bmad_assist.core.loop.epic_phases.execute_phase", side_effect=track_execute
                ):
                    with patch("bmad_assist.core.loop.runner.save_state"):
                        with patch(
                            "bmad_assist.core.loop.runner.handle_story_completion"
                        ) as mock_story:
                            completed_state = State(
                                current_epic=1,
                                current_story="1.1",
                                current_phase=Phase.CODE_REVIEW_SYNTHESIS,
                                completed_stories=["1.1"],
                            )
                            mock_story.return_value = (completed_state, True)  # Last in epic
                            with patch(
                                "bmad_assist.core.loop.runner.handle_epic_completion"
                            ) as mock_epic:
                                final_state = State(
                                    current_epic=1,
                                    current_story="1.1",
                                    current_phase=Phase.RETROSPECTIVE,
                                    completed_epics=[1],
                                )
                                mock_epic.return_value = (final_state, True)  # Project complete

                                run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # DEFAULT_LOOP_CONFIG is minimal (no TEA phases):
        # 6 story phases + RETROSPECTIVE in epic_teardown = 7 total
        # With --tea flag (TEA_FULL_LOOP_CONFIG), would be 10 phases
        assert len(phases_executed) == 7
        assert phases_executed[0] == Phase.CREATE_STORY
        assert phases_executed[-1] == Phase.RETROSPECTIVE


class TestCodeReviewReworkPolicy:
    """Tests for strict handling of unresolved negative code review verdicts."""

    def test_run_loop_completes_negative_verdict_with_zero_unresolved_action_items(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A resolved synthesis should not re-enter dev-story on stale review verdict."""
        from bmad_assist.core.config import LoopConfig
        from bmad_assist.core.loop import LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
                "loop": {
                    "story": ["dev_story", "code_review", "code_review_synthesis"],
                    "code_review_rework": True,
                    "max_rework_attempts": 2,
                },
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            code_review_rework_count=0,
        )
        completed_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            completed_stories=["1.1"],
        )
        final_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[1],
        )
        loop_config = LoopConfig(
            epic_setup=[],
            story=["dev_story", "code_review", "code_review_synthesis"],
            epic_teardown=[],
            code_review_rework=True,
            max_rework_attempts=2,
        )

        phase_result = PhaseResult.ok(
            {
                "verdict": "MAJOR_REWORK",
                "unresolved_code_review_action_items": 0,
            }
        )

        with (
            patch("bmad_assist.core.loop.runner.load_state", return_value=state),
            patch("bmad_assist.core.config.load_loop_config", return_value=loop_config),
            patch(
                "bmad_assist.core.loop.runner.execute_phase",
                return_value=phase_result,
            ) as mock_execute_phase,
            patch("bmad_assist.git.auto_commit_phase"),
            patch("bmad_assist.core.loop.runner.save_state"),
            patch("bmad_assist.core.loop.runner._run_archive_artifacts"),
            patch(
                "bmad_assist.core.loop.runner.handle_story_completion"
            ) as mock_story_completion,
            patch("bmad_assist.core.loop.runner.handle_epic_completion") as mock_epic_completion,
            caplog.at_level(logging.INFO),
        ):
            mock_story_completion.return_value = (completed_state, True)
            mock_epic_completion.return_value = (final_state, True)
            result = run_loop(
                config,
                tmp_path,
                [1],
                lambda x: ["1.1"],
            )

        assert result == LoopExitReason.COMPLETED
        assert mock_execute_phase.call_count == 1
        assert "zero unresolved synthesis action items" in caplog.text
        assert "looping back to DEV_STORY" not in caplog.text
        mock_story_completion.assert_called_once()

    def test_run_loop_reworks_negative_verdict_when_resolution_count_unknown(
        self, tmp_path: Path
    ) -> None:
        """Ambiguous synthesis output should keep fail-closed rework behavior."""
        from bmad_assist.core.config import LoopConfig
        from bmad_assist.core.loop import LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
                "loop": {
                    "story": ["dev_story", "code_review", "code_review_synthesis"],
                    "code_review_rework": True,
                    "max_rework_attempts": 2,
                },
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            code_review_rework_count=0,
        )
        loop_config = LoopConfig(
            epic_setup=[],
            story=["dev_story", "code_review", "code_review_synthesis"],
            epic_teardown=[],
            code_review_rework=True,
            max_rework_attempts=2,
        )
        saved_states: list[State] = []

        def capture_state(next_state: State, *_args) -> None:
            saved_states.append(next_state)

        with (
            patch("bmad_assist.core.loop.runner.load_state", return_value=state),
            patch("bmad_assist.core.config.load_loop_config", return_value=loop_config),
            patch(
                "bmad_assist.core.loop.runner.execute_phase",
                side_effect=[
                    PhaseResult.ok({"verdict": "MAJOR_REWORK"}),
                    PhaseResult.fail("stop after proving rework loop"),
                ],
            ) as mock_execute_phase,
            patch("bmad_assist.git.auto_commit_phase"),
            patch(
                "bmad_assist.core.loop.runner.save_state",
                side_effect=capture_state,
            ),
            patch("bmad_assist.core.loop.runner._run_archive_artifacts") as mock_archive,
            patch(
                "bmad_assist.core.loop.runner.handle_story_completion"
            ) as mock_story_completion,
        ):
            result = run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        assert result == LoopExitReason.GUARDIAN_HALT
        assert mock_execute_phase.call_count == 2
        assert any(
            saved_state.current_phase == Phase.DEV_STORY
            and saved_state.code_review_rework_count == 1
            for saved_state in saved_states
        )
        mock_archive.assert_not_called()
        mock_story_completion.assert_not_called()

    def test_run_loop_stops_when_negative_verdict_remains_unresolved(
        self, tmp_path: Path
    ) -> None:
        """Runner fails closed after exhausting rework attempts by default."""
        from bmad_assist.core.config import LoopConfig
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
                "loop": {
                    "story": ["dev_story", "code_review", "code_review_synthesis"],
                    "code_review_rework": True,
                    "max_rework_attempts": 2,
                },
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            code_review_rework_count=2,
        )
        loop_config = LoopConfig(
            epic_setup=[],
            story=["dev_story", "code_review", "code_review_synthesis"],
            epic_teardown=[],
            code_review_rework=True,
            max_rework_attempts=2,
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.config.load_loop_config", return_value=loop_config):
                with patch(
                    "bmad_assist.core.loop.runner.execute_phase",
                    return_value=PhaseResult.ok({"verdict": "MAJOR_REWORK"}),
                ):
                    with patch("bmad_assist.git.auto_commit_phase"):
                        with patch("bmad_assist.core.loop.runner.save_state"):
                            with patch(
                                "bmad_assist.core.loop.runner._run_archive_artifacts"
                            ) as mock_archive:
                                with patch(
                                    "bmad_assist.core.loop.runner.handle_story_completion"
                                ) as mock_story_completion:
                                    with pytest.raises(
                                        StateError,
                                        match="remains unresolved after max rework attempts",
                                    ):
                                        run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        mock_archive.assert_not_called()
        mock_story_completion.assert_not_called()

    def test_run_loop_can_explicitly_continue_after_negative_verdict(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Legacy continue behavior requires an explicit config opt-out."""
        from bmad_assist.core.config import LoopConfig
        from bmad_assist.core.loop import LoopExitReason, PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
                "loop": {
                    "story": ["dev_story", "code_review", "code_review_synthesis"],
                    "code_review_rework": True,
                    "max_rework_attempts": 2,
                    "fail_on_unresolved_negative_code_review": False,
                },
            }
        )

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            code_review_rework_count=2,
        )
        completed_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CODE_REVIEW_SYNTHESIS,
            completed_stories=["1.1"],
        )
        final_state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.RETROSPECTIVE,
            completed_epics=[1],
        )
        loop_config = LoopConfig(
            epic_setup=[],
            story=["dev_story", "code_review", "code_review_synthesis"],
            epic_teardown=[],
            code_review_rework=True,
            max_rework_attempts=2,
            fail_on_unresolved_negative_code_review=False,
        )

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch("bmad_assist.core.config.load_loop_config", return_value=loop_config):
                with patch(
                    "bmad_assist.core.loop.runner.execute_phase",
                    return_value=PhaseResult.ok({"verdict": "MAJOR_REWORK"}),
                ):
                    with patch("bmad_assist.git.auto_commit_phase"):
                        with patch("bmad_assist.core.loop.runner.save_state"):
                            with patch(
                                "bmad_assist.core.loop.runner._run_archive_artifacts"
                            ) as mock_archive:
                                with patch(
                                    "bmad_assist.core.loop.runner.handle_story_completion"
                                ) as mock_story_completion:
                                    mock_story_completion.return_value = (completed_state, True)
                                    with patch(
                                        "bmad_assist.core.loop.runner.handle_epic_completion"
                                    ) as mock_epic_completion:
                                        mock_epic_completion.return_value = (final_state, True)
                                        with caplog.at_level(logging.WARNING):
                                            result = run_loop(
                                                config, tmp_path, [1], lambda x: ["1.1"]
                                            )

        assert result == LoopExitReason.COMPLETED
        assert (
            "continuing because loop.fail_on_unresolved_negative_code_review=false"
            in caplog.text
        )
        mock_archive.assert_called_once()
        mock_story_completion.assert_called_once()


class TestPhaseTimingReset:
    """Story standalone-03: Phase timing is reset before each phase execution."""

    def test_start_phase_timing_called_before_each_execute_phase(self, tmp_path: Path) -> None:
        """AC1: start_phase_timing() is called BEFORE each execute_phase() in main loop.

        The bug was that start_phase_timing was only called on resume, not before
        each phase execution. This test verifies that for every execute_phase call
        in the main loop, there's a preceding start_phase_timing call.
        """
        from bmad_assist.core.config import load_config
        from bmad_assist.core.loop import PhaseResult, run_loop
        from bmad_assist.core.state import Phase, State

        config = load_config(
            {
                "providers": {"master": {"provider": "claude", "model": "opus_4"}},
                "state_path": str(tmp_path / "state.yaml"),
            }
        )

        # State with phase_started_at already set (so resume path won't call timing)
        from datetime import datetime

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
            phase_started_at=datetime(2026, 1, 1, 12, 0, 0),  # Pre-set to skip resume timing
        )

        call_sequence: list[str] = []
        execute_count = [0]

        def track_start_phase_timing(s):
            call_sequence.append("start_timing")

        def track_execute_phase(s):
            execute_count[0] += 1
            call_sequence.append(f"execute_{execute_count[0]}")
            return PhaseResult.ok()

        with patch("bmad_assist.core.loop.runner.load_state", return_value=state):
            with patch(
                "bmad_assist.core.loop.runner.start_phase_timing",
                side_effect=track_start_phase_timing,
            ):
                with patch(
                    "bmad_assist.core.loop.runner.execute_phase",
                    side_effect=track_execute_phase,
                ):
                    with patch("bmad_assist.core.loop.runner.save_state"):
                        with patch(
                            "bmad_assist.core.loop.runner.handle_epic_completion"
                        ) as mock_epic:
                            mock_epic.return_value = (state, True)

                            run_loop(config, tmp_path, [1], lambda x: ["1.1"])

        # For every execute_phase call, there should be a start_phase_timing before it
        # The pattern should be: start_timing, execute_1, start_timing, execute_2, ...
        timing_count = call_sequence.count("start_timing")
        execute_count_total = execute_count[0]

        # Each execute should have been preceded by a timing call
        assert timing_count >= execute_count_total, (
            f"Expected at least {execute_count_total} start_phase_timing calls "
            f"(one per execute_phase), but got {timing_count}. "
            f"Sequence: {call_sequence}"
        )

        # Verify the pattern: timing should immediately precede each execute
        for i, item in enumerate(call_sequence):
            if item.startswith("execute_"):
                # The item before should be start_timing
                assert i > 0 and call_sequence[i - 1] == "start_timing", (
                    f"execute_phase at position {i} was not preceded by start_timing. "
                    f"Sequence: {call_sequence}"
                )


class TestStory65Exports:
    """Test Story 6.5 functions are properly exported."""

    def test_get_next_phase_exported(self) -> None:
        """get_next_phase is in loop module's __all__."""
        from bmad_assist.core import loop

        assert "get_next_phase" in loop.__all__

    def test_guardian_check_anomaly_exported(self) -> None:
        """guardian_check_anomaly is in loop module's __all__."""
        from bmad_assist.core import loop

        assert "guardian_check_anomaly" in loop.__all__

    def test_run_loop_exported(self) -> None:
        """run_loop is in loop module's __all__."""
        from bmad_assist.core import loop

        assert "run_loop" in loop.__all__


class TestArchiveArtifacts:
    """Tests for _run_archive_artifacts helper function."""

    def test_archive_called_on_code_review_synthesis_success(self, tmp_path: Path) -> None:
        """Archive script is called after CODE_REVIEW_SYNTHESIS phase success."""
        from bmad_assist.core.loop.runner import _run_archive_artifacts

        # Create mock script
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "archive-artifacts.sh"
        script_path.write_text("#!/bin/bash\nexit 0\n")
        script_path.chmod(0o755)

        # subprocess is imported in sprint_sync.py, not runner.py
        with patch("bmad_assist.core.loop.sprint_sync.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _run_archive_artifacts(tmp_path)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert str(script_path) in call_args[0][0]
        assert "-s" in call_args[0][0]  # Silent mode

    def test_archive_skipped_when_script_missing(self, tmp_path: Path) -> None:
        """Archive is skipped gracefully when script doesn't exist."""
        from bmad_assist.core.loop.runner import _run_archive_artifacts

        # subprocess is imported in sprint_sync.py, not runner.py
        with patch("bmad_assist.core.loop.sprint_sync.subprocess.run") as mock_run:
            _run_archive_artifacts(tmp_path)

        mock_run.assert_not_called()

    def test_archive_handles_script_failure(self, tmp_path: Path) -> None:
        """Archive handles script failure gracefully without crashing."""
        from bmad_assist.core.loop.runner import _run_archive_artifacts

        # Create mock script
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script_path = scripts_dir / "archive-artifacts.sh"
        script_path.write_text("#!/bin/bash\nexit 1\n")
        script_path.chmod(0o755)

        # subprocess is imported in sprint_sync.py, not runner.py
        with patch("bmad_assist.core.loop.sprint_sync.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            # Should not raise
            _run_archive_artifacts(tmp_path)
