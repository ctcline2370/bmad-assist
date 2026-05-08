"""Tests for LoopController.

Story: Direct Orchestrator Integration for Dashboard.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.core.loop.types import LoopExitReason
from bmad_assist.dashboard.loop_controller import ControllerState, LoopController


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = MagicMock()
    config.paths = None
    return config


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    project_path = tmp_path / "test-project"
    project_path.mkdir()
    return project_path


class TestControllerState:
    """Tests for ControllerState enum."""

    def test_state_values(self):
        """Verify state enum values."""
        assert ControllerState.IDLE.value == "idle"
        assert ControllerState.STARTING.value == "starting"
        assert ControllerState.RUNNING.value == "running"
        assert ControllerState.PAUSED.value == "paused"
        assert ControllerState.STOPPING.value == "stopping"


class TestLoopControllerInit:
    """Tests for LoopController initialization."""

    def test_initial_state_is_idle(self, mock_config, temp_project):
        """Controller starts in IDLE state."""
        controller = LoopController(temp_project, mock_config)
        try:
            assert controller._state == ControllerState.IDLE
            assert controller.is_running is False
            assert controller.is_paused is False
        finally:
            controller.shutdown()

    def test_initial_status(self, mock_config, temp_project):
        """get_status returns correct initial values."""
        controller = LoopController(temp_project, mock_config)
        try:
            status = controller.get_status()
            assert status["state"] == "idle"
            assert status["running"] is False
            assert status["paused"] is False
            assert status["current_epic"] is None
            assert status["current_story"] is None
            assert status["current_phase"] is None
            assert status["error"] is None
        finally:
            controller.shutdown()


class TestLoopControllerStateTransitions:
    """Tests for state machine transitions."""

    def test_start_when_not_idle_returns_current_status(
        self, mock_config, temp_project
    ):
        """start() when not IDLE just returns status."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Force state to RUNNING
            controller._state = ControllerState.RUNNING

            # start() should return without starting
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(controller.start())
            finally:
                loop.close()

            assert status["state"] == "running"
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()

    def test_stop_when_not_running_returns_current_status(
        self, mock_config, temp_project
    ):
        """stop() when not RUNNING returns status without action."""
        controller = LoopController(temp_project, mock_config)
        try:
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(controller.stop())
            finally:
                loop.close()

            assert status["state"] == "idle"
        finally:
            controller.shutdown()

    def test_pause_when_not_running_returns_current_status(
        self, mock_config, temp_project
    ):
        """pause() when not RUNNING returns status without action."""
        controller = LoopController(temp_project, mock_config)
        try:
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(controller.pause())
            finally:
                loop.close()

            assert status["state"] == "idle"
        finally:
            controller.shutdown()

    def test_resume_when_not_paused_returns_current_status(
        self, mock_config, temp_project
    ):
        """resume() when not PAUSED returns status without action."""
        controller = LoopController(temp_project, mock_config)
        try:
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(controller.resume())
            finally:
                loop.close()

            assert status["state"] == "idle"
        finally:
            controller.shutdown()


class TestLoopControllerUpdatePosition:
    """Tests for position update functionality."""

    def test_update_position_sets_epic(self, mock_config, temp_project):
        """update_position() sets current epic."""
        controller = LoopController(temp_project, mock_config)
        try:
            controller.update_position(epic=1)
            assert controller._current_epic == 1
        finally:
            controller.shutdown()

    def test_update_position_sets_story(self, mock_config, temp_project):
        """update_position() sets current story."""
        controller = LoopController(temp_project, mock_config)
        try:
            controller.update_position(story="1.1")
            assert controller._current_story == "1.1"
        finally:
            controller.shutdown()

    def test_update_position_sets_phase(self, mock_config, temp_project):
        """update_position() sets current phase."""
        controller = LoopController(temp_project, mock_config)
        try:
            controller.update_position(phase="dev_story")
            assert controller._current_phase == "dev_story"
        finally:
            controller.shutdown()

    def test_update_position_thread_safe(self, mock_config, temp_project):
        """update_position() is thread-safe."""
        controller = LoopController(temp_project, mock_config)
        try:
            updates = []

            def update_loop():
                for i in range(50):
                    controller.update_position(epic=i, story=f"{i}.1", phase=f"phase_{i}")
                    updates.append(i)

            threads = [threading.Thread(target=update_loop) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All updates completed without error
            assert len(updates) == 150
        finally:
            controller.shutdown()


class TestLoopControllerPauseResume:
    """Tests for pause/resume with file flags."""

    def test_pause_creates_flag_file(self, mock_config, temp_project):
        """pause() creates pause.flag file."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Force RUNNING state
            controller._state = ControllerState.RUNNING

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(controller.pause())
            finally:
                loop.close()

            pause_flag = temp_project / ".bmad-assist" / "pause.flag"
            assert pause_flag.exists()
            assert controller._state == ControllerState.PAUSED
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()

    def test_resume_removes_flag_file(self, mock_config, temp_project):
        """resume() removes pause.flag file."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Create pause flag
            pause_flag = temp_project / ".bmad-assist" / "pause.flag"
            pause_flag.parent.mkdir(parents=True, exist_ok=True)
            pause_flag.touch()

            # Force PAUSED state
            controller._state = ControllerState.PAUSED

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(controller.resume())
            finally:
                loop.close()

            assert not pause_flag.exists()
            assert controller._state == ControllerState.RUNNING
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()


class TestLoopControllerIntegration:
    """Integration tests with mocked run_loop."""

    def test_start_catches_load_errors(self, mock_config, temp_project):
        """start() catches errors during epic loading and returns to IDLE."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Patch _load_epic_data to raise
            with patch.object(
                controller, "_load_epic_data", side_effect=FileNotFoundError("No BMAD docs")
            ):
                loop = asyncio.new_event_loop()
                try:
                    status = loop.run_until_complete(controller.start())
                finally:
                    loop.close()

            assert status["state"] == "idle"
            assert "No BMAD docs" in status["error"]
        finally:
            controller.shutdown()

    def test_start_with_empty_epic_list_returns_error(self, mock_config, temp_project):
        """start() with empty epic list sets error and returns to IDLE."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Patch _load_epic_data to return empty
            with patch.object(controller, "_load_epic_data", return_value=([], lambda _: [])):
                loop = asyncio.new_event_loop()
                try:
                    status = loop.run_until_complete(controller.start())
                finally:
                    loop.close()

            assert status["state"] == "idle"
            assert status["error"] == "No epics found in project"
        finally:
            controller.shutdown()

    def test_run_loop_wrapper_handles_exception(self, mock_config, temp_project):
        """_run_loop_wrapper catches exceptions and sets error."""
        controller = LoopController(temp_project, mock_config)
        try:
            with patch(
                "bmad_assist.core.loop.runner.run_loop",
                side_effect=RuntimeError("Provider failed"),
            ):
                result = controller._run_loop_wrapper([1], lambda _: ["1.1"])

            assert result == LoopExitReason.ERROR
            assert "Provider failed" in controller._error
            assert controller._state == ControllerState.IDLE
        finally:
            controller.shutdown()

    def test_run_loop_wrapper_returns_exit_reason(self, mock_config, temp_project):
        """_run_loop_wrapper returns exit reason from run_loop."""
        controller = LoopController(temp_project, mock_config)
        try:
            with patch(
                "bmad_assist.core.loop.runner.run_loop",
                return_value=LoopExitReason.CANCELLED,
            ):
                result = controller._run_loop_wrapper([1], lambda _: ["1.1"])

            assert result == LoopExitReason.CANCELLED
            assert controller._state == ControllerState.IDLE
        finally:
            controller.shutdown()


class TestLoopControllerConcurrentOperations:
    """Concurrent operation tests from adversarial review.

    These tests verify state machine correctness under concurrent operations.
    They use synchronous patterns to avoid async/threading timing issues.
    """

    def test_start_while_starting(self, mock_config, temp_project):
        """Concurrent start() calls should not create multiple threads."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Track how many times _load_epic_data is called
            call_count = [0]
            load_started = threading.Event()
            load_continue = threading.Event()

            def controlled_load():
                call_count[0] += 1
                load_started.set()  # Signal we entered the function
                load_continue.wait(timeout=5.0)  # Wait for permission to continue
                return ([1], lambda _: ["1.1"])

            with (
                patch.object(controller, "_load_epic_data", side_effect=controlled_load),
                patch(
                    "bmad_assist.core.loop.runner.run_loop",
                    return_value=LoopExitReason.COMPLETED,
                ),
            ):
                # Start first call in a thread
                    results = [None, None]

                    def start_call(idx):
                        loop = asyncio.new_event_loop()
                        try:
                            results[idx] = loop.run_until_complete(controller.start())
                        finally:
                            loop.close()

                    t1 = threading.Thread(target=start_call, args=(0,))
                    t1.start()

                    # Wait for first call to enter _load_epic_data
                    load_started.wait(timeout=2.0)

                    # Now fire second call - it should see non-IDLE state
                    t2 = threading.Thread(target=start_call, args=(1,))
                    t2.start()

                    # Give second call time to check state and return
                    t2.join(timeout=2.0)

                    # Let first call complete
                    load_continue.set()
                    t1.join(timeout=5.0)

            # Both should return status dicts
            assert all(isinstance(r, dict) for r in results)

            # _load_epic_data should only be called once (state machine prevents second call)
            assert call_count[0] == 1
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()

    def test_stop_during_running(self, mock_config, temp_project):
        """stop() during RUNNING state should cancel and return to IDLE."""
        controller = LoopController(temp_project, mock_config)
        try:
            loop_started = threading.Event()
            loop_continue = threading.Event()

            def blocking_run_loop(*_, **kwargs):
                loop_started.set()
                loop_continue.wait(timeout=5.0)
                cancel_ctx = kwargs.get("cancel_ctx")
                if cancel_ctx and cancel_ctx.is_cancelled:
                    return LoopExitReason.CANCELLED
                return LoopExitReason.COMPLETED

            with (
                patch.object(
                    controller, "_load_epic_data", return_value=([1], lambda _: ["1.1"])
                ),
                patch(
                    "bmad_assist.core.loop.runner.run_loop",
                    side_effect=blocking_run_loop,
                ),
            ):
                # Start in a thread
                    start_result = [None]

                    def do_start():
                        loop = asyncio.new_event_loop()
                        try:
                            start_result[0] = loop.run_until_complete(controller.start())
                        finally:
                            loop.close()

                    start_thread = threading.Thread(target=do_start)
                    start_thread.start()

                    # Wait for run_loop to begin
                    assert loop_started.wait(timeout=5.0), "run_loop did not start"

                    # Wait for state to become RUNNING (may briefly be STARTING)
                    for _ in range(50):  # 50 * 10ms = 500ms max
                        if controller._state == ControllerState.RUNNING:
                            break
                        time.sleep(0.01)
                    assert controller._state == ControllerState.RUNNING

                    # Now stop
                    stop_errors = []

                    def do_stop():
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(controller.stop())
                        except BaseException as exc:
                            stop_errors.append(exc)
                        finally:
                            loop.close()

                    stop_thread = threading.Thread(target=do_stop)
                    stop_thread.start()

                    # Allow run_loop to exit
                    loop_continue.set()

                    # Wait for both to complete
                    start_thread.join(timeout=5.0)
                    stop_thread.join(timeout=5.0)

            assert stop_errors == []

            # Controller should end up idle
            final_status = controller.get_status()
            assert final_status["state"] == "idle"
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()

    async def test_pause_resume_rapid_cycle(self, mock_config, temp_project):
        """Rapid pause/resume cycles don't corrupt state."""
        controller = LoopController(temp_project, mock_config)
        try:
            # Force RUNNING state
            controller._state = ControllerState.RUNNING

            # Rapid pause/resume cycles
            for _ in range(10):
                await controller.pause()
                assert controller._state == ControllerState.PAUSED
                await controller.resume()
                assert controller._state == ControllerState.RUNNING

            # Should end in RUNNING state
            assert controller._state == ControllerState.RUNNING
        finally:
            controller._state = ControllerState.IDLE
            controller.shutdown()
