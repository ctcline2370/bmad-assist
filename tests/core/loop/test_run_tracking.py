"""Tests for run tracking module."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from bmad_assist.core.loop.run_tracking import (
    CurrentPhase,
    MAX_ARG_LENGTH,
    PhaseInvocation,
    PhaseStatus,
    RunLog,
    RunStatus,
    SecurityError,
    _cleanup_old_tmp_files,
    _format_datetime,
    _sanitize_csv_value,
    mask_cli_args,
    reconcile_stale_running_run,
    reconcile_stale_running_runs,
    save_run_log,
)


class TestMaskCliArgs:
    """Tests for mask_cli_args function."""

    def test_masks_token_inline(self) -> None:
        """--token=secret should be masked."""
        args = ["--token=my-secret-value"]
        result = mask_cli_args(args)
        assert result == ["--token=***"]

    def test_masks_token_space_separated(self) -> None:
        """--token secret should mask the following arg."""
        args = ["--token", "my-secret-value"]
        result = mask_cli_args(args)
        assert result == ["--token", "***"]

    def test_masks_case_insensitive(self) -> None:
        """Flag matching should be case-insensitive."""
        args = ["--TOKEN=secret", "--Password=hunter2", "-KEY", "value"]
        result = mask_cli_args(args)
        assert result == ["--TOKEN=***", "--Password=***", "-KEY", "***"]

    def test_preserves_non_sensitive_args(self) -> None:
        """Non-sensitive arguments should be preserved."""
        args = ["--project", "/path/to/project", "--verbose", "-n"]
        result = mask_cli_args(args)
        assert result == ["--project", "/path/to/project", "--verbose", "-n"]

    def test_handles_mixed_args(self) -> None:
        """Mixed sensitive and non-sensitive args."""
        args = ["--project", "./", "--token", "secret", "--verbose"]
        result = mask_cli_args(args)
        assert result == ["--project", "./", "--token", "***", "--verbose"]

    def test_handles_credential_variants(self) -> None:
        """Various sensitive flag patterns should be masked."""
        args = [
            "--credential=abc",
            "--auth=xyz",
            "--secret", "shhh",
            "-password", "pass123",
        ]
        result = mask_cli_args(args)
        assert result == [
            "--credential=***",
            "--auth=***",
            "--secret", "***",
            "-password", "***",
        ]

    def test_truncates_overly_long_args(self) -> None:
        """Args exceeding MAX_ARG_LENGTH should be truncated."""
        long_arg = "a" * (MAX_ARG_LENGTH + 100)
        args = [long_arg]
        result = mask_cli_args(args)
        assert len(result) == 1
        assert result[0].endswith("...[TRUNCATED]")
        assert len(result[0]) < MAX_ARG_LENGTH

    def test_empty_list(self) -> None:
        """Empty args list should return empty."""
        assert mask_cli_args([]) == []

    def test_flag_at_end_of_list(self) -> None:
        """Sensitive flag at end of list (no following value)."""
        args = ["--project", "foo", "--token"]
        result = mask_cli_args(args)
        # mask_next is True but no next arg, so flag is preserved
        assert result == ["--project", "foo", "--token"]


class TestRunLog:
    """Tests for RunLog model."""

    def test_creates_with_defaults(self) -> None:
        """RunLog should create with sensible defaults."""
        log = RunLog()
        assert len(log.run_id) == 8
        assert log.status == RunStatus.RUNNING
        assert log.cli_args == []
        assert log.phases == []
        assert log.started_at is not None

    def test_accepts_all_fields(self) -> None:
        """RunLog should accept all fields."""
        log = RunLog(
            run_id="test1234",
            exit_reason="guardian_halt",
            cli_args=["--project", "."],
            cli_args_masked=["--project", "."],
            epic=22,
            story="22.3",
            project_path="/path/to/project",
        )
        assert log.run_id == "test1234"
        assert log.exit_reason == "guardian_halt"
        assert log.epic == 22
        assert log.story == "22.3"


class TestPhaseInvocation:
    """Tests for PhaseInvocation model."""

    def test_creates_minimal(self) -> None:
        """PhaseInvocation should create with required fields."""
        now = datetime.now(UTC)
        phase = PhaseInvocation(
            phase="CREATE_STORY",
            started_at=now,
            provider="claude",
            model="opus",
            status=PhaseStatus.SUCCESS,
        )
        assert phase.phase == "CREATE_STORY"
        assert phase.status == PhaseStatus.SUCCESS
        assert phase.error_type is None

    def test_creates_with_error(self) -> None:
        """PhaseInvocation should record error details."""
        now = datetime.now(UTC)
        phase = PhaseInvocation(
            phase="DEV_STORY",
            started_at=now,
            ended_at=now,
            duration_ms=5000,
            provider="gemini",
            model="2.5-pro",
            status=PhaseStatus.ERROR,
            error_type="TimeoutError",
        )
        assert phase.status == PhaseStatus.ERROR
        assert phase.error_type == "TimeoutError"


class TestSaveRunLog:
    """Tests for save_run_log function."""

    def test_creates_directory(self) -> None:
        """save_run_log should create .bmad-assist/runs/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            log = RunLog(run_id="test0001")

            result = save_run_log(log, project_path)

            assert result.exists()
            assert (project_path / ".bmad-assist" / "runs").is_dir()

    def test_yaml_filename_format(self) -> None:
        """Saved file should have correct naming format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            log = RunLog(run_id="abcd1234")

            result = save_run_log(log, project_path)

            assert "abcd1234" in result.name
            assert result.suffix == ".yaml"

    def test_csv_export(self) -> None:
        """as_csv=True should create both YAML and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            now = datetime.now(UTC)
            log = RunLog(
                run_id="csv12345",
                exit_reason="guardian_halt",
                epic=1,
                story="1.1",
                phases=[
                    PhaseInvocation(
                        phase="CREATE_STORY",
                        started_at=now,
                        ended_at=now,
                        duration_ms=1000,
                        provider="claude",
                        model="opus",
                        status=PhaseStatus.SUCCESS,
                    )
                ],
            )

            yaml_path = save_run_log(log, project_path, as_csv=True)
            csv_path = yaml_path.with_suffix(".csv")

            assert yaml_path.exists()
            assert csv_path.exists()

            # Check CSV content
            csv_content = csv_path.read_text()
            assert "run_id" in csv_content  # Header
            assert "csv12345" in csv_content  # Data
            assert "# Exit Reason: guardian_halt" in csv_content

    def test_detects_symlink_attack(self) -> None:
        """save_run_log should refuse to write through symlinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            target_dir = project_path / "target"
            target_dir.mkdir()

            # Create symlink at expected runs directory location
            runs_dir = project_path / ".bmad-assist" / "runs"
            runs_dir.parent.mkdir(parents=True, exist_ok=True)
            runs_dir.symlink_to(target_dir)

            log = RunLog(run_id="symlink01")

            with pytest.raises(SecurityError, match="Symlink detected"):
                save_run_log(log, project_path)


class TestStaleRunReconciliation:
    """Tests for stale running run reconciliation."""

    def test_reconciles_newest_running_run(self, tmp_path: Path) -> None:
        """Newest running run should be finalized as crashed."""
        older_started_at = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        newer_started_at = datetime(2026, 4, 20, 12, 5, tzinfo=UTC)

        older_run = RunLog(run_id="older001", started_at=older_started_at)
        newer_run = RunLog(
            run_id="newer001",
            started_at=newer_started_at,
            current_phase=CurrentPhase(
                phase="DEV_STORY",
                started_at=newer_started_at,
                provider="openai",
                model="gpt-5.4",
            ),
        )

        older_yaml = save_run_log(older_run, tmp_path)
        newer_yaml = save_run_log(newer_run, tmp_path, as_csv=True)

        reconciled = reconcile_stale_running_run(tmp_path, "stale_lock_recovered_dead_pid")

        assert reconciled == newer_yaml

        newer_data = yaml.safe_load(newer_yaml.read_text())
        assert newer_data["status"] == RunStatus.CRASHED.value
        assert newer_data["exit_reason"] == "stale_lock_recovered_dead_pid"
        assert newer_data["ended_at"] is not None
        assert newer_data["current_phase"] is None

        older_data = yaml.safe_load(older_yaml.read_text())
        assert older_data["status"] == RunStatus.CRASHED.value
        assert older_data["exit_reason"] == "stale_lock_recovered_dead_pid"
        assert older_data["ended_at"] is not None
        assert older_data["current_phase"] is None

        csv_content = newer_yaml.with_suffix(".csv").read_text()
        assert "# Status: crashed" in csv_content
        assert "# Exit Reason: stale_lock_recovered_dead_pid" in csv_content

    def test_reconciles_all_running_runs_newest_first(self, tmp_path: Path) -> None:
        """Plural helper should finalize the full stale backlog in newest-first order."""
        oldest_started_at = datetime(2026, 4, 20, 9, 0, tzinfo=UTC)
        middle_started_at = datetime(2026, 4, 20, 9, 30, tzinfo=UTC)
        newest_started_at = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)

        oldest_yaml = save_run_log(
            RunLog(
                run_id="oldest01",
                started_at=oldest_started_at,
                current_phase=CurrentPhase(
                    phase="DEV_STORY",
                    started_at=oldest_started_at,
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            tmp_path,
        )
        middle_yaml = save_run_log(
            RunLog(
                run_id="middle01",
                started_at=middle_started_at,
                current_phase=CurrentPhase(
                    phase="DEV_STORY",
                    started_at=middle_started_at,
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            tmp_path,
        )
        newest_yaml = save_run_log(
            RunLog(
                run_id="newest01",
                started_at=newest_started_at,
                current_phase=CurrentPhase(
                    phase="DEV_STORY",
                    started_at=newest_started_at,
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            tmp_path,
        )

        reconciled = reconcile_stale_running_runs(tmp_path, "stale_lock_recovered_dead_pid")

        assert reconciled == [newest_yaml, middle_yaml, oldest_yaml]
        for yaml_path in reconciled:
            data = yaml.safe_load(yaml_path.read_text())
            assert data["status"] == RunStatus.CRASHED.value
            assert data["exit_reason"] == "stale_lock_recovered_dead_pid"
            assert data["ended_at"] is not None
            assert data["current_phase"] is None

    def test_returns_none_when_no_running_run_exists(self, tmp_path: Path) -> None:
        """No-op when every persisted run is already finalized."""
        completed_run = RunLog(
            run_id="done0001",
            started_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
            ended_at=datetime(2026, 4, 20, 12, 1, tzinfo=UTC),
            status=RunStatus.COMPLETED,
        )
        save_run_log(completed_run, tmp_path)

        reconciled = reconcile_stale_running_run(tmp_path, "stale_lock_recovered_dead_pid")

        assert reconciled is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_datetime_with_value(self) -> None:
        """_format_datetime should return ISO format."""
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = _format_datetime(dt)
        assert "2026-01-15" in result
        assert "10:30:00" in result

    def test_format_datetime_none(self) -> None:
        """_format_datetime should return empty string for None."""
        assert _format_datetime(None) == ""

    def test_sanitize_csv_value_normal(self) -> None:
        """Normal values should be unchanged."""
        assert _sanitize_csv_value("hello") == "hello"
        assert _sanitize_csv_value("123") == "123"

    def test_sanitize_csv_value_formula_chars(self) -> None:
        """Formula characters should be escaped."""
        assert _sanitize_csv_value("=SUM(A1)") == "'=SUM(A1)"
        assert _sanitize_csv_value("+1234") == "'+1234"
        assert _sanitize_csv_value("-5") == "'-5"
        assert _sanitize_csv_value("@mention") == "'@mention"

    def test_sanitize_csv_value_none(self) -> None:
        """None should return empty string."""
        assert _sanitize_csv_value(None) == ""

    def test_cleanup_old_tmp_files(self) -> None:
        """Old .tmp files should be removed."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)

            # Create an "old" tmp file (mock by setting mtime in past)
            old_tmp = directory / "old.tmp"
            old_tmp.write_text("old")
            # Set mtime to 2 hours ago
            old_time = time.time() - 7200
            import os
            os.utime(old_tmp, (old_time, old_time))

            # Create a "new" tmp file
            new_tmp = directory / "new.tmp"
            new_tmp.write_text("new")

            _cleanup_old_tmp_files(directory, max_age_hours=1)

            assert not old_tmp.exists()
            assert new_tmp.exists()
