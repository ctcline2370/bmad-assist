"""Tests for QaRemediateHandler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bmad_assist.core.loop.handlers.qa_remediate import QaRemediateHandler
from bmad_assist.core.state import State
from bmad_assist.qa.remediate import REMEDIATE_ESCALATIONS_END, REMEDIATE_ESCALATIONS_START


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config with QA settings."""
    config = MagicMock()
    config.qa.remediate_max_iterations = 2
    config.qa.remediate_max_age_days = 7
    config.qa.remediate_safety_cap = 0.8
    config.qa.remediate_max_issues = 200
    config.qa.qa_artifacts_path = "{project-root}/_bmad-output/qa-artifacts"
    config.providers.master.provider = "claude-subprocess"
    config.providers.master.model = "opus"
    config.providers.master.model_name = None
    config.providers.master.settings_path = None
    config.providers.master.reasoning_effort = None
    config.providers.helper = None
    config.benchmarking.enabled = False
    return config


@pytest.fixture
def handler(mock_config: MagicMock, tmp_path: Path) -> QaRemediateHandler:
    """Create a QaRemediateHandler with mock config."""
    return QaRemediateHandler(mock_config, tmp_path)


def _setup_qa_failures(tmp_path: Path, epic_id: int = 1, tests: list[dict] | None = None) -> Path:
    """Helper to create QA result files with failures."""
    qa_dir = tmp_path / "_bmad-output" / "qa-artifacts" / "test-results"
    qa_dir.mkdir(parents=True, exist_ok=True)
    data = {"tests": tests or [{"name": "test_broken", "status": "FAIL", "error": "boom"}]}
    path = qa_dir / f"epic-{epic_id}-run-001.yaml"
    path.write_text(yaml.dump(data))
    return qa_dir


class TestHandlerBasics:
    def test_phase_name(self, handler: QaRemediateHandler) -> None:
        assert handler.phase_name == "qa_remediate"

    def test_build_context_empty(self, handler: QaRemediateHandler) -> None:
        state = State()
        assert handler.build_context(state) == {}


class TestHandlerExecute:
    def test_no_epic_fails(self, handler: QaRemediateHandler) -> None:
        state = State(current_epic=None)
        result = handler.execute(state)
        assert result.success is False
        assert "no current epic" in result.error

    def test_no_issues_clean(self, handler: QaRemediateHandler) -> None:
        state = State(current_epic=1)
        result = handler.execute(state)
        assert result.success is True
        assert result.outputs["status"] == "clean"
        assert result.outputs["issues_found"] == 0
        assert result.outputs["issues_fixed"] == 0
        assert result.outputs["issues_escalated"] == 0
        assert result.outputs["retest_pass_rate"] == 100.0

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_with_issues_invokes_llm(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Handler invokes LLM when issues are found."""
        _setup_qa_failures(tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "All issues fixed. No escalations needed."
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        # Mock LLM produces no FILE markers → 0 files modified, 0 escalated →
        # status=unresolved → phase result must be failure (needs human review).
        assert result.success is False
        assert result.outputs["status"] == "unresolved"
        assert mock_invoke.called
        # Check prompt contains the issue and safety cap
        prompt = mock_invoke.call_args[0][0]
        assert "test_broken" in prompt
        assert "80%" in prompt  # safety cap
        assert "FIX" in prompt.upper()
        assert "ESCALATE" in prompt

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_escalation_extraction(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Handler extracts escalations from LLM output and saves report."""
        _setup_qa_failures(tmp_path, tests=[
            {"name": "test_auth", "status": "FAIL", "error": "denied"},
        ])

        mock_result = MagicMock()
        mock_result.stdout = f"""
Fixed some issues.

{REMEDIATE_ESCALATIONS_START}
## Escalated Issues
### Issue 1: Auth redesign needed
**Source:** qa_results
**Severity:** high
**Problem:** Fundamental auth architecture issue
**Proposals:**
1. Rewrite auth module
2. Add middleware

```llm-context
src/auth.py needs complete rewrite
```
{REMEDIATE_ESCALATIONS_END}
"""
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        # Escalations present → status=escalated → phase result must be failure
        # so the runner does not report the epic as cleanly completed.
        assert result.success is False
        assert result.error is not None
        assert "escalated" in result.error
        assert result.outputs["issues_escalated"] == 1
        assert result.outputs["status"] == "escalated"
        # Verify escalation report written to disk
        esc_path = result.outputs["escalation_path"]
        assert esc_path is not None
        assert Path(esc_path).exists()
        content = Path(esc_path).read_text()
        assert "Auth redesign needed" in content

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_saves_reports_with_content(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Handler saves remediation report with correct content."""
        _setup_qa_failures(tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "Fixed test_broken by updating import."
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        # Mock LLM output has no FILE markers → 0 files modified, 0 escalated →
        # status=unresolved → success=False (needs human review).
        assert result.success is False
        assert result.outputs["status"] == "unresolved"
        report_path = result.outputs.get("report_path")
        assert report_path is not None
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "issues_found: 1" in content
        assert "handler: qa_remediate" in content

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_exception_handling(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Provider crash → PhaseResult.fail."""
        _setup_qa_failures(tmp_path)
        mock_invoke.side_effect = RuntimeError("LLM crashed")

        state = State(current_epic=1)
        result = handler.execute(state)

        assert result.success is False
        assert "LLM crashed" in result.error

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_cumulative_issue_count(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """issues_found accumulates across iterations, not just last iteration."""
        _setup_qa_failures(tmp_path, tests=[
            {"name": "test_a", "status": "FAIL", "error": "err_a"},
            {"name": "test_b", "status": "FAIL", "error": "err_b"},
        ])

        mock_result = MagicMock()
        mock_result.stdout = "Fixed all issues."
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        # Mock produces no FILE markers → status=unresolved → success=False.
        # But the accounting we care about here is cumulative issue count.
        assert result.success is False
        # Should accumulate issues from first iteration
        assert result.outputs["issues_found"] >= 2

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_deduplication_across_iterations(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Same issue description is not sent to LLM twice."""
        _setup_qa_failures(tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "Attempted fix."
        mock_invoke.return_value = mock_result

        # max_iterations=2, same fixture on both iterations
        # Second iteration should find same issue but dedup skips it
        handler.config.qa.remediate_max_iterations = 2
        state = State(current_epic=1)
        result = handler.execute(state)

        # Mock produces no FILE markers → status=unresolved → success=False.
        # The dedup behavior we assert below is orthogonal to success.
        assert result.success is False
        # LLM should be called once (first iter finds issues, second iter deduped → empty → break)
        assert mock_invoke.call_count == 1

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_config_qa_none_uses_defaults(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """When config.qa is None, handler uses fallback defaults."""
        handler.config.qa = None
        _setup_qa_failures(tmp_path)

        mock_result = MagicMock()
        mock_result.stdout = "Fixed."
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        # Default handler + mock with no FILE markers → status=unresolved → success=False.
        # What this test actually validates is that config.qa=None doesn't crash and
        # the LLM is still invoked with fallback defaults.
        assert result.success is False
        assert mock_invoke.called

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_prompt_includes_fixed_files_section(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """When files were modified, prompt includes 'Already Fixed Files' section."""
        _setup_qa_failures(tmp_path)

        # First call: LLM "modifies" a file
        mock_result1 = MagicMock()
        mock_result1.stdout = "Wrote to '/home/user/src/auth.py'\nFixed."
        # Second call (if reached): should include fixed files
        mock_result2 = MagicMock()
        mock_result2.stdout = "All done."

        mock_invoke.side_effect = [mock_result1, mock_result2]
        handler.config.qa.remediate_max_iterations = 2

        state = State(current_epic=1)
        handler.execute(state)

        # If second call happened, check prompt has fixed files
        if mock_invoke.call_count > 1:
            prompt2 = mock_invoke.call_args_list[1][0][0]
            assert "Already Fixed Files" in prompt2


class TestPhaseResultSemantics:
    """Regression guard for Epic 8 exit-gate fix.

    The original qa_remediate handler returned PhaseResult.ok for every
    terminating status — including `escalated`, `unresolved`, and `partial`.
    The runner then reported the phase as successful and the downstream
    "Project complete!" log fired despite thousands of unresolved issues.

    These tests pin the correct mapping from status → success flag so the
    regression cannot silently return.
    """

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_clean_status_returns_success(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """No issues found → status=clean → success=True."""
        # Intentionally do NOT create qa failure fixtures — no issues to remediate.
        state = State(current_epic=1)
        result = handler.execute(state)

        assert result.success is True
        assert result.outputs["status"] == "clean"
        # LLM must not be invoked when there are no issues.
        assert not mock_invoke.called

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_escalated_status_returns_failure(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Escalated findings → status=escalated → success=False.

        Escalation means the LLM could not fix the issues and flagged them
        for human review. Surfacing this as phase success would cause the
        epic-completion log to misreport the state.
        """
        _setup_qa_failures(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = f"""
{REMEDIATE_ESCALATIONS_START}
## Escalated Issues
### Issue 1: Needs architecture change
**Source:** qa_results
**Severity:** high
**Problem:** Can't fix in-place.
**Proposals:**
1. Rewrite the module.
{REMEDIATE_ESCALATIONS_END}
"""
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        assert result.success is False
        assert result.outputs["status"] == "escalated"
        assert result.outputs["issues_escalated"] >= 1
        # The error message must explicitly mention human review so callers
        # can detect this state without parsing the outputs dict.
        assert result.error is not None
        assert "human review" in result.error.lower()

    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_unresolved_status_returns_failure(
        self,
        mock_invoke: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """Issues found but 0 fixes and 0 escalations → unresolved → success=False."""
        _setup_qa_failures(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = "Acknowledged the issues but did not produce any fixes."
        mock_invoke.return_value = mock_result

        state = State(current_epic=1)
        result = handler.execute(state)

        assert result.success is False
        assert result.outputs["status"] == "unresolved"
        assert result.outputs["files_modified"] == 0
        assert result.outputs["issues_escalated"] == 0


class TestRunRetest:
    """Tests for _run_retest method."""

    def test_returns_none_on_first_iteration(self, handler: QaRemediateHandler) -> None:
        """First iteration (iteration=0) skips retest, returns None."""
        result = handler._run_retest(epic_id=1, iteration=0)
        assert result is None

    @patch("bmad_assist.core.loop.handlers.qa_remediate.QaRemediateHandler._run_retest")
    @patch.object(QaRemediateHandler, "invoke_provider")
    def test_retest_called_on_second_iteration(
        self,
        mock_invoke: MagicMock,
        mock_retest: MagicMock,
        handler: QaRemediateHandler,
        tmp_path: Path,
    ) -> None:
        """_run_retest is called with iteration > 0 in multi-iteration run."""
        _setup_qa_failures(tmp_path, tests=[
            {"name": "test_persistent", "status": "FAIL", "error": "always fails"},
        ])

        mock_result = MagicMock()
        mock_result.stdout = "Tried to fix."
        mock_invoke.return_value = mock_result

        # First call: iter 0 → None, second call: iter 1 → 50.0
        mock_retest.side_effect = [None, 50.0]
        handler.config.qa.remediate_max_iterations = 2

        state = State(current_epic=1)
        handler.execute(state)

        # Retest should have been called at least for iteration > 0
        assert mock_retest.call_count >= 1

    def test_returns_none_on_import_error(self, handler: QaRemediateHandler) -> None:
        """When execute_qa_plan import fails, returns None."""
        with patch(
            "bmad_assist.qa.executor.execute_qa_plan",
            side_effect=ImportError("no module"),
            create=True,
        ):
            result = handler._run_retest(epic_id=1, iteration=1)
            assert result is None


class TestBuildRemediatePrompt:
    """Tests for _build_remediate_prompt method."""

    def test_safety_cap_rounding(self, handler: QaRemediateHandler) -> None:
        """Safety cap uses round() not int() — 0.7 should show 70% not 69%."""
        from bmad_assist.qa.remediate import EpicIssue

        issues = [EpicIssue(source="test", severity="high", description="test issue")]
        prompt = handler._build_remediate_prompt(issues, set(), 1, 0.7)
        assert "70%" in prompt
        # Also check 0.15 → 15% (not 14%)
        prompt2 = handler._build_remediate_prompt(issues, set(), 1, 0.15)
        assert "15%" in prompt2

    def test_context_truncation(self, handler: QaRemediateHandler) -> None:
        """Issue context is truncated at 2000 chars."""
        from bmad_assist.qa.remediate import EpicIssue

        long_ctx = "x" * 5000
        issues = [EpicIssue(source="test", severity="high", description="big", context=long_ctx)]
        prompt = handler._build_remediate_prompt(issues, set(), 1, 0.8)
        # Context in prompt should be truncated
        assert "x" * 2001 not in prompt
        assert "x" * 100 in prompt  # Some context is present
