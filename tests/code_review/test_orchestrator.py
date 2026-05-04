"""Tests for code_review/orchestrator.py.

Story 13.10: Code Review Benchmarking Integration

Tests cover:
- Task 1: Module structure and public API (AC: #1)
- Task 2: Reviewer invocation with metrics (AC: #1, #6)
- Task 10: Unit tests for orchestrator (AC: #1, #2)
"""

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.benchmarking import (
    CollectorContext,
    DeterministicMetrics,
    LLMEvaluationRecord,
)
from bmad_assist.code_review.orchestrator import (
    CODE_REVIEW_SYNTHESIS_WORKFLOW_ID,
    CODE_REVIEW_WORKFLOW_ID,
    CodeReviewError,
    CodeReviewPhaseResult,
    InsufficientReviewsError,
    run_code_review_phase,
)
from bmad_assist.core.config import (
    BenchmarkingConfig,
    Config,
    MasterProviderConfig,
    MultiProviderConfig,
    ProviderConfig,
)
from bmad_assist.core.exceptions import ProviderTimeoutError
from bmad_assist.providers.base import ProviderResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config() -> Config:
    """Create mock Config with multi providers and master."""
    return Config(
        providers=ProviderConfig(
            master=MasterProviderConfig(provider="claude", model="opus"),
            multi=[
                MultiProviderConfig(provider="gemini", model="gemini-2.5-flash"),
                MultiProviderConfig(provider="codex", model="gpt-4o"),
            ],
        ),
        timeout=300,
        parallel_delay=0.0,
        benchmarking=BenchmarkingConfig(enabled=True),
        workflow_variant="default",
    )


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    """Create temp project with story file."""
    from bmad_assist.core.paths import init_paths

    # Initialize paths singleton for this test
    paths = init_paths(tmp_path)
    paths.ensure_directories()

    # Create story file in the new location
    story_file = paths.stories_dir / "13-10-code-review-benchmarking-integration.md"
    story_file.write_text("""# Story 13.10: Code Review Benchmarking Integration

## File List

- `src/bmad_assist/code_review/__init__.py`
- `src/bmad_assist/code_review/orchestrator.py`
- `tests/code_review/test_orchestrator.py`
""")

    return tmp_path


class TestCodeReviewTaskCollection:
    """Tests for bounded code review task collection."""

    def test_calculates_sweep_timeout_from_provider_timeout_and_retries(self) -> None:
        """Sweep timeout includes provider attempts, start delay, security timeout, and grace."""
        from bmad_assist.code_review.orchestrator import _calculate_code_review_sweep_timeout

        timeout = _calculate_code_review_sweep_timeout(
            timeout=120,
            timeout_retries=1,
            task_count=3,
            max_start_delay=5.0,
            security_timeout=400,
        )

        assert timeout == 460.0

    def test_sweep_timeout_preserves_completed_results_and_cancels_stragglers(self) -> None:
        """A stalled reviewer cannot discard already completed reviewer output."""
        from bmad_assist.code_review.orchestrator import _collect_code_review_task_results
        from bmad_assist.validation.anonymizer import ValidationOutput

        async def fast_reviewer():
            output = ValidationOutput(
                provider="fast",
                model="test-model",
                content="fast review",
                timestamp=datetime.now(UTC),
                duration_ms=10,
                token_count=3,
            )
            return "fast", output, None, None

        async def stuck_reviewer():
            await asyncio.sleep(60)
            return "stuck", None, None, "unexpected completion"

        async def run_collection():
            tasks = [
                asyncio.create_task(fast_reviewer()),
                asyncio.create_task(stuck_reviewer()),
            ]
            return await _collect_code_review_task_results(
                tasks,
                ["reviewer:fast", "reviewer:stuck"],
                timeout_seconds=0.01,
            )

        with patch("bmad_assist.code_review.orchestrator.kill_all_child_pgids") as kill_mock:
            results = asyncio.run(run_collection())

        assert any(isinstance(result, tuple) and result[0] == "fast" for result in results)
        assert any(
            isinstance(result, tuple)
            and result[0] == "stuck"
            and result[1] is None
            and "exceeded phase timeout" in str(result[3])
            for result in results
        )
        kill_mock.assert_called_once()

    def test_invoke_reviewer_passes_cancel_token_to_provider(self) -> None:
        """Provider calls receive a cancellation token for orchestrator teardown."""
        from bmad_assist.code_review.orchestrator import _invoke_reviewer

        provider = MagicMock()
        provider.provider_name = "mock"
        provider.invoke.return_value = ProviderResult(
            stdout="<!-- CODE_REVIEW_REPORT_START -->\n# Review\n<!-- CODE_REVIEW_REPORT_END -->",
            stderr="",
            exit_code=0,
            duration_ms=10,
            model="mock-model",
            command=("mock",),
        )

        asyncio.run(
            _invoke_reviewer(
                provider,
                "prompt",
                5,
                "reviewer-a",
                model="mock-model",
                timeout_retries=0,
            )
        )

        cancel_token = provider.invoke.call_args.kwargs["cancel_token"]
        assert isinstance(cancel_token, threading.Event)
        assert cancel_token.is_set() is False

    def test_invoke_reviewer_persists_timeout_artifact_before_retry(
        self,
        tmp_path: Path,
    ) -> None:
        """Retried reviewer timeouts persist partial output for diagnosis."""
        from bmad_assist.code_review.orchestrator import _invoke_reviewer

        provider = MagicMock()
        provider.provider_name = "mock"
        partial = ProviderResult(
            stdout="partial review stdout",
            stderr="partial review stderr",
            exit_code=-1,
            duration_ms=123,
            model="mock-model",
            command=("mock",),
        )
        provider.invoke.side_effect = [
            ProviderTimeoutError("reviewer timed out", partial_result=partial),
            ProviderResult(
                stdout=(
                    "<!-- CODE_REVIEW_REPORT_START -->\n"
                    "# Review\n"
                    "<!-- CODE_REVIEW_REPORT_END -->"
                ),
                stderr="",
                exit_code=0,
                duration_ms=10,
                model="mock-model",
                command=("mock",),
            ),
        ]

        reviewer_id, output, _deterministic, error = asyncio.run(
            _invoke_reviewer(
                provider,
                "prompt",
                5,
                "reviewer-a",
                model="mock-model",
                timeout_retries=1,
                epic_num=8,
                story_num=2,
                cwd=tmp_path,
            )
        )

        assert reviewer_id == "reviewer-a"
        assert output is not None
        assert error is None
        artifacts = list((tmp_path / ".bmad-assist" / "provider-timeouts").glob("*.md"))
        assert len(artifacts) == 1
        artifact = artifacts[0].read_text(encoding="utf-8")
        assert "- Phase: code_review" in artifact
        assert "- Epic: 8" in artifact
        assert "- Story: 2" in artifact
        assert "- ProviderId: reviewer-a" in artifact
        assert "- Attempt: 1" in artifact
        assert "- WillRetry: True" in artifact
        assert "partial review stdout" in artifact
        assert "partial review stderr" in artifact


# ============================================================================
# Task 1: Module Structure Tests (AC: #1)
# ============================================================================


class TestModuleExports:
    """Test module has correct exports."""

    def test_code_review_workflow_id_constant(self) -> None:
        """Test CODE_REVIEW_WORKFLOW_ID constant is defined."""
        assert CODE_REVIEW_WORKFLOW_ID == "code-review"

    def test_code_review_synthesis_workflow_id_constant(self) -> None:
        """Test CODE_REVIEW_SYNTHESIS_WORKFLOW_ID constant is defined."""
        assert CODE_REVIEW_SYNTHESIS_WORKFLOW_ID == "code-review-synthesis"

    def test_code_review_phase_result_dataclass(self) -> None:
        """Test CodeReviewPhaseResult is defined with required fields."""
        result = CodeReviewPhaseResult(
            anonymized_reviews=[],
            session_id="test-session",
            review_count=0,
            reviewers=[],
            failed_reviewers=[],
            evaluation_records=[],
        )
        assert result.session_id == "test-session"
        assert result.review_count == 0

    def test_code_review_error_exception(self) -> None:
        """Test CodeReviewError exception is defined."""
        error = CodeReviewError("test error")
        assert str(error) == "test error"

    def test_insufficient_reviews_error_exception(self) -> None:
        """Test InsufficientReviewsError exception is defined."""
        error = InsufficientReviewsError(count=1, minimum=2)
        assert error.count == 1
        assert error.minimum == 2


# ============================================================================
# Task 10: Unit Tests for Orchestrator (AC: #1, #2)
# ============================================================================


class TestRunCodeReviewPhaseWorkflowId:
    """Test run_code_review_phase creates records with correct workflow.id (AC: #2)."""

    def test_creates_records_with_code_review_workflow_id(
        self,
        mock_config: Config,
        project_path: Path,
    ) -> None:
        """Test that evaluation records have workflow.id = 'code-review'."""
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "<!-- CODE_REVIEW_REPORT_START -->\n# Review\n## Findings\nNo issues\n<!-- CODE_REVIEW_REPORT_END -->"
        mock_result.stderr = None
        mock_result.model = "test-model"
        mock_result.provider_session_id = "session-123"
        mock_result.termination_reason = None

        mock_provider = MagicMock()
        mock_provider.invoke.return_value = mock_result

        with (
            patch(
                "bmad_assist.code_review.orchestrator.get_provider",
                return_value=mock_provider,
            ),
            patch(
                "bmad_assist.code_review.orchestrator._compile_code_review_prompt",
                return_value="compiled prompt",
            ),
            patch(
                "bmad_assist.code_review.orchestrator.should_collect_benchmarking",
                return_value=True,
            ),
            patch(
                "bmad_assist.code_review.orchestrator.collect_deterministic_metrics",
                return_value=MagicMock(spec=DeterministicMetrics),
            ),
            patch(
                "bmad_assist.code_review.orchestrator._run_parallel_extraction",
                return_value=[MagicMock()],
            ),
            patch(
                "bmad_assist.code_review.orchestrator._finalize_evaluation_record",
                return_value=MagicMock(spec=LLMEvaluationRecord),
            ),
        ):
            result = asyncio.run(
                run_code_review_phase(
                    config=mock_config,
                    project_path=project_path,
                    epic_num=13,
                    story_num=10,
                )
            )

        # Verify result structure
        assert result.session_id
        assert result.review_count >= 2  # At least 2 reviewers required


class TestRunCodeReviewPhaseDeterministicMetrics:
    """Test deterministic metrics are collected per reviewer (AC: #1)."""

    def test_collects_deterministic_metrics(
        self,
        mock_config: Config,
        project_path: Path,
    ) -> None:
        """Test that deterministic metrics are collected for each reviewer."""
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "# Code Review\n## Findings\n- Issue 1\n- Issue 2"
        mock_result.stderr = None
        mock_result.model = "test-model"
        mock_result.provider_session_id = "session-123"
        mock_result.termination_reason = None

        mock_provider = MagicMock()
        mock_provider.invoke.return_value = mock_result

        collect_calls: list[tuple[str, CollectorContext]] = []

        def mock_collect(content: str, context: CollectorContext) -> DeterministicMetrics:
            collect_calls.append((content, context))
            return MagicMock(spec=DeterministicMetrics)

        with (
            patch(
                "bmad_assist.code_review.orchestrator.get_provider",
                return_value=mock_provider,
            ),
            patch(
                "bmad_assist.code_review.orchestrator._compile_code_review_prompt",
                return_value="compiled prompt",
            ),
            patch(
                "bmad_assist.code_review.orchestrator.should_collect_benchmarking",
                return_value=True,
            ),
            patch(
                "bmad_assist.code_review.orchestrator.collect_deterministic_metrics",
                side_effect=mock_collect,
            ),
            patch(
                "bmad_assist.code_review.orchestrator._run_parallel_extraction",
                return_value=[],
            ),
        ):
            result = asyncio.run(
                run_code_review_phase(
                    config=mock_config,
                    project_path=project_path,
                    epic_num=13,
                    story_num=10,
                )
            )

        # Each reviewer should have metrics collected
        # 2 multi + 1 master = 3 total reviewers
        assert len(collect_calls) == 3


class TestRunCodeReviewPhaseHandleFailures:
    """Test handling of failed reviewers (AC: #1)."""

    def test_handles_failed_reviewers_gracefully(
        self,
        mock_config: Config,
        project_path: Path,
    ) -> None:
        """Test that failed reviewers are tracked but don't block the phase."""
        call_count = 0

        def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                # First reviewer fails
                mock_result.exit_code = 1
                mock_result.stderr = "Error occurred"
                mock_result.stdout = ""
            else:
                # Others succeed
                mock_result.exit_code = 0
                mock_result.stdout = "# Review\n## Findings\nLooks good"
                mock_result.stderr = None
            mock_result.model = "test-model"
            mock_result.provider_session_id = f"session-{call_count}"
            mock_result.termination_reason = None
            return mock_result

        mock_provider = MagicMock()
        mock_provider.invoke.side_effect = mock_invoke

        with (
            patch(
                "bmad_assist.code_review.orchestrator.get_provider",
                return_value=mock_provider,
            ),
            patch(
                "bmad_assist.code_review.orchestrator._compile_code_review_prompt",
                return_value="compiled prompt",
            ),
            patch(
                "bmad_assist.code_review.orchestrator.should_collect_benchmarking",
                return_value=False,
            ),
        ):
            result = asyncio.run(
                run_code_review_phase(
                    config=mock_config,
                    project_path=project_path,
                    epic_num=13,
                    story_num=10,
                )
            )

        # Should have at least 2 successful (master + one multi)
        assert result.review_count >= 2
        assert len(result.failed_reviewers) == 1


class TestRunCodeReviewPhaseInsufficientReviewers:
    """Test insufficient reviewers raises error."""

    def test_raises_insufficient_reviews_error(
        self,
        project_path: Path,
    ) -> None:
        """Test InsufficientReviewsError when fewer than 2 reviewers succeed."""
        # Config with only 1 multi provider
        config = Config(
            providers=ProviderConfig(
                master=MasterProviderConfig(provider="claude", model="opus"),
                multi=[],  # No multi providers
            ),
            timeout=300,
            parallel_delay=0.0,
            benchmarking=BenchmarkingConfig(enabled=False),
        )

        def mock_invoke(*args, **kwargs):
            # All providers fail
            mock_result = MagicMock()
            mock_result.exit_code = 1
            mock_result.stderr = "Error"
            mock_result.stdout = ""
            mock_result.model = "test"
            mock_result.provider_session_id = "session"
            mock_result.termination_reason = None
            return mock_result

        mock_provider = MagicMock()
        mock_provider.invoke.side_effect = mock_invoke

        with (
            patch(
                "bmad_assist.code_review.orchestrator.get_provider",
                return_value=mock_provider,
            ),
            patch(
                "bmad_assist.code_review.orchestrator._compile_code_review_prompt",
                return_value="compiled prompt",
            ),
        ):
            with pytest.raises(InsufficientReviewsError) as exc_info:
                asyncio.run(
                    run_code_review_phase(
                        config=config,
                        project_path=project_path,
                        epic_num=13,
                        story_num=10,
                    )
                )

            assert exc_info.value.count < exc_info.value.minimum
