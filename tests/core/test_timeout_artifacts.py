"""Regression tests for provider timeout artifact persistence."""

from pathlib import Path

from bmad_assist.core.exceptions import ProviderTimeoutError
from bmad_assist.core.timeout_artifacts import save_provider_timeout_artifact
from bmad_assist.providers.base import ProviderResult


def test_save_provider_timeout_artifact_persists_partial_output(tmp_path: Path) -> None:
    """Timeout artifacts include enough context to diagnose autonomous failures."""
    partial = ProviderResult(
        stdout="partial stdout",
        stderr="partial stderr",
        exit_code=-1,
        duration_ms=1234,
        model="gpt-5.5",
        command=("codex", "exec"),
    )
    error = ProviderTimeoutError("provider timed out", partial_result=partial)

    artifact_path = save_provider_timeout_artifact(
        project_path=tmp_path,
        phase_name="dev/story",
        error=error,
        epic="8",
        story="8.2",
        provider_id="codex/gpt-5.5",
        attempt=1,
        will_retry=True,
    )

    assert artifact_path is not None
    assert artifact_path.is_file()
    assert artifact_path.parent == tmp_path / ".bmad-assist" / "provider-timeouts"
    assert "/" not in artifact_path.name

    content = artifact_path.read_text(encoding="utf-8")
    assert "- Phase: dev/story" in content
    assert "- Epic: 8" in content
    assert "- Story: 8.2" in content
    assert "- ProviderId: codex/gpt-5.5" in content
    assert "- Attempt: 1" in content
    assert "- WillRetry: True" in content
    assert "partial stdout" in content
    assert "partial stderr" in content
