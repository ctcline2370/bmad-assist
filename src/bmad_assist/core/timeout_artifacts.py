"""Provider timeout artifact persistence."""

import logging
import re
from pathlib import Path

from bmad_assist.core.exceptions import ProviderTimeoutError
from bmad_assist.core.io import atomic_write, get_timestamp

logger = logging.getLogger(__name__)


def _safe_token(value: object) -> str:
    """Return a filesystem-safe token for artifact names."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "unknown")).strip("-")
    return token or "unknown"


def save_provider_timeout_artifact(
    *,
    project_path: Path,
    phase_name: str,
    error: ProviderTimeoutError,
    epic: object = None,
    story: object = None,
    provider_id: object = None,
    attempt: int | None = None,
    will_retry: bool | None = None,
) -> Path | None:
    """Persist partial provider output from a timeout for later diagnosis."""
    try:
        artifact_dir = project_path / ".bmad-assist" / "provider-timeouts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        epic_token = _safe_token(epic)
        story_token = _safe_token(story)
        phase_token = _safe_token(phase_name)
        provider_token = _safe_token(provider_id)
        attempt_token = f"-attempt-{attempt}" if attempt is not None else ""
        artifact_path = (
            artifact_dir
            / (
                f"timeout-{epic_token}-{story_token}-{phase_token}-"
                f"{provider_token}{attempt_token}-{get_timestamp()}.md"
            )
        )

        partial = error.partial_result
        stdout = partial.stdout if partial is not None else ""
        stderr = partial.stderr if partial is not None else ""
        model = partial.model if partial is not None else "unknown"
        duration_ms = partial.duration_ms if partial is not None else "unknown"
        exit_code = partial.exit_code if partial is not None else "unknown"

        content = (
            "# Provider Timeout Partial Output\n\n"
            f"- Phase: {phase_name}\n"
            f"- Epic: {epic or 'unknown'}\n"
            f"- Story: {story or 'unknown'}\n"
            f"- ProviderId: {provider_id or 'unknown'}\n"
            f"- Model: {model}\n"
            f"- DurationMs: {duration_ms}\n"
            f"- ExitCode: {exit_code}\n"
            f"- Attempt: {attempt if attempt is not None else 'unknown'}\n"
            f"- WillRetry: {will_retry if will_retry is not None else 'unknown'}\n"
            f"- Error: {error}\n\n"
            "## Partial Stdout\n\n"
            "~~~text\n"
            f"{stdout}\n"
            "~~~\n\n"
            "## Partial Stderr\n\n"
            "~~~text\n"
            f"{stderr}\n"
            "~~~\n"
        )
        atomic_write(artifact_path, content)
        logger.warning("Saved provider timeout partial output: %s", artifact_path)
        return artifact_path
    except OSError as save_error:
        logger.warning("Failed to save provider timeout partial output: %s", save_error)
        return None
