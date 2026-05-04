"""Regression tests for BaseHandler provider retry behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bmad_assist.core.config import Config
from bmad_assist.core.exceptions import ProviderExitCodeError, ProviderTimeoutError
from bmad_assist.core.loop.handlers.base import BaseHandler
from bmad_assist.core.state import Phase, State
from bmad_assist.providers.base import ExitStatus, ProviderResult


class ConcreteHandler(BaseHandler):
    """Small concrete handler for retry tests."""

    def __init__(self, config: Config, project_path: Path, provider: MagicMock) -> None:
        """Initialize the concrete retry-test handler."""
        super().__init__(config, project_path)
        self._provider = provider

    @property
    def phase_name(self) -> str:
        """Return a single-LLM phase name for config lookup."""
        return "retrospective"

    def build_context(self, state: State) -> dict[str, object]:
        """Return an empty template context; invoke_provider tests do not render."""
        return {}

    def get_provider(self) -> MagicMock:
        """Return the injected provider mock."""
        return self._provider


def _make_config() -> Config:
    return Config(
        providers={
            "master": {
                "provider": "codex",
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
            },
        }
    )


def _make_retry_config() -> Config:
    return Config(
        providers={
            "master": {
                "provider": "codex",
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
            },
        },
        timeouts={
            "default": 60,
            "retries": 1,
        },
    )


def _exit_error(stderr: str) -> ProviderExitCodeError:
    return ProviderExitCodeError(
        message=f"Provider failed: {stderr}",
        exit_code=1,
        exit_status=ExitStatus.ERROR,
        stderr=stderr,
        command=("codex", "exec"),
    )


def test_non_transient_exit_code_fails_fast(tmp_path: Path) -> None:
    """Non-transient CLI startup failures are raised without retry delay."""
    provider = MagicMock()
    provider.provider_name = "codex"
    provider.invoke.side_effect = _exit_error("Operation not permitted")
    handler = ConcreteHandler(_make_config(), tmp_path, provider)

    with (
        patch("bmad_assist.core.loop.handlers.base.time.time", side_effect=[0.0, 0.0]),
        patch("bmad_assist.core.loop.handlers.base.time.sleep") as sleep_mock,
        pytest.raises(ProviderExitCodeError),
    ):
        handler.invoke_provider("prompt", retry_timeout_minutes=1, retry_delay=1)

    assert provider.invoke.call_count == 1
    sleep_mock.assert_not_called()


def test_transient_exit_code_still_retries(tmp_path: Path) -> None:
    """Transient provider exit failures still use the outer retry loop."""
    provider = MagicMock()
    provider.provider_name = "codex"
    expected = ProviderResult(
        stdout="ok",
        stderr="",
        exit_code=0,
        duration_ms=10,
        model="gpt-5.5",
        command=("codex", "exec"),
    )
    provider.invoke.side_effect = [_exit_error("rate limit exceeded"), expected]
    handler = ConcreteHandler(_make_config(), tmp_path, provider)

    with (
        patch("bmad_assist.core.loop.handlers.base.time.time", side_effect=[0.0, 0.0, 1.0]),
        patch("bmad_assist.core.loop.handlers.base.time.sleep") as sleep_mock,
    ):
        result = handler.invoke_provider("prompt", retry_timeout_minutes=1, retry_delay=1)

    assert result is expected
    assert provider.invoke.call_count == 2
    sleep_mock.assert_called_once_with(1)


def test_timeout_retry_persists_partial_output_before_success(tmp_path: Path) -> None:
    """Retried timeout attempts leave diagnosis artifacts even when retry succeeds."""
    provider = MagicMock()
    provider.provider_name = "codex"
    partial = ProviderResult(
        stdout="partial stdout",
        stderr="partial stderr",
        exit_code=-1,
        duration_ms=123,
        model="gpt-5.5",
        command=("codex", "exec"),
    )
    expected = ProviderResult(
        stdout="ok",
        stderr="",
        exit_code=0,
        duration_ms=10,
        model="gpt-5.5",
        command=("codex", "exec"),
    )
    provider.invoke.side_effect = [
        ProviderTimeoutError("timeout message", partial_result=partial),
        expected,
    ]
    handler = ConcreteHandler(_make_retry_config(), tmp_path, provider)
    state = State(
        current_epic=8,
        current_story="8.2",
        current_phase=Phase.DEV_STORY,
    )

    result = handler.invoke_provider("prompt", state=state)

    assert result is expected
    assert provider.invoke.call_count == 2
    artifacts = list((tmp_path / ".bmad-assist" / "provider-timeouts").glob("*.md"))
    assert len(artifacts) == 1
    artifact = artifacts[0].read_text(encoding="utf-8")
    assert "partial stdout" in artifact
    assert "partial stderr" in artifact
    assert "- Attempt: 1" in artifact
    assert "- WillRetry: True" in artifact


def test_execute_persists_partial_output_on_provider_timeout(tmp_path: Path) -> None:
    """Timeout failures preserve partial output as an artifact for autonomous diagnosis."""
    provider = MagicMock()
    provider.provider_name = "codex"
    partial = ProviderResult(
        stdout="partial stdout",
        stderr="partial stderr",
        exit_code=-1,
        duration_ms=123,
        model="gpt-5.5",
        command=("codex", "exec"),
    )
    provider.invoke.side_effect = ProviderTimeoutError(
        "timeout message",
        partial_result=partial,
    )
    handler = ConcreteHandler(_make_config(), tmp_path, provider)
    state = State(
        current_epic=8,
        current_story="8.2",
        current_phase=Phase.DEV_STORY,
    )

    with patch.object(handler, "render_prompt", return_value="prompt"):
        result = handler.execute(state)

    assert not result.success
    assert result.error
    assert "Provider timeout: timeout message" in result.error
    assert "timeout_artifact" in result.outputs
    artifact_path = Path(result.outputs["timeout_artifact"])
    assert artifact_path.is_file()
    artifact = artifact_path.read_text(encoding="utf-8")
    assert "partial stdout" in artifact
    assert "partial stderr" in artifact
    assert "timeout message" in artifact
    assert "- Epic: 8" in artifact
    assert "- Story: 8.2" in artifact
