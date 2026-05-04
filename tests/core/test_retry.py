"""Regression tests for provider timeout retry semantics."""

import logging

import pytest

from bmad_assist.core.exceptions import ProviderTimeoutError
from bmad_assist.core.retry import invoke_with_timeout_retry


def test_timeout_retries_none_invokes_once_and_raises() -> None:
    """None disables timeout retries."""
    calls = 0

    def invoke() -> str:
        nonlocal calls
        calls += 1
        raise ProviderTimeoutError("timeout")

    with pytest.raises(ProviderTimeoutError):
        invoke_with_timeout_retry(
            invoke,
            timeout_retries=None,
            phase_name="dev_story",
        )

    assert calls == 1


def test_positive_timeout_retries_are_retries_after_initial_attempt(caplog: pytest.LogCaptureFixture) -> None:
    """A positive retry count allows that many attempts after the first timeout."""
    calls = 0

    def invoke() -> str:
        nonlocal calls
        calls += 1
        raise ProviderTimeoutError("timeout")

    with caplog.at_level(logging.INFO), pytest.raises(ProviderTimeoutError):
        invoke_with_timeout_retry(
            invoke,
            timeout_retries=1,
            phase_name="dev_story",
        )

    assert calls == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("retry 1/1" in message for message in messages)
    assert any("attempt 2/2" in message for message in messages)
    assert all("0 remaining):" not in message for message in messages)


def test_on_timeout_callback_receives_retry_and_terminal_attempts() -> None:
    """Timeout callbacks run for both retried and terminal timeout attempts."""
    calls = 0
    timeout_events: list[tuple[int, bool, str]] = []

    def invoke() -> str:
        nonlocal calls
        calls += 1
        raise ProviderTimeoutError(f"timeout {calls}")

    def on_timeout(
        error: ProviderTimeoutError,
        attempt: int,
        will_retry: bool,
    ) -> None:
        timeout_events.append((attempt, will_retry, str(error)))

    with pytest.raises(ProviderTimeoutError):
        invoke_with_timeout_retry(
            invoke,
            timeout_retries=1,
            phase_name="dev_story",
            on_timeout=on_timeout,
        )

    assert calls == 2
    assert timeout_events == [
        (1, True, "timeout 1"),
        (2, False, "timeout 2"),
    ]


def test_fallback_starts_after_primary_retries_are_exhausted() -> None:
    """Fallback uses its own retry budget after primary timeout attempts fail."""
    primary_calls = 0
    fallback_calls = 0

    def primary() -> str:
        nonlocal primary_calls
        primary_calls += 1
        raise ProviderTimeoutError("primary timeout")

    def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "ok"

    result = invoke_with_timeout_retry(
        primary,
        timeout_retries=1,
        phase_name="dev_story",
        fallback_invoke_fn=fallback,
    )

    assert result == "ok"
    assert primary_calls == 2
    assert fallback_calls == 1


def test_timeout_retries_zero_retries_until_success() -> None:
    """Zero remains the explicit infinite-retry sentinel."""
    calls = 0

    def invoke() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderTimeoutError("timeout")
        return "ok"

    result = invoke_with_timeout_retry(
        invoke,
        timeout_retries=0,
        phase_name="dev_story",
    )

    assert result == "ok"
    assert calls == 3
