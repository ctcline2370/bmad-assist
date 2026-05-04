"""Provider timeout retry logic.

Shared retry wrapper for all provider invocations across single-LLM and
multi-LLM phases. Handles ProviderTimeoutError with configurable retry count.

Story: Per-phase timeout retry configuration.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from bmad_assist.core.exceptions import ProviderTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def invoke_with_timeout_retry(
    invoke_fn: Callable[..., T],
    *,
    timeout_retries: int | None,
    phase_name: str,
    fallback_invoke_fn: Callable[..., T] | None = None,
    fallback_timeout_retries: int | None = None,
    on_timeout: Callable[[ProviderTimeoutError, int, bool], None] | None = None,
    **kwargs: Any,
) -> T:
    """Invoke provider function with timeout retry logic.

    Retries provider invocation on ProviderTimeoutError, preserving all
    parameters (including prompt) across retry attempts. Timer is reset
    for each retry.

    Args:
        invoke_fn: Callable that invokes the primary provider.
        timeout_retries: Retry count from get_phase_retries().
            None = no retry (fail immediately on timeout).
            0 = infinite retry (until success).
            N = retry N times, then fail or fallback.
        phase_name: Phase name for logging (e.g., "dev_story", "validate_story").
        fallback_invoke_fn: Optional fallback callable (e.g., subprocess provider).
            If primary fails after retries, fallback is invoked with reset retry count.
        fallback_timeout_retries: Retry count for fallback (defaults to timeout_retries).
        on_timeout: Optional callback invoked on each timeout before retry/fallback/raise.
            Receives the timeout error, 1-based attempt number, and whether another
            provider invocation will be attempted.
        **kwargs: Arguments to pass to invoke_fn and fallback_invoke_fn.

    Returns:
        Result from invoke_fn or fallback_invoke_fn on success.

    Raises:
        ProviderTimeoutError: If all retry attempts exhausted (primary and fallback).
            - When timeout_retries is None: raised immediately.
            - When timeout_retries is N: raised after N+1 attempts (or fallback fails).
            - When timeout_retries is 0: never raised (infinite retry).

    Examples:
        >>> # Single-LLM phase with fallback
        >>> result = invoke_with_timeout_retry(
        ...     provider.invoke,
        ...     timeout_retries=3,
        ...     phase_name="dev_story",
        ...     fallback_invoke_fn=subprocess_provider.invoke,
        ...     fallback_timeout_retries=3,
        ...     prompt=prompt,
        ...     model=model,
        ...     timeout=timeout,
        ... )

    """
    # Check if retry is configured
    if timeout_retries is None:
        # No retry - invoke once and let timeout propagate
        return invoke_fn(**kwargs)

    # Retry is configured. A positive retry count means "retries after the
    # initial attempt", so total attempts are retry_count + 1.
    timeout_attempt = 0
    max_attempts = None if timeout_retries == 0 else timeout_retries + 1

    while True:
        timeout_attempt += 1
        if timeout_attempt > 1:
            if max_attempts is None:
                attempt_status = f"attempt {timeout_attempt} (infinite retry budget)"
            else:
                attempt_status = f"attempt {timeout_attempt}/{max_attempts}"
            logger.info(
                "Starting provider timeout retry for %s phase (%s)",
                phase_name,
                attempt_status,
            )

        try:
            return invoke_fn(**kwargs)
        except ProviderTimeoutError as e:
            # Check retry limit
            if max_attempts is not None and timeout_attempt >= max_attempts:
                # Primary retries exhausted - try fallback if available
                if fallback_invoke_fn is not None:
                    if on_timeout is not None:
                        on_timeout(e, timeout_attempt, True)
                    logger.warning(
                        "Primary provider timeout in %s phase after %d attempts, "
                        "starting secondary provider...",
                        phase_name,
                        timeout_attempt,
                    )
                    return invoke_with_timeout_retry(
                        fallback_invoke_fn,
                        timeout_retries=fallback_timeout_retries if fallback_timeout_retries is not None else timeout_retries,
                        phase_name=f"{phase_name}(fallback)",
                        fallback_invoke_fn=None,  # No second fallback
                        on_timeout=on_timeout,
                        **kwargs,
                    )

                # No fallback - raise error
                if on_timeout is not None:
                    on_timeout(e, timeout_attempt, False)
                logger.error(
                    "Provider timeout in %s phase after %d attempts (%d retries configured): %s",
                    phase_name,
                    timeout_attempt,
                    timeout_retries,
                    str(e)[:200],
                )
                raise

            # Retry - preserve kwargs (including prompt), reset timer
            if timeout_retries == 0:
                retry_status = "infinite retries"
            else:
                retry_number = timeout_attempt
                remaining_after_next = max(timeout_retries - retry_number, 0)
                retry_status = (
                    f"retry {retry_number}/{timeout_retries}; "
                    f"{remaining_after_next} remaining after next attempt"
                )
            if on_timeout is not None:
                on_timeout(e, timeout_attempt, True)
            logger.warning(
                "Provider timeout in %s phase (attempt %d, %s): %s. Retrying...",
                phase_name,
                timeout_attempt,
                retry_status,
                str(e)[:100],
            )
            # No delay for timeout retry - restart immediately
            continue
