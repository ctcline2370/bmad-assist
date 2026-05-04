"""Lock file management for concurrent run prevention.

Story 22.2: Run-scoped prompts directory initialization.
Story 22.3: PID validation for stale lock detection.
Extracted from runner.py as part of the runner refactoring.

"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.io import get_timestamp, init_run_prompts_dir
from bmad_assist.core.loop.pause import cleanup_stale_pause_flags
from bmad_assist.core.loop.run_tracking import reconcile_stale_running_runs

logger = logging.getLogger(__name__)

__all__ = ["_is_pid_alive", "_read_lock_file", "_running_lock"]


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Uses os.kill(pid, 0) which sends a null signal (doesn't kill the process,
    only checks existence). Returns True if process exists, False if PID not found.

    Story 22.3: PID validation for stale lock detection.

    Args:
        pid: Process ID to check.

    Returns:
        True if process is running, False if PID is not found (stale lock).

    """
    # Validate PID is positive - negative PIDs have special meaning in os.kill()
    # (e.g., -1 = all processes in caller's process group)
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
        return True
    except ProcessLookupError:
        # PID definitely not found - stale lock
        return False
    except PermissionError:
        # Process exists but belongs to another user - treat as alive
        # This prevents overwriting valid locks from other users
        return True
    except OSError:
        # Other OS errors (e.g., invalid PID) - treat as not found
        return False


def _read_lock_file(lock_path: Path) -> tuple[int | None, str | None]:
    """Read PID and timestamp from lock file.

    Story 22.3: Parse lock file for PID validation.

    Args:
        lock_path: Path to running.lock file.

    Returns:
        Tuple of (pid, timestamp) or (None, None) if file is invalid.

    """
    try:
        content = lock_path.read_text().strip().split("\n")
        if len(content) >= 2:
            pid = int(content[0].strip())
            timestamp = content[1].strip()
            return pid, timestamp
    except (ValueError, IndexError, OSError):
        pass
    return None, None


@contextmanager
def _running_lock(project_path: Path) -> Generator[Path, None, None]:
    """Context manager for .bmad-assist/running.lock file.

    Creates lock file with PID and timestamp on enter, removes on exit.
    Dashboard checks this file to detect if run is active.

    Story 22.2: Also initializes run-scoped prompts directory for organized
    prompt tracking during the run.

    Story 22.3: Implements PID validation for stale lock detection and
    concurrent run prevention.

    Args:
        project_path: Project root directory.

    Yields:
        Path to lock file.

    Raises:
        StateError: If another bmad-assist run is already active.

    """
    lock_dir = project_path / ".bmad-assist"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "running.lock"

    # Story 22.3: Check for existing lock file and validate PID
    if lock_path.exists():
        existing_pid, lock_timestamp = _read_lock_file(lock_path)
        if existing_pid is not None:
            if _is_pid_alive(existing_pid):
                # Active lock - abort to prevent concurrent runs
                raise StateError(
                    f"Another bmad-assist run is already active (PID {existing_pid}). "
                    f"If this is incorrect, remove the stale lock file: {lock_path}"
                )
            else:
                # Stale lock - remove and continue with warning
                logger.warning(
                    f"Removed stale lock file from dead process {existing_pid} "
                    f"(locked at {lock_timestamp})"
                )
                try:
                    lock_path.unlink()
                    reconcile_stale_running_runs(
                        project_path,
                        exit_reason="stale_lock_recovered_dead_pid",
                    )
                except OSError as e:
                    logger.warning(f"Failed to remove stale lock file: {e}")

    # A process may be killed after saving a RUNNING log but before the lock
    # finalizer runs, leaving no lock file for reset-lock to discover. Once this
    # process has passed the active-lock check, any pre-existing RUNNING logs are
    # stale and must be finalized before the new run creates its own log.
    reconcile_stale_running_runs(
        project_path,
        exit_reason="stale_run_recovered_missing_lock",
    )

    # Generate run timestamp for run-scoped prompts directory
    run_timestamp = get_timestamp()

    # Initialize run-scoped prompts directory (Story 22.2)
    init_run_prompts_dir(project_path, run_timestamp)

    # Story 22.10: Clean up stale pause flag on startup (AC #7)
    cleanup_stale_pause_flags(project_path)

    # Write lock file with PID and timestamp
    lock_content = f"{os.getpid()}\n{datetime.now(UTC).isoformat()}\n"
    lock_path.write_text(lock_content)

    try:
        yield lock_path
    finally:
        # Story 22.3: Always remove lock file on exit
        # Use contextlib.suppress for robustness if file was externally deleted
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
