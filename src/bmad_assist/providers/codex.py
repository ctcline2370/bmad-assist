"""Codex CLI subprocess-based provider implementation.

This module implements the CodexProvider class that adapts Codex CLI
for use within bmad-assist via subprocess invocation. Codex serves as
a Multi LLM validator for story validation and code review phases.

The prompt is passed via stdin to avoid ARG_MAX / MAX_ARG_STRLEN limits
on POSIX systems. Large compiled BMAD prompts (>128KB) exceed the Linux
per-argument limit and cannot be passed as positional arguments.

⚠️ SECURITY WARNING: When CodexProvider is used as a Multi-LLM validator,
the orchestrator MUST ensure read-only behavior. The provider defaults to
Codex CLI's workspace-write sandbox for implementation phases and switches
to read-only when callers pass allowed tools without an explicit sandbox mode.

JSON Streaming:
    Uses --json flag to capture JSONL event stream for debugging.
    Event types: thread.started, turn.started, turn.completed, item.*, error
    Text extracted from item.completed events with item.type="agent_message".

Example:
    >>> from bmad_assist.providers import CodexProvider
    >>> provider = CodexProvider()
    >>> result = provider.invoke("Review this code", model="o3-mini")
    >>> response = provider.parse_output(result)

"""

import contextlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired, run
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bmad_assist.providers.tool_guard import ToolCallGuard

from bmad_assist.core.debug_logger import DebugJsonLogger
from bmad_assist.core.exceptions import (
    ProviderError,
    ProviderExitCodeError,
    ProviderTimeoutError,
)
from bmad_assist.providers.base import (
    BaseProvider,
    ExitStatus,
    ProviderResult,
    format_tag,
    is_full_stream,
    register_child_pgid,
    should_print_progress,
    unregister_child_pgid,
    validate_settings_file,
    write_progress,
)

logger = logging.getLogger(__name__)

# Note: Model validation removed - Codex CLI accepts any model string.
# The CLI itself will validate and return an error for unknown models.

# Default timeout in seconds (5 minutes)
DEFAULT_TIMEOUT: int = 300

# Maximum provider diagnostic length in error messages before truncation.
#
# Codex CLI can write plugin warnings to stderr while fatal Responses API
# failures arrive later or on stdout JSON events. Keep enough head and tail
# evidence to preserve the real root cause for autonomous run diagnostics.
STDERR_TRUNCATE_LENGTH: int = 4000

# Long-running Codex turns can be silent for many minutes. Emit metadata-only
# progress so autonomous runs can see liveness before the hard timeout path.
IDLE_PROGRESS_MAX_INTERVAL_SECONDS: float = 60.0
IDLE_PROGRESS_MIN_INTERVAL_SECONDS: float = 5.0
PROCESS_SNAPSHOT_MAX_LINES: int = 80


def _idle_progress_interval(timeout_seconds: int) -> float:
    """Return a bounded interval for provider silence warnings."""
    return max(
        IDLE_PROGRESS_MIN_INTERVAL_SECONDS,
        min(IDLE_PROGRESS_MAX_INTERVAL_SECONDS, timeout_seconds / 4),
    )


def _format_timeout_message(
    timeout_seconds: int,
    prompt_len: int,
    partial_result: ProviderResult,
    thread_id: str | None,
) -> str:
    """Build a timeout message without leaking prompt contents."""
    details = [
        f"prompt_len={prompt_len}",
        f"stdout_chars={len(partial_result.stdout)}",
        f"stderr_chars={len(partial_result.stderr)}",
    ]
    if thread_id:
        details.append(f"thread_id={thread_id}")
    return f"Codex CLI timeout after {timeout_seconds}s ({', '.join(details)})"


def _safe_process_id(value: object) -> int | None:
    """Return a positive process id, excluding mocks and invalid values."""
    return value if isinstance(value, int) and value > 0 else None


def _parse_ps_row(line: str) -> tuple[int, int, int, str] | None:
    """Parse a ps row into pid, ppid, pgid, and the original line."""
    parts = line.strip().split(None, 5)
    if len(parts) < 5:
        return None

    try:
        pid = int(parts[0])
        ppid = int(parts[1])
        pgid = int(parts[2])
    except ValueError:
        return None

    return pid, ppid, pgid, line.rstrip()


def _collect_process_snapshot(process_id: int | None, process_group_id: int | None) -> str:
    """Collect a bounded process snapshot for timeout diagnostics."""
    if process_id is None and process_group_id is None:
        return "Process snapshot unavailable: no process id or process group id."

    try:
        snapshot = run(
            ["ps", "-axo", "pid=,ppid=,pgid=,stat=,etime=,command="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, TimeoutExpired) as error:
        return f"Process snapshot unavailable: ps failed with {type(error).__name__}: {error}"

    rows: list[tuple[int, int, int, str]] = []
    children_by_parent: dict[int, list[int]] = {}
    row_by_pid: dict[int, tuple[int, int, int, str]] = {}
    for line in snapshot.stdout.splitlines():
        parsed = _parse_ps_row(line)
        if parsed is None:
            continue
        pid, ppid, _pgid, _raw_line = parsed
        rows.append(parsed)
        row_by_pid[pid] = parsed
        children_by_parent.setdefault(ppid, []).append(pid)

    selected: set[int] = set()
    if process_id is not None:
        selected.add(process_id)
    if process_group_id is not None:
        selected.update(pid for pid, _ppid, pgid, _line in rows if pgid == process_group_id)

    pending = list(selected)
    while pending:
        parent = pending.pop()
        for child in children_by_parent.get(parent, []):
            if child not in selected:
                selected.add(child)
                pending.append(child)

    if not selected:
        return (
            "Process snapshot captured, but no matching processes were found "
            f"(pid={process_id or '-'}, pgid={process_group_id or '-'})."
        )

    lines = ["PID PPID PGID STAT ELAPSED COMMAND"]
    for pid in sorted(selected):
        row = row_by_pid.get(pid)
        if row is not None:
            lines.append(row[3])
        if len(lines) >= PROCESS_SNAPSHOT_MAX_LINES:
            omitted = len(selected) - (PROCESS_SNAPSHOT_MAX_LINES - 1)
            if omitted > 0:
                lines.append(f"... omitted {omitted} matching process rows ...")
            break

    return "\n".join(lines)


def _head_tail_excerpt(text: str, limit: int = STDERR_TRUNCATE_LENGTH) -> str:
    """Return a bounded head-plus-tail excerpt for provider diagnostics."""
    if not text:
        return "(empty)"
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""

    head_len = max(limit // 2, 1)
    tail_len = max(limit - head_len, 0)
    omitted = len(text) - limit
    tail = text[-tail_len:] if tail_len else ""
    return f"{text[:head_len]}\n... [truncated {omitted} chars] ...\n{tail}"


def _extract_message_text(value: Any) -> str | None:
    """Extract a human-readable message from a Codex JSON value."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if not isinstance(value, dict):
        return None

    for key in ("message", "error", "details", "reason"):
        nested = _extract_message_text(value.get(key))
        if nested:
            return nested
    return None


def _collect_codex_stream_error_messages(raw_stdout_lines: list[str]) -> list[str]:
    """Collect error messages emitted by Codex --json on stdout."""
    messages: list[str] = []
    seen: set[str] = set()

    for line in raw_stdout_lines:
        stripped = line.strip()
        if not stripped:
            continue

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        event_type = str(event.get("type", ""))
        is_error_event = "error" in event_type or "failed" in event_type
        if not is_error_event and "error" not in event:
            continue

        message = _extract_message_text(event)
        if message and message not in seen:
            messages.append(message)
            seen.add(message)

    return messages


_AUTH_FAILURE_MARKERS: tuple[str, ...] = (
    "401 unauthorized",
    "missing bearer or basic authentication",
    "authentication required",
    "not authenticated",
    "api key",
    "bearer",
    "basic authentication",
)


def _find_codex_auth_failure_message(messages: list[str]) -> str | None:
    """Return the first message that looks like a Codex auth failure."""
    for message in messages:
        lower = message.lower()
        if any(marker in lower for marker in _AUTH_FAILURE_MARKERS):
            return message
    return None


def _format_codex_failure_diagnostic(
    stderr_content: str,
    raw_stdout_lines: list[str],
    *,
    limit: int = STDERR_TRUNCATE_LENGTH,
) -> str:
    """Build a concise non-zero-exit diagnostic from stderr and JSON events."""
    stdout_error_messages = _collect_codex_stream_error_messages(raw_stdout_lines)
    diagnostic_parts: list[str] = []

    auth_message = _find_codex_auth_failure_message(
        [stderr_content, *stdout_error_messages]
    )
    if auth_message:
        diagnostic_parts.append(
            "Codex CLI authentication failed: "
            f"{_head_tail_excerpt(auth_message, min(limit, 1000))}"
        )

    if stdout_error_messages:
        stdout_errors = "\n".join(f"- {message}" for message in stdout_error_messages)
        diagnostic_parts.append(
            "stdout error events:\n"
            f"{_head_tail_excerpt(stdout_errors, limit)}"
        )

    if stderr_content:
        diagnostic_parts.append(
            "stderr excerpt:\n"
            f"{_head_tail_excerpt(stderr_content, limit)}"
        )

    if not diagnostic_parts:
        return "(empty)"
    return "\n\n".join(diagnostic_parts)


class CodexProvider(BaseProvider):
    """Codex CLI subprocess-based provider implementation.

    Adapts Codex CLI for use within bmad-assist via subprocess invocation.
    Codex serves as a Multi LLM validator for parallel validation phases.

    Supported models (from Codex CLI, December 2025):
        ChatGPT subscription compatible:
        - gpt-5.1-codex-max: Optimized for agentic coding (default)
        - gpt-5.1-codex, gpt-5.1-codex-mini: Codex variants
        - gpt-5-codex, gpt-5, gpt-5-mini, gpt-5-nano: GPT-5 family
        - gpt-5.2: Latest general-purpose model

        API key required:
        - o3, o3-mini, o4-mini: Reasoning models
        - gpt-4.1, gpt-4.1-mini, gpt-4.1-nano: GPT-4.1 variants
        - gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4: Legacy

    Settings File Handling:
        The settings_file parameter is accepted for API consistency with other
        providers but is NOT passed to Codex CLI, which uses environment
        variables (OPENAI_API_KEY, CODEX_API_KEY) and ~/.codex/ config files
        rather than CLI flags. When provided, the file is validated for
        existence (logging a warning if missing) but does not affect CLI
        execution.

    Example:
        >>> provider = CodexProvider()
        >>> result = provider.invoke("Review this code", timeout=60)
        >>> print(provider.parse_output(result))

    """

    @property
    def provider_name(self) -> str:
        """Return unique identifier for this provider.

        Returns:
            The string "codex" as the provider identifier.

        """
        return "codex"

    @property
    def default_model(self) -> str | None:
        """Return default model when none specified.

        Returns:
            The string "gpt-5.1-codex-max" as the default for ChatGPT users.

        """
        return "gpt-5.1-codex-max"

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.

        Args:
            model: Model identifier to check.

        Returns:
            Always True - let Codex CLI validate model names.

        Example:
            >>> provider = CodexProvider()
            >>> provider.supports_model("gpt-5.1-codex-max")
            True
            >>> provider.supports_model("gpt-5.2")
            True
            >>> provider.supports_model("any-model")
            True

        """
        # Always return True - let Codex CLI validate model names
        return True

    def _resolve_settings(
        self,
        settings_file: Path | None,
        model: str,
    ) -> Path | None:
        """Resolve and validate settings file for invocation.

        Internal helper that validates settings file existence and logs
        a warning if missing. Called after model validation, before
        command building.

        Args:
            settings_file: Settings file path from caller, or None.
            model: Model identifier for logging context.

        Returns:
            Validated settings file Path if exists and is a file,
            None otherwise (triggers graceful degradation to defaults).

        """
        if settings_file is None:
            return None

        return validate_settings_file(
            settings_file=settings_file,
            provider_name=self.provider_name,
            model=model,
        )

    def invoke(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout: int | None = None,
        settings_file: Path | None = None,
        cwd: Path | None = None,
        disable_tools: bool = False,
        allowed_tools: list[str] | None = None,
        sandbox_mode: str | None = None,
        no_cache: bool = False,
        color_index: int | None = None,
        display_model: str | None = None,
        thinking: bool | None = None,
        cancel_token: threading.Event | None = None,
        reasoning_effort: str | None = None,
        guard: "ToolCallGuard | None" = None,
    ) -> ProviderResult:
        """Execute Codex CLI with the given prompt using JSON streaming.

        Invokes Codex CLI via Popen with --json flag for JSONL event streaming.
        This enables:
        - Debug logging of raw JSON events to ~/.bmad-assist/debug/json/
        - Real-time output processing
        - Consistent debugging across all providers

        Command Format:
            codex exec "<prompt>" --json --sandbox workspace-write -m <model>
                (normal mode)
            codex exec "<prompt>" --json --sandbox read-only -m <model>  (validator mode)
            codex exec "<prompt>" --json --sandbox workspace-write -m <model>
                (explicit writable sandbox)

        JSON Event Types:
            - thread.started: Session initialization with thread_id
            - turn.started/turn.completed: Turn lifecycle with usage stats
            - item.started/item.completed: Individual items (messages, tools)
            - error: Error events

        Text Extraction:
            Response text is extracted from item.completed events where
            item.type === "agent_message" from the item.text field.

        Args:
            prompt: The prompt text to send to Codex.
            model: Model to use (gpt-5.1-codex-max, o3-mini, etc).
                If None, uses default_model.
            timeout: Timeout in seconds. Must be positive (>= 1) if specified.
                If None, uses DEFAULT_TIMEOUT (300s).
            settings_file: Path to settings file (validated but not used by CLI).
            cwd: Working directory (ignored - Codex CLI doesn't support).
            disable_tools: Disable tools (ignored - Codex CLI doesn't support).
            allowed_tools: List of allowed tools. When set, defaults to
                --sandbox read-only unless sandbox_mode is explicitly provided.
            sandbox_mode: Explicit Codex sandbox mode. When set, overrides the
                implicit read-only sandbox used for allowed_tools so callers
                can opt into workspace-write when edits are required inside a
                guarded tool set.
            no_cache: Disable caching (ignored - Codex CLI doesn't support).
            color_index: Color index for terminal output differentiation.
            reasoning_effort: Reasoning effort level (minimal/low/medium/high/xhigh).
                Passed to Codex CLI as -c model_reasoning_effort="VALUE".

        Returns:
            ProviderResult containing extracted text, stderr, exit code, and timing.

        Raises:
            ValueError: If timeout is not positive (<=0).
            ProviderError: If CLI execution fails.
            ProviderExitCodeError: If CLI returns non-zero exit code.
            ProviderTimeoutError: If CLI execution exceeds timeout.

        """
        # Ignored parameters (Codex CLI doesn't support these flags)
        _ = disable_tools, no_cache

        # cwd IS used - passed to Popen to set working directory
        # This ensures file access is relative to the target project, not bmad-assist

        # Validate timeout parameter
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        # Resolve model with fallback chain: explicit -> default -> literal
        effective_model = model or self.default_model or "gpt-5.1-codex-max"
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        # Validate and resolve settings file
        validated_settings = self._resolve_settings(settings_file, effective_model)

        # Codex CLI uses sandbox modes for tool restrictions. Explicit sandbox
        # selection wins so callers can request workspace-write for synthesis.
        effective_sandbox = sandbox_mode
        if effective_sandbox is None and allowed_tools:
            effective_sandbox = "read-only"
            logger.debug(
                "Codex CLI: using --sandbox read-only for tool restriction (requested: %s)",
                allowed_tools,
            )
        elif effective_sandbox is None:
            effective_sandbox = "workspace-write"
        elif effective_sandbox is not None:
            logger.debug(
                "Codex CLI: using explicit sandbox mode %s (requested tools: %s)",
                effective_sandbox,
                allowed_tools,
            )

        logger.debug(
            "Invoking Codex CLI: model=%s, timeout=%ds, prompt_len=%d, settings=%s, sandbox=%s",
            effective_model,
            effective_timeout,
            len(prompt),
            validated_settings,
            effective_sandbox,
        )

        # Validate reasoning_effort if provided
        valid_reasoning = {"minimal", "low", "medium", "high", "xhigh"}
        if reasoning_effort is not None and reasoning_effort not in valid_reasoning:
            logger.warning(
                "Invalid reasoning_effort '%s', ignoring (valid: %s)",
                reasoning_effort,
                ", ".join(sorted(valid_reasoning)),
            )
            reasoning_effort = None

        # Build command with --json for JSONL streaming
        # Note: prompt passed via stdin to avoid "Argument list too long" error
        # (Linux MAX_ARG_STRLEN is 128KB per argument, compiled prompts exceed this)
        command: list[str] = ["codex", "exec", "--json"]

        command.extend(["--sandbox", effective_sandbox])

        command.extend(["-m", effective_model])

        # Add reasoning effort config override if specified
        if reasoning_effort is not None:
            command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])

        # Persist the actual executed argv. The prompt is written via stdin and
        # must never be reintroduced into ProviderResult metadata.
        original_command: tuple[str, ...] = tuple(command)

        if validated_settings is not None:
            logger.debug(
                "Settings file validated but not passed to Codex CLI: %s",
                validated_settings,
            )

        # Debug JSON logger for raw event stream
        debug_json_logger = DebugJsonLogger()

        # Accumulators for JSON stream parsing
        response_text_parts: list[str] = []
        stderr_chunks: list[str] = []
        raw_stdout_lines: list[str] = []
        thread_id: str | None = None

        start_time = time.perf_counter()
        wall_start_time = time.time()
        child_pgid: int | None = None
        child_env = os.environ.copy()
        if cwd is not None and "CODEX_HOME" not in child_env:
            repo_codex_home = Path(cwd) / ".codex"
            if (repo_codex_home / "auth.json").is_file():
                child_env["CODEX_HOME"] = str(repo_codex_home)

        try:
            process = Popen(
                command,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,  # Use target project directory, not bmad-assist cwd
                env=child_env,
                start_new_session=True,  # Own process group for safe termination
            )
            try:
                child_pgid = os.getpgid(process.pid)
                register_child_pgid(child_pgid)
            except (ProcessLookupError, PermissionError, OSError, TypeError):
                logger.debug("Unable to register Codex child process group", exc_info=True)

            def process_json_stream(
                stream: Any,
                text_parts: list[str],
                raw_lines: list[str],
                json_logger: DebugJsonLogger,
                color_idx: int | None,
            ) -> None:
                """Process Codex --json output, extracting text and logging events."""
                nonlocal thread_id
                for line in iter(stream.readline, ""):
                    raw_lines.append(line)
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Log raw JSON immediately (survives crashes)
                    json_logger.append(stripped)

                    try:
                        msg = json.loads(stripped)
                        msg_type = msg.get("type", "")

                        if msg_type == "thread.started":
                            thread_id = msg.get("thread_id", "?")
                            if should_print_progress():
                                tag = format_tag("INIT", color_idx)
                                write_progress(f"{tag} Thread: {thread_id}")

                        elif msg_type == "item.completed":
                            item = msg.get("item", {})
                            item_type = item.get("type", "")
                            if item_type == "agent_message":
                                text = item.get("text", "")
                                if text:
                                    text_parts.append(text)
                                    if should_print_progress():
                                        tag = format_tag("MESSAGE", color_idx)
                                        if is_full_stream():
                                            write_progress(f"{tag} {text}")
                                        else:
                                            preview = text[:200]
                                            if len(text) > 200:
                                                preview += "..."
                                            write_progress(f"{tag} {preview}")
                            elif item_type == "command_execution":
                                if should_print_progress():
                                    cmd = item.get("command", "?")
                                    tag = format_tag("CMD", color_idx)
                                    if is_full_stream():
                                        write_progress(f"{tag} {cmd}")
                                    else:
                                        cmd_preview = cmd[:60]
                                        if len(cmd) > 60:
                                            cmd_preview += "..."
                                        write_progress(f"{tag} {cmd_preview}")

                        elif msg_type == "turn.completed":
                            if should_print_progress():
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)
                                tag = format_tag("TURN", color_idx)
                                write_progress(f"{tag} in={input_tokens} out={output_tokens}")

                        elif msg_type == "error":
                            if should_print_progress():
                                error_msg = msg.get("message", str(msg))
                                tag = format_tag("ERROR", color_idx)
                                write_progress(f"{tag} {error_msg}")

                    except json.JSONDecodeError:
                        if should_print_progress():
                            tag = format_tag("RAW", color_idx)
                            write_progress(f"{tag} {stripped}")

                stream.close()

            def read_stderr(
                stream: Any,
                chunks: list[str],
                color_idx: int | None,
            ) -> None:
                """Read stderr stream."""
                for line in iter(stream.readline, ""):
                    chunks.append(line)
                    if should_print_progress():
                        tag = format_tag("ERR", color_idx)
                        write_progress(f"{tag} {line.rstrip()}")
                stream.close()

            # Start reader threads
            stdout_thread = threading.Thread(
                target=process_json_stream,
                args=(
                    process.stdout,
                    response_text_parts,
                    raw_stdout_lines,
                    debug_json_logger,
                    color_index,
                ),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=read_stderr,
                args=(process.stderr, stderr_chunks, color_index),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            # Write prompt to stdin in a separate thread to avoid deadlock
            # (if prompt > pipe buffer size and codex writes stdout before
            # finishing stdin read, both sides block without concurrent I/O)
            def write_stdin(
                stream: Any,
                data: str,
            ) -> None:
                """Write prompt to stdin and close."""
                try:
                    stream.write(data)
                except (BrokenPipeError, OSError):
                    pass  # Process died before reading all input
                finally:
                    with contextlib.suppress(OSError):
                        stream.close()

            stdin_thread = threading.Thread(
                target=write_stdin,
                args=(process.stdin, prompt),
                daemon=True,
            )
            stdin_thread.start()

            # Wait for process with timeout and cooperative cancellation.
            deadline = start_time + effective_timeout
            wall_deadline = wall_start_time + effective_timeout
            returncode: int | None = None
            cancelled = False
            idle_progress_interval = _idle_progress_interval(effective_timeout)
            last_activity_at = start_time
            last_wall_activity_at = wall_start_time
            next_idle_log_at = start_time + idle_progress_interval
            last_stdout_line_count = 0
            last_stderr_chunk_count = 0

            while True:
                now = time.perf_counter()
                wall_now = time.time()
                stdout_line_count = len(raw_stdout_lines)
                stderr_chunk_count = len(stderr_chunks)
                if (
                    stdout_line_count != last_stdout_line_count
                    or stderr_chunk_count != last_stderr_chunk_count
                ):
                    last_activity_at = now
                    last_wall_activity_at = wall_now
                    next_idle_log_at = now + idle_progress_interval
                    last_stdout_line_count = stdout_line_count
                    last_stderr_chunk_count = stderr_chunk_count
                elif now >= next_idle_log_at:
                    elapsed_seconds = int(max(now - start_time, wall_now - wall_start_time))
                    idle_seconds = int(max(now - last_activity_at, wall_now - last_wall_activity_at))
                    logger.warning(
                        "Codex CLI still running with no new output: "
                        "provider=%s, model=%s, elapsed_seconds=%d, "
                        "idle_seconds=%d, timeout=%ds, prompt_len=%d, "
                        "stdout_chars=%d, stderr_chars=%d, thread_id=%s",
                        self.provider_name,
                        effective_model,
                        elapsed_seconds,
                        idle_seconds,
                        effective_timeout,
                        len(prompt),
                        sum(len(line) for line in raw_stdout_lines),
                        sum(len(chunk) for chunk in stderr_chunks),
                        thread_id or "-",
                    )
                    next_idle_log_at = now + idle_progress_interval

                if cancel_token is not None and cancel_token.is_set():
                    logger.info("Cancel token set, terminating Codex subprocess")
                    cancelled = True
                    if child_pgid is not None:
                        with contextlib.suppress(
                            ProcessLookupError,
                            PermissionError,
                            OSError,
                        ):
                            os.killpg(child_pgid, signal.SIGKILL)
                    else:
                        with contextlib.suppress(
                            ProcessLookupError,
                            PermissionError,
                            OSError,
                        ):
                            process.kill()
                    with contextlib.suppress(
                        TimeoutExpired,
                        ProcessLookupError,
                        OSError,
                    ):
                        process.wait(timeout=1)
                    returncode = -15
                    break

                if now >= deadline or wall_now >= wall_deadline:
                    process_snapshot = _collect_process_snapshot(
                        _safe_process_id(getattr(process, "pid", None)),
                        child_pgid,
                    )
                    stderr_chunks.append(
                        "\n\n## Codex Process Snapshot At Timeout\n"
                        f"{process_snapshot}\n"
                    )
                    if child_pgid is not None:
                        with contextlib.suppress(
                            ProcessLookupError,
                            PermissionError,
                            OSError,
                        ):
                            os.killpg(child_pgid, signal.SIGKILL)
                    else:
                        with contextlib.suppress(
                            ProcessLookupError,
                            PermissionError,
                            OSError,
                        ):
                            process.kill()
                    with contextlib.suppress(
                        TimeoutExpired,
                        ProcessLookupError,
                        OSError,
                    ):
                        process.wait(timeout=1)
                    stdin_thread.join(timeout=1)
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    duration_ms = int(max(now - start_time, wall_now - wall_start_time) * 1000)

                    partial_result = ProviderResult(
                        stdout="".join(response_text_parts),
                        stderr="".join(stderr_chunks),
                        exit_code=-1,
                        duration_ms=duration_ms,
                        model=effective_model,
                        command=original_command,
                    )
                    timeout_message = _format_timeout_message(
                        timeout_seconds=effective_timeout,
                        prompt_len=len(prompt),
                        partial_result=partial_result,
                        thread_id=thread_id,
                    )

                    logger.warning(
                        "Provider timeout: provider=%s, model=%s, timeout=%ds, "
                        "duration_ms=%d, prompt_len=%d, stdout_chars=%d, "
                        "stderr_chars=%d, thread_id=%s",
                        self.provider_name,
                        effective_model,
                        effective_timeout,
                        duration_ms,
                        len(prompt),
                        len(partial_result.stdout),
                        len(partial_result.stderr),
                        thread_id or "-",
                    )

                    raise ProviderTimeoutError(
                        timeout_message,
                        partial_result=partial_result,
                    ) from None

                try:
                    returncode = process.wait(timeout=0.5)
                    break
                except TimeoutExpired:
                    continue

            if cancelled:
                stdin_thread.join(timeout=1)
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                duration_ms = int(
                    max(
                        time.perf_counter() - start_time,
                        time.time() - wall_start_time,
                    )
                    * 1000
                )
                return ProviderResult(
                    stdout="".join(response_text_parts),
                    stderr="Cancelled by orchestration",
                    exit_code=-15,
                    duration_ms=duration_ms,
                    model=effective_model,
                    command=original_command,
                    provider_session_id=debug_json_logger.provider_session_id,
                )

            # Wait for threads to finish (timeout prevents hang if reader stuck)
            stdin_thread.join(timeout=5)
            stdout_thread.join(timeout=10)
            stderr_thread.join(timeout=10)

        except FileNotFoundError as e:
            logger.error("Codex CLI not found in PATH")
            raise ProviderError("Codex CLI not found. Is 'codex' in PATH?") from e
        finally:
            if child_pgid is not None:
                unregister_child_pgid(child_pgid)
            debug_json_logger.close()

        duration_ms = int(
            max(
                time.perf_counter() - start_time,
                time.time() - wall_start_time,
            )
            * 1000
        )
        stderr_content = "".join(stderr_chunks)

        if returncode != 0:
            exit_status = ExitStatus.from_code(returncode)
            failure_diagnostic = _format_codex_failure_diagnostic(
                stderr_content,
                raw_stdout_lines,
            )
            diagnostic_stream = stderr_content
            stdout_error_messages = _collect_codex_stream_error_messages(raw_stdout_lines)
            if stdout_error_messages:
                stdout_error_text = "\n".join(stdout_error_messages)
                diagnostic_stream = (
                    f"{stderr_content}\n\nCodex stdout error events:\n"
                    f"{stdout_error_text}"
                ).strip()

            logger.error(
                "Codex CLI failed: exit_code=%d, status=%s, model=%s, diagnostic=%s",
                returncode,
                exit_status.name,
                effective_model,
                failure_diagnostic,
            )

            if exit_status == ExitStatus.SIGNAL:
                signal_num = ExitStatus.get_signal_number(returncode)
                message = (
                    f"Codex CLI failed with exit code {returncode} "
                    f"(signal {signal_num}): {failure_diagnostic}"
                )
            elif exit_status == ExitStatus.NOT_FOUND:
                message = (
                    f"Codex CLI failed with exit code {returncode} "
                    f"(command not found - check PATH): {failure_diagnostic}"
                )
            elif exit_status == ExitStatus.CANNOT_EXECUTE:
                message = (
                    f"Codex CLI failed with exit code {returncode} "
                    f"(permission denied): {failure_diagnostic}"
                )
            else:
                message = f"Codex CLI failed with exit code {returncode}: {failure_diagnostic}"

            raise ProviderExitCodeError(
                message,
                exit_code=returncode,
                exit_status=exit_status,
                stderr=diagnostic_stream,
                command=original_command,
            )

        # Combine extracted text parts
        response_text = "\n".join(response_text_parts)

        # Get provider session_id (thread_id for Codex)
        provider_session_id = debug_json_logger.provider_session_id

        logger.info(
            "Codex CLI completed: duration=%dms, exit_code=%d, text_len=%d",
            duration_ms,
            returncode,
            len(response_text),
        )

        return ProviderResult(
            stdout=response_text,
            stderr=stderr_content,
            exit_code=returncode,
            duration_ms=duration_ms,
            model=effective_model,
            command=original_command,
            provider_session_id=provider_session_id,
        )

    def parse_output(self, result: ProviderResult) -> str:
        r"""Extract response text from Codex CLI output.

        Codex CLI outputs progress to stderr and final message to stdout.
        No JSON parsing is needed - the response is the raw stdout with
        leading/trailing whitespace stripped.

        Args:
            result: ProviderResult from invoke() containing raw output.

        Returns:
            Extracted response text with whitespace stripped.
            Empty string if stdout is empty.

        Example:
            >>> result = ProviderResult(stdout="  Code review complete  \n", ...)
            >>> provider.parse_output(result)
            'Code review complete'

        """
        return result.stdout.strip()
