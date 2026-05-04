"""Shared fixtures and helpers for provider tests.

This module provides mock factories for testing Popen-based providers that use
JSON streaming output (Codex with --json, Gemini with --output-format stream-json).

## Mock Process Factories

- `create_mock_process()` - Generic Popen mock with configurable stdout/stderr
- `create_codex_mock_process()` - Codex-specific mock with JSON stream format
- `create_gemini_mock_process()` - Gemini-specific mock with JSON stream format

## JSON Output Generators

- `make_codex_json_output()` - Creates Codex JSONL stream (thread.started, item.completed, etc.)
- `make_gemini_json_output()` - Creates Gemini JSONL stream (init, message, result)
- `make_claude_stream_json_output()` - Creates Claude stream-json format (system/init, assistant, result)

## Usage Example

```python
from .conftest import create_codex_mock_process

def test_codex_invoke(provider):
    with patch("bmad_assist.providers.codex.Popen") as mock_popen:
        mock_popen.return_value = create_codex_mock_process(
            response_text="Expected response",
            returncode=0,
        )
        result = provider.invoke("Hello")
        assert result.stdout == "Expected response"
```
"""

import json
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest


def make_claude_stream_json_output(
    text: str = "Mock response", session_id: str = "test-session"
) -> str:
    """Create stream-json format output for testing.

    Args:
        text: Response text to include.
        session_id: Session ID for init message.

    Returns:
        Multi-line string with JSON stream messages.

    Note:
        The text appears ONLY in the assistant message, not in the result message,
        because the provider extracts text from assistant messages. The result
        message contains metadata only (cost, duration, turns).

    """
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": session_id}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            }
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.001,
                "duration_ms": 100,
                "num_turns": 1,
                "session_id": session_id,
            }
        ),
    ]
    return "\n".join(lines) + "\n"


def create_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    poll_returns_none_count: int = 1,
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for testing.

    Args:
        stdout_content: Raw content for stdout. If None, generates stream-json.
        stderr_content: Content to return from stderr.readline()
        returncode: Exit code to return from poll() when done
        wait_side_effect: Exception to raise from wait() (e.g., TimeoutExpired)
        response_text: Text to include in stream-json output (if stdout_content is None)
        poll_returns_none_count: Number of poll() calls that return None before
            returning returncode. Set to 1 for instant completion.
        never_finish: If True, poll() always returns None (for timeout tests).

    Returns:
        MagicMock configured to behave like a Popen process

    """
    mock_process = MagicMock()

    # Generate stream-json if no raw content provided
    if stdout_content is None:
        stdout_content = make_stream_json_output(response_text)

    # Create iterators for stdout/stderr line reading
    stdout_lines = stdout_content.split("\n") if stdout_content else []
    stderr_lines = stderr_content.split("\n") if stderr_content else []

    # Add newlines back except for empty strings at end
    stdout_iter = iter([line + "\n" if line else "" for line in stdout_lines])
    stderr_iter = iter([line + "\n" if line else "" for line in stderr_lines])

    mock_process.stdout.readline.side_effect = lambda: next(stdout_iter, "")
    mock_process.stderr.readline.side_effect = lambda: next(stderr_iter, "")
    mock_process.stdout.close = MagicMock()
    mock_process.stderr.close = MagicMock()

    # Mock stdin for providers that use stdin (Claude, Gemini)
    mock_process.stdin = MagicMock()

    # Track write() calls for verification
    write_args: list[str] = []

    def capture_write(arg: str) -> None:
        write_args.append(arg)

    mock_process.stdin.write = MagicMock(side_effect=capture_write)
    mock_process.stdin.write_args = write_args  # type: ignore
    mock_process.stdin.close = MagicMock()

    # Mock poll() for the polling loop (used by claude.py and others)
    if never_finish:
        # Always return None (process never finishes - for timeout tests)
        mock_process.poll.return_value = None
        # Also set wait to raise TimeoutExpired for providers using wait()
        mock_process.wait.side_effect = TimeoutExpired(cmd=["mock"], timeout=5)
    else:
        # Return None poll_returns_none_count times, then returncode
        poll_call_count = [0]

        def poll_side_effect():
            poll_call_count[0] += 1
            if poll_call_count[0] <= poll_returns_none_count:
                return None
            return returncode

        mock_process.poll.side_effect = poll_side_effect

        # Legacy wait mock for backward compatibility
        if wait_side_effect:
            mock_process.wait.side_effect = wait_side_effect
        else:
            mock_process.wait.return_value = returncode

    mock_process.kill = MagicMock()

    return mock_process


def make_codex_json_output(text: str = "Mock response", thread_id: str = "test-thread") -> str:
    """Create Codex --json format output for testing.

    Args:
        text: Response text to include.
        thread_id: Thread ID for init message.

    Returns:
        Multi-line string with Codex JSON stream messages.

    """
    lines = [
        json.dumps({"type": "thread.started", "thread_id": thread_id}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "item_1", "type": "agent_message", "text": text},
            }
        ),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}}),
    ]
    return "\n".join(lines) + "\n"


def make_gemini_json_output(text: str = "Mock response", session_id: str = "test-session") -> str:
    """Create Gemini --output-format stream-json output for testing.

    Args:
        text: Response text to include.
        session_id: Session ID for init message.

    Returns:
        Multi-line string with Gemini JSON stream messages.

    """
    lines = [
        json.dumps({"type": "init", "session_id": session_id, "model": "gemini-2.5-flash"}),
        json.dumps({"type": "message", "role": "user", "content": "test prompt"}),
        json.dumps({"type": "message", "role": "assistant", "content": text}),
        json.dumps(
            {
                "type": "result",
                "status": "success",
                "stats": {"total_tokens": 150, "duration_ms": 1000},
            }
        ),
    ]
    return "\n".join(lines) + "\n"


# Backward compatibility alias
make_stream_json_output = make_claude_stream_json_output


@pytest.fixture
def mock_popen_success():
    """Fixture that mocks Popen for successful invocation."""
    with patch("bmad_assist.providers.claude.Popen") as mock:
        mock.return_value = create_mock_process(
            stdout_content="Mock response",
            stderr_content="",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_popen_timeout():
    """Fixture that mocks Popen for timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.claude.Popen") as mock,
        patch("bmad_assist.providers.claude.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.claude.time.sleep"),
    ):
        mock.return_value = create_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_popen_error():
    """Fixture that mocks Popen for non-zero exit."""
    with patch("bmad_assist.providers.claude.Popen") as mock:
        mock.return_value = create_mock_process(
            stdout_content="",
            stderr_content="Error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_popen_not_found():
    """Fixture that mocks Popen when CLI not found."""
    with patch("bmad_assist.providers.claude.Popen") as mock:
        mock.side_effect = FileNotFoundError("claude")
        yield mock


# =============================================================================
# Codex Provider Popen Fixtures
# =============================================================================


def create_codex_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Codex testing."""
    if stdout_content is None:
        stdout_content = make_codex_json_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_codex_popen_success():
    """Fixture that mocks Popen for successful Codex invocation."""
    with patch("bmad_assist.providers.codex.Popen") as mock:
        mock.return_value = create_codex_mock_process(
            response_text="Mock Codex response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_codex_popen_timeout():
    """Fixture that mocks Popen for Codex timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.codex.Popen") as mock,
        patch("bmad_assist.providers.codex.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.codex.time.sleep"),
    ):
        mock.return_value = create_codex_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_codex_popen_error():
    """Fixture that mocks Popen for non-zero Codex exit."""
    with patch("bmad_assist.providers.codex.Popen") as mock:
        mock.return_value = create_codex_mock_process(
            stdout_content="",
            stderr_content="Codex error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_codex_popen_not_found():
    """Fixture that mocks Popen when Codex CLI not found."""
    with patch("bmad_assist.providers.codex.Popen") as mock:
        mock.side_effect = FileNotFoundError("codex")
        yield mock


# =============================================================================
# Gemini Provider Popen Fixtures
# =============================================================================


def create_gemini_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Gemini testing."""
    if stdout_content is None:
        stdout_content = make_gemini_json_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_gemini_popen_success():
    """Fixture that mocks Popen for successful Gemini invocation."""
    with patch("bmad_assist.providers.gemini.Popen") as mock:
        mock.return_value = create_gemini_mock_process(
            response_text="Mock Gemini response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_gemini_popen_timeout():
    """Fixture that mocks Popen for Gemini timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.gemini.Popen") as mock,
        patch("bmad_assist.providers.gemini.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.gemini.time.sleep"),
    ):
        mock.return_value = create_gemini_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_gemini_popen_error():
    """Fixture that mocks Popen for non-zero Gemini exit."""
    with patch("bmad_assist.providers.gemini.Popen") as mock:
        mock.return_value = create_gemini_mock_process(
            stdout_content="",
            stderr_content="Gemini error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_gemini_popen_not_found():
    """Fixture that mocks Popen when Gemini CLI not found."""
    with patch("bmad_assist.providers.gemini.Popen") as mock:
        mock.side_effect = FileNotFoundError("gemini")
        yield mock


# =============================================================================
# OpenCode Provider Popen Fixtures
# =============================================================================


def make_opencode_json_output(
    text: str = "Mock response",
    session_id: str = "ses_test",
) -> str:
    """Create OpenCode --format json output for testing.

    Args:
        text: Response text to include.
        session_id: Session ID for step_start message.

    Returns:
        Multi-line string with OpenCode JSON stream messages.

    """
    lines = [
        json.dumps({"type": "step_start", "sessionID": session_id, "part": {"type": "step-start"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": text}}),
        json.dumps(
            {
                "type": "step_finish",
                "part": {"type": "step-finish", "reason": "stop", "cost": 0.001, "tokens": {}},
            }
        ),
    ]
    return "\n".join(lines) + "\n"


def create_opencode_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for OpenCode testing."""
    if stdout_content is None:
        stdout_content = make_opencode_json_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_opencode_popen_success():
    """Fixture that mocks Popen for successful OpenCode invocation."""
    with patch("bmad_assist.providers.opencode.Popen") as mock:
        mock.return_value = create_opencode_mock_process(
            response_text="Mock OpenCode response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_opencode_popen_timeout():
    """Fixture that mocks Popen for OpenCode timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.opencode.Popen") as mock,
        patch("bmad_assist.providers.opencode.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.opencode.time.sleep"),
    ):
        mock.return_value = create_opencode_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_opencode_popen_error():
    """Fixture that mocks Popen for non-zero OpenCode exit."""
    with patch("bmad_assist.providers.opencode.Popen") as mock:
        mock.return_value = create_opencode_mock_process(
            stdout_content="",
            stderr_content="OpenCode error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_opencode_popen_not_found():
    """Fixture that mocks Popen when OpenCode CLI not found."""
    with patch("bmad_assist.providers.opencode.Popen") as mock:
        mock.side_effect = FileNotFoundError("opencode")
        yield mock


# =============================================================================
# Amp Provider Popen Fixtures
# =============================================================================


def make_amp_json_output(
    text: str = "Mock response",
    session_id: str = "T-test",
) -> str:
    """Create Amp -x --stream-json output for testing.

    Args:
        text: Response text to include.
        session_id: Session ID for system message.

    Returns:
        Multi-line string with Amp JSON stream messages.

    """
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": session_id, "tools": []}),
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": "test prompt"}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                },
            }
        ),
        json.dumps(
            {"type": "result", "subtype": "success", "duration_ms": 100, "result": text}
        ),
    ]
    return "\n".join(lines) + "\n"


def create_amp_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Amp testing."""
    if stdout_content is None:
        stdout_content = make_amp_json_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_amp_popen_success():
    """Fixture that mocks Popen for successful Amp invocation."""
    with patch("bmad_assist.providers.amp.Popen") as mock:
        mock.return_value = create_amp_mock_process(
            response_text="Mock Amp response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_amp_popen_timeout():
    """Fixture that mocks Popen for Amp timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.amp.Popen") as mock,
        patch("bmad_assist.providers.amp.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.amp.time.sleep"),
    ):
        mock.return_value = create_amp_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_amp_popen_error():
    """Fixture that mocks Popen for non-zero Amp exit."""
    with patch("bmad_assist.providers.amp.Popen") as mock:
        mock.return_value = create_amp_mock_process(
            stdout_content="",
            stderr_content="Amp error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_amp_popen_not_found():
    """Fixture that mocks Popen when Amp CLI not found."""
    with patch("bmad_assist.providers.amp.Popen") as mock:
        mock.side_effect = FileNotFoundError("amp")
        yield mock


# =============================================================================
# Copilot Provider Popen Fixtures
# =============================================================================


def make_copilot_output(text: str = "Mock response") -> str:
    """Create plain text output for Copilot testing.

    Args:
        text: Response text to include.

    Returns:
        Plain text output with trailing newline.

    """
    return text + "\n"


def create_copilot_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Copilot testing.

    Copilot outputs plain text (no JSON parsing needed).

    Args:
        stdout_content: Raw stdout content. If None, generates from response_text.
        stderr_content: Content for stderr.
        returncode: Exit code for wait().
        wait_side_effect: Exception to raise from wait() (e.g., TimeoutExpired).
        response_text: Text to use if stdout_content is None.
        never_finish: If True, poll() always returns None.

    Returns:
        MagicMock configured to behave like a Popen process.

    """
    if stdout_content is None:
        stdout_content = make_copilot_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_copilot_popen_success():
    """Fixture that mocks Popen for successful Copilot invocation."""
    with patch("bmad_assist.providers.copilot.Popen") as mock:
        mock.return_value = create_copilot_mock_process(
            response_text="Mock Copilot response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_copilot_popen_timeout():
    """Fixture that mocks Popen for Copilot timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.copilot.Popen") as mock,
        patch("bmad_assist.providers.copilot.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.copilot.time.sleep"),
    ):
        mock.return_value = create_copilot_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_copilot_popen_error():
    """Fixture that mocks Popen for non-zero Copilot exit."""
    with patch("bmad_assist.providers.copilot.Popen") as mock:
        mock.return_value = create_copilot_mock_process(
            stdout_content="",
            stderr_content="Copilot error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_copilot_popen_not_found():
    """Fixture that mocks Popen when Copilot CLI not found."""
    with patch("bmad_assist.providers.copilot.Popen") as mock:
        mock.side_effect = FileNotFoundError("copilot")
        yield mock


# =============================================================================
# Cursor Agent Provider Popen Fixtures
# =============================================================================


def make_cursor_output(text: str = "Mock response") -> str:
    """Create plain text output for Cursor Agent testing.

    Args:
        text: Response text to include.

    Returns:
        Plain text output with trailing newline.

    """
    return text + "\n"


def create_cursor_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Cursor Agent testing.

    Cursor Agent with --print outputs plain text (no JSON parsing needed).

    Args:
        stdout_content: Raw stdout content. If None, generates from response_text.
        stderr_content: Content for stderr.
        returncode: Exit code for wait().
        wait_side_effect: Exception to raise from wait() (e.g., TimeoutExpired).
        response_text: Text to use if stdout_content is None.
        never_finish: If True, poll() always returns None.

    Returns:
        MagicMock configured to behave like a Popen process.

    """
    if stdout_content is None:
        stdout_content = make_cursor_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_cursor_popen_success():
    """Fixture that mocks Popen for successful Cursor Agent invocation."""
    with patch("bmad_assist.providers.cursor_agent.Popen") as mock:
        mock.return_value = create_cursor_mock_process(
            response_text="Mock Cursor Agent response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_cursor_popen_timeout():
    """Fixture that mocks Popen for Cursor Agent timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.cursor_agent.Popen") as mock,
        patch("bmad_assist.providers.cursor_agent.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.cursor_agent.time.sleep"),
    ):
        mock.return_value = create_cursor_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_cursor_popen_error():
    """Fixture that mocks Popen for non-zero Cursor Agent exit."""
    with patch("bmad_assist.providers.cursor_agent.Popen") as mock:
        mock.return_value = create_cursor_mock_process(
            stdout_content="",
            stderr_content="Cursor Agent error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_cursor_popen_not_found():
    """Fixture that mocks Popen when Cursor Agent CLI not found."""
    with patch("bmad_assist.providers.cursor_agent.Popen") as mock:
        mock.side_effect = FileNotFoundError("cursor-agent")
        yield mock


# =============================================================================
# Kimi Provider Popen Fixtures
# =============================================================================


def make_kimi_json_output(
    text: str = "Mock response",
    reasoning_content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> str:
    """Create Kimi OpenAI-style JSONL output for testing.

    Args:
        text: Response text to include in content field.
        reasoning_content: Optional thinking content (for --thinking mode).
        tool_calls: Optional list of tool call dicts to include.

    Returns:
        Multi-line string with Kimi JSONL messages.

    """
    msg: dict = {"role": "assistant", "content": text}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps(msg) + "\n"


def make_kimi_multi_message_output(messages: list[str]) -> str:
    """Create Kimi JSONL with multiple assistant messages.

    Args:
        messages: List of content strings for each assistant message.

    Returns:
        Multi-line string with multiple Kimi JSONL messages.

    """
    lines = [json.dumps({"role": "assistant", "content": text}) for text in messages]
    return "\n".join(lines) + "\n"


def create_kimi_mock_process(
    stdout_content: str | None = None,
    stderr_content: str = "",
    returncode: int = 0,
    wait_side_effect: Exception | None = None,
    response_text: str = "Mock response",
    never_finish: bool = False,
) -> MagicMock:
    """Create a mock Popen process for Kimi testing.

    Args:
        stdout_content: Raw stdout content. If None, generates from response_text.
        stderr_content: Content for stderr.
        returncode: Exit code for wait().
        wait_side_effect: Exception to raise from wait() (e.g., TimeoutExpired).
        response_text: Text to use if stdout_content is None.
        never_finish: If True, poll() always returns None.

    Returns:
        MagicMock configured to behave like a Popen process.

    """
    if stdout_content is None:
        stdout_content = make_kimi_json_output(response_text)
    return create_mock_process(
        stdout_content=stdout_content,
        stderr_content=stderr_content,
        returncode=returncode,
        wait_side_effect=wait_side_effect,
        never_finish=never_finish,
    )


@pytest.fixture
def mock_kimi_popen_success():
    """Fixture that mocks Popen for successful Kimi invocation."""
    with patch("bmad_assist.providers.kimi.Popen") as mock:
        mock.return_value = create_kimi_mock_process(
            response_text="Mock Kimi response",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_kimi_popen_timeout():
    """Fixture that mocks Popen for Kimi timeout with accelerated time."""
    call_count = [0]

    def mock_perf_counter():
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.kimi.Popen") as mock,
        patch("bmad_assist.providers.kimi.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.kimi.time.sleep"),
    ):
        mock.return_value = create_kimi_mock_process(never_finish=True)
        yield mock


@pytest.fixture
def mock_kimi_popen_error():
    """Fixture that mocks Popen for non-zero Kimi exit."""
    with patch("bmad_assist.providers.kimi.Popen") as mock:
        mock.return_value = create_kimi_mock_process(
            stdout_content="",
            stderr_content="Kimi error message",
            returncode=1,
        )
        yield mock


@pytest.fixture
def mock_kimi_popen_not_found():
    """Fixture that mocks Popen when Kimi CLI not found."""
    with patch("bmad_assist.providers.kimi.Popen") as mock:
        mock.side_effect = FileNotFoundError("kimi")
        yield mock


# =============================================================================
# Time Acceleration for Timeout Tests
# =============================================================================


@pytest.fixture
def accelerated_time():
    """Fixture that accelerates time for instant timeout tests.

    Patches time.perf_counter to return rapidly increasing values and
    time.sleep to not actually sleep. This makes timeout tests complete
    instantly instead of waiting for real timeouts.

    Usage:
        def test_timeout(accelerated_time, provider):
            with patch("bmad_assist.providers.claude.Popen") as mock_popen:
                mock_popen.return_value = create_mock_process(never_finish=True)
                with pytest.raises(ProviderTimeoutError):
                    provider.invoke("Hello", timeout=5)  # Returns instantly

    """
    call_count = [0]

    def mock_perf_counter():
        # Return rapidly increasing values: 0, 10, 20, 30...
        # This ensures any timeout (e.g., 5s) is exceeded after 1 iteration
        result = call_count[0] * 10
        call_count[0] += 1
        return result

    with (
        patch("bmad_assist.providers.claude.time.perf_counter", side_effect=mock_perf_counter),
        patch("bmad_assist.providers.claude.time.sleep"),
        patch("bmad_assist.providers.codex.time.perf_counter", side_effect=mock_perf_counter),
    ):
        yield
