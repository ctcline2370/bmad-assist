"""Main loop orchestration package for bmad-assist.

This package provides the main development loop that orchestrates the BMAD
methodology workflow: create story -> validate -> develop -> code review -> retrospective.

Public API (what CLI uses):
    run_loop: Execute the main BMAD development loop
    LoopExitReason: Enum indicating how run_loop() exited

Note: This __init__.py temporarily re-exports all symbols for backward
compatibility during the refactor. After test imports are updated,
this will be reduced to just run_loop and LoopExitReason.

"""

# Re-export from state for backward compatibility (Phase, State, etc.)
# These are needed because tests patch bmad_assist.core.loop.load_state etc.
# Dispatch
from bmad_assist.core.loop.dispatch import (
    execute_phase,
    get_handler,
    init_handlers,
)

# Epic transitions
from bmad_assist.core.loop.epic_transitions import (
    advance_to_next_epic,
    complete_epic,
    get_next_epic,
    handle_epic_completion,
    is_last_epic,
    persist_epic_completion,
)

# Guardian
from bmad_assist.core.loop.guardian import (
    get_next_phase,
    guardian_check_anomaly,
)

# Handlers (stub functions for backward compatibility)
from bmad_assist.core.loop.handlers_stub import (
    WORKFLOW_HANDLERS,
    code_review_handler,
    code_review_synthesis_handler,
    create_story_handler,
    dev_story_handler,
    retrospective_handler,
    validate_story_handler,
    validate_story_synthesis_handler,
)

# Locking (re-exports for backward compatibility - tests may import from loop)
from bmad_assist.core.loop.locking import (
    _is_pid_alive,
    _read_lock_file,
    _running_lock,
)

# Runner
from bmad_assist.core.loop.runner import run_loop

# Signals
from bmad_assist.core.loop.signals import (
    _get_interrupt_exit_reason,
    _handle_sigint,
    _handle_sigterm,
    get_received_signal,
    register_pre_exit_cleanup,
    register_signal_handlers,
    request_shutdown,
    reset_shutdown,
    shutdown_requested,
    unregister_pre_exit_cleanup,
    unregister_signal_handlers,
)

# Story transitions
from bmad_assist.core.loop.story_transitions import (
    advance_to_next_story,
    complete_story,
    get_next_story_id,
    handle_story_completion,
    is_last_story_in_epic,
    persist_story_completion,
)

# Types
from bmad_assist.core.loop.types import (
    GuardianDecision,
    LoopExitReason,
    PhaseHandler,
    PhaseResult,
)
from bmad_assist.core.state import (
    STATE_DIR,
    STATE_FILENAME,
    Phase,
    State,
    get_state_path,
    load_state,
    save_state,
)

__all__ = [
    # Public API
    "run_loop",
    "LoopExitReason",
    # Re-exports from state
    "Phase",
    "State",
    "STATE_DIR",
    "STATE_FILENAME",
    "load_state",
    "save_state",
    "get_state_path",
    # Types
    "GuardianDecision",
    "PhaseResult",
    "PhaseHandler",
    # Signals
    "shutdown_requested",
    "request_shutdown",
    "reset_shutdown",
    "get_received_signal",
    "register_pre_exit_cleanup",
    "register_signal_handlers",
    "unregister_signal_handlers",
    "unregister_pre_exit_cleanup",
    "_get_interrupt_exit_reason",
    "_handle_sigint",
    "_handle_sigterm",
    # Handlers
    "create_story_handler",
    "validate_story_handler",
    "validate_story_synthesis_handler",
    "dev_story_handler",
    "code_review_handler",
    "code_review_synthesis_handler",
    "retrospective_handler",
    "WORKFLOW_HANDLERS",
    # Dispatch
    "init_handlers",
    "get_handler",
    "execute_phase",
    # Story transitions
    "complete_story",
    "is_last_story_in_epic",
    "get_next_story_id",
    "advance_to_next_story",
    "persist_story_completion",
    "handle_story_completion",
    # Epic transitions
    "complete_epic",
    "is_last_epic",
    "get_next_epic",
    "advance_to_next_epic",
    "persist_epic_completion",
    "handle_epic_completion",
    # Guardian
    "get_next_phase",
    "guardian_check_anomaly",
    # Locking
    "_is_pid_alive",
    "_read_lock_file",
    "_running_lock",
]
