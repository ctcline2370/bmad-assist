"""Retrospective report extraction and persistence module.

Bug Fix: Retrospective Report Persistence

This module provides:
- extract_retrospective_report(): Extract report from LLM output using markers
- save_retrospective_report(): Save retrospective report to file

Pattern follows validation/reports.py extraction strategy.
"""

from bmad_assist.retrospective.feedback_loop import (
    FeedbackLoopValidationResult,
    FeedbackSnapshot,
    capture_feedback_snapshot,
    validate_retrospective_feedback_loop,
)
from bmad_assist.retrospective.reports import (
    extract_retrospective_report,
    save_retrospective_report,
)

__all__ = [
    "FeedbackLoopValidationResult",
    "FeedbackSnapshot",
    "capture_feedback_snapshot",
    "extract_retrospective_report",
    "save_retrospective_report",
    "validate_retrospective_feedback_loop",
]
