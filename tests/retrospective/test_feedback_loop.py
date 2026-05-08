"""Tests for retrospective upstream document feedback-loop validation."""

from pathlib import Path

from bmad_assist.retrospective.feedback_loop import (
    GOVERNANCE_CANDIDATE_ROOT,
    capture_feedback_snapshot,
    validate_retrospective_feedback_loop,
)


def _report_with_matrix(row: str) -> str:
    return f"""# Epic 11 Retrospective

## Summary
The epic is complete.

## Document Feedback Matrix
| Finding | Disposition | Evidence | Owner | Blocked Downstream Work |
|---|---|---|---|---|
{row}
"""


def test_no_change_matrix_passes_with_concrete_rationale(tmp_path: Path) -> None:
    """No-change rows pass when they include a concrete rationale."""
    snapshot = capture_feedback_snapshot(tmp_path)
    report = _report_with_matrix(
        "| No upstream document changes were required. | no-change | Reviewed PRD, architecture, epic, story, and readiness context; findings were already covered by existing source documents. | BMAD-ASSIST | none |"
    )

    result = validate_retrospective_feedback_loop(report, tmp_path, snapshot)

    assert result.valid
    assert result.dispositions == ("no-change",)


def test_missing_document_feedback_matrix_fails(tmp_path: Path) -> None:
    """A retrospective without the required matrix fails closed."""
    snapshot = capture_feedback_snapshot(tmp_path)

    result = validate_retrospective_feedback_loop(
        "# Epic 11 Retrospective\n\n## Summary\nDone.\n",
        tmp_path,
        snapshot,
    )

    assert not result.valid
    assert "missing required 'Document Feedback Matrix' section" in result.errors[0]


def test_updated_path_must_change_since_snapshot(tmp_path: Path) -> None:
    """Updated rows must point at files changed after the pre-run snapshot."""
    prd = tmp_path / "docs" / "prd.md"
    prd.parent.mkdir()
    prd.write_text("# PRD\n\nInitial requirements.\n")
    snapshot = capture_feedback_snapshot(tmp_path)

    unchanged_report = _report_with_matrix(
        "| API standards need explicit naming rules. | updated | Updated `docs/prd.md` with the API naming rule. | BMAD-ASSIST | none |"
    )

    unchanged_result = validate_retrospective_feedback_loop(
        unchanged_report,
        tmp_path,
        snapshot,
    )

    assert not unchanged_result.valid
    assert "was not changed during retrospective" in unchanged_result.errors[0]

    prd.write_text("# PRD\n\nInitial requirements.\n\nAPI naming rules hardened.\n")
    changed_result = validate_retrospective_feedback_loop(
        unchanged_report,
        tmp_path,
        snapshot,
    )

    assert changed_result.valid
    assert changed_result.dispositions == ("updated",)


def test_updated_path_rejects_volatile_bmad_output(tmp_path: Path) -> None:
    """Updated rows cannot use volatile BMAD output folders as upstream evidence."""
    artifact = tmp_path / "_bmad-output" / "planning-artifacts" / "prd.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("# PRD\n\nInitial.\n")
    snapshot = capture_feedback_snapshot(tmp_path)
    artifact.write_text("# PRD\n\nChanged.\n")
    report = _report_with_matrix(
        "| Finding only updated volatile output. | updated | Updated `_bmad-output/planning-artifacts/prd.md`. | BMAD-ASSIST | none |"
    )

    result = validate_retrospective_feedback_loop(report, tmp_path, snapshot)

    assert not result.valid
    assert "inside volatile BMAD output/config" in result.errors[0]


def test_governance_candidate_requires_candidate_handoff_path(tmp_path: Path) -> None:
    """Governance-candidate rows pass only with approved candidate handoff paths."""
    snapshot = capture_feedback_snapshot(tmp_path)
    candidate = tmp_path / GOVERNANCE_CANDIDATE_ROOT / "retro-api-standardization.md"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# Candidate\n\nRule proposal from retrospective.\n")
    report = _report_with_matrix(
        "| API standards need a governance rule. | governance-candidate | Drafted `.output/adr-control-plane/bmad-governance-candidates/retro-api-standardization.md`. | BMAD-ASSIST | Governance approval before canonical rule update. |"
    )

    result = validate_retrospective_feedback_loop(report, tmp_path, snapshot)

    assert result.valid
    assert result.dispositions == ("governance-candidate",)


def test_governance_candidate_rejects_canonical_governance_path(tmp_path: Path) -> None:
    """Governance-candidate rows cannot point at canonical governance files."""
    canonical = tmp_path / ".rules" / "rules-index.json"
    canonical.parent.mkdir()
    canonical.write_text("{}\n")
    snapshot = capture_feedback_snapshot(tmp_path)
    canonical.write_text('{"changed": true}\n')
    report = _report_with_matrix(
        "| API standards need a governance rule. | governance-candidate | Updated `.rules/rules-index.json`. | BMAD-ASSIST | Governance approval before canonical rule update. |"
    )

    result = validate_retrospective_feedback_loop(report, tmp_path, snapshot)

    assert not result.valid
    assert "must be under .output/adr-control-plane/bmad-governance-candidates/" in result.errors[0]


def test_human_gated_requires_owner_evidence_and_blocked_work(tmp_path: Path) -> None:
    """Human-gated rows must record ownership, evidence, and blocked work."""
    snapshot = capture_feedback_snapshot(tmp_path)
    invalid_report = _report_with_matrix(
        "| Security standard needs product owner decision. | human-gated | TBD | TBD | none |"
    )

    invalid_result = validate_retrospective_feedback_loop(
        invalid_report,
        tmp_path,
        snapshot,
    )

    assert not invalid_result.valid
    assert len(invalid_result.errors) == 3

    valid_report = _report_with_matrix(
        "| Security standard needs product owner decision. | human-gated | Product acceptance depends on human approval of the rollout exception. | Chris | Story creation for downstream rollout automation. |"
    )

    valid_result = validate_retrospective_feedback_loop(valid_report, tmp_path, snapshot)

    assert valid_result.valid
    assert valid_result.dispositions == ("human-gated",)
