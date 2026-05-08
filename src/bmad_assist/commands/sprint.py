"""Sprint subcommand group for bmad-assist CLI.

Commands for sprint-status management and validation.
"""

import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer

from bmad_assist.cli_utils import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    _error,
    _success,
    _validate_project_path,
    _warning,
    console,
)
from bmad_assist.core.config import load_config_with_project, load_loop_config
from bmad_assist.core.exceptions import BmadAssistError, ConfigError, StateError

if TYPE_CHECKING:
    from bmad_assist.bmad.state_reader import ProjectState
    from bmad_assist.sprint.reconciler import StatusChange

sprint_app = typer.Typer(
    name="sprint",
    help="Sprint-status management commands",
    no_args_is_help=True,
)


# --------------------------------------------------------------------------
# Sprint CLI Types and Helpers
# --------------------------------------------------------------------------


class DiscrepancySeverity(Enum):
    """Severity level for sprint-status discrepancies."""

    ERROR = "error"  # Requires fixing - causes exit code 1
    WARN = "warn"  # Advisory only - exit code 0


@dataclass(frozen=True)
class Discrepancy:
    """A discrepancy between sprint-status and artifact evidence."""

    key: str
    sprint_status: str
    inferred_status: str
    severity: DiscrepancySeverity
    reason: str
    evidence: str  # Description of artifact evidence


def _get_sprint_status_path(project_root: Path) -> Path:
    """Get sprint-status.yaml path using paths singleton.

    Args:
        project_root: Project root directory (kept for backward compatibility).

    Returns:
        Path to sprint-status.yaml in implementation artifacts.

    """
    from bmad_assist.core.paths import get_paths

    return get_paths().sprint_status_file


def _setup_sprint_context(project: str) -> tuple[Path, Path, bool]:
    """Validate paths and return (project_root, sprint_path, is_legacy_only).

    If only legacy location exists (docs/sprint-artifacts/sprint-status.yaml),
    uses that path and signals that auto_exclude_legacy should be disabled.

    Args:
        project: Project path string.

    Returns:
        Tuple of (project_root, sprint_path, is_legacy_only).
        is_legacy_only=True means auto_exclude_legacy should be False.

    Raises:
        typer.Exit: If project path is invalid.

    """
    from bmad_assist.core.paths import get_paths, init_paths

    project_root = _validate_project_path(project)

    # Initialize paths singleton if not already done
    try:
        get_paths()
    except RuntimeError:
        # Load config to get external paths if configured
        try:
            config = load_config_with_project(project_path=project_root)
            paths_config = {}
            if config.paths:
                if config.paths.output_folder:
                    paths_config["output_folder"] = config.paths.output_folder
                if config.paths.project_knowledge:
                    paths_config["project_knowledge"] = config.paths.project_knowledge
            # Add bmad_paths.epics if configured (supports custom epic locations)
            if config.bmad_paths and config.bmad_paths.epics:
                paths_config["epics"] = config.bmad_paths.epics
            init_paths(project_root, paths_config)
        except ConfigError:
            # No config - use defaults
            init_paths(project_root, {})

    paths = get_paths()
    sprint_path = paths.sprint_status_file
    legacy_path = paths.legacy_sprint_artifacts / "sprint-status.yaml"

    # If only legacy location exists, use it
    is_legacy_only = False
    if not sprint_path.exists() and legacy_path.exists():
        sprint_path = legacy_path
        is_legacy_only = True

    return project_root, sprint_path, is_legacy_only


def _display_changes_table(changes: "list[StatusChange]") -> None:
    """Display changes in a Rich table.

    Args:
        changes: List of StatusChange objects from reconciler.

    """
    from rich.table import Table

    table = Table(title="Changes Applied")
    table.add_column("Key", style="cyan")
    table.add_column("Old Status", style="yellow")
    table.add_column("New Status", style="green")
    table.add_column("Reason")
    table.add_column("Confidence", style="dim")

    for change in changes:
        old = change.old_status or "(new)"
        conf = change.confidence.name if change.confidence else "-"
        table.add_row(
            change.key,
            old,
            change.new_status,
            change.reason,
            conf,
        )

    console.print(table)


def _display_discrepancies_table(discrepancies: list[Discrepancy]) -> None:
    """Display discrepancies in a Rich table.

    Args:
        discrepancies: List of Discrepancy objects.

    """
    from rich.table import Table

    table = Table(title="Discrepancies Found")
    table.add_column("Key", style="cyan")
    table.add_column("Severity")
    table.add_column("Sprint Status", style="yellow")
    table.add_column("Inferred", style="green")
    table.add_column("Reason")

    for d in discrepancies:
        severity_str = (
            "[red]ERROR[/red]"
            if d.severity == DiscrepancySeverity.ERROR
            else "[yellow]WARN[/yellow]"
        )
        table.add_row(
            d.key,
            severity_str,
            d.sprint_status,
            d.inferred_status,
            d.reason,
        )

    console.print(table)


def _story_key_match(key: str) -> re.Match[str] | None:
    """Match the epic/story prefix of a sprint status key."""
    return re.match(r"^(?P<epic>[a-z0-9-]+?)-(?P<story>\d+)(?:-|$)", key, re.IGNORECASE)


def _story_epic_id(key: str) -> str | None:
    """Return the epic identifier for a sprint story key."""
    match = _story_key_match(key)
    if not match:
        return None
    return match.group("epic").lower()


def _story_sort_key(key: str) -> tuple[int, int, int, str, int, str]:
    """Sort sprint story keys by epic, story number, then full key."""
    match = _story_key_match(key)
    if not match:
        return (1, 1, 0, key.lower(), 0, key.lower())

    epic = match.group("epic").lower()
    story = int(match.group("story"))
    if epic.isdigit():
        return (0, 0, int(epic), "", story, key.lower())
    return (0, 1, 0, epic, story, key.lower())


def _is_story_entry(entry_type: object) -> bool:
    """Return True for status entries that represent sprint work items."""
    from bmad_assist.sprint.classifier import EntryType

    return entry_type not in (EntryType.EPIC_META, EntryType.RETROSPECTIVE)


def _parse_status_timestamp(value: datetime | None) -> datetime | None:
    """Normalize sprint metadata timestamps for staleness checks."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_sprint_status_summary(
    project_root: Path,
    sprint_path: Path,
) -> dict[str, object]:
    """Build a read-only summary matching the BMAD sprint-status workflow."""
    from bmad_assist.sprint import parse_sprint_status
    from bmad_assist.sprint.classifier import EntryType

    sprint_status = parse_sprint_status(sprint_path)

    story_entries = [
        entry for entry in sprint_status.entries.values() if _is_story_entry(entry.entry_type)
    ]
    epic_entries = [
        entry
        for entry in sprint_status.entries.values()
        if entry.entry_type == EntryType.EPIC_META
    ]
    retrospective_entries = [
        entry
        for entry in sprint_status.entries.values()
        if entry.entry_type == EntryType.RETROSPECTIVE
    ]

    story_counts = {
        "backlog": 0,
        "ready-for-dev": 0,
        "in-progress": 0,
        "review": 0,
        "done": 0,
        "blocked": 0,
        "deferred": 0,
    }
    for entry in story_entries:
        story_counts.setdefault(entry.status, 0)
        story_counts[entry.status] += 1

    epic_counts = {"backlog": 0, "in-progress": 0, "done": 0, "blocked": 0, "deferred": 0}
    for entry in epic_entries:
        epic_counts.setdefault(entry.status, 0)
        epic_counts[entry.status] += 1

    retrospective_counts = {"optional": 0, "done": 0}
    for entry in retrospective_entries:
        retrospective_counts.setdefault(entry.status, 0)
        retrospective_counts[entry.status] += 1

    sorted_story_entries = sorted(story_entries, key=lambda entry: _story_sort_key(entry.key))
    sorted_retro_entries = sorted(retrospective_entries, key=lambda entry: entry.key.lower())

    next_workflow_id: str | None = None
    next_story_id: str | None = None
    next_agent: str | None = None

    for status, workflow in [
        ("in-progress", "dev-story"),
        ("review", "code-review"),
        ("ready-for-dev", "dev-story"),
        ("backlog", "create-story"),
    ]:
        match = next((entry for entry in sorted_story_entries if entry.status == status), None)
        if match is not None:
            next_workflow_id = workflow
            next_story_id = match.key
            next_agent = "DEV"
            break

    if next_workflow_id is None:
        retro_match = next(
            (entry for entry in sorted_retro_entries if entry.status == "optional"),
            None,
        )
        if retro_match is not None:
            next_workflow_id = "retrospective"
            next_story_id = retro_match.key
            next_agent = "DEV"

    risks: list[str] = []
    review_keys = [entry.key for entry in sorted_story_entries if entry.status == "review"]
    if review_keys:
        risks.append(
            "Story status 'review' detected; run code-review for "
            + ", ".join(review_keys[:3])
            + (" ..." if len(review_keys) > 3 else "")
        )

    in_progress_keys = [
        entry.key for entry in sorted_story_entries if entry.status == "in-progress"
    ]
    if in_progress_keys and story_counts.get("ready-for-dev", 0) == 0:
        risks.append(
            "Active story in progress and no ready-for-dev stories; stay focused on "
            + in_progress_keys[0]
        )

    if (
        epic_entries
        and all(entry.status == "backlog" for entry in epic_entries)
        and story_counts.get("ready-for-dev", 0) == 0
    ):
        risks.append("All epics are backlog and no ready-for-dev story is available.")

    status_time = _parse_status_timestamp(
        sprint_status.metadata.last_updated or sprint_status.metadata.generated
    )
    if status_time is not None:
        age_days = (datetime.now(UTC) - status_time).days
        if age_days > 7:
            risks.append("sprint-status.yaml may be stale; last update is over 7 days old.")

    epic_ids = {
        entry.key.removeprefix("epic-")
        for entry in epic_entries
        if entry.entry_type == EntryType.EPIC_META
    }
    for entry in sorted_story_entries:
        epic_id = _story_epic_id(entry.key)
        if epic_id is None:
            continue
        if epic_id not in epic_ids:
            risks.append(f"Orphaned story detected: {entry.key} has no epic-{epic_id}.")

    stories_by_epic: dict[str, int] = {}
    for entry in sorted_story_entries:
        epic_id = _story_epic_id(entry.key)
        if epic_id is None:
            continue
        stories_by_epic[epic_id] = stories_by_epic.get(epic_id, 0) + 1

    for entry in epic_entries:
        epic_id = entry.key.removeprefix("epic-")
        if entry.status == "in-progress" and stories_by_epic.get(epic_id, 0) == 0:
            risks.append(f"In-progress epic has no associated stories: {entry.key}.")

    return {
        "project_root": str(project_root),
        "sprint_path": str(sprint_path),
        "project": sprint_status.metadata.project or project_root.name,
        "project_key": sprint_status.metadata.project_key,
        "tracking_system": sprint_status.metadata.tracking_system,
        "generated": (
            sprint_status.metadata.generated.isoformat()
            if sprint_status.metadata.generated
            else None
        ),
        "last_updated": (
            sprint_status.metadata.last_updated.isoformat()
            if sprint_status.metadata.last_updated
            else None
        ),
        "story_counts": story_counts,
        "epic_counts": epic_counts,
        "retrospective_counts": retrospective_counts,
        "next": {
            "workflow_id": next_workflow_id,
            "story_id": next_story_id,
            "agent": next_agent,
        },
        "risks": risks,
        "stories_by_status": {
            status: [entry.key for entry in sorted_story_entries if entry.status == status]
            for status in story_counts
        },
        "epics_by_status": {
            status: [entry.key for entry in epic_entries if entry.status == status]
            for status in epic_counts
        },
        "retrospectives_by_status": {
            status: [entry.key for entry in sorted_retro_entries if entry.status == status]
            for status in retrospective_counts
        },
        "entry_count": len(sprint_status.entries),
    }


def _iter_lifecycle_knowledge_roots() -> list[Path]:
    """Return deduplicated lifecycle knowledge roots in runtime search order."""
    from bmad_assist.core.paths import get_paths

    paths = get_paths()
    seen: set[Path] = set()
    roots: list[Path] = []

    for candidate in [
        paths.project_knowledge,
        paths.planning_artifacts,
        paths.project_docs_fallback,
    ]:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)

    return roots


def _load_lifecycle_project_state() -> "ProjectState":
    """Load lifecycle project state from the first viable knowledge root.

    The runtime resolves BMAD knowledge from multiple roots. Validation must
    mirror that search order instead of crashing when the first configured root
    is absent or empty.
    """
    from bmad_assist.bmad import read_project_state
    from bmad_assist.core.exceptions import StateError

    candidate_notes: list[str] = []

    for candidate in _iter_lifecycle_knowledge_roots():
        if not candidate.exists():
            candidate_notes.append(f"{candidate} (missing)")
            continue

        try:
            project_state = read_project_state(candidate, use_sprint_status=True)
        except FileNotFoundError:
            candidate_notes.append(f"{candidate} (missing)")
            continue

        if project_state.all_stories:
            return project_state

        candidate_notes.append(f"{candidate} (no epic stories)")

    searched = ", ".join(candidate_notes) if candidate_notes else "no knowledge roots configured"
    raise StateError(
        "Unable to load lifecycle project state from any BMAD knowledge root. "
        f"Searched: {searched}"
    )


# --------------------------------------------------------------------------
# Sprint Commands
# --------------------------------------------------------------------------


@sprint_app.command("status")
def sprint_status(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    format_: str = typer.Option(
        "plain",
        "--format",
        "-f",
        help="Output format: plain or json",
    ),
) -> None:
    """Summarize sprint-status and recommend the next BMAD workflow action."""
    import json

    project_root, sprint_path, _is_legacy_only = _setup_sprint_context(project)

    format_lower = format_.lower()
    if format_lower not in ("plain", "json"):
        _error(f"Invalid --format: {format_}. Use 'plain' or 'json'.")
        raise typer.Exit(code=EXIT_ERROR)

    if not sprint_path.exists():
        _error(f"Sprint-status not found: {sprint_path}")
        _error("Run sprint planning or `bmad-assist sprint generate` first.")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        summary = _build_sprint_status_summary(project_root, sprint_path)
    except BmadAssistError as e:
        _error(f"Failed to read sprint status: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    if format_lower == "json":
        sys.stdout.write(json.dumps(summary, indent=2))
        sys.stdout.write("\n")
        raise typer.Exit(code=EXIT_SUCCESS)

    story_counts = cast(dict[str, int], summary["story_counts"])
    epic_counts = cast(dict[str, int], summary["epic_counts"])
    retrospective_counts = cast(dict[str, int], summary["retrospective_counts"])
    next_action = cast(dict[str, str | None], summary["next"])
    risks = cast(list[str], summary["risks"])

    console.print()
    console.print("[bold]Sprint Status[/bold]")
    console.print(f"  Project: {summary['project']} ({summary.get('project_key') or 'NOKEY'})")
    console.print(f"  Tracking: {summary.get('tracking_system') or 'unknown'}")
    console.print(f"  Status file: {summary['sprint_path']}")
    console.print(
        "  Stories: "
        f"backlog {story_counts['backlog']}, "
        f"ready-for-dev {story_counts['ready-for-dev']}, "
        f"in-progress {story_counts['in-progress']}, "
        f"review {story_counts['review']}, "
        f"done {story_counts['done']}"
    )
    console.print(
        "  Epics: "
        f"backlog {epic_counts['backlog']}, "
        f"in-progress {epic_counts['in-progress']}, "
        f"done {epic_counts['done']}"
    )
    console.print(
        "  Retrospectives: "
        f"optional {retrospective_counts['optional']}, "
        f"done {retrospective_counts['done']}"
    )

    if next_action["workflow_id"]:
        console.print(
            "  Next Recommendation: "
            f"{next_action['workflow_id']} ({next_action['story_id']})"
        )
    else:
        console.print("  Next Recommendation: all implementation items complete")

    if risks:
        console.print()
        console.print("[bold yellow]Risks[/bold yellow]")
        for risk in risks:
            console.print(f"  - {risk}")

    raise typer.Exit(code=EXIT_SUCCESS)


@sprint_app.command("generate")
def sprint_generate(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    include_legacy: bool = typer.Option(
        False,
        "--include-legacy",
        help="Include legacy epics (normally auto-excluded if tracked in docs/sprint-artifacts/)",
    ),
) -> None:
    """Generate sprint-status entries from epic files.

    Scans epic files in docs/epics/ and _bmad-output/planning-artifacts/epics/,
    extracts story definitions, and merges with existing sprint-status.

    By default, auto-excludes epics tracked in docs/sprint-artifacts/sprint-status.yaml
    and skips epics in archive/ directories.
    """
    from bmad_assist.sprint import (
        ArtifactIndex,
        SprintStatus,
        generate_from_epics,
        parse_sprint_status,
        reconcile,
        write_sprint_status,
    )

    project_root, sprint_path, is_legacy_only = _setup_sprint_context(project)

    # Load existing sprint-status or create empty
    if sprint_path.exists():
        existing = parse_sprint_status(sprint_path)
        console.print(
            f"[dim]Loaded existing sprint-status with {len(existing.entries)} entries[/dim]"
        )
    else:
        existing = SprintStatus.empty(project=project_root.name)
        console.print("[dim]Creating new sprint-status file[/dim]")

    # Generate entries from epic files
    # Disable auto_exclude_legacy if only legacy location exists or user requests include
    effective_auto_exclude = not include_legacy and not is_legacy_only
    generated = generate_from_epics(project_root, auto_exclude_legacy=effective_auto_exclude)
    console.print(
        f"[dim]Generated {len(generated.entries)} entries from {generated.files_processed} files[/dim]"  # noqa: E501
    )

    if generated.duplicates_skipped > 0:
        _warning(f"{generated.duplicates_skipped} duplicate entries skipped")
    if generated.files_failed > 0:
        _warning(f"{generated.files_failed} files failed to parse")

    # Create empty artifact index (merge-only, no evidence inference)
    index = ArtifactIndex()

    # Reconcile (merge without evidence-based inference)
    reconciliation = reconcile(existing, generated, index)

    # Write result
    write_sprint_status(reconciliation.status, sprint_path, preserve_comments=True)

    # Summary
    console.print()
    _success(f"Generated {len(reconciliation.status.entries)} entries")
    console.print(f"  Output: {sprint_path}")
    console.print(f"  {reconciliation.summary()}")

    if verbose and reconciliation.changes:
        console.print()
        _display_changes_table(reconciliation.changes)


@sprint_app.command("repair")
def sprint_repair(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show changes without applying",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Apply repair without confirmation prompt",
    ),
    include_legacy: bool = typer.Option(
        False,
        "--include-legacy",
        help="Include legacy epics (normally auto-excluded if tracked in docs/sprint-artifacts/)",
    ),
) -> None:
    """Repair sprint-status from artifact evidence.

    Scans project artifacts (stories, code reviews, validations, retrospectives)
    and repairs sprint-status using evidence-based inference.

    By default, auto-excludes epics tracked in docs/sprint-artifacts/sprint-status.yaml
    and skips epics in archive/ directories.
    """
    from bmad_assist.sprint import (
        ArtifactIndex,
        RepairMode,
        SprintStatus,
        generate_from_epics,
        parse_sprint_status,
        reconcile,
        repair_sprint_status,
    )

    project_root, sprint_path, is_legacy_only = _setup_sprint_context(project)

    if dry_run:
        # Generate what would change without writing
        try:
            if sprint_path.exists():
                existing = parse_sprint_status(sprint_path)
            else:
                existing = SprintStatus.empty(project=project_root.name)

            # Disable auto_exclude_legacy if only legacy location exists
            effective_auto_exclude = not include_legacy and not is_legacy_only
            generated = generate_from_epics(
                project_root, auto_exclude_legacy=effective_auto_exclude
            )
            index = ArtifactIndex.scan(project_root)
            reconciliation = reconcile(existing, generated, index)
        except BmadAssistError as e:
            _error(f"Failed to analyze sprint status: {e}")
            raise typer.Exit(code=EXIT_ERROR) from None

        console.print("[yellow]Dry run - no changes written[/yellow]")
        console.print()
        console.print(f"Would apply {len(reconciliation.changes)} changes")
        console.print(f"  {reconciliation.summary()}")

        if verbose and reconciliation.changes:
            console.print()
            _display_changes_table(reconciliation.changes)
    else:
        # Check if we need confirmation before overwriting
        from bmad_assist.core.loop.interactive import is_non_interactive

        if sprint_path.exists() and not yes and not is_non_interactive():
            console.print(f"\n[yellow]Warning:[/yellow] This will overwrite {sprint_path}")
            console.print(
                "[dim]You can restore using BMAD workflow: /bmad:bmm:workflows:sprint-planning[/dim]"
            )  # noqa: E501
            if not typer.confirm("Continue?", default=False):
                console.print("[dim]Aborted.[/dim]")
                raise typer.Exit(code=EXIT_SUCCESS)

        # Actually perform repair
        # Note: repair_sprint_status also detects legacy-only internally, but we pass the flag
        effective_auto_exclude = not include_legacy and not is_legacy_only
        result = repair_sprint_status(
            project_root, RepairMode.SILENT, auto_exclude_legacy=effective_auto_exclude
        )

        if result.errors:
            _error(f"Repair completed with errors: {', '.join(result.errors)}")
            raise typer.Exit(code=EXIT_ERROR)

        console.print()
        _success(result.summary())
        console.print(f"  Output: {sprint_path}")

        if verbose:
            console.print(f"  Divergence: {result.divergence_pct:.1f}%")


@sprint_app.command("validate")
def sprint_validate(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output with evidence",
    ),
    format_: str = typer.Option(
        "plain",
        "--format",
        "-f",
        help="Output format: plain or json",
    ),
) -> None:
    """Validate sprint-status against artifact evidence.

    Compares sprint-status entries with evidence from code reviews,
    validations, and story files. Reports discrepancies with severity.

    Exit code 0 if no ERROR discrepancies (WARN only is OK).
    Exit code 1 if any ERROR discrepancies found.
    """
    import json

    from bmad_assist.core.state import get_state_path, load_state
    from bmad_assist.core.types import EpicId, epic_sort_key, parse_epic_id
    from bmad_assist.sprint import (
        ArtifactIndex,
        InferenceConfidence,
        infer_story_status_detailed,
        parse_sprint_status,
    )
    from bmad_assist.sprint.classifier import EntryType
    from bmad_assist.sprint.resume_validation import validate_resume_state

    project_root, sprint_path, _is_legacy_only = _setup_sprint_context(project)

    # Validate format
    format_lower = format_.lower()
    if format_lower not in ("plain", "json"):
        _error(f"Invalid --format: {format_}. Use 'plain' or 'json'.")
        raise typer.Exit(code=EXIT_ERROR)

    # Load sprint-status
    if not sprint_path.exists():
        _error(f"Sprint-status not found: {sprint_path}")
        raise typer.Exit(code=EXIT_ERROR)

    sprint_status = parse_sprint_status(sprint_path)

    # Scan artifacts
    index = ArtifactIndex.scan(project_root)
    loop_config = load_loop_config(project_root)
    require_test_review_for_done = "test_review" in loop_config.story

    # Compare each entry with inferred status
    discrepancies: list[Discrepancy] = []

    try:
        loaded_config = load_config_with_project(project_path=project_root)
        state_path = get_state_path(loaded_config, project_root=project_root)
    except ConfigError:
        state_path = project_root / ".bmad-assist" / "state.yaml"

    if state_path.exists():
        try:
            project_state = _load_lifecycle_project_state()

            lifecycle_stories_by_epic: dict[EpicId, list[str]] = {}
            for story in project_state.all_stories:
                epic_id = parse_epic_id(story.number.split(".", 1)[0])
                lifecycle_stories_by_epic.setdefault(epic_id, []).append(story.number)

            lifecycle_epic_list = sorted(lifecycle_stories_by_epic, key=epic_sort_key)

            def lifecycle_epic_stories_loader(epic_id: EpicId) -> list[str]:
                return list(lifecycle_stories_by_epic.get(epic_id, []))

            validate_resume_state(
                load_state(state_path),
                project_root,
                lifecycle_epic_list,
                lifecycle_epic_stories_loader,
                require_completion_artifacts_for_done=True,
                require_test_review_for_done=require_test_review_for_done,
            )
        except StateError as exc:
            discrepancies.append(
                Discrepancy(
                    key="loop-state",
                    sprint_status="current state",
                    inferred_status="invalid lifecycle progression",
                    severity=DiscrepancySeverity.ERROR,
                    reason=str(exc),
                    evidence=str(state_path),
                )
            )

    for key, entry in sprint_status.entries.items():
        # Skip non-story entries (epic meta, retrospectives)
        if entry.entry_type in (EntryType.EPIC_META, EntryType.RETROSPECTIVE):
            continue

        # Get inferred status
        result = infer_story_status_detailed(
            key,
            index,
            require_test_review_for_done=require_test_review_for_done,
        )
        inferred = result.status
        confidence = result.confidence

        # Compare
        if entry.status == inferred:
            continue  # No discrepancy

        # Determine severity based on rules from story
        severity = DiscrepancySeverity.WARN
        reason = ""
        evidence = ""

        # Build evidence description
        if result.evidence_sources:
            evidence = f"Found: {result.evidence_sources[0].name}"
            if len(result.evidence_sources) > 1:
                evidence += f" (+{len(result.evidence_sources) - 1} more)"
        else:
            evidence = "No artifacts found"

        # Classification rules from AC4
        if entry.status == "done" and not index.has_master_review(key):
            severity = DiscrepancySeverity.ERROR
            reason = "Sprint says 'done' but no master code review exists"
        elif entry.status == "backlog" and index.has_master_review(key):
            severity = DiscrepancySeverity.ERROR
            reason = "Sprint says 'backlog' but master code review exists (missed update)"
        elif entry.status == "in-progress" and index.has_any_review(key):
            severity = DiscrepancySeverity.WARN
            reason = "Sprint says 'in-progress' but code reviews exist (should be 'review')"
        elif entry.status == "review" and index.has_master_review(key):
            severity = DiscrepancySeverity.WARN
            reason = "Sprint says 'review' but master review exists (should be 'done')"
        elif confidence == InferenceConfidence.EXPLICIT:
            severity = DiscrepancySeverity.WARN
            reason = "Story file Status differs from sprint-status (possible manual override)"
        else:
            reason = f"Status mismatch: sprint={entry.status}, inferred={inferred}"

        discrepancies.append(
            Discrepancy(
                key=key,
                sprint_status=entry.status,
                inferred_status=inferred,
                severity=severity,
                reason=reason,
                evidence=evidence,
            )
        )

    # Count by severity
    error_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.ERROR)
    warn_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.WARN)

    # Output
    if format_lower == "json":
        output = {
            "success": error_count == 0,
            "exit_code": 1 if error_count > 0 else 0,
            "summary": {
                "total": len(discrepancies),
                "error_count": error_count,
                "warn_count": warn_count,
            },
            "discrepancies": [
                {
                    "key": d.key,
                    "sprint_status": d.sprint_status,
                    "inferred_status": d.inferred_status,
                    "severity": d.severity.value,
                    "reason": d.reason,
                    "evidence": d.evidence,
                }
                for d in discrepancies
            ],
        }
        # JSON to stdout
        sys.stdout.write(json.dumps(output, indent=2))
        sys.stdout.write("\n")
    else:
        # Plain output
        if not discrepancies:
            _success("No discrepancies found")
            console.print(f"  Validated {len(sprint_status.entries)} entries")
        else:
            console.print()
            console.print(f"Found {len(discrepancies)} discrepancies:")
            console.print(f"  [red]ERROR:[/red] {error_count}")
            console.print(f"  [yellow]WARN:[/yellow] {warn_count}")
            console.print()
            _display_discrepancies_table(discrepancies)

            if verbose:
                console.print()
                console.print("[bold]Evidence Details:[/bold]")
                for d in discrepancies:
                    console.print(f"  {d.key}: {d.evidence}")

    # Exit code based on ERROR count
    if error_count > 0:
        raise typer.Exit(code=EXIT_ERROR)
    raise typer.Exit(code=EXIT_SUCCESS)


@sprint_app.command("sync")
def sprint_sync(
    project: str = typer.Option(
        ".",
        "--project",
        "-p",
        help="Path to project directory",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """Sync sprint-status from state.yaml.

    One-way sync: state.yaml (runtime authority) -> sprint-status.yaml (BMAD view).
    Updates current story status based on phase and marks completed stories/epics.
    """
    from bmad_assist.core.state import get_state_path, load_state
    from bmad_assist.sprint import trigger_sync

    project_root, sprint_path, _is_legacy_only = _setup_sprint_context(project)

    # Try to load config for state path resolution
    try:
        loaded_config = load_config_with_project(project_path=project_root)
        state_path = get_state_path(loaded_config, project_root=project_root)
    except ConfigError:
        # No config - use default state path
        state_path = project_root / ".bmad-assist" / "state.yaml"

    # Check state.yaml exists
    if not state_path.exists():
        _error(f"state.yaml not found: {state_path}")
        _error("Cannot sync without state file. Run the development loop first.")
        raise typer.Exit(code=EXIT_ERROR)

    # Load state
    try:
        state = load_state(state_path)
    except Exception as e:
        _error(f"Failed to load state: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Perform sync
    try:
        result = trigger_sync(state, project_root)
    except Exception as e:
        _error(f"Sync failed: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    # Summary
    console.print()
    _success(result.summary())
    console.print(f"  Output: {sprint_path}")

    if verbose and result.skipped_keys:
        console.print()
        _warning(f"Skipped {len(result.skipped_keys)} missing keys:")
        for key in result.skipped_keys:
            console.print(f"    - {key}")
