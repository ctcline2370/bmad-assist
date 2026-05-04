"""Utilities for resolving durable retrospective artifacts."""

from __future__ import annotations

from pathlib import Path

from bmad_assist.core.paths import get_paths
from bmad_assist.core.types import EpicId


def _get_retrospective_search_roots(project_path: Path) -> list[Path]:
    """Return canonical and legacy retrospective roots in search order."""
    try:
        paths = get_paths()
        implementation_artifacts = paths.implementation_artifacts
        retrospectives_dir = paths.retrospectives_dir
    except RuntimeError:
        implementation_artifacts = project_path / "_bmad-output" / "implementation-artifacts"
        retrospectives_dir = implementation_artifacts / "retrospectives"

    roots: list[Path] = []
    for candidate in (retrospectives_dir, implementation_artifacts):
        if candidate not in roots:
            roots.append(candidate)
    return roots


def find_durable_retrospective_artifacts(epic_id: EpicId, project_path: Path) -> list[Path]:
    """Find durable retrospective artifacts for an epic.

    Search the canonical retrospectives folder first, then fall back to the
    legacy implementation-artifacts root used by older retrospectives.
    """
    retro_pattern = f"epic-{epic_id}-retro-*.md"

    for root in _get_retrospective_search_roots(project_path):
        matches = sorted(root.glob(retro_pattern))
        if matches:
            return matches

    return []


def has_durable_retrospective_artifact(epic_id: EpicId, project_path: Path) -> bool:
    """Return whether an epic has any durable retrospective artifact."""
    return bool(find_durable_retrospective_artifacts(epic_id, project_path))
