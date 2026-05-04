"""Tests for the retrospective workflow compiler."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from bmad_assist.compiler.parser import parse_workflow
from bmad_assist.compiler.types import CompilerContext
from bmad_assist.core.exceptions import CompilerError


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project structure for retrospective compilation tests."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "project-context.md").write_text("# Project Context\n\nCore implementation rules.\n")

    planning_dir = docs / "planning-artifacts"
    planning_dir.mkdir()
    (planning_dir / "architecture.md").write_text("# Architecture\n\nArchitectural constraints.\n")
    (planning_dir / "prd.md").write_text("# PRD\n\nProduct requirements.\n")

    epics_dir = docs / "epics"
    epics_dir.mkdir()
    (epics_dir / "epic-7.md").write_text("# Epic 7\n\nEpic scope.\n")

    sprint_artifacts = docs / "sprint-artifacts"
    sprint_artifacts.mkdir()
    (sprint_artifacts / "sprint-status.yaml").write_text("current_story: 7.2\n")
    (sprint_artifacts / "7-1-alpha.md").write_text("# Story 7.1\n\nCompleted work.\n")
    (sprint_artifacts / "7-2-beta.md").write_text("# Story 7.2\n\nCompleted work.\n")

    retrospectives = docs / "retrospectives"
    retrospectives.mkdir()
    (retrospectives / "epic-6-retro-2026-04-19.md").write_text(
        "# Epic 6 Retrospective\n\nPrevious learnings.\n"
    )

    workflow_dir = tmp_path / "_bmad" / "bmm" / "workflows" / "4-implementation" / "retrospective"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "workflow.yaml").write_text(
        """name: retrospective
description: "Run epic retrospective."
config_source: "{project-root}/_bmad/bmm/config.yaml"
template: false
instructions: "{installed_path}/instructions.xml"
standalone: true
"""
    )
    (workflow_dir / "instructions.xml").write_text(
        """<workflow>
  <critical>YOU ARE THE RETROSPECTIVE AGENT</critical>
  <step n="1" goal="Review the completed epic">
    <action>Inspect epic artifacts and implementation evidence</action>
  </step>
</workflow>
"""
    )

    config_dir = tmp_path / "_bmad" / "bmm"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        f"""project_name: test-project
output_folder: '{docs}'
sprint_artifacts: '{sprint_artifacts}'
user_name: TestUser
communication_language: English
document_output_language: English
"""
    )

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main")
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0")

    return tmp_path


def create_test_context(
    project: Path,
    epic_num: int = 7,
    session_id: str = "test-session-123",
    **extra_vars: Any,
) -> CompilerContext:
    """Create a CompilerContext for retrospective tests."""
    resolved_vars = {
        "epic_num": epic_num,
        "session_id": session_id,
        **extra_vars,
    }
    workflow_dir = project / "_bmad" / "bmm" / "workflows" / "4-implementation" / "retrospective"
    workflow_ir = parse_workflow(workflow_dir) if workflow_dir.exists() else None
    return CompilerContext(
        project_root=project,
        output_folder=project / "docs",
        resolved_variables=resolved_vars,
        workflow_ir=workflow_ir,
    )


class _BudgetConfig:
    def __init__(self, budget: int) -> None:
        self._budget = budget

    def get_budget(self, workflow_name: str) -> int:
        assert workflow_name == "retrospective"
        return self._budget


class TestRetrospectiveCompiler:
    """Tests for RetrospectiveCompiler."""

    @pytest.mark.parametrize(
        ("configured_budget", "expected"),
        [
            (12000, 12000),
            (50000, 45000),
        ],
    )
    def test_context_cap_uses_configured_budget(
        self,
        configured_budget: int,
        expected: int,
    ) -> None:
        """Configured retrospective budgets are honored and clamped to the hard cap."""
        from bmad_assist.compiler.workflows.retrospective import (
            RETROSPECTIVE_CONTEXT_HARD_CAP_TOKENS,
            RetrospectiveCompiler,
        )

        assert RETROSPECTIVE_CONTEXT_HARD_CAP_TOKENS == 45000

        compiler = RetrospectiveCompiler()
        fake_config = SimpleNamespace(
            compiler=SimpleNamespace(
                source_context=SimpleNamespace(
                    budgets=_BudgetConfig(configured_budget),
                )
            )
        )

        with patch(
            "bmad_assist.core.config.loaders.get_config",
            return_value=fake_config,
        ):
            assert compiler._get_context_token_cap() == expected

    def test_prune_context_files_drops_optional_sections_before_story_files(
        self,
        tmp_project: Path,
    ) -> None:
        """Low-value retrospective context is pruned before story files are touched."""
        from bmad_assist.compiler.workflows.retrospective import RetrospectiveCompiler

        compiler = RetrospectiveCompiler()
        context = create_test_context(tmp_project)
        resolved = {"epic_num": 7}
        token_cap = 50000

        project_context_files = {"project-context": "project"}
        architecture_files = {"architecture": "architecture"}
        prd_files = {"prd": "prd"}
        epic_files = {"epic": "epic"}
        sprint_status_files = {"sprint-status": "status"}
        story_files = {
            "7-1-alpha.md": "story-a",
            "7-2-beta.md": "story-b",
        }
        tea_files = {"tea-context": "trace"}
        previous_retro_files = {"previous-retrospective": "retro"}

        with patch.object(
            compiler,
            "_estimate_prompt_tokens",
            side_effect=[100000, 90000, 80000, 70000, 40000],
        ):
            files, tokens, dropped_sections, dropped_story_files = compiler._prune_context_files(
                project_context_files,
                architecture_files,
                prd_files,
                epic_files,
                sprint_status_files,
                story_files,
                tea_files,
                previous_retro_files,
                token_cap,
                context=context,
                resolved=resolved,
                mission="mission",
                filtered_instructions="instructions",
            )

        assert tokens == 40000
        assert dropped_sections == [
            "previous-retrospective",
            "project-context",
            "architecture",
            "prd",
        ]
        assert dropped_story_files == []
        assert "previous-retrospective" not in files
        assert "project-context" not in files
        assert "architecture" not in files
        assert "prd" not in files
        assert "epic" in files
        assert "sprint-status" in files
        assert "tea-context" in files
        assert "7-1-alpha.md" in files
        assert "7-2-beta.md" in files

    def test_prune_context_files_drops_oldest_story_only_after_optional_sections(
        self,
        tmp_project: Path,
    ) -> None:
        """Story files are pruned only after optional sections and in insertion order."""
        from bmad_assist.compiler.workflows.retrospective import RetrospectiveCompiler

        compiler = RetrospectiveCompiler()
        context = create_test_context(tmp_project)
        resolved = {"epic_num": 7}
        token_cap = 50000

        story_files = {
            "7-1-alpha.md": "story-a",
            "7-2-beta.md": "story-b",
        }

        with patch.object(
            compiler,
            "_estimate_prompt_tokens",
            side_effect=[100000, 90000, 80000, 70000, 60000, 45000],
        ):
            files, tokens, dropped_sections, dropped_story_files = compiler._prune_context_files(
                {"project-context": "project"},
                {"architecture": "architecture"},
                {"prd": "prd"},
                {"epic": "epic"},
                {"sprint-status": "status"},
                story_files,
                {"tea-context": "trace"},
                {"previous-retrospective": "retro"},
                token_cap,
                context=context,
                resolved=resolved,
                mission="mission",
                filtered_instructions="instructions",
            )

        assert tokens == 45000
        assert dropped_sections == [
            "previous-retrospective",
            "project-context",
            "architecture",
            "prd",
        ]
        assert dropped_story_files == ["7-1-alpha.md"]
        assert "7-1-alpha.md" not in files
        assert "7-2-beta.md" in files
        assert "epic" in files
        assert "sprint-status" in files
        assert "tea-context" in files

    def test_compile_raises_when_prompt_still_exceeds_cap_after_pruning(
        self,
        tmp_project: Path,
    ) -> None:
        """Compilation fails closed when retrospective prompt size remains unsafe."""
        from bmad_assist.compiler.workflows.retrospective import RetrospectiveCompiler

        compiler = RetrospectiveCompiler()
        context = create_test_context(tmp_project)

        with (
            patch.object(compiler, "_get_context_token_cap", return_value=10),
            patch.object(compiler, "_estimate_prompt_tokens", return_value=99999),
            patch(
                "bmad_assist.compiler.workflows.retrospective.collect_tea_context",
                return_value={},
            ),
            pytest.raises(
                CompilerError,
                match="retrospective context still exceeds the operational cap after pruning",
            ),
        ):
            compiler.compile(context)

    def test_find_previous_retrospective_prefers_canonical_subdirectory(
        self,
        tmp_project: Path,
    ) -> None:
        """Canonical retrospectives directory wins when both locations contain legacy data."""
        from bmad_assist.compiler.workflows.retrospective import RetrospectiveCompiler

        compiler = RetrospectiveCompiler()
        context = create_test_context(tmp_project)
        legacy_root = tmp_project / "docs" / "epic-6-retro-2026-04-20.md"
        legacy_root.write_text("# Legacy Epic 6 Retrospective\n\nLegacy path.\n")

        result = compiler._find_previous_retrospective(context, 6)

        assert result == tmp_project / "docs" / "retrospectives" / "epic-6-retro-2026-04-19.md"

    def test_find_previous_retrospective_falls_back_to_legacy_root(
        self,
        tmp_project: Path,
    ) -> None:
        """Legacy top-level retrospectives remain readable for upgrade-safe compatibility."""
        from bmad_assist.compiler.workflows.retrospective import RetrospectiveCompiler

        compiler = RetrospectiveCompiler()
        context = create_test_context(tmp_project)
        canonical_path = tmp_project / "docs" / "retrospectives" / "epic-6-retro-2026-04-19.md"
        canonical_path.unlink()
        legacy_root = tmp_project / "docs" / "epic-6-retro-2026-04-20.md"
        legacy_root.write_text("# Legacy Epic 6 Retrospective\n\nLegacy path.\n")

        result = compiler._find_previous_retrospective(context, 6)

        assert result == legacy_root
