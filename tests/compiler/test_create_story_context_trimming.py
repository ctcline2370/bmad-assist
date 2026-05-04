"""Focused tests for create-story context trimming behavior."""

from pathlib import Path
from typing import Any

import pytest

import bmad_assist.compiler.strategic_context as strategic_context_module
from bmad_assist.compiler.parser import parse_workflow
from bmad_assist.compiler.source_context import SourceContextService
from bmad_assist.compiler.strategic_context import StrategicContextService
from bmad_assist.compiler.types import CompilerContext
from bmad_assist.compiler.workflows import create_story as create_story_module
from bmad_assist.compiler.workflows.create_story import CreateStoryCompiler


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project structure for testing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sprint-artifacts").mkdir()
    (docs / "epics").mkdir()

    workflow_dir = tmp_path / "_bmad" / "bmm" / "workflows" / "4-implementation" / "create-story"
    workflow_dir.mkdir(parents=True)

    (workflow_dir / "workflow.yaml").write_text(
        """name: create-story
description: "Create the next user story from epics+stories with enhanced context"
config_source: "{project-root}/_bmad/bmm/config.yaml"
template: "{installed_path}/template.md"
instructions: "{installed_path}/instructions.xml"
""",
        encoding="utf-8",
    )
    (workflow_dir / "instructions.xml").write_text(
        """<workflow>
  <step n="1" goal="Analyze epic">
    <action>Load epic file</action>
  </step>
</workflow>
""",
        encoding="utf-8",
    )
    (workflow_dir / "template.md").write_text(
        """# Story {{epic_num}}.{{story_num}}: {{story_title}}""",
        encoding="utf-8",
    )

    config_dir = tmp_path / "_bmad" / "bmm"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        f"""project_name: test-project
output_folder: '{tmp_path}/docs'
sprint_artifacts: '{tmp_path}/docs/sprint-artifacts'
user_name: TestUser
communication_language: English
document_output_language: English
""",
        encoding="utf-8",
    )

    (docs / "project_context.md").write_text(
        """# Project Context for AI Agents""",
        encoding="utf-8",
    )
    return tmp_path


def create_test_context(
    project: Path,
    epic_num: int = 10,
    story_num: int = 7,
    **extra_vars: Any,
) -> CompilerContext:
    """Create a CompilerContext for testing."""
    resolved_vars = {
        "epic_num": epic_num,
        "story_num": story_num,
        **extra_vars,
    }
    workflow_dir = project / "_bmad" / "bmm" / "workflows" / "4-implementation" / "create-story"
    workflow_ir = parse_workflow(workflow_dir) if workflow_dir.exists() else None
    return CompilerContext(
        project_root=project,
        output_folder=project / "docs",
        resolved_variables=resolved_vars,
        workflow_ir=workflow_ir,
    )


class TestCreateStoryContextTrimming:
    """Validate continuity-preserving context pruning for create-story."""

    def test_build_context_files_trims_story_and_epic_context(
        self,
        tmp_project: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Previous story and epic context are pruned without losing handoff-critical details."""
        previous_story = tmp_project / "docs" / "sprint-artifacts" / "story-10.6.md"
        previous_story.write_text(
            """# Story 10.6: Previous Story

## Story

Carry forward the implementation context.

## Acceptance Criteria

- AC 1

## Tasks / Subtasks

- [x] Implemented the core flow

## Debug Transcript

Very long transcript that should not be carried forward.

## File List

- `src/Smartgistics.Infrastructure/Observability/Dashboards/DefaultDashboards.cs`
- `tests/Smartgistics.Infrastructure.UnitTests/Observability/Dashboards/DefaultDashboardsTests.cs`
- `tests/Smartgistics.Infrastructure.UnitTests/Observability/OpenTelemetryServiceCollectionExtensionsTests.cs`
- `tests/Smartgistics.Infrastructure.UnitTests/Observability/TenantMetricsPublisherTests.cs`

## QA Review

- Fix the edge-case validation branch before moving on.
""",
            encoding="utf-8",
        )

        epic_file = tmp_project / "docs" / "epics" / "epics.md"
        epic_file.write_text(
            """# Epic 9: Previous Epic

## Story 9.1: Old Story

Old content.

# Epic 10: Current Epic

## Story 10.7: Current Story

Current epic content.
""",
            encoding="utf-8",
        )

        captured_file_list_paths: list[str] = []

        monkeypatch.setattr(
            StrategicContextService,
            "collect",
            lambda self: {"strategic.md": "strategic context"},
        )
        monkeypatch.setattr(
            strategic_context_module,
            "load_antipatterns",
            lambda context, artifact_type: {"antipatterns.md": f"{artifact_type} antipatterns"},
        )
        monkeypatch.setattr(
            create_story_module,
            "find_previous_stories",
            lambda context, resolved, max_stories=1: [previous_story],
        )

        def fake_collect_files(self: SourceContextService, file_list_paths: list[str], diff_paths: list[str] | None) -> dict[str, str]:
            captured_file_list_paths.extend(file_list_paths)
            assert diff_paths is None
            return {
                "src/Smartgistics.Infrastructure/Observability/Dashboards/DefaultDashboards.cs": "dashboard source",
                "tests/Smartgistics.Infrastructure.UnitTests/Observability/Dashboards/DefaultDashboardsTests.cs": "dashboard tests",
            }

        monkeypatch.setattr(SourceContextService, "collect_files", fake_collect_files)

        compiler = CreateStoryCompiler()
        monkeypatch.setattr(
            compiler,
            "_find_epic_context_files",
            lambda context, resolved: [epic_file],
        )

        context = create_test_context(
            tmp_project,
            epic_num=10,
            story_num=7,
            story_title="Deliver Governed Workbooks and Dashboard Assets",
        )
        result = compiler._build_context_files(context, context.resolved_variables)

        assert list(result) == [
            "strategic.md",
            "antipatterns.md",
            str(previous_story),
            "src/Smartgistics.Infrastructure/Observability/Dashboards/DefaultDashboards.cs",
            "tests/Smartgistics.Infrastructure.UnitTests/Observability/Dashboards/DefaultDashboardsTests.cs",
            str(epic_file),
        ]

        trimmed_story = result[str(previous_story)]
        assert "## Story" in trimmed_story
        assert "## Acceptance Criteria" in trimmed_story
        assert "## Tasks / Subtasks" in trimmed_story
        assert "## File List" in trimmed_story
        assert "## QA Review" in trimmed_story
        assert "Debug Transcript" not in trimmed_story

        trimmed_epic = result[str(epic_file)]
        assert trimmed_epic.startswith("# Epic 10: Current Epic")
        assert "## Story 10.7: Current Story" in trimmed_epic
        assert "# Epic 9: Previous Epic" not in trimmed_epic

        assert captured_file_list_paths == [
            "src/Smartgistics.Infrastructure/Observability/Dashboards/DefaultDashboards.cs",
            "tests/Smartgistics.Infrastructure.UnitTests/Observability/Dashboards/DefaultDashboardsTests.cs",
        ]

    def test_select_previous_story_file_paths_uses_conservative_fallback_when_title_has_no_overlap(
        self,
        tmp_project: Path,
    ) -> None:
        """Fallback keeps continuity minimal when no path matches the story title."""
        compiler = CreateStoryCompiler()
        context = create_test_context(tmp_project, epic_num=10, story_num=8)

        selected = compiler._select_previous_story_file_paths(
            [
                "src/observability/defaults.py",
                "tests/observability/defaults_tests.py",
            ],
            {
                **context.resolved_variables,
                "story_title": "Publish Tenant Billing Exports",
            },
        )

        assert selected == ["src/observability/defaults.py"]
