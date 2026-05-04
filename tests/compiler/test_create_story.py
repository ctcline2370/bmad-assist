"""Tests for the create-story workflow compiler.

Tests the CreateStoryCompiler class which orchestrates all compiler
pipeline components for the create-story workflow.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from bmad_assist.compiler.parser import parse_workflow
from bmad_assist.compiler.types import CompiledWorkflow, CompilerContext
from bmad_assist.compiler.workflows.create_story import CreateStoryCompiler
from bmad_assist.core.exceptions import CompilerError


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project structure for testing."""
    # Create docs directory structure
    docs = tmp_path / "docs"
    docs.mkdir()

    # Create sprint-artifacts
    sprint_artifacts = docs / "sprint-artifacts"
    sprint_artifacts.mkdir()

    # Create epics directory
    epics = docs / "epics"
    epics.mkdir()

    # Create BMAD workflow directory structure
    workflow_dir = tmp_path / "_bmad" / "bmm" / "workflows" / "4-implementation" / "create-story"
    workflow_dir.mkdir(parents=True)

    # Create workflow.yaml
    workflow_yaml = workflow_dir / "workflow.yaml"
    workflow_yaml.write_text("""name: create-story
description: "Create the next user story from epics+stories with enhanced context"
config_source: "{project-root}/_bmad/bmm/config.yaml"
template: "{installed_path}/template.md"
instructions: "{installed_path}/instructions.xml"
""")

    # Create instructions.xml
    instructions_xml = workflow_dir / "instructions.xml"
    instructions_xml.write_text("""<workflow>
  <step n="1" goal="Analyze epic">
    <action>Load epic file</action>
    <action>Extract story requirements</action>
  </step>
  <step n="2" goal="Create story">
    <action>Generate story content</action>
    <check if="story has acceptance criteria">
      <action>Validate acceptance criteria</action>
    </check>
  </step>
</workflow>
""")

    # Create template.md
    template_md = workflow_dir / "template.md"
    template_md.write_text("""# Story {{epic_num}}.{{story_num}}: {{story_title}}

Status: drafted

## Story

As a {{role}},
I want {{action}},
so that {{benefit}}.

## Acceptance Criteria

1. [Add acceptance criteria]

## Tasks / Subtasks

- [ ] Task 1 (AC: #)
""")

    # Create config.yaml
    config_dir = tmp_path / "_bmad" / "bmm"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_yaml = config_dir / "config.yaml"
    config_yaml.write_text(f"""project_name: test-project
output_folder: '{tmp_path}/docs'
sprint_artifacts: '{tmp_path}/docs/sprint-artifacts'
user_name: TestUser
communication_language: English
document_output_language: English
""")

    # Create project_context.md (required)
    project_context = docs / "project_context.md"
    project_context.write_text("""# Project Context for AI Agents

## Technology Stack

- Python 3.11+
- pytest for testing

## Critical Rules

- Type hints required on all functions
- Google-style docstrings
""")

    return tmp_path


def create_test_context(
    project: Path,
    epic_num: int = 10,
    story_num: int = 7,
    **extra_vars: Any,
) -> CompilerContext:
    """Create a CompilerContext for testing.

    Pre-loads workflow_ir from the workflow directory (normally done by core.compile_workflow).
    """
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


class TestWorkflowProperties:
    """Tests for CreateStoryCompiler properties."""

    def test_workflow_name(self) -> None:
        """workflow_name returns 'create-story'."""
        compiler = CreateStoryCompiler()
        assert compiler.workflow_name == "create-story"

    def test_get_required_files(self) -> None:
        """get_required_files returns expected patterns."""
        compiler = CreateStoryCompiler()
        patterns = compiler.get_required_files()

        assert "**/project_context.md" in patterns
        assert "**/architecture*.md" in patterns
        assert "**/prd*.md" in patterns
        assert "**/sprint-status.yaml" in patterns
        assert "**/epic*.md" in patterns

    def test_get_variables(self) -> None:
        """get_variables returns expected variable names."""
        compiler = CreateStoryCompiler()
        variables = compiler.get_variables()

        assert "epic_num" in variables
        assert "story_num" in variables
        assert "story_key" in variables
        assert "story_id" in variables
        assert "date" in variables


class TestValidateContext:
    """Tests for validate_context method."""

    def test_missing_project_root_raises(self, tmp_project: Path) -> None:
        """Missing project_root raises CompilerError."""
        context = CompilerContext(
            project_root=None,  # type: ignore
            output_folder=tmp_project / "docs",
            resolved_variables={"epic_num": 10, "story_num": 7},
        )
        compiler = CreateStoryCompiler()

        with pytest.raises(CompilerError, match="project_root"):
            compiler.validate_context(context)

    def test_missing_output_folder_raises(self, tmp_project: Path) -> None:
        """Missing output_folder raises CompilerError."""
        context = CompilerContext(
            project_root=tmp_project,
            output_folder=None,  # type: ignore
            resolved_variables={"epic_num": 10, "story_num": 7},
        )
        compiler = CreateStoryCompiler()

        with pytest.raises(CompilerError, match="output_folder"):
            compiler.validate_context(context)

    def test_missing_epic_num_raises(self, tmp_project: Path) -> None:
        """Missing epic_num raises CompilerError."""
        context = create_test_context(tmp_project, epic_num=None, story_num=7)  # type: ignore
        compiler = CreateStoryCompiler()

        with pytest.raises(CompilerError, match="epic_num"):
            compiler.validate_context(context)

    def test_missing_story_num_raises(self, tmp_project: Path) -> None:
        """Missing story_num raises CompilerError."""
        context = create_test_context(tmp_project, epic_num=10, story_num=None)  # type: ignore
        compiler = CreateStoryCompiler()

        with pytest.raises(CompilerError, match="story_num"):
            compiler.validate_context(context)

    def test_missing_workflow_dir_uses_bundled_fallback(self, tmp_project: Path) -> None:
        """Missing BMAD workflow uses bundled fallback.

        With bundled workflows, validate_context should NOT fail when
        the BMAD directory doesn't exist - it falls back to bundled.
        """
        # Remove workflow directory from BMAD
        workflow_dir = (
            tmp_project / "_bmad" / "bmm" / "workflows" / "4-implementation" / "create-story"
        )
        for f in workflow_dir.iterdir():
            f.unlink()
        workflow_dir.rmdir()

        context = create_test_context(tmp_project)
        compiler = CreateStoryCompiler()

        # Should NOT raise - uses bundled workflow as fallback
        compiler.validate_context(context)  # No exception expected

    def test_missing_project_context_raises(self, tmp_project: Path) -> None:
        """Missing project_context.md raises CompilerError."""
        # Remove project_context.md
        (tmp_project / "docs" / "project_context.md").unlink()

        context = create_test_context(tmp_project)
        compiler = CreateStoryCompiler()

        with pytest.raises(CompilerError, match="project_context.md"):
            compiler.validate_context(context)

    def test_valid_context_passes(self, tmp_project: Path) -> None:
        """Valid context passes validation."""
        context = create_test_context(tmp_project)
        compiler = CreateStoryCompiler()

        # Should not raise
        compiler.validate_context(context)


class TestCompile:
    """Tests for compile method."""

    def test_full_pipeline_compilation(self, tmp_project: Path) -> None:
        """Complete pipeline produces valid CompiledWorkflow."""
        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("""# Epic 10: Test Epic

## Story 10.7: Test Story

As a developer,
I want to test compilation,
so that I can verify the pipeline works.

### Acceptance Criteria

1. Pipeline completes successfully
""")

        # Create sprint-status.yaml
        sprint_status = tmp_project / "docs" / "sprint-artifacts" / "sprint-status.yaml"
        sprint_status.write_text("""development_status:
  10-7-test-story: backlog
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert result.workflow_name == "create-story"
        assert result.mission  # Non-empty
        assert result.instructions  # Non-empty (filtered)
        assert result.output_template  # Template loaded
        assert result.token_estimate > 0

    def test_compiled_workflow_structure(self, tmp_project: Path) -> None:
        """CompiledWorkflow has all required fields."""
        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("""# Epic 10

## Story 10.7: Test

Content here.
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert isinstance(result, CompiledWorkflow)
        assert result.workflow_name == "create-story"
        assert isinstance(result.mission, str)
        assert isinstance(result.context, str)
        assert isinstance(result.variables, dict)
        assert isinstance(result.instructions, str)
        assert isinstance(result.output_template, str)
        assert isinstance(result.token_estimate, int)

    def test_variables_resolved(self, tmp_project: Path) -> None:
        """Variables are resolved correctly."""
        # Create sprint-status.yaml
        sprint_status = tmp_project / "docs" / "sprint-artifacts" / "sprint-status.yaml"
        sprint_status.write_text("""development_status:
  10-7-variable-test: backlog
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert result.variables.get("epic_num") == 10
        assert result.variables.get("story_num") == 7
        assert result.variables.get("story_id") == "10.7"
        assert result.variables.get("story_key") == "10-7-variable-test"
        assert "date" in result.variables


class TestPreviousStoryInclusion:
    """Tests for previous story file discovery (AC3)."""

    def test_previous_story_included(self, tmp_project: Path) -> None:
        """Previous story is included in context when story_num > 1."""
        from bmad_assist.compiler.shared_utils import find_previous_stories

        # Create previous story file
        prev_story = tmp_project / "docs" / "sprint-artifacts" / "10-6-xml-output-generator.md"
        prev_story.write_text("# Previous story content\n\nThis is the previous story.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)

        prev_stories = find_previous_stories(context, context.resolved_variables)

        assert len(prev_stories) >= 1
        assert "10-6-" in str(prev_stories[0])

    def test_no_previous_story_for_first(self, tmp_project: Path) -> None:
        """Story 1 has no previous story."""
        from bmad_assist.compiler.shared_utils import find_previous_stories

        context = create_test_context(tmp_project, epic_num=10, story_num=1)

        prev_stories = find_previous_stories(context, context.resolved_variables)

        assert prev_stories == []

    def test_previous_story_missing_handled(self, tmp_project: Path) -> None:
        """Missing previous story is handled gracefully."""
        from bmad_assist.compiler.shared_utils import find_previous_stories

        # Don't create previous story file
        context = create_test_context(tmp_project, epic_num=10, story_num=7)

        prev_stories = find_previous_stories(context, context.resolved_variables)

        assert prev_stories == []  # Should return empty list, not raise


class TestEpicContextFiles:
    """Tests for epic context file discovery (AC5)."""

    def test_sharded_epics_includes_support_files_and_current_epic(self, tmp_project: Path) -> None:
        """Sharded epics include support files + current epic only."""
        epics_dir = tmp_project / "docs" / "epics"

        # Create support files
        (epics_dir / "index.md").write_text("# Epics Index")
        (epics_dir / "summary.md").write_text("# Summary")

        # Create multiple epic files
        (epics_dir / "epic-9-previous.md").write_text("# Epic 9")
        (epics_dir / "epic-10-current.md").write_text("# Epic 10: Current")
        (epics_dir / "epic-11-next.md").write_text("# Epic 11")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()

        files = compiler._find_epic_context_files(context, context.resolved_variables)
        filenames = [f.name for f in files]

        # Should include support files
        assert "index.md" in filenames
        assert "summary.md" in filenames
        # Should include only current epic
        assert "epic-10-current.md" in filenames
        # Should NOT include other epics
        assert "epic-9-previous.md" not in filenames
        assert "epic-11-next.md" not in filenames

    def test_single_file_epic_fallback(self, tmp_project: Path) -> None:
        """Falls back to single epic file when no epics/ directory."""
        # Remove epics directory, create single file in docs/
        epics_dir = tmp_project / "docs" / "epics"
        if epics_dir.exists():
            import shutil

            shutil.rmtree(epics_dir)

        epic_file = tmp_project / "docs" / "epic-10-test.md"
        epic_file.write_text("# Epic 10: Single File")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()

        files = compiler._find_epic_context_files(context, context.resolved_variables)

        assert len(files) == 1
        assert files[0] == epic_file

    def test_missing_epic_returns_empty(self, tmp_project: Path) -> None:
        """Missing epic file returns empty list."""
        context = create_test_context(tmp_project, epic_num=99, story_num=1)
        compiler = CreateStoryCompiler()

        files = compiler._find_epic_context_files(context, context.resolved_variables)

        assert files == []


class TestSprintStatusIntegration:
    """Tests for sprint-status.yaml integration (AC6)."""

    def test_sprint_status_title_extraction(self, tmp_project: Path) -> None:
        """Story title is extracted from sprint-status.yaml."""
        sprint_status = tmp_project / "docs" / "sprint-artifacts" / "sprint-status.yaml"
        sprint_status.write_text("""development_status:
  10-7-create-story-workflow-compiler: backlog
  10-8-next-story: backlog
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert result.variables.get("story_title") == "create-story-workflow-compiler"
        assert result.variables.get("story_key") == "10-7-create-story-workflow-compiler"

    def test_missing_sprint_status_fallback(self, tmp_project: Path) -> None:
        """Missing sprint-status.yaml uses fallback naming."""
        # Don't create sprint-status.yaml
        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        # Should use fallback
        assert "story-7" in result.variables.get("story_title", "")


class TestTemplateLoading:
    """Tests for template loading (AC4)."""

    def test_template_loaded(self, tmp_project: Path) -> None:
        """Template.md is loaded with placeholders preserved."""
        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert "{{epic_num}}" in result.output_template
        assert "{{story_num}}" in result.output_template
        assert "{{story_title}}" in result.output_template

    def test_empty_template_handled(self, tmp_project: Path) -> None:
        """Empty template file results in empty output_template."""
        # Overwrite template with empty content
        template_path = (
            tmp_project
            / "_bmad"
            / "bmm"
            / "workflows"
            / "4-implementation"
            / "create-story"
            / "template.md"
        )
        template_path.write_text("")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert result.output_template == ""


class TestRecencyBiasOrdering:
    """Tests for context files recency-bias ordering (AC3)."""

    def test_context_recency_bias_ordering(self, tmp_project: Path) -> None:
        """Context files are ordered: general -> specific."""
        # Create PRD and architecture files
        (tmp_project / "docs" / "prd.md").write_text("# PRD\n\nProduct requirements.")
        (tmp_project / "docs" / "architecture.md").write_text("# Architecture\n\nTech details.")

        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("# Epic 10\n\n## Story 10.7: Test\n\nContent.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        context_files = compiler._build_context_files(context, context.resolved_variables)
        paths = list(context_files.keys())

        # Find indices
        ctx_idx = next((i for i, p in enumerate(paths) if "project_context" in p), -1)
        arch_idx = next((i for i, p in enumerate(paths) if "architecture" in p), -1)
        epic_idx = next((i for i, p in enumerate(paths) if "epic" in p), -1)

        # Verify ordering: project_context < architecture < epic
        if ctx_idx >= 0 and arch_idx >= 0:
            assert ctx_idx < arch_idx
        if arch_idx >= 0 and epic_idx >= 0:
            assert arch_idx < epic_idx


class TestDeterminism:
    """Tests for deterministic compilation (NFR11)."""

    def test_compilation_deterministic(self, tmp_project: Path) -> None:
        """Same input produces identical output."""
        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("# Epic 10\n\n## Story 10.7: Test\n\nContent.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7, date="2025-01-01")
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result1 = compiler.compile(context)

        # Reset context for second run
        context2 = create_test_context(tmp_project, epic_num=10, story_num=7, date="2025-01-01")
        compiler2 = CreateStoryCompiler()
        compiler2.validate_context(context2)

        result2 = compiler2.compile(context2)

        assert result1.mission == result2.mission
        assert result1.instructions == result2.instructions
        assert result1.output_template == result2.output_template


class TestXMLOutput:
    """Tests for XML output structure (AC8)."""

    def test_xml_parseable(self, tmp_project: Path) -> None:
        """Generated XML is parseable by ElementTree."""
        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("# Epic 10\n\n## Story 10.7: Test\n\nContent.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        # XML in context field should be parseable
        root = ET.fromstring(result.context)
        assert root.tag == "compiled-workflow"

    def test_xml_has_required_sections(self, tmp_project: Path) -> None:
        """XML output has all required sections."""
        # Create epic file
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("# Epic 10\n\n## Story 10.7: Test\n\nContent.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        root = ET.fromstring(result.context)

        assert root.find("mission") is not None
        assert root.find("context") is not None
        assert root.find("variables") is not None
        assert root.find("instructions") is not None
        assert root.find("output-template") is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content_handled(self, tmp_project: Path) -> None:
        """Unicode content in files is handled correctly."""
        # Create project_context with Unicode
        (tmp_project / "docs" / "project_context.md").write_text(
            "# Project Context\n\nPolish: ąęćżźół\nEmoji: 🎉\nKanji: 日本語"
        )

        # Create epic with Unicode
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text("# Epic 10\n\n## Story 10.7: Tëst Störy\n\nÜñíçödé content.")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        # Should contain Unicode
        assert "ąęćżźół" in result.context or "emoji" in result.context.lower()

    def test_large_epic_file_handled(self, tmp_project: Path) -> None:
        """Large epic files are handled without errors."""
        # Create large epic file (>100KB)
        large_content = "# Epic 10\n\n## Story 10.7: Test\n\n" + "X" * (150 * 1024)
        epic_file = tmp_project / "docs" / "epics" / "epic-10.md"
        epic_file.write_text(large_content)

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        # Should not raise
        result = compiler.compile(context)
        assert result.token_estimate > 0

    def test_multiple_epic_files_first_selected(self, tmp_project: Path) -> None:
        """Multiple matching epic files - first alphabetically is selected (determinism)."""
        # Create multiple epic files for same epic
        (tmp_project / "docs" / "epics" / "epic-10-a.md").write_text("# Epic 10 A")
        (tmp_project / "docs" / "epics" / "epic-10-b.md").write_text("# Epic 10 B")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()

        files = compiler._find_epic_context_files(context, context.resolved_variables)
        epic_files = [f for f in files if f.name.startswith("epic-10")]

        # Should be the first alphabetically
        assert len(epic_files) >= 1
        assert "epic-10-a.md" in str(epic_files[0])

    def test_undefined_template_vars_preserved(self, tmp_project: Path) -> None:
        """Template variables not in resolved_variables are preserved as-is."""
        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        # {{role}}, {{action}}, {{benefit}} are not in resolved variables
        # They should be preserved for LLM to fill
        assert "{{role}}" in result.output_template
        assert "{{action}}" in result.output_template
        assert "{{benefit}}" in result.output_template


class TestPathSecurity:
    """Tests for path security (path traversal prevention)."""

    def test_path_traversal_in_template_rejected(self, tmp_project: Path) -> None:
        """Path traversal in template path is rejected."""
        # Modify workflow.yaml to have path traversal
        workflow_yaml = (
            tmp_project
            / "_bmad"
            / "bmm"
            / "workflows"
            / "4-implementation"
            / "create-story"
            / "workflow.yaml"
        )
        workflow_yaml.write_text("""name: create-story
description: "Test"
config_source: "{project-root}/_bmad/bmm/config.yaml"
template: "/etc/passwd"
instructions: "{installed_path}/instructions.xml"
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        with pytest.raises(CompilerError, match="Path security violation"):
            compiler.compile(context)


class TestMissionGeneration:
    """Tests for mission description generation."""

    def test_mission_includes_story_info(self, tmp_project: Path) -> None:
        """Mission includes story number and title."""
        sprint_status = tmp_project / "docs" / "sprint-artifacts" / "sprint-status.yaml"
        sprint_status.write_text("""development_status:
  10-7-workflow-compiler: backlog
""")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert "10.7" in result.mission
        assert "workflow-compiler" in result.mission

    def test_mission_without_title(self, tmp_project: Path) -> None:
        """Mission is generated even without story title."""
        # No sprint-status.yaml
        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        compiler = CreateStoryCompiler()
        compiler.validate_context(context)

        result = compiler.compile(context)

        assert "10.7" in result.mission
        assert result.mission  # Non-empty


class TestFileListExtraction:
    """Tests for extracting file paths from File List sections."""

    def test_extract_file_paths_basic(self) -> None:
        """Extracts file paths from basic File List section."""
        from bmad_assist.compiler.source_context import extract_file_paths_from_story

        story_content = """# Story 6.3

## Acceptance Criteria
Some text here.

## File List

**Modified:**
- `src/bmad_assist/core/loop.py` - Added functions
- `tests/core/test_loop.py` - Added 35 tests

## Change Log
"""
        paths = extract_file_paths_from_story(story_content)

        assert len(paths) == 2
        assert "src/bmad_assist/core/loop.py" in paths
        assert "tests/core/test_loop.py" in paths

    def test_extract_file_paths_with_h3(self) -> None:
        """Extracts file paths from ### File List section."""
        from bmad_assist.compiler.source_context import extract_file_paths_from_story

        story_content = """### File List

- `src/module/file.py` - Description
- `tests/test_file.py`
"""
        paths = extract_file_paths_from_story(story_content)

        assert len(paths) == 2
        assert "src/module/file.py" in paths
        assert "tests/test_file.py" in paths

    def test_extract_file_paths_no_section(self) -> None:
        """Returns empty list if no File List section."""
        from bmad_assist.compiler.source_context import extract_file_paths_from_story

        story_content = """# Story

## Implementation
Code here.
"""
        paths = extract_file_paths_from_story(story_content)

        assert paths == []

    def test_extract_file_paths_without_backticks(self) -> None:
        """Extracts paths without backticks."""
        from bmad_assist.compiler.source_context import extract_file_paths_from_story

        story_content = """## File List

- src/plain/path.ts - Plain path
* tests/another.py - Asterisk bullet
"""
        paths = extract_file_paths_from_story(story_content)

        assert "src/plain/path.ts" in paths
        assert "tests/another.py" in paths

    def test_extract_file_paths_various_extensions(self) -> None:
        """Extracts paths with various file extensions."""
        from bmad_assist.compiler.source_context import extract_file_paths_from_story

        story_content = """## File List

- `src/code.py`
- `src/component.tsx`
- `config/settings.yaml`
- `src/util.go`
- `src/main.rs`
"""
        paths = extract_file_paths_from_story(story_content)

        assert len(paths) == 5
        assert "src/code.py" in paths
        assert "src/component.tsx" in paths
        assert "config/settings.yaml" in paths
        assert "src/util.go" in paths
        assert "src/main.rs" in paths


class TestSourceFilesCollection:
    """Tests for collecting source files via SourceContextService."""

    def test_collect_source_files_basic(self, tmp_project: Path) -> None:
        """Collects source files from File List paths."""
        from bmad_assist.compiler.source_context import SourceContextService

        # Create source file
        src_dir = tmp_project / "src"
        src_dir.mkdir()
        source_file = src_dir / "module.py"
        source_file.write_text("def hello():\n    return 'world'")

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        service = SourceContextService(context, "create_story")

        result = service.collect_files(["src/module.py"], None)

        assert len(result) == 1
        assert "def hello():" in list(result.values())[0]

    def test_source_context_defaults(self, tmp_project: Path) -> None:
        """SourceContextService uses correct defaults for create_story."""
        from bmad_assist.compiler.source_context import SourceContextService

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        service = SourceContextService(context, "create_story")

        assert service.budget == 20000  # create_story default
        assert service.is_enabled()

    def test_collect_source_files_skips_missing(self, tmp_project: Path) -> None:
        """Skips files that don't exist."""
        from bmad_assist.compiler.source_context import SourceContextService

        context = create_test_context(tmp_project, epic_num=10, story_num=7)
        service = SourceContextService(context, "create_story")

        result = service.collect_files(["src/nonexistent.py"], None)

        assert len(result) == 0

    def test_create_story_prioritizes_live_code_and_tests_over_docs(
        self, tmp_project: Path
    ) -> None:
        """create_story keeps implementation context ahead of stale markdown noise."""
        from bmad_assist.compiler.source_context import SourceContextService

        src_dir = tmp_project / "src"
        tests_dir = tmp_project / "tests"
        docs_dir = tmp_project / "docs"
        artifacts_dir = tmp_project / "_bmad-output" / "implementation-artifacts"
        src_dir.mkdir()
        tests_dir.mkdir()
        docs_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(parents=True)

        (src_dir / "service.py").write_text("def handle() -> None:\n    return None\n")
        (tests_dir / "test_service.py").write_text(
            "def test_handle() -> None:\n    assert True\n"
        )
        (docs_dir / "notes.md").write_text("# Design Notes\n")
        (artifacts_dir / "7-2-current.md").write_text("# Prior Story Artifact\n")

        context = create_test_context(
            tmp_project,
            epic_num=7,
            story_num=3,
            story_id="7.3",
            story_key="7-3",
        )
        service = SourceContextService(context, "create_story")
        service.config = service.config.model_copy(
            update={
                "extraction": service.config.extraction.model_copy(
                    update={"max_files": 2}
                ),
                "scoring": service.config.scoring.model_copy(
                    update={"is_test_file": -5}
                ),
            }
        )

        result = service.collect_files(
            [
                "src/service.py",
                "tests/test_service.py",
                "docs/notes.md",
                "_bmad-output/implementation-artifacts/7-2-current.md",
            ],
            None,
        )

        ordered_paths = [
            Path(path).relative_to(tmp_project).as_posix() for path in result.keys()
        ]
        assert ordered_paths == [
            "src/service.py",
            "tests/test_service.py",
        ]
