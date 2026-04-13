"""Compiler for the dev-story workflow.

This module implements the WorkflowCompiler protocol for the dev-story
workflow, producing standalone prompts for story implementation with
all necessary context embedded.

Public API:
    DevStoryCompiler: Workflow compiler class implementing WorkflowCompiler protocol
"""

import logging
from pathlib import Path
from typing import Any

from bmad_assist.compiler.filtering import filter_instructions
from bmad_assist.compiler.output import generate_output
from bmad_assist.compiler.shared_utils import (
    apply_post_process,
    context_snapshot,
    estimate_tokens,
    find_epic_file,
    find_file_in_output_folder,
    find_sprint_status_file,
    resolve_story_file,
    safe_read_file,
)
from bmad_assist.compiler.source_context import (
    SourceContextService,
    extract_file_paths_from_story,
)
from bmad_assist.compiler.strategic_context import StrategicContextService
from bmad_assist.compiler.types import CompiledWorkflow, CompilerContext, WorkflowIR
from bmad_assist.compiler.variable_utils import substitute_variables
from bmad_assist.compiler.variables import resolve_variables
from bmad_assist.core.exceptions import CompilerError
from bmad_assist.testarch.context import collect_tea_context, is_tea_context_enabled

logger = logging.getLogger(__name__)

DEV_STORY_CONTEXT_HARD_CAP_TOKENS = 28000


class DevStoryCompiler:
    """Compiler for the dev-story workflow.

    Implements the WorkflowCompiler protocol to compile the dev-story
    workflow into a standalone prompt. The dev-story workflow is an
    action-workflow (no template output), focused on implementing
    stories with all necessary context embedded.

    Context embedding follows recency-bias ordering:
    1. project_context.md (general)
    2. prd.md (full, no filtering)
    3. ux.md (optional)
    4. architecture.md (technical)
    5. epic file (current epic)
    6. source files from File List (with token budget)
    7. story file (LAST - closest to instructions)

    """

    @property
    def workflow_name(self) -> str:
        """Unique workflow identifier."""
        return "dev-story"

    def get_required_files(self) -> list[str]:
        """Return list of required file glob patterns.

        Returns:
            Glob patterns for files needed by dev-story workflow.

        """
        return [
            "**/project_context.md",
            "**/project-context.md",
            "**/architecture*.md",
            "**/prd*.md",
            "**/ux*.md",
            "**/sprint-status.yaml",
            "**/epic*.md",
        ]

    def get_variables(self) -> dict[str, Any]:
        """Return workflow-specific variables to resolve.

        Returns:
            Variables needed for dev-story compilation.

        """
        return {
            "epic_num": None,
            "story_num": None,
            "story_key": None,
            "story_id": None,
            "story_file": None,
            "story_title": None,
            "date": None,
        }

    def get_workflow_dir(self, context: CompilerContext) -> Path:
        """Return the workflow directory for this compiler.

        Args:
            context: The compilation context with project paths.

        Returns:
            Path to the workflow directory containing workflow.yaml.

        Raises:
            CompilerError: If workflow directory not found.

        """
        from bmad_assist.compiler.workflow_discovery import (
            discover_workflow_dir,
            get_workflow_not_found_message,
        )

        workflow_dir = discover_workflow_dir(self.workflow_name, context.project_root)
        if workflow_dir is None:
            raise CompilerError(
                get_workflow_not_found_message(self.workflow_name, context.project_root)
            )
        return workflow_dir

    def validate_context(self, context: CompilerContext) -> None:
        """Validate context before compilation.

        Args:
            context: The compilation context to validate.

        Raises:
            CompilerError: If required context is missing.

        """
        if context.project_root is None:
            raise CompilerError("project_root is required in context")
        if context.output_folder is None:
            raise CompilerError("output_folder is required in context")

        epic_num = context.resolved_variables.get("epic_num")
        story_num = context.resolved_variables.get("story_num")

        if epic_num is None:
            raise CompilerError(
                "epic_num is required for dev-story compilation.\n"
                "  Suggestion: Provide epic_num via invocation params or ensure "
                "sprint-status.yaml has a ready-for-dev story"
            )
        if story_num is None:
            raise CompilerError(
                "story_num is required for dev-story compilation.\n"
                "  Suggestion: Provide story_num via invocation params or ensure "
                "sprint-status.yaml has a ready-for-dev story"
            )

        # Workflow directory is validated by get_workflow_dir via discovery
        workflow_dir = self.get_workflow_dir(context)
        if not workflow_dir.exists():
            raise CompilerError(
                f"Workflow directory not found: {workflow_dir}\n"
                f"  Why it's needed: Contains workflow.yaml and instructions.xml\n"
                f"  How to fix: Reinstall bmad-assist or ensure BMAD is properly installed"
            )

        story_path, _, _ = resolve_story_file(context, epic_num, story_num)
        if story_path is None:
            raise CompilerError(
                f"Story file not found for {epic_num}-{story_num}-*.md\n"
                f"  Expected pattern: docs/sprint-artifacts/{epic_num}-{story_num}-*.md\n"
                f"  Suggestion: Run 'create-story' workflow first to create the story"
            )

    def compile(self, context: CompilerContext) -> CompiledWorkflow:
        """Compile dev-story workflow with given context.

        Executes the full compilation pipeline:
        1. Use pre-loaded workflow_ir from context
        2. Resolve variables with sprint-status lookup
        3. Build context files with recency-bias ordering (story LAST)
        4. Filter instructions
        5. Generate XML output

        Args:
            context: The compilation context with:
                - workflow_ir: Pre-loaded WorkflowIR
                - patch_path: Path to patch file (for post_process)

        Returns:
            CompiledWorkflow ready for output.

        Raises:
            CompilerError: If compilation fails at any stage.

        """
        workflow_ir = context.workflow_ir
        if workflow_ir is None:
            raise CompilerError(
                "workflow_ir not set in context. This is a bug - core.py should have loaded it."
            )

        workflow_dir = self.get_workflow_dir(context)

        with context_snapshot(context):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Using workflow from %s", workflow_dir)

            invocation_params = {
                k: v
                for k, v in context.resolved_variables.items()
                if k in ("epic_num", "story_num", "story_title", "date")
            }

            sprint_status_path = find_sprint_status_file(context)

            epic_num = invocation_params.get("epic_num")
            epics_path = find_epic_file(context, epic_num) if epic_num else None

            resolved = resolve_variables(context, invocation_params, sprint_status_path, epics_path)

            story_path, story_key, _ = resolve_story_file(
                context,
                resolved.get("epic_num"),
                resolved.get("story_num"),
            )
            if story_path:
                resolved["story_file"] = str(story_path)
            if story_key:
                resolved["story_key"] = story_key

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Resolved %d variables", len(resolved))

            context_files = self._build_context_files(context, resolved)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Built context with %d files", len(context_files))

            filtered_instructions = filter_instructions(workflow_ir)
            filtered_instructions = substitute_variables(filtered_instructions, resolved)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Filtered instructions: %d bytes", len(filtered_instructions))

            mission = self._build_mission(workflow_ir, resolved)

            compiled = CompiledWorkflow(
                workflow_name=self.workflow_name,
                mission=mission,
                context="",
                variables=resolved,
                instructions=filtered_instructions,
                output_template="",  # action-workflow, no template
                token_estimate=0,
            )

            result = generate_output(
                compiled,
                project_root=context.project_root,
                context_files=context_files,
                links_only=context.links_only,
            )

            final_xml = apply_post_process(result.xml, context)

            return CompiledWorkflow(
                workflow_name=self.workflow_name,
                mission=mission,
                context=final_xml,
                variables=resolved,
                instructions=filtered_instructions,
                output_template="",
                token_estimate=result.token_estimate,
            )

    def _build_context_files(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
    ) -> dict[str, str]:
        """Build context files dict with recency-bias ordering.

        Files are ordered from general (early) to specific (late):
        1. Strategic docs via StrategicContextService (project-context, PRD/UX/Arch)
        1b. code-antipatterns.md (if exists - general guidance)
        2. epic file (current epic)
        3. ATDD checklist (if exists)
        4. source files from File List (with token budget)
        5. story file (LAST - closest to instructions)

        Args:
            context: Compilation context with paths.
            resolved: Resolved variables containing epic_num and story_num.

        Returns:
            Dictionary mapping file paths to content, ordered by recency-bias.

        """
        project_root = context.project_root
        strategic_files: dict[str, str] = {}
        antipattern_files: dict[str, str] = {}
        epic_files: dict[str, str] = {}
        tea_test_design_files: dict[str, str] = {}
        tea_other_files: dict[str, str] = {}
        tea_atdd_files: dict[str, str] = {}
        source_files: dict[str, str] = {}
        story_files: dict[str, str] = {}

        # 1. Strategic docs (project-context, PRD, UX, Architecture) via service
        # Default config for dev_story: project-context only (other docs rarely cited)
        strategic_service = StrategicContextService(context, "dev_story")
        strategic_files.update(strategic_service.collect())

        # 1b. Include code antipatterns from previous code reviews (if exists)
        from bmad_assist.compiler.strategic_context import load_antipatterns

        antipattern_files.update(load_antipatterns(context, "code"))

        # 2. Epic file (current epic)
        epic_num = resolved.get("epic_num")
        if epic_num:
            epic_path = find_epic_file(context, epic_num)
            if epic_path:
                content = safe_read_file(epic_path, project_root)
                if content:
                    epic_files[str(epic_path)] = content

        # 3. TEA Context (test-design, ATDD checklists) via TEAContextService
        # F4 Fix: Backward compatible - uses TEA config if available, falls back to legacy
        story_id = resolved.get("story_id")

        if is_tea_context_enabled(context):
            # New TEA context loader (includes test-design + ATDD)
            tea_test_design_files, tea_other_files, tea_atdd_files = self._split_tea_files(
                collect_tea_context(context, "dev_story", resolved)
            )
        elif story_id:
            # F4 BACKWARD COMPATIBILITY: Legacy hardcoded ATDD for projects without TEA config
            # This preserves existing behavior until project migrates to TEA config
            atdd_pattern = f"*atdd-checklist*{story_id}*.md"
            atdd_path = find_file_in_output_folder(context, atdd_pattern)
            if atdd_path:
                content = safe_read_file(atdd_path, project_root)
                if content:
                    tea_atdd_files[str(atdd_path)] = content
                    logger.info("ATDD checklist loaded (legacy mode): %s", atdd_path)

        # 4. Source files from story's File List using SourceContextService
        story_path_str = resolved.get("story_file")
        if story_path_str:
            story_path = Path(story_path_str)
            story_content = safe_read_file(story_path, project_root)
            file_list_paths: list[str] = []
            if story_content:
                file_list_paths = extract_file_paths_from_story(story_content)

            service = SourceContextService(context, "dev_story")
            source_files.update(service.collect_files(file_list_paths, None))

        # 5. Story file (LAST - closest to instructions per recency-bias)
        if story_path_str:
            story_path = Path(story_path_str)
            content = safe_read_file(story_path, project_root)
            if content:
                story_files[str(story_path)] = content

        files = self._merge_context_sections(
            strategic_files,
            antipattern_files,
            epic_files,
            tea_test_design_files,
            tea_other_files,
            tea_atdd_files,
            source_files,
            story_files,
        )

        total_tokens = self._estimate_context_tokens(files)
        if total_tokens <= DEV_STORY_CONTEXT_HARD_CAP_TOKENS:
            return files

        logger.warning(
            "Dev story context exceeded hard cap: %d tokens (cap=%d). Pruning optional context.",
            total_tokens,
            DEV_STORY_CONTEXT_HARD_CAP_TOKENS,
        )
        pruned_files, pruned_tokens, dropped_sections = self._prune_context_files(
            strategic_files,
            antipattern_files,
            epic_files,
            tea_test_design_files,
            tea_other_files,
            tea_atdd_files,
            source_files,
            story_files,
        )
        if pruned_tokens > DEV_STORY_CONTEXT_HARD_CAP_TOKENS:
            raise CompilerError(
                "dev-story context still exceeds the operational cap after pruning "
                f"optional sections ({', '.join(dropped_sections) or 'none'}): "
                f"{pruned_tokens} tokens > {DEV_STORY_CONTEXT_HARD_CAP_TOKENS}. "
                "Reduce source context or trim story-linked files before invoking the provider."
            )

        logger.info(
            "Dev story context pruned to %d tokens after dropping optional sections: %s",
            pruned_tokens,
            ", ".join(dropped_sections),
        )
        return pruned_files

    def _merge_context_sections(self, *sections: dict[str, str]) -> dict[str, str]:
        """Merge ordered context sections while preserving recency-bias."""
        merged: dict[str, str] = {}
        for section in sections:
            merged.update(section)
        return merged

    def _estimate_context_tokens(self, files: dict[str, str]) -> int:
        """Estimate total token count for assembled context files."""
        return sum(estimate_tokens(content) for content in files.values())

    def _prune_context_files(
        self,
        strategic_files: dict[str, str],
        antipattern_files: dict[str, str],
        epic_files: dict[str, str],
        tea_test_design_files: dict[str, str],
        tea_other_files: dict[str, str],
        tea_atdd_files: dict[str, str],
        source_files: dict[str, str],
        story_files: dict[str, str],
    ) -> tuple[dict[str, str], int, list[str]]:
        """Prune low-value duplicate context before provider invocation."""
        optional_sections: list[tuple[str, dict[str, str]]] = [
            ("antipatterns", antipattern_files),
            ("tea-other", tea_other_files),
            ("tea-test-design", tea_test_design_files),
        ]
        dropped_sections: list[str] = []

        candidate_files = self._merge_context_sections(
            strategic_files,
            antipattern_files,
            epic_files,
            tea_test_design_files,
            tea_other_files,
            tea_atdd_files,
            source_files,
            story_files,
        )
        candidate_tokens = self._estimate_context_tokens(candidate_files)
        if candidate_tokens <= DEV_STORY_CONTEXT_HARD_CAP_TOKENS:
            return candidate_files, candidate_tokens, dropped_sections

        for section_name, _ in optional_sections:
            dropped_sections.append(section_name)
            candidate_files = self._merge_context_sections(
                strategic_files,
                {} if "antipatterns" in dropped_sections else antipattern_files,
                epic_files,
                {} if "tea-test-design" in dropped_sections else tea_test_design_files,
                {} if "tea-other" in dropped_sections else tea_other_files,
                tea_atdd_files,
                source_files,
                story_files,
            )
            candidate_tokens = self._estimate_context_tokens(candidate_files)
            if candidate_tokens <= DEV_STORY_CONTEXT_HARD_CAP_TOKENS:
                return candidate_files, candidate_tokens, dropped_sections

        return candidate_files, candidate_tokens, dropped_sections

    def _split_tea_files(
        self,
        tea_files: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Split TEA files into test-design, other, and ATDD buckets.

        This keeps ATDD story evidence closest to the prompt while allowing
        broader TEA context to be pruned first under pressure.
        """
        test_design_files: dict[str, str] = {}
        other_files: dict[str, str] = {}
        atdd_files: dict[str, str] = {}

        for path, content in tea_files.items():
            lower_path = path.lower()
            if "atdd-checklist" in lower_path:
                atdd_files[path] = content
            elif "test-design" in lower_path or "test_design" in lower_path or "test-plan" in lower_path:
                test_design_files[path] = content
            else:
                other_files[path] = content

        return test_design_files, other_files, atdd_files

    def _build_mission(
        self,
        workflow_ir: WorkflowIR,
        resolved: dict[str, Any],
    ) -> str:
        """Build mission description for compiled workflow.

        Args:
            workflow_ir: Workflow IR with description.
            resolved: Resolved variables.

        Returns:
            Mission description string.

        """
        base_description = workflow_ir.raw_config.get(
            "description", "Execute a story by implementing tasks/subtasks, writing tests"
        )

        epic_num = resolved.get("epic_num", "?")
        story_num = resolved.get("story_num", "?")
        story_title = resolved.get("story_title", "")

        if story_title:
            mission = (
                f"{base_description}\n\n"
                f"Target: Story {epic_num}.{story_num} - {story_title}\n"
                f"Implement all tasks and subtasks following TDD methodology."
            )
        else:
            mission = (
                f"{base_description}\n\n"
                f"Target: Story {epic_num}.{story_num}\n"
                f"Implement all tasks and subtasks following TDD methodology."
            )

        return mission
