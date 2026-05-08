"""Compiler for the retrospective workflow.

This module implements the WorkflowCompiler protocol for the retrospective
workflow, producing standalone prompts for epic retrospectives with
all necessary context embedded including sprint-status, stories, and
previous retrospective if available.

Public API:
    RetrospectiveCompiler: Workflow compiler class implementing WorkflowCompiler protocol
"""

import logging
import re
from pathlib import Path
from typing import Any

from bmad_assist.compiler.filtering import filter_instructions
from bmad_assist.compiler.output import generate_output
from bmad_assist.compiler.shared_utils import (
    apply_post_process,
    context_snapshot,
    estimate_tokens,
    find_file_in_planning_dir,
    find_project_context_file,
    find_sprint_status_file,
    get_stories_dir,
    safe_read_file,
)
from bmad_assist.compiler.types import CompiledWorkflow, CompilerContext, WorkflowIR
from bmad_assist.compiler.variable_utils import substitute_variables
from bmad_assist.compiler.variables import resolve_variables
from bmad_assist.core.exceptions import CompilerError, ConfigError
from bmad_assist.testarch.context import collect_tea_context

logger = logging.getLogger(__name__)

# Workflow path relative to project root
_WORKFLOW_RELATIVE_PATH = "_bmad/bmm/workflows/4-implementation/retrospective"

# Pattern for story files
_STORY_FILE_PATTERN = re.compile(r"^(\d+)-(\d+)-.+\.md$")

RETROSPECTIVE_CONTEXT_HARD_CAP_TOKENS = 45000
_RETROSPECTIVE_COMPACTED_SECTION_TARGET_TOKENS = 2500
_RETROSPECTIVE_COMPACTED_SECTION_MIN_TOKENS = 400
_RETROSPECTIVE_COMPACTION_NOTICE = (
    "\n\n<!-- Retrospective context compacted: source content exceeded the "
    "operational token budget. Read the referenced source file directly if "
    "more detail is required. -->"
)
_RETROSPECTIVE_LEGACY_SCOPE_SENTENCE = (
    "Your ONLY output should be a structured retrospective report wrapped in "
    "extraction markers. Do NOT use party-mode dialogue. Do NOT wait for user "
    "input. Do NOT write files. Generate the complete retrospective in ONE response."
)
_RETROSPECTIVE_FEEDBACK_SCOPE_SENTENCE = (
    "You MUST complete the upstream document feedback loop before producing the "
    "final structured retrospective report wrapped in extraction markers. Do NOT "
    "use party-mode dialogue. Do NOT wait for user input. Do NOT save the "
    "retrospective report file directly; BMAD-ASSIST persists the final report. "
    "You MAY edit upstream source documents and approved governance-candidate "
    "handoff files when retrospective findings require it."
)
_RETROSPECTIVE_LEGACY_SAVE_SENTENCE = "Do NOT save files - output to stdout only."
_RETROSPECTIVE_FEEDBACK_SAVE_SENTENCE = (
    "Do NOT save the retrospective report file directly; BMAD-ASSIST persists "
    "the final report. Upstream source-document or approved governance-candidate "
    "updates are allowed when required by the feedback loop."
)
_RETROSPECTIVE_FEEDBACK_LOOP_REQUIREMENTS = """

<critical>RETROSPECTIVE FEEDBACK LOOP REQUIREMENTS:
Before final output, classify every meaningful retrospective finding as one of:
- updated
- governance-candidate
- human-gated
- no-change

Update upstream PRD, architecture, epic, story, backlog, and readiness docs first when
findings change requirements, NFRs, assumptions, scope, success metrics, acceptance
criteria, workflow behavior, rollout/support/operational assumptions, tenant isolation,
authorization, API behavior, or AI automation.

Governance canonical changes are draft-first only under
.output/adr-control-plane/bmad-governance-candidates/. Do not write canonical
governance files directly.

The final retrospective report MUST include a markdown table under the exact heading:
## Document Feedback Matrix

The table MUST use these columns:
Finding | Disposition | Evidence | Owner | Blocked Downstream Work

Disposition evidence rules:
- updated rows must list changed upstream source-document paths.
- governance-candidate rows must list changed candidate handoff paths under
  .output/adr-control-plane/bmad-governance-candidates/.
- human-gated rows must include owner, evidence, and blocked downstream work.
- no-change rows must include a concrete rationale explaining why no upstream change
  was needed.

BMAD-ASSIST validates this matrix and fails the retrospective if it is missing,
unsupported, or unevidenced.</critical>
"""


class RetrospectiveCompiler:
    """Compiler for the retrospective workflow.

    Implements the WorkflowCompiler protocol to compile the retrospective
    workflow into a standalone prompt. The retrospective workflow is an
    action-workflow (no template output), focused on epic completion review.

    Context embedding follows recency-bias ordering:
    1. project_context.md (general)
    2. architecture.md (technical constraints)
    3. prd.md (product requirements)
    4. epic file (what was built)
    5. sprint-status.yaml (completion status)
    6. story files for epic (implementation details)
    7. previous retrospective (LAST - continuity)

    """

    @property
    def workflow_name(self) -> str:
        """Unique workflow identifier."""
        return "retrospective"

    def get_required_files(self) -> list[str]:
        """Return list of required file glob patterns.

        Returns:
            Glob patterns for files needed by retrospective workflow.

        """
        return [
            "**/project_context.md",
            "**/project-context.md",
            "**/architecture*.md",
            "**/prd*.md",
            "**/epic*.md",
            "**/sprint-status.yaml",
        ]

    def get_variables(self) -> dict[str, Any]:
        """Return workflow-specific variables to resolve.

        Returns:
            Variables needed for retrospective compilation.

        """
        return {
            "epic_num": None,
            "prev_epic_num": None,
            "next_epic_num": None,
            "epic_title": None,
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

        if epic_num is None:
            raise CompilerError(
                "epic_num is required for retrospective compilation.\n"
                "  Suggestion: Provide epic_num via invocation params"
            )

        # Workflow directory is validated by get_workflow_dir via discovery
        workflow_dir = self.get_workflow_dir(context)
        if not workflow_dir.exists():
            raise CompilerError(
                f"Workflow directory not found: {workflow_dir}\n"
                f"  Why it's needed: Contains workflow.yaml and instructions.md\n"
                f"  How to fix: Reinstall bmad-assist or ensure BMAD is properly installed"
            )

    def compile(self, context: CompilerContext) -> CompiledWorkflow:
        """Compile retrospective workflow with given context.

        Executes the full compilation pipeline:
        1. Use pre-loaded workflow_ir from context
        2. Resolve variables
        3. Build context files with recency-bias ordering
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
                k: v for k, v in context.resolved_variables.items() if k in ("epic_num", "date")
            }

            sprint_status_path = find_sprint_status_file(context)

            resolved = resolve_variables(context, invocation_params, sprint_status_path, None)

            # Calculate prev/next epic numbers
            epic_num = resolved.get("epic_num")
            if epic_num is not None:
                try:
                    epic_int = int(epic_num)
                    resolved["prev_epic_num"] = epic_int - 1 if epic_int > 1 else None
                    resolved["next_epic_num"] = epic_int + 1
                except (ValueError, TypeError):
                    # Non-numeric epic (like "testarch")
                    resolved["prev_epic_num"] = None
                    resolved["next_epic_num"] = None

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Resolved %d variables", len(resolved))

            filtered_instructions = filter_instructions(workflow_ir)
            filtered_instructions = substitute_variables(filtered_instructions, resolved)
            filtered_instructions = self._inject_feedback_loop_requirements(
                filtered_instructions
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Filtered instructions: %d bytes", len(filtered_instructions))

            mission = self._build_mission(workflow_ir, resolved)
            context_files = self._build_context_files(
                context,
                resolved,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Built context with %d files", len(context_files))

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
                token_estimate=estimate_tokens(final_xml),
            )

    def _inject_feedback_loop_requirements(self, instructions: str) -> str:
        """Require upstream document feedback evidence in retrospective prompts."""
        hardened = instructions.replace(
            _RETROSPECTIVE_LEGACY_SCOPE_SENTENCE,
            _RETROSPECTIVE_FEEDBACK_SCOPE_SENTENCE,
        )
        hardened = hardened.replace(
            _RETROSPECTIVE_LEGACY_SAVE_SENTENCE,
            _RETROSPECTIVE_FEEDBACK_SAVE_SENTENCE,
        )
        if "RETROSPECTIVE FEEDBACK LOOP REQUIREMENTS" in hardened:
            return hardened
        return hardened.rstrip() + _RETROSPECTIVE_FEEDBACK_LOOP_REQUIREMENTS

    def _build_context_files(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
        *,
        mission: str | None = None,
        filtered_instructions: str | None = None,
    ) -> dict[str, str]:
        """Build context files dict with recency-bias ordering.

        Files are ordered from general (early) to specific (late):
        1. project_context.md (general)
        2. architecture.md (technical constraints)
        3. prd.md (product requirements)
        4. epic file (what was built)
        5. sprint-status.yaml (completion status)
        6. story files for epic (implementation details)
        7. previous retrospective (LAST - continuity)

        Args:
            context: Compilation context with paths.
            resolved: Resolved variables containing epic_num.
            mission: Compiled mission text when prompt-aware token estimation is needed.
            filtered_instructions: Filtered workflow instructions for prompt-aware sizing.

        Returns:
            Dictionary mapping file paths to content, ordered by recency-bias.

        """
        (
            project_context_files,
            architecture_files,
            prd_files,
            epic_files,
            sprint_status_files,
            story_files,
            tea_files,
            previous_retro_files,
        ) = self._build_context_sections(context, resolved)

        files = self._merge_context_sections(
            project_context_files,
            architecture_files,
            prd_files,
            epic_files,
            sprint_status_files,
            story_files,
            tea_files,
            previous_retro_files,
        )

        token_cap = self._get_context_token_cap()
        total_tokens = self._estimate_prompt_tokens(
            context,
            resolved,
            files,
            mission=mission,
            filtered_instructions=filtered_instructions,
        )
        if total_tokens <= token_cap:
            return files

        logger.warning(
            "Retrospective context exceeded operational cap: %d tokens (cap=%d). "
            "Pruning optional context.",
            total_tokens,
            token_cap,
        )
        pruned_files, pruned_tokens, dropped_sections, dropped_story_files = self._prune_context_files(
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
            mission=mission,
            filtered_instructions=filtered_instructions,
        )
        if pruned_tokens > token_cap:
            dropped_story_summary = ", ".join(dropped_story_files) or "none"
            raise CompilerError(
                "retrospective context still exceeds the operational cap after pruning "
                f"optional sections ({', '.join(dropped_sections) or 'none'}): "
                f"{pruned_tokens} tokens > {token_cap}. "
                f"Dropped story files: {dropped_story_summary}. "
                "Reduce retrospective context or trim epic story artifacts before invoking the provider."
            )

        dropped_story_summary = ", ".join(dropped_story_files) or "none"
        logger.info(
            "Retrospective context pruned to %d tokens after dropping optional sections: %s; "
            "dropped story files: %s",
            pruned_tokens,
            ", ".join(dropped_sections) or "none",
            dropped_story_summary,
        )
        return pruned_files

    def _build_context_sections(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
    ]:
        """Build ordered retrospective context sections before budget enforcement."""
        project_context_files: dict[str, str] = {}
        architecture_files: dict[str, str] = {}
        prd_files: dict[str, str] = {}
        epic_files: dict[str, str] = {}
        sprint_status_files: dict[str, str] = {}
        story_files: dict[str, str] = {}
        tea_files: dict[str, str] = {}
        previous_retro_files: dict[str, str] = {}
        project_root = context.project_root

        # 1. Project context (general)
        project_context_path = find_project_context_file(context)
        if project_context_path:
            content = safe_read_file(project_context_path, project_root)
            if content:
                project_context_files[str(project_context_path)] = content

        # 2. Architecture (technical constraints)
        arch_path = find_file_in_planning_dir(context, "*architecture*.md")
        if arch_path:
            content = safe_read_file(arch_path, project_root)
            if content:
                architecture_files[str(arch_path)] = content

        # 3. PRD (product requirements)
        prd_path = find_file_in_planning_dir(context, "*prd*.md")
        if prd_path:
            content = safe_read_file(prd_path, project_root)
            if content:
                prd_files[str(prd_path)] = content

        # 4. Epic file
        epic_num = resolved.get("epic_num")
        if epic_num is not None:
            epic_path = self._find_epic_file(context, epic_num)
            if epic_path:
                content = safe_read_file(epic_path, project_root)
                if content:
                    epic_files[str(epic_path)] = content

        # 5. Sprint status
        sprint_status_path = find_sprint_status_file(context)
        if sprint_status_path:
            content = safe_read_file(sprint_status_path, project_root)
            if content:
                sprint_status_files[str(sprint_status_path)] = content

        # 6. Story files for this epic
        if epic_num is not None:
            story_files.update(self._collect_story_files(context, epic_num))

        # 6b. TEA Context (trace matrix) for traceability review
        # F19 Fix: Trace matrix may not exist on first retrospective (created BY trace workflow)
        tea_files.update(collect_tea_context(context, "retrospective", resolved))

        # 7. Previous retrospective (LAST - closest to instructions)
        prev_epic_num = resolved.get("prev_epic_num")
        if prev_epic_num is not None:
            prev_retro_path = self._find_previous_retrospective(context, prev_epic_num)
            if prev_retro_path:
                content = safe_read_file(prev_retro_path, project_root)
                if content:
                    previous_retro_files[str(prev_retro_path)] = content

        return (
            project_context_files,
            architecture_files,
            prd_files,
            epic_files,
            sprint_status_files,
            story_files,
            tea_files,
            previous_retro_files,
        )

    def _merge_context_sections(self, *sections: dict[str, str]) -> dict[str, str]:
        """Merge ordered context sections while preserving recency-bias."""
        merged: dict[str, str] = {}
        for section in sections:
            merged.update(section)
        return merged

    def _estimate_context_tokens(self, files: dict[str, str]) -> int:
        """Estimate total token count for assembled context files."""
        return sum(estimate_tokens(content) for content in files.values())

    def _estimate_prompt_tokens(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
        files: dict[str, str],
        *,
        mission: str | None = None,
        filtered_instructions: str | None = None,
    ) -> int:
        """Estimate final retrospective prompt tokens for the assembled payload."""
        if mission is None or filtered_instructions is None:
            return self._estimate_context_tokens(files)

        compiled = CompiledWorkflow(
            workflow_name=self.workflow_name,
            mission=mission,
            context="",
            variables=resolved,
            instructions=filtered_instructions,
            output_template="",
            token_estimate=0,
        )
        result = generate_output(
            compiled,
            project_root=context.project_root,
            context_files=files,
            links_only=context.links_only,
        )
        final_xml = apply_post_process(result.xml, context)
        return estimate_tokens(final_xml)

    def _get_context_token_cap(self) -> int:
        """Return the effective operational token cap for retrospective context."""
        try:
            from bmad_assist.core.config.loaders import get_config

            config = get_config()
            budgets = getattr(getattr(config, "compiler", None), "source_context", None)
            budget_config = getattr(budgets, "budgets", None)
            if budget_config is not None:
                configured_budget = budget_config.get_budget("retrospective")
                if isinstance(configured_budget, int) and configured_budget > 0:
                    return min(configured_budget, RETROSPECTIVE_CONTEXT_HARD_CAP_TOKENS)
        except ConfigError:
            logger.debug(
                "Falling back to retrospective hard cap because config could not be loaded."
            )

        return RETROSPECTIVE_CONTEXT_HARD_CAP_TOKENS

    def _prune_context_files(
        self,
        project_context_files: dict[str, str],
        architecture_files: dict[str, str],
        prd_files: dict[str, str],
        epic_files: dict[str, str],
        sprint_status_files: dict[str, str],
        story_files: dict[str, str],
        tea_files: dict[str, str],
        previous_retro_files: dict[str, str],
        token_cap: int,
        *,
        context: CompilerContext,
        resolved: dict[str, Any],
        mission: str | None = None,
        filtered_instructions: str | None = None,
    ) -> tuple[dict[str, str], int, list[str], list[str]]:
        """Prune low-value retrospective context before provider invocation."""
        optional_sections: list[tuple[str, dict[str, str]]] = [
            ("previous-retrospective", previous_retro_files),
            ("project-context", project_context_files),
            ("architecture", architecture_files),
            ("prd", prd_files),
        ]
        dropped_sections: list[str] = []
        dropped_story_files: list[str] = []

        candidate_files = self._merge_context_sections(
            project_context_files,
            architecture_files,
            prd_files,
            epic_files,
            sprint_status_files,
            story_files,
            tea_files,
            previous_retro_files,
        )
        candidate_tokens = self._estimate_prompt_tokens(
            context,
            resolved,
            candidate_files,
            mission=mission,
            filtered_instructions=filtered_instructions,
        )
        if candidate_tokens <= token_cap:
            return candidate_files, candidate_tokens, dropped_sections, dropped_story_files

        for section_name, _ in optional_sections:
            dropped_sections.append(section_name)
            candidate_files = self._merge_context_sections(
                {} if "project-context" in dropped_sections else project_context_files,
                {} if "architecture" in dropped_sections else architecture_files,
                {} if "prd" in dropped_sections else prd_files,
                epic_files,
                sprint_status_files,
                story_files,
                tea_files,
                {} if "previous-retrospective" in dropped_sections else previous_retro_files,
            )
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            if candidate_tokens <= token_cap:
                return candidate_files, candidate_tokens, dropped_sections, dropped_story_files

        remaining_story_items = list(story_files.items())
        while remaining_story_items and candidate_tokens > token_cap:
            dropped_path, _ = remaining_story_items.pop(0)
            dropped_story_files.append(dropped_path)
            candidate_files = self._merge_context_sections(
                {} if "project-context" in dropped_sections else project_context_files,
                {} if "architecture" in dropped_sections else architecture_files,
                {} if "prd" in dropped_sections else prd_files,
                epic_files,
                sprint_status_files,
                dict(remaining_story_items),
                tea_files,
                {} if "previous-retrospective" in dropped_sections else previous_retro_files,
            )
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            if candidate_tokens <= token_cap:
                return candidate_files, candidate_tokens, dropped_sections, dropped_story_files

        if candidate_tokens > token_cap:
            for compact_token_budget in self._required_context_compaction_budgets(token_cap):
                compacted_epic_files, compacted_epic = self._compact_epic_files(
                    epic_files,
                    resolved,
                    compact_token_budget,
                )
                compacted_sprint_status_files, compacted_sprint_status = (
                    self._compact_sprint_status_files(
                        sprint_status_files,
                        resolved,
                        compact_token_budget,
                    )
                )
                compacted_tea_files, compacted_tea = self._compact_context_file_map(
                    tea_files,
                    compact_token_budget,
                )

                candidate_files = self._merge_context_sections(
                    {} if "project-context" in dropped_sections else project_context_files,
                    {} if "architecture" in dropped_sections else architecture_files,
                    {} if "prd" in dropped_sections else prd_files,
                    compacted_epic_files,
                    compacted_sprint_status_files,
                    dict(remaining_story_items),
                    compacted_tea_files,
                    {} if "previous-retrospective" in dropped_sections else previous_retro_files,
                )
                candidate_tokens = self._estimate_prompt_tokens(
                    context,
                    resolved,
                    candidate_files,
                    mission=mission,
                    filtered_instructions=filtered_instructions,
                )

                compacted_sections = self._get_compacted_section_names(
                    compacted_epic,
                    compacted_sprint_status,
                    compacted_tea,
                )
                if compacted_sections:
                    for section_name in compacted_sections:
                        if section_name not in dropped_sections:
                            dropped_sections.append(section_name)

                if candidate_tokens <= token_cap:
                    return candidate_files, candidate_tokens, dropped_sections, dropped_story_files

            placeholder_epic_files = self._replace_context_with_placeholders(
                epic_files,
                "epic",
            )
            placeholder_sprint_status_files = self._replace_context_with_placeholders(
                sprint_status_files,
                "sprint-status",
            )
            placeholder_tea_files = self._replace_context_with_placeholders(
                tea_files,
                "tea",
            )
            candidate_files = self._merge_context_sections(
                {} if "project-context" in dropped_sections else project_context_files,
                {} if "architecture" in dropped_sections else architecture_files,
                {} if "prd" in dropped_sections else prd_files,
                placeholder_epic_files,
                placeholder_sprint_status_files,
                dict(remaining_story_items),
                placeholder_tea_files,
                {} if "previous-retrospective" in dropped_sections else previous_retro_files,
            )
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            for section_name in [
                "deferred-epic",
                "deferred-sprint-status",
                "deferred-tea",
            ]:
                if section_name not in dropped_sections:
                    dropped_sections.append(section_name)

        return candidate_files, candidate_tokens, dropped_sections, dropped_story_files

    def _required_context_compaction_budgets(self, token_cap: int) -> list[int]:
        """Return progressively smaller budgets for required retrospective context."""
        initial_budget = min(
            _RETROSPECTIVE_COMPACTED_SECTION_TARGET_TOKENS,
            max(_RETROSPECTIVE_COMPACTED_SECTION_MIN_TOKENS, token_cap // 8),
        )
        tighter_budget = max(_RETROSPECTIVE_COMPACTED_SECTION_MIN_TOKENS, token_cap // 16)
        tightest_budget = max(_RETROSPECTIVE_COMPACTED_SECTION_MIN_TOKENS, token_cap // 32)

        budgets: list[int] = []
        for budget in [initial_budget, tighter_budget, tightest_budget]:
            if budget not in budgets:
                budgets.append(budget)

        return budgets

    @staticmethod
    def _get_compacted_section_names(
        compacted_epic: bool,
        compacted_sprint_status: bool,
        compacted_tea: bool,
    ) -> list[str]:
        """Return compaction labels for observability and compiler errors."""
        section_names: list[str] = []
        if compacted_epic:
            section_names.append("compacted-epic")
        if compacted_sprint_status:
            section_names.append("compacted-sprint-status")
        if compacted_tea:
            section_names.append("compacted-tea")

        return section_names

    def _compact_epic_files(
        self,
        epic_files: dict[str, str],
        resolved: dict[str, Any],
        token_budget: int,
    ) -> tuple[dict[str, str], bool]:
        """Compact epic context while preserving the current epic section when possible."""
        epic_num = resolved.get("epic_num")
        compacted_files: dict[str, str] = {}
        changed = False

        for path, content in epic_files.items():
            candidate = self._extract_current_epic_section(content, epic_num)
            if candidate != content:
                changed = True

            compacted = self._compact_content(candidate, token_budget)
            if compacted != content:
                changed = True
            compacted_files[path] = compacted

        return compacted_files, changed

    def _compact_sprint_status_files(
        self,
        sprint_status_files: dict[str, str],
        resolved: dict[str, Any],
        token_budget: int,
    ) -> tuple[dict[str, str], bool]:
        """Compact sprint status to current epic and retrospective gate excerpts."""
        epic_num = resolved.get("epic_num")
        compacted_files: dict[str, str] = {}
        changed = False

        for path, content in sprint_status_files.items():
            candidate = self._extract_current_epic_sprint_status(content, epic_num)
            if candidate != content:
                changed = True

            compacted = self._compact_content(candidate, token_budget)
            if compacted != content:
                changed = True
            compacted_files[path] = compacted

        return compacted_files, changed

    def _compact_context_file_map(
        self,
        files: dict[str, str],
        token_budget: int,
    ) -> tuple[dict[str, str], bool]:
        """Compact a context file map with markdown-aware truncation."""
        compacted_files: dict[str, str] = {}
        changed = False

        for path, content in files.items():
            compacted = self._compact_content(content, token_budget)
            if compacted != content:
                changed = True
            compacted_files[path] = compacted

        return compacted_files, changed

    def _compact_content(self, content: str, token_budget: int) -> str:
        """Trim oversized retrospective context at a readable markdown boundary."""
        if estimate_tokens(content) <= token_budget:
            return content

        target_chars = max(1, token_budget * 4)
        if len(content) <= target_chars:
            return content

        search_start = int(target_chars * 0.8)
        search_end = min(target_chars, len(content))
        search_region = content[search_start:search_end]
        best_cut = target_chars

        header_matches = list(re.finditer(r"\n#{1,4}\s+[^\n]+\n", search_region))
        if header_matches:
            best_cut = search_start + header_matches[-1].start()
        else:
            blank_line_matches = list(re.finditer(r"\n\s*\n", search_region))
            if blank_line_matches:
                best_cut = search_start + blank_line_matches[-1].end()
            else:
                for index in range(min(target_chars, len(content) - 1), search_start, -1):
                    if content[index] in " \n\t.,;:!?":
                        best_cut = index + 1
                        break

        return content[:best_cut].rstrip() + _RETROSPECTIVE_COMPACTION_NOTICE

    def _extract_current_epic_section(self, content: str, epic_num: Any) -> str:
        """Extract the current epic section from a whole epics document."""
        if epic_num is None:
            return content

        epic_text = str(epic_num).strip()
        if not epic_text:
            return content

        heading_pattern = re.compile(
            rf"^(?P<marker>#+)\s+Epic\s+0*{re.escape(epic_text)}(?:\b|[\s:.-]).*$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = heading_pattern.search(content)
        if match is None:
            return content

        heading_level = len(match.group("marker"))
        next_epic_pattern = re.compile(
            rf"^#{{1,{heading_level}}}\s+Epic\s+\d+(?:\b|[\s:.-]).*$",
            re.IGNORECASE | re.MULTILINE,
        )
        next_match = next_epic_pattern.search(content, match.end())
        end = next_match.start() if next_match else len(content)
        return content[match.start() : end].strip()

    def _extract_current_epic_sprint_status(self, content: str, epic_num: Any) -> str:
        """Extract current-epic sprint status and retrospective gate excerpts."""
        if epic_num is None:
            return content

        epic_text = str(epic_num).strip()
        if not epic_text:
            return content

        epic_key = f"epic-{epic_text}"
        story_prefix = f"{epic_text}-"
        metadata_lines: list[str] = []
        current_epic_lines: list[str] = []
        gate_lines: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            lower_stripped = stripped.lower()
            if stripped.startswith("# Generated") or stripped.startswith("# Sprint Status"):
                metadata_lines.append(line)
                continue

            if stripped in {"metadata:", "development_status:"}:
                current_epic_lines.append(line)
                continue

            if stripped == f"# Epic {epic_text}":
                current_epic_lines.append(line)
                continue

            if (
                stripped.startswith(f"{epic_key}:")
                or stripped.startswith(f"{epic_key}-retrospective:")
                or stripped.startswith(story_prefix)
            ):
                current_epic_lines.append(line)
                continue

            if "retrospective" in lower_stripped and len(gate_lines) < 30:
                gate_lines.append(line)

        if not current_epic_lines:
            return content

        parts = [
            "# Sprint status compacted for retrospective context",
            "",
        ]
        if metadata_lines:
            parts.extend(metadata_lines)
            parts.append("")

        parts.extend(
            [
                "## Current Epic Development Status",
                "```yaml",
                *current_epic_lines,
                "```",
            ]
        )

        if gate_lines:
            parts.extend(
                [
                    "",
                    "## Retrospective Gate Excerpts",
                    "```yaml",
                    *gate_lines,
                    "```",
                ]
            )

        parts.append(_RETROSPECTIVE_COMPACTION_NOTICE.strip())
        return "\n".join(parts)

    @staticmethod
    def _replace_context_with_placeholders(
        files: dict[str, str],
        section_name: str,
    ) -> dict[str, str]:
        """Replace oversized required context with source-path placeholders."""
        placeholders: dict[str, str] = {}
        for path in files:
            placeholders[path] = (
                f"# {section_name} context deferred due token cap\n\n"
                "Full content was omitted from the preloaded retrospective prompt "
                "to keep the autonomous run within the operational token cap.\n\n"
                f"Path: `{path}`\n\n"
                "Required action: read this file directly before finalizing the "
                "retrospective evidence for this section.\n"
            )

        return placeholders

    def _find_epic_file(self, context: CompilerContext, epic_num: Any) -> Path | None:
        """Find epic file by number.

        Searches both sharded and whole epic patterns.

        Args:
            context: Compilation context.
            epic_num: Epic number to find.

        Returns:
            Path to epic file, or None if not found.

        """
        planning_dir = context.output_folder
        if planning_dir is None:
            return None

        # Try docs/epics/epic-N.md pattern (sharded)
        docs_dir = context.project_root / "docs"
        if docs_dir.exists():
            sharded_path = docs_dir / "epics" / f"epic-{epic_num}.md"
            if sharded_path.exists():
                return sharded_path

            # Also try epic-N-*.md pattern
            epics_dir = docs_dir / "epics"
            if epics_dir.exists():
                for f in epics_dir.glob(f"epic-{epic_num}*.md"):
                    return f

        # Try whole document pattern
        whole_path = find_file_in_planning_dir(context, f"*epic*{epic_num}*.md")
        if whole_path:
            return whole_path

        return None

    def _collect_story_files(self, context: CompilerContext, epic_num: Any) -> dict[str, str]:
        """Collect all story files for an epic.

        Args:
            context: Compilation context.
            epic_num: Epic number to collect stories for.

        Returns:
            Dictionary mapping story file paths to content.

        """
        result: dict[str, str] = {}
        project_root = context.project_root

        # Look in implementation artifacts using shared utility
        stories_dir = get_stories_dir(context)
        if not stories_dir.exists():
            # Fallback to legacy location
            stories_dir = project_root / "docs" / "sprint-artifacts"
            if not stories_dir.exists():
                return result

        # Find all stories matching epic number
        for story_file in sorted(stories_dir.glob(f"{epic_num}-*.md")):
            match = _STORY_FILE_PATTERN.match(story_file.name)
            if match and match.group(1) == str(epic_num):
                content = safe_read_file(story_file, project_root)
                if content:
                    result[str(story_file)] = content

        if result:
            logger.debug("Collected %d story files for epic %s", len(result), epic_num)

        return result

    def _find_previous_retrospective(
        self, context: CompilerContext, prev_epic_num: int
    ) -> Path | None:
        """Find previous epic's retrospective file.

        Args:
            context: Compilation context.
            prev_epic_num: Previous epic number.

        Returns:
            Path to previous retrospective, or None if not found.

        """
        impl_artifacts = context.output_folder
        if impl_artifacts is None:
            return None

        # Find most recent retro for previous epic
        pattern = f"epic-{prev_epic_num}-retro-*.md"
        search_dirs = [impl_artifacts / "retrospectives", impl_artifacts]
        for retro_dir in search_dirs:
            if not retro_dir.exists():
                continue
            retros = sorted(retro_dir.glob(pattern), reverse=True)
            if retros:
                return retros[0]

        return None

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
            Mission description string for retrospective.

        """
        base_description = workflow_ir.raw_config.get(
            "description", "Run epic retrospective to review success and extract lessons learned"
        )

        epic_num = resolved.get("epic_num", "?")
        epic_title = resolved.get("epic_title", "")

        if epic_title:
            mission = (
                f"{base_description}\n\n"
                f"Target: Epic {epic_num} - {epic_title}\n"
                f"Generate retrospective report with extraction markers."
            )
        else:
            mission = (
                f"{base_description}\n\n"
                f"Target: Epic {epic_num}\n"
                f"Generate retrospective report with extraction markers."
            )

        return mission
