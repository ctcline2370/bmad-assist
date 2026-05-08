"""Compiler for the dev-story workflow.

This module implements the WorkflowCompiler protocol for the dev-story
workflow, producing standalone prompts for story implementation with
all necessary context embedded.

Public API:
    DevStoryCompiler: Workflow compiler class implementing WorkflowCompiler protocol
"""

import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from bmad_assist.bmad.parser import extract_epic_markdown, extract_markdown_sections
from bmad_assist.compiler.filtering import filter_instructions
from bmad_assist.compiler.output import (
    DEFAULT_HARD_LIMIT_TOKENS,
    DEFAULT_SOFT_LIMIT_TOKENS,
    SOFT_LIMIT_RATIO,
    generate_output,
)
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
from bmad_assist.core.exceptions import CompilerError, ConfigError
from bmad_assist.testarch.context import collect_tea_context, is_tea_context_enabled

logger = logging.getLogger(__name__)

DEV_STORY_CONTEXT_HARD_CAP_TOKENS = DEFAULT_HARD_LIMIT_TOKENS
DEV_STORY_REQUIRED_SECTION_HEADINGS = {
    "story",
    "acceptance criteria",
    "validation requirements",
    "tasks / subtasks",
}
DEV_STORY_EXCLUDED_DEV_NOTES_CHILDREN = {
    "agent model used",
    "completion notes list",
    "debug log references",
    "file list",
}
_CODE_REVIEW_SYNTHESIS_DIRS = (
    ("code-reviews",),
    ("implementation-artifacts", "code-reviews"),
    ("sprint-artifacts", "code-reviews"),
)


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

    @staticmethod
    def _filter_story_file_from_source_paths(
        file_list_paths: list[str],
        story_path: Path,
        project_root: Path,
    ) -> list[str]:
        """Exclude the active story artifact from source-context collection.

        Story files can list themselves in the File List section. When that happens,
        dev_story would otherwise inject the same artifact twice: once as source
        context and once again as the final story section.
        """
        story_full_path = story_path.resolve(strict=False)
        filtered_paths: list[str] = []

        for raw_path in file_list_paths:
            candidate_path = Path(raw_path)
            if not candidate_path.is_absolute():
                candidate_path = project_root / candidate_path

            if candidate_path.resolve(strict=False) == story_full_path:
                logger.info("Skipping self-listed story artifact from source context: %s", raw_path)
                continue

            filtered_paths.append(raw_path)

        return filtered_paths

    @staticmethod
    def _normalize_markdown_heading(heading: str) -> str:
        """Normalize markdown headings for deterministic comparisons."""
        return re.sub(r"\s+", " ", heading.strip()).lower()

    def _trim_epic_context(self, path: Path, content: str, epic_num: Any) -> str:
        """Trim multi-epic markdown files down to the active epic."""
        trimmed = extract_epic_markdown(content, epic_num)
        if not trimmed:
            return content

        stripped_content = content.strip()
        if trimmed == stripped_content:
            return content

        logger.info("Trimmed epic context to active epic for dev_story: %s", path)
        return trimmed

    def _select_story_section_headings(self, content: str) -> list[str]:
        """Select the implementation-relevant story headings to retain."""
        selected_headings: list[str] = []
        current_h2_heading = ""
        saw_dev_notes = False
        selected_dev_notes_child = False

        for match in re.finditer(r"^(#{2,6})\s+(.+?)\s*$", content, re.MULTILINE):
            level = len(match.group(1))
            heading = match.group(2).strip()
            normalized_heading = self._normalize_markdown_heading(heading)

            if level == 2:
                current_h2_heading = heading
                if normalized_heading in DEV_STORY_REQUIRED_SECTION_HEADINGS:
                    selected_headings.append(heading)
                elif normalized_heading == "dev notes":
                    saw_dev_notes = True
                continue

            if (
                level >= 3
                and self._normalize_markdown_heading(current_h2_heading) == "dev notes"
                and normalized_heading not in DEV_STORY_EXCLUDED_DEV_NOTES_CHILDREN
            ):
                selected_headings.append(heading)
                selected_dev_notes_child = True

        if saw_dev_notes and not selected_dev_notes_child:
            selected_headings.append("Dev Notes")

        return selected_headings

    def _trim_story_context(self, path: Path, content: str) -> str:
        """Trim story markdown to implementation-critical sections."""
        selected_headings = self._select_story_section_headings(content)
        if not selected_headings:
            return content

        extracted_sections = extract_markdown_sections(content, selected_headings)
        if not extracted_sections:
            return content

        first_h2 = re.search(r"^##(?!#)\s+", content, re.MULTILINE)
        preamble = content[: first_h2.start()].strip() if first_h2 else content.strip()

        trimmed = "\n\n".join(part for part in (preamble, extracted_sections) if part).strip()
        if not trimmed:
            return content

        if len(trimmed) < len(content.strip()):
            logger.info("Trimmed story context to implementation sections for dev_story: %s", path)
            return trimmed

        return content

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

            filtered_instructions = filter_instructions(workflow_ir)
            filtered_instructions = substitute_variables(filtered_instructions, resolved)

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
        story_content: str | None = None

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
                    epic_files[str(epic_path)] = self._trim_epic_context(epic_path, content, epic_num)

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
            file_list_paths: list[str] = []
            story_content = safe_read_file(story_path, project_root)
            if story_content:
                file_list_paths = extract_file_paths_from_story(story_content)
                file_list_paths = self._filter_story_file_from_source_paths(
                    file_list_paths,
                    story_path,
                    project_root,
                )

            service = SourceContextService(context, "dev_story")
            source_files.update(service.collect_files(file_list_paths, None))

        # 5. Story file (LAST - closest to instructions per recency-bias)
        if story_path_str:
            story_path = Path(story_path_str)
            if story_content:
                story_files[str(story_path)] = self._trim_story_context(story_path, story_content)

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

        token_cap = self._get_context_token_cap()
        total_tokens = self._estimate_prompt_tokens(
            context,
            resolved,
            files,
            mission=mission,
            filtered_instructions=filtered_instructions,
        )
        soft_token_cap = self._get_context_soft_token_cap(token_cap)
        if total_tokens <= token_cap:
            if total_tokens <= soft_token_cap:
                return files

            protected_source_files = self._find_review_critical_source_files(
                context,
                resolved,
                source_files,
            )
            logger.info(
                "Dev story context exceeded soft target: %d tokens (soft=%d, hard=%d). "
                "Pruning optional context only.",
                total_tokens,
                soft_token_cap,
                token_cap,
            )
            pruned_files, pruned_tokens, dropped_sections, _ = self._prune_context_files(
                strategic_files,
                antipattern_files,
                epic_files,
                tea_test_design_files,
                tea_other_files,
                tea_atdd_files,
                source_files,
                story_files,
                soft_token_cap,
                context=context,
                resolved=resolved,
                mission=mission,
                filtered_instructions=filtered_instructions,
                prune_source=False,
                protected_source_files=protected_source_files,
            )
            if dropped_sections:
                logger.info(
                    "Dev story context pruned toward soft target to %d tokens after dropping "
                    "optional sections: %s",
                    pruned_tokens,
                    ", ".join(dropped_sections),
                )
                return pruned_files

            return files

        logger.warning(
            "Dev story context exceeded operational cap: %d tokens (cap=%d). Pruning optional context.",
            total_tokens,
            token_cap,
        )
        protected_source_files = self._find_review_critical_source_files(
            context,
            resolved,
            source_files,
        )
        pruned_files, pruned_tokens, dropped_sections, dropped_source_files = self._prune_context_files(
            strategic_files,
            antipattern_files,
            epic_files,
            tea_test_design_files,
            tea_other_files,
            tea_atdd_files,
            source_files,
            story_files,
            soft_token_cap,
            context=context,
            resolved=resolved,
            mission=mission,
            filtered_instructions=filtered_instructions,
            prune_source=False,
            protected_source_files=protected_source_files,
        )
        if pruned_tokens > token_cap:
            pruned_files, pruned_tokens, dropped_sections, dropped_source_files = self._prune_context_files(
                strategic_files,
                antipattern_files,
                epic_files,
                tea_test_design_files,
                tea_other_files,
                tea_atdd_files,
                source_files,
                story_files,
                token_cap,
                context=context,
                resolved=resolved,
                mission=mission,
                filtered_instructions=filtered_instructions,
                protected_source_files=protected_source_files,
                prune_epic=True,
            )
        if pruned_tokens > token_cap:
            dropped_source_summary = ", ".join(dropped_source_files) or "none"
            protected_source_summary = ", ".join(sorted(protected_source_files)) or "none"
            raise CompilerError(
                "dev-story context still exceeds the operational cap after pruning "
                f"optional sections ({', '.join(dropped_sections) or 'none'}): "
                f"{pruned_tokens} tokens > {token_cap}. "
                f"Dropped source files: {dropped_source_summary}. "
                f"Protected review-critical source files: {protected_source_summary}. "
                "Reduce source context or trim story-linked files before invoking the provider."
            )

        dropped_source_summary = ", ".join(dropped_source_files) or "none"
        logger.info(
            "Dev story context pruned to %d tokens after dropping optional sections: %s; "
            "dropped source files: %s",
            pruned_tokens,
            ", ".join(dropped_sections) or "none",
            dropped_source_summary,
        )
        return pruned_files

    def _find_review_critical_source_files(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
        source_files: dict[str, str],
    ) -> set[str]:
        """Return collected source files referenced by the latest review synthesis."""
        if not source_files:
            return set()

        synthesis_path = self._find_latest_code_review_synthesis_report(context, resolved)
        if synthesis_path is None:
            return set()

        content = safe_read_file(synthesis_path, context.project_root)
        if not content:
            return set()

        content_lower = content.lower()
        protected_files: set[str] = set()
        for source_path in source_files:
            variants = self._source_path_match_variants(source_path, context.project_root)
            if any(variant.lower() in content_lower for variant in variants):
                protected_files.add(source_path)

        if protected_files:
            logger.info(
                "Protected %d review-critical source file(s) from dev-story pruning based on %s: %s",
                len(protected_files),
                synthesis_path,
                ", ".join(sorted(protected_files)),
            )

        return protected_files

    def _find_latest_code_review_synthesis_report(
        self,
        context: CompilerContext,
        resolved: dict[str, Any],
    ) -> Path | None:
        """Find the newest code-review synthesis report for the active story."""
        epic_num = resolved.get("epic_num")
        story_num = resolved.get("story_num")
        if epic_num is None or story_num is None:
            return None

        epic = self._sanitize_synthesis_path_part(epic_num)
        story = self._sanitize_synthesis_path_part(story_num)
        if not epic or not story:
            return None

        candidate_roots = self._code_review_synthesis_roots(context)
        matches: list[Path] = []
        pattern = f"synthesis-{epic}-{story}-*.md"
        for root in candidate_roots:
            if root.is_dir():
                matches.extend(path for path in root.glob(pattern) if path.is_file())

        if not matches:
            return None

        return max(matches, key=lambda path: path.stat().st_mtime)

    def _code_review_synthesis_roots(self, context: CompilerContext) -> list[Path]:
        """Return likely code-review report directories for current and legacy layouts."""
        candidates: list[Path] = []
        base_roots = [
            context.output_folder,
            context.output_folder.parent,
            context.project_root / "_bmad-output",
            context.project_root / "docs",
        ]

        for base in base_roots:
            candidates.extend(base.joinpath(*parts) for parts in _CODE_REVIEW_SYNTHESIS_DIRS)

        return self._dedupe_paths(candidates)

    @staticmethod
    def _source_path_match_variants(source_path: str, project_root: Path) -> set[str]:
        """Return text variants likely to appear in synthesis markdown."""
        path = Path(source_path)
        variants = {source_path, path.name}
        normalized = source_path.replace("\\", "/")
        variants.add(normalized)

        if path.is_absolute():
            with suppress(ValueError):
                variants.add(str(path.relative_to(project_root)).replace("\\", "/"))

        return {variant for variant in variants if variant}

    @staticmethod
    def _sanitize_synthesis_path_part(value: Any) -> str:
        """Normalize epic/story identifiers to report filename components."""
        return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip()).strip("-")

    @staticmethod
    def _dedupe_paths(paths: list[Path]) -> list[Path]:
        """Deduplicate paths while preserving order."""
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in paths:
            normalized = path.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(path)

        return unique

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
        """Estimate the final prompt tokens for the assembled dev-story payload."""
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
        """Return the effective operational token cap for dev-story context."""
        try:
            from bmad_assist.core.config.loaders import get_config

            config = get_config()
            budgets = getattr(getattr(config, "compiler", None), "source_context", None)
            budget_config = getattr(budgets, "budgets", None)
            if budget_config is not None:
                configured_budget = budget_config.get_budget("dev_story")
                if isinstance(configured_budget, int) and configured_budget > 0:
                    return min(configured_budget, DEV_STORY_CONTEXT_HARD_CAP_TOKENS)
        except ConfigError:
            logger.debug("Falling back to dev-story hard cap because config could not be loaded.")

        return DEV_STORY_CONTEXT_HARD_CAP_TOKENS

    def _get_context_soft_token_cap(self, token_cap: int) -> int:
        """Return the soft pruning target that matches output budget warnings."""
        if token_cap == DEFAULT_HARD_LIMIT_TOKENS:
            return DEFAULT_SOFT_LIMIT_TOKENS

        return max(1, int(token_cap * SOFT_LIMIT_RATIO))

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
        token_cap: int,
        *,
        context: CompilerContext,
        resolved: dict[str, Any],
        mission: str | None = None,
        filtered_instructions: str | None = None,
        prune_source: bool = True,
        protected_source_files: set[str] | None = None,
        prune_epic: bool = False,
    ) -> tuple[dict[str, str], int, list[str], list[str]]:
        """Prune low-value duplicate context before provider invocation."""
        optional_sections: list[tuple[str, dict[str, str]]] = [
            ("antipatterns", antipattern_files),
            ("tea-other", tea_other_files),
            ("tea-test-design", tea_test_design_files),
        ]
        dropped_sections: list[str] = []
        dropped_source_files: list[str] = []

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
        candidate_tokens = self._estimate_prompt_tokens(
            context,
            resolved,
            candidate_files,
            mission=mission,
            filtered_instructions=filtered_instructions,
        )
        if candidate_tokens <= token_cap:
            return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

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
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            if candidate_tokens <= token_cap:
                return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

        if not prune_source:
            return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

        remaining_source_items = list(source_files.items())
        protected_source_files = protected_source_files or set()
        while candidate_tokens > token_cap:
            drop_index = self._find_prunable_source_index(
                remaining_source_items,
                protected_source_files,
            )
            if drop_index is None:
                break

            dropped_path, _ = remaining_source_items.pop(drop_index)
            dropped_source_files.append(dropped_path)
            candidate_files = self._merge_context_sections(
                strategic_files,
                {} if "antipatterns" in dropped_sections else antipattern_files,
                epic_files,
                {} if "tea-test-design" in dropped_sections else tea_test_design_files,
                {} if "tea-other" in dropped_sections else tea_other_files,
                tea_atdd_files,
                dict(remaining_source_items),
                story_files,
            )
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            if candidate_tokens <= token_cap:
                return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

        candidate_source_files = dict(remaining_source_items)
        if candidate_tokens > token_cap and protected_source_files:
            placeholder_source_files = self._replace_protected_source_with_placeholders(
                candidate_source_files,
                protected_source_files,
            )
            if placeholder_source_files != candidate_source_files:
                candidate_source_files = placeholder_source_files
                candidate_files = self._merge_context_sections(
                    strategic_files,
                    {} if "antipatterns" in dropped_sections else antipattern_files,
                    epic_files,
                    {} if "tea-test-design" in dropped_sections else tea_test_design_files,
                    {} if "tea-other" in dropped_sections else tea_other_files,
                    tea_atdd_files,
                    candidate_source_files,
                    story_files,
                )
                candidate_tokens = self._estimate_prompt_tokens(
                    context,
                    resolved,
                    candidate_files,
                    mission=mission,
                    filtered_instructions=filtered_instructions,
                )
                placeholder_count = sum(
                    1 for path in placeholder_source_files if path in protected_source_files
                )
                logger.warning(
                    "Dev story context retained %d review-critical source path "
                    "placeholder(s) after full source context exceeded cap",
                    placeholder_count,
                )

        if candidate_tokens > token_cap and prune_epic and epic_files:
            dropped_sections.append("epic-context")
            candidate_files = self._merge_context_sections(
                strategic_files,
                {} if "antipatterns" in dropped_sections else antipattern_files,
                {},
                {} if "tea-test-design" in dropped_sections else tea_test_design_files,
                {} if "tea-other" in dropped_sections else tea_other_files,
                tea_atdd_files,
                candidate_source_files,
                story_files,
            )
            candidate_tokens = self._estimate_prompt_tokens(
                context,
                resolved,
                candidate_files,
                mission=mission,
                filtered_instructions=filtered_instructions,
            )
            if candidate_tokens <= token_cap:
                return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

        return candidate_files, candidate_tokens, dropped_sections, dropped_source_files

    @staticmethod
    def _find_prunable_source_index(
        remaining_source_items: list[tuple[str, str]],
        protected_source_files: set[str],
    ) -> int | None:
        """Return the last source-file index that is not protected."""
        for index in range(len(remaining_source_items) - 1, -1, -1):
            path, _ = remaining_source_items[index]
            if path not in protected_source_files:
                return index

        return None

    @staticmethod
    def _replace_protected_source_with_placeholders(
        source_files: dict[str, str],
        protected_source_files: set[str],
    ) -> dict[str, str]:
        """Keep review-critical paths while deferring full source bodies."""
        placeholder_source_files: dict[str, str] = {}
        for path, content in source_files.items():
            if path not in protected_source_files:
                placeholder_source_files[path] = content
                continue

            placeholder_source_files[path] = (
                "# Source context deferred due token cap\n\n"
                "The full contents of this review-critical file were omitted from the "
                "preloaded dev-story prompt to keep the autonomous run within the "
                "operational token cap.\n\n"
                f"Path: `{path}`\n\n"
                "Required action: read this file directly before editing or validating "
                "related review findings.\n"
            )

        return placeholder_source_files

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
