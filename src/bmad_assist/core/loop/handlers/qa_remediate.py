"""QA_REMEDIATE phase handler.

Collects epic issues from multiple sources, runs Deep Verify scans for
fresh findings, invokes master LLM to auto-fix or escalate, then re-scans.

Iteration loop:
  1. Run DV scan on epic source files → fresh findings on disk
  2. Collect all issues (DV + historical sources)
  3. If 0 issues → clean, done
  4. Invoke master LLM to fix
  5. Run DV scan on modified files only → updated findings
  6. Collect issues again → compare
  7. If 0 → done. If same → escalate. If fewer → iterate.
  8. Max iterations default 3.

Uses direct invocation pattern (like qa_plan_generate/execute) — calls
Python functions directly instead of compiling workflow prompts.

NOTE: Like QaPlanGenerateHandler and QaPlanExecuteHandler, this handler
overrides BaseHandler.execute() entirely (direct invocation pattern).
Prompt saving and timing tracking are not applicable for this pattern.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from bmad_assist.core.loop.handlers.base import BaseHandler
from bmad_assist.core.loop.types import PhaseResult
from bmad_assist.core.state import State
from bmad_assist.core.types import EpicId
from bmad_assist.qa.prompts import get_remediate_prompt
from bmad_assist.qa.remediate import (
    REMEDIATE_ESCALATIONS_END,
    REMEDIATE_ESCALATIONS_START,
    EscalationItem,
    collect_epic_issues,
    extract_escalations,
    extract_modified_files,
    save_escalation_report,
    save_remediation_report,
)

logger = logging.getLogger(__name__)


class QaRemediateHandler(BaseHandler):
    """Handler for QA_REMEDIATE phase.

    Collects issues from QA results, code reviews, retro, scorecard,
    validations, Deep Verify, and security reports. Runs fresh DV scans
    before each iteration. Invokes master LLM to fix code. Loops until
    clean or max iterations reached.

    This phase runs after QA_PLAN_EXECUTE in epic_teardown.
    """

    @property
    def phase_name(self) -> str:
        """Return the phase name."""
        return "qa_remediate"

    def build_context(self, state: State) -> dict[str, Any]:
        """Not used — direct invocation pattern."""
        return {}

    def execute(self, state: State) -> PhaseResult:
        """Execute QA remediation with DV-driven iteration loop.

        Each iteration:
          1. Run DV scan → fresh findings on disk
          2. Collect all issues (DV findings + historical sources)
          3. If 0 → clean, done
          4. Invoke master LLM with issues → fixes code
          5. Next iteration re-scans (DV on modified files only)

        Args:
            state: Current loop state with epic information.

        Returns:
            PhaseResult with remediation summary.

        """
        if state.current_epic is None:
            return PhaseResult.fail("Cannot remediate: no current epic set")

        epic_id = state.current_epic
        logger.info("Starting QA remediation for epic %s...", epic_id)

        try:
            qa_config = self.config.qa
            max_iterations = qa_config.remediate_max_iterations if qa_config else 3
            max_age = qa_config.remediate_max_age_days if qa_config else 7
            safety_cap = qa_config.remediate_safety_cap if qa_config else 0.8
            max_issues = qa_config.remediate_max_issues if qa_config else 200

            fixed_files: set[str] = set()
            all_escalations: list[EscalationItem] = []
            final_pass_rate: float | None = None
            prev_pass_rate: float | None = None
            iterations_run = 0
            total_issues_accumulated = 0
            prev_issue_descriptions: set[str] = set()
            last_esc_path: str | None = None

            # Detect project stacks once (used by file filter + DV context)
            try:
                from bmad_assist.deep_verify.stack_detector import detect_project_stacks

                project_stacks = tuple(
                    detect_project_stacks(self.project_path, config=self.config)
                )
            except (ImportError, RuntimeError):
                project_stacks = ()

            # Resolve epic source files once (reused across iterations)
            epic_source_files = self._resolve_epic_source_files(
                epic_id, project_stacks=project_stacks,
            )
            if epic_source_files:
                logger.info(
                    "Epic %s: %d source files for DV scanning",
                    epic_id, len(epic_source_files),
                )

            for iteration in range(max_iterations):
                iterations_run = iteration + 1
                logger.info(
                    "Remediation iteration %d/%d for epic %s",
                    iterations_run, max_iterations, epic_id,
                )

                # 1. Run DV scan for fresh findings
                if iteration == 0:
                    # First iteration: scan all epic source files
                    scan_files = epic_source_files
                else:
                    # Subsequent iterations: scan only files LLM modified
                    scan_files = [
                        f for f in epic_source_files
                        if str(f) in fixed_files
                        or any(str(f).endswith(fp) for fp in fixed_files)
                    ]

                dv_findings_count = self._run_dv_scan(
                    epic_id=epic_id,
                    files=scan_files,
                    iteration=iterations_run,
                    project_stacks=project_stacks,
                )

                # 2. Collect issues from all sources (DV picks up fresh scan)
                collection = collect_epic_issues(
                    epic_id=epic_id,
                    project_path=self.project_path,
                    max_age_days=max_age,
                )

                if collection.stale_sources:
                    logger.warning("Stale sources detected: %s", collection.stale_sources)

                # Log collection breakdown by source
                if collection.issues:
                    from collections import Counter
                    source_counts = Counter(i.source for i in collection.issues)
                    logger.info(
                        "Collected %d issues from %d sources: %s",
                        len(collection.issues),
                        collection.sources_found,
                        dict(source_counts),
                    )

                # Deduplicate within this iteration's collection
                seen_this_iter: set[str] = set()
                unique_issues = []
                for i in collection.issues:
                    if i.description not in seen_this_iter:
                        seen_this_iter.add(i.description)
                        unique_issues.append(i)

                # On iteration 2+, check if issues changed
                current_descriptions = {i.description for i in unique_issues}
                if iteration > 0 and current_descriptions == prev_issue_descriptions:
                    logger.warning(
                        "Same %d issues persist after fix attempt — "
                        "LLM did not resolve them. Escalating for epic %s.",
                        len(unique_issues),
                        epic_id,
                    )
                    break

                # Log improvement between iterations
                if iteration > 0 and prev_issue_descriptions:
                    resolved = prev_issue_descriptions - current_descriptions
                    new_found = current_descriptions - prev_issue_descriptions
                    if resolved:
                        logger.info(
                            "Iteration %d resolved %d issues, %d new issues found.",
                            iterations_run, len(resolved), len(new_found),
                        )

                prev_issue_descriptions = current_descriptions
                new_issues = unique_issues

                if not new_issues:
                    logger.info("No issues found — epic %s is clean.", epic_id)
                    if iterations_run == 1:
                        return PhaseResult.ok({
                            "status": "clean",
                            "report_path": None,
                            "escalation_path": None,
                            "iterations": iterations_run,
                            "issues_found": 0,
                            "issues_fixed": 0,
                            "issues_escalated": 0,
                            "files_modified": 0,
                            "dv_findings": dv_findings_count,
                            "retest_pass_rate": 100.0,
                        })
                    break  # Clean after a fix iteration

                total_issues_accumulated += len(new_issues)

                # Apply hard limit before invoking LLM (prevents overflow crash)
                if len(new_issues) > max_issues:
                    logger.warning(
                        "Truncating %d issues to max_issues=%d for epic %s iteration %d",
                        len(new_issues),
                        max_issues,
                        epic_id,
                        iterations_run,
                    )
                    from bmad_assist.qa.remediate import _apply_issue_limit
                    new_issues = _apply_issue_limit(new_issues, max_issues)

                # 3. Build prompt + invoke LLM
                prompt = self._build_remediate_prompt(
                    new_issues, fixed_files, epic_id, safety_cap,
                )
                result = self.invoke_provider(prompt, state=state)

                # 4. Track modified files
                new_fixed = extract_modified_files(result.stdout)
                refixed = new_fixed & fixed_files
                if refixed:
                    logger.warning("Files re-fixed (will escalate): %s", refixed)
                fixed_files |= new_fixed

                # Log actual modifications
                if new_fixed:
                    logger.info(
                        "Iteration %d: LLM modified %d file(s): %s",
                        iterations_run, len(new_fixed),
                        sorted(new_fixed)[:10],
                    )
                else:
                    logger.warning(
                        "Iteration %d: LLM modified 0 files "
                        "(sent %d issues but no file edits detected in output).",
                        iterations_run, len(new_issues),
                    )

                # 5. Extract escalations
                escalations = extract_escalations(result.stdout)
                all_escalations.extend(escalations)

                if escalations:
                    esc_path = save_escalation_report(
                        escalations=escalations,
                        epic_id=epic_id,
                        project_path=self.project_path,
                        iteration=iterations_run,
                        total_issues=total_issues_accumulated,
                        auto_fixed=len(fixed_files),
                    )
                    last_esc_path = str(esc_path)
                    logger.info("Escalation report: %s", esc_path)

                # 6. Re-test (if qa executor available)
                pass_rate = self._run_retest(epic_id, iteration)

                if pass_rate is not None:
                    if prev_pass_rate is not None and pass_rate < prev_pass_rate:
                        logger.warning(
                            "Regression detected: pass rate dropped %.1f%% → %.1f%%",
                            prev_pass_rate,
                            pass_rate,
                        )
                    prev_pass_rate = pass_rate
                    final_pass_rate = pass_rate

                    if pass_rate >= 99.99:
                        logger.info("All tests pass after iteration %d — done.", iterations_run)
                        break

            # Honest accounting
            files_modified = len(fixed_files)
            issues_escalated = len(all_escalations)

            # Determine status
            if total_issues_accumulated == 0:
                status = "clean"
            elif files_modified == 0 and issues_escalated == 0:
                status = "unresolved"
            elif issues_escalated > 0:
                status = "escalated"
            elif files_modified > 0 and (final_pass_rate is None or final_pass_rate >= 99.99):
                status = "fixed"
            else:
                status = "partial"

            report_path = save_remediation_report(
                epic_id=epic_id,
                project_path=self.project_path,
                status=status,
                iterations=iterations_run,
                issues_found=total_issues_accumulated,
                issues_fixed=files_modified,
                issues_escalated=issues_escalated,
                pass_rate=final_pass_rate if final_pass_rate is not None else 0.0,
            )

            logger.info(
                "Remediation complete for epic %s: status=%s, "
                "issues=%d, files_modified=%d, escalated=%d",
                epic_id, status, total_issues_accumulated,
                files_modified, issues_escalated,
            )

            outputs = {
                "status": status,
                "report_path": str(report_path),
                "escalation_path": last_esc_path,
                "iterations": iterations_run,
                "issues_found": total_issues_accumulated,
                "issues_fixed": files_modified,
                "issues_escalated": issues_escalated,
                "files_modified": files_modified,
                "retest_pass_rate": final_pass_rate if final_pass_rate is not None else 0.0,
            }

            # Only treat as a successful phase when the code was either clean
            # from the start or the handler actually resolved every finding.
            # `escalated` with zero fixes, `unresolved`, and `partial` all
            # leave real work for a human — surfacing them as success=True
            # causes the runner (and any downstream "project complete" log)
            # to misreport the epic as done.
            success_statuses = {"clean", "fixed"}
            if status in success_statuses:
                return PhaseResult.ok(outputs)

            error_msg = (
                f"QA remediation needs human review: status={status}, "
                f"issues_found={total_issues_accumulated}, "
                f"files_modified={files_modified}, "
                f"escalated={issues_escalated}. "
                f"See escalation report: {last_esc_path}"
            )
            return PhaseResult(success=False, error=error_msg, outputs=outputs)

        except Exception as e:
            logger.error("QA remediation failed for epic %s: %s", epic_id, e, exc_info=True)
            return PhaseResult.fail(f"QA remediation error: {e}")

    # ------------------------------------------------------------------
    # DV scan
    # ------------------------------------------------------------------

    def _resolve_epic_source_files(
        self,
        epic_id: EpicId,
        project_stacks: tuple[str, ...] = (),
    ) -> list[Path]:
        """Resolve source files for all stories in an epic.

        Uses story File List sections to discover files. Filters out
        non-scannable files via DVFileFilter, validates existence, deduplicates.

        Returns:
            List of existing, non-test source file paths.

        """
        try:
            from bmad_assist.compiler.source_context import (
                _extract_file_list_section,
                extract_file_paths_from_section,
            )
            from bmad_assist.core.paths import get_paths
        except (ImportError, RuntimeError) as e:
            logger.debug("Cannot resolve epic source files: %s", e)
            return []

        try:
            stories_dir = get_paths().stories_dir
        except RuntimeError:
            logger.debug("Paths not initialized, skipping file resolution")
            return []

        if not stories_dir.exists():
            return []

        # Find all story files for this epic
        pattern = f"{epic_id}-*-*.md"
        story_files = sorted(stories_dir.glob(pattern))
        if not story_files:
            return []

        # Collect file paths from all stories
        all_paths: set[str] = set()
        for sf in story_files:
            try:
                content = sf.read_text(encoding="utf-8")
            except OSError:
                continue
            section = _extract_file_list_section(content)
            if section:
                paths = extract_file_paths_from_section(section)
                all_paths.update(paths)

        if not all_paths:
            logger.info("No files found in story File Lists for epic %s", epic_id)
            return []

        # Stack-aware file filter (excludes tests, configs, binaries, etc.)
        try:
            from bmad_assist.deep_verify.file_filter import DVFileFilter

            file_filter = DVFileFilter(stacks=list(project_stacks))
        except (ImportError, RuntimeError):
            file_filter = None

        # Resolve and filter
        project_root = self.project_path.resolve()
        resolved: list[Path] = []
        seen: set[Path] = set()

        for rel_path in sorted(all_paths):
            # Stack-aware exclusion (tests, configs, binaries, etc.)
            if file_filter and file_filter.should_exclude(rel_path):
                continue

            path = (self.project_path / rel_path).resolve()

            # Path traversal prevention
            try:
                path.relative_to(project_root)
            except ValueError:
                continue

            if path in seen or not path.is_file():
                continue

            seen.add(path)
            resolved.append(path)

        logger.debug(
            "Epic %s: resolved %d source files from %d story files "
            "(filtered %d non-scannable)",
            epic_id, len(resolved), len(story_files),
            len(all_paths) - len(resolved),
        )
        return resolved

    def _run_dv_scan(
        self,
        epic_id: EpicId,
        files: list[Path],
        iteration: int,
        project_stacks: tuple[str, ...] = (),
    ) -> int:
        """Run Deep Verify batch scan on files and save reports.

        Uses batch mode (multi-turn sessions) to scan all files efficiently.
        Saves reports to deep-verify directory so collect_epic_issues()
        picks them up.

        Args:
            epic_id: Epic identifier.
            files: Source files to scan.
            iteration: Current iteration number (1-based).
            project_stacks: Detected project stacks for pattern selection.

        Returns:
            Total number of DV findings across all files.

        """
        if not files:
            logger.debug("No files to DV scan for epic %s iteration %d", epic_id, iteration)
            return 0

        # Check if DV is enabled
        dv_config = getattr(self.config, "deep_verify", None)
        if dv_config is None or not dv_config.enabled:
            logger.debug("Deep Verify disabled, skipping scan")
            return 0

        try:
            from bmad_assist.deep_verify.integration.code_review_hook import (
                run_deep_verify_code_review_batch,
            )
            from bmad_assist.deep_verify.integration.reports import (
                save_deep_verify_batch_report,
            )
        except ImportError as e:
            logger.warning("Deep Verify not available: %s", e)
            return 0

        # Get output directory for DV reports
        dv_dir = self.project_path / "_bmad-output" / "implementation-artifacts" / "deep-verify"
        dv_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running DV batch scan: %d files for epic %s (iteration %d)",
            len(files), epic_id, iteration,
        )

        # Build batch file list
        batch_files: list[tuple[Path, str]] = []
        for file_path in files:
            try:
                code_content = file_path.read_text(encoding="utf-8")
                batch_files.append((file_path, code_content))
            except OSError as e:
                logger.warning("Cannot read %s for DV scan: %s", file_path, e)

        if not batch_files:
            return 0

        scan_start = time.perf_counter()

        try:
            batch_results = asyncio.run(
                run_deep_verify_code_review_batch(
                    files=batch_files,
                    config=self.config,
                    project_path=self.project_path,
                    epic_num=epic_id,
                    story_num=f"remediate-iter{iteration}",
                    base_timeout=90,
                    project_stacks=project_stacks,
                    file_hunk_ranges=None,
                )
            )
        except Exception as e:
            logger.warning("DV batch scan failed for epic %s: %s", epic_id, e)
            return 0

        total_findings = sum(len(r.findings) for r in batch_results.values())

        # Save single consolidated report with per-file breakdown
        if total_findings > 0:
            save_deep_verify_batch_report(
                batch_results=batch_results,
                epic=epic_id,
                story=f"remediate-iter{iteration}",
                output_dir=dv_dir,
                phase_type="remediation",
            )

        scan_duration = time.perf_counter() - scan_start
        logger.info(
            "DV batch scan complete: %d findings across %d files (%.1fs)",
            total_findings, len(batch_files), scan_duration,
        )
        return total_findings

    # ------------------------------------------------------------------
    # Re-test
    # ------------------------------------------------------------------

    def _run_retest(
        self,
        epic_id: Any,
        iteration: int,
    ) -> float | None:
        """Attempt re-test using existing QA executor.

        Returns pass rate as float, or None if retest was skipped/unavailable.
        """
        if iteration == 0:
            return None

        try:
            from bmad_assist.qa.executor import execute_qa_plan

            retest_result = execute_qa_plan(
                config=self.config,
                project_path=self.project_path,
                epic_id=epic_id,
                retry=True,
                batch_mode="batch",
            )
            return retest_result.pass_rate
        except Exception as e:
            logger.warning("Re-test failed (continuing): %s", e)
            return None

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_remediate_prompt(
        self,
        issues: list[Any],
        fixed_files: set[str],
        epic_id: Any,
        safety_cap: float,
    ) -> str:
        """Build the LLM prompt for remediation.

        Loads the XML template from qa/prompts/remediate.xml and populates
        dynamic sections (issues, fixed files, escalation markers).

        Args:
            issues: List of EpicIssue to fix.
            fixed_files: Files already modified (exclude from fixes).
            epic_id: Epic identifier.
            safety_cap: Max fraction of issues that can be AUTO-FIX.

        Returns:
            Prompt string.

        """
        # Build issues XML block
        issue_lines: list[str] = []
        for i, issue in enumerate(issues, 1):
            issue_lines.append(f'    <issue n="{i}" source="{issue.source}" severity="{issue.severity}">')
            issue_lines.append(f"      <description>{issue.description}</description>")
            if issue.file_path:
                issue_lines.append(f"      <file>{issue.file_path}</file>")
            if issue.context:
                truncated = issue.context[:2000]
                issue_lines.append(f"      <context>{truncated}</context>")
            issue_lines.append("    </issue>")
        issues_xml = "\n".join(issue_lines)

        # Build fixed files section (or empty string)
        if fixed_files:
            ff_lines = ["  <already_fixed_files>"]
            for fp in sorted(fixed_files):
                ff_lines.append(f"    <file>{fp}</file>")
            ff_lines.append("  </already_fixed_files>")
            fixed_files_section = "\n".join(ff_lines)
        else:
            fixed_files_section = ""

        template = get_remediate_prompt()
        return template.format(
            epic_id=epic_id,
            issues_count=len(issues),
            safety_cap_pct=round(safety_cap * 100),
            escalation_start=REMEDIATE_ESCALATIONS_START,
            escalation_end=REMEDIATE_ESCALATIONS_END,
            fixed_files_section=fixed_files_section,
            issues_xml=issues_xml,
        )
