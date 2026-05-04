"""Tests for code review workflow prompt filtering."""

from bmad_assist.compiler.workflows.code_review import _extract_modified_files_from_stat


class TestExtractModifiedFilesFromStat:
    """Tests for filtering generated and control-plane files from review context."""

    def test_excludes_generated_and_control_plane_files(self) -> None:
        """Only product source and tests should remain in the review set."""
        stat_output = """
 src/main.py | 20 +++++
 tests/test_main.py | 10 +++
 .codex/config.toml | 6 ++
 .agents/skills/index.json | 4 +
 .claude/settings.json | 2 +
 AGENTS.md | 8 ++
 CLAUDE.md | 3 +
 _bmad-output/implementation-artifacts/report.md | 11 +++
 .bmad-assist/state.yaml | 5 ++
 9 files changed, 69 insertions(+)
"""

        modified_files = _extract_modified_files_from_stat(stat_output)

        assert modified_files == [("src/main.py", 20), ("tests/test_main.py", 10)]

    def test_excludes_control_plane_rename_targets(self) -> None:
        """Renamed control-plane paths should not leak into the review prompt."""
        stat_output = """
 .codex/old-config.toml => .codex/config.toml | 5 ++
 src/old.py => src/new.py | 4 ++
 2 files changed, 9 insertions(+)
"""

        modified_files = _extract_modified_files_from_stat(stat_output)

        assert modified_files == [("src/new.py", 4)]
