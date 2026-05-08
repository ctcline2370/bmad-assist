"""Tests for Epic File Parser (Story 2.2).

Tests cover all acceptance criteria:
- AC1: Parse standard story sections (## and ### headers)
- AC2: Extract story estimates
- AC3: Handle epic file with no stories
- AC4: Handle malformed story headers
- AC5: Extract epic metadata from frontmatter
- AC6: Handle consolidated epics.md file
- AC7: Extract story dependencies
- AC8: Return type consistency
- AC9: Infer status from content
- AC9b: Status priority (explicit Status field wins)
- AC10: Handle story status from acceptance criteria checkboxes
"""

import logging
from pathlib import Path

import pytest

from bmad_assist.bmad.parser import (
    EpicDocument,
    EpicStory,
    extract_epic_markdown,
    extract_markdown_sections,
    parse_epic_file,
)


class TestParseStandardStorySections:
    """Test AC1: Parse standard story sections."""

    def test_parse_h2_story_headers(self, tmp_path: Path) -> None:
        """Parse stories with ## Story X.Y: Title headers."""
        content = """---
epic_num: 2
title: BMAD File Integration
---

# Epic 2: BMAD File Integration

## Story 2.1: Markdown Frontmatter Parser

**As a** developer...

**Estimate:** 2 SP

---

## Story 2.2: Epic File Parser

**As a** developer...

**Estimate:** 3 SP
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert len(result.stories) == 2
        assert result.stories[0].number == "2.1"
        assert result.stories[0].title == "Markdown Frontmatter Parser"
        assert result.stories[1].number == "2.2"
        assert result.stories[1].title == "Epic File Parser"

    def test_parse_h3_story_headers(self, tmp_path: Path) -> None:
        """Parse stories with ### Story X.Y: Title headers."""
        content = """---
epic_num: 2
---

### Story 2.1: First Story

Content here.

### Story 2.2: Second Story

More content.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert len(result.stories) == 2
        assert result.stories[0].number == "2.1"
        assert result.stories[1].number == "2.2"

    def test_parse_mixed_h2_h3_headers(self, tmp_path: Path) -> None:
        """Parse stories with mix of ## and ### headers."""
        content = """---
---

## Story 1.1: H2 Story

Content.

### Story 1.2: H3 Story

Content.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert len(result.stories) == 2
        assert result.stories[0].number == "1.1"
        assert result.stories[0].title == "H2 Story"
        assert result.stories[1].number == "1.2"
        assert result.stories[1].title == "H3 Story"

    def test_parse_story_with_large_numbers(self, tmp_path: Path) -> None:
        """Parse story numbers > 9 (e.g., Story 12.15)."""
        content = """---
---

## Story 12.15: Large Number Story

Content here.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert len(result.stories) == 1
        assert result.stories[0].number == "12.15"
        assert result.stories[0].title == "Large Number Story"


class TestExtractStoryEstimate:
    """Test AC2: Extract story estimates."""

    def test_extract_estimate_format(self, tmp_path: Path) -> None:
        """Extract estimate from **Estimate:** 3 SP format."""
        content = """---
---

## Story 1.1: Test Story

**Estimate:** 3 SP
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].estimate == 3

    def test_extract_story_points_format(self, tmp_path: Path) -> None:
        """Extract estimate from **Story Points:** 5 format."""
        content = """---
---

## Story 1.1: Test Story

**Story Points:** 5
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].estimate == 5

    def test_extract_estimate_without_sp_suffix(self, tmp_path: Path) -> None:
        """Extract estimate without SP suffix."""
        content = """---
---

## Story 1.1: Test Story

**Estimate:** 8
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].estimate == 8

    def test_missing_estimate_returns_none(self, tmp_path: Path) -> None:
        """Missing estimate returns None."""
        content = """---
---

## Story 1.1: Test Story

No estimate here.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].estimate is None


class TestParseEpicNoStories:
    """Test AC3: Handle epic file with no stories."""

    def test_epic_with_only_header(self, tmp_path: Path) -> None:
        """Epic file with only epic header returns empty stories list."""
        content = """---
epic_num: 5
---

# Epic 5: Power-Prompts Engine

**Goal:** System can load and inject context-aware prompts...

**FRs:** FR22, FR23, FR24, FR25
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories == []
        assert result.epic_num == 5

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty file returns empty stories list."""
        content = """---
---
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories == []


class TestMalformedStoryHeaders:
    """Test AC4: Handle malformed story headers."""

    def test_skip_malformed_headers(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Malformed headers are skipped with warning."""
        content = """---
---

## Story 2.1 - Missing Colon

Invalid content.

## Story 2.2: Valid Story

Valid content.

## Invalid: No Number

More invalid.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        with caplog.at_level(logging.WARNING):
            result = parse_epic_file(path)

        # Only valid story should be parsed
        assert len(result.stories) == 1
        assert result.stories[0].number == "2.2"
        assert result.stories[0].title == "Valid Story"

    def test_story_without_epic_prefix(self, tmp_path: Path) -> None:
        """Story without epic number prefix is skipped."""
        content = """---
---

## Story: No Epic Prefix

Invalid.

## Story 1.1: Valid Story

Valid.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert len(result.stories) == 1
        assert result.stories[0].number == "1.1"


class TestExtractEpicMetadata:
    """Test AC5: Extract epic metadata from frontmatter."""

    def test_extract_epic_metadata_from_frontmatter(self, tmp_path: Path) -> None:
        """Extract epic_num, title, status from frontmatter."""
        content = """---
epic_num: 2
title: BMAD File Integration
status: in-progress
---

## Story 2.1: Markdown Frontmatter Parser
...
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.epic_num == 2
        assert result.title == "BMAD File Integration"
        assert result.status == "in-progress"

    def test_missing_frontmatter_fields(self, tmp_path: Path) -> None:
        """Missing frontmatter fields return None."""
        content = """---
---

## Story 2.1: Test
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.epic_num is None
        assert result.title is None
        assert result.status is None


class TestConsolidatedEpicsFile:
    """Test AC6: Handle consolidated epics.md file."""

    def test_multi_epic_file_returns_none_metadata(self, tmp_path: Path) -> None:
        """Multi-epic file returns epic_num=None, title=None, status=None."""
        content = """---
total_epics: 2
---

# Epic 1: Project Foundation

## Story 1.1: Project Initialization
**Estimate:** 2 SP

## Story 1.2: Configuration Models
**Estimate:** 3 SP

# Epic 2: BMAD File Integration

## Story 2.1: Markdown Frontmatter Parser
**Estimate:** 2 SP

## Story 2.2: Epic File Parser
**Estimate:** 3 SP
"""
        path = tmp_path / "epics.md"
        path.write_text(content)

        result = parse_epic_file(path)

        # Multi-epic file should have None for single-epic metadata
        assert result.epic_num is None
        assert result.title is None
        assert result.status is None

        # But should contain all stories from all epics
        assert len(result.stories) == 4
        assert result.stories[0].number == "1.1"
        assert result.stories[1].number == "1.2"
        assert result.stories[2].number == "2.1"
        assert result.stories[3].number == "2.2"

    def test_multi_epic_file_ignores_epic_list_without_blank_line(self, tmp_path: Path) -> None:
        """Epic List followed immediately by Epic 1 still yields real stories."""
        content = """# Feature Plan

## Epic List
### Epic 1: Project Foundation

#### Story 1.1: Project Initialization
**Estimate:** 2 SP

### Epic 2: BMAD File Integration

#### Story 2.1: Markdown Frontmatter Parser
**Estimate:** 3 SP
"""
        path = tmp_path / "epics.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert [story.number for story in result.stories] == ["1.1", "2.1"]

    def test_stories_ordered_by_appearance(self, tmp_path: Path) -> None:
        """Stories are ordered by appearance in file."""
        content = """---
---

# Epic 2: Second Epic

## Story 2.1: Second Epic First Story

# Epic 1: First Epic

## Story 1.1: First Epic First Story
"""
        path = tmp_path / "epics.md"
        path.write_text(content)

        result = parse_epic_file(path)

        # Order should be by appearance, not by epic number
        assert len(result.stories) == 2
        assert result.stories[0].number == "2.1"
        assert result.stories[1].number == "1.1"


class TestMarkdownExtractionHelpers:
    """Tests for helper extraction utilities used by create-story."""

    def test_extract_epic_markdown_returns_target_block(self) -> None:
        """Extracts only the requested epic from consolidated markdown."""
        content = """# Epic 1: Project Foundation

## Story 1.1: Initialize Project

Alpha content.

# Epic 2: BMAD File Integration

## Story 2.1: Markdown Frontmatter Parser

Beta content.
"""

        extracted = extract_epic_markdown(content, 2)

        assert extracted is not None
        assert extracted.startswith("# Epic 2: BMAD File Integration")
        assert "## Story 2.1: Markdown Frontmatter Parser" in extracted
        assert "# Epic 1: Project Foundation" not in extracted

    def test_extract_epic_markdown_skips_epic_list_section_heading(self) -> None:
        """Document headings like Epic List do not hide the first real epic."""
        content = """# Feature Plan

## Epic Ownership and Collision Analysis

This is planning guidance, not an epic.

## Epic List
### Epic 1: Foundation Governance

#### Story 1.1: Foundation Readiness

Alpha content.

### Epic 2: Domain Substrate

#### Story 2.1: Persistence Standards

Beta content.
"""

        extracted = extract_epic_markdown(content, 1)

        assert extracted is not None
        assert extracted.startswith("### Epic 1: Foundation Governance")
        assert "#### Story 1.1: Foundation Readiness" in extracted
        assert "## Epic List" not in extracted
        assert "### Epic 2: Domain Substrate" not in extracted

    def test_extract_markdown_sections_preserves_source_order(self) -> None:
        """Selected sections preserve original document order."""
        content = """## Story

Story body.

## Acceptance Criteria

- AC 1

### File List

- `src/example.py`
"""

        extracted = extract_markdown_sections(
            content,
            ["Acceptance Criteria", "Story", "File List"],
        )

        assert extracted.index("## Story") < extracted.index("## Acceptance Criteria")
        assert extracted.index("## Acceptance Criteria") < extracted.index("### File List")

    def test_extract_markdown_sections_avoids_nested_duplication(self) -> None:
        """Parent sections win when parent and child headings are both requested."""
        content = """## Story

Story body.

### Completion Notes List

- implemented

## Acceptance Criteria

- AC 1
"""

        extracted = extract_markdown_sections(
            content,
            ["Story", "Completion Notes List"],
        )

        assert extracted.count("### Completion Notes List") == 1
        assert extracted.startswith("## Story")


class TestDependencyExtraction:
    """Test AC7: Extract story dependencies."""

    def test_extract_story_dependencies(self, tmp_path: Path) -> None:
        """Extract dependencies from **Dependencies:** line."""
        content = """---
---

## Story 3.5: Resume Interrupted Loop

**Dependencies:** Story 3.2 (Atomic State Persistence), Story 3.4 (Loop Position Tracking)
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].dependencies == ["3.2", "3.4"]

    def test_extract_dependencies_short_format(self, tmp_path: Path) -> None:
        """Extract dependencies in short format (just numbers)."""
        content = """---
---

## Story 2.1: Test Story

**Dependencies:** 1.2, 1.3, 1.5
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].dependencies == ["1.2", "1.3", "1.5"]

    def test_no_dependencies_returns_empty_list(self, tmp_path: Path) -> None:
        """No dependencies returns empty list."""
        content = """---
---

## Story 1.1: Test Story

No dependencies here.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].dependencies == []


class TestReturnTypes:
    """Test AC8: Return type consistency."""

    def test_epic_document_structure(self, tmp_path: Path) -> None:
        """EpicDocument has correct structure."""
        content = """---
epic_num: 1
title: Test Epic
status: complete
---

## Story 1.1: Test Story
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert isinstance(result, EpicDocument)
        assert isinstance(result.epic_num, int)
        assert isinstance(result.title, str)
        assert isinstance(result.status, str)
        assert isinstance(result.stories, list)
        assert isinstance(result.path, str)

    def test_epic_story_structure(self, tmp_path: Path) -> None:
        """EpicStory has correct structure."""
        content = """---
---

## Story 2.1: Test Story

**Estimate:** 3 SP
**Status:** in-progress
**Dependencies:** 1.1, 1.2

- [x] AC1: First
- [ ] AC2: Second
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)
        story = result.stories[0]

        assert isinstance(story, EpicStory)
        assert isinstance(story.number, str)
        assert isinstance(story.title, str)
        assert isinstance(story.estimate, int)
        assert isinstance(story.status, str)
        assert isinstance(story.dependencies, list)
        assert isinstance(story.completed_criteria, int)
        assert isinstance(story.total_criteria, int)


class TestStatusInference:
    """Test AC9: Infer status from content."""

    def test_extract_explicit_status(self, tmp_path: Path) -> None:
        """Extract status from **Status:** field."""
        content = """---
---

## Story 2.1: Markdown Frontmatter Parser
**Status:** done

## Story 2.2: Epic File Parser
**Status:** in-progress
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].status == "done"
        assert result.stories[1].status == "in-progress"

    def test_missing_status_returns_none(self, tmp_path: Path) -> None:
        """Missing status returns None."""
        content = """---
---

## Story 1.1: Test Story

No status here.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].status is None

    def test_extract_multi_word_status(self, tmp_path: Path) -> None:
        """Extract multi-word status values correctly."""
        content = """---
---

## Story 1.1: Test
**Status:** Ready for Review

## Story 1.2: Test
**Status:** Waiting for Dependencies
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].status == "Ready for Review"
        assert result.stories[1].status == "Waiting for Dependencies"


class TestStatusPriority:
    """Test AC9b: Status field takes priority over checkbox counts."""

    def test_explicit_status_overrides_checkboxes(self, tmp_path: Path) -> None:
        """Explicit **Status:** field wins over checkbox inference."""
        content = """---
---

## Story 2.1: Parser
**Status:** done

**Acceptance Criteria:**
- [ ] AC1: Parse frontmatter
- [ ] AC2: Handle errors
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)
        story = result.stories[0]

        # Explicit status should be "done"
        assert story.status == "done"
        # But checkboxes are tracked separately
        assert story.completed_criteria == 0
        assert story.total_criteria == 2


class TestAcceptanceCriteriaCheckboxes:
    """Test AC10: Handle story status from acceptance criteria checkboxes."""

    def test_count_checked_and_unchecked(self, tmp_path: Path) -> None:
        """Count checked and unchecked acceptance criteria."""
        content = """---
---

## Story 2.1: Markdown Frontmatter Parser

**Acceptance Criteria:**
- [x] AC1: Parse valid frontmatter
- [x] AC2: Parse file without frontmatter
- [ ] AC3: Handle malformed frontmatter
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].completed_criteria == 2
        assert result.stories[0].total_criteria == 3

    def test_all_checked(self, tmp_path: Path) -> None:
        """All criteria checked."""
        content = """---
---

## Story 1.1: Test

- [x] AC1
- [x] AC2
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].completed_criteria == 2
        assert result.stories[0].total_criteria == 2

    def test_none_checked(self, tmp_path: Path) -> None:
        """No criteria checked."""
        content = """---
---

## Story 1.1: Test

- [ ] AC1
- [ ] AC2
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].completed_criteria == 0
        assert result.stories[0].total_criteria == 2

    def test_no_checkboxes_returns_none(self, tmp_path: Path) -> None:
        """No checkboxes returns None for both counts."""
        content = """---
---

## Story 1.1: Test

No checkboxes here.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].completed_criteria is None
        assert result.stories[0].total_criteria is None

    def test_uppercase_x_in_checkbox(self, tmp_path: Path) -> None:
        """Handle uppercase [X] in checkboxes."""
        content = """---
---

## Story 1.1: Test

- [X] AC1 (uppercase)
- [x] AC2 (lowercase)
- [ ] AC3
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].completed_criteria == 2
        assert result.stories[0].total_criteria == 3


class TestLogging:
    """Test logging behavior for malformed headers."""

    def test_malformed_header_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed headers trigger warning with correct format."""
        content = """---
---

## Story 2.1: Valid Story

Content.

## Story Missing Colon Here

Invalid.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        with caplog.at_level(logging.WARNING, logger="bmad_assist.bmad.parser"):
            result = parse_epic_file(path)

        # Should only parse valid story
        assert len(result.stories) == 1
        # No warnings should be logged since the malformed header doesn't match
        # the regex pattern at all


class TestRealEpicsFile:
    """Test with sample project epics.md fixture file."""

    def test_parse_real_epics_file(self) -> None:
        """Parse fixture epics.md file (60 stories expected)."""
        epics_path = Path(__file__).parents[2] / "tests/fixtures/bmad-sample-project/docs/epics.md"

        assert epics_path.exists(), f"Fixture file not found: {epics_path}"

        result = parse_epic_file(epics_path)

        # Multi-epic file should have None metadata
        assert result.epic_num is None
        assert result.title is None
        assert result.status is None

        # Should have 60 stories according to frontmatter
        assert len(result.stories) == 60

        # Check first story
        assert result.stories[0].number == "1.1"
        assert "Project Initialization" in result.stories[0].title

        # Check last story
        assert result.stories[-1].number == "9.8"

        # All stories should have estimates
        for story in result.stories:
            assert story.estimate is not None
            assert story.estimate > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            parse_epic_file(tmp_path / "nonexistent.md")

    def test_path_object_accepted(self, tmp_path: Path) -> None:
        """Path object is accepted as argument."""
        content = """---
---

## Story 1.1: Test
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)  # Path object

        assert len(result.stories) == 1

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """String path is accepted as argument."""
        content = """---
---

## Story 1.1: Test
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(str(path))  # String

        assert len(result.stories) == 1

    def test_path_stored_as_string(self, tmp_path: Path) -> None:
        """Path is stored as string in result."""
        content = """---
---

## Story 1.1: Test
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert isinstance(result.path, str)
        assert result.path == str(path)

    def test_story_title_stripped(self, tmp_path: Path) -> None:
        """Story title is stripped of whitespace."""
        content = """---
---

## Story 1.1:   Title with spaces

Content.
"""
        path = tmp_path / "epic.md"
        path.write_text(content)

        result = parse_epic_file(path)

        assert result.stories[0].title == "Title with spaces"
