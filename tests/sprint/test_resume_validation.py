"""Tests for sprint resume validation module."""

from pathlib import Path

import pytest

from bmad_assist.core.exceptions import StateError
from bmad_assist.core.state import Phase, State
from bmad_assist.sprint.resume_validation import (
    ResumeValidationResult,
    _find_next_incomplete_epic,
    _find_next_incomplete_story,
    _is_epic_done_in_sprint,
    _is_story_done_in_sprint,
    validate_resume_state,
)


def _write_sprint_status(tmp_path: Path, yaml_content: str) -> Path:
    """Write sprint-status.yaml to the canonical implementation-artifacts path."""
    impl_artifacts = tmp_path / "_bmad-output" / "implementation-artifacts"
    impl_artifacts.mkdir(parents=True, exist_ok=True)
    sprint_status_path = impl_artifacts / "sprint-status.yaml"
    sprint_status_path.write_text(yaml_content)
    return sprint_status_path


def _write_retro_artifact(tmp_path: Path, epic_id: int) -> Path:
    """Write a durable retrospective artifact for the given epic."""
    retrospectives_dir = tmp_path / "_bmad-output" / "implementation-artifacts" / "retrospectives"
    retrospectives_dir.mkdir(parents=True, exist_ok=True)
    retro_path = retrospectives_dir / f"epic-{epic_id}-retro-20260101_000000.md"
    retro_path.write_text(f"# Epic {epic_id} retrospective\n")
    return retro_path


def _write_legacy_retro_artifact(tmp_path: Path, epic_id: int) -> Path:
    """Write a durable retrospective artifact to the legacy root location."""
    implementation_artifacts = tmp_path / "_bmad-output" / "implementation-artifacts"
    implementation_artifacts.mkdir(parents=True, exist_ok=True)
    retro_path = implementation_artifacts / f"epic-{epic_id}-retro-20260101_000000.md"
    retro_path.write_text(f"# Epic {epic_id} retrospective\n")
    return retro_path


def _write_story_completion_artifacts(
    tmp_path: Path,
    story_id: str,
    *,
    include_test_review: bool = False,
) -> None:
    """Write durable story completion artifacts for the given story."""
    story_key = story_id.replace(".", "-")
    implementation_artifacts = tmp_path / "_bmad-output" / "implementation-artifacts"

    code_reviews_dir = implementation_artifacts / "code-reviews"
    code_reviews_dir.mkdir(parents=True, exist_ok=True)
    (code_reviews_dir / f"synthesis-{story_key}-20260101T000000Z.md").write_text(
        f"# Synthesis {story_id}\n"
    )

    if include_test_review:
        test_reviews_dir = implementation_artifacts / "test-reviews"
        test_reviews_dir.mkdir(parents=True, exist_ok=True)
        (test_reviews_dir / f"test-review-{story_key}-20260101T000000Z.md").write_text(
            "# Test Review\n\nQuality Score: 90/100\n"
        )


@pytest.fixture
def basic_sprint_status_yaml() -> str:
    """Sprint-status YAML with mixed done/in-progress entries."""
    return """
generated: '2026-01-01T00:00:00'
project: test-project
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first-story: done
  1-2-second-story: done
  epic-2: in-progress
  2-1-first-story: done
  2-2-second-story: in-progress
  2-3-third-story: backlog
  epic-3: backlog
  3-1-first-story: backlog
"""


@pytest.fixture
def all_done_sprint_status_yaml() -> str:
    """Sprint-status YAML where all epics, stories, and retrospectives are done."""
    return """
generated: '2026-01-01T00:00:00'
project: test-project
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first-story: done
  1-2-second-story: done
  epic-2: done
  epic-2-retrospective: done
  2-1-first-story: done
  2-2-second-story: done
  epic-3: done
  epic-3-retrospective: done
  3-1-first-story: done
"""


@pytest.fixture
def state_at_epic_2_story_1() -> State:
    """State pointing to Epic 2, Story 2.1, CREATE_STORY phase."""
    return State(
        current_epic=2,
        current_story="2.1",
        current_phase=Phase.CREATE_STORY,
        completed_stories=[],
        completed_epics=[1],
    )


@pytest.fixture
def epic_stories_loader():
    """Mock epic_stories_loader that returns predefined stories."""

    def loader(epic_id):
        stories = {
            1: ["1.1", "1.2"],
            2: ["2.1", "2.2", "2.3"],
            3: ["3.1"],
        }
        return stories.get(epic_id, [])

    return loader


class TestIsStoryDoneInSprint:
    """Tests for _is_story_done_in_sprint helper."""

    def test_story_done(self, tmp_path, basic_sprint_status_yaml):
        """Story marked done returns True."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_story_done_in_sprint("1.1", sprint_status) is True
        assert _is_story_done_in_sprint("2.1", sprint_status) is True

    def test_story_in_progress(self, tmp_path, basic_sprint_status_yaml):
        """Story marked in-progress returns False."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_story_done_in_sprint("2.2", sprint_status) is False

    def test_story_backlog(self, tmp_path, basic_sprint_status_yaml):
        """Story marked backlog returns False."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_story_done_in_sprint("2.3", sprint_status) is False

    def test_story_not_found(self, tmp_path, basic_sprint_status_yaml):
        """Story not in sprint-status returns False."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_story_done_in_sprint("99.1", sprint_status) is False


class TestIsEpicDoneInSprint:
    """Tests for _is_epic_done_in_sprint helper."""

    def test_epic_done_with_retro_done(self, tmp_path):
        """Epic is done only when both epic AND retrospective are done."""
        from bmad_assist.sprint.parser import parse_sprint_status

        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
"""
        path = tmp_path / "sprint-status.yaml"
        path.write_text(yaml_content)
        sprint_status = parse_sprint_status(path)

        assert _is_epic_done_in_sprint(1, sprint_status) is True

    def test_epic_done_but_retro_backlog(self, tmp_path):
        """Epic is NOT done if retrospective is backlog."""
        from bmad_assist.sprint.parser import parse_sprint_status

        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: backlog
"""
        path = tmp_path / "sprint-status.yaml"
        path.write_text(yaml_content)
        sprint_status = parse_sprint_status(path)

        assert _is_epic_done_in_sprint(1, sprint_status) is False

    def test_epic_done_no_retro_entry(self, tmp_path):
        """Epic is NOT done if no retrospective entry exists."""
        from bmad_assist.sprint.parser import parse_sprint_status

        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
"""
        path = tmp_path / "sprint-status.yaml"
        path.write_text(yaml_content)
        sprint_status = parse_sprint_status(path)

        assert _is_epic_done_in_sprint(1, sprint_status) is False

    def test_epic_in_progress(self, tmp_path, basic_sprint_status_yaml):
        """Epic marked in-progress returns False."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_epic_done_in_sprint(2, sprint_status) is False

    def test_epic_backlog(self, tmp_path, basic_sprint_status_yaml):
        """Epic marked backlog returns False."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        assert _is_epic_done_in_sprint(3, sprint_status) is False


class TestFindNextIncompleteStory:
    """Tests for _find_next_incomplete_story helper."""

    def test_finds_first_incomplete(self, tmp_path, basic_sprint_status_yaml):
        """Finds first story not done in sprint-status."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        result = _find_next_incomplete_story("2.1", ["2.1", "2.2", "2.3"], ["2.1"], sprint_status)
        assert result == "2.2"

    def test_skips_done_stories(self, tmp_path):
        """Skips stories that are done in sprint-status."""
        from bmad_assist.sprint.parser import parse_sprint_status

        yaml_content = """
generated: '2026-01-01'
development_status:
  1-1-first: done
  1-2-second: done
  1-3-third: in-progress
"""
        path = tmp_path / "sprint-status.yaml"
        path.write_text(yaml_content)
        sprint_status = parse_sprint_status(path)

        result = _find_next_incomplete_story("1.1", ["1.1", "1.2", "1.3"], [], sprint_status)
        assert result == "1.3"

    def test_returns_none_when_all_done(self, tmp_path, all_done_sprint_status_yaml):
        """Returns None when all remaining stories are done."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(all_done_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        result = _find_next_incomplete_story("1.1", ["1.1", "1.2"], [], sprint_status)
        assert result is None

    def test_done_story_without_required_artifacts_is_incomplete(self, tmp_path):
        """Does not skip a sprint-status done story without durable completion artifacts."""
        from bmad_assist.sprint.parser import parse_sprint_status

        yaml_content = """
generated: '2026-01-01'
development_status:
  1-1-first: done
  1-2-second: done
"""
        path = tmp_path / "sprint-status.yaml"
        path.write_text(yaml_content)
        sprint_status = parse_sprint_status(path)
        _write_story_completion_artifacts(tmp_path, "1.1", include_test_review=True)

        result = _find_next_incomplete_story(
            "1.1",
            ["1.1", "1.2"],
            [],
            sprint_status,
            project_path=tmp_path,
            require_test_review_for_done=True,
        )
        assert result == "1.2"


class TestFindNextIncompleteEpic:
    """Tests for _find_next_incomplete_epic helper."""

    def test_finds_first_incomplete(self, tmp_path, basic_sprint_status_yaml):
        """Finds first epic not done in sprint-status."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        result = _find_next_incomplete_epic(1, [1, 2, 3], [1], sprint_status, tmp_path)
        assert result == 2

    def test_skips_completed_epics(self, tmp_path, basic_sprint_status_yaml):
        """Skips epics in completed_epics list."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(basic_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)

        result = _find_next_incomplete_epic(1, [1, 2, 3], [1, 2], sprint_status, tmp_path)
        assert result == 3

    def test_returns_none_when_all_done(self, tmp_path, all_done_sprint_status_yaml):
        """Returns None when all epics are done."""
        from bmad_assist.sprint.parser import parse_sprint_status

        path = tmp_path / "sprint-status.yaml"
        path.write_text(all_done_sprint_status_yaml)
        sprint_status = parse_sprint_status(path)
        _write_retro_artifact(tmp_path, 1)
        _write_retro_artifact(tmp_path, 2)
        _write_retro_artifact(tmp_path, 3)

        result = _find_next_incomplete_epic(1, [1, 2, 3], [], sprint_status, tmp_path)
        assert result is None


class TestValidateResumeState:
    """Tests for validate_resume_state main function."""

    def test_no_sprint_status_file(self, tmp_path, state_at_epic_2_story_1, epic_stories_loader):
        """Returns unchanged state when no sprint-status file exists."""
        result = validate_resume_state(
            state_at_epic_2_story_1,
            tmp_path,
            [1, 2, 3],
            epic_stories_loader,
        )

        assert result.state == state_at_epic_2_story_1
        assert result.advanced is False
        assert result.stories_skipped == []
        assert result.epics_skipped == []
        assert result.project_complete is False

    def test_skips_done_story(
        self, tmp_path, state_at_epic_2_story_1, epic_stories_loader, basic_sprint_status_yaml
    ):
        """Advances state when current story is done in sprint-status."""
        # Setup: story 2.1 is marked done in sprint-status
        _write_sprint_status(tmp_path, basic_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)

        result = validate_resume_state(
            state_at_epic_2_story_1,
            tmp_path,
            [1, 2, 3],
            epic_stories_loader,
        )

        assert result.advanced is True
        assert "2.1" in result.stories_skipped
        assert result.state.current_story == "2.2"
        assert result.state.current_phase == Phase.CREATE_STORY
        assert "2.1" in result.state.completed_stories

    def test_done_story_without_synthesis_raises_when_completion_artifacts_required(
        self,
        tmp_path,
        state_at_epic_2_story_1,
        epic_stories_loader,
        basic_sprint_status_yaml,
    ):
        """Fails closed when sprint-status says done but synthesis evidence is missing."""
        _write_sprint_status(tmp_path, basic_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)

        with pytest.raises(
            StateError,
            match="Story 2.1 completion is incomplete .*durable code-review synthesis artifact is missing",
        ):
            validate_resume_state(
                state_at_epic_2_story_1,
                tmp_path,
                [1, 2, 3],
                epic_stories_loader,
                require_test_review_for_done=True,
            )

    def test_done_story_without_test_review_raises_when_test_review_required(
        self,
        tmp_path,
        state_at_epic_2_story_1,
        epic_stories_loader,
        basic_sprint_status_yaml,
    ):
        """Fails closed when TEA completion is missing a durable test-review artifact."""
        _write_sprint_status(tmp_path, basic_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)
        _write_story_completion_artifacts(tmp_path, "2.1", include_test_review=False)

        with pytest.raises(
            StateError,
            match="Story 2.1 completion is incomplete .*durable test-review artifact is missing",
        ):
            validate_resume_state(
                state_at_epic_2_story_1,
                tmp_path,
                [1, 2, 3],
                epic_stories_loader,
                require_test_review_for_done=True,
            )

    def test_test_review_phase_is_not_skipped_when_sprint_status_says_done(
        self,
        tmp_path,
        epic_stories_loader,
        basic_sprint_status_yaml,
    ):
        """Keeps a resume state at TEST_REVIEW when test-review evidence is missing."""
        _write_sprint_status(tmp_path, basic_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)
        _write_story_completion_artifacts(tmp_path, "2.1", include_test_review=False)

        state = State(
            current_epic=2,
            current_story="2.1",
            current_phase=Phase.TEST_REVIEW,
            completed_stories=[],
            completed_epics=[1],
        )

        result = validate_resume_state(
            state,
            tmp_path,
            [1, 2, 3],
            epic_stories_loader,
            require_test_review_for_done=True,
        )

        assert result.advanced is False
        assert result.state == state
        assert result.stories_skipped == []

    def test_skips_done_epic(self, tmp_path, epic_stories_loader):
        """Advances to next epic when current epic is done in sprint-status."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: backlog
  2-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
            completed_stories=[],
            completed_epics=[],
        )

        result = validate_resume_state(state, tmp_path, [1, 2], epic_stories_loader)

        assert result.advanced is True
        assert 1 in result.epics_skipped
        assert result.state.current_epic == 2
        assert result.state.current_story == "2.1"
        assert 1 in result.state.completed_epics

    def test_all_stories_done_without_epic_teardown_raises_state_error(
        self, tmp_path, epic_stories_loader
    ):
        """Fails closed when stories are done but epic teardown is incomplete."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  epic-2-retrospective: backlog
  2-1-first: done
  2-2-second: done
  2-3-third: done
  epic-3: backlog
  3-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.1",
            current_phase=Phase.CREATE_STORY,
            completed_stories=[],
            completed_epics=[1],
        )

        with pytest.raises(
            StateError,
            match="Epic 2 has all stories marked done in sprint-status, but epic teardown is incomplete",
        ):
            validate_resume_state(state, tmp_path, [1, 2, 3], epic_stories_loader)

    def test_configured_trace_teardown_phase_is_not_skipped_when_all_stories_done(
        self, tmp_path, epic_stories_loader
    ):
        """Allows configured TRACE teardown to execute before retrospective closes the epic."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  epic-2-retrospective: backlog
  2-1-first: done
  2-2-second: done
  2-3-third: done
  epic-3: backlog
  3-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.3",
            current_phase=Phase.TRACE,
            completed_stories=["2.1", "2.2", "2.3"],
            completed_epics=[1],
        )

        result = validate_resume_state(
            state,
            tmp_path,
            [1, 2, 3],
            epic_stories_loader,
            epic_teardown_phases=["trace", "tea_nfr_assess", "retrospective"],
        )

        assert result.advanced is False
        assert result.state == state
        assert result.stories_skipped == []

    def test_configured_tea_nfr_teardown_phase_is_not_skipped_when_all_stories_done(
        self, tmp_path, epic_stories_loader
    ):
        """Allows configured TEA NFR teardown to execute before retrospective closes the epic."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  epic-2-retrospective: backlog
  2-1-first: done
  2-2-second: done
  2-3-third: done
  epic-3: backlog
  3-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.3",
            current_phase=Phase.TEA_NFR_ASSESS,
            completed_stories=["2.1", "2.2", "2.3"],
            completed_epics=[1],
        )

        result = validate_resume_state(
            state,
            tmp_path,
            [1, 2, 3],
            epic_stories_loader,
            epic_teardown_phases=["trace", "tea_nfr_assess", "retrospective"],
        )

        assert result.advanced is False
        assert result.state == state
        assert result.stories_skipped == []

    def test_unconfigured_trace_teardown_phase_still_fails_closed(
        self, tmp_path, epic_stories_loader
    ):
        """Does not treat TRACE as resumable teardown unless the loop config includes it."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  epic-2-retrospective: backlog
  2-1-first: done
  2-2-second: done
  2-3-third: done
  epic-3: backlog
  3-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.3",
            current_phase=Phase.TRACE,
            completed_stories=["2.1", "2.2", "2.3"],
            completed_epics=[1],
        )

        with pytest.raises(
            StateError,
            match="Epic 2 has all stories marked done in sprint-status, but epic teardown is incomplete",
        ):
            validate_resume_state(
                state,
                tmp_path,
                [1, 2, 3],
                epic_stories_loader,
                epic_teardown_phases=["retrospective"],
            )

    def test_project_complete_all_done(
        self, tmp_path, epic_stories_loader, all_done_sprint_status_yaml
    ):
        """Detects project completion when all epics are done."""
        _write_sprint_status(tmp_path, all_done_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)
        _write_retro_artifact(tmp_path, 2)
        _write_retro_artifact(tmp_path, 3)

        state = State(
            current_epic=1,
            current_story="1.1",
            current_phase=Phase.CREATE_STORY,
            completed_stories=[],
            completed_epics=[],
        )

        result = validate_resume_state(state, tmp_path, [1, 2, 3], epic_stories_loader)

        assert result.project_complete is True
        assert result.advanced is True

    def test_no_changes_when_current_not_done(
        self, tmp_path, epic_stories_loader, basic_sprint_status_yaml
    ):
        """No changes when current story is not done."""
        _write_sprint_status(tmp_path, basic_sprint_status_yaml)
        _write_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.2",  # This is in-progress in sprint-status
            current_phase=Phase.DEV_STORY,
            completed_stories=["2.1"],
            completed_epics=[1],
        )

        result = validate_resume_state(state, tmp_path, [1, 2, 3], epic_stories_loader)

        assert result.advanced is False
        assert result.state == state

    def test_rejects_state_advanced_past_epic_without_durable_retrospective_artifact(
        self, tmp_path, epic_stories_loader
    ):
        """Fails closed when state advanced past an epic missing its durable retro artifact."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  2-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)

        state = State(
            current_epic=2,
            current_story="2.1",
            current_phase=Phase.CREATE_STORY,
            completed_stories=[],
            completed_epics=[1],
        )

        with pytest.raises(
            StateError,
            match="Epic 1 teardown is incomplete .*durable retrospective artifact is missing",
        ):
            validate_resume_state(state, tmp_path, [1, 2], epic_stories_loader)

    def test_accepts_state_advanced_past_epic_with_legacy_retro_artifact(
        self, tmp_path, epic_stories_loader
    ):
        """Allows prior-epic advancement when only the legacy retro artifact exists."""
        yaml_content = """
generated: '2026-01-01'
development_status:
  epic-1: done
  epic-1-retrospective: done
  1-1-first: done
  1-2-second: done
  epic-2: in-progress
  2-1-first: backlog
"""
        _write_sprint_status(tmp_path, yaml_content)
        _write_legacy_retro_artifact(tmp_path, 1)

        state = State(
            current_epic=2,
            current_story="2.1",
            current_phase=Phase.CREATE_STORY,
            completed_stories=[],
            completed_epics=[1],
        )

        result = validate_resume_state(state, tmp_path, [1, 2], epic_stories_loader)

        assert result.advanced is False
        assert result.state == state

    def test_result_summary(self):
        """ResumeValidationResult.summary() produces readable output."""
        state = State()

        # No changes
        result = ResumeValidationResult(
            state=state,
            stories_skipped=[],
            epics_skipped=[],
            advanced=False,
            project_complete=False,
        )
        assert "no changes" in result.summary()

        # Stories skipped
        result = ResumeValidationResult(
            state=state,
            stories_skipped=["1.1", "1.2"],
            epics_skipped=[],
            advanced=True,
            project_complete=False,
        )
        assert "2 done stories" in result.summary()

        # Project complete
        result = ResumeValidationResult(
            state=state, stories_skipped=[], epics_skipped=[1], advanced=True, project_complete=True
        )
        assert "project complete" in result.summary()
