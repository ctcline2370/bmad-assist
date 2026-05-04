from unittest.mock import MagicMock

from bmad_assist.bmad.parser import EpicStory
from bmad_assist.bmad.state_reader import ProjectState
from bmad_assist.core.epic_lifecycle import get_epic_lifecycle_status
from bmad_assist.core.paths import _reset_paths, get_paths, init_paths
from bmad_assist.core.state import Phase


def _build_project_state() -> ProjectState:
    stories = [
        EpicStory(number="7.1", title="Story 7.1", status="done"),
        EpicStory(number="7.2", title="Story 7.2", status="done"),
    ]
    return ProjectState(
        epics=[],
        all_stories=stories,
        completed_stories=["7.1", "7.2"],
        current_epic=7,
        current_story="7.2",
        bmad_path="docs",
    )


def test_sprint_status_done_does_not_replace_retrospective_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BMAD_QA_ENABLED", raising=False)
    _reset_paths()
    try:
        init_paths(tmp_path)
        project_state = _build_project_state()
        paths = get_paths()
        paths.sprint_status_file.parent.mkdir(parents=True, exist_ok=True)
        paths.sprint_status_file.write_text(
            "development_status:\n  epic-7-retrospective: done\n",
            encoding="utf-8",
        )

        status = get_epic_lifecycle_status(7, project_state, MagicMock(), tmp_path)

        assert status.all_stories_done is True
        assert status.retro_completed is False
        assert status.next_phase == Phase.RETROSPECTIVE
        assert status.describe() == "ready for retrospective"
        assert status.is_fully_completed is False
    finally:
        _reset_paths()


def test_retrospective_artifact_marks_epic_as_fully_completed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BMAD_QA_ENABLED", raising=False)
    _reset_paths()
    try:
        init_paths(tmp_path)
        project_state = _build_project_state()
        paths = get_paths()
        paths.retrospectives_dir.mkdir(parents=True, exist_ok=True)
        (paths.retrospectives_dir / "epic-7-retro-20260420.md").write_text(
            "# Epic 7 retrospective\n",
            encoding="utf-8",
        )

        status = get_epic_lifecycle_status(7, project_state, MagicMock(), tmp_path)

        assert status.retro_completed is True
        assert status.next_phase is None
        assert status.describe() == "fully completed"
        assert status.is_fully_completed is True
    finally:
        _reset_paths()


def test_legacy_root_retrospective_artifact_marks_epic_as_fully_completed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BMAD_QA_ENABLED", raising=False)
    _reset_paths()
    try:
        init_paths(tmp_path)
        project_state = _build_project_state()
        paths = get_paths()
        paths.implementation_artifacts.mkdir(parents=True, exist_ok=True)
        (paths.implementation_artifacts / "epic-7-retro-20260420.md").write_text(
            "# Epic 7 retrospective\n",
            encoding="utf-8",
        )

        status = get_epic_lifecycle_status(7, project_state, MagicMock(), tmp_path)

        assert status.retro_completed is True
        assert status.next_phase is None
        assert status.describe() == "fully completed"
        assert status.is_fully_completed is True
    finally:
        _reset_paths()
