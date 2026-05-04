"""Regression tests for dashboard story phase truthfulness.

The dashboard must not display an in-progress story phase based solely on
state.yaml. A matching running run log is required to confirm live execution.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from bmad_assist.core.loop.run_tracking import CurrentPhase, RunLog, RunStatus, save_run_log
from bmad_assist.core.state import Phase, State, get_state_path, save_state
from bmad_assist.dashboard.server import DashboardServer


def _create_server(tmp_path: Path) -> DashboardServer:
    impl_dir = tmp_path / "_bmad-output" / "implementation-artifacts"
    impl_dir.mkdir(parents=True, exist_ok=True)
    (impl_dir / "sprint-status.yaml").write_text("epics: []\n", encoding="utf-8")
    return DashboardServer(project_root=tmp_path)


def _save_current_state(tmp_path: Path) -> None:
    save_state(
        State(
            current_epic=7,
            current_story="7.3",
            current_phase=Phase.DEV_STORY,
            updated_at=datetime.now(UTC),
            phase_started_at=datetime.now(UTC),
        ),
        get_state_path(project_root=tmp_path),
    )


def _story_phase_statuses(server: DashboardServer) -> dict[str, str]:
    server.get_sprint_status = lambda: {
        "epics": [
            {
                "id": 7,
                "title": "Epic 7",
                "status": "in-progress",
                "stories": [
                    {
                        "id": "3",
                        "title": "Story 7.3",
                        "status": "backlog",
                    }
                ],
            }
        ]
    }
    stories = server.get_stories()["epics"][0]["stories"][0]["phases"]
    return {phase["id"]: phase["status"] for phase in stories}


def test_story_phase_stays_pending_without_active_run(tmp_path: Path) -> None:
    """Stale state.yaml alone must not mark dev_story as running."""
    server = _create_server(tmp_path)
    _save_current_state(tmp_path)

    phases = _story_phase_statuses(server)

    assert phases["dev_story"] == "pending"
    assert all(status != "in-progress" for status in phases.values())


def test_story_phase_uses_matching_running_run(tmp_path: Path) -> None:
    """A matching running run log should mark dev_story as in-progress."""
    server = _create_server(tmp_path)
    _save_current_state(tmp_path)
    assist_dir = tmp_path / ".bmad-assist"
    assist_dir.mkdir(parents=True, exist_ok=True)
    (assist_dir / "running.lock").write_text(
        f"{os.getpid()}\n{datetime.now(UTC).isoformat()}\n",
        encoding="utf-8",
    )

    save_run_log(
        RunLog(
            run_id="truth123",
            started_at=datetime.now(UTC),
            status=RunStatus.RUNNING,
            epic=7,
            story=3,
            project_path=str(tmp_path),
            current_phase=CurrentPhase(
                phase="dev_story",
                started_at=datetime.now(UTC),
                provider="openai",
                model="gpt-5",
            ),
        ),
        tmp_path,
    )

    phases = _story_phase_statuses(server)

    assert phases["dev_story"] == "in-progress"
