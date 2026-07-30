"""Unit tests for durable Agent-run lifecycle operations."""

from pathlib import Path

import pytest

from tubetalk.agent.runs import AgentEventType, AgentRun, NewAgentRunEvent
from tubetalk.infrastructure.repositories.sqlite_agent_runs import (
    SQLiteAgentRunRepository,
)
from tubetalk.services.agent_run_service import (
    AgentRunService,
    AgentRunTransitionError,
)


class StubSession:
    """A deterministic durable-session substitute for service lifecycle tests."""

    def __init__(self, repository: SQLiteAgentRunRepository, run: AgentRun) -> None:
        self._repository = repository
        self._run = run
        self.resumed = False

    @property
    def run_id(self) -> str:
        return self._run.run_id

    def ask(self, request: str) -> str:
        self._repository.append_event(
            NewAgentRunEvent(
                run_id=self.run_id,
                event_type=AgentEventType.USER_REQUEST,
                payload={"content": request},
            )
        )
        self._repository.append_event(
            NewAgentRunEvent(
                run_id=self.run_id,
                event_type=AgentEventType.FINAL_RESPONSE,
                payload={"text": "launched"},
            )
        )
        return "launched"

    def resume(self) -> str:
        self.resumed = True
        self._repository.append_event(
            NewAgentRunEvent(
                run_id=self.run_id,
                event_type=AgentEventType.FINAL_RESPONSE,
                payload={"text": "resumed"},
            )
        )
        return "resumed"


class StubSessions:
    def __init__(self, repository: SQLiteAgentRunRepository) -> None:
        self._repository = repository
        self.created: list[StubSession] = []

    def __call__(self, run: AgentRun | None = None) -> StubSession:
        durable_run = run or AgentRun(run_id=f"run-{len(self.created) + 1}")
        if run is None:
            self._repository.create_run(durable_run)
        session = StubSession(self._repository, durable_run)
        self.created.append(session)
        return session


def _service(
    tmp_path: Path,
) -> tuple[AgentRunService, SQLiteAgentRunRepository, StubSessions]:
    repository = SQLiteAgentRunRepository(tmp_path / "runs.sqlite3")
    sessions = StubSessions(repository)
    return AgentRunService(repository, sessions), repository, sessions


def _pending_run(repository: SQLiteAgentRunRepository, run_id: str = "pending") -> None:
    repository.create_run(AgentRun(run_id=run_id))
    repository.append_event(
        NewAgentRunEvent(run_id=run_id, event_type=AgentEventType.USER_REQUEST)
    )
    repository.append_event(
        NewAgentRunEvent(run_id=run_id, event_type=AgentEventType.APPROVAL_REQUESTED)
    )


def test_launch_returns_completed_durable_run(tmp_path: Path) -> None:
    """Launching delegates work to a new durable session."""
    service, _, _ = _service(tmp_path)

    result = service.launch("영상 목록 보여줘")

    assert result.response == "launched"
    assert result.state.status == "completed"


def test_approve_and_reject_enforce_pending_approval_state(tmp_path: Path) -> None:
    """Only pending runs can be approved or rejected."""
    service, repository, _ = _service(tmp_path)
    _pending_run(repository)

    approved = service.approve("pending")
    assert approved.status == "running"
    with pytest.raises(AgentRunTransitionError):
        service.approve("pending")

    _pending_run(repository, "rejected")
    assert service.reject("rejected").status == "cancelled"


def test_pause_and_resume_continue_the_existing_run_once(tmp_path: Path) -> None:
    """Resume binds the prior run and does not create a replacement execution."""
    service, repository, sessions = _service(tmp_path)
    repository.create_run(AgentRun(run_id="run"))
    repository.append_event(
        NewAgentRunEvent(run_id="run", event_type=AgentEventType.USER_REQUEST)
    )

    assert service.pause("run").status == "paused"
    result = service.resume("run")

    assert result.response == "resumed"
    assert result.state.status == "completed"
    assert sessions.created[-1].run_id == "run"
    assert [run.run_id for run in repository.list_runs()] == ["run"]


def test_resume_rejects_terminal_and_pending_runs(tmp_path: Path) -> None:
    """Only a paused or already-running execution may resume."""
    service, repository, _ = _service(tmp_path)
    _pending_run(repository)

    with pytest.raises(AgentRunTransitionError, match="pending_approval"):
        service.resume("pending")
