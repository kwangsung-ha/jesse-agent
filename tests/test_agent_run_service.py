"""Unit tests for durable Agent-run lifecycle operations."""

from pathlib import Path

import pytest

from jesseagent.agent.runs import AgentEventType, AgentRun, NewAgentRunEvent
from jesseagent.application.agent_runs.service import (
    AgentRunService,
    AgentRunTransitionError,
)
from jesseagent.infrastructure.sqlite.agent_runs import (
    SQLiteAgentRunRepository,
)
from jesseagent.sinks.contracts import SinkApplyResult, SinkPlan


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


def test_sink_plan_requires_preview_approval_and_applies_exactly_once(
    tmp_path: Path,
) -> None:
    """A durable Sink plan cannot mutate before approval or be replayed."""
    repository = SQLiteAgentRunRepository(tmp_path / "runs.sqlite3")
    sessions = StubSessions(repository)
    applied: list[SinkPlan] = []

    def apply(plan: SinkPlan) -> SinkApplyResult:
        applied.append(plan)
        return SinkApplyResult(
            plan_id=plan.plan_id, summary="장보기 목록을 반영했습니다."
        )

    service = AgentRunService(repository, sessions, apply)
    repository.create_run(AgentRun(run_id="sink-run"))
    repository.append_event(
        NewAgentRunEvent(
            run_id="sink-run",
            event_type=AgentEventType.USER_REQUEST,
            payload={"content": "장보기 목록을 저장해줘"},
        )
    )
    plan = SinkPlan(
        sink_id="fake",
        operation="create_list",
        preview="장보기 항목 3개를 생성합니다.",
        payload={"items": ["양파", "감자", "카레"]},
    )

    pending = service.request_sink_plan("sink-run", plan)
    assert pending.status == "pending_approval"
    assert pending.approval_preview == "장보기 항목 3개를 생성합니다."
    assert applied == []

    service.approve("sink-run")
    completed = service.resume("sink-run")
    assert completed.state.status == "completed"
    assert completed.response == "장보기 목록을 반영했습니다."
    assert applied == [plan]
    with pytest.raises(AgentRunTransitionError, match="completed"):
        service.resume("sink-run")


def test_rejected_sink_plan_never_applies(tmp_path: Path) -> None:
    repository = SQLiteAgentRunRepository(tmp_path / "runs.sqlite3")
    sessions = StubSessions(repository)
    applied: list[SinkPlan] = []
    service = AgentRunService(
        repository,
        sessions,
        lambda plan: (
            applied.append(plan)
            or SinkApplyResult(plan_id=plan.plan_id, summary="applied")
        ),
    )
    repository.create_run(AgentRun(run_id="rejected-sink"))
    repository.append_event(
        NewAgentRunEvent(run_id="rejected-sink", event_type=AgentEventType.USER_REQUEST)
    )
    service.request_sink_plan(
        "rejected-sink",
        SinkPlan(sink_id="fake", operation="write", preview="파일을 씁니다."),
    )

    assert service.reject("rejected-sink").status == "cancelled"
    assert applied == []
