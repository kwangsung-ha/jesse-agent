"""Application API for durable Agent-run lifecycle operations."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from jesseagent.agent.orchestrator import AgentSession
from jesseagent.agent.reducer import reduce_run
from jesseagent.agent.runs import (
    AgentEventType,
    AgentRun,
    AgentRunState,
    AgentRunStatus,
    NewAgentRunEvent,
)
from jesseagent.ports.agent_run_repository import AgentRunRepository


class AgentRunTransitionError(ValueError):
    """Raised when an operation is not allowed in the run's current state."""


class AgentRunResult(BaseModel):
    """A lifecycle operation's run state and optional rendered answer."""

    model_config = ConfigDict(frozen=True)

    state: AgentRunState
    response: str | None = None


class AgentSessionFactory(Protocol):
    """Create a new or an existing durable Agent session."""

    def __call__(self, run: AgentRun | None = None) -> AgentSession:
        """Return a session bound to the supplied run when resuming."""


class AgentRunService:
    """Own lifecycle transitions above the Agent's deterministic event loop."""

    def __init__(
        self, repository: AgentRunRepository, sessions: AgentSessionFactory
    ) -> None:
        self._repository = repository
        self._sessions = sessions

    def launch(self, request: str) -> AgentRunResult:
        """Create and execute one new run."""
        session = self._sessions()
        response = session.ask(request)
        return AgentRunResult(
            state=self.get_status(_require_run_id(session)), response=response
        )

    def get_status(self, run_id: str) -> AgentRunState:
        """Return state reconstructed from a run's append-only events."""
        run = self._repository.get_run(run_id)
        return reduce_run(run, self._repository.list_events(run_id))

    def list_runs(self) -> tuple[AgentRunState, ...]:
        """Return all durable runs with reducer-derived current states."""
        return tuple(
            self.get_status(run.run_id) for run in self._repository.list_runs()
        )

    def approve(self, run_id: str) -> AgentRunState:
        """Resolve a pending approval positively without executing work yet."""
        self._require_status(run_id, AgentRunStatus.PENDING_APPROVAL)
        self._append_resolution(run_id, approved=True)
        return self.get_status(run_id)

    def reject(self, run_id: str) -> AgentRunState:
        """Cancel a pending approval without executing the requested work."""
        self._require_status(run_id, AgentRunStatus.PENDING_APPROVAL)
        self._append_resolution(run_id, approved=False)
        return self.get_status(run_id)

    def pause(self, run_id: str) -> AgentRunState:
        """Persist an explicit pause so a later resume has a stable boundary."""
        self._require_status(run_id, AgentRunStatus.RUNNING)
        self._repository.append_event(
            NewAgentRunEvent(run_id=run_id, event_type=AgentEventType.PAUSED)
        )
        return self.get_status(run_id)

    def resume(self, run_id: str) -> AgentRunResult:
        """Continue only a paused or approved run from its durable context."""
        state = self.get_status(run_id)
        if state.status not in {AgentRunStatus.PAUSED, AgentRunStatus.RUNNING}:
            raise AgentRunTransitionError(
                f"Cannot resume a run in '{state.status}' state"
            )
        run = self._repository.get_run(run_id)
        response = self._sessions(run).resume()
        return AgentRunResult(state=self.get_status(run_id), response=response)

    def delete_run(self, run_id: str) -> None:
        """Explicitly delete one run and all of its events."""
        self._repository.delete_run(run_id)

    def _require_status(self, run_id: str, expected: AgentRunStatus) -> None:
        status = self.get_status(run_id).status
        if status != expected:
            raise AgentRunTransitionError(
                f"Expected '{expected}' state, got '{status}'"
            )

    def _append_resolution(self, run_id: str, *, approved: bool) -> None:
        self._repository.append_event(
            NewAgentRunEvent(
                run_id=run_id,
                event_type=AgentEventType.APPROVAL_RESOLVED,
                payload={"approved": approved},
            )
        )


def _require_run_id(session: AgentSession) -> str:
    run_id = session.run_id
    if run_id is None:
        raise RuntimeError("AgentRunService requires a durable AgentSession")
    return run_id
