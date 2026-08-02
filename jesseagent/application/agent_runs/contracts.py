"""Contracts required by durable Agent-run workflows."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from jesseagent.agent.runs import AgentRun, AgentRunEvent, NewAgentRunEvent

if TYPE_CHECKING:
    from jesseagent.agent.orchestrator import AgentSession
    from jesseagent.agent.runs import AgentRunState
    from jesseagent.application.agent_runs.service import AgentRunResult


class AgentRunRepositoryError(Exception):
    """Raised when durable Agent run persistence cannot complete an operation."""


class AgentRunNotFoundError(AgentRunRepositoryError):
    """Raised when the requested Agent run does not exist."""


class AgentRunRepository(Protocol):
    """Store and retrieve ordered events for durable Agent runs."""

    @abstractmethod
    def create_run(self, run: AgentRun) -> None: ...

    @abstractmethod
    def append_event(self, event: NewAgentRunEvent) -> AgentRunEvent: ...

    @abstractmethod
    def get_run(self, run_id: str) -> AgentRun: ...

    @abstractmethod
    def list_runs(self) -> tuple[AgentRun, ...]: ...

    @abstractmethod
    def list_events(self, run_id: str) -> tuple[AgentRunEvent, ...]: ...

    @abstractmethod
    def delete_run(self, run_id: str) -> None: ...


class AgentSessionFactory(Protocol):
    """Create a new or an existing durable Agent session."""

    def __call__(self, run: AgentRun | None = None) -> "AgentSession":
        """Return a session bound to the supplied run when resuming."""


class AgentRunTrigger(Protocol):
    """Launch and manage durable runs without owning Agent control flow."""

    def launch(self, request: str) -> "AgentRunResult": ...

    def get_status(self, run_id: str) -> "AgentRunState": ...

    def list_runs(self) -> tuple["AgentRunState", ...]: ...

    def approve(self, run_id: str) -> "AgentRunState": ...

    def reject(self, run_id: str) -> "AgentRunState": ...

    def resume(self, run_id: str) -> "AgentRunResult": ...

    def delete_run(self, run_id: str) -> None: ...
