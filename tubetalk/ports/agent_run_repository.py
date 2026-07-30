"""Persistence port for append-only Agent run events."""

from abc import abstractmethod
from typing import Protocol

from tubetalk.agent.runs import AgentRun, AgentRunEvent, NewAgentRunEvent


class AgentRunRepositoryError(Exception):
    """Raised when durable Agent run persistence cannot complete an operation."""


class AgentRunNotFoundError(AgentRunRepositoryError):
    """Raised when the requested Agent run does not exist."""


class AgentRunRepository(Protocol):
    """Store and retrieve ordered events for durable Agent runs."""

    @abstractmethod
    def create_run(self, run: AgentRun) -> None:
        """Create an empty run exactly once."""

    @abstractmethod
    def append_event(self, event: NewAgentRunEvent) -> AgentRunEvent:
        """Atomically append an event and return its assigned sequence."""

    @abstractmethod
    def get_run(self, run_id: str) -> AgentRun:
        """Load one run or raise ``AgentRunNotFoundError``."""

    @abstractmethod
    def list_runs(self) -> tuple[AgentRun, ...]:
        """Return all runs, newest first."""

    @abstractmethod
    def list_events(self, run_id: str) -> tuple[AgentRunEvent, ...]:
        """Return one run's events in append order."""

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Explicitly remove one run and all of its events."""
