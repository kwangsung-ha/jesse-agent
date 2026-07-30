"""Interface used by CLI and future external Agent-run triggers."""

from abc import abstractmethod
from typing import Protocol

from tubetalk.agent.runs import AgentRunState
from tubetalk.services.agent_run_service import AgentRunResult


class AgentRunTrigger(Protocol):
    """Launch and manage durable runs without owning Agent control flow."""

    @abstractmethod
    def launch(self, request: str) -> AgentRunResult: ...

    @abstractmethod
    def get_status(self, run_id: str) -> AgentRunState: ...

    @abstractmethod
    def list_runs(self) -> tuple[AgentRunState, ...]: ...

    @abstractmethod
    def approve(self, run_id: str) -> AgentRunState: ...

    @abstractmethod
    def reject(self, run_id: str) -> AgentRunState: ...

    @abstractmethod
    def resume(self, run_id: str) -> AgentRunResult: ...

    @abstractmethod
    def delete_run(self, run_id: str) -> None: ...
