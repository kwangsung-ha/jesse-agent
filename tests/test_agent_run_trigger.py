"""Static contract test for the shared Agent-run trigger boundary."""

from jesseagent.application.agent_runs.contracts import AgentRunTrigger
from jesseagent.application.agent_runs.service import AgentRunService


def _accept_trigger(_: AgentRunTrigger) -> None:
    """Type-check the interface consumed by future trigger adapters."""


def test_agent_run_service_implements_trigger_port() -> None:
    """The CLI service satisfies the future-adapter lifecycle contract."""
    _accept_trigger(AgentRunService)
