"""Static contract test for the shared Agent-run trigger boundary."""

from jesseagent.ports.agent_run_trigger import AgentRunTrigger
from jesseagent.services.agent_run_service import AgentRunService


def _accept_trigger(_: AgentRunTrigger) -> None:
    """Type-check the interface consumed by future trigger adapters."""


def test_agent_run_service_implements_trigger_port() -> None:
    """The CLI service satisfies the future-adapter lifecycle contract."""
    _accept_trigger(AgentRunService)
