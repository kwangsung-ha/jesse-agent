"""Provider-neutral model boundary for the Agent decision loop."""

from typing import Protocol

from jesseagent.agent.contracts import AgentDecision, AgentMessage


class AgentModelError(Exception):
    """Raised when the model cannot produce a usable Agent decision."""


class AgentModel(Protocol):
    """Choose the next native tool call or return a final natural-language reply."""

    def decide(
        self,
        messages: tuple[AgentMessage, ...],
        declarations: tuple[dict[str, object], ...],
        current_video_id: str | None,
    ) -> AgentDecision:
        """Return one model decision for the supplied compact conversation."""
