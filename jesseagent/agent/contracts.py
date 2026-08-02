"""Typed contracts used by the Agent model loop."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from jesseagent.tools.contracts import ToolCall


class AgentMessage(BaseModel):
    """One compact event supplied to the Agent model."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "tool"]
    content: str


class AgentDecision(BaseModel):
    """The next model action: one or more tools, or a final response."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
