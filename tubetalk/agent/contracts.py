"""Typed contracts shared by the Agent loop and its deterministic tools."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentMessage(BaseModel):
    """One compact event supplied to the Agent model."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "tool"]
    content: str


class ToolCall(BaseModel):
    """A model-proposed call that must be validated before execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """The next model action: one or more tools, or a final response."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ToolResult(BaseModel):
    """JSON-safe result returned to the Agent after deterministic execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool
    content: dict[str, Any]
