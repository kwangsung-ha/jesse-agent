"""Typed contracts shared by the Agent loop and its deterministic tools."""

from typing import Any, Literal
from uuid import uuid4

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
    call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
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
    call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    error_code: str | None = None
    user_summary: str = ""
    next_action: str | None = None
