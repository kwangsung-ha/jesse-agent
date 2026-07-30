"""Immutable contracts for durable Agent executions."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class AgentRunStatus(StrEnum):
    """A durable Agent run's lifecycle state."""

    RUNNING = "running"
    PENDING_APPROVAL = "pending_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentEventType(StrEnum):
    """Events from which an Agent run is reconstructed."""

    USER_REQUEST = "user_request"
    MODEL_DECISION = "model_decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    PAUSED = "paused"
    FINAL_RESPONSE = "final_response"
    FAILURE = "failure"


class AgentRun(BaseModel):
    """Identity and creation time for one append-only Agent execution."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRunState(BaseModel):
    """Reducer-owned snapshot contract; persistence remains event-based."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    status: AgentRunStatus = AgentRunStatus.RUNNING
    last_sequence: int = Field(default=0, ge=0)
    current_video_id: str | None = None


class NewAgentRunEvent(BaseModel):
    """A validated event before the repository assigns its sequence."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    event_type: AgentEventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRunEvent(NewAgentRunEvent):
    """An immutable persisted Agent event with repository-assigned ordering."""

    sequence: int = Field(ge=1)
