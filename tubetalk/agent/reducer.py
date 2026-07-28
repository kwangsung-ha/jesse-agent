"""Deterministically reconstruct Agent state and model context from events."""

import json

from tubetalk.agent.contracts import AgentMessage
from tubetalk.agent.runs import (
    AgentEventType,
    AgentRun,
    AgentRunEvent,
    AgentRunState,
    AgentRunStatus,
)


class AgentRunReductionError(ValueError):
    """Raised when persisted events cannot form one valid Agent execution."""


def reduce_run(run: AgentRun, events: tuple[AgentRunEvent, ...]) -> AgentRunState:
    """Return the lifecycle snapshot derived only from ordered events."""
    status = AgentRunStatus.RUNNING
    current_video_id: str | None = None
    previous_sequence = 0
    for event in events:
        if event.run_id != run.run_id or event.sequence != previous_sequence + 1:
            raise AgentRunReductionError("Agent run events must be contiguous")
        previous_sequence = event.sequence
        if event.event_type == AgentEventType.USER_REQUEST:
            status = AgentRunStatus.RUNNING
        elif event.event_type == AgentEventType.APPROVAL_REQUESTED:
            status = AgentRunStatus.PENDING_APPROVAL
        elif event.event_type == AgentEventType.APPROVAL_RESOLVED:
            status = AgentRunStatus.RUNNING
        elif event.event_type == AgentEventType.FINAL_RESPONSE:
            status = AgentRunStatus.COMPLETED
        elif event.event_type == AgentEventType.FAILURE:
            status = AgentRunStatus.FAILED
        current_video_id = _video_id_from_event(event) or current_video_id
    return AgentRunState(
        run_id=run.run_id,
        status=status,
        last_sequence=previous_sequence,
        current_video_id=current_video_id,
    )


def model_messages(events: tuple[AgentRunEvent, ...]) -> tuple[AgentMessage, ...]:
    """Build the model's compact conversation from durable user/tool events."""
    messages: list[AgentMessage] = []
    for event in events:
        if event.event_type == AgentEventType.USER_REQUEST:
            content = event.payload.get("content")
            if not isinstance(content, str):
                raise AgentRunReductionError("User request events require text content")
            messages.append(AgentMessage(role="user", content=content))
        elif event.event_type == AgentEventType.TOOL_RESULT:
            messages.append(
                AgentMessage(
                    role="tool",
                    content=json.dumps(event.payload, ensure_ascii=False),
                )
            )
    return tuple(messages)


def _video_id_from_event(event: AgentRunEvent) -> str | None:
    if event.event_type != AgentEventType.TOOL_RESULT:
        return None
    content = event.payload.get("content")
    if not isinstance(content, dict):
        return None
    video_id = content.get("video_id")
    return video_id if isinstance(video_id, str) else None
