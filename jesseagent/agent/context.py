"""Deterministic model-context budgeting without changing persisted events."""

from pydantic import BaseModel, ConfigDict, Field

from jesseagent.agent.contracts import AgentMessage


class AgentContextBudget(BaseModel):
    """Maximum message count and characters sent to the Agent model."""

    model_config = ConfigDict(frozen=True)

    max_messages: int = Field(default=24, ge=2)
    max_characters: int = Field(default=12000, ge=32)


def compact_messages(
    messages: tuple[AgentMessage, ...], budget: AgentContextBudget
) -> tuple[AgentMessage, ...]:
    """Keep the newest useful context in a deterministic bounded representation."""
    omitted_by_count = max(0, len(messages) - (budget.max_messages - 1))
    candidates = (
        messages[-(budget.max_messages - 1) :] if omitted_by_count else messages
    )
    notice = (
        f"[Earlier context omitted: {omitted_by_count} message(s).]"
        if omitted_by_count
        else ""
    )
    retained: list[AgentMessage] = []
    remaining = budget.max_characters - len(notice)
    for message in reversed(candidates):
        if remaining <= 0:
            break
        content = message.content
        if len(content) > remaining:
            content = _tail(content, remaining)
        retained.append(AgentMessage(role=message.role, content=content))
        remaining -= len(content)
    retained.reverse()
    omitted = len(messages) - len(retained)
    if omitted:
        retained.insert(
            0,
            AgentMessage(
                role="tool",
                content=f"[Earlier context omitted: {omitted} message(s).]",
            ),
        )
    return tuple(retained)


def _tail(content: str, limit: int) -> str:
    if limit <= 3:
        return content[-limit:]
    return f"...{content[-(limit - 3) :]}"
