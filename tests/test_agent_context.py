"""Tests for deterministic Agent model-context compaction."""

from tubetalk.agent.context import AgentContextBudget, compact_messages
from tubetalk.agent.contracts import AgentMessage


def test_compaction_keeps_newest_messages_within_both_budgets() -> None:
    """Oldest messages are omitted before model invocation, never from storage."""
    messages = (
        AgentMessage(role="user", content="first"),
        AgentMessage(role="tool", content="second"),
        AgentMessage(role="user", content="third"),
    )

    compacted = compact_messages(
        messages, AgentContextBudget(max_messages=2, max_characters=100)
    )

    assert compacted[-1] == messages[-1]
    assert "Earlier context omitted" in compacted[0].content


def test_compaction_truncates_the_oldest_retained_message_deterministically() -> None:
    """The newest context survives when one message exhausts character budget."""
    messages = (
        AgentMessage(role="user", content="a" * 300),
        AgentMessage(role="tool", content="citation timestamp 01:20"),
    )

    compacted = compact_messages(
        messages, AgentContextBudget(max_messages=4, max_characters=40)
    )

    assert compacted[-1].content == "citation timestamp 01:20"
    assert compacted[-2].content.startswith("...")
    assert sum(len(message.content) for message in compacted[-2:]) <= 40
