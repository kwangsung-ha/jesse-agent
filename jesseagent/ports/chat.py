"""Port for grounded answer generation."""

from typing import Protocol

from jesseagent.domain.retrieval import ChatAnswer, ChatTurn, RetrievalHit


class ChatProviderError(Exception):
    """Raised when an answer backend cannot return a valid answer."""


class ChatProvider(Protocol):
    """Generate a grounded answer from fresh evidence and conversation context."""

    def answer(
        self,
        question: str,
        evidence: tuple[RetrievalHit, ...],
        history: tuple[ChatTurn, ...],
    ) -> ChatAnswer:
        """Return an answer with citations into the supplied evidence."""
