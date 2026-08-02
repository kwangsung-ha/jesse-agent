"""Shared embedding contract used by video and knowledge workflows."""

from typing import Protocol


class EmbeddingProviderError(Exception):
    """Raised when an embedding backend cannot serve a request."""


class EmbeddingProvider(Protocol):
    """Create vectors in a named, fixed-dimensional embedding space."""

    model: str
    dimension: int

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Return one vector for each document."""

    def embed_query(self, query: str) -> list[float]:
        """Return one vector in the same space as indexed documents."""
