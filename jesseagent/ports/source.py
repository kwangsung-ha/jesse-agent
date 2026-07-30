"""Contracts for read-only systems that provide JesseAgent knowledge."""

from typing import Protocol

from jesseagent.domain.knowledge import KnowledgeDocument


class SourceConnectorError(Exception):
    """Raised when a configured knowledge source cannot be read safely."""


class SourceConnector(Protocol):
    """Expose stable, source-neutral documents for a future sync service."""

    source_id: str

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        """Return the source's currently readable documents in stable order."""
