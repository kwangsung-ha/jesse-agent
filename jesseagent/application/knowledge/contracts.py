"""Contracts required by knowledge synchronization and search workflows."""

from typing import Protocol

from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument


class KnowledgeCatalog(Protocol):
    def content_hashes(self, source_id: str) -> dict[str, str]: ...

    def replace_document(
        self, document: KnowledgeDocument, chunks: tuple[KnowledgeChunk, ...]
    ) -> None: ...

    def delete_documents(self, source_id: str, document_ids: set[str]) -> None: ...


class KnowledgeVectorIndex(Protocol):
    def replace_document(
        self, chunks: tuple[KnowledgeChunk, ...], provider: EmbeddingProvider
    ) -> None: ...

    def delete_documents(self, document_ids: set[str]) -> None: ...


class KnowledgeCatalogSearch(Protocol):
    def search(self, query: str, limit: int) -> list[dict[str, object]]: ...


class KnowledgeVectorSearch(Protocol):
    def search(self, embedding: list[float], limit: int) -> list[dict[str, object]]: ...
