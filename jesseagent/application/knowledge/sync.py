"""Incrementally synchronize source-neutral knowledge into local search stores."""

from dataclasses import dataclass
from typing import Callable

from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.application.knowledge.contracts import (
    KnowledgeCatalog,
    KnowledgeVectorIndex,
)
from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument
from jesseagent.sources.contracts import SourceConnector


@dataclass(frozen=True)
class KnowledgeSyncResult:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0


class KnowledgeSyncService:
    """Coordinate source reads, catalog updates, and vector replacement."""

    def __init__(
        self,
        catalog: KnowledgeCatalog,
        index: KnowledgeVectorIndex,
        provider_factory: Callable[[], EmbeddingProvider],
        chunker: Callable[[KnowledgeDocument], tuple[KnowledgeChunk, ...]],
    ) -> None:
        self._catalog, self._index = catalog, index
        self._provider_factory, self._chunker = provider_factory, chunker

    def sync(self, source: SourceConnector) -> KnowledgeSyncResult:
        existing = self._catalog.content_hashes(source.source_id)
        documents = source.list_documents()
        seen = {document.document_id for document in documents}
        added = updated = unchanged = 0
        provider: EmbeddingProvider | None = None
        for document in documents:
            if existing.get(document.document_id) == document.content_hash:
                unchanged += 1
                continue
            chunks = self._chunker(document)
            if provider is None:
                provider = self._provider_factory()
            self._index.replace_document(chunks, provider)
            self._catalog.replace_document(document, chunks)
            if document.document_id in existing:
                updated += 1
            else:
                added += 1
        deleted_ids = set(existing) - seen
        if deleted_ids:
            self._index.delete_documents(deleted_ids)
            self._catalog.delete_documents(source.source_id, deleted_ids)
        return KnowledgeSyncResult(added, updated, unchanged, len(deleted_ids))
