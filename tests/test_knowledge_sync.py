"""Tests for incremental common-knowledge synchronization."""

from datetime import datetime, timezone
from pathlib import Path

from jesseagent.application.knowledge.sync import KnowledgeSyncService
from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument
from jesseagent.infrastructure.sqlite.knowledge import (
    SQLiteKnowledgeCatalog,
)


class Source:
    source_id = "obsidian"

    def __init__(self, documents: tuple[KnowledgeDocument, ...]) -> None:
        self.documents = documents

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        return self.documents


class Provider:
    model, dimension = "test", 2

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in documents]

    def embed_query(self, query: str) -> list[float]:
        return [0.0, 1.0]


class Index:
    def __init__(self) -> None:
        self.replaced: list[str] = []
        self.deleted: set[str] = set()

    def replace_document(
        self, chunks: tuple[KnowledgeChunk, ...], provider: Provider
    ) -> None:
        self.replaced.append(chunks[0].document_id)

    def delete_documents(self, document_ids: set[str]) -> None:
        self.deleted.update(document_ids)


def _document(identifier: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        source_id="obsidian",
        document_id=identifier,
        uri="obsidian://open",
        title="Note",
        content=content,
        content_hash=KnowledgeDocument.content_digest(content),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"wiki_links": ["Other"]},
    )


def _chunks(document: KnowledgeDocument) -> tuple[KnowledgeChunk, ...]:
    return (
        KnowledgeChunk(
            chunk_id=f"{document.document_id}:0",
            document_id=document.document_id,
            ordinal=0,
            text=document.content,
        ),
    )


def test_sync_skips_unchanged_replaces_changed_and_removes_deleted(
    tmp_path: Path,
) -> None:
    catalog, index = SQLiteKnowledgeCatalog(tmp_path / "knowledge.sqlite3"), Index()
    service = KnowledgeSyncService(catalog, index, lambda: Provider(), _chunks)
    one = _document("obsidian:one.md", "one")
    assert service.sync(Source((one,))).added == 1
    assert service.sync(Source((one,))).unchanged == 1
    changed = _document("obsidian:one.md", "changed")
    result = service.sync(Source((changed,)))
    assert (result.updated, index.replaced) == (
        1,
        ["obsidian:one.md", "obsidian:one.md"],
    )
    assert service.sync(Source(())).deleted == 1
    assert index.deleted == {"obsidian:one.md"}
