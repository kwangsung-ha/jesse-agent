"""SQLite catalog and FTS5 index for common knowledge documents."""
# ruff: noqa: E501

import json
import sqlite3
from pathlib import Path

from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument


class SQLiteKnowledgeCatalog:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
              document_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, uri TEXT NOT NULL,
              title TEXT NOT NULL, content_hash TEXT NOT NULL, updated_at TEXT, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
              ordinal INTEGER NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(chunk_id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS wiki_links (
              document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE, target TEXT NOT NULL
            );
        """)

    def content_hashes(self, source_id: str) -> dict[str, str]:
        return dict(
            self._connection.execute(
                "SELECT document_id, content_hash FROM documents WHERE source_id = ?",
                (source_id,),
            )
        )

    def replace_document(
        self, document: KnowledgeDocument, chunks: tuple[KnowledgeChunk, ...]
    ) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE document_id = ?)",
                (document.document_id,),
            )
            self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document.document_id,)
            )
            self._connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.source_id,
                    document.uri,
                    document.title,
                    document.content_hash,
                    document.updated_at.isoformat() if document.updated_at else None,
                    json.dumps(document.metadata),
                ),
            )
            for chunk in chunks:
                self._connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        json.dumps(chunk.metadata),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO chunk_fts VALUES (?, ?)", (chunk.chunk_id, chunk.text)
                )
            links = document.metadata.get("wiki_links", [])
            for target in links if isinstance(links, list) else []:
                if isinstance(target, str):
                    self._connection.execute(
                        "INSERT INTO wiki_links VALUES (?, ?)",
                        (document.document_id, target),
                    )

    def delete_documents(self, source_id: str, document_ids: set[str]) -> None:
        with self._connection:
            for document_id in document_ids:
                self._connection.execute(
                    "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE document_id = ?)",
                    (document_id,),
                )
                self._connection.execute(
                    "DELETE FROM documents WHERE source_id = ? AND document_id = ?",
                    (source_id, document_id),
                )
