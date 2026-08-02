"""One Chroma collection shared by all knowledge sources."""

from pathlib import Path

import chromadb

from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.domain.knowledge import KnowledgeChunk


class ChromaKnowledgeIndex:
    def __init__(self, path: Path, model: str, dimension: int) -> None:
        self._model, self._dimension = model, dimension
        path.mkdir(parents=True, exist_ok=True)
        self._collection = chromadb.PersistentClient(
            path=str(path)
        ).get_or_create_collection(
            "knowledge_chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def replace_document(
        self, chunks: tuple[KnowledgeChunk, ...], provider: EmbeddingProvider
    ) -> None:
        if provider.model != self._model or provider.dimension != self._dimension:
            raise ValueError(
                "Embedding provider settings do not match the knowledge index"
            )
        if not chunks:
            return
        document_id = chunks[0].document_id
        self._collection.delete(where={"document_id": document_id})
        embeddings = provider.embed_documents([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks) or any(
            len(item) != self._dimension for item in embeddings
        ):
            raise ValueError("Embedding provider returned unexpected knowledge vectors")
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[
                {"document_id": chunk.document_id, "ordinal": chunk.ordinal}
                for chunk in chunks
            ],
        )

    def delete_documents(self, document_ids: set[str]) -> None:
        for document_id in document_ids:
            self._collection.delete(where={"document_id": document_id})

    def search(self, embedding: list[float], limit: int) -> list[dict[str, object]]:
        if len(embedding) != self._dimension:
            raise ValueError("Query embedding has an unexpected dimension")
        result = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=limit,
        )
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            {"chunk_id": str(chunk_id), "text": str(text), "distance": float(distance)}
            for chunk_id, text, distance in zip(ids, documents, distances, strict=True)
        ]
