"""Internal shared mechanics for video-scoped Chroma vector indexes."""

import json
from pathlib import Path
from typing import Any

import chromadb

from tubetalk.ports.embedding import EmbeddingProvider


class ChromaVectorRepositoryBase:
    """Own common Chroma lifecycle and vector-space validation mechanics."""

    collection_name: str

    def _initialize(
        self,
        *,
        video_id: str,
        data_dir: Path,
        manifest_filename: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self.video_id = video_id
        self.path = data_dir / video_id / "chromadb"
        self.manifest_path = data_dir / video_id / manifest_filename
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._create_collection()

    def count(self) -> int:
        """Return the number of documents in the active collection."""
        return int(self._collection.count())

    def _create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"video_id": self.video_id, "hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _recreate_collection(self) -> Any:
        self._client.delete_collection(name=self.collection_name)
        return self._create_collection()

    def _validate_provider(self, embedding_provider: EmbeddingProvider) -> None:
        if (
            embedding_provider.model != self.embedding_model
            or embedding_provider.dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding provider settings do not match the vector repository"
            )

    def _validate_embeddings(
        self, embeddings: list[list[float]], expected_count: int
    ) -> None:
        if len(embeddings) != expected_count:
            raise ValueError("Embedding provider returned an unexpected vector count")
        if any(len(vector) != self.embedding_dimension for vector in embeddings):
            raise ValueError(
                "Embedding provider returned an unexpected vector dimension"
            )

    def _load_manifest_data(self) -> dict[str, Any]:
        data = json.loads(self.manifest_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Index manifest must be a JSON object")
        return data

    def _save_manifest_data(self, data: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(data, indent=2) + "\n")
