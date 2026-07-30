"""Internal shared mechanics for video-scoped Chroma vector indexes."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb

from jesseagent.ports.embedding import EmbeddingProvider


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

    def _create_collection(self, name: str | None = None) -> Any:
        return self._client.get_or_create_collection(
            name=name or self.collection_name,
            metadata={"video_id": self.video_id, "hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _create_generation_collection(self) -> tuple[str, Any]:
        """Create an inactive collection that can safely replace the active one."""
        name = f"{self.collection_name}__{uuid4().hex}"
        return name, self._create_collection(name)

    def _active_collection_name(self) -> str:
        """Read the manifest-selected collection, supporting legacy manifests."""
        if not self.manifest_path.is_file():
            return self.collection_name
        try:
            data = self._load_manifest_data()
        except (OSError, ValueError, json.JSONDecodeError):
            return self.collection_name
        name = data.get("collection_name")
        return name if isinstance(name, str) and name else self.collection_name

    def _collection_for_manifest(self, data: dict[str, Any]) -> Any:
        """Select the manifest's active collection for status validation."""
        name = data.get("collection_name")
        if not isinstance(name, str) or not name:
            name = self.collection_name
        self._collection = self._create_collection(name)
        return self._collection

    def _retire_collection(self, name: str) -> None:
        """Best-effort cleanup after a new generation is already active."""
        if name == self.collection_name or name.startswith(f"{self.collection_name}__"):
            try:
                self._client.delete_collection(name=name)
            except Exception:
                pass

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
        serialized = json.dumps(data, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.manifest_path.parent,
            prefix=f".{self.manifest_path.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w") as temporary_file:
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, self.manifest_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
