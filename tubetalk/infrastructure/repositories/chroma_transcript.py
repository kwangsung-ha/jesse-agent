"""ChromaDB implementation of the transcript-index repository port."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.errors import ChromaError

from tubetalk.core.config import settings
from tubetalk.domain.transcript import Transcript
from tubetalk.domain.transcript_index import (
    CHUNK_POLICY_VERSION,
    INDEX_SCHEMA_VERSION,
    IndexManifest,
    chunk_transcript,
    format_document,
    transcript_sha256,
)
from tubetalk.ports.embedding import EmbeddingProvider
from tubetalk.ports.transcript_index_repository import (
    TranscriptIndexRepositoryError,
    TranscriptIndexStatus,
)


class ChromaTranscriptIndexRepository:
    """Persist a video's explicit transcript embeddings in local ChromaDB."""

    collection_name = "transcript_collection"

    def __init__(
        self,
        video_id: str,
        data_dir: Optional[Path] = None,
        embedding_model: str = settings.embedding_model,
        embedding_dimension: int = settings.embedding_dimension,
    ) -> None:
        """Open the Chroma database beneath the video's cache directory."""
        root_dir = data_dir or settings.data_dir
        self.video_id = video_id
        self.path = root_dir / video_id / "chromadb"
        self.manifest_path = root_dir / video_id / "index_manifest.json"
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._create_collection()
        except (ChromaError, OSError) as error:
            raise TranscriptIndexRepositoryError(str(error)) from error

    def needs_indexing(self, transcript: Transcript) -> bool:
        """Return whether the local index differs from its source transcript."""
        return self.get_index_status(transcript).state != "current"

    def get_index_status(
        self, transcript: Optional[Transcript]
    ) -> TranscriptIndexStatus:
        """Read manifest and collection state without exposing Chroma details."""
        if not self.manifest_path.is_file():
            return TranscriptIndexStatus(state="missing")
        try:
            manifest = IndexManifest(**json.loads(self.manifest_path.read_text()))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return TranscriptIndexStatus(state="invalid")
        status = TranscriptIndexStatus(
            state="stale",
            chunk_count=manifest.chunk_count,
            embedding_model=manifest.embedding_model,
            embedding_dimension=manifest.embedding_dimension,
            indexed_at=manifest.indexed_at,
        )
        if transcript is None:
            return status
        try:
            expected_chunks = len(chunk_transcript(transcript))
            is_current = (
                manifest.schema_version == INDEX_SCHEMA_VERSION
                and manifest.transcript_sha256 == transcript_sha256(transcript)
                and manifest.embedding_model == self.embedding_model
                and manifest.embedding_dimension == self.embedding_dimension
                and manifest.chunk_policy_version == CHUNK_POLICY_VERSION
                and manifest.chunk_count == expected_chunks
                and manifest.chunk_count == self.count()
            )
        except (ChromaError, OSError, TypeError, ValueError):
            return TranscriptIndexStatus(
                state="invalid",
                chunk_count=manifest.chunk_count,
                embedding_model=manifest.embedding_model,
                embedding_dimension=manifest.embedding_dimension,
                indexed_at=manifest.indexed_at,
            )
        return (
            TranscriptIndexStatus(
                state="current",
                chunk_count=manifest.chunk_count,
                embedding_model=manifest.embedding_model,
                embedding_dimension=manifest.embedding_dimension,
                indexed_at=manifest.indexed_at,
            )
            if is_current
            else status
        )

    def index_transcript(
        self,
        transcript: Transcript,
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int:
        """Rebuild the collection and persist an index manifest for segments."""
        if (
            embedding_provider.model != self.embedding_model
            or embedding_provider.dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding provider settings do not match the vector repository"
            )
        chunks = chunk_transcript(transcript)
        documents = [format_document(chunk.text, title) for chunk in chunks]
        embeddings = embedding_provider.embed_documents(documents)
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding provider returned an unexpected vector count")
        if any(len(vector) != self.embedding_dimension for vector in embeddings):
            raise ValueError(
                "Embedding provider returned an unexpected vector dimension"
            )
        try:
            self._collection = self._recreate_collection()
            if chunks:
                self._collection.upsert(
                    ids=[f"{self.video_id}:chunk:{chunk.index}" for chunk in chunks],
                    documents=[chunk.text for chunk in chunks],
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "video_id": self.video_id,
                            "chunk_index": chunk.index,
                            "start_sec": chunk.start_sec,
                            "end_sec": chunk.end_sec,
                            "first_segment_index": chunk.first_segment_index,
                            "last_segment_index": chunk.last_segment_index,
                            "embedding_model": self.embedding_model,
                            "embedding_dimension": self.embedding_dimension,
                        }
                        for chunk in chunks
                    ],
                )
            self._save_manifest(transcript, len(chunks))
        except (ChromaError, OSError) as error:
            raise TranscriptIndexRepositoryError(str(error)) from error
        return len(chunks)

    def count(self) -> int:
        """Return the number of transcript chunks in this video's collection."""
        return self._collection.count()

    def _create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"video_id": self.video_id, "hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _recreate_collection(self) -> Any:
        self._client.delete_collection(name=self.collection_name)
        return self._create_collection()

    def _save_manifest(self, transcript: Transcript, chunk_count: int) -> None:
        manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            transcript_sha256=transcript_sha256(transcript),
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            chunk_policy_version=CHUNK_POLICY_VERSION,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
