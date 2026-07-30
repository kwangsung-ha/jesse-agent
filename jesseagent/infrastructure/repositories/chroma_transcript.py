"""ChromaDB implementation of the transcript-index repository port."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chromadb.errors import ChromaError

from jesseagent.core.logging import logger
from jesseagent.domain.retrieval import RetrievalHit
from jesseagent.domain.state import CacheState
from jesseagent.domain.transcript import Transcript
from jesseagent.domain.transcript_index import (
    CHUNK_POLICY_VERSION,
    DEFAULT_TRANSCRIPT_CHUNK_POLICY,
    INDEX_SCHEMA_VERSION,
    IndexManifest,
    TranscriptChunkPolicy,
    chunk_transcript,
    format_document,
    transcript_sha256,
)
from jesseagent.infrastructure.repositories.chroma_base import (
    ChromaVectorRepositoryBase,
)
from jesseagent.ports.embedding import EmbeddingProvider
from jesseagent.ports.transcript_index_repository import (
    TranscriptIndexRepositoryError,
    TranscriptIndexStatus,
)


class ChromaTranscriptIndexRepository(ChromaVectorRepositoryBase):
    """Persist a video's explicit transcript embeddings in local ChromaDB."""

    collection_name = "transcript_collection"

    def __init__(
        self,
        video_id: str,
        data_dir: Optional[Path] = None,
        embedding_model: str = "gemini-embedding-2",
        embedding_dimension: int = 768,
        chunk_policy: TranscriptChunkPolicy = DEFAULT_TRANSCRIPT_CHUNK_POLICY,
    ) -> None:
        """Open the Chroma database beneath the video's cache directory."""
        root_dir = data_dir or Path("./data")
        self.chunk_policy = chunk_policy
        try:
            self._initialize(
                video_id=video_id,
                data_dir=root_dir,
                manifest_filename="index_manifest.json",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        except (ChromaError, OSError) as error:
            raise TranscriptIndexRepositoryError(str(error)) from error

    def needs_indexing(self, transcript: Transcript) -> bool:
        """Return whether the local index differs from its source transcript."""
        status = self.get_index_status(transcript)
        logger.bind(event="chroma.transcript.status", video_id=self.video_id).debug(
            "state={}", status.state
        )
        return status.state != "current"

    def get_index_status(
        self, transcript: Optional[Transcript]
    ) -> TranscriptIndexStatus:
        """Read manifest and collection state without exposing Chroma details."""
        if not self.manifest_path.is_file():
            return TranscriptIndexStatus(state=CacheState.MISSING)
        try:
            manifest_data = self._load_manifest_data()
            self._collection_for_manifest(manifest_data)
            manifest_data["indexed_at"] = datetime.fromisoformat(
                manifest_data["indexed_at"]
            )
            manifest = IndexManifest(**manifest_data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return TranscriptIndexStatus(state=CacheState.INVALID)
        status = TranscriptIndexStatus(
            state=CacheState.STALE,
            chunk_count=manifest.chunk_count,
            embedding_model=manifest.embedding_model,
            embedding_dimension=manifest.embedding_dimension,
            indexed_at=manifest.indexed_at,
        )
        if transcript is None:
            return status
        try:
            expected_chunks = len(chunk_transcript(transcript, self.chunk_policy))
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
                state=CacheState.INVALID,
                chunk_count=manifest.chunk_count,
                embedding_model=manifest.embedding_model,
                embedding_dimension=manifest.embedding_dimension,
                indexed_at=manifest.indexed_at,
            )
        return (
            TranscriptIndexStatus(
                state=CacheState.CURRENT,
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
        self._validate_provider(embedding_provider)
        chunks = chunk_transcript(transcript, self.chunk_policy)
        documents = [format_document(chunk.text, title) for chunk in chunks]
        embeddings = embedding_provider.embed_documents(documents)
        self._validate_embeddings(embeddings, len(chunks))
        try:
            previous_collection = self._active_collection_name()
            generation_name, generation = self._create_generation_collection()
            if chunks:
                generation.upsert(
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
            self._save_manifest(transcript, len(chunks), generation_name)
        except (ChromaError, OSError) as error:
            raise TranscriptIndexRepositoryError(str(error)) from error
        self._collection = generation
        self._retire_collection(previous_collection)
        logger.bind(event="chroma.transcript.index", video_id=self.video_id).debug(
            "chunks={} collection={}", len(chunks), generation_name
        )
        return len(chunks)

    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        """Search the active transcript generation with an explicit vector."""
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError("Query embedding has an unexpected dimension")
        try:
            self._collection_for_manifest(self._load_manifest_data())
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            ids = result["ids"][0]
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            hits = [
                RetrievalHit(
                    source_id=str(item_id),
                    source="transcript",
                    text=str(document),
                    start_sec=float(metadata["start_sec"]),
                    end_sec=float(metadata["end_sec"]),
                    rank=rank,
                    distance=float(distance),
                )
                for rank, (item_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances), start=1
                )
            ]
            logger.bind(event="chroma.transcript.search", video_id=self.video_id).debug(
                "limit={} hits={}", limit, len(hits)
            )
            return hits
        except (ChromaError, OSError, KeyError, TypeError, ValueError) as error:
            raise TranscriptIndexRepositoryError(str(error)) from error

    def _save_manifest(
        self, transcript: Transcript, chunk_count: int, collection_name: str
    ) -> None:
        manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            transcript_sha256=transcript_sha256(transcript),
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            chunk_policy_version=CHUNK_POLICY_VERSION,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc),
            collection_name=collection_name,
        )
        self._save_manifest_data(manifest.model_dump(mode="json"))
