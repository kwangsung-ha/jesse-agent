"""Gemini-backed local ChromaDB storage for transcript chunks."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import chromadb
from google import genai
from google.genai import types

from tubetalk.core.config import settings

INDEX_SCHEMA_VERSION = 1
CHUNK_POLICY_VERSION = "45s-1200chars-v1"


@dataclass(frozen=True)
class TranscriptChunk:
    """A retrieval-sized, timestamped group of transcript segments."""

    index: int
    text: str
    start_sec: float
    end_sec: float
    first_segment_index: int
    last_segment_index: int


@dataclass(frozen=True)
class IndexManifest:
    """Records the inputs and settings used to build a vector index."""

    schema_version: int
    transcript_sha256: str
    embedding_model: str
    embedding_dimension: int
    chunk_policy_version: str
    chunk_count: int
    indexed_at: str


class EmbeddingProvider(Protocol):
    """Minimal embedding API required by ``TranscriptVectorStore``."""

    model: str
    dimension: int

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Return one vector for each formatted document."""


class GeminiEmbeddingProvider:
    """Generate explicit Gemini Embedding 2 vectors for retrieval."""

    def __init__(
        self,
        api_key: str,
        model: str = settings.embedding_model,
        dimension: int = settings.embedding_dimension,
        client: Optional[Any] = None,
    ) -> None:
        """Create a provider using the supplied Gemini API key."""
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to create transcript embeddings"
            )
        self.model = model
        self.dimension = dimension
        self._client = client or genai.Client(api_key=api_key)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed documents one at a time to preserve one vector per chunk."""
        return [self._embed(document) for document in documents]

    def embed_query(self, query: str) -> list[float]:
        """Embed a future Q&A query in the matching retrieval space."""
        return self._embed(f"task: question answering | query: {query}")

    def _embed(self, content: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=content,
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        embeddings = response.embeddings
        if not embeddings or len(embeddings) != 1:
            raise ValueError("Gemini embedding response must contain one embedding")
        values = list(embeddings[0].values)
        if len(values) != self.dimension:
            raise ValueError(
                f"Gemini embedding dimension {len(values)} does not match "
                f"configured dimension {self.dimension}"
            )
        return values


def chunk_transcript(
    segments: list[dict[str, Any]],
    max_seconds: float = settings.transcript_chunk_max_seconds,
    max_characters: int = settings.transcript_chunk_max_characters,
) -> list[TranscriptChunk]:
    """Merge consecutive transcript segments into bounded retrieval chunks."""
    if max_seconds <= 0 or max_characters <= 0:
        raise ValueError("Transcript chunk limits must be positive")

    chunks: list[TranscriptChunk] = []
    chunk_texts: list[str] = []
    chunk_start: Optional[float] = None
    chunk_end: Optional[float] = None
    first_index: Optional[int] = None
    previous_start: Optional[float] = None

    def emit(last_index: int) -> None:
        if chunk_start is None or chunk_end is None or first_index is None:
            return
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(chunk_texts),
                start_sec=chunk_start,
                end_sec=chunk_end,
                first_segment_index=first_index,
                last_segment_index=last_index,
            )
        )

    for segment_index, segment in enumerate(segments):
        text, start_sec, end_sec = _validate_segment(segment)
        if previous_start is not None and start_sec < previous_start:
            raise ValueError("Transcript segments must be ordered by start_sec")
        previous_start = start_sec

        candidate_characters = len(" ".join([*chunk_texts, text]))
        candidate_duration = end_sec - (
            chunk_start if chunk_start is not None else start_sec
        )
        if chunk_texts and (
            candidate_characters > max_characters or candidate_duration > max_seconds
        ):
            emit(segment_index - 1)
            chunk_texts = []
            chunk_start = None
            chunk_end = None
            first_index = None

        if not chunk_texts:
            chunk_start = start_sec
            first_index = segment_index
        chunk_texts.append(text)
        chunk_end = end_sec

    if chunk_texts:
        emit(len(segments) - 1)
    return chunks


def format_document(text: str, title: str) -> str:
    """Format a text chunk with Gemini Embedding 2's retrieval prompt."""
    return f"title: {title} | text: {text}"


class TranscriptVectorStore:
    """Persist a video's explicit Gemini transcript embeddings in ChromaDB."""

    collection_name = "transcript_collection"

    def __init__(
        self,
        video_id: str,
        data_dir: Optional[Path] = None,
        embedding_model: str = settings.embedding_model,
        embedding_dimension: int = settings.embedding_dimension,
    ) -> None:
        """Open the vector database stored beneath the video's cache directory."""
        root_dir = data_dir or settings.data_dir
        self.video_id = video_id
        self.path = root_dir / video_id / "chromadb"
        self.manifest_path = root_dir / video_id / "index_manifest.json"
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._create_collection()

    def needs_indexing(self, segments: list[dict[str, Any]]) -> bool:
        """Return whether the local index differs from its source transcript."""
        manifest = self._load_manifest()
        if manifest is None:
            return True
        return not (
            manifest.schema_version == INDEX_SCHEMA_VERSION
            and manifest.transcript_sha256 == _transcript_sha256(segments)
            and manifest.embedding_model == self.embedding_model
            and manifest.embedding_dimension == self.embedding_dimension
            and manifest.chunk_policy_version == CHUNK_POLICY_VERSION
            and manifest.chunk_count == self.count()
        )

    def index_transcript(
        self,
        segments: list[dict[str, Any]],
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int:
        """Rebuild the collection and persist an index manifest for *segments*."""
        if (
            embedding_provider.model != self.embedding_model
            or embedding_provider.dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding provider settings do not match the vector store"
            )

        chunks = chunk_transcript(segments)
        documents = [format_document(chunk.text, title) for chunk in chunks]
        embeddings = embedding_provider.embed_documents(documents)
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding provider returned an unexpected vector count")
        if any(len(vector) != self.embedding_dimension for vector in embeddings):
            raise ValueError(
                "Embedding provider returned an unexpected vector dimension"
            )

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
        self._save_manifest(segments, len(chunks))
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

    def _load_manifest(self) -> Optional[IndexManifest]:
        try:
            return IndexManifest(**json.loads(self.manifest_path.read_text()))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_manifest(self, segments: list[dict[str, Any]], chunk_count: int) -> None:
        manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            transcript_sha256=_transcript_sha256(segments),
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            chunk_policy_version=CHUNK_POLICY_VERSION,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def _validate_segment(segment: dict[str, Any]) -> tuple[str, float, float]:
    """Validate a source segment and return its text and time boundaries."""
    text = segment.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each transcript segment requires non-empty text")
    start_sec = segment.get("start_sec")
    if not isinstance(start_sec, (int, float)):
        raise ValueError("Each transcript segment requires numeric start_sec")
    duration_sec = segment.get("duration_sec", 0.0)
    if not isinstance(duration_sec, (int, float)):
        raise ValueError("duration_sec must be numeric when provided")
    return text.strip(), float(start_sec), float(start_sec + duration_sec)


def _transcript_sha256(segments: list[dict[str, Any]]) -> str:
    """Return a stable digest used to detect transcript changes."""
    serialized = json.dumps(
        segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
