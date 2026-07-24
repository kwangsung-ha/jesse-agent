"""Repository port for a video's transcript vector index."""

from abc import abstractmethod
from datetime import datetime
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict

from tubetalk.domain.retrieval import RetrievalHit
from tubetalk.domain.state import CacheState
from tubetalk.domain.transcript import Transcript
from tubetalk.ports.embedding import EmbeddingProvider


class TranscriptIndexRepositoryError(Exception):
    """Raised when a transcript-index backend cannot complete an operation."""


class TranscriptIndexStatus(BaseModel):
    """Repository-owned status data for a transcript index."""

    model_config = ConfigDict(frozen=True)

    state: CacheState
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    indexed_at: Optional[datetime] = None


class TranscriptIndexRepository(Protocol):
    """Persist and validate one video's transcript search index."""

    @abstractmethod
    def needs_indexing(self, transcript: Transcript) -> bool:
        """Return whether the stored index differs from its source transcript."""

    @abstractmethod
    def get_index_status(
        self, transcript: Optional[Transcript]
    ) -> TranscriptIndexStatus:
        """Return current, stale, invalid, or missing index state."""

    @abstractmethod
    def index_transcript(
        self,
        transcript: Transcript,
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int:
        """Replace the index and return the stored chunk count."""

    @abstractmethod
    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        """Return the nearest transcript chunks for one embedded query."""
