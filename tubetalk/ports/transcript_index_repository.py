"""Repository port for a video's transcript vector index."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from tubetalk.domain.transcript import Transcript
from tubetalk.ports.embedding import EmbeddingProvider


class TranscriptIndexRepositoryError(Exception):
    """Raised when a transcript-index backend cannot complete an operation."""


@dataclass(frozen=True)
class TranscriptIndexStatus:
    """Repository-owned status data for a transcript index."""

    state: str
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    indexed_at: Optional[str] = None


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
