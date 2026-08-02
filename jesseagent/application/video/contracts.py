"""Contracts required by the video processing application workflow."""

from abc import abstractmethod
from datetime import datetime
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict

from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.domain.retrieval import ChatAnswer, ChatTurn, RetrievalHit
from jesseagent.domain.state import CacheState
from jesseagent.domain.summary import VideoSummary
from jesseagent.domain.transcript import Transcript
from jesseagent.domain.vision import VisionScene, VisionSource


class ChatProviderError(Exception):
    """Raised when an answer backend cannot return a valid answer."""


class ChatProvider(Protocol):
    """Generate a grounded answer from evidence and conversation context."""

    def answer(
        self,
        question: str,
        evidence: tuple[RetrievalHit, ...],
        history: tuple[ChatTurn, ...],
    ) -> ChatAnswer:
        """Return an answer with citations into the supplied evidence."""


class SummaryProviderError(Exception):
    """Raised when a summary provider cannot produce a valid summary."""


class SummaryProvider(Protocol):
    """Generate a structured summary from timestamped transcript segments."""

    def generate_summary(
        self,
        transcript: Transcript,
        *,
        title: str,
        language: str,
    ) -> VideoSummary:
        """Return a concise summary and chronological chapter titles."""


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
    def needs_indexing(self, transcript: Transcript) -> bool: ...

    @abstractmethod
    def get_index_status(
        self, transcript: Optional[Transcript]
    ) -> TranscriptIndexStatus: ...

    @abstractmethod
    def index_transcript(
        self,
        transcript: Transcript,
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int: ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], limit: int
    ) -> list[RetrievalHit]: ...


class VisionProviderError(Exception):
    """Raised when a vision provider cannot produce a valid scene index."""


class VisionAnalyzer(Protocol):
    """Describe a video source without exposing provider-specific API details."""

    def describe(
        self, source: VisionSource, *, title: str, duration_sec: float
    ) -> tuple[VisionScene, ...]:
        """Return chronologically ordered visual scenes for the source."""


class VisionIndexRepositoryError(Exception):
    """Raised when a vision-index backend cannot complete an operation."""


class VisionVectorIndexStatus(BaseModel):
    """Repository-owned status data for a visual-scene vector index."""

    model_config = ConfigDict(frozen=True)

    state: CacheState
    scene_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    indexed_at: Optional[datetime] = None


class VisionIndexRepository(Protocol):
    """Persist and validate one video's visual-scene search index."""

    @abstractmethod
    def needs_indexing(self, scenes: tuple[VisionScene, ...]) -> bool: ...

    @abstractmethod
    def get_index_status(
        self, scenes: Optional[tuple[VisionScene, ...]]
    ) -> VisionVectorIndexStatus: ...

    @abstractmethod
    def index_scenes(
        self,
        scenes: tuple[VisionScene, ...],
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int: ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], limit: int
    ) -> list[RetrievalHit]: ...
