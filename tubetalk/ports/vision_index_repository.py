"""Repository port for a video's visual-scene vector index."""

from abc import abstractmethod
from datetime import datetime
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict

from tubetalk.domain.retrieval import RetrievalHit
from tubetalk.domain.state import CacheState
from tubetalk.domain.vision import VisionScene
from tubetalk.ports.embedding import EmbeddingProvider


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
    def needs_indexing(self, scenes: tuple[VisionScene, ...]) -> bool:
        """Return whether the stored index differs from its source scenes."""

    @abstractmethod
    def get_index_status(
        self, scenes: Optional[tuple[VisionScene, ...]]
    ) -> VisionVectorIndexStatus:
        """Return current, stale, invalid, or missing index state."""

    @abstractmethod
    def index_scenes(
        self,
        scenes: tuple[VisionScene, ...],
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int:
        """Replace the index and return the stored scene count."""

    @abstractmethod
    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        """Return the nearest visual scenes for one embedded query."""
