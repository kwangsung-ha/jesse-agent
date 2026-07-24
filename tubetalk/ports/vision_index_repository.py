"""Repository port for a video's visual-scene vector index."""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from tubetalk.domain.state import CacheState
from tubetalk.domain.vision import VisionScene
from tubetalk.ports.embedding import EmbeddingProvider


class VisionIndexRepositoryError(Exception):
    """Raised when a vision-index backend cannot complete an operation."""


@dataclass(frozen=True)
class VisionVectorIndexStatus:
    """Repository-owned status data for a visual-scene vector index."""

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
