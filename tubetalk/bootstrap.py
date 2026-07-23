"""Production dependency wiring selected from application settings."""

from collections.abc import Callable

from tubetalk.core.cache import LocalCacheManager
from tubetalk.core.config import Settings, settings
from tubetalk.infrastructure.embeddings.gemini import GeminiEmbeddingProvider
from tubetalk.infrastructure.repositories.chroma_transcript import (
    ChromaTranscriptIndexRepository,
)
from tubetalk.infrastructure.summaries.gemini import GeminiSummaryProvider
from tubetalk.pipeline.loader import YouTubeLoader
from tubetalk.ports.embedding import EmbeddingProvider
from tubetalk.ports.summary import SummaryProvider
from tubetalk.ports.transcript_index_repository import TranscriptIndexRepository
from tubetalk.services.video_service import VideoService


def create_video_service(config: Settings = settings) -> VideoService:
    """Build the production service using the configured infrastructure."""
    return VideoService(
        cache=LocalCacheManager(data_dir=config.data_dir),
        loader=YouTubeLoader(),
        embedding_provider_factory=_embedding_provider_factory(config),
        transcript_index_repository_factory=_repository_factory(config),
        summary_provider_factory=_summary_provider_factory(config),
    )


def _embedding_provider_factory(
    config: Settings,
) -> Callable[[], EmbeddingProvider]:
    match config.embedding_provider:
        case "gemini":
            return lambda: GeminiEmbeddingProvider(
                api_key=config.gemini_api_key,
                model=config.embedding_model,
                dimension=config.embedding_dimension,
            )


def _repository_factory(
    config: Settings,
) -> Callable[[str], TranscriptIndexRepository]:
    match config.vector_repository:
        case "chroma":
            return lambda video_id: ChromaTranscriptIndexRepository(
                video_id,
                data_dir=config.data_dir,
                embedding_model=config.embedding_model,
                embedding_dimension=config.embedding_dimension,
            )


def _summary_provider_factory(config: Settings) -> Callable[[], SummaryProvider]:
    """Build a factory so summary credentials are only resolved when needed."""
    return lambda: GeminiSummaryProvider(
        api_key=config.gemini_api_key,
        model=config.summary_model,
    )
