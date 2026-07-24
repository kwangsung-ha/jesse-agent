"""Production dependency wiring selected from application settings."""

from collections.abc import Callable

from tubetalk.core.cache import CacheFreshnessPolicy, LocalCacheManager
from tubetalk.core.config import Settings, settings
from tubetalk.domain.transcript_index import TranscriptChunkPolicy
from tubetalk.infrastructure.embeddings.gemini import GeminiEmbeddingProvider
from tubetalk.infrastructure.repositories.chroma_transcript import (
    ChromaTranscriptIndexRepository,
)
from tubetalk.infrastructure.repositories.chroma_vision import (
    ChromaVisionIndexRepository,
)
from tubetalk.infrastructure.summaries.gemini import GeminiSummaryProvider
from tubetalk.infrastructure.visions.gemini import GeminiVisionAnalyzer
from tubetalk.pipeline.loader import YouTubeLoader
from tubetalk.ports.embedding import EmbeddingProvider
from tubetalk.ports.summary import SummaryProvider
from tubetalk.ports.transcript_index_repository import TranscriptIndexRepository
from tubetalk.ports.vision import VisionAnalyzer
from tubetalk.ports.vision_index_repository import VisionIndexRepository
from tubetalk.services.video_service import VideoService


def create_video_service(config: Settings = settings) -> VideoService:
    """Build the production service using the configured infrastructure."""
    return VideoService(
        cache=LocalCacheManager(
            data_dir=config.data_dir,
            freshness_policy=CacheFreshnessPolicy(
                summary_model=config.summary_model,
                summary_prompt_version=config.summary_prompt_version,
                summary_language=config.summary_language,
                vision_model=config.vision_model,
                vision_prompt_version=config.vision_prompt_version,
                embedding_model=config.embedding_model,
                embedding_dimension=config.embedding_dimension,
            ),
        ),
        loader=YouTubeLoader(),
        embedding_provider_factory=_embedding_provider_factory(config),
        transcript_index_repository_factory=_repository_factory(config),
        summary_provider_factory=_summary_provider_factory(config),
        vision_analyzer_factory=_vision_analyzer_factory(config),
        vision_index_repository_factory=_vision_repository_factory(config),
        summary_model=config.summary_model,
        summary_prompt_version=config.summary_prompt_version,
        summary_language=config.summary_language,
        vision_model=config.vision_model,
        vision_prompt_version=config.vision_prompt_version,
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
                chunk_policy=TranscriptChunkPolicy(
                    max_seconds=config.transcript_chunk_max_seconds,
                    max_characters=config.transcript_chunk_max_characters,
                ),
            )


def _summary_provider_factory(config: Settings) -> Callable[[], SummaryProvider]:
    """Build a factory so summary credentials are only resolved when needed."""
    return lambda: GeminiSummaryProvider(
        api_key=config.gemini_api_key,
        model=config.summary_model,
    )


def _vision_analyzer_factory(config: Settings) -> Callable[[], VisionAnalyzer]:
    """Create the configured public-URL vision analyzer lazily."""
    return lambda: GeminiVisionAnalyzer(
        api_key=config.gemini_api_key,
        model=config.vision_model,
    )


def _vision_repository_factory(
    config: Settings,
) -> Callable[[str], VisionIndexRepository]:
    """Create a video-scoped visual-scene vector repository."""
    return lambda video_id: ChromaVisionIndexRepository(
        video_id,
        data_dir=config.data_dir,
        embedding_model=config.embedding_model,
        embedding_dimension=config.embedding_dimension,
    )
