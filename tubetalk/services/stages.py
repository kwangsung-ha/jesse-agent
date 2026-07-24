"""Independently testable processing stages used by the video-service facade."""

from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ConfigDict

from tubetalk.core.cache import LocalCacheManager
from tubetalk.domain.state import CacheState, SyncState
from tubetalk.domain.summary import (
    SUMMARY_SCHEMA_VERSION,
    SummaryCacheEntry,
    SummaryManifest,
)
from tubetalk.domain.transcript import Transcript
from tubetalk.domain.transcript_index import transcript_sha256
from tubetalk.domain.video import CachedVideo, VideoMetadata
from tubetalk.domain.vision import (
    VISION_SCHEMA_VERSION,
    VisionIndexEntry,
    VisionManifest,
    VisionScene,
    YouTubeUrlVisionSource,
)
from tubetalk.pipeline.loader import YouTubeLoader
from tubetalk.ports.embedding import EmbeddingProvider, EmbeddingProviderError
from tubetalk.ports.summary import SummaryProvider, SummaryProviderError
from tubetalk.ports.transcript_index_repository import (
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
)
from tubetalk.ports.vision import VisionAnalyzer, VisionProviderError
from tubetalk.ports.vision_index_repository import (
    VisionIndexRepository,
    VisionIndexRepositoryError,
    VisionVectorIndexStatus,
)
from tubetalk.services.results import IndexingResult, SummaryResult, VisionResult


class IngestionResult(BaseModel):
    """Typed output of loading cached resources or collecting a video."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    cache_hit: bool
    video: CachedVideo


class VideoIngestionStage:
    """Own cache lookup and the external collection boundary."""

    def __init__(self, cache: LocalCacheManager, loader: YouTubeLoader) -> None:
        self._cache = cache
        self._loader = loader

    def load_or_collect(self, url: str) -> IngestionResult:
        """Reuse a valid cache or collect and persist typed video resources."""
        video_id = self._loader.extract_video_id(url)
        if self._cache.has_cache(video_id):
            return IngestionResult(
                video_id=video_id,
                cache_hit=True,
                video=self._cache.load_video(video_id),
            )
        metadata = self._loader.fetch_metadata(video_id, url).model_copy(
            update={"processed_at": datetime.now(timezone.utc)}
        )
        video = CachedVideo(
            metadata=metadata,
            transcript=self._loader.fetch_transcript(video_id),
        )
        self._cache.save_video(video)
        return IngestionResult(video_id=video_id, cache_hit=False, video=video)


class TranscriptIndexingStage:
    """Synchronise the derived transcript vector index."""

    def __init__(
        self,
        repository_factory: Callable[[str], TranscriptIndexRepository],
        provider_factory: Callable[[], EmbeddingProvider],
    ) -> None:
        self._repository_factory = repository_factory
        self._provider_factory = provider_factory

    def sync(
        self, video_id: str, metadata: VideoMetadata, transcript: Transcript
    ) -> IndexingResult:
        try:
            repository = self._repository_factory(video_id)
            if not repository.needs_indexing(transcript):
                return IndexingResult(state=SyncState.CURRENT)
            count = repository.index_transcript(
                transcript,
                metadata.title or f"YouTube video {video_id}",
                self._provider_factory(),
            )
            return IndexingResult(state=SyncState.INDEXED, chunk_count=count)
        except (
            EmbeddingProviderError,
            OSError,
            TranscriptIndexRepositoryError,
            ValueError,
        ) as error:
            return IndexingResult(state=SyncState.WARNING, warning=str(error))


class SummaryGenerationStage:
    """Reuse or generate a transcript-grounded summary."""

    def __init__(
        self,
        cache: LocalCacheManager,
        provider_factory: Callable[[], SummaryProvider],
        model: str,
        prompt_version: str,
        language: str,
    ) -> None:
        self._cache = cache
        self._provider_factory = provider_factory
        self._model = model
        self._prompt_version = prompt_version
        self._language = language

    def sync(
        self, video_id: str, metadata: VideoMetadata, transcript: Transcript
    ) -> SummaryResult:
        try:
            status = self._cache.get_summary_status(
                video_id,
                transcript,
                model=self._model,
                prompt_version=self._prompt_version,
                language=self._language,
            )
            if status.state == CacheState.CURRENT and status.entry is not None:
                return SummaryResult(
                    state=SyncState.CURRENT, summary=status.entry.summary
                )
            summary = self._provider_factory().generate_summary(
                transcript,
                title=metadata.title or f"YouTube video {video_id}",
                language=self._language,
            )
            self._cache.save_summary(
                video_id,
                SummaryCacheEntry(
                    summary=summary,
                    manifest=SummaryManifest(
                        schema_version=SUMMARY_SCHEMA_VERSION,
                        transcript_sha256=transcript_sha256(transcript),
                        model=self._model,
                        prompt_version=self._prompt_version,
                        language=self._language,
                        generated_at=datetime.now(timezone.utc),
                    ),
                ),
            )
            return SummaryResult(state=SyncState.GENERATED, summary=summary)
        except (OSError, SummaryProviderError, ValueError) as error:
            return SummaryResult(state=SyncState.WARNING, warning=str(error))


class VisionIndexingStage:
    """Synchronise visual-scene cache and its derived vector index."""

    def __init__(
        self,
        cache: LocalCacheManager,
        analyzer_factory: Callable[[], VisionAnalyzer],
        repository_factory: Callable[[str], VisionIndexRepository],
        embedding_provider_factory: Callable[[], EmbeddingProvider],
        model: str,
        prompt_version: str,
    ) -> None:
        self._cache = cache
        self._analyzer_factory = analyzer_factory
        self._repository_factory = repository_factory
        self._embedding_provider_factory = embedding_provider_factory
        self._model = model
        self._prompt_version = prompt_version

    def sync(self, video_id: str, metadata: VideoMetadata) -> VisionResult:
        """Reuse or generate scenes, then synchronise their vector index."""
        duration = metadata.duration_sec
        if duration is None:
            return VisionResult(
                state=SyncState.WARNING,
                warning="Cached metadata does not contain a valid duration",
            )
        try:
            status = self._cache.get_vision_index_status(
                video_id,
                source_url=metadata.source_url,
                model=self._model,
                prompt_version=self._prompt_version,
            )
            if status.state == CacheState.CURRENT and status.entry is not None:
                scenes = status.entry.scenes
                state = SyncState.CURRENT
            else:
                scenes = self._analyzer_factory().describe(
                    YouTubeUrlVisionSource(metadata.source_url),
                    title=metadata.title or f"YouTube video {video_id}",
                    duration_sec=float(duration),
                )
                self._cache.save_vision_index(
                    video_id,
                    VisionIndexEntry(
                        scenes=scenes,
                        manifest=VisionManifest(
                            schema_version=VISION_SCHEMA_VERSION,
                            source_url=metadata.source_url,
                            model=self._model,
                            prompt_version=self._prompt_version,
                            generated_at=datetime.now(timezone.utc),
                        ),
                    ),
                )
                state = SyncState.GENERATED
            return VisionResult(
                state=state,
                scene_count=len(scenes),
                indexing=self._sync_vectors(video_id, metadata, scenes),
            )
        except (OSError, ValueError, VisionProviderError) as error:
            return VisionResult(state=SyncState.WARNING, warning=str(error))

    def get_vector_status(self, video_id: str) -> VisionVectorIndexStatus:
        """Read vector-index status without constructing an embedding provider."""
        try:
            source_url = self._cache.load_video(video_id).metadata.source_url
            entry = self._cache.get_vision_index_status(
                video_id,
                source_url=source_url,
                model=self._model,
                prompt_version=self._prompt_version,
            ).entry
            return self._repository_factory(video_id).get_index_status(
                entry.scenes if entry else None
            )
        except (OSError, ValueError, VisionIndexRepositoryError):
            return VisionVectorIndexStatus(state=CacheState.INVALID)

    def _sync_vectors(
        self, video_id: str, metadata: VideoMetadata, scenes: tuple[VisionScene, ...]
    ) -> IndexingResult:
        try:
            repository = self._repository_factory(video_id)
            if not repository.needs_indexing(scenes):
                return IndexingResult(state=SyncState.CURRENT)
            count = repository.index_scenes(
                scenes,
                metadata.title or f"YouTube video {video_id}",
                self._embedding_provider_factory(),
            )
            return IndexingResult(state=SyncState.INDEXED, chunk_count=count)
        except (
            EmbeddingProviderError,
            OSError,
            ValueError,
            VisionIndexRepositoryError,
        ) as error:
            return IndexingResult(state=SyncState.WARNING, warning=str(error))
