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
from tubetalk.pipeline.loader import YouTubeLoader
from tubetalk.ports.embedding import EmbeddingProvider, EmbeddingProviderError
from tubetalk.ports.summary import SummaryProvider, SummaryProviderError
from tubetalk.ports.transcript_index_repository import (
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
)
from tubetalk.services.results import IndexingResult, SummaryResult


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
