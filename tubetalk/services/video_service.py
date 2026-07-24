"""Application use cases for ingesting videos and reading their status."""

from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Optional

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
from tubetalk.domain.video_status import VideoStatus
from tubetalk.domain.vision import (
    VISION_SCHEMA_VERSION,
    VisionIndexEntry,
    VisionManifest,
    VisionScene,
    YouTubeUrlVisionSource,
)
from tubetalk.pipeline.loader import (
    InvalidVideoUrlError as LoaderInvalidVideoUrlError,
)
from tubetalk.pipeline.loader import (
    VideoLoaderError,
    YouTubeLoader,
)
from tubetalk.ports.embedding import EmbeddingProvider, EmbeddingProviderError
from tubetalk.ports.summary import SummaryProvider, SummaryProviderError
from tubetalk.ports.transcript_index_repository import (
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
    TranscriptIndexStatus,
)
from tubetalk.ports.vision import VisionAnalyzer, VisionProviderError
from tubetalk.ports.vision_index_repository import (
    VisionIndexRepository,
    VisionIndexRepositoryError,
    VisionVectorIndexStatus,
)
from tubetalk.services.results import (
    IndexingResult,
    ProcessResult,
    ProcessTiming,
    SummaryResult,
    VisionResult,
)
from tubetalk.services.stages import (
    SummaryGenerationStage,
    TranscriptIndexingStage,
    VideoIngestionStage,
)


class VideoServiceError(Exception):
    """Base exception for failures that an interface can present to a user."""


class InvalidVideoUrlError(VideoServiceError):
    """Raised when a URL does not identify a supported YouTube video."""


class VideoNotFoundError(VideoServiceError):
    """Raised when a requested video is absent from the local cache."""


class VideoIngestionError(VideoServiceError):
    """Raised when metadata or transcript collection fails."""


class SummaryUnavailableError(VideoServiceError):
    """Raised when a requested summary is absent or stale without generation."""


class SummaryGenerationError(VideoServiceError):
    """Raised when an explicitly requested summary cannot be generated."""


class VideoService:
    """Coordinate cache, YouTube loading, and transcript indexing use cases."""

    def __init__(
        self,
        cache: LocalCacheManager,
        loader: YouTubeLoader,
        embedding_provider_factory: Callable[[], EmbeddingProvider],
        transcript_index_repository_factory: Callable[[str], TranscriptIndexRepository],
        summary_provider_factory: Callable[[], SummaryProvider],
        vision_analyzer_factory: Callable[[], VisionAnalyzer],
        vision_index_repository_factory: Callable[[str], VisionIndexRepository],
        summary_model: str,
        summary_prompt_version: str,
        summary_language: str,
        vision_model: str,
        vision_prompt_version: str,
    ) -> None:
        """Create a service with explicit infrastructure dependencies."""
        self._cache = cache
        self._loader = loader
        self._transcript_index_repository_factory = transcript_index_repository_factory
        self._embedding_provider_factory = embedding_provider_factory
        self._summary_provider_factory = summary_provider_factory
        self._vision_analyzer_factory = vision_analyzer_factory
        self._vision_index_repository_factory = vision_index_repository_factory
        self._summary_model = summary_model
        self._summary_prompt_version = summary_prompt_version
        self._summary_language = summary_language
        self._vision_model = vision_model
        self._vision_prompt_version = vision_prompt_version
        self._ingestion_stage = VideoIngestionStage(cache, loader)
        self._transcript_indexing_stage = TranscriptIndexingStage(
            transcript_index_repository_factory, embedding_provider_factory
        )
        self._summary_stage = SummaryGenerationStage(
            cache,
            summary_provider_factory,
            summary_model,
            summary_prompt_version,
            summary_language,
        )

    def process(self, url: str) -> ProcessResult:
        """Fetch or reuse a video cache, then bring its index up to date."""
        total_started = perf_counter()
        ingestion_started = perf_counter()
        try:
            ingestion = self._ingestion_stage.load_or_collect(url)
        except LoaderInvalidVideoUrlError as error:
            raise InvalidVideoUrlError(str(error)) from error
        except VideoLoaderError as error:
            raise VideoIngestionError(f"Failed to process: {error}") from error
        ingestion_sec = perf_counter() - ingestion_started
        video_id = ingestion.video_id
        cached_video = ingestion.video
        indexing_started = perf_counter()
        indexing = self._transcript_indexing_stage.sync(
            video_id, cached_video.metadata, cached_video.transcript
        )
        transcript_index_sec = perf_counter() - indexing_started
        summary_started = perf_counter()
        summary = self._summary_stage.sync(
            video_id, cached_video.metadata, cached_video.transcript
        )
        summary_sec = perf_counter() - summary_started
        vision_started = perf_counter()
        vision = self._sync_vision_index(video_id, cached_video.metadata)
        vision_sec = perf_counter() - vision_started
        return ProcessResult(
            video_id=video_id,
            cache_hit=ingestion.cache_hit,
            transcript_segments=len(cached_video.transcript),
            indexing=indexing,
            summary=summary,
            vision=vision,
            timing=ProcessTiming(
                ingestion_sec=ingestion_sec,
                transcript_index_sec=transcript_index_sec,
                summary_sec=summary_sec,
                vision_sec=vision_sec,
                total_sec=perf_counter() - total_started,
            ),
        )

    def list_statuses(self) -> list[VideoStatus]:
        """Return status for every locally cached video."""
        return [
            self._video_status(status) for status in self._cache.list_cached_videos()
        ]

    def get_status(self, video_id: str) -> VideoStatus:
        """Return one cached video's status or raise a service-level error."""
        status = self._cache.get_video_status(video_id)
        if status is None:
            raise VideoNotFoundError(f"Video '{video_id}' not found in local cache.")
        return self._video_status(status)

    def get_summary(self, video_id: str, *, generate: bool = False) -> SummaryResult:
        """Return a current cached summary or explicitly generate one when allowed."""
        if not self._cache.has_cache(video_id):
            raise VideoNotFoundError(f"Video '{video_id}' not found in local cache.")
        cached_video = self._load_cached_resources(video_id)
        status = self._cache.get_summary_status(
            video_id,
            cached_video.transcript,
            model=self._summary_model,
            prompt_version=self._summary_prompt_version,
            language=self._summary_language,
        )
        if status.state == CacheState.CURRENT and status.entry is not None:
            return SummaryResult(state=SyncState.CURRENT, summary=status.entry.summary)
        if not generate:
            raise SummaryUnavailableError(
                f"Summary for '{video_id}' is {status.state}. "
                f"Run 'tubetalk summary {video_id} --generate' to create it."
            )
        result = self._sync_summary(
            video_id, cached_video.metadata, cached_video.transcript
        )
        if result.state == "warning":
            raise SummaryGenerationError(result.warning or "Failed to generate summary")
        return result

    def _video_status(self, status: VideoStatus) -> VideoStatus:
        video_id = status.video_id
        transcript: Optional[Transcript] = None
        if status.has_transcript:
            try:
                transcript = self._cache.load_video(video_id).transcript
            except (OSError, ValueError):
                pass
        try:
            repository = self._transcript_index_repository_factory(video_id)
            index_status = repository.get_index_status(transcript)
        except (OSError, TranscriptIndexRepositoryError):
            index_status = TranscriptIndexStatus(state=CacheState.INVALID)
        vision_vector_status = self._vision_vector_status(video_id)
        return replace(
            status,
            transcript_index_state=index_status.state,
            transcript_index_chunks=index_status.chunk_count,
            transcript_index_model=index_status.embedding_model,
            transcript_index_dimension=index_status.embedding_dimension,
            transcript_indexed_at=index_status.indexed_at,
            vision_vector_index_state=vision_vector_status.state,
            vision_vector_index_scenes=vision_vector_status.scene_count,
            vision_vector_index_model=vision_vector_status.embedding_model,
            vision_vector_index_dimension=vision_vector_status.embedding_dimension,
            vision_vector_indexed_at=vision_vector_status.indexed_at,
        )

    def _load_cached_resources(self, video_id: str) -> CachedVideo:
        try:
            return self._cache.load_video(video_id)
        except (OSError, ValueError) as error:
            raise VideoIngestionError("Cached video has an invalid format") from error

    def _sync_transcript_index(
        self,
        video_id: str,
        metadata: VideoMetadata,
        transcript: Transcript,
    ) -> IndexingResult:
        try:
            repository = self._transcript_index_repository_factory(video_id)
            if not repository.needs_indexing(transcript):
                return IndexingResult(state=SyncState.CURRENT)

            provider = self._embedding_provider_factory()
            chunk_count = repository.index_transcript(
                transcript, self._video_title(metadata), provider
            )
            return IndexingResult(state=SyncState.INDEXED, chunk_count=chunk_count)
        except (
            EmbeddingProviderError,
            OSError,
            TranscriptIndexRepositoryError,
            ValueError,
        ) as error:
            return IndexingResult(state=SyncState.WARNING, warning=str(error))

    def _sync_summary(
        self,
        video_id: str,
        metadata: VideoMetadata,
        transcript: Transcript,
    ) -> SummaryResult:
        """Reuse or regenerate the summary without discarding cached resources."""
        try:
            status = self._cache.get_summary_status(
                video_id,
                transcript,
                model=self._summary_model,
                prompt_version=self._summary_prompt_version,
                language=self._summary_language,
            )
            if status.state == CacheState.CURRENT and status.entry is not None:
                return SummaryResult(
                    state=SyncState.CURRENT, summary=status.entry.summary
                )

            provider = self._summary_provider_factory()
            summary = provider.generate_summary(
                transcript,
                title=self._video_title(metadata),
                language=self._summary_language,
            )
            self._cache.save_summary(
                video_id,
                SummaryCacheEntry(
                    summary=summary,
                    manifest=SummaryManifest(
                        schema_version=SUMMARY_SCHEMA_VERSION,
                        transcript_sha256=transcript_sha256(transcript),
                        model=self._summary_model,
                        prompt_version=self._summary_prompt_version,
                        language=self._summary_language,
                        generated_at=datetime.now(timezone.utc),
                    ),
                ),
            )
            return SummaryResult(state=SyncState.GENERATED, summary=summary)
        except (OSError, SummaryProviderError, ValueError) as error:
            return SummaryResult(state=SyncState.WARNING, warning=str(error))

    def _sync_vision_index(
        self, video_id: str, metadata: VideoMetadata
    ) -> VisionResult:
        """Reuse or generate visual scenes without affecting text-cache success."""
        source_url = metadata.source_url
        duration = metadata.duration_sec
        if duration is None:
            return VisionResult(
                state=SyncState.WARNING,
                warning="Cached metadata does not contain a valid duration",
            )
        try:
            status = self._cache.get_vision_index_status(
                video_id,
                source_url=source_url,
                model=self._vision_model,
                prompt_version=self._vision_prompt_version,
            )
            if status.state == CacheState.CURRENT and status.entry is not None:
                return VisionResult(
                    state=SyncState.CURRENT,
                    scene_count=len(status.entry.scenes),
                    indexing=self._sync_vision_vectors(
                        video_id, metadata, status.entry.scenes
                    ),
                )
            analyzer = self._vision_analyzer_factory()
            scenes = analyzer.describe(
                YouTubeUrlVisionSource(source_url),
                title=self._video_title(metadata),
                duration_sec=float(duration),
            )
            self._cache.save_vision_index(
                video_id,
                VisionIndexEntry(
                    scenes=scenes,
                    manifest=VisionManifest(
                        schema_version=VISION_SCHEMA_VERSION,
                        source_url=source_url,
                        model=self._vision_model,
                        prompt_version=self._vision_prompt_version,
                        generated_at=datetime.now(timezone.utc),
                    ),
                ),
            )
            return VisionResult(
                state=SyncState.GENERATED,
                scene_count=len(scenes),
                indexing=self._sync_vision_vectors(video_id, metadata, scenes),
            )
        except (OSError, ValueError, VisionProviderError) as error:
            return VisionResult(state=SyncState.WARNING, warning=str(error))

    def _sync_vision_vectors(
        self, video_id: str, metadata: VideoMetadata, scenes: tuple[VisionScene, ...]
    ) -> IndexingResult:
        """Index visual scene descriptions without failing the scene cache."""
        try:
            repository = self._vision_index_repository_factory(video_id)
            if not repository.needs_indexing(scenes):
                return IndexingResult(state=SyncState.CURRENT)
            provider = self._embedding_provider_factory()
            count = repository.index_scenes(
                scenes, self._video_title(metadata), provider
            )
            return IndexingResult(state=SyncState.INDEXED, chunk_count=count)
        except (
            EmbeddingProviderError,
            OSError,
            ValueError,
            VisionIndexRepositoryError,
        ) as error:
            return IndexingResult(state=SyncState.WARNING, warning=str(error))

    def _vision_vector_status(self, video_id: str) -> VisionVectorIndexStatus:
        """Read the scene-vector manifest without invoking an embedding provider."""
        try:
            source_url = self._cache.load_video(video_id).metadata.source_url
            vision_entry = self._cache.get_vision_index_status(
                video_id,
                source_url=source_url,
                model=self._vision_model,
                prompt_version=self._vision_prompt_version,
            ).entry
            return self._vision_index_repository_factory(video_id).get_index_status(
                vision_entry.scenes if vision_entry else None
            )
        except (OSError, ValueError, VisionIndexRepositoryError):
            return VisionVectorIndexStatus(state=CacheState.INVALID)

    @staticmethod
    def _video_title(metadata: VideoMetadata) -> str:
        """Choose a useful fallback title for provider prompts."""
        return metadata.title or f"YouTube video {metadata.video_id}"
