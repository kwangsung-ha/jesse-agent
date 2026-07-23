"""Application use cases for ingesting videos and reading their status."""

import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from youtube_transcript_api._errors import YouTubeTranscriptApiException

from tubetalk.core.cache import LocalCacheManager
from tubetalk.domain.summary import (
    SUMMARY_SCHEMA_VERSION,
    SummaryCacheEntry,
    SummaryManifest,
    VideoSummary,
)
from tubetalk.domain.transcript_index import transcript_sha256
from tubetalk.domain.video_status import VideoStatus
from tubetalk.domain.vision import (
    VISION_SCHEMA_VERSION,
    VisionIndexEntry,
    VisionManifest,
    YouTubeUrlVisionSource,
)
from tubetalk.pipeline.loader import YouTubeLoader
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


@dataclass(frozen=True)
class IndexingResult:
    """Outcome of checking or updating a transcript vector index."""

    state: str
    chunk_count: Optional[int] = None
    warning: Optional[str] = None


@dataclass(frozen=True)
class SummaryResult:
    """Outcome of checking or updating a cached video summary."""

    state: str
    summary: Optional[VideoSummary] = None
    warning: Optional[str] = None


@dataclass(frozen=True)
class VisionResult:
    """Outcome of checking or generating a cached visual scene index."""

    state: str
    scene_count: Optional[int] = None
    warning: Optional[str] = None
    indexing: IndexingResult = field(
        default_factory=lambda: IndexingResult(state="missing")
    )


@dataclass(frozen=True)
class ProcessResult:
    """Outcome of processing a video URL into the local cache."""

    video_id: str
    cache_hit: bool
    transcript_segments: int
    indexing: IndexingResult
    summary: SummaryResult
    vision: VisionResult = field(default_factory=lambda: VisionResult(state="missing"))


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

    def process(self, url: str) -> ProcessResult:
        """Fetch or reuse a video cache, then bring its index up to date."""
        try:
            video_id = self._loader.extract_video_id(url)
        except ValueError as error:
            raise InvalidVideoUrlError(str(error)) from error

        if self._cache.has_cache(video_id):
            metadata, transcript = self._load_cached_resources(video_id)
            return ProcessResult(
                video_id=video_id,
                cache_hit=True,
                transcript_segments=len(transcript),
                indexing=self._sync_transcript_index(video_id, metadata, transcript),
                summary=self._sync_summary(video_id, metadata, transcript),
                vision=self._sync_vision_index(video_id, metadata),
            )

        try:
            metadata = self._loader.fetch_metadata(url)
            transcript = self._loader.fetch_transcript(video_id)
        except (
            json.JSONDecodeError,
            OSError,
            subprocess.CalledProcessError,
            YouTubeTranscriptApiException,
        ) as error:
            raise VideoIngestionError(
                f"Failed to process {video_id}: {error}"
            ) from error

        metadata.update(
            {
                "video_id": video_id,
                "source_url": url,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._cache.save_json(video_id, "metadata.json", metadata)
        self._cache.save_json(video_id, "transcript.json", transcript)
        return ProcessResult(
            video_id=video_id,
            cache_hit=False,
            transcript_segments=len(transcript),
            indexing=self._sync_transcript_index(video_id, metadata, transcript),
            summary=self._sync_summary(video_id, metadata, transcript),
            vision=self._sync_vision_index(video_id, metadata),
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
        metadata, transcript = self._load_cached_resources(video_id)
        status = self._cache.get_summary_status(
            video_id,
            transcript,
            model=self._summary_model,
            prompt_version=self._summary_prompt_version,
            language=self._summary_language,
        )
        if status.state == "current" and status.entry is not None:
            return SummaryResult(state="current", summary=status.entry.summary)
        if not generate:
            raise SummaryUnavailableError(
                f"Summary for '{video_id}' is {status.state}. "
                f"Run 'tubetalk summary {video_id} --generate' to create it."
            )
        result = self._sync_summary(video_id, metadata, transcript)
        if result.state == "warning":
            raise SummaryGenerationError(result.warning or "Failed to generate summary")
        return result

    def _video_status(self, status: VideoStatus) -> VideoStatus:
        video_id = status.video_id
        segments: Optional[list[dict[str, Any]]] = None
        if status.has_transcript:
            try:
                loaded_segments = self._cache.load_json(video_id, "transcript.json")
                if isinstance(loaded_segments, list):
                    segments = loaded_segments
            except (OSError, ValueError):
                pass
        try:
            repository = self._transcript_index_repository_factory(video_id)
            index_status = repository.get_index_status(segments)
        except (OSError, TranscriptIndexRepositoryError):
            index_status = TranscriptIndexStatus(state="invalid")
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

    def _load_cached_resources(
        self, video_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metadata = self._cache.load_json(video_id, "metadata.json")
        transcript = self._cache.load_json(video_id, "transcript.json")
        if not isinstance(metadata, dict) or not isinstance(transcript, list):
            raise VideoIngestionError(
                "Cached metadata or transcript has an invalid format"
            )
        return metadata, transcript

    def _sync_transcript_index(
        self,
        video_id: str,
        metadata: dict[str, Any],
        transcript: list[dict[str, Any]],
    ) -> IndexingResult:
        try:
            repository = self._transcript_index_repository_factory(video_id)
            if not repository.needs_indexing(transcript):
                return IndexingResult(state="current")

            title = metadata.get("title")
            if not isinstance(title, str) or not title:
                title = f"YouTube video {video_id}"
            provider = self._embedding_provider_factory()
            chunk_count = repository.index_transcript(transcript, title, provider)
            return IndexingResult(state="indexed", chunk_count=chunk_count)
        except (
            EmbeddingProviderError,
            OSError,
            TranscriptIndexRepositoryError,
            ValueError,
        ) as error:
            return IndexingResult(state="warning", warning=str(error))

    def _sync_summary(
        self,
        video_id: str,
        metadata: dict[str, Any],
        transcript: list[dict[str, Any]],
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
            if status.state == "current" and status.entry is not None:
                return SummaryResult(state="current", summary=status.entry.summary)

            provider = self._summary_provider_factory()
            summary = provider.generate_summary(
                transcript,
                title=self._video_title(video_id, metadata),
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
                        generated_at=datetime.now(timezone.utc).isoformat(),
                    ),
                ),
            )
            return SummaryResult(state="generated", summary=summary)
        except (OSError, SummaryProviderError, ValueError) as error:
            return SummaryResult(state="warning", warning=str(error))

    def _sync_vision_index(
        self, video_id: str, metadata: dict[str, Any]
    ) -> VisionResult:
        """Reuse or generate visual scenes without affecting text-cache success."""
        source_url = metadata.get("source_url")
        duration = metadata.get("duration")
        if not isinstance(source_url, str) or not source_url:
            return VisionResult(
                state="warning", warning="Cached metadata does not contain a source URL"
            )
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            return VisionResult(
                state="warning",
                warning="Cached metadata does not contain a valid duration",
            )
        try:
            status = self._cache.get_vision_index_status(
                video_id,
                source_url=source_url,
                model=self._vision_model,
                prompt_version=self._vision_prompt_version,
            )
            if status.state == "current" and status.entry is not None:
                return VisionResult(
                    state="current",
                    scene_count=len(status.entry.scenes),
                    indexing=self._sync_vision_vectors(
                        video_id, metadata, status.entry.scenes
                    ),
                )
            analyzer = self._vision_analyzer_factory()
            scenes = analyzer.describe(
                YouTubeUrlVisionSource(source_url),
                title=self._video_title(video_id, metadata),
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
                        generated_at=datetime.now(timezone.utc).isoformat(),
                    ),
                ),
            )
            return VisionResult(
                state="generated",
                scene_count=len(scenes),
                indexing=self._sync_vision_vectors(video_id, metadata, scenes),
            )
        except (OSError, ValueError, VisionProviderError) as error:
            return VisionResult(state="warning", warning=str(error))

    def _sync_vision_vectors(
        self, video_id: str, metadata: dict[str, Any], scenes: tuple[Any, ...]
    ) -> IndexingResult:
        """Index visual scene descriptions without failing the scene cache."""
        try:
            repository = self._vision_index_repository_factory(video_id)
            if not repository.needs_indexing(scenes):
                return IndexingResult(state="current")
            provider = self._embedding_provider_factory()
            count = repository.index_scenes(
                scenes, self._video_title(video_id, metadata), provider
            )
            return IndexingResult(state="indexed", chunk_count=count)
        except (
            EmbeddingProviderError,
            OSError,
            ValueError,
            VisionIndexRepositoryError,
        ) as error:
            return IndexingResult(state="warning", warning=str(error))

    def _vision_vector_status(self, video_id: str) -> VisionVectorIndexStatus:
        """Read the scene-vector manifest without invoking an embedding provider."""
        try:
            metadata = self._cache.load_json(video_id, "metadata.json")
            source_url = (
                metadata.get("source_url") if isinstance(metadata, dict) else None
            )
            if not isinstance(source_url, str):
                return VisionVectorIndexStatus(state="missing")
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
            return VisionVectorIndexStatus(state="invalid")

    @staticmethod
    def _video_title(video_id: str, metadata: dict[str, Any]) -> str:
        """Choose a useful fallback title for provider prompts."""
        title = metadata.get("title")
        if isinstance(title, str) and title:
            return title
        return f"YouTube video {video_id}"
