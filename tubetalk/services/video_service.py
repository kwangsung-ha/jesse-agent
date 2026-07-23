"""Application use cases for ingesting videos and reading their status."""

import json
import subprocess
from dataclasses import dataclass, replace
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
from tubetalk.pipeline.loader import YouTubeLoader
from tubetalk.ports.embedding import EmbeddingProvider, EmbeddingProviderError
from tubetalk.ports.summary import SummaryProvider, SummaryProviderError
from tubetalk.ports.transcript_index_repository import (
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
    TranscriptIndexStatus,
)


class VideoServiceError(Exception):
    """Base exception for failures that an interface can present to a user."""


class InvalidVideoUrlError(VideoServiceError):
    """Raised when a URL does not identify a supported YouTube video."""


class VideoNotFoundError(VideoServiceError):
    """Raised when a requested video is absent from the local cache."""


class VideoIngestionError(VideoServiceError):
    """Raised when metadata or transcript collection fails."""


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
class ProcessResult:
    """Outcome of processing a video URL into the local cache."""

    video_id: str
    cache_hit: bool
    transcript_segments: int
    indexing: IndexingResult
    summary: SummaryResult


class VideoService:
    """Coordinate cache, YouTube loading, and transcript indexing use cases."""

    def __init__(
        self,
        cache: LocalCacheManager,
        loader: YouTubeLoader,
        embedding_provider_factory: Callable[[], EmbeddingProvider],
        transcript_index_repository_factory: Callable[[str], TranscriptIndexRepository],
        summary_provider_factory: Callable[[], SummaryProvider],
        summary_model: str,
        summary_prompt_version: str,
        summary_language: str,
    ) -> None:
        """Create a service with explicit infrastructure dependencies."""
        self._cache = cache
        self._loader = loader
        self._transcript_index_repository_factory = transcript_index_repository_factory
        self._embedding_provider_factory = embedding_provider_factory
        self._summary_provider_factory = summary_provider_factory
        self._summary_model = summary_model
        self._summary_prompt_version = summary_prompt_version
        self._summary_language = summary_language

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
        return replace(
            status,
            transcript_index_state=index_status.state,
            transcript_index_chunks=index_status.chunk_count,
            transcript_index_model=index_status.embedding_model,
            transcript_index_dimension=index_status.embedding_dimension,
            transcript_indexed_at=index_status.indexed_at,
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

    @staticmethod
    def _video_title(video_id: str, metadata: dict[str, Any]) -> str:
        """Choose a useful fallback title for provider prompts."""
        title = metadata.get("title")
        if isinstance(title, str) and title:
            return title
        return f"YouTube video {video_id}"
