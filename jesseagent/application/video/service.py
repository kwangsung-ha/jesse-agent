"""Application use cases for ingesting videos and reading their status."""

from time import perf_counter
from typing import Callable, Optional

from jesseagent.agent.retriever import HybridRetrievalError, HybridRetriever
from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.application.video.contracts import (
    ChatProvider,
    ChatProviderError,
    SummaryProvider,
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
    TranscriptIndexStatus,
    VisionAnalyzer,
    VisionIndexRepository,
)
from jesseagent.application.video.results import (
    ProcessResult,
    ProcessTiming,
    SummaryResult,
)
from jesseagent.application.video.stages import (
    SummaryGenerationStage,
    TranscriptIndexingStage,
    VideoIngestionStage,
    VisionIndexingStage,
)
from jesseagent.core.cache import LocalCacheManager
from jesseagent.core.logging import logger
from jesseagent.domain.chaptering import ChapterBlockPolicy, ChapterWindowPolicy
from jesseagent.domain.retrieval import ChatAnswer, ChatTurn, RetrievalHit
from jesseagent.domain.state import CacheState, SyncState
from jesseagent.domain.transcript import Transcript
from jesseagent.domain.video import CachedVideo
from jesseagent.domain.video_status import VideoStatus
from jesseagent.domain.vision import VisionScene
from jesseagent.pipeline.loader import (
    InvalidVideoUrlError as LoaderInvalidVideoUrlError,
)
from jesseagent.pipeline.loader import (
    VideoLoaderError,
    YouTubeLoader,
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


class ChatUnavailableError(VideoServiceError):
    """Raised when a video cannot provide current dual-index evidence."""


class ChatGenerationError(VideoServiceError):
    """Raised when an answer provider cannot return valid grounded output."""


class ChatSession:
    """An in-memory, video-scoped conversational session."""

    def __init__(
        self,
        retriever: HybridRetriever,
        provider: ChatProvider,
        cached_video: CachedVideo,
        scenes: tuple[VisionScene, ...],
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._cached_video = cached_video
        self._scenes = scenes
        self._history: list[ChatTurn] = []
        self._last_evidence: tuple[RetrievalHit, ...] = ()

    @property
    def last_evidence(self) -> tuple[RetrievalHit, ...]:
        """Return the evidence used for the most recent successful response."""
        return self._last_evidence

    def ask(self, question: str) -> ChatAnswer:
        """Retrieve fresh evidence and append one validated answer to this session."""
        try:
            evidence = self._retriever.retrieve(
                question, self._cached_video.transcript, self._scenes
            )
        except HybridRetrievalError as error:
            raise ChatUnavailableError(str(error)) from error
        try:
            answer = self._provider.answer(question, evidence, tuple(self._history))
        except ChatProviderError as error:
            raise ChatGenerationError(str(error)) from error
        self._last_evidence = evidence
        self._history.append(ChatTurn(question=question, answer=answer))
        return answer


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
        chat_provider_factory: Callable[[], ChatProvider] | None = None,
        summary_chapter_window_policy: str = ChapterWindowPolicy().cache_key,
        summary_chapter_block_policy: str = ChapterBlockPolicy().cache_key,
    ) -> None:
        """Create a service with explicit infrastructure dependencies."""
        self._cache = cache
        self._loader = loader
        self._transcript_index_repository_factory = transcript_index_repository_factory
        self._embedding_provider_factory = embedding_provider_factory
        self._summary_provider_factory = summary_provider_factory
        self._vision_analyzer_factory = vision_analyzer_factory
        self._vision_index_repository_factory = vision_index_repository_factory
        self._chat_provider_factory = chat_provider_factory
        self._summary_model = summary_model
        self._summary_prompt_version = summary_prompt_version
        self._summary_language = summary_language
        self._summary_chapter_window_policy = summary_chapter_window_policy
        self._summary_chapter_block_policy = summary_chapter_block_policy
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
            summary_chapter_window_policy,
            summary_chapter_block_policy,
        )
        self._vision_stage = VisionIndexingStage(
            cache,
            vision_analyzer_factory,
            vision_index_repository_factory,
            embedding_provider_factory,
            vision_model,
            vision_prompt_version,
        )

    def process(self, url: str) -> ProcessResult:
        """Fetch or reuse a video cache, then bring its index up to date."""
        logger.bind(event="service.process.start").debug("url={}", url)
        total_started = perf_counter()
        ingestion_started = perf_counter()
        try:
            ingestion = self._ingestion_stage.load_or_collect(url)
        except LoaderInvalidVideoUrlError as error:
            raise InvalidVideoUrlError(str(error)) from error
        except VideoLoaderError as error:
            raise VideoIngestionError(f"Failed to process: {error}") from error
        ingestion_sec = perf_counter() - ingestion_started
        logger.bind(
            event="service.ingestion.complete", video_id=ingestion.video_id
        ).debug("cache_hit={} elapsed_sec={:.3f}", ingestion.cache_hit, ingestion_sec)
        video_id = ingestion.video_id
        cached_video = ingestion.video
        indexing_started = perf_counter()
        indexing = self._transcript_indexing_stage.sync(
            video_id, cached_video.metadata, cached_video.transcript
        )
        transcript_index_sec = perf_counter() - indexing_started
        logger.bind(event="service.transcript_index.complete", video_id=video_id).debug(
            "state={} chunks={} elapsed_sec={:.3f}",
            indexing.state,
            indexing.chunk_count,
            transcript_index_sec,
        )
        summary_started = perf_counter()
        summary = self._summary_stage.sync(
            video_id, cached_video.metadata, cached_video.transcript
        )
        summary_sec = perf_counter() - summary_started
        logger.bind(event="service.summary.complete", video_id=video_id).debug(
            "state={} elapsed_sec={:.3f}", summary.state, summary_sec
        )
        vision_started = perf_counter()
        vision = self._vision_stage.sync(video_id, cached_video.metadata)
        vision_sec = perf_counter() - vision_started
        logger.bind(event="service.vision.complete", video_id=video_id).debug(
            "state={} scenes={} elapsed_sec={:.3f}",
            vision.state,
            vision.scene_count,
            vision_sec,
        )
        logger.bind(event="service.process.complete", video_id=video_id).debug(
            "elapsed_sec={:.3f}", perf_counter() - total_started
        )
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
        statuses = [
            self._video_status(status) for status in self._cache.list_cached_videos()
        ]
        logger.bind(event="service.status.list").debug(
            "cached_videos={}", len(statuses)
        )
        return statuses

    def get_status(self, video_id: str) -> VideoStatus:
        """Return one cached video's status or raise a service-level error."""
        status = self._cache.get_video_status(video_id)
        if status is None:
            raise VideoNotFoundError(f"Video '{video_id}' not found in local cache.")
        logger.bind(event="service.status.get", video_id=video_id).debug("loaded")
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
            chapter_window_policy=self._summary_chapter_window_policy,
            chapter_block_policy=self._summary_chapter_block_policy,
        )
        if status.state == CacheState.CURRENT and status.entry is not None:
            logger.bind(event="service.summary.cache", video_id=video_id).debug(
                "state=current"
            )
            return SummaryResult(state=SyncState.CURRENT, summary=status.entry.summary)
        if not generate:
            raise SummaryUnavailableError(
                f"Summary for '{video_id}' is {status.state}. "
                "Ask JesseAgent to generate or refresh the summary."
            )
        result = self._summary_stage.sync(
            video_id, cached_video.metadata, cached_video.transcript
        )
        logger.bind(event="service.summary.generate", video_id=video_id).debug(
            "state={}", result.state
        )
        if result.state == "warning":
            raise SummaryGenerationError(result.warning or "Failed to generate summary")
        return result

    def create_chat_session(self, video_id: str) -> ChatSession:
        """Create one non-persistent session after verifying cached source data."""
        if not self._cache.has_cache(video_id):
            raise VideoNotFoundError(f"Video '{video_id}' not found in local cache.")
        cached_video = self._load_cached_resources(video_id)
        vision_status = self._cache.get_vision_index_status(
            video_id,
            source_url=cached_video.metadata.source_url,
            model=self._vision_model,
            prompt_version=self._vision_prompt_version,
        )
        if vision_status.entry is None:
            raise ChatUnavailableError(
                "Vision scenes are unavailable or stale. "
                "Ask JesseAgent to process the video first."
            )
        return ChatSession(
            HybridRetriever(
                self._embedding_provider_factory(),
                self._transcript_index_repository_factory(video_id),
                self._vision_index_repository_factory(video_id),
            ),
            self._get_chat_provider(),
            cached_video,
            vision_status.entry.scenes,
        )

    def _get_chat_provider(self) -> ChatProvider:
        if self._chat_provider_factory is None:
            raise ChatGenerationError("Chat provider is not configured")
        return self._chat_provider_factory()

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
        vision_vector_status = self._vision_stage.get_vector_status(video_id)
        return status.model_copy(
            update={
                "transcript_index_state": index_status.state,
                "transcript_index_chunks": index_status.chunk_count,
                "transcript_index_model": index_status.embedding_model,
                "transcript_index_dimension": index_status.embedding_dimension,
                "transcript_indexed_at": index_status.indexed_at,
                "vision_vector_index_state": vision_vector_status.state,
                "vision_vector_index_scenes": vision_vector_status.scene_count,
                "vision_vector_index_model": vision_vector_status.embedding_model,
                "vision_vector_index_dimension": (
                    vision_vector_status.embedding_dimension
                ),
                "vision_vector_indexed_at": vision_vector_status.indexed_at,
            }
        )

    def _load_cached_resources(self, video_id: str) -> CachedVideo:
        try:
            return self._cache.load_video(video_id)
        except (OSError, ValueError) as error:
            raise VideoIngestionError("Cached video has an invalid format") from error
