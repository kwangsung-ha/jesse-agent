"""Hybrid transcript and vision retrieval with reciprocal-rank fusion."""

from collections.abc import Iterable

from jesseagent.application.embedding import EmbeddingProvider, EmbeddingProviderError
from jesseagent.application.video.contracts import (
    TranscriptIndexRepository,
    TranscriptIndexRepositoryError,
    VisionIndexRepository,
    VisionIndexRepositoryError,
)
from jesseagent.core.logging import logger
from jesseagent.domain.retrieval import RetrievalHit
from jesseagent.domain.state import CacheState
from jesseagent.domain.transcript import Transcript
from jesseagent.domain.vision import VisionScene

RRF_K = 60
RETRIEVAL_LIMIT = 5


class HybridRetrievalError(Exception):
    """Raised when current dual-index evidence cannot be retrieved."""


class HybridRetriever:
    """Retrieve each modality independently, then fuse their ranked results."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        transcript_repository: TranscriptIndexRepository,
        vision_repository: VisionIndexRepository,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._transcript_repository = transcript_repository
        self._vision_repository = vision_repository

    def retrieve(
        self,
        query: str,
        transcript: Transcript,
        scenes: tuple[VisionScene, ...],
    ) -> tuple[RetrievalHit, ...]:
        """Return fused evidence after checking both source indexes are current."""
        if not query.strip():
            raise HybridRetrievalError("Question must not be empty")
        transcript_status = self._transcript_repository.get_index_status(transcript)
        vision_status = self._vision_repository.get_index_status(scenes)
        if (
            transcript_status.state != CacheState.CURRENT
            or vision_status.state != CacheState.CURRENT
        ):
            raise HybridRetrievalError(
                "Transcript and vision indexes must be current. "
                "Ask JesseAgent to process the video first."
            )
        try:
            embedding = self._embedding_provider.embed_query(query)
            transcript_hits = self._transcript_repository.search(
                embedding, RETRIEVAL_LIMIT
            )
            vision_hits = self._vision_repository.search(embedding, RETRIEVAL_LIMIT)
        except (
            EmbeddingProviderError,
            TranscriptIndexRepositoryError,
            VisionIndexRepositoryError,
            OSError,
            ValueError,
        ) as error:
            raise HybridRetrievalError(str(error)) from error
        fused = _rrf_fuse(transcript_hits, vision_hits)
        if not fused:
            raise HybridRetrievalError("No indexed evidence was found for this video")
        selected = tuple(fused[:RETRIEVAL_LIMIT])
        logger.bind(event="retrieval.complete").debug(
            "query={}\n{}",
            query,
            "\n".join(
                f"{hit.source_id} source={hit.source} interval="
                f"{hit.start_sec:.3f}-{hit.end_sec:.3f} score={hit.score:.5f}"
                for hit in selected
            ),
        )
        return selected


def _rrf_fuse(*ranked_lists: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """Combine ranked lists using RRF and deterministic tie breakers."""
    scores: dict[str, float] = {}
    hits: dict[str, RetrievalHit] = {}
    for ranked in ranked_lists:
        for hit in ranked:
            scores[hit.source_id] = scores.get(hit.source_id, 0.0) + 1 / (
                RRF_K + hit.rank
            )
            hits[hit.source_id] = hit
    return sorted(
        (
            hits[source_id].model_copy(update={"score": score})
            for source_id, score in scores.items()
        ),
        key=lambda hit: (-hit.score, hit.start_sec, hit.source_id),
    )
