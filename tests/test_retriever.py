"""Hybrid retrieval and RRF tests without a Chroma dependency."""

from typing import Any

import pytest

from tubetalk.agent.retriever import HybridRetrievalError, HybridRetriever, _rrf_fuse
from tubetalk.domain.retrieval import RetrievalHit
from tubetalk.domain.transcript import Transcript, TranscriptSegment
from tubetalk.domain.vision import VisionScene
from tubetalk.ports.transcript_index_repository import TranscriptIndexStatus
from tubetalk.ports.vision_index_repository import VisionVectorIndexStatus


def _hit(source_id: str, source: str, rank: int, start: float = 0) -> RetrievalHit:
    return RetrievalHit(
        source_id=source_id,
        source=source,
        text="evidence",
        start_sec=start,
        end_sec=start + 2,
        rank=rank,
        distance=0.1,
    )


def test_rrf_fuses_and_orders_results_deterministically() -> None:
    fused = _rrf_fuse(
        [_hit("text-1", "transcript", 1, 10)],
        [_hit("vision-1", "vision", 1, 5)],
    )

    assert [hit.source_id for hit in fused] == ["vision-1", "text-1"]
    assert fused[0].score == pytest.approx(1 / 61)


def test_retriever_requires_both_current_indexes(mocker: Any) -> None:
    embeddings = mocker.Mock()
    transcript_repo = mocker.Mock()
    vision_repo = mocker.Mock()
    transcript_repo.get_index_status.return_value = TranscriptIndexStatus(
        state="current"
    )
    vision_repo.get_index_status.return_value = VisionVectorIndexStatus(state="stale")
    retriever = HybridRetriever(embeddings, transcript_repo, vision_repo)

    with pytest.raises(HybridRetrievalError, match="Ask TubeTalk to process"):
        retriever.retrieve(
            "what happened?",
            Transcript(segments=(TranscriptSegment(start_sec=0, text="hello"),)),
            (VisionScene(0, 1, "title", ()),),
        )

    embeddings.embed_query.assert_not_called()


def test_retriever_embeds_once_and_limits_fused_results(mocker: Any) -> None:
    embeddings = mocker.Mock()
    embeddings.embed_query.return_value = [0.2, 0.3]
    transcript_repo = mocker.Mock()
    vision_repo = mocker.Mock()
    transcript_repo.get_index_status.return_value = TranscriptIndexStatus(
        state="current"
    )
    vision_repo.get_index_status.return_value = VisionVectorIndexStatus(state="current")
    transcript_repo.search.return_value = [_hit("text-1", "transcript", 1)]
    vision_repo.search.return_value = [_hit("vision-1", "vision", 1)]
    retriever = HybridRetriever(embeddings, transcript_repo, vision_repo)

    results = retriever.retrieve(
        "what happened?",
        Transcript(segments=(TranscriptSegment(start_sec=0, text="hello"),)),
        (VisionScene(0, 1, "title", ()),),
    )

    assert len(results) == 2
    embeddings.embed_query.assert_called_once_with("what happened?")
    transcript_repo.search.assert_called_once_with([0.2, 0.3], 5)
    vision_repo.search.assert_called_once_with([0.2, 0.3], 5)


def test_retriever_maps_query_backend_failures(mocker: Any) -> None:
    embeddings = mocker.Mock()
    embeddings.embed_query.return_value = [0.2]
    transcript_repo = mocker.Mock()
    vision_repo = mocker.Mock()
    transcript_repo.get_index_status.return_value = TranscriptIndexStatus(
        state="current"
    )
    vision_repo.get_index_status.return_value = VisionVectorIndexStatus(state="current")
    transcript_repo.search.side_effect = ValueError("bad query")
    retriever = HybridRetriever(embeddings, transcript_repo, vision_repo)

    with pytest.raises(HybridRetrievalError, match="bad query"):
        retriever.retrieve(
            "what happened?",
            Transcript(segments=(TranscriptSegment(start_sec=0, text="hello"),)),
            (VisionScene(0, 1, "title", ()),),
        )
