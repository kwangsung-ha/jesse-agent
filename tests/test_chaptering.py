"""Tests for coverage-first transcript chapter extraction primitives."""

import pytest

from tubetalk.domain.chaptering import (
    ChapterBlockPolicy,
    ChapterCandidate,
    ChapterWindowPolicy,
    block_transcript_segments,
    window_transcript,
)
from tubetalk.domain.transcript import Transcript, TranscriptSegment


def _transcript(*segments: tuple[float, float, str]) -> Transcript:
    return Transcript(
        segments=tuple(
            TranscriptSegment(start_sec=start, duration_sec=duration, text=text)
            for start, duration, text in segments
        )
    )


def test_window_transcript_preserves_overlap_at_duration_boundaries() -> None:
    """Topics beside a request boundary should be visible to both windows."""
    windows = window_transcript(
        _transcript(
            (0, 20, "one"),
            (20, 20, "two"),
            (40, 20, "three"),
            (60, 20, "four"),
        ),
        ChapterWindowPolicy(max_seconds=45, max_characters=100, overlap_seconds=30),
    )

    assert [window.first_segment_index for window in windows] == [0, 1, 2]
    assert [window.last_segment_index for window in windows] == [1, 2, 3]
    assert [window.start_sec for window in windows] == [0, 20, 40]
    assert [window.end_sec for window in windows] == [40, 60, 80]


def test_window_transcript_splits_by_characters_without_losing_segments() -> None:
    """Character bounds should not discard any source segment."""
    windows = window_transcript(
        _transcript((0, 0, "alpha"), (1, 0, "bravo"), (2, 0, "charlie")),
        ChapterWindowPolicy(max_seconds=45, max_characters=9, overlap_seconds=0),
    )

    assert [window.first_segment_index for window in windows] == [0, 1, 2]
    assert [window.last_segment_index for window in windows] == [0, 1, 2]


def test_window_transcript_keeps_a_single_oversized_segment() -> None:
    """One long segment must still receive a candidate-extraction request."""
    windows = window_transcript(
        _transcript((0, 100, "x" * 100), (100, 1, "next")),
        ChapterWindowPolicy(max_seconds=45, max_characters=10, overlap_seconds=5),
    )

    assert [window.last_segment_index for window in windows] == [0, 1]


def test_window_transcript_returns_no_windows_for_empty_transcript() -> None:
    """Empty transcripts cannot produce extraction requests."""
    assert window_transcript(Transcript(segments=())) == []


def test_block_transcript_segments_reconstructs_caption_fragments() -> None:
    """Fragments remain readable while retaining the first source timestamp."""
    blocks = block_transcript_segments(
        _transcript(
            (0, 1, "오늘 시장은"), (1, 1, "상승했습니다."), (2, 1, "다음 주제")
        ).segments
    )

    assert [(block.start_sec, block.end_sec, block.text) for block in blocks] == [
        (0, 2, "오늘 시장은 상승했습니다."),
        (2, 3, "다음 주제"),
    ]


def test_block_transcript_segments_breaks_at_gaps_and_hard_limits() -> None:
    """A prompt block never crosses silence or configured size limits."""
    blocks = block_transcript_segments(
        _transcript(
            (0, 2, "alpha"), (2, 2, "bravo"), (10, 1, "charlie"), (11, 1, "delta")
        ).segments,
        ChapterBlockPolicy(max_seconds=10, max_characters=10, max_gap_seconds=1.5),
    )

    assert [block.text for block in blocks] == ["alpha", "bravo", "charlie", "delta"]


def test_block_transcript_segments_keeps_an_oversized_segment() -> None:
    """An individual long caption remains available to candidate extraction."""
    blocks = block_transcript_segments(
        _transcript((0, 30, "x" * 500), (30, 1, "next")).segments,
        ChapterBlockPolicy(max_seconds=20, max_characters=400, max_gap_seconds=1.5),
    )

    assert [block.text for block in blocks] == ["x" * 500, "next"]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            {"max_seconds": 30, "max_characters": 100, "overlap_seconds": 30},
            "overlap",
        ),
        (
            {"max_seconds": 30, "max_characters": 100, "overlap_seconds": -1},
            "greater than or equal",
        ),
    ],
)
def test_chapter_window_policy_rejects_invalid_overlap(
    policy: dict[str, float | int], message: str
) -> None:
    """A window policy must always permit forward progress."""
    with pytest.raises(ValueError, match=message):
        ChapterWindowPolicy(**policy)


def test_chapter_candidate_requires_identifying_and_display_text() -> None:
    """Candidates need stable provenance before global consolidation."""
    candidate = ChapterCandidate(
        candidate_id=" window-0-candidate-1 ",
        window_index=0,
        start_sec=15,
        title=" 핵심 주장 ",
    )

    assert candidate.candidate_id == "window-0-candidate-1"
    assert candidate.title == "핵심 주장"
