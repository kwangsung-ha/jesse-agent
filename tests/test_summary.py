"""Unit tests for transcript-grounded summary domain models."""

import pytest

from tubetalk.domain.summary import Chapter, VideoSummary


def test_video_summary_accepts_ordered_chapters() -> None:
    """Chronological chapters should form a valid summary."""
    summary = VideoSummary(
        text="영상의 핵심 내용을 요약합니다.",
        chapters=(
            Chapter(start_sec=0, title="소개"),
            Chapter(start_sec=42.5, title="핵심 설명"),
        ),
    )

    assert summary.chapters[1].start_sec == 42.5


@pytest.mark.parametrize(
    ("start_sec", "title"),
    [(-1, "소개"), (0, " ")],
)
def test_chapter_rejects_invalid_values(start_sec: float, title: str) -> None:
    """Chapters require a non-negative timestamp and a title."""
    with pytest.raises(ValueError):
        Chapter(start_sec=start_sec, title=title)


def test_video_summary_rejects_out_of_order_chapters() -> None:
    """A table of contents must remain chronological."""
    with pytest.raises(ValueError, match="ordered"):
        VideoSummary(
            text="요약입니다.",
            chapters=(
                Chapter(start_sec=30, title="나중"),
                Chapter(start_sec=0, title="처음"),
            ),
        )
