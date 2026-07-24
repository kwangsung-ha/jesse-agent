"""Unit tests for YouTubeLoader (external APIs are mocked)."""

import json
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tubetalk.pipeline.loader import YouTubeLoader

# ------------------------------------------------------------------
# extract_video_id
# ------------------------------------------------------------------


class TestExtractVideoId:
    """Tests for YouTubeLoader.extract_video_id."""

    @pytest.mark.parametrize(
        "url, expected_id",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120",
                "dQw4w9WgXcQ",
            ),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=30", "dQw4w9WgXcQ"),
            (
                "https://www.youtube.com/shorts/dQw4w9WgXcQ",
                "dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/embed/dQw4w9WgXcQ",
                "dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/v/dQw4w9WgXcQ",
                "dQw4w9WgXcQ",
            ),
        ],
    )
    def test_valid_urls(self, url: str, expected_id: str) -> None:
        """Various valid YouTube URL formats should yield the correct id."""
        assert YouTubeLoader.extract_video_id(url) == expected_id

    def test_invalid_url_raises(self) -> None:
        """Non-YouTube URLs should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot extract video_id"):
            YouTubeLoader.extract_video_id("https://example.com/page")

    def test_empty_string_raises(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            YouTubeLoader.extract_video_id("")


# ------------------------------------------------------------------
# fetch_transcript (mocked)
# ------------------------------------------------------------------


def _make_transcript_segments() -> list[Any]:
    """Return fake transcript segment objects with start/duration/text."""
    return [
        SimpleNamespace(start=0.0, duration=3.5, text="Hello world"),
        SimpleNamespace(start=3.5, duration=4.0, text="안녕하세요"),
        SimpleNamespace(start=7.5, duration=2.0, text="End"),
    ]


class TestFetchTranscript:
    """Tests for YouTubeLoader.fetch_transcript."""

    def test_returns_formatted_segments(self, mocker: Any) -> None:
        """fetch_transcript should return validated transcript segments."""
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript_segments()
        mocker.patch(
            "tubetalk.pipeline.loader.YouTubeTranscriptApi",
            return_value=mock_api,
        )

        result = YouTubeLoader.fetch_transcript("abc123")

        assert len(result) == 3
        assert result.segments[0].start_sec == 0.0
        assert result.segments[0].duration_sec == 3.5
        assert result.segments[0].text == "Hello world"
        assert result.segments[1].text == "안녕하세요"
        mock_api.fetch.assert_called_once_with("abc123", languages=["ko", "en"])

    def test_custom_languages(self, mocker: Any) -> None:
        """fetch_transcript should forward the languages argument."""
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript_segments()
        mocker.patch(
            "tubetalk.pipeline.loader.YouTubeTranscriptApi",
            return_value=mock_api,
        )

        YouTubeLoader.fetch_transcript("abc123", languages=["ja"])
        mock_api.fetch.assert_called_once_with("abc123", languages=["ja"])


# ------------------------------------------------------------------
# fetch_metadata (mocked)
# ------------------------------------------------------------------

_FAKE_YTDLP_JSON: dict[str, Any] = {
    "title": "Test Video Title",
    "channel": "TestChannel",
    "duration": 300,
    "upload_date": "20240101",
    "view_count": 12345,
    "thumbnail": "https://img.youtube.com/vi/abc/default.jpg",
    "extra_field": "ignored",
}


class TestFetchMetadata:
    """Tests for YouTubeLoader.fetch_metadata."""

    def test_returns_expected_keys(self, mocker: Any) -> None:
        """fetch_metadata should return typed metadata."""
        mocker.patch(
            "tubetalk.pipeline.loader.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(_FAKE_YTDLP_JSON),
                stderr="",
            ),
        )

        result = YouTubeLoader.fetch_metadata(
            "abc123", "https://www.youtube.com/watch?v=abc123"
        )

        assert result.video_id == "abc123"
        assert result.title == "Test Video Title"
        assert result.channel == "TestChannel"
        assert result.duration_sec == 300
        assert result.upload_date == "20240101"
        assert result.view_count == 12345
        assert result.thumbnail_url == "https://img.youtube.com/vi/abc/default.jpg"

    def test_calls_ytdlp_correctly(self, mocker: Any) -> None:
        """fetch_metadata should invoke yt-dlp with the right arguments."""
        mock_run = mocker.patch(
            "tubetalk.pipeline.loader.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(_FAKE_YTDLP_JSON),
                stderr="",
            ),
        )

        url = "https://youtu.be/xyz789"
        YouTubeLoader.fetch_metadata("xyz789", url)

        mock_run.assert_called_once_with(
            ["yt-dlp", "--dump-json", "--no-download", "--no-warnings", url],
            capture_output=True,
            text=True,
            check=True,
        )
