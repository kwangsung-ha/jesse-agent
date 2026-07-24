"""YouTube video loader — URL parsing, transcript & metadata fetching."""

import json
import re
import subprocess
from typing import Any, Optional

from youtube_transcript_api import YouTubeTranscriptApi

from tubetalk.domain.transcript import Transcript, TranscriptSegment
from tubetalk.domain.video import VideoMetadata

# ------------------------------------------------------------------
# URL → video_id extraction
# ------------------------------------------------------------------

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|shorts/|embed/|v/)"
    r"|youtu\.be/)"
    r"(?P<id>[A-Za-z0-9_-]{11})"
)


class YouTubeLoader:
    """Parse YouTube URLs and fetch transcript / metadata."""

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract the 11-char ``video_id`` from a YouTube URL.

        Raises:
            ValueError: If the URL does not match any known YouTube pattern.
        """
        match = _YOUTUBE_RE.search(url)
        if not match:
            raise ValueError(f"Cannot extract video_id from URL: {url}")
        return match.group("id")

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_transcript(
        video_id: str,
        languages: Optional[list[str]] = None,
    ) -> Transcript:
        """Fetch transcript segments via ``youtube-transcript-api``.

        Returns validated, chronologically ordered transcript segments.
        """
        if languages is None:
            languages = ["ko", "en"]

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=languages)

        return Transcript(
            segments=tuple(
                TranscriptSegment(
                    start_sec=round(seg.start, 3),
                    duration_sec=round(seg.duration, 3),
                    text=seg.text,
                )
                for seg in transcript
            )
        )

    # ------------------------------------------------------------------
    # Metadata (via yt-dlp)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_metadata(video_id: str, url: str) -> VideoMetadata:
        """Fetch video metadata using ``yt-dlp --dump-json``.

        Returns validated metadata for the requested video.
        """
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-warnings",
                url,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info: dict[str, Any] = json.loads(result.stdout)
        return VideoMetadata(
            video_id=video_id,
            source_url=url,
            title=_optional_text(info.get("title")),
            channel=_optional_text(info.get("channel")),
            duration_sec=_optional_number(info.get("duration")),
            upload_date=_optional_text(info.get("upload_date")),
            view_count=_optional_int(info.get("view_count")),
            thumbnail_url=_optional_text(info.get("thumbnail")),
        )


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value
