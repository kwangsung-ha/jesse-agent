"""YouTube video loader — URL parsing, transcript & metadata fetching."""

import json
import re
import subprocess
from typing import Any, Optional

from youtube_transcript_api import YouTubeTranscriptApi

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
    ) -> list[dict[str, Any]]:
        """Fetch transcript segments via ``youtube-transcript-api``.

        Returns a list of dicts with keys:
        ``start_sec``, ``duration_sec``, ``text``.
        """
        if languages is None:
            languages = ["ko", "en"]

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=languages)

        return [
            {
                "start_sec": round(seg.start, 3),
                "duration_sec": round(seg.duration, 3),
                "text": seg.text,
            }
            for seg in transcript
        ]

    # ------------------------------------------------------------------
    # Metadata (via yt-dlp)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_metadata(url: str) -> dict[str, Any]:
        """Fetch video metadata using ``yt-dlp --dump-json``.

        Returns a dict with keys:
        ``title``, ``channel``, ``duration``, ``upload_date``,
        ``view_count``, ``thumbnail``.
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
        return {
            "title": info.get("title"),
            "channel": info.get("channel"),
            "duration": info.get("duration"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "thumbnail": info.get("thumbnail"),
        }
