"""Local cache manager for TubeTalk video data."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tubetalk.core.config import settings
from tubetalk.domain.summary import (
    SUMMARY_SCHEMA_VERSION,
    Chapter,
    SummaryCacheEntry,
    SummaryCacheStatus,
    SummaryManifest,
    VideoSummary,
)
from tubetalk.domain.transcript_index import transcript_sha256
from tubetalk.domain.video_status import VideoStatus


class LocalCacheManager:
    """Manages local JSON file cache under ``data/{video_id}/``."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or settings.data_dir

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def get_video_dir(self, video_id: str) -> Path:
        """Return (and create) ``data/{video_id}/`` directory."""
        video_dir = self._data_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    # ------------------------------------------------------------------
    # Cache detection
    # ------------------------------------------------------------------

    def has_cache(self, video_id: str) -> bool:
        """Check whether both ``metadata.json`` and ``transcript.json`` exist."""
        video_dir = self._data_dir / video_id
        return (video_dir / "metadata.json").is_file() and (
            video_dir / "transcript.json"
        ).is_file()

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def save_json(self, video_id: str, filename: str, data: Any) -> None:
        """Serialize *data* to ``data/{video_id}/{filename}``."""
        video_dir = self.get_video_dir(video_id)
        filepath = video_dir / filename
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def load_json(self, video_id: str, filename: str) -> Any:
        """Deserialize and return content of ``data/{video_id}/{filename}``."""
        filepath = self._data_dir / video_id / filename
        return json.loads(filepath.read_text())

    def save_summary(self, video_id: str, entry: SummaryCacheEntry) -> None:
        """Persist a generated summary and the inputs used to create it."""
        self.save_json(
            video_id,
            "summary.json",
            {
                "summary": entry.summary.text,
                "chapters": [
                    {"start_sec": chapter.start_sec, "title": chapter.title}
                    for chapter in entry.summary.chapters
                ],
                "schema_version": entry.manifest.schema_version,
                "transcript_sha256": entry.manifest.transcript_sha256,
                "model": entry.manifest.model,
                "prompt_version": entry.manifest.prompt_version,
                "language": entry.manifest.language,
                "generated_at": entry.manifest.generated_at,
            },
        )

    def get_summary_status(
        self,
        video_id: str,
        transcript: list[dict[str, Any]],
        *,
        model: str,
        prompt_version: str,
        language: str,
    ) -> SummaryCacheStatus:
        """Return whether a cached summary matches its transcript and settings."""
        summary_path = self._data_dir / video_id / "summary.json"
        if not summary_path.is_file():
            return SummaryCacheStatus(state="missing")
        try:
            entry = _summary_cache_entry(json.loads(summary_path.read_text()))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return SummaryCacheStatus(state="invalid")
        manifest = entry.manifest
        if (
            manifest.transcript_sha256 != transcript_sha256(transcript)
            or manifest.model != model
            or manifest.prompt_version != prompt_version
            or manifest.language != language
        ):
            return SummaryCacheStatus(state="stale", entry=entry)
        return SummaryCacheStatus(state="current", entry=entry)

    # ------------------------------------------------------------------
    # Listing / status helpers
    # ------------------------------------------------------------------

    def list_cached_videos(self) -> list[VideoStatus]:
        """Return a summary list of every cached video under ``data/``."""
        results: list[VideoStatus] = []
        if not self._data_dir.is_dir():
            return results

        for child in sorted(self._data_dir.iterdir()):
            if not child.is_dir():
                continue
            video_id = child.name
            status = self.get_video_status(video_id)
            if status is not None:
                results.append(status)
        return results

    def get_video_status(self, video_id: str) -> Optional[VideoStatus]:
        """Return detailed cache status for *video_id*, or ``None``."""
        video_dir = self._data_dir / video_id
        if not video_dir.is_dir():
            return None

        has_metadata = (video_dir / "metadata.json").is_file()
        has_transcript = (video_dir / "transcript.json").is_file()
        has_vision_index = (video_dir / "vision_index.json").is_file()

        transcript_index = self._get_transcript_index_status(video_dir)

        title: Optional[str] = None
        duration: Optional[float] = None
        channel: Optional[str] = None
        if has_metadata:
            try:
                meta = json.loads((video_dir / "metadata.json").read_text())
                title = meta.get("title")
                duration = meta.get("duration")
                channel = meta.get("channel")
            except (json.JSONDecodeError, OSError):
                pass

        transcript_count = 0
        if has_transcript:
            try:
                transcript = json.loads((video_dir / "transcript.json").read_text())
                if isinstance(transcript, list):
                    transcript_count = len(transcript)
            except (json.JSONDecodeError, OSError):
                pass

        cached_at: Optional[str] = None
        ref_file = video_dir / "metadata.json"
        if ref_file.is_file():
            mtime = ref_file.stat().st_mtime
            cached_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        return VideoStatus(
            video_id=video_id,
            title=title if isinstance(title, str) else None,
            channel=channel if isinstance(channel, str) else None,
            duration=float(duration) if isinstance(duration, (int, float)) else None,
            has_metadata=has_metadata,
            has_transcript=has_transcript,
            has_vision_index=has_vision_index,
            transcript_segments=transcript_count,
            transcript_index_state=str(transcript_index["transcript_index_state"]),
            transcript_index_chunks=_optional_int(
                transcript_index["transcript_index_chunks"]
            ),
            transcript_index_model=_optional_string(
                transcript_index["transcript_index_model"]
            ),
            transcript_index_dimension=_optional_int(
                transcript_index["transcript_index_dimension"]
            ),
            transcript_indexed_at=_optional_string(
                transcript_index["transcript_indexed_at"]
            ),
            cached_at=cached_at,
        )

    def _get_transcript_index_status(self, video_dir: Path) -> dict[str, Any]:
        """Return transcript index metadata without opening the Chroma database."""
        manifest_path = video_dir / "index_manifest.json"
        result: dict[str, Any] = {
            "transcript_index_state": "missing",
            "transcript_index_chunks": None,
            "transcript_index_model": None,
            "transcript_index_dimension": None,
            "transcript_indexed_at": None,
        }
        if not manifest_path.is_file():
            return result

        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                return {**result, "transcript_index_state": "invalid"}
            chunks = manifest.get("chunk_count")
            result.update(
                {
                    "transcript_index_chunks": chunks
                    if isinstance(chunks, int)
                    else None,
                    "transcript_index_model": manifest.get("embedding_model"),
                    "transcript_index_dimension": manifest.get("embedding_dimension"),
                    "transcript_indexed_at": manifest.get("indexed_at"),
                }
            )
            transcript_path = video_dir / "transcript.json"
            if not transcript_path.is_file():
                return {**result, "transcript_index_state": "stale"}
            transcript = json.loads(transcript_path.read_text())
            transcript_sha256 = hashlib.sha256(
                json.dumps(
                    transcript,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if (
                manifest.get("transcript_sha256") != transcript_sha256
                or manifest.get("embedding_model") != settings.embedding_model
                or manifest.get("embedding_dimension") != settings.embedding_dimension
            ):
                return {**result, "transcript_index_state": "stale"}

            return {**result, "transcript_index_state": "current"}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return {**result, "transcript_index_state": "invalid"}


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) else None


def _summary_cache_entry(data: Any) -> SummaryCacheEntry:
    """Parse and validate the persisted ``summary.json`` schema."""
    if not isinstance(data, dict):
        raise ValueError("Summary cache must be a JSON object")
    chapters_data = data.get("chapters")
    if not isinstance(chapters_data, list):
        raise ValueError("Summary chapters must be a list")
    chapters = tuple(
        Chapter(
            start_sec=_required_number(chapter, "start_sec"),
            title=_required_text(chapter, "title"),
        )
        for chapter in chapters_data
    )
    manifest = SummaryManifest(
        schema_version=_required_int(data, "schema_version"),
        transcript_sha256=_required_text(data, "transcript_sha256"),
        model=_required_text(data, "model"),
        prompt_version=_required_text(data, "prompt_version"),
        language=_required_text(data, "language"),
        generated_at=_required_text(data, "generated_at"),
    )
    if manifest.schema_version != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Unsupported summary cache schema version")
    return SummaryCacheEntry(
        summary=VideoSummary(text=_required_text(data, "summary"), chapters=chapters),
        manifest=manifest,
    )


def _required_text(data: Any, key: str) -> str:
    if not isinstance(data, dict):
        raise ValueError(f"{key} must belong to a JSON object")
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


def _required_number(data: Any, key: str) -> float:
    if not isinstance(data, dict):
        raise ValueError(f"{key} must belong to a JSON object")
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _required_int(data: Any, key: str) -> int:
    if not isinstance(data, dict):
        raise ValueError(f"{key} must belong to a JSON object")
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value
