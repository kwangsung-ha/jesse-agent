"""Local cache manager for TubeTalk video data."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tubetalk.core.config import settings


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

    # ------------------------------------------------------------------
    # Listing / status helpers
    # ------------------------------------------------------------------

    def list_cached_videos(self) -> list[dict[str, Any]]:
        """Return a summary list of every cached video under ``data/``."""
        results: list[dict[str, Any]] = []
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

    def get_video_status(self, video_id: str) -> Optional[dict[str, Any]]:
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

        return {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "duration": duration,
            "has_metadata": has_metadata,
            "has_transcript": has_transcript,
            "has_vision_index": has_vision_index,
            "transcript_segments": transcript_count,
            **transcript_index,
            "cached_at": cached_at,
        }

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
            if manifest.get("transcript_sha256") != transcript_sha256:
                return {**result, "transcript_index_state": "stale"}
            if (
                manifest.get("embedding_model") != settings.embedding_model
                or manifest.get("embedding_dimension") != settings.embedding_dimension
            ):
                return {**result, "transcript_index_state": "stale"}
            return {**result, "transcript_index_state": "current"}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return {**result, "transcript_index_state": "invalid"}
