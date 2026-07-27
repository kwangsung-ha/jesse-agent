"""Local cache manager for TubeTalk video data."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from tubetalk.core.logging import logger
from tubetalk.domain.chaptering import ChapterBlockPolicy, ChapterWindowPolicy
from tubetalk.domain.state import CacheState
from tubetalk.domain.summary import (
    SUMMARY_SCHEMA_VERSION,
    Chapter,
    SummaryCacheEntry,
    SummaryCacheStatus,
    SummaryManifest,
    VideoSummary,
)
from tubetalk.domain.transcript import Transcript
from tubetalk.domain.transcript_index import transcript_sha256
from tubetalk.domain.video import CachedVideo, VideoMetadata
from tubetalk.domain.video_status import VideoStatus
from tubetalk.domain.vision import (
    VISION_SCHEMA_VERSION,
    VisionIndexEntry,
    VisionIndexStatus,
    VisionManifest,
    VisionScene,
)


class CacheFreshnessPolicy(BaseModel):
    """Expected derived-artifact settings used for cache status checks."""

    model_config = ConfigDict(frozen=True)

    summary_model: str = "gemini-3.5-flash-lite"
    summary_prompt_version: str = "summary-chapters-v2"
    summary_language: str = "ko"
    summary_chapter_window_policy: str = ChapterWindowPolicy().cache_key
    summary_chapter_block_policy: str = ChapterBlockPolicy().cache_key
    vision_model: str = "gemini-3.5-flash"
    vision_prompt_version: str = "vision-scenes-v2-30s"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768


class LocalCacheManager:
    """Manages local JSON file cache under ``data/{video_id}/``."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        freshness_policy: CacheFreshnessPolicy = CacheFreshnessPolicy(),
    ) -> None:
        self._data_dir = data_dir or Path("./data")
        self._freshness_policy = freshness_policy

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
        exists = (video_dir / "metadata.json").is_file() and (
            video_dir / "transcript.json"
        ).is_file()
        logger.bind(event="cache.check", video_id=video_id).debug("hit={}", exists)
        return exists

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def save_json(self, video_id: str, filename: str, data: Any) -> None:
        """Serialize *data* to ``data/{video_id}/{filename}``."""
        video_dir = self.get_video_dir(video_id)
        filepath = video_dir / filename
        _atomic_write_json(filepath, data)
        logger.bind(event="cache.save", video_id=video_id).debug("file={}", filename)

    def load_json(self, video_id: str, filename: str) -> Any:
        """Deserialize and return content of ``data/{video_id}/{filename}``."""
        filepath = self._data_dir / video_id / filename
        data = json.loads(filepath.read_text())
        logger.bind(event="cache.load", video_id=video_id).debug("file={}", filename)
        return data

    def save_video(self, video: CachedVideo) -> None:
        """Persist the typed resources required for a reusable video cache."""
        self.save_json(
            video.metadata.video_id,
            "metadata.json",
            video.metadata.model_dump(mode="json"),
        )
        self.save_json(
            video.metadata.video_id,
            "transcript.json",
            video.transcript.model_dump(mode="json")["segments"],
        )

    def load_video(self, video_id: str) -> CachedVideo:
        """Load and validate a complete video cache as domain objects."""
        return CachedVideo(
            metadata=VideoMetadata.model_validate(
                self.load_json(video_id, "metadata.json")
            ),
            transcript=Transcript.model_validate(
                {"segments": self.load_json(video_id, "transcript.json")}
            ),
        )

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
                "chapter_window_policy": entry.manifest.chapter_window_policy,
                "chapter_block_policy": entry.manifest.chapter_block_policy,
                "generated_at": entry.manifest.generated_at.isoformat(),
            },
        )

    def get_summary_status(
        self,
        video_id: str,
        transcript: Transcript,
        *,
        model: str,
        prompt_version: str,
        language: str,
        chapter_window_policy: str | None = None,
        chapter_block_policy: str | None = None,
    ) -> SummaryCacheStatus:
        """Return whether a cached summary matches its transcript and settings."""
        summary_path = self._data_dir / video_id / "summary.json"
        if not summary_path.is_file():
            logger.bind(event="cache.summary.status", video_id=video_id).debug(
                "state=missing"
            )
            return SummaryCacheStatus(state=CacheState.MISSING)
        try:
            entry = _summary_cache_entry(json.loads(summary_path.read_text()))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            status = SummaryCacheStatus(state=CacheState.INVALID)
            logger.bind(event="cache.summary.status", video_id=video_id).debug(
                "state=invalid"
            )
            return status
        manifest = entry.manifest
        if (
            manifest.transcript_sha256 != transcript_sha256(transcript)
            or manifest.model != model
            or manifest.prompt_version != prompt_version
            or manifest.language != language
            or manifest.chapter_window_policy
            != (chapter_window_policy or ChapterWindowPolicy().cache_key)
            or manifest.chapter_block_policy
            != (chapter_block_policy or ChapterBlockPolicy().cache_key)
        ):
            state = CacheState.STALE
        else:
            state = CacheState.CURRENT
        logger.bind(event="cache.summary.status", video_id=video_id).debug(
            "state={}", state
        )
        return SummaryCacheStatus(state=state, entry=entry)

    def save_vision_index(self, video_id: str, entry: VisionIndexEntry) -> None:
        """Persist generated visual scenes and their generation provenance."""
        self.save_json(
            video_id,
            "vision_index.json",
            {
                "scenes": [
                    {
                        "start_sec": scene.start_sec,
                        "end_sec": scene.end_sec,
                        "visual_summary": scene.visual_summary,
                        "detected_objects": list(scene.detected_objects),
                    }
                    for scene in entry.scenes
                ],
                "schema_version": entry.manifest.schema_version,
                "source_url": entry.manifest.source_url,
                "model": entry.manifest.model,
                "prompt_version": entry.manifest.prompt_version,
                "generated_at": entry.manifest.generated_at.isoformat(),
            },
        )

    def get_vision_index_status(
        self,
        video_id: str,
        *,
        source_url: str,
        model: str,
        prompt_version: str,
    ) -> VisionIndexStatus:
        """Return whether a visual index matches its source and settings."""
        vision_path = self._data_dir / video_id / "vision_index.json"
        if not vision_path.is_file():
            logger.bind(event="cache.vision.status", video_id=video_id).debug(
                "state=missing"
            )
            return VisionIndexStatus(state=CacheState.MISSING)
        try:
            entry = _vision_index_entry(json.loads(vision_path.read_text()))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            status = VisionIndexStatus(state=CacheState.INVALID)
            logger.bind(event="cache.vision.status", video_id=video_id).debug(
                "state=invalid"
            )
            return status
        manifest = entry.manifest
        if (
            manifest.source_url != source_url
            or manifest.model != model
            or manifest.prompt_version != prompt_version
        ):
            state = CacheState.STALE
        else:
            state = CacheState.CURRENT
        logger.bind(event="cache.vision.status", video_id=video_id).debug(
            "state={}", state
        )
        return VisionIndexStatus(state=state, entry=entry)

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
        metadata: dict[str, Any] = {}
        if has_metadata:
            try:
                loaded_metadata = json.loads((video_dir / "metadata.json").read_text())
                if isinstance(loaded_metadata, dict):
                    metadata = loaded_metadata
                title = metadata.get("title")
                duration = metadata.get("duration")
                channel = metadata.get("channel")
            except (json.JSONDecodeError, OSError):
                pass

        vision_status = VisionIndexStatus(state=CacheState.MISSING)
        source_url = metadata.get("source_url")
        if has_vision_index:
            if isinstance(source_url, str) and source_url:
                vision_status = self.get_vision_index_status(
                    video_id,
                    source_url=source_url,
                    model=self._freshness_policy.vision_model,
                    prompt_version=self._freshness_policy.vision_prompt_version,
                )
            else:
                vision_status = VisionIndexStatus(state=CacheState.INVALID)
        vision_entry = vision_status.entry

        transcript_count = 0
        transcript = Transcript(segments=())
        if has_transcript:
            try:
                loaded_transcript = json.loads(
                    (video_dir / "transcript.json").read_text()
                )
                transcript = Transcript.model_validate({"segments": loaded_transcript})
                transcript_count = len(transcript)
            except (json.JSONDecodeError, OSError):
                pass

        summary_status = self.get_summary_status(
            video_id,
            transcript,
            model=self._freshness_policy.summary_model,
            prompt_version=self._freshness_policy.summary_prompt_version,
            language=self._freshness_policy.summary_language,
            chapter_window_policy=(
                self._freshness_policy.summary_chapter_window_policy
            ),
            chapter_block_policy=self._freshness_policy.summary_chapter_block_policy,
        )
        summary_entry = summary_status.entry

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
            transcript_index_state=CacheState(
                str(transcript_index["transcript_index_state"])
            ),
            transcript_index_chunks=_optional_int(
                transcript_index["transcript_index_chunks"]
            ),
            transcript_index_model=_optional_string(
                transcript_index["transcript_index_model"]
            ),
            transcript_index_dimension=_optional_int(
                transcript_index["transcript_index_dimension"]
            ),
            transcript_indexed_at=_optional_datetime(
                transcript_index["transcript_indexed_at"]
            ),
            summary_state=summary_status.state,
            summary_chapters=(
                len(summary_entry.summary.chapters)
                if summary_entry is not None
                else None
            ),
            summary_model=(
                summary_entry.manifest.model if summary_entry is not None else None
            ),
            summary_prompt_version=(
                summary_entry.manifest.prompt_version
                if summary_entry is not None
                else None
            ),
            summary_language=(
                summary_entry.manifest.language if summary_entry is not None else None
            ),
            summary_generated_at=(
                summary_entry.manifest.generated_at
                if summary_entry is not None
                else None
            ),
            cached_at=cached_at,
            vision_index_state=vision_status.state,
            vision_scene_count=(len(vision_entry.scenes) if vision_entry else None),
            vision_model=(vision_entry.manifest.model if vision_entry else None),
            vision_prompt_version=(
                vision_entry.manifest.prompt_version if vision_entry else None
            ),
            vision_generated_at=(
                vision_entry.manifest.generated_at if vision_entry else None
            ),
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
                or manifest.get("embedding_model")
                != self._freshness_policy.embedding_model
                or manifest.get("embedding_dimension")
                != self._freshness_policy.embedding_dimension
            ):
                return {**result, "transcript_index_state": "stale"}

            return {**result, "transcript_index_state": "current"}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return {**result, "transcript_index_state": "invalid"}


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _optional_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _atomic_write_json(path: Path, data: Any) -> None:
    """Replace a JSON cache file only after its complete temporary write."""
    serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w") as temporary_file:
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


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
        chapter_window_policy=data.get(
            "chapter_window_policy", ChapterWindowPolicy().cache_key
        ),
        chapter_block_policy=data.get(
            "chapter_block_policy", ChapterBlockPolicy().cache_key
        ),
        generated_at=datetime.fromisoformat(_required_text(data, "generated_at")),
    )
    if manifest.schema_version != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Unsupported summary cache schema version")
    return SummaryCacheEntry(
        summary=VideoSummary(text=_required_text(data, "summary"), chapters=chapters),
        manifest=manifest,
    )


def _vision_index_entry(data: Any) -> VisionIndexEntry:
    """Parse and validate the persisted ``vision_index.json`` schema."""
    if not isinstance(data, dict):
        raise ValueError("Vision index cache must be a JSON object")
    scenes_data = data.get("scenes")
    if not isinstance(scenes_data, list):
        raise ValueError("Vision scenes must be a list")
    scenes = tuple(
        VisionScene(
            start_sec=_required_number(scene, "start_sec"),
            end_sec=_required_number(scene, "end_sec"),
            visual_summary=_required_text(scene, "visual_summary"),
            detected_objects=_required_text_items(scene, "detected_objects"),
        )
        for scene in scenes_data
    )
    if any(
        current.start_sec < previous.start_sec
        for previous, current in zip(scenes, scenes[1:])
    ):
        raise ValueError("Vision scenes must be ordered by start_sec")
    manifest = VisionManifest(
        schema_version=_required_int(data, "schema_version"),
        source_url=_required_text(data, "source_url"),
        model=_required_text(data, "model"),
        prompt_version=_required_text(data, "prompt_version"),
        generated_at=datetime.fromisoformat(_required_text(data, "generated_at")),
    )
    if manifest.schema_version != VISION_SCHEMA_VERSION:
        raise ValueError("Unsupported vision cache schema version")
    return VisionIndexEntry(scenes=scenes, manifest=manifest)


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


def _required_text_items(data: Any, key: str) -> tuple[str, ...]:
    if not isinstance(data, dict):
        raise ValueError(f"{key} must belong to a JSON object")
    values = data.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"{key} must be a list of non-empty text")
    return tuple(values)


def _required_int(data: Any, key: str) -> int:
    if not isinstance(data, dict):
        raise ValueError(f"{key} must belong to a JSON object")
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value
