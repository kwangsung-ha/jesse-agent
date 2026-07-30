"""Pydantic domain models for cached YouTube-video resources."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from jesseagent.domain.transcript import Transcript


class VideoMetadata(BaseModel):
    """The stable metadata collected for a YouTube video."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_url: str
    title: str | None = None
    channel: str | None = None
    duration_sec: float | None = None
    upload_date: str | None = None
    view_count: int | None = None
    thumbnail_url: str | None = None
    processed_at: datetime | None = None

    @field_validator("video_id", "source_url")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Video identity fields must not be empty")
        return value

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("Video duration must be non-negative")
        return value

    @field_validator("view_count")
    @classmethod
    def validate_view_count(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Video view count must be non-negative")
        return value


class CachedVideo(BaseModel):
    """The typed resources that constitute a reusable video cache."""

    model_config = ConfigDict(frozen=True)

    metadata: VideoMetadata
    transcript: Transcript
