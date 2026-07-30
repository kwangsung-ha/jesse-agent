"""Domain models for timestamped visual scene indexes."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from jesseagent.domain.state import CacheState

VISION_SCHEMA_VERSION = 1


class VisionSource(BaseModel):
    """Marker base class for provider-independent visual-analysis inputs."""


class YouTubeUrlVisionSource(VisionSource):
    """A public YouTube URL that Gemini can inspect directly."""

    model_config = ConfigDict(frozen=True)

    url: str

    def __init__(self, url: str | None = None, **data: object) -> None:
        """Accept the legacy single positional URL while retaining BaseModel data."""
        if url is not None:
            data["url"] = url
        super().__init__(**data)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value.startswith(("https://www.youtube.com/", "https://youtu.be/")):
            raise ValueError("Vision source must be a public YouTube URL")
        return value


class VisionScene(BaseModel):
    """One chronologically bounded, visually grounded video scene."""

    model_config = ConfigDict(frozen=True)

    start_sec: float
    end_sec: float
    visual_summary: str
    detected_objects: tuple[str, ...]

    def __init__(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        visual_summary: str | None = None,
        detected_objects: tuple[str, ...] | None = None,
        **data: object,
    ) -> None:
        """Keep the prior positional constructor compatible during migration."""
        values = {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "visual_summary": visual_summary,
            "detected_objects": detected_objects,
        }
        data.update({key: value for key, value in values.items() if value is not None})
        super().__init__(**data)

    @model_validator(mode="after")
    def validate_scene(self) -> "VisionScene":
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("Vision scene timestamps must be non-negative and ordered")
        if not self.visual_summary.strip():
            raise ValueError("Vision scene summary must not be empty")
        if any(not item.strip() for item in self.detected_objects):
            raise ValueError("Detected objects must not contain empty text")
        return self


class VisionManifest(BaseModel):
    """Inputs and settings used to generate a visual scene index."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    source_url: str
    model: str
    prompt_version: str
    generated_at: datetime


class VisionIndexEntry(BaseModel):
    """A scene index together with the provenance needed to reuse it safely."""

    model_config = ConfigDict(frozen=True)

    scenes: tuple[VisionScene, ...]
    manifest: VisionManifest


class VisionIndexStatus(BaseModel):
    """Validity of a cached visual scene index for the requested settings."""

    model_config = ConfigDict(frozen=True)

    state: CacheState
    entry: VisionIndexEntry | None = None
