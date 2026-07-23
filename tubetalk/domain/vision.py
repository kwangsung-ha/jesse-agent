"""Domain models for timestamped visual scene indexes."""

from dataclasses import dataclass
from typing import Literal

VISION_SCHEMA_VERSION = 1


class VisionSource:
    """Marker base class for provider-independent visual-analysis inputs."""


@dataclass(frozen=True)
class YouTubeUrlVisionSource(VisionSource):
    """A public YouTube URL that Gemini can inspect directly."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://www.youtube.com/", "https://youtu.be/")):
            raise ValueError("Vision source must be a public YouTube URL")


@dataclass(frozen=True)
class VisionScene:
    """One chronologically bounded, visually grounded video scene."""

    start_sec: float
    end_sec: float
    visual_summary: str
    detected_objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("Vision scene timestamps must be non-negative and ordered")
        if not self.visual_summary.strip():
            raise ValueError("Vision scene summary must not be empty")
        if any(not item.strip() for item in self.detected_objects):
            raise ValueError("Detected objects must not contain empty text")


@dataclass(frozen=True)
class VisionManifest:
    """Inputs and settings used to generate a visual scene index."""

    schema_version: int
    source_url: str
    model: str
    prompt_version: str
    generated_at: str


@dataclass(frozen=True)
class VisionIndexEntry:
    """A scene index together with the provenance needed to reuse it safely."""

    scenes: tuple[VisionScene, ...]
    manifest: VisionManifest


@dataclass(frozen=True)
class VisionIndexStatus:
    """Validity of a cached visual scene index for the requested settings."""

    state: Literal["missing", "current", "stale", "invalid"]
    entry: VisionIndexEntry | None = None
