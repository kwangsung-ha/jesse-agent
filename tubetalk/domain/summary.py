"""Domain models for transcript-grounded video summaries."""

from dataclasses import dataclass
from datetime import datetime

from tubetalk.domain.state import CacheState

SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Chapter:
    """One timestamped entry in a generated video table of contents."""

    start_sec: float
    title: str

    def __post_init__(self) -> None:
        if self.start_sec < 0:
            raise ValueError("Chapter start_sec must be non-negative")
        if not self.title.strip():
            raise ValueError("Chapter title must not be empty")


@dataclass(frozen=True)
class VideoSummary:
    """A concise summary and chronological transcript-based chapters."""

    text: str
    chapters: tuple[Chapter, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Summary text must not be empty")
        if any(
            current.start_sec < previous.start_sec
            for previous, current in zip(self.chapters, self.chapters[1:])
        ):
            raise ValueError("Chapters must be ordered by start_sec")


@dataclass(frozen=True)
class SummaryManifest:
    """Inputs and settings used to generate a cached video summary."""

    schema_version: int
    transcript_sha256: str
    model: str
    prompt_version: str
    language: str
    generated_at: datetime

    def __post_init__(self) -> None:
        if isinstance(self.generated_at, str):
            object.__setattr__(
                self, "generated_at", datetime.fromisoformat(self.generated_at)
            )


@dataclass(frozen=True)
class SummaryCacheEntry:
    """A summary together with the manifest that establishes its freshness."""

    summary: VideoSummary
    manifest: SummaryManifest


@dataclass(frozen=True)
class SummaryCacheStatus:
    """The validity of a cached summary for the requested generation inputs."""

    state: CacheState
    entry: SummaryCacheEntry | None = None
