"""Domain models for transcript-grounded video summaries."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from jesseagent.domain.chaptering import ChapterBlockPolicy, ChapterWindowPolicy
from jesseagent.domain.state import CacheState

SUMMARY_SCHEMA_VERSION = 1


class Chapter(BaseModel):
    """One timestamped entry in a generated video table of contents."""

    model_config = ConfigDict(frozen=True)

    start_sec: float
    title: str

    @field_validator("start_sec")
    @classmethod
    def validate_start_sec(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Chapter start_sec must be non-negative")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chapter title must not be empty")
        return value.strip()


class VideoSummary(BaseModel):
    """A concise summary and chronological transcript-based chapters."""

    model_config = ConfigDict(frozen=True)

    text: str
    chapters: tuple[Chapter, ...]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Summary text must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_chapter_order(self) -> "VideoSummary":
        if any(
            current.start_sec < previous.start_sec
            for previous, current in zip(self.chapters, self.chapters[1:])
        ):
            raise ValueError("Chapters must be ordered by start_sec")
        return self


class SummaryManifest(BaseModel):
    """Inputs and settings used to generate a cached video summary."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    transcript_sha256: str
    model: str
    prompt_version: str
    language: str
    chapter_window_policy: str = ChapterWindowPolicy().cache_key
    chapter_block_policy: str = ChapterBlockPolicy().cache_key
    generated_at: datetime


class SummaryCacheEntry(BaseModel):
    """A summary together with the manifest that establishes its freshness."""

    model_config = ConfigDict(frozen=True)

    summary: VideoSummary
    manifest: SummaryManifest


class SummaryCacheStatus(BaseModel):
    """The validity of a cached summary for the requested generation inputs."""

    model_config = ConfigDict(frozen=True)

    state: CacheState
    entry: SummaryCacheEntry | None = None
