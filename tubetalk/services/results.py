"""Typed outcomes shared by video-processing stages and their facade."""

from pydantic import BaseModel, ConfigDict, Field

from tubetalk.domain.state import SyncState
from tubetalk.domain.summary import VideoSummary


class IndexingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: SyncState
    chunk_count: int | None = None
    warning: str | None = None


class SummaryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: SyncState
    summary: VideoSummary | None = None
    warning: str | None = None


class VisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: SyncState
    scene_count: int | None = None
    warning: str | None = None
    indexing: IndexingResult = Field(
        default_factory=lambda: IndexingResult(state=SyncState.MISSING)
    )


class ProcessTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    ingestion_sec: float = 0.0
    transcript_index_sec: float = 0.0
    summary_sec: float = 0.0
    vision_sec: float = 0.0
    total_sec: float = 0.0


class ProcessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    cache_hit: bool
    transcript_segments: int
    indexing: IndexingResult
    summary: SummaryResult
    vision: VisionResult = Field(
        default_factory=lambda: VisionResult(state=SyncState.MISSING)
    )
    timing: ProcessTiming = Field(default_factory=ProcessTiming)
