"""Typed local-video status shared by cache and application services."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from tubetalk.domain.state import CacheState


class VectorIndexStatus(BaseModel):
    """Presentation-neutral status for one vector index."""

    state: CacheState
    item_count: Optional[int]
    embedding_model: Optional[str]
    embedding_dimension: Optional[int]
    indexed_at: Optional[datetime]


class SummaryStatus(BaseModel):
    """Presentation-neutral status and provenance for a transcript summary."""

    state: CacheState
    chapter_count: Optional[int]
    model: Optional[str]
    prompt_version: Optional[str]
    language: Optional[str]
    generated_at: Optional[datetime]


class VisionStatus(BaseModel):
    """Presentation-neutral status and provenance for visual scenes."""

    state: CacheState
    scene_count: Optional[int]
    model: Optional[str]
    prompt_version: Optional[str]
    generated_at: Optional[datetime]


class VideoStatusDetails(BaseModel):
    """Nested status view used by interfaces instead of flat field groups."""

    transcript_segment_count: int
    transcript_index: VectorIndexStatus
    summary: SummaryStatus
    vision: VisionStatus
    vision_vector_index: VectorIndexStatus


class VideoStatus(BaseModel):
    """UI-independent status for one locally known video."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    title: Optional[str]
    channel: Optional[str]
    duration: Optional[float]
    has_metadata: bool
    has_transcript: bool
    has_vision_index: bool
    transcript_segments: int
    transcript_index_state: CacheState
    transcript_index_chunks: Optional[int]
    transcript_index_model: Optional[str]
    transcript_index_dimension: Optional[int]
    transcript_indexed_at: Optional[datetime]
    summary_state: CacheState
    summary_chapters: Optional[int]
    summary_model: Optional[str]
    summary_prompt_version: Optional[str]
    summary_language: Optional[str]
    summary_generated_at: Optional[datetime]
    cached_at: Optional[str]
    vision_index_state: CacheState = CacheState.MISSING
    vision_scene_count: Optional[int] = None
    vision_model: Optional[str] = None
    vision_prompt_version: Optional[str] = None
    vision_generated_at: Optional[datetime] = None
    vision_vector_index_state: CacheState = CacheState.MISSING
    vision_vector_index_scenes: Optional[int] = None
    vision_vector_index_model: Optional[str] = None
    vision_vector_index_dimension: Optional[int] = None
    vision_vector_indexed_at: Optional[datetime] = None

    @property
    def details(self) -> VideoStatusDetails:
        """Return the grouped status model consumed by interface adapters."""
        return VideoStatusDetails(
            transcript_segment_count=self.transcript_segments,
            transcript_index=VectorIndexStatus(
                state=self.transcript_index_state,
                item_count=self.transcript_index_chunks,
                embedding_model=self.transcript_index_model,
                embedding_dimension=self.transcript_index_dimension,
                indexed_at=self.transcript_indexed_at,
            ),
            summary=SummaryStatus(
                state=self.summary_state,
                chapter_count=self.summary_chapters,
                model=self.summary_model,
                prompt_version=self.summary_prompt_version,
                language=self.summary_language,
                generated_at=self.summary_generated_at,
            ),
            vision=VisionStatus(
                state=self.vision_index_state,
                scene_count=self.vision_scene_count,
                model=self.vision_model,
                prompt_version=self.vision_prompt_version,
                generated_at=self.vision_generated_at,
            ),
            vision_vector_index=VectorIndexStatus(
                state=self.vision_vector_index_state,
                item_count=self.vision_vector_index_scenes,
                embedding_model=self.vision_vector_index_model,
                embedding_dimension=self.vision_vector_index_dimension,
                indexed_at=self.vision_vector_indexed_at,
            ),
        )
