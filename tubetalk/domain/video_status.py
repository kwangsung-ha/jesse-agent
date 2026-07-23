"""Typed local-video status shared by cache and application services."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoStatus:
    """UI-independent status for one locally known video."""

    video_id: str
    title: Optional[str]
    channel: Optional[str]
    duration: Optional[float]
    has_metadata: bool
    has_transcript: bool
    has_vision_index: bool
    transcript_segments: int
    transcript_index_state: str
    transcript_index_chunks: Optional[int]
    transcript_index_model: Optional[str]
    transcript_index_dimension: Optional[int]
    transcript_indexed_at: Optional[str]
    summary_state: str
    summary_chapters: Optional[int]
    summary_model: Optional[str]
    summary_prompt_version: Optional[str]
    summary_language: Optional[str]
    summary_generated_at: Optional[str]
    cached_at: Optional[str]
