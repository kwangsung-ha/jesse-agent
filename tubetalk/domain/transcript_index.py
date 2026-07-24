"""Transcript chunking and manifest models independent of vector backends."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tubetalk.core.config import settings
from tubetalk.domain.transcript import Transcript

INDEX_SCHEMA_VERSION = 1
CHUNK_POLICY_VERSION = "45s-1200chars-v1"


@dataclass(frozen=True)
class TranscriptChunk:
    """A retrieval-sized, timestamped group of transcript segments."""

    index: int
    text: str
    start_sec: float
    end_sec: float
    first_segment_index: int
    last_segment_index: int


@dataclass(frozen=True)
class IndexManifest:
    """Records the inputs and settings used to build a vector index."""

    schema_version: int
    transcript_sha256: str
    embedding_model: str
    embedding_dimension: int
    chunk_policy_version: str
    chunk_count: int
    indexed_at: datetime

    def __post_init__(self) -> None:
        if isinstance(self.indexed_at, str):
            object.__setattr__(
                self, "indexed_at", datetime.fromisoformat(self.indexed_at)
            )


def chunk_transcript(
    transcript: Transcript,
    max_seconds: float = settings.transcript_chunk_max_seconds,
    max_characters: int = settings.transcript_chunk_max_characters,
) -> list[TranscriptChunk]:
    """Merge consecutive transcript segments into bounded retrieval chunks."""
    if max_seconds <= 0 or max_characters <= 0:
        raise ValueError("Transcript chunk limits must be positive")
    chunks: list[TranscriptChunk] = []
    chunk_texts: list[str] = []
    chunk_start: Optional[float] = None
    chunk_end: Optional[float] = None
    first_index: Optional[int] = None
    previous_start: Optional[float] = None

    def emit(last_index: int) -> None:
        if chunk_start is None or chunk_end is None or first_index is None:
            return
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(chunk_texts),
                start_sec=chunk_start,
                end_sec=chunk_end,
                first_segment_index=first_index,
                last_segment_index=last_index,
            )
        )

    for segment_index, segment in enumerate(transcript.segments):
        text, start_sec, end_sec = (
            segment.text.strip(),
            segment.start_sec,
            segment.end_sec,
        )
        if previous_start is not None and start_sec < previous_start:
            raise ValueError("Transcript segments must be ordered by start_sec")
        previous_start = start_sec
        candidate_characters = len(" ".join([*chunk_texts, text]))
        candidate_duration = end_sec - (
            chunk_start if chunk_start is not None else start_sec
        )
        if chunk_texts and (
            candidate_characters > max_characters or candidate_duration > max_seconds
        ):
            emit(segment_index - 1)
            chunk_texts = []
            chunk_start = None
            chunk_end = None
            first_index = None
        if not chunk_texts:
            chunk_start = start_sec
            first_index = segment_index
        chunk_texts.append(text)
        chunk_end = end_sec
    if chunk_texts:
        emit(len(transcript) - 1)
    return chunks


def format_document(text: str, title: str) -> str:
    """Format a text chunk for document embedding."""
    return f"title: {title} | text: {text}"


def transcript_sha256(transcript: Transcript) -> str:
    """Return a stable digest used to detect transcript changes."""
    serialized = json.dumps(
        [
            {
                "start_sec": segment.start_sec,
                "duration_sec": segment.duration_sec,
                "text": segment.text,
            }
            for segment in transcript.segments
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
