"""Transcript chunking and manifest models independent of vector backends."""

import hashlib
import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from tubetalk.domain.transcript import Transcript

INDEX_SCHEMA_VERSION = 1
CHUNK_POLICY_VERSION = "45s-1200chars-v1"


class TranscriptChunkPolicy(BaseModel):
    """Application-supplied limits for retrieval-sized transcript chunks."""

    model_config = ConfigDict(frozen=True)

    max_seconds: float = Field(gt=0)
    max_characters: int = Field(gt=0)


DEFAULT_TRANSCRIPT_CHUNK_POLICY = TranscriptChunkPolicy(
    max_seconds=45.0, max_characters=1200
)


class TranscriptChunk(BaseModel):
    """A retrieval-sized, timestamped group of transcript segments."""

    model_config = ConfigDict(frozen=True)

    index: int
    text: str
    start_sec: float
    end_sec: float
    first_segment_index: int
    last_segment_index: int


class IndexManifest(BaseModel):
    """Records the inputs and settings used to build a vector index."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    transcript_sha256: str
    embedding_model: str
    embedding_dimension: int
    chunk_policy_version: str
    chunk_count: int
    indexed_at: datetime
    collection_name: str = "transcript_collection"


def chunk_transcript(
    transcript: Transcript,
    policy: TranscriptChunkPolicy = DEFAULT_TRANSCRIPT_CHUNK_POLICY,
) -> list[TranscriptChunk]:
    """Merge consecutive transcript segments into bounded retrieval chunks."""
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
            candidate_characters > policy.max_characters
            or candidate_duration > policy.max_seconds
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
