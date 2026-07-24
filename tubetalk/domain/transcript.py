"""Pydantic domain models for timestamped YouTube transcripts."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TranscriptSegment(BaseModel):
    """One non-empty piece of timed transcript text."""

    model_config = ConfigDict(frozen=True)

    start_sec: float
    duration_sec: float = 0.0
    text: str

    @field_validator("start_sec", "duration_sec")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("Transcript timestamps must be non-negative and finite")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Transcript text must not be empty")
        return value.strip()

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


class Transcript(BaseModel):
    """Chronologically ordered transcript segments for one video."""

    model_config = ConfigDict(frozen=True)

    segments: tuple[TranscriptSegment, ...]

    @model_validator(mode="after")
    def validate_order(self) -> "Transcript":
        if any(
            current.start_sec < previous.start_sec
            for previous, current in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("Transcript segments must be ordered by start_sec")
        return self

    def __len__(self) -> int:
        return len(self.segments)
