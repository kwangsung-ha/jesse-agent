"""Typed, timestamped evidence returned by hybrid retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalHit(BaseModel):
    """One searchable transcript chunk or visual scene."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source: Literal["transcript", "vision"]
    text: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    rank: int = Field(ge=1)
    distance: float = Field(ge=0)
    score: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "RetrievalHit":
        if not self.source_id.strip() or not self.text.strip():
            raise ValueError("Retrieval hits require a source ID and text")
        if self.end_sec < self.start_sec:
            raise ValueError("Retrieval hit timestamps must be ordered")
        return self


class Citation(BaseModel):
    """A model-selected timestamp tied to a retrieved source."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    timestamp_sec: float = Field(ge=0)


class ChatAnswer(BaseModel):
    """A grounded answer with citations that have passed validation."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[Citation, ...]


class ChatTurn(BaseModel):
    """One process-local question and its validated answer."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: ChatAnswer
