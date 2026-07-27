"""Domain primitives for coverage-first transcript chapter extraction."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tubetalk.domain.transcript import Transcript, TranscriptSegment


class ChapterWindowPolicy(BaseModel):
    """Limits and overlap used to preserve topic transitions between requests."""

    model_config = ConfigDict(frozen=True)

    max_seconds: float = Field(default=480.0, gt=0)
    max_characters: int = Field(default=12000, gt=0)
    overlap_seconds: float = Field(default=30.0, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChapterWindowPolicy":
        if self.overlap_seconds >= self.max_seconds:
            raise ValueError("Chapter window overlap must be shorter than its duration")
        return self

    @property
    def cache_key(self) -> str:
        """Return a stable cache-freshness key for this extraction policy."""
        return (
            f"{self.max_seconds:g}s-{self.max_characters}chars-"
            f"{self.overlap_seconds:g}s-v1"
        )


class ChapterBlockPolicy(BaseModel):
    """Rules for turning caption fragments into readable prompt blocks."""

    model_config = ConfigDict(frozen=True)

    max_seconds: float = Field(default=20.0, gt=0)
    max_characters: int = Field(default=400, gt=0)
    max_gap_seconds: float = Field(default=1.5, ge=0)

    @property
    def cache_key(self) -> str:
        """Return a stable cache-freshness key for this prompt-shaping policy."""
        return (
            f"{self.max_seconds:g}s-{self.max_characters}chars-"
            f"{self.max_gap_seconds:g}s-gap-v1"
        )


class TranscriptBlock(BaseModel):
    """A readable, timestamped group of adjacent source caption fragments."""

    model_config = ConfigDict(frozen=True)

    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Transcript block text must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_interval(self) -> "TranscriptBlock":
        if self.end_sec < self.start_sec:
            raise ValueError("Transcript block end_sec must not precede start_sec")
        return self


class TranscriptWindow(BaseModel):
    """One overlapping, bounded transcript interval sent to candidate extraction."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    segments: tuple[TranscriptSegment, ...]
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    first_segment_index: int = Field(ge=0)
    last_segment_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "TranscriptWindow":
        if not self.segments:
            raise ValueError("Transcript windows must contain at least one segment")
        if self.end_sec < self.start_sec:
            raise ValueError("Transcript window end_sec must not precede start_sec")
        if self.last_segment_index < self.first_segment_index:
            raise ValueError("Transcript window segment indexes must be ordered")
        return self


class ChapterCandidate(BaseModel):
    """A locally extracted semantic transition awaiting global consolidation."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    window_index: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    title: str

    @field_validator("candidate_id", "title")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chapter candidate text must not be empty")
        return value.strip()


def block_transcript_segments(
    segments: tuple[TranscriptSegment, ...],
    policy: ChapterBlockPolicy | None = None,
) -> list[TranscriptBlock]:
    """Merge caption fragments into readable blocks without losing their interval."""
    if not segments:
        return []

    policy = policy or ChapterBlockPolicy()
    blocks: list[TranscriptBlock] = []
    current: list[TranscriptSegment] = []

    def emit() -> None:
        if not current:
            return
        blocks.append(
            TranscriptBlock(
                start_sec=current[0].start_sec,
                end_sec=current[-1].end_sec,
                text=" ".join(segment.text for segment in current),
            )
        )

    for segment in segments:
        if current:
            previous = current[-1]
            candidate_text = " ".join([*(item.text for item in current), segment.text])
            candidate_duration = segment.end_sec - current[0].start_sec
            should_break = (
                segment.start_sec - previous.end_sec > policy.max_gap_seconds
                or _ends_sentence(previous.text)
                or candidate_duration > policy.max_seconds
                or len(candidate_text) > policy.max_characters
            )
            if should_break:
                emit()
                current = []
        current.append(segment)
    emit()
    return blocks


def window_transcript(
    transcript: Transcript,
    policy: ChapterWindowPolicy | None = None,
) -> list[TranscriptWindow]:
    """Split a transcript into bounded windows with deterministic time overlap.

    A source segment that by itself exceeds a limit remains in one window so every
    part of the transcript is still submitted for candidate extraction.
    """
    if not transcript:
        return []

    policy = policy or ChapterWindowPolicy()

    segments = transcript.segments
    windows: list[TranscriptWindow] = []
    first_index = 0

    while first_index < len(segments):
        start_sec = segments[first_index].start_sec
        character_count = 0
        last_index = first_index - 1

        for segment_index in range(first_index, len(segments)):
            segment = segments[segment_index]
            candidate_characters = character_count + len(segment.text)
            candidate_duration = segment.end_sec - start_sec
            if last_index >= first_index and (
                candidate_characters > policy.max_characters
                or candidate_duration > policy.max_seconds
            ):
                break
            character_count = candidate_characters
            last_index = segment_index

        window_segments = segments[first_index : last_index + 1]
        end_sec = window_segments[-1].end_sec
        windows.append(
            TranscriptWindow(
                index=len(windows),
                segments=window_segments,
                start_sec=start_sec,
                end_sec=end_sec,
                first_segment_index=first_index,
                last_segment_index=last_index,
            )
        )
        if last_index == len(segments) - 1:
            break

        overlap_start = end_sec - policy.overlap_seconds
        next_first_index = first_index + 1
        while (
            next_first_index <= last_index
            and segments[next_first_index].end_sec <= overlap_start
        ):
            next_first_index += 1
        first_index = next_first_index

    return windows


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？"))
