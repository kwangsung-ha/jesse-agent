"""Gemini adapter for structured transcript summaries."""

import json
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from tubetalk.domain.summary import Chapter, VideoSummary
from tubetalk.domain.transcript import Transcript
from tubetalk.ports.summary import SummaryProviderError


class GeminiSummaryProvider:
    """Generate concise, timestamped summaries with a Gemini text model."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        client: Optional[Any] = None,
    ) -> None:
        """Create a summary provider using the configured Gemini model."""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate summaries")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)

    def generate_summary(
        self,
        transcript: Transcript,
        *,
        title: str,
        language: str,
    ) -> VideoSummary:
        """Generate and validate a structured summary from the transcript."""
        prompt, last_timestamp = _summary_prompt(transcript, title, language)
        response = self._generate_content(prompt)
        try:
            return _parse_response(response, last_timestamp)
        except ChapterTimestampOutOfRangeError as error:
            corrected_response = self._generate_content(
                _correction_prompt(prompt, error)
            )
            try:
                return _parse_response(corrected_response, last_timestamp)
            except ChapterTimestampOutOfRangeError as corrected_error:
                raise SummaryProviderError(
                    "Gemini returned an out-of-range chapter timestamp after "
                    f"correction: {corrected_error}"
                ) from corrected_error

    def _generate_content(self, prompt: str) -> Any:
        """Request one structured response from Gemini."""
        try:
            return self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "chapters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_sec": {"type": "number"},
                                        "title": {"type": "string"},
                                    },
                                    "required": ["start_sec", "title"],
                                },
                            },
                        },
                        "required": ["summary", "chapters"],
                    },
                ),
            )
        except (APIError, HTTPError) as error:
            raise SummaryProviderError(str(error)) from error


def _summary_prompt(
    transcript: Transcript, title: str, language: str
) -> tuple[str, float]:
    """Build the grounded prompt and find the final valid transcript timestamp."""
    if not transcript:
        raise SummaryProviderError("Cannot summarize an empty transcript")
    lines: list[str] = []
    last_timestamp = 0.0
    for segment in transcript.segments:
        start = segment.start_sec
        end = segment.end_sec
        last_timestamp = max(last_timestamp, end)
        lines.append(f"[{_timestamp(start)}] {segment.text.strip()}")
    prompt = (
        "Summarize the following YouTube transcript. Use only facts supported by "
        "the transcript. Return JSON with a 3-5 sentence `summary` and a "
        "chronological `chapters` array. Every chapter needs a `start_sec` "
        "timestamp from the transcript and a concise `title`. The only valid "
        f"range for every start_sec is 0.0 through {last_timestamp:.3f}; do not "
        "invent or round timestamps beyond that range. "
        f"Write all text in {language}.\n\n"
        f"Video title: {title}\n\nTranscript:\n" + "\n".join(lines)
    )
    return prompt, last_timestamp


def _correction_prompt(prompt: str, error: "ChapterTimestampOutOfRangeError") -> str:
    """Request a complete corrected response after timestamp validation fails."""
    return (
        f"{prompt}\n\nValidation feedback: a chapter start_sec of "
        f"{error.start_sec:.3f} was outside the valid range 0.0 through "
        f"{error.last_timestamp:.3f}. Return a complete corrected JSON response "
        "with every chapter timestamp inside that range."
    )


def _parse_response(response: Any, last_timestamp: float) -> VideoSummary:
    """Convert the provider's JSON response into validated domain models."""
    try:
        payload = json.loads(response.text)
        summary = payload["summary"]
        chapters_data = payload["chapters"]
        if not isinstance(summary, str) or not isinstance(chapters_data, list):
            raise ValueError("Summary response has invalid field types")
        chapters = tuple(
            _chapter_from_data(chapter, last_timestamp) for chapter in chapters_data
        )
        return VideoSummary(text=summary, chapters=chapters)
    except ChapterTimestampOutOfRangeError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise SummaryProviderError(
            f"Invalid Gemini summary response: {error}"
        ) from error


def _chapter_from_data(data: Any, last_timestamp: float) -> Chapter:
    """Validate one Gemini chapter against the source transcript duration."""
    if not isinstance(data, dict):
        raise ValueError("Chapter must be an object")
    start_sec = data.get("start_sec")
    title = data.get("title")
    if not isinstance(start_sec, (int, float)) or isinstance(start_sec, bool):
        raise ValueError("Chapter start_sec must be numeric")
    timestamp = float(start_sec)
    if timestamp < 0 or timestamp > last_timestamp:
        raise ChapterTimestampOutOfRangeError(timestamp, last_timestamp)
    if not isinstance(title, str):
        raise ValueError("Chapter title must be text")
    return Chapter(start_sec=timestamp, title=title)


class ChapterTimestampOutOfRangeError(ValueError):
    """A model-generated chapter timestamp does not cite the source transcript."""

    def __init__(self, start_sec: float, last_timestamp: float) -> None:
        self.start_sec = start_sec
        self.last_timestamp = last_timestamp
        super().__init__(
            f"Chapter start_sec {start_sec:.3f} exceeds transcript duration "
            f"{last_timestamp:.3f}"
        )


def _timestamp(seconds: float) -> str:
    """Render a transcript timestamp for the model prompt."""
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
