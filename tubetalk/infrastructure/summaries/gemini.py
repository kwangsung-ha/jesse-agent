"""Gemini adapter for structured transcript summaries."""

import json
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from tubetalk.core.logging import logger
from tubetalk.core.prompts import PromptCatalog, PromptTemplateError
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
        prompt_version: str = "summary-chapters-v1",
        prompts: PromptCatalog | None = None,
    ) -> None:
        """Create a summary provider using the configured Gemini model."""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate summaries")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)
        self._prompt_version = prompt_version
        self._prompts = prompts or PromptCatalog()

    def generate_summary(
        self,
        transcript: Transcript,
        *,
        title: str,
        language: str,
    ) -> VideoSummary:
        """Generate and validate a structured summary from the transcript."""
        prompt, last_timestamp = self._summary_prompt(transcript, title, language)
        response = self._generate_content(prompt)
        try:
            return _parse_response(response, last_timestamp)
        except ChapterTimestampOutOfRangeError as error:
            corrected_response = self._generate_content(
                self._correction_prompt(prompt, error)
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
        logger.bind(event="gemini.summary.request", model=self.model).debug(
            "--- prompt ---\n{}\n--- end prompt ---", prompt
        )
        try:
            response = self._client.models.generate_content(
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
            logger.bind(event="gemini.summary.response", model=self.model).trace(
                "{}", _response_text(response)
            )
            return response
        except (APIError, HTTPError) as error:
            raise SummaryProviderError(str(error)) from error

    def _summary_prompt(
        self, transcript: Transcript, title: str, language: str
    ) -> tuple[str, float]:
        if not transcript:
            raise SummaryProviderError("Cannot summarize an empty transcript")
        lines: list[str] = []
        last_timestamp = 0.0
        for segment in transcript.segments:
            last_timestamp = max(last_timestamp, segment.end_sec)
            lines.append(f"[{_timestamp(segment.start_sec)}] {segment.text.strip()}")
        try:
            prompt = self._prompts.render(
                "summary",
                self._prompt_version,
                {
                    "last_timestamp": f"{last_timestamp:.3f}",
                    "language": language,
                    "title": title,
                    "transcript": "\n".join(lines),
                },
            )
        except PromptTemplateError as error:
            raise SummaryProviderError(str(error)) from error
        return prompt, last_timestamp

    def _correction_prompt(
        self, prompt: str, error: "ChapterTimestampOutOfRangeError"
    ) -> str:
        try:
            return self._prompts.render(
                "summary_correction",
                "summary-correction-v1",
                {
                    "prompt": prompt,
                    "start_sec": f"{error.start_sec:.3f}",
                    "last_timestamp": f"{error.last_timestamp:.3f}",
                },
            )
        except PromptTemplateError as template_error:
            raise SummaryProviderError(str(template_error)) from template_error


def _response_text(response: Any) -> str:
    """Extract model text for verbose diagnostics without assuming SDK details."""
    return str(getattr(response, "text", response))


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
