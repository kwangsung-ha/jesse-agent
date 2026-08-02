"""Gemini adapter for coverage-first transcript summaries and chapters."""

import json
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from jesseagent.application.video.contracts import SummaryProviderError
from jesseagent.core.logging import logger
from jesseagent.core.prompts import PromptCatalog, PromptTemplateError
from jesseagent.domain.chaptering import (
    ChapterBlockPolicy,
    ChapterCandidate,
    ChapterWindowPolicy,
    TranscriptBlock,
    TranscriptWindow,
    block_transcript_segments,
    window_transcript,
)
from jesseagent.domain.summary import Chapter, VideoSummary
from jesseagent.domain.transcript import Transcript


class GeminiSummaryProvider:
    """Generate coverage-first, timestamped summaries with Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        client: Optional[Any] = None,
        prompt_version: str = "summary-chapters-v2",
        prompts: PromptCatalog | None = None,
        chapter_window_policy: ChapterWindowPolicy | None = None,
        chapter_block_policy: ChapterBlockPolicy | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate summaries")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)
        self._prompt_version = prompt_version
        self._prompts = prompts or PromptCatalog()
        self._chapter_window_policy = chapter_window_policy or ChapterWindowPolicy()
        self._chapter_block_policy = chapter_block_policy or ChapterBlockPolicy()

    def generate_summary(
        self, transcript: Transcript, *, title: str, language: str
    ) -> VideoSummary:
        """Extract local candidates, then consolidate every candidate globally."""
        if not transcript:
            raise SummaryProviderError("Cannot summarize an empty transcript")
        # Validate the configured final prompt before making any paid request.
        self._consolidation_prompt((), title, language)
        last_timestamp = max(segment.end_sec for segment in transcript.segments)
        candidates = self._extract_candidates(transcript, title, language)
        prompt = self._consolidation_prompt(candidates, title, language)
        response = self._generate_content(prompt, _consolidation_schema())
        try:
            return _parse_consolidation_response(
                response, candidates, transcript, last_timestamp
            )
        except ChapterTimestampOutOfRangeError as error:
            corrected_response = self._generate_content(
                self._correction_prompt(prompt, error), _consolidation_schema()
            )
            try:
                return _parse_consolidation_response(
                    corrected_response, candidates, transcript, last_timestamp
                )
            except ChapterTimestampOutOfRangeError as corrected_error:
                raise SummaryProviderError(
                    "Gemini returned an out-of-range chapter timestamp after "
                    f"correction: {corrected_error}"
                ) from corrected_error

    def _extract_candidates(
        self, transcript: Transcript, title: str, language: str
    ) -> tuple[ChapterCandidate, ...]:
        candidates: list[ChapterCandidate] = []
        for window in window_transcript(transcript, self._chapter_window_policy):
            blocks = block_transcript_segments(
                window.segments, self._chapter_block_policy
            )
            prompt = self._candidate_prompt(window, blocks, title, language)
            response = self._generate_content(prompt, _candidate_schema())
            try:
                extracted = _parse_candidate_response(
                    response, window, blocks, candidates
                )
            except CandidateBlockIndexOutOfRangeError as error:
                corrected_response = self._generate_content(
                    self._candidate_correction_prompt(prompt, error),
                    _candidate_schema(),
                )
                try:
                    extracted = _parse_candidate_response(
                        corrected_response, window, blocks, candidates
                    )
                except CandidateBlockIndexOutOfRangeError as corrected_error:
                    raise SummaryProviderError(
                        "Gemini returned an invalid candidate block index after "
                        f"correction: {corrected_error}"
                    ) from corrected_error
            candidates.extend(extracted)
        if not candidates:
            raise SummaryProviderError(
                "Candidate extraction returned no topic transitions"
            )
        return tuple(candidates)

    def _generate_content(self, prompt: str, schema: dict[str, Any]) -> Any:
        logger.bind(event="gemini.summary.request", model=self.model).debug(
            "--- prompt ---\n{}\n--- end prompt ---", prompt
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema
                ),
            )
            logger.bind(event="gemini.summary.response", model=self.model).trace(
                "{}", _response_text(response)
            )
            return response
        except (APIError, HTTPError) as error:
            raise SummaryProviderError(str(error)) from error

    def _candidate_prompt(
        self,
        window: TranscriptWindow,
        blocks: list[TranscriptBlock],
        title: str,
        language: str,
    ) -> str:
        try:
            return self._prompts.render(
                "chapter_candidates",
                "chapter-candidates-v1",
                {
                    "language": language,
                    "title": title,
                    "window_start": f"{window.start_sec:.3f}",
                    "window_end": f"{window.end_sec:.3f}",
                    "transcript": _format_blocks(blocks),
                },
            )
        except PromptTemplateError as error:
            raise SummaryProviderError(str(error)) from error

    def _consolidation_prompt(
        self, candidates: tuple[ChapterCandidate, ...], title: str, language: str
    ) -> str:
        serialized = json.dumps(
            [candidate.model_dump() for candidate in candidates], ensure_ascii=False
        )
        try:
            return self._prompts.render(
                "summary",
                self._prompt_version,
                {"language": language, "title": title, "candidates": serialized},
            )
        except PromptTemplateError as error:
            raise SummaryProviderError(str(error)) from error

    def _candidate_correction_prompt(
        self, prompt: str, error: "CandidateBlockIndexOutOfRangeError"
    ) -> str:
        try:
            return self._prompts.render(
                "chapter_candidates_correction",
                "chapter-candidates-correction-v1",
                {
                    "prompt": prompt,
                    "block_index": str(error.block_index),
                    "last_block_index": str(error.block_count - 1),
                },
            )
        except PromptTemplateError as template_error:
            raise SummaryProviderError(str(template_error)) from template_error

    def _correction_prompt(
        self, prompt: str, error: "ChapterTimestampOutOfRangeError"
    ) -> str:
        try:
            return self._prompts.render(
                "summary_correction",
                "summary-correction-v2",
                {
                    "prompt": prompt,
                    "start_sec": f"{error.start_sec:.3f}",
                    "last_timestamp": f"{error.last_timestamp:.3f}",
                },
            )
        except PromptTemplateError as template_error:
            raise SummaryProviderError(str(template_error)) from template_error


def _candidate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "block_index": {"type": "integer"},
                        "title": {"type": "string"},
                    },
                    "required": ["block_index", "title"],
                },
            }
        },
        "required": ["candidates"],
    }


def _consolidation_schema() -> dict[str, Any]:
    return {
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
                        "candidate_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["start_sec", "title", "candidate_ids"],
                },
            },
        },
        "required": ["summary", "chapters"],
    }


def _parse_candidate_response(
    response: Any,
    window: TranscriptWindow,
    blocks: list[TranscriptBlock],
    existing: list[ChapterCandidate],
) -> list[ChapterCandidate]:
    try:
        payload = json.loads(response.text)
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("Candidate response has invalid field types")
        candidates = []
        for offset, raw_candidate in enumerate(raw_candidates):
            if not isinstance(raw_candidate, dict):
                raise ValueError("Candidate must be an object")
            block_index = _candidate_block_index(raw_candidate, len(blocks))
            title = raw_candidate.get("title")
            if not isinstance(title, str):
                raise ValueError("Candidate title must be text")
            candidates.append(
                ChapterCandidate(
                    candidate_id=f"candidate-{len(existing) + offset}",
                    window_index=window.index,
                    start_sec=blocks[block_index].start_sec,
                    title=title,
                )
            )
        return candidates
    except CandidateBlockIndexOutOfRangeError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise SummaryProviderError(
            f"Invalid Gemini candidate response: {error}"
        ) from error


def _parse_consolidation_response(
    response: Any,
    candidates: tuple[ChapterCandidate, ...],
    transcript: Transcript,
    last_timestamp: float,
) -> VideoSummary:
    try:
        payload = json.loads(response.text)
        summary = payload["summary"]
        raw_chapters = payload["chapters"]
        if not isinstance(summary, str) or not isinstance(raw_chapters, list):
            raise ValueError("Summary response has invalid field types")
        assigned_ids: list[str] = []
        chapters = []
        for raw_chapter in raw_chapters:
            if not isinstance(raw_chapter, dict):
                raise ValueError("Chapter must be an object")
            candidate_ids = raw_chapter.get("candidate_ids")
            if not isinstance(candidate_ids, list) or not all(
                isinstance(candidate_id, str) for candidate_id in candidate_ids
            ):
                raise ValueError("Chapter candidate_ids must be a list of text")
            assigned_ids.extend(candidate_ids)
            timestamp = _required_timestamp(raw_chapter, last_timestamp)
            title = raw_chapter.get("title")
            if not isinstance(title, str):
                raise ValueError("Chapter title must be text")
            chapters.append(
                Chapter(
                    start_sec=_snap_to_segment_start(timestamp, transcript), title=title
                )
            )
        expected_ids = {candidate.candidate_id for candidate in candidates}
        if (
            len(assigned_ids) != len(set(assigned_ids))
            or set(assigned_ids) != expected_ids
        ):
            raise ValueError("Consolidation must assign every candidate exactly once")
        return VideoSummary(text=summary, chapters=tuple(chapters))
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


def _required_timestamp(data: dict[str, Any], last_timestamp: float) -> float:
    value = data.get("start_sec")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Chapter start_sec must be numeric")
    timestamp = float(value)
    if timestamp < 0 or timestamp > last_timestamp:
        raise ChapterTimestampOutOfRangeError(timestamp, last_timestamp)
    return timestamp


def _candidate_block_index(data: dict[str, Any], block_count: int) -> int:
    value = data.get("block_index")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Candidate block_index must be an integer")
    if value < 0 or value >= block_count:
        raise CandidateBlockIndexOutOfRangeError(value, block_count)
    return value


def _snap_to_segment_start(timestamp: float, transcript: Transcript) -> float:
    starts = [segment.start_sec for segment in transcript.segments]
    return max((start for start in starts if start <= timestamp), default=starts[0])


def _format_blocks(blocks: list[TranscriptBlock]) -> str:
    return "\n".join(
        f"[B{index} | {_timestamp(block.start_sec)}–{_timestamp(block.end_sec)}] "
        f"{block.text}"
        for index, block in enumerate(blocks)
    )


def _response_text(response: Any) -> str:
    return str(getattr(response, "text", response))


class ChapterTimestampOutOfRangeError(ValueError):
    """A model-generated timestamp lies outside its source transcript."""

    def __init__(self, start_sec: float, last_timestamp: float) -> None:
        self.start_sec = start_sec
        self.last_timestamp = last_timestamp
        super().__init__(
            f"Chapter start_sec {start_sec:.3f} exceeds transcript duration "
            f"{last_timestamp:.3f}"
        )


class CandidateBlockIndexOutOfRangeError(ValueError):
    """A local candidate refers to a prompt block that was not supplied."""

    def __init__(self, block_index: int, block_count: int) -> None:
        self.block_index = block_index
        self.block_count = block_count
        super().__init__(
            f"Candidate block_index {block_index} is outside valid block indexes "
            f"0 through {block_count - 1}"
        )


def _timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
