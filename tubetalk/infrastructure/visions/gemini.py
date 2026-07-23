"""Gemini adapter for public YouTube URL scene analysis."""

import json
from typing import Any, Optional

from google import genai
from google.genai.errors import APIError
from httpx import HTTPError

from tubetalk.core.config import settings
from tubetalk.domain.vision import VisionScene, VisionSource, YouTubeUrlVisionSource
from tubetalk.ports.vision import VisionProviderError


class GeminiVisionAnalyzer:
    """Create structured visual-scene descriptions from a YouTube URL."""

    def __init__(
        self,
        api_key: str,
        model: str = settings.vision_model,
        client: Optional[Any] = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to analyze video scenes")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)

    def describe(
        self, source: VisionSource, *, title: str, duration_sec: float
    ) -> tuple[VisionScene, ...]:
        """Request a timestamped visual index that covers the whole video."""
        if not isinstance(source, YouTubeUrlVisionSource):
            raise VisionProviderError(
                "Gemini URL-video analysis requires a public YouTube URL source"
            )
        if duration_sec <= 0:
            raise VisionProviderError(
                "Video duration must be positive for scene coverage"
            )
        scenes = _parse_response(
            self._generate(source, _vision_prompt(title, duration_sec))
        )
        errors = _coverage_errors(scenes, duration_sec)
        if not errors:
            return scenes
        corrected = _parse_response(
            self._generate(source, _correction_prompt(title, duration_sec, errors))
        )
        corrected_errors = _coverage_errors(corrected, duration_sec)
        if corrected_errors:
            raise VisionProviderError(
                "Gemini scene coverage remained incomplete after correction: "
                + "; ".join(corrected_errors)
            )
        return corrected

    def _generate(self, source: YouTubeUrlVisionSource, prompt: str) -> Any:
        try:
            return self._client.interactions.create(
                model=self.model,
                input=[
                    {"type": "video", "uri": source.url},
                    {"type": "text", "text": prompt},
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _response_schema(),
                },
            )
        except (APIError, HTTPError) as error:
            raise VisionProviderError(str(error)) from error


def _vision_prompt(title: str, duration_sec: float) -> str:
    return (
        "Analyze this public YouTube video visually. Return chronological scenes "
        f"that cover every moment from 0 through {duration_sec:.3f} seconds. "
        "Each scene must be no longer than 30 seconds, contiguous with its "
        "neighbors, and use exact video-supported timestamps. Include visual "
        "objects, people, text, charts, or actions; do not summarize speech unless "
        f"it is visibly shown. Video title: {title}"
    )


def _correction_prompt(title: str, duration_sec: float, errors: list[str]) -> str:
    return (
        _vision_prompt(title, duration_sec)
        + " Return a complete replacement JSON response. Validation errors: "
        + "; ".join(errors)
    )


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "visual_summary": {"type": "string"},
                        "detected_objects": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "start_sec",
                        "end_sec",
                        "visual_summary",
                        "detected_objects",
                    ],
                },
            }
        },
        "required": ["scenes"],
    }


def _parse_response(response: Any) -> tuple[VisionScene, ...]:
    try:
        output_text = response.output_text
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError(
                f"Gemini returned no scene JSON ({_response_status(response)})"
            )
        payload = json.loads(output_text)
        scenes_data = payload["scenes"]
        if not isinstance(scenes_data, list):
            raise ValueError("Scenes must be a list")
        scenes = tuple(_scene_from_data(item) for item in scenes_data)
        if any(
            current.start_sec < previous.start_sec
            for previous, current in zip(scenes, scenes[1:])
        ):
            raise ValueError("Scenes must be ordered by start_sec")
        return scenes
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise VisionProviderError(f"Invalid Gemini vision response: {error}") from error


def _response_status(response: Any) -> str:
    """Return compact interaction state without assuming a specific SDK shape."""
    status = getattr(response, "status", None)
    if isinstance(status, str) and status:
        return f"status={status}"
    steps = getattr(response, "steps", None)
    if isinstance(steps, list) and steps:
        return f"steps={len(steps)}"
    return "no status details"


def _coverage_errors(scenes: tuple[VisionScene, ...], duration_sec: float) -> list[str]:
    """Return all coverage violations for the 30-second scene policy."""
    if not scenes:
        return ["no scenes returned"]
    errors: list[str] = []
    covered_until = 0.0
    for scene in scenes:
        if scene.start_sec > covered_until + 0.001:
            errors.append(f"uncovered {covered_until:.3f}-{scene.start_sec:.3f}")
        if scene.end_sec - scene.start_sec > 30.001:
            errors.append(f"scene exceeds 30 seconds at {scene.start_sec:.3f}")
        if scene.end_sec > duration_sec + 0.001:
            errors.append(f"scene exceeds video duration at {scene.end_sec:.3f}")
        covered_until = max(covered_until, scene.end_sec)
    if covered_until < duration_sec - 0.001:
        errors.append(f"uncovered {covered_until:.3f}-{duration_sec:.3f}")
    return errors


def _scene_from_data(data: Any) -> VisionScene:
    if not isinstance(data, dict):
        raise ValueError("Scene must be an object")
    start_sec = _required_number(data, "start_sec")
    end_sec = _required_number(data, "end_sec")
    summary = _required_text(data, "visual_summary")
    objects = data.get("detected_objects")
    if not isinstance(objects, list) or not all(
        isinstance(item, str) for item in objects
    ):
        raise ValueError("detected_objects must be a list of text")
    return VisionScene(
        start_sec=start_sec,
        end_sec=end_sec,
        visual_summary=summary,
        detected_objects=tuple(objects),
    )


def _required_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value
