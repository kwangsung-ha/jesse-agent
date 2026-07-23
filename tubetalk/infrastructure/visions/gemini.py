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

    def describe(self, source: VisionSource, *, title: str) -> tuple[VisionScene, ...]:
        """Request a concise, timestamped visual index for a public video."""
        if not isinstance(source, YouTubeUrlVisionSource):
            raise VisionProviderError(
                "Gemini URL-video analysis requires a public YouTube URL source"
            )
        try:
            response = self._client.interactions.create(
                model=self.model,
                input=[
                    {"type": "video", "uri": source.url},
                    {"type": "text", "text": _vision_prompt(title)},
                ],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": _response_schema(),
                },
            )
        except (APIError, HTTPError) as error:
            raise VisionProviderError(str(error)) from error
        return _parse_response(response)


def _vision_prompt(title: str) -> str:
    return (
        "Analyze this public YouTube video visually. Return a chronological list "
        "of salient visual scenes only; do not summarize spoken content unless it "
        "is visibly shown. Use exact timestamps supported by the video, cover "
        "meaningful changes, and include useful visible objects, people, text, "
        "charts, or actions. Keep each visual_summary concise. "
        f"Video title: {title}"
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
        payload = json.loads(response.output_text)
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
