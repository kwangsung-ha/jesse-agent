"""Gemini adapter for public YouTube URL scene analysis."""

import json
import math
from typing import Any, Optional

from google import genai
from google.genai.errors import APIError
from httpx import HTTPError

from jesseagent.application.video.contracts import VisionProviderError
from jesseagent.core.logging import logger
from jesseagent.core.prompts import PromptCatalog, PromptTemplateError
from jesseagent.domain.vision import VisionScene, VisionSource, YouTubeUrlVisionSource


class GeminiVisionAnalyzer:
    """Create structured visual-scene descriptions from a YouTube URL."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        client: Optional[Any] = None,
        prompt_version: str = "vision-scenes-v2-30s",
        prompts: PromptCatalog | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to analyze video scenes")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)
        self._prompt_version = prompt_version
        self._prompts = prompts or PromptCatalog()

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
        try:
            prompt = self._prompts.render(
                "vision",
                self._prompt_version,
                {"title": title, "duration_sec": f"{duration_sec:.3f}"},
            )
        except PromptTemplateError as error:
            raise VisionProviderError(str(error)) from error
        return _parse_response(self._generate(source, prompt), duration_sec)

    def _generate(self, source: YouTubeUrlVisionSource, prompt: str) -> Any:
        logger.bind(event="gemini.vision.request", model=self.model).debug(
            "source={}\n--- prompt ---\n{}\n--- end prompt ---", source.url, prompt
        )
        try:
            response = self._client.interactions.create(
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
            logger.bind(event="gemini.vision.response", model=self.model).trace(
                "{}", str(getattr(response, "output_text", response))
            )
            return response
        except (APIError, HTTPError) as error:
            raise VisionProviderError(str(error)) from error


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


def _parse_response(response: Any, duration_sec: float) -> tuple[VisionScene, ...]:
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
        scenes = tuple(
            scene
            for item in scenes_data
            if (scene := _scene_from_data(item, duration_sec)) is not None
        )
        scenes = tuple(sorted(scenes, key=lambda scene: scene.start_sec))
        if not scenes:
            raise ValueError("Gemini returned no usable visual scenes")
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


def _scene_from_data(data: Any, duration_sec: float) -> VisionScene | None:
    if not isinstance(data, dict):
        raise ValueError("Scene must be an object")
    start_sec = _clamp_timestamp(_required_number(data, "start_sec"), duration_sec)
    end_sec = _clamp_timestamp(_required_number(data, "end_sec"), duration_sec)
    summary = _required_text(data, "visual_summary")
    objects = data.get("detected_objects")
    if not isinstance(objects, list) or not all(
        isinstance(item, str) for item in objects
    ):
        raise ValueError("detected_objects must be a list of text")
    if end_sec <= start_sec:
        return None
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
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _clamp_timestamp(value: float, duration_sec: float) -> float:
    """Keep usable model timestamps inside the known video range."""
    return min(max(value, 0.0), duration_sec)


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value
