"""Unit tests for the Gemini URL-video scene analyzer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tubetalk.domain.vision import VisionScene, VisionSource, YouTubeUrlVisionSource
from tubetalk.infrastructure.visions.gemini import (
    GeminiVisionAnalyzer,
    _coverage_errors,
)
from tubetalk.ports.vision import VisionProviderError


def test_describe_sends_public_video_and_returns_ordered_scenes() -> None:
    """The provider should request JSON scenes from the supplied YouTube URL."""
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output_text=(
                '{"scenes":[{"start_sec":0,"end_sec":5,'
                '"visual_summary":"A presenter appears.",'
                '"detected_objects":["presenter"]}]}'
            )
        )

    interactions = SimpleNamespace(create=create)
    client = SimpleNamespace(interactions=interactions)
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    scenes = analyzer.describe(
        YouTubeUrlVisionSource("https://www.youtube.com/watch?v=abc123"),
        title="Example video",
        duration_sec=5,
    )

    assert scenes[0].start_sec == 0.0
    assert scenes[0].detected_objects == ("presenter",)
    assert captured["model"] == "gemini-3.5-flash"
    assert captured["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": {
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
        },
    }
    assert captured["input"][0] == {
        "type": "video",
        "uri": "https://www.youtube.com/watch?v=abc123",
    }
    assert (
        "cover every moment from 0 through 5.000 seconds"
        in captured["input"][1]["text"]
    )


def test_describe_rejects_invalid_scene_responses() -> None:
    """Malformed provider output must not become a cached scene index."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text='{"scenes":[{}]}')
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    with pytest.raises(VisionProviderError, match="Invalid Gemini vision response"):
        analyzer.describe(
            YouTubeUrlVisionSource("https://youtu.be/abc123"),
            title="Example",
            duration_sec=5,
        )


def test_describe_reports_empty_interaction_output_with_status() -> None:
    """Empty model output should identify the interaction state for diagnosis."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text="", status="failed")
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    with pytest.raises(VisionProviderError, match="status=failed"):
        analyzer.describe(
            YouTubeUrlVisionSource("https://youtu.be/abc123"),
            title="Example",
            duration_sec=5,
        )


def test_describe_retries_once_when_scenes_do_not_cover_duration() -> None:
    """A sparse first response should be replaced by complete 30-second scenes."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=Mock(
                side_effect=[
                    SimpleNamespace(
                        output_text=(
                            '{"scenes":[{"start_sec":0,"end_sec":30,'
                            '"visual_summary":"A presenter appears.",'
                            '"detected_objects":[]}]}'
                        )
                    ),
                    SimpleNamespace(
                        output_text=(
                            '{"scenes":[{"start_sec":0,"end_sec":30,'
                            '"visual_summary":"A presenter appears.",'
                            '"detected_objects":[]},{"start_sec":30,'
                            '"end_sec":60,"visual_summary":"A chart appears.",'
                            '"detected_objects":["chart"]}]}'
                        )
                    ),
                ]
            )
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    scenes = analyzer.describe(
        YouTubeUrlVisionSource("https://youtu.be/abc123"),
        title="Example",
        duration_sec=60,
    )

    assert len(scenes) == 2
    assert client.interactions.create.call_count == 2


def test_coverage_validation_reports_gaps_long_scenes_and_duration_overflow() -> None:
    """Coverage checks should identify every reason a scene list is unusable."""
    errors = _coverage_errors(
        (
            VisionScene(5, 40, "A long scene.", ()),
            VisionScene(45, 70, "An overflow scene.", ()),
        ),
        60,
    )

    assert "uncovered 0.000-5.000" in errors
    assert "scene exceeds 30 seconds at 5.000" in errors
    assert "scene exceeds video duration at 70.000" in errors
    assert _coverage_errors((), 60) == ["no scenes returned"]


def test_describe_rejects_non_positive_duration() -> None:
    """Coverage generation requires a real video duration."""
    analyzer = GeminiVisionAnalyzer(api_key="key", client=SimpleNamespace())

    with pytest.raises(VisionProviderError, match="duration must be positive"):
        analyzer.describe(
            YouTubeUrlVisionSource("https://youtu.be/abc123"),
            title="Example",
            duration_sec=0,
        )


def test_youtube_source_rejects_non_youtube_urls() -> None:
    """Only the Phase 2 public YouTube source type is accepted."""
    with pytest.raises(ValueError, match="public YouTube URL"):
        YouTubeUrlVisionSource("https://example.com/video.mp4")


def test_gemini_analyzer_rejects_another_vision_source_type() -> None:
    """The provider boundary reserves local sources for a future implementation."""
    analyzer = GeminiVisionAnalyzer(api_key="key", client=SimpleNamespace())

    with pytest.raises(VisionProviderError, match="public YouTube URL source"):
        analyzer.describe(VisionSource(), title="Example", duration_sec=5)


@pytest.mark.parametrize(
    ("start_sec", "end_sec", "summary", "objects"),
    [
        (-1, 0, "A scene", ()),
        (5, 0, "A scene", ()),
        (0, 1, " ", ()),
    ],
)
def test_vision_scene_rejects_invalid_values(
    start_sec: float, end_sec: float, summary: str, objects: tuple[str, ...]
) -> None:
    """Invalid scene values should be rejected at the domain boundary."""
    with pytest.raises(ValueError):
        VisionScene(start_sec, end_sec, summary, objects)
