"""Unit tests for the Gemini URL-video scene analyzer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jesseagent.application.video.contracts import VisionProviderError
from jesseagent.domain.vision import VisionScene, VisionSource, YouTubeUrlVisionSource
from jesseagent.infrastructure.visions.gemini import (
    GeminiVisionAnalyzer,
)


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


def test_describe_keeps_sparse_scenes_without_a_second_api_request() -> None:
    """Coverage gaps should not spend another request on a correction attempt."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    output_text=(
                        '{"scenes":[{"start_sec":0,"end_sec":30,'
                        '"visual_summary":"A presenter appears.",'
                        '"detected_objects":[]}]}'
                    )
                )
            )
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    scenes = analyzer.describe(
        YouTubeUrlVisionSource("https://youtu.be/abc123"),
        title="Example",
        duration_sec=60,
    )

    assert len(scenes) == 1
    assert client.interactions.create.call_count == 1


def test_describe_clamps_timestamps_and_drops_only_non_positive_ranges() -> None:
    """One malformed timestamp range must not discard the usable scenes."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    output_text=(
                        '{"scenes":[{"start_sec":-5,"end_sec":10,'
                        '"visual_summary":"First.","detected_objects":[]},'
                        '{"start_sec":50,"end_sec":20,'
                        '"visual_summary":"Drop.","detected_objects":[]},'
                        '{"start_sec":20,"end_sec":80,'
                        '"visual_summary":"Last.","detected_objects":[]}]}'
                    )
                )
            )
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    scenes = analyzer.describe(
        YouTubeUrlVisionSource("https://youtu.be/abc123"),
        title="Example",
        duration_sec=60,
    )

    assert [(scene.start_sec, scene.end_sec) for scene in scenes] == [
        (0.0, 10.0),
        (20.0, 60.0),
    ]


def test_describe_fails_only_when_every_scene_is_unusable() -> None:
    """An index cannot be created when normalization removes every scene."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    output_text=(
                        '{"scenes":[{"start_sec":30,"end_sec":20,'
                        '"visual_summary":"Invalid.","detected_objects":[]}]}'
                    )
                )
            )
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    with pytest.raises(VisionProviderError, match="no usable visual scenes"):
        analyzer.describe(
            YouTubeUrlVisionSource("https://youtu.be/abc123"),
            title="Example",
            duration_sec=60,
        )


def test_describe_rejects_non_finite_timestamps() -> None:
    """NaN timestamps are not safe to clamp or persist."""
    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    output_text=(
                        '{"scenes":[{"start_sec":NaN,"end_sec":10,'
                        '"visual_summary":"Invalid.","detected_objects":[]}]}'
                    )
                )
            )
        )
    )
    analyzer = GeminiVisionAnalyzer(api_key="key", client=client)

    with pytest.raises(VisionProviderError, match="start_sec must be finite"):
        analyzer.describe(
            YouTubeUrlVisionSource("https://youtu.be/abc123"),
            title="Example",
            duration_sec=60,
        )


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
