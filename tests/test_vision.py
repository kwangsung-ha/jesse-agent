"""Unit tests for the Gemini URL-video scene analyzer."""

from types import SimpleNamespace

import pytest

from tubetalk.domain.vision import VisionScene, VisionSource, YouTubeUrlVisionSource
from tubetalk.infrastructure.visions.gemini import GeminiVisionAnalyzer
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
    )

    assert scenes[0].start_sec == 0.0
    assert scenes[0].detected_objects == ("presenter",)
    assert captured["model"] == "gemini-3.5-flash"
    assert captured["input"] == [
        {"type": "video", "uri": "https://www.youtube.com/watch?v=abc123"},
        {
            "type": "text",
            "text": (
                "Analyze this public YouTube video visually. Return a chronological "
                "list of salient visual scenes only; do not summarize spoken content "
                "unless it is visibly shown. Use exact timestamps supported by the "
                "video, cover meaningful changes, and include useful visible objects, "
                "people, text, charts, or actions. Keep each visual_summary concise. "
                "Video title: Example video"
            ),
        },
    ]


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
            YouTubeUrlVisionSource("https://youtu.be/abc123"), title="Example"
        )


def test_youtube_source_rejects_non_youtube_urls() -> None:
    """Only the Phase 2 public YouTube source type is accepted."""
    with pytest.raises(ValueError, match="public YouTube URL"):
        YouTubeUrlVisionSource("https://example.com/video.mp4")


def test_gemini_analyzer_rejects_another_vision_source_type() -> None:
    """The provider boundary reserves local sources for a future implementation."""
    analyzer = GeminiVisionAnalyzer(api_key="key", client=SimpleNamespace())

    with pytest.raises(VisionProviderError, match="public YouTube URL source"):
        analyzer.describe(VisionSource(), title="Example")


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
