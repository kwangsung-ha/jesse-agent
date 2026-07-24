"""CLI adapter tests using a mocked application service."""

from dataclasses import replace
from typing import Any

from typer.testing import CliRunner

from tubetalk.cli.main import _format_vision_state, _format_vision_summary, app
from tubetalk.domain.summary import Chapter, VideoSummary
from tubetalk.services.results import IndexingResult
from tubetalk.services.video_service import (
    InvalidVideoUrlError,
    ProcessResult,
    SummaryGenerationError,
    SummaryResult,
    SummaryUnavailableError,
    VideoIngestionError,
    VideoNotFoundError,
    VideoStatus,
)

runner = CliRunner()


def _summary() -> VideoSummary:
    return VideoSummary(
        text="영상의 핵심 내용을 요약합니다.",
        chapters=(
            Chapter(start_sec=0, title="소개"),
            Chapter(start_sec=65, title="핵심"),
        ),
    )


def _status(video_id: str = "vid_a") -> VideoStatus:
    return VideoStatus(
        video_id=video_id,
        title="Alpha Video",
        channel="Alpha Channel",
        duration=120.0,
        has_metadata=True,
        has_transcript=True,
        has_vision_index=False,
        transcript_segments=3,
        transcript_index_state="current",
        transcript_index_chunks=2,
        transcript_index_model="gemini-embedding-2",
        transcript_index_dimension=768,
        transcript_indexed_at="2026-07-23T00:00:00+00:00",
        summary_state="current",
        summary_chapters=3,
        summary_model="gemini-3.5-flash-lite",
        summary_prompt_version="summary-chapters-v1",
        summary_language="ko",
        summary_generated_at="2026-07-23T00:00:00+00:00",
        cached_at="2026-07-23T00:00:00+00:00",
    )


def _mock_service(mocker: Any) -> Any:
    service = mocker.Mock()
    mocker.patch("tubetalk.cli.main.create_video_service", return_value=service)
    return service


def test_status_shows_empty_message(mocker: Any) -> None:
    """The CLI renders an empty result without cache-specific logic."""
    service = _mock_service(mocker)
    service.list_statuses.return_value = []

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "No cached videos found" in result.output


def test_status_shows_list_and_detail(mocker: Any) -> None:
    """Status output is rendered from typed service results."""
    service = _mock_service(mocker)
    service.list_statuses.return_value = [_status()]
    service.get_status.return_value = _status("vid_x")

    list_result = runner.invoke(app, ["status"])
    detail_result = runner.invoke(app, ["status", "vid_x"])

    assert list_result.exit_code == 0
    assert "Alpha Video" in list_result.output
    assert "✅ 2" in list_result.output
    assert "✅ 3" in list_result.output
    assert detail_result.exit_code == 0
    assert "vid_x" in detail_result.output
    assert "120s" in detail_result.output
    assert "Current" in detail_result.output
    assert "gemini-3.5-flash-lite" in detail_result.output


def test_status_not_found_maps_service_error_to_exit_code(mocker: Any) -> None:
    """The CLI owns its exit-code representation for a missing video."""
    service = _mock_service(mocker)
    service.get_status.side_effect = VideoNotFoundError("Video 'missing' not found")

    result = runner.invoke(app, ["status", "missing"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_vision_status_formatters_show_freshness_and_scene_count() -> None:
    """Vision freshness should have concise list and detailed renderings."""
    missing = _status()
    current = replace(missing, vision_index_state="current", vision_scene_count=4)
    stale = replace(missing, vision_index_state="stale")
    invalid = replace(missing, vision_index_state="invalid")

    assert _format_vision_summary(missing) == "—"
    assert _format_vision_summary(current) == "✅ 4"
    assert _format_vision_summary(stale) == "⚠️ stale"
    assert _format_vision_summary(invalid) == "❌ invalid"
    assert _format_vision_state(current) == "✅ Current"
    assert _format_vision_state(stale) == "⚠️ Stale"
    assert _format_vision_state(invalid) == "❌ Invalid"
    assert _format_vision_state(missing) == "❌ Missing"


def test_process_renders_cache_miss_and_index_result(mocker: Any) -> None:
    """The CLI presents a service result without invoking infrastructure."""
    service = _mock_service(mocker)
    service.process.return_value = ProcessResult(
        video_id="dQw4w9WgXcQ",
        cache_hit=False,
        transcript_segments=1,
        indexing=IndexingResult(state="indexed", chunk_count=1),
        summary=SummaryResult(state="generated", summary=_summary()),
    )
    service.get_status.return_value = _status("dQw4w9WgXcQ")

    result = runner.invoke(app, ["process", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "Processing video data" in result.output
    assert "Completed in 0.00s" in result.output
    assert "Saved 1 transcript segments" in result.output
    assert "Indexed 1 transcript chunks" in result.output
    assert "영상의 핵심 내용을 요약합니다" in result.output
    assert "01:05" in result.output
    service.process.assert_called_once_with("https://youtu.be/dQw4w9WgXcQ")


def test_process_renders_cache_hit_and_index_warning(mocker: Any) -> None:
    """A non-fatal indexing warning remains visible to CLI users."""
    service = _mock_service(mocker)
    service.process.return_value = ProcessResult(
        video_id="dQw4w9WgXcQ",
        cache_hit=True,
        transcript_segments=1,
        indexing=IndexingResult(state="warning", warning="GEMINI_API_KEY is required"),
        summary=SummaryResult(state="warning", warning="Gemini unavailable"),
    )
    service.get_status.return_value = _status("dQw4w9WgXcQ")

    result = runner.invoke(app, ["process", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "Cache hit" in result.output
    assert "transcript index was not updated" in result.output


def test_process_maps_service_errors_to_existing_exit_codes(mocker: Any) -> None:
    """URL and ingestion errors retain their user-facing CLI contracts."""
    service = _mock_service(mocker)
    service.process.side_effect = InvalidVideoUrlError("Cannot extract video_id")

    invalid_result = runner.invoke(app, ["process", "https://example.com/video"])

    service.process.side_effect = VideoIngestionError("Failed to process video")
    failed_result = runner.invoke(app, ["process", "https://youtu.be/dQw4w9WgXcQ"])

    assert invalid_result.exit_code == 2
    assert "Cannot extract video_id" in invalid_result.output
    assert failed_result.exit_code == 1
    assert "Failed to process video" in failed_result.output


def test_summary_displays_current_cache_or_generates_when_requested(
    mocker: Any,
) -> None:
    """The CLI renders service summaries and forwards the generation flag."""
    service = _mock_service(mocker)
    service.get_summary.return_value = SummaryResult(
        state="current", summary=_summary()
    )

    current_result = runner.invoke(app, ["summary", "vid_a"])

    service.get_summary.return_value = SummaryResult(
        state="generated", summary=_summary()
    )
    generated_result = runner.invoke(app, ["summary", "vid_a", "--generate"])

    assert current_result.exit_code == 0
    assert "영상의 핵심 내용을 요약합니다" in current_result.output
    assert "01:05" in current_result.output
    service.get_summary.assert_called_with("vid_a", generate=True)
    assert generated_result.exit_code == 0
    assert "Generated transcript summary" in generated_result.output


def test_summary_without_video_id_prompts_from_cached_list(mocker: Any) -> None:
    """The interactive form uses a numbered cache list instead of opaque IDs."""
    service = _mock_service(mocker)
    service.list_statuses.return_value = [_status("vid_a"), _status("vid_b")]
    service.get_summary.return_value = SummaryResult(
        state="current", summary=_summary()
    )

    result = runner.invoke(app, ["summary"], input="2\n")

    assert result.exit_code == 0
    assert "Select a Cached Video" in result.output
    assert "Alpha Video" in result.output
    assert "Select a video number" in result.output
    service.get_summary.assert_called_once_with("vid_b", generate=False)


def test_summary_without_video_id_handles_empty_and_invalid_selections(
    mocker: Any,
) -> None:
    """The selection prompt gives clear errors for unavailable or invalid choices."""
    service = _mock_service(mocker)
    service.list_statuses.return_value = []

    empty_result = runner.invoke(app, ["summary"])

    service.list_statuses.return_value = [_status()]
    invalid_result = runner.invoke(app, ["summary"], input="9\n")

    assert empty_result.exit_code == 1
    assert "No cached videos found" in empty_result.output
    assert invalid_result.exit_code == 2
    assert "displayed list" in invalid_result.output


def test_summary_maps_unavailable_and_generation_errors(mocker: Any) -> None:
    """Summary cache and provider failures are user-facing command errors."""
    service = _mock_service(mocker)
    service.get_summary.side_effect = SummaryUnavailableError("Run --generate")

    unavailable_result = runner.invoke(app, ["summary", "vid_a"])

    service.get_summary.side_effect = SummaryGenerationError("Gemini unavailable")
    failed_result = runner.invoke(app, ["summary", "vid_a", "--generate"])

    assert unavailable_result.exit_code == 1
    assert "Run --generate" in unavailable_result.output
    assert failed_result.exit_code == 1
    assert "Gemini unavailable" in failed_result.output
