"""CLI adapter tests using a mocked application service."""

from typing import Any

from typer.testing import CliRunner

from tubetalk.cli.main import app
from tubetalk.services.video_service import (
    IndexingResult,
    InvalidVideoUrlError,
    ProcessResult,
    SummaryResult,
    VideoIngestionError,
    VideoNotFoundError,
    VideoStatus,
)

runner = CliRunner()


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


def test_process_renders_cache_miss_and_index_result(mocker: Any) -> None:
    """The CLI presents a service result without invoking infrastructure."""
    service = _mock_service(mocker)
    service.process.return_value = ProcessResult(
        video_id="dQw4w9WgXcQ",
        cache_hit=False,
        transcript_segments=1,
        indexing=IndexingResult(state="indexed", chunk_count=1),
        summary=SummaryResult(state="generated"),
    )
    service.get_status.return_value = _status("dQw4w9WgXcQ")

    result = runner.invoke(app, ["process", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "Saved 1 transcript segments" in result.output
    assert "Indexed 1 transcript chunks" in result.output
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
