"""Unit tests for interface-independent video application services."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tubetalk.core.cache import LocalCacheManager
from tubetalk.ports.transcript_index_repository import TranscriptIndexStatus
from tubetalk.services.video_service import (
    InvalidVideoUrlError,
    VideoIngestionError,
    VideoNotFoundError,
    VideoService,
)


def _service(tmp_path: Path, mocker: Any) -> tuple[VideoService, Any, Any, Any]:
    cache = LocalCacheManager(data_dir=tmp_path)
    loader = mocker.Mock()
    store = mocker.Mock()
    store.get_index_status.return_value = TranscriptIndexStatus(state="missing")
    provider_factory = mocker.Mock(return_value=mocker.Mock())
    service = VideoService(
        cache=cache,
        loader=loader,
        embedding_provider_factory=provider_factory,
        transcript_index_repository_factory=mocker.Mock(return_value=store),
    )
    return service, loader, store, provider_factory


def test_process_cache_miss_saves_resources_and_indexes(
    tmp_path: Path, mocker: Any
) -> None:
    """A cache miss coordinates collection, persistence, and indexing."""
    service, loader, store, provider_factory = _service(tmp_path, mocker)
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.return_value = {"title": "Example"}
    loader.fetch_transcript.return_value = [{"start_sec": 0, "text": "Hello"}]
    store.needs_indexing.return_value = True
    store.index_transcript.return_value = 1

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.cache_hit is False
    assert result.transcript_segments == 1
    assert result.indexing.state == "indexed"
    assert result.indexing.chunk_count == 1
    assert (tmp_path / "dQw4w9WgXcQ" / "metadata.json").is_file()
    loader.fetch_metadata.assert_called_once_with("https://youtu.be/dQw4w9WgXcQ")
    loader.fetch_transcript.assert_called_once_with("dQw4w9WgXcQ")
    provider_factory.assert_called_once_with()


def test_process_cache_hit_skips_remote_loading_and_keeps_current_index(
    tmp_path: Path, mocker: Any
) -> None:
    """A complete cache is reused without another loader call."""
    service, loader, store, provider_factory = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json("dQw4w9WgXcQ", "metadata.json", {"title": "Example"})
    cache.save_json(
        "dQw4w9WgXcQ", "transcript.json", [{"start_sec": 0, "text": "Hello"}]
    )
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    store.needs_indexing.return_value = False

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.cache_hit is True
    assert result.indexing.state == "current"
    provider_factory.assert_not_called()
    loader.fetch_metadata.assert_not_called()
    loader.fetch_transcript.assert_not_called()
    store.index_transcript.assert_not_called()


def test_process_keeps_cache_when_indexing_fails(tmp_path: Path, mocker: Any) -> None:
    """Embedding failures are returned as warnings after JSON is persisted."""
    service, loader, store, provider_factory = _service(tmp_path, mocker)
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.return_value = {"title": "Example"}
    loader.fetch_transcript.return_value = [{"start_sec": 0, "text": "Hello"}]
    store.needs_indexing.return_value = True
    provider_factory.side_effect = ValueError("GEMINI_API_KEY is required")

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.indexing.state == "warning"
    assert "GEMINI_API_KEY" in result.indexing.warning
    assert LocalCacheManager(data_dir=tmp_path).has_cache("dQw4w9WgXcQ") is True


def test_process_raises_service_errors_for_invalid_url_and_ingestion(
    tmp_path: Path, mocker: Any
) -> None:
    """Interfaces receive stable error types instead of loader exceptions."""
    service, loader, _, _ = _service(tmp_path, mocker)
    loader.extract_video_id.side_effect = ValueError("Cannot extract video_id")

    with pytest.raises(InvalidVideoUrlError, match="Cannot extract video_id"):
        service.process("https://example.com/video")

    loader.extract_video_id.side_effect = None
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.side_effect = subprocess.CalledProcessError(
        1, "yt-dlp", stderr="yt-dlp failed"
    )
    with pytest.raises(VideoIngestionError, match="Failed to process"):
        service.process("https://youtu.be/dQw4w9WgXcQ")


def test_process_does_not_hide_unexpected_indexing_errors(
    tmp_path: Path, mocker: Any
) -> None:
    """Programming errors must not be incorrectly reported as API warnings."""
    service, loader, store, _ = _service(tmp_path, mocker)
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.return_value = {"title": "Example"}
    loader.fetch_transcript.return_value = [{"start_sec": 0, "text": "Hello"}]
    store.needs_indexing.side_effect = RuntimeError("unexpected bug")

    with pytest.raises(RuntimeError, match="unexpected bug"):
        service.process("https://youtu.be/dQw4w9WgXcQ")


def test_statuses_are_typed_and_missing_video_raises(
    tmp_path: Path, mocker: Any
) -> None:
    """Status data is reusable without CLI table formatting."""
    service, _, _, _ = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json(
        "video123",
        "metadata.json",
        {"title": "Example", "channel": "Channel", "duration": 45},
    )
    cache.save_json("video123", "transcript.json", [{"start_sec": 0, "text": "Hi"}])

    statuses = service.list_statuses()
    status = service.get_status("video123")

    assert statuses == [status]
    assert status.title == "Example"
    assert status.transcript_segments == 1
    with pytest.raises(VideoNotFoundError, match="not found"):
        service.get_status("missing")
