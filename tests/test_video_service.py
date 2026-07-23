"""Unit tests for interface-independent video application services."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tubetalk.core.cache import LocalCacheManager
from tubetalk.domain.summary import (
    SUMMARY_SCHEMA_VERSION,
    Chapter,
    SummaryCacheEntry,
    SummaryManifest,
    VideoSummary,
)
from tubetalk.domain.transcript_index import transcript_sha256
from tubetalk.domain.vision import (
    VISION_SCHEMA_VERSION,
    VisionIndexEntry,
    VisionManifest,
    VisionScene,
)
from tubetalk.ports.summary import SummaryProviderError
from tubetalk.ports.transcript_index_repository import TranscriptIndexStatus
from tubetalk.ports.vision import VisionProviderError
from tubetalk.services.video_service import (
    InvalidVideoUrlError,
    SummaryGenerationError,
    SummaryUnavailableError,
    VideoIngestionError,
    VideoNotFoundError,
    VideoService,
)


def _service(tmp_path: Path, mocker: Any) -> tuple[VideoService, Any, Any, Any, Any]:
    cache = LocalCacheManager(data_dir=tmp_path)
    loader = mocker.Mock()
    store = mocker.Mock()
    store.get_index_status.return_value = TranscriptIndexStatus(state="missing")
    provider_factory = mocker.Mock(return_value=mocker.Mock())
    summary_provider = mocker.Mock()
    summary_provider.generate_summary.return_value = VideoSummary(
        text="요약입니다.", chapters=(Chapter(start_sec=0, title="소개"),)
    )
    summary_provider_factory = mocker.Mock(return_value=summary_provider)
    vision_analyzer = mocker.Mock()
    vision_analyzer.describe.return_value = (
        VisionScene(0, 5, "A presenter appears.", ("presenter",)),
    )
    vision_store = mocker.Mock()
    vision_store.needs_indexing.return_value = True
    vision_store.index_scenes.return_value = 1
    service = VideoService(
        cache=cache,
        loader=loader,
        embedding_provider_factory=provider_factory,
        transcript_index_repository_factory=mocker.Mock(return_value=store),
        summary_provider_factory=summary_provider_factory,
        vision_analyzer_factory=mocker.Mock(return_value=vision_analyzer),
        vision_index_repository_factory=mocker.Mock(return_value=vision_store),
        summary_model="gemini-3.5-flash-lite",
        summary_prompt_version="summary-chapters-v1",
        summary_language="ko",
        vision_model="gemini-3.5-flash",
        vision_prompt_version="vision-scenes-v1",
    )
    return service, loader, store, provider_factory, summary_provider_factory


def test_process_cache_miss_saves_resources_and_indexes(
    tmp_path: Path, mocker: Any
) -> None:
    """A cache miss coordinates collection, persistence, and indexing."""
    service, loader, store, provider_factory, summary_provider_factory = _service(
        tmp_path, mocker
    )
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
    assert provider_factory.call_count == 2
    summary_provider_factory.assert_called_once_with()
    assert result.summary.state == "generated"
    assert result.vision.state == "generated"
    assert result.vision.scene_count == 1
    assert result.vision.indexing.state == "indexed"


def test_process_reuses_current_vision_index_without_provider_call(
    tmp_path: Path, mocker: Any
) -> None:
    """A current visual index should not trigger a second Gemini request."""
    service, loader, store, _, _ = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json(
        "dQw4w9WgXcQ",
        "metadata.json",
        {
            "title": "Example",
            "source_url": "https://youtu.be/dQw4w9WgXcQ",
        },
    )
    cache.save_json(
        "dQw4w9WgXcQ", "transcript.json", [{"start_sec": 0, "text": "Hello"}]
    )
    cache.save_vision_index(
        "dQw4w9WgXcQ",
        VisionIndexEntry(
            scenes=(VisionScene(0, 5, "A presenter appears.", ("presenter",)),),
            manifest=VisionManifest(
                schema_version=VISION_SCHEMA_VERSION,
                source_url="https://youtu.be/dQw4w9WgXcQ",
                model="gemini-3.5-flash",
                prompt_version="vision-scenes-v1",
                generated_at="2026-07-24T00:00:00+00:00",
            ),
        ),
    )
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    store.needs_indexing.return_value = False

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.vision.state == "current"
    service._vision_analyzer_factory.assert_not_called()


def test_process_keeps_text_cache_when_vision_generation_fails(
    tmp_path: Path, mocker: Any
) -> None:
    """Vision-provider errors must not discard transcript or summary resources."""
    service, loader, store, _, _ = _service(tmp_path, mocker)
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.return_value = {"title": "Example"}
    loader.fetch_transcript.return_value = [{"start_sec": 0, "text": "Hello"}]
    store.needs_indexing.return_value = False
    service._vision_analyzer_factory.return_value.describe.side_effect = (
        VisionProviderError("video is unavailable")
    )

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.vision.state == "warning"
    assert result.vision.warning == "video is unavailable"
    assert (tmp_path / "dQw4w9WgXcQ" / "transcript.json").is_file()


def test_process_cache_hit_skips_remote_loading_and_keeps_current_index(
    tmp_path: Path, mocker: Any
) -> None:
    """A complete cache is reused without another loader call."""
    service, loader, store, provider_factory, summary_provider_factory = _service(
        tmp_path, mocker
    )
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
    summary_provider_factory.assert_called_once_with()
    assert result.summary.state == "generated"


def test_process_reuses_current_summary_without_provider_call(
    tmp_path: Path, mocker: Any
) -> None:
    """A current summary manifest avoids another Gemini request."""
    service, loader, store, _, summary_provider_factory = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    transcript = [{"start_sec": 0, "text": "Hello"}]
    cache.save_json("dQw4w9WgXcQ", "metadata.json", {"title": "Example"})
    cache.save_json("dQw4w9WgXcQ", "transcript.json", transcript)
    cached_summary = VideoSummary(
        text="캐시된 요약입니다.", chapters=(Chapter(start_sec=0, title="소개"),)
    )
    cache.save_summary(
        "dQw4w9WgXcQ",
        SummaryCacheEntry(
            summary=cached_summary,
            manifest=SummaryManifest(
                schema_version=SUMMARY_SCHEMA_VERSION,
                transcript_sha256=transcript_sha256(transcript),
                model="gemini-3.5-flash-lite",
                prompt_version="summary-chapters-v1",
                language="ko",
                generated_at="2026-07-23T00:00:00+00:00",
            ),
        ),
    )
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    store.needs_indexing.return_value = False

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.summary.state == "current"
    assert result.summary.summary == cached_summary
    summary_provider_factory.assert_not_called()


def test_process_keeps_cache_when_summary_generation_fails(
    tmp_path: Path, mocker: Any
) -> None:
    """Summary failures are warnings and leave the collected cache intact."""
    service, loader, store, _, summary_provider_factory = _service(tmp_path, mocker)
    loader.extract_video_id.return_value = "dQw4w9WgXcQ"
    loader.fetch_metadata.return_value = {"title": "Example"}
    loader.fetch_transcript.return_value = [{"start_sec": 0, "text": "Hello"}]
    store.needs_indexing.return_value = False
    summary_provider_factory.side_effect = SummaryProviderError("Gemini unavailable")

    result = service.process("https://youtu.be/dQw4w9WgXcQ")

    assert result.summary.state == "warning"
    assert result.summary.warning == "Gemini unavailable"
    assert LocalCacheManager(data_dir=tmp_path).has_cache("dQw4w9WgXcQ") is True
    assert not (tmp_path / "dQw4w9WgXcQ" / "summary.json").exists()


def test_get_summary_requires_generate_for_missing_cache(
    tmp_path: Path, mocker: Any
) -> None:
    """The summary use case does not call Gemini without explicit permission."""
    service, _, _, _, summary_provider_factory = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json("video123", "metadata.json", {"title": "Example"})
    cache.save_json("video123", "transcript.json", [{"start_sec": 0, "text": "Hi"}])

    with pytest.raises(SummaryUnavailableError, match="--generate"):
        service.get_summary("video123")

    result = service.get_summary("video123", generate=True)

    assert result.state == "generated"
    assert result.summary is not None
    summary_provider_factory.assert_called_once_with()


def test_get_summary_maps_generation_warning_to_service_error(
    tmp_path: Path, mocker: Any
) -> None:
    """Explicit generation exposes provider failures to the CLI."""
    service, _, _, _, summary_provider_factory = _service(tmp_path, mocker)
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json("video123", "metadata.json", {"title": "Example"})
    cache.save_json("video123", "transcript.json", [{"start_sec": 0, "text": "Hi"}])
    summary_provider_factory.side_effect = SummaryProviderError("Gemini unavailable")

    with pytest.raises(SummaryGenerationError, match="Gemini unavailable"):
        service.get_summary("video123", generate=True)


def test_process_keeps_cache_when_indexing_fails(tmp_path: Path, mocker: Any) -> None:
    """Embedding failures are returned as warnings after JSON is persisted."""
    service, loader, store, provider_factory, _ = _service(tmp_path, mocker)
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
    service, loader, _, _, _ = _service(tmp_path, mocker)
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
    service, loader, store, _, _ = _service(tmp_path, mocker)
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
    service, _, _, _, _ = _service(tmp_path, mocker)
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
