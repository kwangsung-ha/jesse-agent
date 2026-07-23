"""Unit tests for LocalCacheManager."""

import hashlib
import json
from pathlib import Path

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

# ------------------------------------------------------------------
# get_video_dir
# ------------------------------------------------------------------


def test_get_video_dir_creates_directory(tmp_path: Path) -> None:
    """get_video_dir should create and return the video directory."""
    cache = LocalCacheManager(data_dir=tmp_path)
    video_dir = cache.get_video_dir("abc123")
    assert video_dir == tmp_path / "abc123"
    assert video_dir.is_dir()


# ------------------------------------------------------------------
# has_cache
# ------------------------------------------------------------------


def test_has_cache_returns_false_when_empty(tmp_path: Path) -> None:
    """has_cache returns False when no files exist."""
    cache = LocalCacheManager(data_dir=tmp_path)
    (tmp_path / "abc123").mkdir()
    assert cache.has_cache("abc123") is False


def test_has_cache_returns_false_when_partial(tmp_path: Path) -> None:
    """has_cache returns False when only metadata.json exists."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "abc123"
    vdir.mkdir()
    (vdir / "metadata.json").write_text("{}")
    assert cache.has_cache("abc123") is False


def test_has_cache_returns_true_when_both_exist(tmp_path: Path) -> None:
    """has_cache returns True when both metadata.json and transcript.json exist."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "abc123"
    vdir.mkdir()
    (vdir / "metadata.json").write_text("{}")
    (vdir / "transcript.json").write_text("[]")
    assert cache.has_cache("abc123") is True


# ------------------------------------------------------------------
# save_json / load_json
# ------------------------------------------------------------------


def test_save_and_load_json(tmp_path: Path) -> None:
    """save_json persists data that load_json can read back."""
    cache = LocalCacheManager(data_dir=tmp_path)
    data = {"title": "Hello", "duration": 120.5}
    cache.save_json("vid1", "metadata.json", data)

    loaded = cache.load_json("vid1", "metadata.json")
    assert loaded == data


def test_save_json_unicode(tmp_path: Path) -> None:
    """save_json should handle non-ASCII content correctly."""
    cache = LocalCacheManager(data_dir=tmp_path)
    data = [{"text": "안녕하세요", "start_sec": 0.0}]
    cache.save_json("vid1", "transcript.json", data)

    raw = (tmp_path / "vid1" / "transcript.json").read_text()
    assert "안녕하세요" in raw

    loaded = cache.load_json("vid1", "transcript.json")
    assert loaded == data


# ------------------------------------------------------------------
# summary cache
# ------------------------------------------------------------------


def _summary_entry(transcript: list[dict[str, object]]) -> SummaryCacheEntry:
    return SummaryCacheEntry(
        summary=VideoSummary(
            text="영상의 핵심 내용을 설명합니다.",
            chapters=(
                Chapter(start_sec=0, title="소개"),
                Chapter(start_sec=30, title="주요 내용"),
            ),
        ),
        manifest=SummaryManifest(
            schema_version=SUMMARY_SCHEMA_VERSION,
            transcript_sha256=transcript_sha256(transcript),
            model="gemini-3.5-flash-lite",
            prompt_version="summary-chapters-v1",
            language="ko",
            generated_at="2026-07-23T00:00:00+00:00",
        ),
    )


def test_summary_cache_round_trip_and_current_status(tmp_path: Path) -> None:
    """A summary with matching provenance should be reusable."""
    cache = LocalCacheManager(data_dir=tmp_path)
    transcript: list[dict[str, object]] = [
        {"start_sec": 0.0, "duration_sec": 10.0, "text": "안녕하세요"}
    ]
    cache.save_summary("vid1", _summary_entry(transcript))

    status = cache.get_summary_status(
        "vid1",
        transcript,
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v1",
        language="ko",
    )

    assert status.state == "current"
    assert status.entry is not None
    assert status.entry.summary.chapters[1].title == "주요 내용"


def test_summary_cache_marks_changed_inputs_stale(tmp_path: Path) -> None:
    """Changing transcript or generation settings must invalidate a summary."""
    cache = LocalCacheManager(data_dir=tmp_path)
    transcript: list[dict[str, object]] = [
        {"start_sec": 0.0, "duration_sec": 10.0, "text": "안녕하세요"}
    ]
    cache.save_summary("vid1", _summary_entry(transcript))

    stale_by_transcript = cache.get_summary_status(
        "vid1",
        [{"start_sec": 0.0, "duration_sec": 10.0, "text": "변경된 자막"}],
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v1",
        language="ko",
    )
    stale_by_model = cache.get_summary_status(
        "vid1",
        transcript,
        model="other-model",
        prompt_version="summary-chapters-v1",
        language="ko",
    )
    stale_by_prompt = cache.get_summary_status(
        "vid1",
        transcript,
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v2",
        language="ko",
    )
    stale_by_language = cache.get_summary_status(
        "vid1",
        transcript,
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v1",
        language="en",
    )

    assert [
        stale_by_transcript.state,
        stale_by_model.state,
        stale_by_prompt.state,
        stale_by_language.state,
    ] == ["stale", "stale", "stale", "stale"]


def test_summary_cache_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    """Missing and malformed summary caches must be distinguished."""
    cache = LocalCacheManager(data_dir=tmp_path)
    transcript: list[dict[str, object]] = []

    missing = cache.get_summary_status(
        "vid1",
        transcript,
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v1",
        language="ko",
    )
    cache.save_json("vid1", "summary.json", {"summary": "broken"})
    invalid = cache.get_summary_status(
        "vid1",
        transcript,
        model="gemini-3.5-flash-lite",
        prompt_version="summary-chapters-v1",
        language="ko",
    )

    assert missing.state == "missing"
    assert invalid.state == "invalid"


def test_vision_index_cache_round_trip_and_freshness(tmp_path: Path) -> None:
    """Vision cache reuse must be tied to its URL, model, and prompt version."""
    cache = LocalCacheManager(data_dir=tmp_path)
    entry = VisionIndexEntry(
        scenes=(VisionScene(0, 5, "A presenter appears.", ("presenter",)),),
        manifest=VisionManifest(
            schema_version=VISION_SCHEMA_VERSION,
            source_url="https://www.youtube.com/watch?v=vid1",
            model="gemini-3.5-flash",
            prompt_version="vision-scenes-v1",
            generated_at="2026-07-24T00:00:00+00:00",
        ),
    )
    cache.save_vision_index("vid1", entry)

    current = cache.get_vision_index_status(
        "vid1",
        source_url="https://www.youtube.com/watch?v=vid1",
        model="gemini-3.5-flash",
        prompt_version="vision-scenes-v1",
    )
    stale = cache.get_vision_index_status(
        "vid1",
        source_url="https://www.youtube.com/watch?v=vid1",
        model="gemini-2.5-flash",
        prompt_version="vision-scenes-v1",
    )

    assert current.state == "current"
    assert current.entry == entry
    assert stale.state == "stale"


def test_video_status_includes_current_summary_metadata(tmp_path: Path) -> None:
    """Status should expose summary freshness and generation provenance."""
    cache = LocalCacheManager(data_dir=tmp_path)
    transcript: list[dict[str, object]] = [
        {"start_sec": 0.0, "duration_sec": 10.0, "text": "안녕하세요"}
    ]
    cache.save_json("vid1", "metadata.json", {"title": "예시"})
    cache.save_json("vid1", "transcript.json", transcript)
    cache.save_summary("vid1", _summary_entry(transcript))

    status = cache.get_video_status("vid1")

    assert status is not None
    assert status.summary_state == "current"
    assert status.summary_chapters == 2
    assert status.summary_model == "gemini-3.5-flash-lite"
    assert status.summary_language == "ko"


# ------------------------------------------------------------------
# list_cached_videos
# ------------------------------------------------------------------


def test_list_cached_videos_empty(tmp_path: Path) -> None:
    """list_cached_videos returns empty list when data dir is empty."""
    cache = LocalCacheManager(data_dir=tmp_path)
    assert cache.list_cached_videos() == []


def test_list_cached_videos_nonexistent_dir(tmp_path: Path) -> None:
    """list_cached_videos returns empty list when data dir doesn't exist."""
    cache = LocalCacheManager(data_dir=tmp_path / "nonexistent")
    assert cache.list_cached_videos() == []


def test_list_cached_videos_with_data(tmp_path: Path) -> None:
    """list_cached_videos returns entries for cached video directories."""
    cache = LocalCacheManager(data_dir=tmp_path)

    # Create two video caches
    for vid in ("vid_a", "vid_b"):
        vdir = tmp_path / vid
        vdir.mkdir()
        (vdir / "metadata.json").write_text(
            json.dumps({"title": f"Title {vid}", "channel": "ch", "duration": 60})
        )
        (vdir / "transcript.json").write_text(
            json.dumps([{"start_sec": 0, "text": "hi"}])
        )

    results = cache.list_cached_videos()
    assert len(results) == 2
    assert results[0].video_id == "vid_a"
    assert results[1].video_id == "vid_b"


# ------------------------------------------------------------------
# get_video_status
# ------------------------------------------------------------------


def test_get_video_status_nonexistent(tmp_path: Path) -> None:
    """get_video_status returns None for missing video_id."""
    cache = LocalCacheManager(data_dir=tmp_path)
    assert cache.get_video_status("no_such_video") is None


def test_get_video_status_full(tmp_path: Path) -> None:
    """get_video_status returns rich status when cache files exist."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid1"
    vdir.mkdir()
    (vdir / "metadata.json").write_text(
        json.dumps({"title": "Test Video", "channel": "MyChannel", "duration": 180.0})
    )
    (vdir / "transcript.json").write_text(
        json.dumps(
            [
                {"start_sec": 0, "text": "A"},
                {"start_sec": 5, "text": "B"},
                {"start_sec": 10, "text": "C"},
            ]
        )
    )
    (vdir / "vision_index.json").write_text("[]")

    status = cache.get_video_status("vid1")
    assert status is not None
    assert status.video_id == "vid1"
    assert status.title == "Test Video"
    assert status.channel == "MyChannel"
    assert status.duration == 180.0
    assert status.has_metadata is True
    assert status.has_transcript is True
    assert status.has_vision_index is True
    assert status.transcript_segments == 3
    assert status.transcript_index_state == "missing"
    assert status.cached_at is not None


def test_get_video_status_includes_current_transcript_index(tmp_path: Path) -> None:
    """Status should expose manifest metadata for a current text index."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid_indexed"
    vdir.mkdir()
    transcript = [{"start_sec": 0, "text": "Hello"}]
    (vdir / "transcript.json").write_text(json.dumps(transcript))
    digest = hashlib.sha256(
        json.dumps(
            transcript,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (vdir / "index_manifest.json").write_text(
        json.dumps(
            {
                "transcript_sha256": digest,
                "chunk_count": 1,
                "embedding_model": "gemini-embedding-2",
                "embedding_dimension": 768,
                "indexed_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )

    status = cache.get_video_status("vid_indexed")

    assert status is not None
    assert status.transcript_index_state == "current"
    assert status.transcript_index_chunks == 1
    assert status.transcript_index_model == "gemini-embedding-2"
    assert status.transcript_index_dimension == 768
    assert status.transcript_indexed_at == "2026-07-23T00:00:00+00:00"


def test_get_video_status_marks_changed_transcript_index_stale(tmp_path: Path) -> None:
    """A manifest for another transcript must not be shown as current."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid_stale"
    vdir.mkdir()
    (vdir / "transcript.json").write_text(json.dumps([{"start_sec": 0, "text": "New"}]))
    (vdir / "index_manifest.json").write_text(
        json.dumps({"transcript_sha256": "outdated", "chunk_count": 2})
    )

    status = cache.get_video_status("vid_stale")

    assert status is not None
    assert status.transcript_index_state == "stale"
    assert status.transcript_index_chunks == 2


def test_get_video_status_marks_another_embedding_model_stale(tmp_path: Path) -> None:
    """A manifest built with another vector space must be marked stale."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid_old_model"
    vdir.mkdir()
    transcript = [{"start_sec": 0, "text": "Hello"}]
    (vdir / "transcript.json").write_text(json.dumps(transcript))
    digest = hashlib.sha256(
        json.dumps(
            transcript,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (vdir / "index_manifest.json").write_text(
        json.dumps(
            {
                "transcript_sha256": digest,
                "chunk_count": 1,
                "embedding_model": "gemini-embedding-001",
                "embedding_dimension": 768,
            }
        )
    )

    status = cache.get_video_status("vid_old_model")

    assert status is not None
    assert status.transcript_index_state == "stale"


def test_get_video_status_partial(tmp_path: Path) -> None:
    """get_video_status works with only a directory (no JSON files)."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid2"
    vdir.mkdir()

    status = cache.get_video_status("vid2")
    assert status is not None
    assert status.has_metadata is False
    assert status.has_transcript is False
    assert status.has_vision_index is False
    assert status.transcript_segments == 0
    assert status.title is None
    assert status.cached_at is None


def test_get_video_status_corrupt_json(tmp_path: Path) -> None:
    """get_video_status handles corrupt JSON files gracefully."""
    cache = LocalCacheManager(data_dir=tmp_path)
    vdir = tmp_path / "vid3"
    vdir.mkdir()
    (vdir / "metadata.json").write_text("NOT_VALID_JSON")
    (vdir / "transcript.json").write_text("NOT_VALID_JSON")

    status = cache.get_video_status("vid3")
    assert status is not None
    assert status.has_metadata is True
    assert status.title is None
    assert status.transcript_segments == 0
