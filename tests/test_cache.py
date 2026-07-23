"""Unit tests for LocalCacheManager."""

import hashlib
import json
from pathlib import Path

from tubetalk.core.cache import LocalCacheManager

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
