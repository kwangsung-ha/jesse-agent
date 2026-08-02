"""Tests for source-neutral knowledge records and the YouTube adapter."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jesseagent.domain.knowledge import KnowledgeDocument
from jesseagent.domain.transcript import Transcript, TranscriptSegment
from jesseagent.domain.video import CachedVideo, VideoMetadata
from jesseagent.infrastructure.local.video_cache import LocalCacheManager
from jesseagent.infrastructure.youtube.cached_source import YouTubeSourceConnector


def _video(video_id: str, text: str = "첫 번째 문장") -> CachedVideo:
    return CachedVideo(
        metadata=VideoMetadata(
            video_id=video_id,
            source_url=f"https://youtu.be/{video_id}",
            title="테스트 영상",
            channel="Jesse",
            duration_sec=75,
            upload_date="20260730",
            processed_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        ),
        transcript=Transcript(
            segments=(
                TranscriptSegment(start_sec=5, duration_sec=3, text=text),
                TranscriptSegment(start_sec=65, duration_sec=3, text="두 번째 문장"),
            )
        ),
    )


def test_knowledge_document_requires_a_sha256_content_hash() -> None:
    """A connector cannot create an untrackable document."""
    with pytest.raises(ValueError, match="SHA-256"):
        KnowledgeDocument(
            source_id="test",
            document_id="test:one",
            uri="test://one",
            title="One",
            content="content",
            content_hash="not-a-digest",
        )


def test_youtube_connector_projects_cached_transcripts(tmp_path: Path) -> None:
    """Cached YouTube data is available through the common source contract."""
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_video(_video("video-1"))

    documents = YouTubeSourceConnector(cache).list_documents()

    assert len(documents) == 1
    document = documents[0]
    assert document.source_id == "youtube"
    assert document.document_id == "youtube:video-1"
    assert document.uri == "https://youtu.be/video-1"
    assert document.content == "[00:05] 첫 번째 문장\n[01:05] 두 번째 문장"
    assert document.content_hash == KnowledgeDocument.content_digest(document.content)
    assert document.metadata == {
        "video_id": "video-1",
        "channel": "Jesse",
        "duration_sec": 75.0,
        "upload_date": "20260730",
    }


def test_youtube_connector_skips_partial_or_empty_caches(tmp_path: Path) -> None:
    """Only complete, indexable video caches become knowledge documents."""
    cache = LocalCacheManager(data_dir=tmp_path)
    cache.save_json("partial", "metadata.json", {"video_id": "partial"})
    empty = _video("empty")
    cache.save_video(empty.model_copy(update={"transcript": Transcript(segments=())}))

    assert YouTubeSourceConnector(cache).list_documents() == ()
