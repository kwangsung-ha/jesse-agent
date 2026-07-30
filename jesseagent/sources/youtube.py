"""Expose cached YouTube transcripts as source-neutral knowledge documents."""

from jesseagent.core.cache import LocalCacheManager
from jesseagent.domain.knowledge import KnowledgeDocument
from jesseagent.domain.video import CachedVideo


class YouTubeSourceConnector:
    """Adapt the existing local YouTube cache to the SourceConnector contract."""

    source_id = "youtube"

    def __init__(self, cache: LocalCacheManager) -> None:
        self._cache = cache

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        """Return one transcript document per complete cached video."""
        documents: list[KnowledgeDocument] = []
        for status in self._cache.list_cached_videos():
            if not status.has_metadata or not status.has_transcript:
                continue
            video = self._cache.load_video(status.video_id)
            document = _transcript_document(video)
            if document is not None:
                documents.append(document)
        return tuple(documents)


def _transcript_document(video: CachedVideo) -> KnowledgeDocument | None:
    """Build a timestamp-preserving transcript record when it has usable text."""
    content = "\n".join(
        f"[{_timestamp(segment.start_sec)}] {segment.text}"
        for segment in video.transcript.segments
    )
    if not content:
        return None
    metadata = video.metadata
    return KnowledgeDocument(
        source_id="youtube",
        document_id=f"youtube:{metadata.video_id}",
        uri=metadata.source_url,
        title=metadata.title or f"YouTube video {metadata.video_id}",
        content=content,
        content_hash=KnowledgeDocument.content_digest(content),
        updated_at=metadata.processed_at,
        metadata={
            "video_id": metadata.video_id,
            "channel": metadata.channel,
            "duration_sec": metadata.duration_sec,
            "upload_date": metadata.upload_date,
        },
    )


def _timestamp(seconds: float) -> str:
    """Render a transcript offset without losing its hour component."""
    total_seconds = int(seconds)
    minutes, second = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{second:02d}"
    return f"{minutes:02d}:{second:02d}"
