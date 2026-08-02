"""Unit tests for Gemini-backed local transcript vector storage."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jesseagent.application.video.contracts import TranscriptIndexRepositoryError
from jesseagent.domain.transcript import Transcript, TranscriptSegment
from jesseagent.domain.transcript_index import (
    CHUNK_POLICY_VERSION,
    TranscriptChunkPolicy,
    chunk_transcript,
    format_document,
)
from jesseagent.infrastructure.embeddings.gemini import GeminiEmbeddingProvider
from jesseagent.infrastructure.repositories.chroma_transcript import (
    ChromaTranscriptIndexRepository,
)


class FakeEmbeddingProvider:
    """Deterministic embedding provider used without an external API."""

    model = "gemini-embedding-2"
    dimension = 3

    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        self.documents = documents
        return [[float(index), 0.0, 1.0] for index in range(len(documents))]


def _transcript(*segments: tuple[float, float, str]) -> Transcript:
    return Transcript(
        segments=tuple(
            TranscriptSegment(start_sec=start, duration_sec=duration, text=text)
            for start, duration, text in segments
        )
    )


def _make_store(
    tmp_path: Path, mocker: Any
) -> tuple[ChromaTranscriptIndexRepository, Any, Any]:
    """Create a store with its Chroma client replaced by a mock."""
    collection = mocker.Mock()
    collection.count.return_value = 0
    client = mocker.Mock()
    client.get_or_create_collection.return_value = collection
    persistent_client = mocker.patch(
        "jesseagent.infrastructure.repositories.chroma_base.chromadb.PersistentClient",
        return_value=client,
    )

    store = ChromaTranscriptIndexRepository(
        "video123",
        data_dir=tmp_path,
        embedding_dimension=3,
    )
    persistent_client.assert_called_once_with(
        path=str(tmp_path / "video123" / "chromadb")
    )
    client.get_or_create_collection.assert_called_once_with(
        name="transcript_collection",
        metadata={"video_id": "video123", "hnsw:space": "cosine"},
        embedding_function=None,
    )
    return store, client, collection


def test_creates_video_scoped_chroma_directory(tmp_path: Path, mocker: Any) -> None:
    """The vector database path should live below a video's local cache."""
    store, _, _ = _make_store(tmp_path, mocker)

    assert store.path == tmp_path / "video123" / "chromadb"
    assert store.path.is_dir()


def test_chunk_transcript_splits_by_duration_and_preserves_boundaries() -> None:
    """Chunks should stop before exceeding the configured duration."""
    chunks = chunk_transcript(
        _transcript((0, 20, "One"), (20, 20, "Two"), (40, 20, "Three")),
        policy=TranscriptChunkPolicy(max_seconds=45, max_characters=1200),
    )

    assert chunks[0].text == "One Two"
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 40.0
    assert chunks[0].first_segment_index == 0
    assert chunks[0].last_segment_index == 1
    assert chunks[1].text == "Three"
    assert chunks[1].start_sec == 40.0
    assert chunks[1].end_sec == 60.0


def test_chunk_transcript_splits_by_characters_without_overlap() -> None:
    """Character limits should create separate, non-duplicated chunks."""
    chunks = chunk_transcript(
        _transcript((0, 0, "alpha"), (1, 0, "bravo"), (2, 0, "charlie")),
        policy=TranscriptChunkPolicy(max_seconds=45, max_characters=11),
    )

    assert [chunk.text for chunk in chunks] == ["alpha bravo", "charlie"]
    assert [chunk.first_segment_index for chunk in chunks] == [0, 2]


@pytest.mark.parametrize("args", [(0, 0, ""), (-1, 0, "Hello"), (0, -1, "Hello")])
def test_chunk_transcript_rejects_invalid_segments(
    args: tuple[float, float, str],
) -> None:
    """Invalid source segments should never reach the embedding provider."""
    with pytest.raises(ValueError):
        _transcript(args)


def test_gemini_provider_formats_query_and_validates_vector_dimension(
    mocker: Any,
) -> None:
    """Gemini calls should use explicit 768-style dimension configuration."""
    client = mocker.Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimension=3,
        client=client,
    )

    assert provider.embed_query("When is the goal?") == [0.1, 0.2, 0.3]
    _, kwargs = client.models.embed_content.call_args
    assert kwargs["model"] == "gemini-embedding-2"
    assert kwargs["contents"] == "task: search result | query: When is the goal?"
    assert kwargs["config"].output_dimensionality == 3


def test_index_transcript_rebuilds_collection_and_writes_manifest(
    tmp_path: Path, mocker: Any
) -> None:
    """A successful index stores explicit vectors and a current manifest."""
    store, client, collection = _make_store(tmp_path, mocker)
    provider = FakeEmbeddingProvider()
    segments = _transcript((0, 2.5, "Hello"), (2.5, 3, "World"))

    assert store.index_transcript(segments, "Example video", provider) == 1
    assert provider.documents == [format_document("Hello World", "Example video")]
    client.delete_collection.assert_called_once_with(name="transcript_collection")
    collection.upsert.assert_called_once_with(
        ids=["video123:chunk:0"],
        documents=["Hello World"],
        embeddings=[[0.0, 0.0, 1.0]],
        metadatas=[
            {
                "video_id": "video123",
                "chunk_index": 0,
                "start_sec": 0.0,
                "end_sec": 5.5,
                "first_segment_index": 0,
                "last_segment_index": 1,
                "embedding_model": "gemini-embedding-2",
                "embedding_dimension": 3,
            }
        ],
    )
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["embedding_model"] == "gemini-embedding-2"
    assert manifest["chunk_policy_version"] == CHUNK_POLICY_VERSION
    assert manifest["chunk_count"] == 1
    assert manifest["collection_name"].startswith("transcript_collection__")


def test_failed_generation_keeps_previous_transcript_index_active(
    tmp_path: Path, mocker: Any
) -> None:
    """A failed manifest switch must not delete the collection named by it."""
    store, client, collection = _make_store(tmp_path, mocker)
    store.manifest_path.write_text('{"collection_name": "transcript_collection"}')
    mocker.patch(
        "jesseagent.infrastructure.repositories.chroma_base.os.replace",
        side_effect=OSError("write failed"),
    )

    with pytest.raises(TranscriptIndexRepositoryError, match="write failed"):
        store.index_transcript(
            _transcript((0, 0, "Hello")), "Example", FakeEmbeddingProvider()
        )

    assert json.loads(store.manifest_path.read_text()) == {
        "collection_name": "transcript_collection"
    }
    client.delete_collection.assert_not_called()


def test_needs_indexing_detects_current_and_stale_transcripts(
    tmp_path: Path, mocker: Any
) -> None:
    """The manifest fingerprint should skip only a matching complete index."""
    store, _, collection = _make_store(tmp_path, mocker)
    provider = FakeEmbeddingProvider()
    segments = _transcript((0, 0, "Hello"))

    assert store.needs_indexing(segments) is True
    store.index_transcript(segments, "Example", provider)
    collection.count.return_value = 1
    assert store.needs_indexing(segments) is False
    assert store.needs_indexing(_transcript((0, 0, "Changed"))) is True


def test_index_transcript_rejects_provider_with_wrong_configuration(
    tmp_path: Path, mocker: Any
) -> None:
    """Vectors from another model or dimension cannot enter this collection."""
    store, _, _ = _make_store(tmp_path, mocker)
    provider = FakeEmbeddingProvider()
    provider.dimension = 4

    with pytest.raises(ValueError, match="settings do not match"):
        store.index_transcript(_transcript((0, 0, "Hello")), "Example", provider)
