"""Unit tests for Gemini-backed local visual-scene vector storage."""

import json
from pathlib import Path
from typing import Any

import pytest

from tubetalk.domain.vision import VisionScene
from tubetalk.infrastructure.repositories.chroma_vision import (
    ChromaVisionIndexRepository,
    format_scene_document,
)
from tubetalk.ports.vision_index_repository import VisionIndexRepositoryError


class FakeEmbeddingProvider:
    """Deterministic explicit-vector provider used without an external API."""

    model = "gemini-embedding-2"
    dimension = 3

    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        self.documents = documents
        return [[float(index), 0.0, 1.0] for index in range(len(documents))]


def _make_store(
    tmp_path: Path, mocker: Any
) -> tuple[ChromaVisionIndexRepository, Any, Any]:
    collection = mocker.Mock()
    collection.count.return_value = 0
    client = mocker.Mock()
    client.get_or_create_collection.return_value = collection
    mocker.patch(
        "tubetalk.infrastructure.repositories.chroma_base.chromadb.PersistentClient",
        return_value=client,
    )
    store = ChromaVisionIndexRepository(
        "video123", data_dir=tmp_path, embedding_dimension=3
    )
    return store, client, collection


def _scenes() -> tuple[VisionScene, ...]:
    return (
        VisionScene(0, 5, "A presenter appears.", ("presenter",)),
        VisionScene(5, 10, "A chart appears.", ("chart",)),
    )


def test_creates_video_scoped_vision_collection(tmp_path: Path, mocker: Any) -> None:
    """The vision collection shares the video's local Chroma database."""
    store, client, _ = _make_store(tmp_path, mocker)

    assert store.path == tmp_path / "video123" / "chromadb"
    client.get_or_create_collection.assert_called_once_with(
        name="vision_collection",
        metadata={"video_id": "video123", "hnsw:space": "cosine"},
        embedding_function=None,
    )


def test_index_scenes_rebuilds_collection_and_writes_manifest(
    tmp_path: Path, mocker: Any
) -> None:
    """Each description is stored with an explicit vector and scene metadata."""
    store, client, collection = _make_store(tmp_path, mocker)
    provider = FakeEmbeddingProvider()

    assert store.index_scenes(_scenes(), "Example", provider) == 2
    assert provider.documents == [
        "title: Example | text: A presenter appears. Objects: presenter",
        "title: Example | text: A chart appears. Objects: chart",
    ]
    client.delete_collection.assert_called_once_with(name="vision_collection")
    collection.upsert.assert_called_once_with(
        ids=["video123:scene:0:description", "video123:scene:1:description"],
        documents=["A presenter appears.", "A chart appears."],
        embeddings=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        metadatas=[
            {
                "video_id": "video123",
                "scene_id": 0,
                "vector_type": "description",
                "start_sec": 0,
                "end_sec": 5,
                "detected_objects": "presenter",
                "embedding_model": "gemini-embedding-2",
                "embedding_dimension": 3,
            },
            {
                "video_id": "video123",
                "scene_id": 1,
                "vector_type": "description",
                "start_sec": 5,
                "end_sec": 10,
                "detected_objects": "chart",
                "embedding_model": "gemini-embedding-2",
                "embedding_dimension": 3,
            },
        ],
    )
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["scene_count"] == 2
    assert manifest["embedding_model"] == "gemini-embedding-2"
    assert manifest["collection_name"].startswith("vision_collection__")


def test_failed_generation_keeps_previous_vision_index_active(
    tmp_path: Path, mocker: Any
) -> None:
    """A failed generation cannot replace or delete the active scene index."""
    store, client, collection = _make_store(tmp_path, mocker)
    store.manifest_path.write_text('{"collection_name": "vision_collection"}')
    collection.upsert.side_effect = OSError("write failed")

    with pytest.raises(VisionIndexRepositoryError, match="write failed"):
        store.index_scenes(_scenes(), "Example", FakeEmbeddingProvider())

    assert json.loads(store.manifest_path.read_text()) == {
        "collection_name": "vision_collection"
    }
    client.delete_collection.assert_not_called()


def test_index_status_detects_current_stale_missing_and_invalid_indexes(
    tmp_path: Path, mocker: Any
) -> None:
    """A manifest is current only when its source scenes and collection match."""
    store, _, collection = _make_store(tmp_path, mocker)
    missing = store.get_index_status(_scenes())
    store.index_scenes(_scenes(), "Example", FakeEmbeddingProvider())
    collection.count.return_value = 2
    current = store.get_index_status(_scenes())
    stale = store.get_index_status((VisionScene(0, 5, "Changed", ()),))
    store.manifest_path.write_text("not json")
    invalid = store.get_index_status(_scenes())

    assert missing.state == "missing"
    assert current.state == "current"
    assert current.scene_count == 2
    assert stale.state == "stale"
    assert invalid.state == "invalid"


def test_index_scenes_rejects_wrong_provider_shape(tmp_path: Path, mocker: Any) -> None:
    """Vectors from another configured space cannot enter the collection."""
    store, _, _ = _make_store(tmp_path, mocker)
    provider = FakeEmbeddingProvider()
    provider.dimension = 4

    with pytest.raises(ValueError, match="settings do not match"):
        store.index_scenes(_scenes(), "Example", provider)


def test_scene_document_omits_empty_object_suffix() -> None:
    """A scene without object labels still has a valid retrieval document."""
    assert format_scene_document(VisionScene(0, 1, "A title card.", ()), "Example") == (
        "title: Example | text: A title card."
    )
