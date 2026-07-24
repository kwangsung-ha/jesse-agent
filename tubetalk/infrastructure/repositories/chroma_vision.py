"""ChromaDB implementation of the visual-scene index repository port."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chromadb.errors import ChromaError
from pydantic import BaseModel, ConfigDict

from tubetalk.domain.retrieval import RetrievalHit
from tubetalk.domain.state import CacheState
from tubetalk.domain.vision import VisionScene
from tubetalk.infrastructure.repositories.chroma_base import ChromaVectorRepositoryBase
from tubetalk.ports.embedding import EmbeddingProvider
from tubetalk.ports.vision_index_repository import (
    VisionIndexRepositoryError,
    VisionVectorIndexStatus,
)

VISION_VECTOR_SCHEMA_VERSION = 1


class VisionVectorManifest(BaseModel):
    """Records the source scenes and settings of a vision vector index."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    scenes_sha256: str
    embedding_model: str
    embedding_dimension: int
    scene_count: int
    indexed_at: datetime
    collection_name: str = "vision_collection"


class ChromaVisionIndexRepository(ChromaVectorRepositoryBase):
    """Persist explicit visual-scene description vectors in local ChromaDB."""

    collection_name = "vision_collection"

    def __init__(
        self,
        video_id: str,
        data_dir: Optional[Path] = None,
        embedding_model: str = "gemini-embedding-2",
        embedding_dimension: int = 768,
    ) -> None:
        root_dir = data_dir or Path("./data")
        try:
            self._initialize(
                video_id=video_id,
                data_dir=root_dir,
                manifest_filename="vision_vector_manifest.json",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        except (ChromaError, OSError) as error:
            raise VisionIndexRepositoryError(str(error)) from error

    def needs_indexing(self, scenes: tuple[VisionScene, ...]) -> bool:
        return self.get_index_status(scenes).state != "current"

    def get_index_status(
        self, scenes: Optional[tuple[VisionScene, ...]]
    ) -> VisionVectorIndexStatus:
        if not self.manifest_path.is_file():
            return VisionVectorIndexStatus(state=CacheState.MISSING)
        try:
            manifest_data = self._load_manifest_data()
            self._collection_for_manifest(manifest_data)
            manifest_data["indexed_at"] = datetime.fromisoformat(
                manifest_data["indexed_at"]
            )
            manifest = VisionVectorManifest(**manifest_data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return VisionVectorIndexStatus(state=CacheState.INVALID)
        status = VisionVectorIndexStatus(
            state=CacheState.STALE,
            scene_count=manifest.scene_count,
            embedding_model=manifest.embedding_model,
            embedding_dimension=manifest.embedding_dimension,
            indexed_at=manifest.indexed_at,
        )
        if scenes is None:
            return status
        try:
            current = (
                manifest.schema_version == VISION_VECTOR_SCHEMA_VERSION
                and manifest.scenes_sha256 == scenes_sha256(scenes)
                and manifest.embedding_model == self.embedding_model
                and manifest.embedding_dimension == self.embedding_dimension
                and manifest.scene_count == len(scenes)
                and manifest.scene_count == self.count()
            )
        except (ChromaError, OSError, TypeError, ValueError):
            return VisionVectorIndexStatus(state=CacheState.INVALID)
        return (
            VisionVectorIndexStatus(
                state=CacheState.CURRENT,
                scene_count=manifest.scene_count,
                embedding_model=manifest.embedding_model,
                embedding_dimension=manifest.embedding_dimension,
                indexed_at=manifest.indexed_at,
            )
            if current
            else status
        )

    def index_scenes(
        self,
        scenes: tuple[VisionScene, ...],
        title: str,
        embedding_provider: EmbeddingProvider,
    ) -> int:
        self._validate_provider(embedding_provider)
        documents = [format_scene_document(scene, title) for scene in scenes]
        embeddings = embedding_provider.embed_documents(documents)
        self._validate_embeddings(embeddings, len(scenes))
        try:
            previous_collection = self._active_collection_name()
            generation_name, generation = self._create_generation_collection()
            if scenes:
                generation.upsert(
                    ids=[
                        f"{self.video_id}:scene:{index}:description"
                        for index in range(len(scenes))
                    ],
                    documents=[scene.visual_summary for scene in scenes],
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "video_id": self.video_id,
                            "scene_id": index,
                            "vector_type": "description",
                            "start_sec": scene.start_sec,
                            "end_sec": scene.end_sec,
                            "detected_objects": ", ".join(scene.detected_objects),
                            "embedding_model": self.embedding_model,
                            "embedding_dimension": self.embedding_dimension,
                        }
                        for index, scene in enumerate(scenes)
                    ],
                )
            self._save_manifest(scenes, generation_name)
        except (ChromaError, OSError) as error:
            raise VisionIndexRepositoryError(str(error)) from error
        self._collection = generation
        self._retire_collection(previous_collection)
        return len(scenes)

    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        """Search the active visual-scene generation with an explicit vector."""
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError("Query embedding has an unexpected dimension")
        try:
            self._collection_for_manifest(self._load_manifest_data())
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            ids = result["ids"][0]
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            return [
                RetrievalHit(
                    source_id=str(item_id),
                    source="vision",
                    text=str(document),
                    start_sec=float(metadata["start_sec"]),
                    end_sec=float(metadata["end_sec"]),
                    rank=rank,
                    distance=float(distance),
                )
                for rank, (item_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances), start=1
                )
            ]
        except (ChromaError, OSError, KeyError, TypeError, ValueError) as error:
            raise VisionIndexRepositoryError(str(error)) from error

    def _save_manifest(
        self, scenes: tuple[VisionScene, ...], collection_name: str
    ) -> None:
        manifest = VisionVectorManifest(
            schema_version=VISION_VECTOR_SCHEMA_VERSION,
            scenes_sha256=scenes_sha256(scenes),
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            scene_count=len(scenes),
            indexed_at=datetime.now(timezone.utc),
            collection_name=collection_name,
        )
        self._save_manifest_data(manifest.model_dump(mode="json"))


def format_scene_document(scene: VisionScene, title: str) -> str:
    """Format a visual scene according to the Gemini Embedding 2 document form."""
    objects = ", ".join(scene.detected_objects)
    details = (
        f"{scene.visual_summary} Objects: {objects}"
        if objects
        else scene.visual_summary
    )
    return f"title: {title} | text: {details}"


def scenes_sha256(scenes: tuple[VisionScene, ...]) -> str:
    """Return a stable digest of the scene content used for embedding."""
    serialized = json.dumps(
        [scene.model_dump(mode="json") for scene in scenes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
