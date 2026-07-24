"""ChromaDB implementation of the visual-scene index repository port."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.errors import ChromaError

from tubetalk.domain.state import CacheState
from tubetalk.domain.vision import VisionScene
from tubetalk.ports.embedding import EmbeddingProvider
from tubetalk.ports.vision_index_repository import (
    VisionIndexRepositoryError,
    VisionVectorIndexStatus,
)

VISION_VECTOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VisionVectorManifest:
    """Records the source scenes and settings of a vision vector index."""

    schema_version: int
    scenes_sha256: str
    embedding_model: str
    embedding_dimension: int
    scene_count: int
    indexed_at: datetime


class ChromaVisionIndexRepository:
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
        self.video_id = video_id
        self.path = root_dir / video_id / "chromadb"
        self.manifest_path = root_dir / video_id / "vision_vector_manifest.json"
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._create_collection()
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
            manifest_data = json.loads(self.manifest_path.read_text())
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
        if (
            embedding_provider.model != self.embedding_model
            or embedding_provider.dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding provider settings do not match the vector repository"
            )
        documents = [format_scene_document(scene, title) for scene in scenes]
        embeddings = embedding_provider.embed_documents(documents)
        if len(embeddings) != len(scenes) or any(
            len(vector) != self.embedding_dimension for vector in embeddings
        ):
            raise ValueError("Embedding provider returned an unexpected vector shape")
        try:
            self._collection = self._recreate_collection()
            if scenes:
                self._collection.upsert(
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
            self._save_manifest(scenes)
        except (ChromaError, OSError) as error:
            raise VisionIndexRepositoryError(str(error)) from error
        return len(scenes)

    def count(self) -> int:
        return int(self._collection.count())

    def _create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"video_id": self.video_id, "hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _recreate_collection(self) -> Any:
        self._client.delete_collection(name=self.collection_name)
        return self._create_collection()

    def _save_manifest(self, scenes: tuple[VisionScene, ...]) -> None:
        manifest = VisionVectorManifest(
            schema_version=VISION_VECTOR_SCHEMA_VERSION,
            scenes_sha256=scenes_sha256(scenes),
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            scene_count=len(scenes),
            indexed_at=datetime.now(timezone.utc),
        )
        payload = asdict(manifest)
        payload["indexed_at"] = manifest.indexed_at.isoformat()
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


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
        [asdict(scene) for scene in scenes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
