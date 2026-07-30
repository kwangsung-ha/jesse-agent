"""Source-neutral knowledge records used by every JesseAgent connector."""

from datetime import datetime
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class KnowledgeDocument(BaseModel):
    """One complete, source-owned record eligible for indexing."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    uri: str
    title: str
    content: str
    content_hash: str
    updated_at: datetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("source_id", "document_id", "uri", "title", "content")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Knowledge document text fields must not be empty")
        return value.strip()

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("content_hash must be a lowercase SHA-256 digest")
        return value

    @classmethod
    def content_digest(cls, content: str) -> str:
        """Return the stable SHA-256 digest used for incremental indexing."""
        return sha256(content.encode("utf-8")).hexdigest()


class KnowledgeChunk(BaseModel):
    """A searchable portion of one knowledge document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Knowledge chunk text fields must not be empty")
        return value.strip()
