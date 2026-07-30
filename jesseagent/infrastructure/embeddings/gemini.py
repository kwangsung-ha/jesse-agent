"""Gemini Embedding API adapter."""

from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from jesseagent.ports.embedding import EmbeddingProviderError


class GeminiEmbeddingProvider:
    """Generate explicit Gemini Embedding 2 vectors for retrieval."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        client: Optional[Any] = None,
    ) -> None:
        """Create a provider using the supplied Gemini API key."""
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to create transcript embeddings"
            )
        self.model = model
        self.dimension = dimension
        self._client = client or genai.Client(api_key=api_key)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed documents one at a time to preserve one vector per chunk."""
        return [self._embed(document) for document in documents]

    def embed_query(self, query: str) -> list[float]:
        """Embed a future Q&A query in the matching retrieval space."""
        return self._embed(f"task: search result | query: {query}")

    def _embed(self, content: str) -> list[float]:
        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=content,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
        except (APIError, HTTPError) as error:
            raise EmbeddingProviderError(str(error)) from error
        embeddings = response.embeddings
        if not embeddings or len(embeddings) != 1:
            raise ValueError("Gemini embedding response must contain one embedding")
        raw_values = embeddings[0].values
        if not isinstance(raw_values, list):
            raise ValueError("Gemini embedding response must contain vector values")
        values = [float(value) for value in raw_values]
        if len(values) != self.dimension:
            raise ValueError(
                f"Gemini embedding dimension {len(values)} does not match "
                f"configured dimension {self.dimension}"
            )
        return values
