"""Hybrid vector and FTS5 search over common knowledge chunks."""

from collections.abc import Iterable

from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.application.knowledge.contracts import (
    KnowledgeCatalogSearch,
    KnowledgeVectorSearch,
)

RRF_K = 60
SEARCH_LIMIT = 5


class KnowledgeSearchService:
    def __init__(
        self,
        catalog: KnowledgeCatalogSearch,
        vectors: KnowledgeVectorSearch,
        provider: EmbeddingProvider,
    ) -> None:
        self._catalog, self._vectors, self._provider = catalog, vectors, provider

    def search(self, query: str) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        keyword = self._catalog.search(query, SEARCH_LIMIT)
        vector = self._vectors.search(self._provider.embed_query(query), SEARCH_LIMIT)
        details = {item["chunk_id"]: item for item in keyword}
        details.update(
            {
                item["chunk_id"]: item
                for item in vector
                if item["chunk_id"] not in details
            }
        )
        scores = _rrf(keyword, vector)
        return [
            details[chunk_id] | {"score": score}
            for chunk_id, score in scores[:SEARCH_LIMIT]
        ]


def _rrf(*lists: Iterable[dict[str, object]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = str(item["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
