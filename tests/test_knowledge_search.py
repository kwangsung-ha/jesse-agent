"""Tests for common-knowledge hybrid retrieval and the Agent query tool."""

from typing import Any

from jesseagent.agent.contracts import ToolCall
from jesseagent.agent.tools import VideoToolExecutor
from jesseagent.services.knowledge_search import KnowledgeSearchService


class Provider:
    model = "test"
    dimension = 2

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in documents]

    def embed_query(self, query: str) -> list[float]:
        return [0.0, 1.0]


class Catalog:
    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "obsidian:meal.md:chunk:0",
                "text": "카레 레시피",
                "title": "저녁 식단",
                "uri": "obsidian://open?vault=Jesse&file=meal.md",
                "metadata": {"heading_path": ["카레"]},
            }
        ]


class Vectors:
    def search(self, embedding: list[float], limit: int) -> list[dict[str, object]]:
        return [
            {"chunk_id": "obsidian:meal.md:chunk:0", "text": "카레 레시피"},
            {"chunk_id": "obsidian:notes.md:chunk:0", "text": "재료 메모"},
        ]


def test_knowledge_search_rrf_preserves_obsidian_uri_evidence() -> None:
    results = KnowledgeSearchService(Catalog(), Vectors(), Provider()).search("카레")

    assert [item["chunk_id"] for item in results] == [
        "obsidian:meal.md:chunk:0",
        "obsidian:notes.md:chunk:0",
    ]
    assert results[0]["uri"] == "obsidian://open?vault=Jesse&file=meal.md"
    assert results[0]["score"] > results[1]["score"]


def test_search_knowledge_tool_returns_registered_evidence(mocker: Any) -> None:
    search = mocker.Mock()
    search.search.return_value = [{"chunk_id": "obsidian:note:chunk:0"}]
    executor = VideoToolExecutor(mocker.Mock(), search)

    result = executor.execute(
        ToolCall(name="search_knowledge", arguments={"query": "식단"})
    )

    assert result.ok is True
    assert result.content == {"results": [{"chunk_id": "obsidian:note:chunk:0"}]}
    assert {item["name"] for item in executor.declarations} >= {"search_knowledge"}
