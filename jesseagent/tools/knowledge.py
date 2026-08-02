"""Knowledge tool input and handler over the knowledge search workflow."""

from typing import Any

from pydantic import BaseModel, Field

from jesseagent.application.knowledge.search import KnowledgeSearchService
from jesseagent.application.video.service import VideoServiceError


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=1)


class KnowledgeTools:
    """Translate validated knowledge inputs into application service calls."""

    def __init__(self, service: KnowledgeSearchService | None) -> None:
        self._service = service

    def search_knowledge(self, payload: BaseModel) -> dict[str, Any]:
        if self._service is None:
            raise VideoServiceError("Knowledge search is not configured")
        query = KnowledgeSearchInput.model_validate(payload).query
        return {"results": self._service.search(query)}


KNOWLEDGE_TOOL_DESCRIPTIONS = {
    "search_knowledge": (
        "Search indexed personal knowledge and return URI-backed evidence."
    )
}
