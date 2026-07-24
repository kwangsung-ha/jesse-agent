"""Gemini adapter for citation-grounded video Q&A."""

import json
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from tubetalk.domain.retrieval import ChatAnswer, ChatTurn, Citation, RetrievalHit
from tubetalk.ports.chat import ChatProviderError


class GeminiChatProvider:
    """Generate answers whose citations resolve to supplied evidence records."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        client: Optional[Any] = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to chat")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)

    def answer(
        self,
        question: str,
        evidence: tuple[RetrievalHit, ...],
        history: tuple[ChatTurn, ...],
    ) -> ChatAnswer:
        """Return a citation-validated answer, correcting once when necessary."""
        prompt = _chat_prompt(question, evidence, history)
        answer = _parse_answer(self._generate_content(prompt))
        try:
            _validate_citations(answer, evidence)
            return answer
        except ChatProviderError as error:
            corrected = _parse_answer(
                self._generate_content(f"{prompt}\n\nValidation feedback: {error}")
            )
            _validate_citations(corrected, evidence)
            return corrected

    def _generate_content(self, prompt: str) -> Any:
        try:
            return self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "timestamp_sec": {"type": "number"},
                                    },
                                    "required": ["source_id", "timestamp_sec"],
                                },
                            },
                        },
                        "required": ["answer", "citations"],
                    },
                ),
            )
        except (APIError, HTTPError) as error:
            raise ChatProviderError(str(error)) from error


def _chat_prompt(
    question: str, evidence: tuple[RetrievalHit, ...], history: tuple[ChatTurn, ...]
) -> str:
    history_text = (
        "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer.answer}"
            for turn in history
        )
        or "(none)"
    )
    evidence_text = "\n".join(
        f"id={hit.source_id}; source={hit.source}; interval={hit.start_sec:.3f}-"
        f"{hit.end_sec:.3f}; text={hit.text}"
        for hit in evidence
    )
    return (
        "Answer the user's question using only the evidence below. Respond in JSON. "
        "Give a concise answer and at least one citation. Every citation source_id "
        "must exactly match an evidence id, and timestamp_sec must be inside that "
        "evidence interval. Do not invent facts or citations.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Evidence:\n{evidence_text}\n\nQuestion: {question}"
    )


def _parse_answer(response: Any) -> ChatAnswer:
    try:
        payload = json.loads(response.text)
        return ChatAnswer(
            answer=payload["answer"],
            citations=tuple(
                Citation.model_validate(item) for item in payload["citations"]
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ChatProviderError("Gemini returned an invalid chat response") from error


def _validate_citations(answer: ChatAnswer, evidence: tuple[RetrievalHit, ...]) -> None:
    if not answer.answer.strip() or not answer.citations:
        raise ChatProviderError("Answer must contain text and at least one citation")
    sources = {hit.source_id: hit for hit in evidence}
    for citation in answer.citations:
        hit = sources.get(citation.source_id)
        if hit is None:
            raise ChatProviderError(f"Unknown citation source_id: {citation.source_id}")
        if not hit.start_sec <= citation.timestamp_sec <= hit.end_sec:
            raise ChatProviderError(
                f"Citation {citation.source_id} timestamp {citation.timestamp_sec} "
                f"is outside {hit.start_sec}-{hit.end_sec}"
            )
