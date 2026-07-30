"""Gemini adapter for citation-grounded video Q&A."""

import json
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from jesseagent.core.logging import logger
from jesseagent.core.prompts import PromptCatalog, PromptTemplateError
from jesseagent.domain.retrieval import ChatAnswer, ChatTurn, Citation, RetrievalHit
from jesseagent.ports.chat import ChatProviderError


class GeminiChatProvider:
    """Generate answers whose citations resolve to supplied evidence records."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        client: Optional[Any] = None,
        prompt_version: str = "grounded-chat-v1",
        prompts: PromptCatalog | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to chat")
        self.model = model
        self._client = client or genai.Client(api_key=api_key)
        self._prompt_version = prompt_version
        self._prompts = prompts or PromptCatalog()

    def answer(
        self,
        question: str,
        evidence: tuple[RetrievalHit, ...],
        history: tuple[ChatTurn, ...],
    ) -> ChatAnswer:
        """Return a citation-validated answer, correcting once when necessary."""
        prompt = self._chat_prompt(question, evidence, history)
        answer = _parse_answer(self._generate_content(prompt))
        try:
            _validate_citations(answer, evidence)
            return answer
        except ChatProviderError as error:
            corrected = _parse_answer(
                self._generate_content(self._correction_prompt(prompt, str(error)))
            )
            _validate_citations(corrected, evidence)
            return corrected

    def _generate_content(self, prompt: str) -> Any:
        logger.bind(event="gemini.chat.request", model=self.model).debug(
            "--- prompt ---\n{}\n--- end prompt ---", prompt
        )
        try:
            response = self._client.models.generate_content(
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
            logger.bind(event="gemini.chat.response", model=self.model).trace(
                "{}", str(getattr(response, "text", response))
            )
            return response
        except (APIError, HTTPError) as error:
            raise ChatProviderError(str(error)) from error

    def _chat_prompt(
        self,
        question: str,
        evidence: tuple[RetrievalHit, ...],
        history: tuple[ChatTurn, ...],
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
        try:
            return self._prompts.render(
                "chat",
                self._prompt_version,
                {
                    "history": history_text,
                    "evidence": evidence_text,
                    "question": question,
                },
            )
        except PromptTemplateError as error:
            raise ChatProviderError(str(error)) from error

    def _correction_prompt(self, prompt: str, feedback: str) -> str:
        try:
            return self._prompts.render(
                "chat_correction",
                "chat-correction-v1",
                {"prompt": prompt, "validation_feedback": feedback},
            )
        except PromptTemplateError as error:
            raise ChatProviderError(str(error)) from error


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
