"""Grounded Gemini chat-provider tests with mocked transport."""

from types import SimpleNamespace
from typing import Any

import pytest

from jesseagent.application.video.contracts import ChatProviderError
from jesseagent.domain.retrieval import RetrievalHit
from jesseagent.infrastructure.chats.gemini import GeminiChatProvider


def _evidence() -> tuple[RetrievalHit, ...]:
    return (
        RetrievalHit(
            source_id="video:chunk:0",
            source="transcript",
            text="A presenter introduces the topic.",
            start_sec=10,
            end_sec=20,
            rank=1,
            distance=0.1,
        ),
    )


def test_chat_provider_returns_valid_grounded_answer(mocker: Any) -> None:
    client = mocker.Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=(
            '{"answer":"The presenter introduces it.","citations":'
            '[{"source_id":"video:chunk:0","timestamp_sec":12}]}'
        )
    )
    provider = GeminiChatProvider("key", client=client)

    answer = provider.answer("What happens?", _evidence(), ())

    assert answer.citations[0].timestamp_sec == 12
    assert client.models.generate_content.call_count == 1


def test_chat_provider_corrects_invalid_citation_once(mocker: Any) -> None:
    client = mocker.Mock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(
            text='{"answer":"Answer","citations":[{"source_id":"bad","timestamp_sec":1}]}'
        ),
        SimpleNamespace(
            text='{"answer":"Answer","citations":[{"source_id":"video:chunk:0","timestamp_sec":10}]}'
        ),
    ]
    provider = GeminiChatProvider("key", client=client)

    answer = provider.answer("What happens?", _evidence(), ())

    assert answer.citations[0].source_id == "video:chunk:0"
    assert client.models.generate_content.call_count == 2


def test_chat_provider_rejects_persistently_invalid_citations(
    mocker: Any,
) -> None:
    client = mocker.Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"answer":"Answer","citations":[]}'
    )
    provider = GeminiChatProvider("key", client=client)

    with pytest.raises(ChatProviderError, match="at least one citation"):
        provider.answer("What happens?", _evidence(), ())
