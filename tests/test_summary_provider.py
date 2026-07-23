"""Unit tests for the Gemini transcript-summary adapter."""

import json
from typing import Any

import pytest
from httpx import HTTPError

from tubetalk.infrastructure.summaries.gemini import GeminiSummaryProvider
from tubetalk.ports.summary import SummaryProviderError


def _transcript() -> list[dict[str, object]]:
    return [
        {"start_sec": 0.0, "duration_sec": 10.0, "text": "영상 소개입니다."},
        {"start_sec": 10.0, "duration_sec": 15.0, "text": "핵심 내용을 설명합니다."},
    ]


def test_gemini_summary_provider_requests_structured_flash_lite_output(
    mocker: Any,
) -> None:
    """The adapter must request JSON from the configured summary model."""
    client = mocker.Mock()
    client.models.generate_content.return_value.text = json.dumps(
        {
            "summary": "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
            "chapters": [
                {"start_sec": 0, "title": "소개"},
                {"start_sec": 10, "title": "핵심 내용"},
            ],
        }
    )
    provider = GeminiSummaryProvider(
        api_key="key", model="gemini-3.5-flash-lite", client=client
    )

    result = provider.generate_summary(_transcript(), title="예시 영상", language="ko")

    assert result.text.startswith("첫 문장")
    assert [chapter.title for chapter in result.chapters] == ["소개", "핵심 내용"]
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-3.5-flash-lite"
    assert "[00:10] 핵심 내용을 설명합니다." in call.kwargs["contents"]
    assert call.kwargs["config"].response_mime_type == "application/json"
    assert call.kwargs["config"].response_schema["required"] == [
        "summary",
        "chapters",
    ]


@pytest.mark.parametrize(
    "response_text",
    [
        "not json",
        json.dumps(
            {"summary": "요약", "chapters": [{"start_sec": 30, "title": "초과"}]}
        ),
        json.dumps(
            {"summary": "요약", "chapters": [{"start_sec": True, "title": "오류"}]}
        ),
    ],
)
def test_gemini_summary_provider_rejects_invalid_responses(
    mocker: Any, response_text: str
) -> None:
    """Malformed output or unsupported timestamps must not enter the cache."""
    client = mocker.Mock()
    client.models.generate_content.return_value.text = response_text
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="Invalid Gemini summary response"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")


def test_gemini_summary_provider_converts_api_errors(mocker: Any) -> None:
    """Transport errors should be exposed through the provider-specific error."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = HTTPError("offline")
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="offline"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")


def test_gemini_summary_provider_rejects_empty_transcript(mocker: Any) -> None:
    """An empty transcript fails before the provider is called."""
    client = mocker.Mock()
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="empty transcript"):
        provider.generate_summary([], title="예시 영상", language="ko")
    client.models.generate_content.assert_not_called()
