"""Unit tests for the coverage-first Gemini transcript-summary adapter."""

import json
from typing import Any

import pytest
from httpx import HTTPError

from jesseagent.application.video.contracts import SummaryProviderError
from jesseagent.domain.transcript import Transcript, TranscriptSegment
from jesseagent.infrastructure.summaries.gemini import GeminiSummaryProvider, _timestamp


def _transcript() -> Transcript:
    return Transcript(
        segments=(
            TranscriptSegment(
                start_sec=0.0, duration_sec=10.0, text="영상 소개입니다."
            ),
            TranscriptSegment(
                start_sec=10.0, duration_sec=15.0, text="핵심 내용을 설명합니다."
            ),
        )
    )


def _response(payload: dict[str, object], mocker: Any) -> Any:
    return mocker.Mock(text=json.dumps(payload))


def test_gemini_summary_provider_extracts_then_consolidates_all_candidates(
    mocker: Any,
) -> None:
    """The final request must retain provenance for every local candidate."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = [
        _response(
            {
                "candidates": [
                    {"block_index": 0, "title": "소개"},
                    {"block_index": 1, "title": "핵심 내용"},
                ]
            },
            mocker,
        ),
        _response(
            {
                "summary": "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
                "chapters": [
                    {
                        "start_sec": 0,
                        "title": "소개",
                        "candidate_ids": ["candidate-0"],
                    },
                    {
                        "start_sec": 11,
                        "title": "핵심 내용",
                        "candidate_ids": ["candidate-1"],
                    },
                ],
            },
            mocker,
        ),
    ]
    provider = GeminiSummaryProvider(api_key="key", client=client)

    result = provider.generate_summary(_transcript(), title="예시 영상", language="ko")

    assert [chapter.title for chapter in result.chapters] == ["소개", "핵심 내용"]
    assert result.chapters[1].start_sec == 10
    assert client.models.generate_content.call_count == 2
    candidate_call, consolidation_call = client.models.generate_content.call_args_list
    assert (
        "[B1 | 00:10–00:25] 핵심 내용을 설명합니다."
        in candidate_call.kwargs["contents"]
    )
    assert candidate_call.kwargs["config"].response_schema["required"] == ["candidates"]
    assert '"candidate_id": "candidate-1"' in consolidation_call.kwargs["contents"]
    assert consolidation_call.kwargs["config"].response_schema["required"] == [
        "summary",
        "chapters",
    ]


def test_gemini_summary_provider_rejects_omitted_candidate(mocker: Any) -> None:
    """A consolidation response may merge candidates but may never discard one."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = [
        _response(
            {
                "candidates": [
                    {"block_index": 0, "title": "소개"},
                    {"block_index": 1, "title": "핵심"},
                ]
            },
            mocker,
        ),
        _response(
            {
                "summary": "요약입니다.",
                "chapters": [
                    {
                        "start_sec": 0,
                        "title": "소개",
                        "candidate_ids": ["candidate-0"],
                    }
                ],
            },
            mocker,
        ),
    ]
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="every candidate exactly once"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")


def test_gemini_summary_provider_retries_out_of_range_consolidation(
    mocker: Any,
) -> None:
    """Only final chapter timestamps use the existing corrective retry path."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = [
        _response({"candidates": [{"block_index": 0, "title": "소개"}]}, mocker),
        _response(
            {
                "summary": "요약입니다.",
                "chapters": [
                    {
                        "start_sec": 30,
                        "title": "초과",
                        "candidate_ids": ["candidate-0"],
                    }
                ],
            },
            mocker,
        ),
        _response(
            {
                "summary": "요약입니다.",
                "chapters": [
                    {
                        "start_sec": 10,
                        "title": "수정",
                        "candidate_ids": ["candidate-0"],
                    }
                ],
            },
            mocker,
        ),
    ]
    provider = GeminiSummaryProvider(api_key="key", client=client)

    result = provider.generate_summary(_transcript(), title="예시 영상", language="ko")

    assert result.chapters[0].title == "수정"
    assert client.models.generate_content.call_count == 3
    correction = client.models.generate_content.call_args_list[2].kwargs["contents"]
    assert "30.000 outside 0.0 through" in correction


def test_gemini_summary_provider_retries_out_of_window_candidate(
    mocker: Any,
) -> None:
    """A local timestamp error corrects only that extraction window."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = [
        _response({"candidates": [{"block_index": 99, "title": "초과"}]}, mocker),
        _response({"candidates": [{"block_index": 1, "title": "수정"}]}, mocker),
        _response(
            {
                "summary": "요약입니다.",
                "chapters": [
                    {
                        "start_sec": 10,
                        "title": "수정",
                        "candidate_ids": ["candidate-0"],
                    }
                ],
            },
            mocker,
        ),
    ]
    provider = GeminiSummaryProvider(api_key="key", client=client)

    result = provider.generate_summary(_transcript(), title="예시 영상", language="ko")

    assert result.chapters[0].title == "수정"
    assert client.models.generate_content.call_count == 3
    correction = client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert (
        "block_index 99, but the only valid\nblock indexes are 0 through 1"
        in correction
    )


def test_gemini_summary_provider_rejects_invalid_candidate_response(
    mocker: Any,
) -> None:
    """Malformed candidate output never proceeds to summary consolidation."""
    client = mocker.Mock()
    client.models.generate_content.return_value = _response(
        {"candidates": [{}]}, mocker
    )
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="Invalid Gemini candidate response"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")
    assert client.models.generate_content.call_count == 1


def test_gemini_summary_provider_converts_api_errors(mocker: Any) -> None:
    """Transport errors should be exposed through the provider-specific error."""
    client = mocker.Mock()
    client.models.generate_content.side_effect = HTTPError("offline")
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="offline"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")


def test_gemini_summary_provider_rejects_empty_transcript_without_a_request(
    mocker: Any,
) -> None:
    """An empty source cannot silently become an empty chapter list."""
    client = mocker.Mock()
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="empty transcript"):
        provider.generate_summary(
            Transcript(segments=()), title="예시 영상", language="ko"
        )
    client.models.generate_content.assert_not_called()


def test_gemini_summary_provider_rejects_empty_candidate_extraction(
    mocker: Any,
) -> None:
    """A model failure to identify topics cannot be cached as a valid summary."""
    client = mocker.Mock()
    client.models.generate_content.return_value = _response({"candidates": []}, mocker)
    provider = GeminiSummaryProvider(api_key="key", client=client)

    with pytest.raises(SummaryProviderError, match="no topic transitions"):
        provider.generate_summary(_transcript(), title="예시 영상", language="ko")
    assert client.models.generate_content.call_count == 1


def test_gemini_summary_provider_requires_an_api_key() -> None:
    """The provider cannot be configured without production credentials."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiSummaryProvider(api_key="")


def test_timestamp_formats_hour_values() -> None:
    """Long video transcript prompts retain an unambiguous hour component."""
    assert _timestamp(3661) == "01:01:01"
