"""Tests for opt-in diagnostics and versioned prompt resources."""

from io import StringIO

import pytest
from loguru import logger

from jesseagent.application.video.contracts import SummaryProviderError
from jesseagent.core import logging as logging_config
from jesseagent.core.logging import configure_debug_logging
from jesseagent.core.prompts import PromptCatalog, PromptTemplateError
from jesseagent.domain.transcript import Transcript, TranscriptSegment
from jesseagent.infrastructure.summaries.gemini import GeminiSummaryProvider


def test_prompt_catalog_renders_versioned_summary_template() -> None:
    prompt = PromptCatalog().render(
        "summary",
        "summary-chapters-v1",
        {
            "last_timestamp": "12.000",
            "language": "ko",
            "title": "Example",
            "transcript": "[00:00] Hello",
        },
    )

    assert "Video title: Example" in prompt
    assert "[00:00] Hello" in prompt
    assert "$title" not in prompt


def test_prompt_catalog_renders_candidate_template() -> None:
    prompt = PromptCatalog().render(
        "chapter_candidates",
        "chapter-candidates-v1",
        {
            "language": "ko",
            "title": "Example",
            "window_start": "0.000",
            "window_end": "12.000",
            "transcript": "[00:00] Hello",
        },
    )

    assert "Window: 0.000 through 12.000 seconds" in prompt
    assert "[00:00] Hello" in prompt


def test_prompt_catalog_renders_candidate_correction_template() -> None:
    prompt = PromptCatalog().render(
        "chapter_candidates_correction",
        "chapter-candidates-correction-v1",
        {
            "prompt": "original prompt",
            "block_index": "99",
            "last_block_index": "1",
        },
    )

    assert "block_index 99, but the only valid\nblock indexes are 0 through 1" in prompt


def test_prompt_catalog_rejects_missing_or_unknown_templates() -> None:
    catalog = PromptCatalog()

    with pytest.raises(PromptTemplateError, match="not found"):
        catalog.render("chat", "unknown", {})
    with pytest.raises(PromptTemplateError, match="could not be rendered"):
        catalog.render("vision", "vision-scenes-v2-30s", {})
    with pytest.raises(PromptTemplateError, match="Unknown prompt kind"):
        catalog.render("missing", "v1", {})


def test_summary_provider_reports_unknown_prompt_version_before_api_request() -> None:
    provider = GeminiSummaryProvider(api_key="key", prompt_version="missing")
    transcript = Transcript(
        segments=(TranscriptSegment(start_sec=0, duration_sec=1, text="Hello"),)
    )

    with pytest.raises(SummaryProviderError, match="was not found"):
        provider.generate_summary(transcript, title="Example", language="en")


def test_debug_logging_hides_trace_events_without_verbose_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = StringIO()
    monkeypatch.setattr(logging_config.sys, "stderr", output)
    configure_debug_logging(debug=True, verbose=False)

    logger.bind(event="stage.complete").debug("done")
    logger.bind(event="gemini.response").trace("raw response")

    configure_debug_logging(debug=False, verbose=False)

    assert "stage.complete" in output.getvalue()
    assert "raw response" not in output.getvalue()
