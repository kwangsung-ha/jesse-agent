"""Port for providers that generate transcript-grounded summaries."""

from typing import Any, Protocol

from tubetalk.domain.summary import VideoSummary


class SummaryProviderError(Exception):
    """Raised when a summary provider cannot produce a valid summary."""


class SummaryProvider(Protocol):
    """Generate a structured summary from timestamped transcript segments."""

    def generate_summary(
        self,
        transcript: list[dict[str, Any]],
        *,
        title: str,
        language: str,
    ) -> VideoSummary:
        """Return a concise summary and chronological chapter titles."""
