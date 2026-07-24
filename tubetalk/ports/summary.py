"""Port for providers that generate transcript-grounded summaries."""

from typing import Protocol

from tubetalk.domain.summary import VideoSummary
from tubetalk.domain.transcript import Transcript


class SummaryProviderError(Exception):
    """Raised when a summary provider cannot produce a valid summary."""


class SummaryProvider(Protocol):
    """Generate a structured summary from timestamped transcript segments."""

    def generate_summary(
        self,
        transcript: Transcript,
        *,
        title: str,
        language: str,
    ) -> VideoSummary:
        """Return a concise summary and chronological chapter titles."""
