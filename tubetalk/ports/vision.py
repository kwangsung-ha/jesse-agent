"""Port for providers that turn video inputs into timestamped visual scenes."""

from typing import Protocol

from tubetalk.domain.vision import VisionScene, VisionSource


class VisionProviderError(Exception):
    """Raised when a vision provider cannot produce a valid scene index."""


class VisionAnalyzer(Protocol):
    """Describe a video source without exposing provider-specific API details."""

    def describe(self, source: VisionSource, *, title: str) -> tuple[VisionScene, ...]:
        """Return chronologically ordered visual scenes for the source."""
