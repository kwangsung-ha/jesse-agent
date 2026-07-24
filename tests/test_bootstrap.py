"""Tests for environment-selected production dependency wiring."""

from pathlib import Path
from typing import Any

from tubetalk.bootstrap import create_video_service
from tubetalk.core.config import Settings
from tubetalk.domain.transcript_index import TranscriptChunkPolicy


def test_create_video_service_selects_gemini_and_chroma(
    tmp_path: Path, mocker: Any
) -> None:
    """The default supported settings build the installed adapters lazily."""
    gemini_provider = mocker.patch("tubetalk.bootstrap.GeminiEmbeddingProvider")
    summary_provider = mocker.patch("tubetalk.bootstrap.GeminiSummaryProvider")
    vision_analyzer = mocker.patch("tubetalk.bootstrap.GeminiVisionAnalyzer")
    chroma_repository = mocker.patch(
        "tubetalk.bootstrap.ChromaTranscriptIndexRepository"
    )
    service = create_video_service(Settings(data_dir=tmp_path, gemini_api_key="key"))

    provider = service._embedding_provider_factory()
    repository = service._transcript_index_repository_factory("video123")
    summary = service._summary_provider_factory()
    vision = service._vision_analyzer_factory()

    assert provider is gemini_provider.return_value
    assert repository is chroma_repository.return_value
    assert summary is summary_provider.return_value
    assert vision is vision_analyzer.return_value
    gemini_provider.assert_called_once_with(
        api_key="key", model="gemini-embedding-2", dimension=768
    )
    chroma_repository.assert_called_once_with(
        "video123",
        data_dir=tmp_path,
        embedding_model="gemini-embedding-2",
        embedding_dimension=768,
        chunk_policy=TranscriptChunkPolicy(max_seconds=45.0, max_characters=1200),
    )
    summary_provider.assert_called_once_with(
        api_key="key", model="gemini-3.5-flash-lite"
    )
    vision_analyzer.assert_called_once_with(api_key="key", model="gemini-3.5-flash")
