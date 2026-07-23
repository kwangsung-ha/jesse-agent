"""Unit tests for package version and configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tubetalk import __version__
from tubetalk.core.config import Settings, settings


def test_package_version():
    """Verify package version constant."""
    assert __version__ == "0.1.0"


def test_settings_defaults():
    """Verify default settings values and custom instantiation."""
    s = Settings(gemini_api_key="test-key")
    assert s.gemini_api_key == "test-key"
    assert s.default_sample_interval_sec == 5.0
    assert s.vision_model == "gemini-2.5-flash"
    assert s.llm_model == "gemini-2.5-pro"
    assert s.summary_model == "gemini-3.5-flash-lite"
    assert s.summary_language == "ko"
    assert s.summary_prompt_version == "summary-chapters-v1"
    assert s.embedding_model == "gemini-embedding-2"
    assert s.embedding_dimension == 768
    assert s.embedding_provider == "gemini"
    assert s.vector_repository == "chroma"
    assert s.transcript_chunk_max_seconds == 45.0
    assert s.transcript_chunk_max_characters == 1200
    assert settings is not None


def test_ensure_data_dir(tmp_path: Path):
    """Verify that ensure_data_dir creates the target directory if it does not exist."""
    target_dir = tmp_path / "custom_data_dir"
    s = Settings(data_dir=target_dir)
    assert not target_dir.exists()

    created_dir = s.ensure_data_dir()
    assert created_dir.exists()
    assert created_dir == target_dir


@pytest.mark.parametrize(
    ("field", "value"),
    [("embedding_provider", "local"), ("vector_repository", "qdrant")],
)
def test_settings_reject_unsupported_infrastructure(field: str, value: str) -> None:
    """Only implementations installed in this release are selectable."""
    with pytest.raises(ValidationError):
        Settings(**{field: value})
