"""Unit tests for package version and configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from jesseagent import __version__
from jesseagent.core.config import Settings, settings


def test_package_version():
    """Verify package version constant."""
    assert __version__ == "0.1.0"


def test_settings_defaults():
    """Verify default settings values and custom instantiation."""
    s = Settings(gemini_api_key="test-key")
    assert s.gemini_api_key == "test-key"
    assert s.vision_model == "gemini-3.5-flash"
    assert s.vision_prompt_version == "vision-scenes-v2-30s"
    assert s.llm_model == "gemini-3.5-flash-lite"
    assert s.summary_model == "gemini-3.5-flash-lite"
    assert s.summary_language == "ko"
    assert s.summary_prompt_version == "summary-chapters-v2"
    assert s.chapter_window_policy.cache_key == "480s-12000chars-30s-v1"
    assert s.chapter_block_policy.cache_key == "20s-400chars-1.5s-gap-v1"
    assert s.chat_prompt_version == "grounded-chat-v1"
    assert s.agent_prompt_version == "tool-agent-v1"
    assert s.agent_max_steps == 8
    assert s.agent_context_max_messages == 24
    assert s.agent_context_max_characters == 12000
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


def test_settings_accepts_chapter_window_overrides() -> None:
    """Chapter extraction limits can be selected through environment-backed fields."""
    settings = Settings(
        chapter_window_max_seconds=300,
        chapter_window_max_characters=8000,
        chapter_window_overlap_seconds=20,
    )

    assert settings.chapter_window_policy.cache_key == "300s-8000chars-20s-v1"


def test_settings_reads_chapter_window_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uppercase environment variables configure the chapter policy."""
    monkeypatch.setenv("CHAPTER_WINDOW_MAX_SECONDS", "300")
    monkeypatch.setenv("CHAPTER_WINDOW_MAX_CHARACTERS", "8000")
    monkeypatch.setenv("CHAPTER_WINDOW_OVERLAP_SECONDS", "20")

    assert Settings().chapter_window_policy.cache_key == "300s-8000chars-20s-v1"


def test_settings_reads_chapter_block_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt-shaping block limits are independently configurable."""
    monkeypatch.setenv("CHAPTER_BLOCK_MAX_SECONDS", "15")
    monkeypatch.setenv("CHAPTER_BLOCK_MAX_CHARACTERS", "300")
    monkeypatch.setenv("CHAPTER_BLOCK_MAX_GAP_SECONDS", "1")

    assert Settings().chapter_block_policy.cache_key == "15s-300chars-1s-gap-v1"


def test_settings_rejects_an_overlap_that_prevents_window_progress() -> None:
    """Environment overrides must preserve a positive non-overlapping advance."""
    with pytest.raises(ValidationError, match="overlap"):
        Settings(chapter_window_max_seconds=30, chapter_window_overlap_seconds=30)
