"""Unit tests for package version and configuration settings."""

from pathlib import Path

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
    assert settings is not None


def test_ensure_data_dir(tmp_path: Path):
    """Verify that ensure_data_dir creates the target directory if it does not exist."""
    target_dir = tmp_path / "custom_data_dir"
    s = Settings(data_dir=target_dir)
    assert not target_dir.exists()

    created_dir = s.ensure_data_dir()
    assert created_dir.exists()
    assert created_dir == target_dir
