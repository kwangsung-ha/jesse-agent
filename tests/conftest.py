"""Pytest global configuration and fixtures."""

import pytest


@pytest.fixture
def sample_fixture() -> str:
    """Sample fixture for testing harness setup."""
    return "tubetalk_fixture"
