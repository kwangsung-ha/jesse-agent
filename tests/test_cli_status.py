"""Tests for the natural-language CLI adapter."""

from typing import Any

from typer.testing import CliRunner

from jesseagent.cli.main import app
from jesseagent.core.config import Settings

runner = CliRunner()


def test_repl_keeps_one_agent_session_until_exit(mocker: Any) -> None:
    session = mocker.Mock()
    session.ask.return_value = "목록입니다."
    mocker.patch("jesseagent.cli.main.create_agent_session", return_value=session)

    result = runner.invoke(app, ["run"], input="영상 목록 보여줘\nexit\n")

    assert result.exit_code == 0
    assert "Ask about a video" in result.output
    session.ask.assert_called_once_with("영상 목록 보여줘")


def test_verbose_requires_debug() -> None:
    result = runner.invoke(app, ["--verbose", "run"])

    assert result.exit_code == 2
    assert "requires --debug" in result.output


def test_sources_sync_obsidian_uses_configured_vault(mocker: Any, tmp_path) -> None:
    service = mocker.Mock()
    service.sync.return_value = mocker.Mock(added=1, updated=2, unchanged=3, deleted=4)
    mocker.patch("jesseagent.cli.main.settings", Settings(obsidian_vault_path=tmp_path))
    mocker.patch(
        "jesseagent.cli.main.create_knowledge_sync_service", return_value=service
    )

    result = runner.invoke(app, ["sources", "sync", "obsidian"])

    assert result.exit_code == 0
    assert "added=1 updated=2 unchanged=3 deleted=4" in result.output
