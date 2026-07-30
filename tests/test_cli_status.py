"""Tests for the natural-language CLI adapter."""

from typing import Any

from typer.testing import CliRunner

from jesseagent.cli.main import app

runner = CliRunner()


def test_one_shot_request_runs_agent_and_renders_response(mocker: Any) -> None:
    session = mocker.Mock()
    session.ask.return_value = "영상 처리를 완료했습니다."
    factory = mocker.patch(
        "jesseagent.cli.main.create_agent_session", return_value=session
    )

    result = runner.invoke(app, ["이 URL을 처리해줘"])

    assert result.exit_code == 0
    assert "영상 처리를 완료했습니다" in result.output
    session.ask.assert_called_once_with("이 URL을 처리해줘")
    assert factory.call_args.kwargs["on_tool_result"] is not None


def test_repl_keeps_one_agent_session_until_exit(mocker: Any) -> None:
    session = mocker.Mock()
    session.ask.return_value = "목록입니다."
    mocker.patch("jesseagent.cli.main.create_agent_session", return_value=session)

    result = runner.invoke(app, input="영상 목록 보여줘\nexit\n")

    assert result.exit_code == 0
    assert "Ask about a video" in result.output
    session.ask.assert_called_once_with("영상 목록 보여줘")


def test_verbose_requires_debug() -> None:
    result = runner.invoke(app, ["--verbose", "목록 보여줘"])

    assert result.exit_code == 2
    assert "requires --debug" in result.output
