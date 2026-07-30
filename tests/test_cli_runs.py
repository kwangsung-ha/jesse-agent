"""Tests for the durable Agent-run CLI."""

from typing import Any

from typer.testing import CliRunner

from tubetalk.cli.runs import app

runner = CliRunner()


def test_run_commands_delegate_to_service(mocker: Any) -> None:
    service = mocker.Mock()
    state = mocker.Mock(run_id="run-1", status="pending_approval")
    state.model_dump_json.return_value = '{"run_id":"run-1"}'
    service.list_runs.return_value = (state,)
    service.get_status.return_value = state
    service.approve.return_value = state
    service.reject.return_value = state
    service.resume.return_value = mocker.Mock(response="resumed")
    factory = mocker.patch(
        "tubetalk.cli.runs.create_agent_run_service", return_value=service
    )

    assert runner.invoke(app, ["list"]).exit_code == 0
    assert runner.invoke(app, ["status", "run-1"]).exit_code == 0
    assert runner.invoke(app, ["approve", "run-1"]).exit_code == 0
    assert runner.invoke(app, ["reject", "run-1"]).exit_code == 0
    assert runner.invoke(app, ["resume", "run-1"]).exit_code == 0
    assert runner.invoke(app, ["delete", "run-1"]).exit_code == 0
    assert factory.call_count == 6
