"""CLI for inspecting and controlling durable Agent runs."""

import typer

from tubetalk.bootstrap import create_agent_run_service

app = typer.Typer(name="tubetalk-runs", help="Inspect and control Agent runs.")


@app.command("list")
def list_runs() -> None:
    """List durable Agent runs and their current state."""
    for state in create_agent_run_service().list_runs():
        typer.echo(f"{state.run_id}  {state.status}")


@app.command("status")
def status(run_id: str) -> None:
    """Show one durable Agent run's reducer-derived state."""
    typer.echo(create_agent_run_service().get_status(run_id).model_dump_json())


@app.command("approve")
def approve(run_id: str) -> None:
    """Approve a pending costly operation."""
    typer.echo(create_agent_run_service().approve(run_id).model_dump_json())


@app.command("reject")
def reject(run_id: str) -> None:
    """Reject and cancel a pending costly operation."""
    typer.echo(create_agent_run_service().reject(run_id).model_dump_json())


@app.command("resume")
def resume(run_id: str) -> None:
    """Resume a paused or approved run."""
    result = create_agent_run_service().resume(run_id)
    if result.response:
        typer.echo(result.response)


@app.command("delete")
def delete(run_id: str) -> None:
    """Delete a durable run and all of its events."""
    create_agent_run_service().delete_run(run_id)
    typer.echo(f"Deleted {run_id}")
