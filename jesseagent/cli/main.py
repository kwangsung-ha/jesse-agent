"""Natural-language JesseAgent command-line Agent."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jesseagent.agent.contracts import ToolResult
from jesseagent.bootstrap import create_agent_session, create_knowledge_sync_service
from jesseagent.cli.runs import (
    approve,
    list_runs,
    reject,
    resume,
    status,
)
from jesseagent.cli.runs import (
    delete as delete_run,
)
from jesseagent.core.config import settings
from jesseagent.core.logging import configure_debug_logging
from jesseagent.sources.obsidian import ObsidianSourceConnector

app = typer.Typer(
    name="jesseagent",
    help="Personal knowledge and task Agent — natural-language CLI",
    no_args_is_help=True,
)
console = Console()
sources_app = typer.Typer(help="Synchronize configured knowledge sources.")
run_app = typer.Typer(
    help="Start the Agent REPL or manage durable runs.", invoke_without_command=True
)


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show diagnostic logs."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Include raw model responses (requires --debug)."
    ),
) -> None:
    """Personal knowledge and task Agent commands."""
    if verbose and not debug:
        raise typer.BadParameter("--verbose requires --debug", param_hint="--verbose")
    configure_debug_logging(debug=debug, verbose=verbose)
    if ctx.invoked_subcommand is not None:
        return


@run_app.callback()
def run(ctx: typer.Context) -> None:
    """Start a multi-turn conversation when no management command is selected."""
    if ctx.invoked_subcommand is not None:
        return
    session = create_agent_session(on_tool_result=show_tool_result)
    console.print("[dim]Ask about a video, or type 'exit' to finish.[/dim]")
    while True:
        try:
            message = typer.prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if message.lower() in {"exit", "quit"}:
            return
        if message:
            _show_response(session.ask(message))


def _show_response(response: str) -> None:
    """Render the Agent's final natural-language response consistently."""
    console.print(Panel(response, title="JesseAgent", border_style="cyan"))


def show_tool_result(result: ToolResult) -> None:
    """Render concise progressive feedback for a completed Agent tool call."""
    state = "completed" if result.ok else "failed"
    console.print(f"[dim]{result.name} {state}[/dim]")
    if result.name != "answer_video_question" or not result.ok:
        return
    citations = result.content.get("citations", [])
    if not citations:
        return
    table = Table(title="Citations", show_header=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Evidence")
    for citation in citations:
        table.add_row(
            _format_timestamp(float(citation["timestamp_sec"])),
            str(citation["source"]),
            str(citation["evidence"]),
        )
    console.print(table)


def _format_timestamp(seconds: float) -> str:
    """Format a cited offset as a compact timestamp."""
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@sources_app.command("sync")
def sync_source(source: str) -> None:
    """Incrementally synchronize one configured source."""
    if source != "obsidian":
        raise typer.BadParameter("Only 'obsidian' is supported")
    if settings.obsidian_vault_path is None:
        raise typer.BadParameter("OBSIDIAN_VAULT_PATH must be configured")
    result = create_knowledge_sync_service().sync(
        ObsidianSourceConnector(settings.obsidian_vault_path)
    )
    typer.echo(
        "obsidian: "
        f"added={result.added} updated={result.updated} "
        f"unchanged={result.unchanged} deleted={result.deleted}"
    )


app.add_typer(sources_app, name="sources")
run_app.command("list")(list_runs)
run_app.command("status")(status)
run_app.command("approve")(approve)
run_app.command("reject")(reject)
run_app.command("resume")(resume)
run_app.command("delete")(delete_run)
app.add_typer(run_app, name="run")
