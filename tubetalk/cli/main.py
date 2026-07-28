"""Natural-language TubeTalk command-line Agent."""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tubetalk.agent.contracts import ToolResult
from tubetalk.bootstrap import create_agent_session
from tubetalk.core.logging import configure_debug_logging

app = typer.Typer(
    name="tubetalk",
    help="YouTube Video Intelligence Agent — natural-language CLI",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


@app.callback()
def main(
    request: Optional[str] = typer.Argument(
        default=None, help="A one-shot natural-language request."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show diagnostic logs."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Include raw model responses (requires --debug)."
    ),
) -> None:
    """Run one request or start a multi-turn TubeTalk conversation."""
    if verbose and not debug:
        raise typer.BadParameter("--verbose requires --debug", param_hint="--verbose")
    configure_debug_logging(debug=debug, verbose=verbose)
    session = create_agent_session(on_tool_result=show_tool_result)
    if request is not None:
        _show_response(session.ask(request))
        return
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
    console.print(Panel(response, title="TubeTalk", border_style="cyan"))


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
