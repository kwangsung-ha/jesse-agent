"""TubeTalk CLI entry point powered by Typer & Rich."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tubetalk.bootstrap import create_video_service
from tubetalk.services.video_service import (
    InvalidVideoUrlError,
    ProcessResult,
    VideoIngestionError,
    VideoNotFoundError,
    VideoStatus,
)

app = typer.Typer(
    name="tubetalk",
    help="YouTube Video Intelligence Agent — CLI",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Configure the TubeTalk command group."""


@app.command()
def status(
    video_id: Optional[str] = typer.Argument(default=None),
) -> None:
    """Show local cache status for all or a specific video."""
    service = create_video_service()
    if video_id is None:
        _show_all(service.list_statuses())
        return
    try:
        _show_detail(service.get_status(video_id))
    except VideoNotFoundError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error


@app.command()
def process(url: str = typer.Argument(..., metavar="YOUTUBE_URL")) -> None:
    """Fetch a video's data, cache it, and synchronise its text index."""
    service = create_video_service()
    try:
        result = service.process(url)
    except InvalidVideoUrlError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except VideoIngestionError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if result.cache_hit:
        console.print(
            f"[green]Cache hit for {result.video_id}; using saved data.[/green]"
        )
    else:
        console.print(
            f"[green]Saved {result.transcript_segments} transcript segments for "
            f"{result.video_id}.[/green]"
        )
    _show_indexing_result(result)
    try:
        _show_detail(service.get_status(result.video_id))
    except VideoNotFoundError as error:
        console.print(f"[red]{error}[/red]")


def _show_indexing_result(result: ProcessResult) -> None:
    """Render the non-fatal transcript indexing outcome."""
    if result.indexing.state == "current":
        console.print("[dim]Transcript index is current.[/dim]")
    elif result.indexing.state == "indexed":
        console.print(
            f"[green]Indexed {result.indexing.chunk_count} transcript chunks.[/green]"
        )
    elif result.indexing.warning:
        console.print(
            "[yellow]Warning: transcript index was not updated: "
            f"{result.indexing.warning}[/yellow]"
        )


def _show_all(videos: list[VideoStatus]) -> None:
    """Display a Rich table summarising every cached video."""
    if not videos:
        console.print("[yellow]No cached videos found.[/yellow]")
        return
    table = Table(title="🎬 Cached Videos", show_lines=True)
    table.add_column("Video ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Segments", justify="right")
    table.add_column("Text Index", justify="center")
    table.add_column("Summary", justify="center")
    table.add_column("Vision", justify="center")
    for video in videos:
        table.add_row(
            video.video_id,
            video.title or "—",
            str(video.transcript_segments),
            _format_index_summary(video),
            _format_summary_summary(video),
            "✅" if video.has_vision_index else "—",
        )
    console.print(table)


def _show_detail(status_info: VideoStatus) -> None:
    """Display detailed metadata for a single cached video."""
    table = Table(
        title=f"📺 Video Detail — {status_info.video_id}",
        show_header=False,
        show_lines=True,
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")
    duration = status_info.duration
    duration_str = f"{duration:.0f}s" if duration is not None else "—"
    rows = [
        ("Video ID", status_info.video_id),
        ("Title", status_info.title or "—"),
        ("Channel", status_info.channel or "—"),
        ("Duration", duration_str),
        ("Metadata", "✅" if status_info.has_metadata else "❌"),
        ("Transcript", "✅" if status_info.has_transcript else "❌"),
        ("Transcript Segments", str(status_info.transcript_segments)),
        ("Transcript Index", _format_index_state(status_info)),
        (
            "Indexed Chunks",
            str(status_info.transcript_index_chunks)
            if status_info.transcript_index_chunks is not None
            else "—",
        ),
        ("Embedding Model", status_info.transcript_index_model or "—"),
        (
            "Embedding Dimension",
            str(status_info.transcript_index_dimension)
            if status_info.transcript_index_dimension is not None
            else "—",
        ),
        ("Transcript Indexed At", status_info.transcript_indexed_at or "—"),
        ("Summary", _format_summary_state(status_info)),
        (
            "Summary Chapters",
            str(status_info.summary_chapters)
            if status_info.summary_chapters is not None
            else "—",
        ),
        ("Summary Model", status_info.summary_model or "—"),
        ("Summary Prompt", status_info.summary_prompt_version or "—"),
        ("Summary Language", status_info.summary_language or "—"),
        ("Summary Generated At", status_info.summary_generated_at or "—"),
        ("Vision Index", "✅" if status_info.has_vision_index else "❌"),
        ("Cached At", status_info.cached_at or "—"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _format_index_summary(status_info: VideoStatus) -> str:
    """Format text index state for the all-videos table."""
    if status_info.transcript_index_state == "current":
        return (
            f"✅ {status_info.transcript_index_chunks}"
            if status_info.transcript_index_chunks is not None
            else "✅"
        )
    if status_info.transcript_index_state == "stale":
        return "⚠️ stale"
    if status_info.transcript_index_state == "invalid":
        return "❌ invalid"
    return "—"


def _format_index_state(status_info: VideoStatus) -> str:
    """Format text index state for the detail table."""
    if status_info.transcript_index_state == "current":
        return "✅ Current"
    if status_info.transcript_index_state == "stale":
        return "⚠️ Stale"
    if status_info.transcript_index_state == "invalid":
        return "❌ Invalid"
    return "❌ Missing"


def _format_summary_summary(status_info: VideoStatus) -> str:
    """Format summary state for the all-videos table."""
    if status_info.summary_state == "current":
        return (
            f"✅ {status_info.summary_chapters}"
            if status_info.summary_chapters is not None
            else "✅"
        )
    if status_info.summary_state == "stale":
        return "⚠️ stale"
    if status_info.summary_state == "invalid":
        return "❌ invalid"
    return "—"


def _format_summary_state(status_info: VideoStatus) -> str:
    """Format summary state for the detailed status table."""
    if status_info.summary_state == "current":
        return "✅ Current"
    if status_info.summary_state == "stale":
        return "⚠️ Stale"
    if status_info.summary_state == "invalid":
        return "❌ Invalid"
    return "❌ Missing"
