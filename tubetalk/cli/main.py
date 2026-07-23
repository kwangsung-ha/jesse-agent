"""TubeTalk CLI entry point powered by Typer & Rich."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tubetalk.bootstrap import create_video_service
from tubetalk.domain.summary import VideoSummary
from tubetalk.services.video_service import (
    InvalidVideoUrlError,
    ProcessResult,
    SummaryGenerationError,
    SummaryUnavailableError,
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
    _show_process_summary(result)
    _show_process_vision(result)
    try:
        _show_detail(service.get_status(result.video_id))
    except VideoNotFoundError as error:
        console.print(f"[red]{error}[/red]")


@app.command(name="summary")
def show_summary(
    video_id: Optional[str] = typer.Argument(default=None),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Generate the summary when it is missing or stale.",
    ),
) -> None:
    """Display a cached transcript summary and its timestamp chapters."""
    service = create_video_service()
    if video_id is None:
        video_id = _select_summary_video(service.list_statuses())
    try:
        result = service.get_summary(video_id, generate=generate)
    except (
        VideoNotFoundError,
        SummaryUnavailableError,
        SummaryGenerationError,
    ) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    if result.summary is None:
        console.print("[red]Summary result did not contain content.[/red]")
        raise typer.Exit(code=1)
    if result.state == "generated":
        console.print("[green]Generated transcript summary.[/green]")
    _show_summary(result.summary)


def _select_summary_video(videos: list[VideoStatus]) -> str:
    """Show cached videos and return the ID chosen by an interactive user."""
    if not videos:
        console.print("[yellow]No cached videos found.[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title="Select a Cached Video", show_lines=True)
    table.add_column("No.", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Summary", justify="center")
    for index, video in enumerate(videos, start=1):
        table.add_row(
            str(index),
            video.title or video.video_id,
            _format_summary_summary(video),
        )
    console.print(table)
    choice = typer.prompt("Select a video number")
    try:
        selected = int(choice)
    except ValueError as error:
        console.print("[red]Enter a valid video number.[/red]")
        raise typer.Exit(code=2) from error
    if selected < 1 or selected > len(videos):
        console.print("[red]Select a number from the displayed list.[/red]")
        raise typer.Exit(code=2)
    return videos[selected - 1].video_id


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


def _show_process_summary(result: ProcessResult) -> None:
    """Render the summary created or reused by ``process``."""
    if result.summary.warning:
        console.print(
            "[yellow]Warning: summary was not updated: "
            f"{result.summary.warning}[/yellow]"
        )
        return
    if result.summary.summary is not None:
        if result.summary.state == "generated":
            console.print("[green]Generated transcript summary.[/green]")
        else:
            console.print("[dim]Transcript summary is current.[/dim]")
        _show_summary(result.summary.summary)


def _show_process_vision(result: ProcessResult) -> None:
    """Render the non-fatal visual-scene indexing outcome."""
    if result.vision.warning:
        console.print(
            "[yellow]Warning: vision index was not updated: "
            f"{result.vision.warning}[/yellow]"
        )
    elif result.vision.state == "generated":
        console.print(
            f"[green]Generated {result.vision.scene_count} visual scenes.[/green]"
        )
    elif result.vision.state == "current":
        console.print("[dim]Vision index is current.[/dim]")


def _show_summary(summary: VideoSummary) -> None:
    """Render a summary and its timestamp chapters consistently across commands."""
    console.print("[bold cyan]Summary[/bold cyan]")
    console.print(summary.text)
    if not summary.chapters:
        return
    table = Table(title="Timestamp Chapters", show_header=True)
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    for chapter in summary.chapters:
        table.add_row(_format_timestamp(chapter.start_sec), chapter.title)
    console.print(table)


def _format_timestamp(start_sec: float) -> str:
    """Format a chapter timestamp as ``MM:SS`` or ``HH:MM:SS``."""
    total_seconds = int(start_sec)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
            _format_vision_summary(video),
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
        ("Vision Index", _format_vision_state(status_info)),
        (
            "Vision Scenes",
            str(status_info.vision_scene_count)
            if status_info.vision_scene_count is not None
            else "—",
        ),
        ("Vision Model", status_info.vision_model or "—"),
        ("Vision Prompt", status_info.vision_prompt_version or "—"),
        ("Vision Generated At", status_info.vision_generated_at or "—"),
        ("Vision Vector Index", _format_vision_vector_state(status_info)),
        (
            "Vision Vector Scenes",
            str(status_info.vision_vector_index_scenes)
            if status_info.vision_vector_index_scenes is not None
            else "—",
        ),
        ("Vision Embedding Model", status_info.vision_vector_index_model or "—"),
        (
            "Vision Embedding Dimension",
            str(status_info.vision_vector_index_dimension)
            if status_info.vision_vector_index_dimension is not None
            else "—",
        ),
        ("Vision Indexed At", status_info.vision_vector_indexed_at or "—"),
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


def _format_vision_summary(status_info: VideoStatus) -> str:
    """Format visual-scene index freshness for the all-videos table."""
    if status_info.vision_index_state == "current":
        return (
            f"✅ {status_info.vision_scene_count}"
            if status_info.vision_scene_count is not None
            else "✅"
        )
    if status_info.vision_index_state == "stale":
        return "⚠️ stale"
    if status_info.vision_index_state == "invalid":
        return "❌ invalid"
    return "—"


def _format_vision_state(status_info: VideoStatus) -> str:
    """Format visual-scene index freshness for a detail view."""
    if status_info.vision_index_state == "current":
        return "✅ Current"
    if status_info.vision_index_state == "stale":
        return "⚠️ Stale"
    if status_info.vision_index_state == "invalid":
        return "❌ Invalid"
    return "❌ Missing"


def _format_vision_vector_state(status_info: VideoStatus) -> str:
    """Format visual vector-index freshness for a detail view."""
    if status_info.vision_vector_index_state == "current":
        return "✅ Current"
    if status_info.vision_vector_index_state == "stale":
        return "⚠️ Stale"
    if status_info.vision_vector_index_state == "invalid":
        return "❌ Invalid"
    return "❌ Missing"
