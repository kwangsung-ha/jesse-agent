"""TubeTalk CLI entry point powered by Typer & Rich."""

from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tubetalk.core.cache import LocalCacheManager
from tubetalk.pipeline.loader import YouTubeLoader

app = typer.Typer(
    name="tubetalk",
    help="YouTube Video Intelligence Agent — CLI",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main():
    pass


@app.command()
def status(
    video_id: Optional[str] = typer.Argument(default=None),
) -> None:
    """Show local cache status for all or a specific video."""
    cache = LocalCacheManager()

    if video_id is None:
        _show_all(cache)
    else:
        _show_detail(cache, video_id)


@app.command()
def process(url: str = typer.Argument(..., metavar="YOUTUBE_URL")) -> None:
    """Fetch a video's metadata and transcript, then save them locally.

    This initial ingestion command deliberately stops after text collection.
    Vision indexing and summaries are added by later pipeline stages.
    """
    loader = YouTubeLoader()
    try:
        video_id = loader.extract_video_id(url)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    cache = LocalCacheManager()
    if cache.has_cache(video_id):
        console.print(f"[green]Cache hit for {video_id}; using saved data.[/green]")
        _show_detail(cache, video_id)
        return

    console.print(f"[cyan]Collecting metadata and transcript for {video_id}...[/cyan]")
    try:
        metadata = loader.fetch_metadata(url)
        transcript = loader.fetch_transcript(video_id)
    except Exception as error:
        console.print(f"[red]Failed to process {video_id}: {error}[/red]")
        raise typer.Exit(code=1) from error

    metadata.update(
        {
            "video_id": video_id,
            "source_url": url,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    cache.save_json(video_id, "metadata.json", metadata)
    cache.save_json(video_id, "transcript.json", transcript)

    console.print(
        f"[green]Saved {len(transcript)} transcript segments for {video_id}.[/green]"
    )
    _show_detail(cache, video_id)


def _show_all(cache: LocalCacheManager) -> None:
    """Display a Rich table summarising every cached video."""
    videos = cache.list_cached_videos()
    if not videos:
        console.print("[yellow]No cached videos found.[/yellow]")
        return

    table = Table(title="🎬 Cached Videos", show_lines=True)
    table.add_column("Video ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Channel", style="magenta")
    table.add_column("Segments", justify="right")
    table.add_column("Vision", justify="center")
    table.add_column("Cached At", style="dim")

    for v in videos:
        table.add_row(
            v["video_id"],
            v.get("title") or "—",
            v.get("channel") or "—",
            str(v.get("transcript_segments", 0)),
            "✅" if v.get("has_vision_index") else "—",
            v.get("cached_at") or "—",
        )

    console.print(table)


def _show_detail(cache: LocalCacheManager, video_id: str) -> None:
    """Display detailed metadata for a single cached video."""
    status_info = cache.get_video_status(video_id)
    if status_info is None:
        console.print(f"[red]Video '{video_id}' not found in local cache.[/red]")
        raise typer.Exit(code=1)

    table = Table(
        title=f"📺 Video Detail — {video_id}",
        show_header=False,
        show_lines=True,
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    duration = status_info.get("duration")
    duration_str = f"{duration:.0f}s" if duration is not None else "—"

    rows = [
        ("Video ID", status_info["video_id"]),
        ("Title", status_info.get("title") or "—"),
        ("Channel", status_info.get("channel") or "—"),
        ("Duration", duration_str),
        ("Metadata", "✅" if status_info["has_metadata"] else "❌"),
        ("Transcript", "✅" if status_info["has_transcript"] else "❌"),
        (
            "Transcript Segments",
            str(status_info.get("transcript_segments", 0)),
        ),
        ("Vision Index", "✅" if status_info["has_vision_index"] else "❌"),
        ("Cached At", status_info.get("cached_at") or "—"),
    ]
    for key, value in rows:
        table.add_row(key, value)

    console.print(table)
