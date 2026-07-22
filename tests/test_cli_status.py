"""Unit tests for the TubeTalk CLI status and process commands."""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from tubetalk.cli.main import app

runner = CliRunner()


# ------------------------------------------------------------------
# Helper: populate a fake data directory
# ------------------------------------------------------------------


def _create_video_cache(
    data_dir: Path,
    video_id: str,
    title: str = "Test Title",
    channel: str = "TestChannel",
    duration: float = 120.0,
    segments: int = 3,
    vision: bool = False,
) -> None:
    """Create a minimal cached video directory for testing."""
    vdir = data_dir / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "metadata.json").write_text(
        json.dumps({"title": title, "channel": channel, "duration": duration})
    )
    transcript = [{"start_sec": float(i), "text": f"seg{i}"} for i in range(segments)]
    (vdir / "transcript.json").write_text(json.dumps(transcript))
    if vision:
        (vdir / "vision_index.json").write_text("[]")


# ------------------------------------------------------------------
# status (no args → list all)
# ------------------------------------------------------------------


class TestStatusListAll:
    """Tests for ``tubetalk status`` (no VIDEO_ID)."""

    def test_empty_cache(self, tmp_path: Path, mocker: Any) -> None:
        """Show a message when no videos are cached."""
        mocker.patch(
            "tubetalk.cli.main.LocalCacheManager",
            return_value=_make_cache(tmp_path),
        )
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No cached videos found" in result.output

    def test_shows_table_with_videos(self, tmp_path: Path, mocker: Any) -> None:
        """status should display a Rich table when videos exist."""
        _create_video_cache(tmp_path, "vid_a", title="Alpha Video", vision=True)
        _create_video_cache(tmp_path, "vid_b", title="Beta Video")
        mocker.patch(
            "tubetalk.cli.main.LocalCacheManager",
            return_value=_make_cache(tmp_path),
        )

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "vid_a" in result.output
        assert "Alpha Video" in result.output
        assert "vid_b" in result.output
        assert "Beta Video" in result.output


# ------------------------------------------------------------------
# status <VIDEO_ID>
# ------------------------------------------------------------------


class TestStatusDetail:
    """Tests for ``tubetalk status <VIDEO_ID>``."""

    def test_not_found(self, tmp_path: Path, mocker: Any) -> None:
        """Exit code 1 when VIDEO_ID is not cached."""
        mocker.patch(
            "tubetalk.cli.main.LocalCacheManager",
            return_value=_make_cache(tmp_path),
        )
        result = runner.invoke(app, ["status", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_shows_detail(self, tmp_path: Path, mocker: Any) -> None:
        """status <VIDEO_ID> should print a detail table."""
        _create_video_cache(
            tmp_path,
            "vid_x",
            title="X Video",
            channel="XChan",
            duration=250.0,
            segments=5,
            vision=True,
        )
        mocker.patch(
            "tubetalk.cli.main.LocalCacheManager",
            return_value=_make_cache(tmp_path),
        )

        result = runner.invoke(app, ["status", "vid_x"])
        assert result.exit_code == 0
        assert "vid_x" in result.output
        assert "X Video" in result.output
        assert "XChan" in result.output
        assert "250s" in result.output
        assert "5" in result.output


# ------------------------------------------------------------------
# process
# ------------------------------------------------------------------


class TestProcess:
    """Tests for initial metadata/transcript ingestion."""

    def test_fetches_and_saves_a_cache_miss(self, tmp_path: Path, mocker: Any) -> None:
        """A cache miss should fetch both resources and persist the results."""
        cache = _make_cache(tmp_path)
        loader = mocker.Mock()
        loader.extract_video_id.return_value = "dQw4w9WgXcQ"
        loader.fetch_metadata.return_value = {
            "title": "Never Gonna Give You Up",
            "channel": "RickAstleyVEVO",
            "duration": 213,
        }
        loader.fetch_transcript.return_value = [
            {"start_sec": 0.0, "duration_sec": 3.0, "text": "We're no strangers"}
        ]
        mocker.patch("tubetalk.cli.main.LocalCacheManager", return_value=cache)
        mocker.patch("tubetalk.cli.main.YouTubeLoader", return_value=loader)

        url = "https://youtu.be/dQw4w9WgXcQ"
        result = runner.invoke(app, ["process", url])

        assert result.exit_code == 0
        assert "Saved 1 transcript segments" in result.output
        loader.fetch_metadata.assert_called_once_with(url)
        loader.fetch_transcript.assert_called_once_with("dQw4w9WgXcQ")
        assert cache.load_json("dQw4w9WgXcQ", "transcript.json") == (
            loader.fetch_transcript.return_value
        )
        metadata = cache.load_json("dQw4w9WgXcQ", "metadata.json")
        assert metadata["video_id"] == "dQw4w9WgXcQ"
        assert metadata["source_url"] == url
        assert metadata["processed_at"]

    def test_uses_existing_cache_without_fetching(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """A complete cache should bypass all remote loader calls."""
        _create_video_cache(tmp_path, "dQw4w9WgXcQ")
        cache = _make_cache(tmp_path)
        loader = mocker.Mock()
        loader.extract_video_id.return_value = "dQw4w9WgXcQ"
        mocker.patch("tubetalk.cli.main.LocalCacheManager", return_value=cache)
        mocker.patch("tubetalk.cli.main.YouTubeLoader", return_value=loader)

        result = runner.invoke(app, ["process", "https://youtu.be/dQw4w9WgXcQ"])

        assert result.exit_code == 0
        assert "Cache hit" in result.output
        loader.fetch_metadata.assert_not_called()
        loader.fetch_transcript.assert_not_called()

    def test_rejects_an_invalid_youtube_url(self, mocker: Any) -> None:
        """An invalid URL should exit without constructing a cache manager."""
        loader = mocker.Mock()
        loader.extract_video_id.side_effect = ValueError("Cannot extract video_id")
        cache_manager = mocker.patch("tubetalk.cli.main.LocalCacheManager")
        mocker.patch("tubetalk.cli.main.YouTubeLoader", return_value=loader)

        result = runner.invoke(app, ["process", "https://example.com/video"])

        assert result.exit_code == 2
        assert "Cannot extract video_id" in result.output
        cache_manager.assert_not_called()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

from tubetalk.core.cache import LocalCacheManager  # noqa: E402


def _make_cache(data_dir: Path) -> LocalCacheManager:
    return LocalCacheManager(data_dir=data_dir)
