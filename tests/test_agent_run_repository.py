"""Unit tests for the SQLite append-only Agent run repository."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jesseagent.agent.runs import AgentEventType, AgentRun, NewAgentRunEvent
from jesseagent.application.agent_runs.contracts import (
    AgentRunNotFoundError,
    AgentRunRepositoryError,
)
from jesseagent.infrastructure.repositories.sqlite_agent_runs import (
    SQLiteAgentRunRepository,
)


def _repository(tmp_path: Path) -> SQLiteAgentRunRepository:
    return SQLiteAgentRunRepository(tmp_path / "agent_runs.sqlite3")


def _run(run_id: str = "run-1", minutes: int = 0) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        created_at=datetime(2026, 7, 28, tzinfo=timezone.utc)
        + timedelta(minutes=minutes),
    )


def test_create_and_get_run_round_trip(tmp_path: Path) -> None:
    """A created run survives a new repository instance."""
    repository = _repository(tmp_path)
    run = _run()
    repository.create_run(run)

    loaded = _repository(tmp_path).get_run(run.run_id)

    assert loaded == run


def test_create_run_rejects_duplicate_id(tmp_path: Path) -> None:
    """Run IDs are unique durable identifiers."""
    repository = _repository(tmp_path)
    repository.create_run(_run())

    with pytest.raises(AgentRunRepositoryError, match="already exists"):
        repository.create_run(_run())


def test_list_runs_returns_newest_first(tmp_path: Path) -> None:
    """Run listing is deterministic for CLI inspection."""
    repository = _repository(tmp_path)
    repository.create_run(_run("older"))
    repository.create_run(_run("newer", minutes=1))

    assert [run.run_id for run in repository.list_runs()] == ["newer", "older"]


def test_append_event_assigns_monotonic_sequence_and_persists_payload(
    tmp_path: Path,
) -> None:
    """Each append receives the next per-run sequence in one durable log."""
    repository = _repository(tmp_path)
    repository.create_run(_run())

    first = repository.append_event(
        NewAgentRunEvent(
            run_id="run-1",
            event_type=AgentEventType.USER_REQUEST,
            payload={"text": "안녕"},
        )
    )
    second = repository.append_event(
        NewAgentRunEvent(
            run_id="run-1",
            event_type=AgentEventType.FINAL_RESPONSE,
            payload={"text": "반가워"},
        )
    )

    assert (first.sequence, second.sequence) == (1, 2)
    assert _repository(tmp_path).list_events("run-1") == (first, second)


def test_append_and_list_events_reject_unknown_run(tmp_path: Path) -> None:
    """Events cannot be orphaned from a durable run."""
    repository = _repository(tmp_path)

    with pytest.raises(AgentRunNotFoundError):
        repository.append_event(
            NewAgentRunEvent(run_id="missing", event_type=AgentEventType.FAILURE)
        )
    with pytest.raises(AgentRunNotFoundError):
        repository.list_events("missing")


def test_delete_run_removes_its_events_and_requires_an_existing_run(
    tmp_path: Path,
) -> None:
    """Explicit deletion cascades to events without affecting other runs."""
    repository = _repository(tmp_path)
    repository.create_run(_run("first"))
    repository.create_run(_run("second"))
    repository.append_event(
        NewAgentRunEvent(run_id="first", event_type=AgentEventType.FAILURE)
    )

    repository.delete_run("first")

    assert [run.run_id for run in repository.list_runs()] == ["second"]
    with pytest.raises(AgentRunNotFoundError):
        repository.get_run("first")
    with pytest.raises(AgentRunNotFoundError):
        repository.delete_run("first")
