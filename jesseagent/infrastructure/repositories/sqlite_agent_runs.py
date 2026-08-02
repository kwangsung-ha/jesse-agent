"""SQLite implementation of the durable Agent run event log."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from jesseagent.agent.runs import (
    AgentEventType,
    AgentRun,
    AgentRunEvent,
    NewAgentRunEvent,
)
from jesseagent.application.agent_runs.contracts import (
    AgentRunNotFoundError,
    AgentRunRepositoryError,
)


class SQLiteAgentRunRepository:
    """Persist Agent runs as append-only, per-run ordered SQLite events."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_run(self, run: AgentRun) -> None:
        """Create an empty durable run."""
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO agent_runs (run_id, created_at) VALUES (?, ?)",
                    (run.run_id, run.created_at.isoformat()),
                )
        except sqlite3.IntegrityError as error:
            raise AgentRunRepositoryError(
                f"Agent run '{run.run_id}' already exists"
            ) from error
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error

    def append_event(self, event: NewAgentRunEvent) -> AgentRunEvent:
        """Append an event in one transaction and assign the next sequence."""
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._require_run(connection, event.run_id)
                row = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence "
                    "FROM agent_events WHERE run_id = ?",
                    (event.run_id,),
                ).fetchone()
                sequence = int(row["sequence"])
                connection.execute(
                    "INSERT INTO agent_events "
                    "(run_id, sequence, event_type, payload, occurred_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        event.run_id,
                        sequence,
                        event.event_type.value,
                        json.dumps(
                            event.payload, ensure_ascii=False, separators=(",", ":")
                        ),
                        event.occurred_at.isoformat(),
                    ),
                )
        except AgentRunNotFoundError:
            raise
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error
        return AgentRunEvent(sequence=sequence, **event.model_dump())

    def get_run(self, run_id: str) -> AgentRun:
        """Load one run."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT run_id, created_at FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error
        if row is None:
            raise AgentRunNotFoundError(f"Agent run '{run_id}' was not found")
        return AgentRun(
            run_id=str(row["run_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    def list_runs(self) -> tuple[AgentRun, ...]:
        """Load all runs in descending creation order."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT run_id, created_at FROM agent_runs "
                    "ORDER BY created_at DESC, run_id DESC"
                ).fetchall()
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error
        return tuple(
            AgentRun(
                run_id=str(row["run_id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
            )
            for row in rows
        )

    def list_events(self, run_id: str) -> tuple[AgentRunEvent, ...]:
        """Load a run's events in durable append order."""
        try:
            with self._connect() as connection:
                self._require_run(connection, run_id)
                rows = connection.execute(
                    "SELECT sequence, event_type, payload, occurred_at "
                    "FROM agent_events "
                    "WHERE run_id = ? ORDER BY sequence ASC",
                    (run_id,),
                ).fetchall()
        except AgentRunNotFoundError:
            raise
        except (json.JSONDecodeError, sqlite3.Error) as error:
            raise AgentRunRepositoryError(str(error)) from error
        return tuple(
            AgentRunEvent(
                run_id=run_id,
                sequence=int(row["sequence"]),
                event_type=AgentEventType(str(row["event_type"])),
                payload=json.loads(str(row["payload"])),
                occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            )
            for row in rows
        )

    def delete_run(self, run_id: str) -> None:
        """Delete one run and its events through a foreign-key cascade."""
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM agent_runs WHERE run_id = ?", (run_id,)
                )
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error
        if cursor.rowcount != 1:
            raise AgentRunNotFoundError(f"Agent run '{run_id}' was not found")

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS agent_events (
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, sequence),
                        FOREIGN KEY (run_id) REFERENCES agent_runs (run_id)
                            ON DELETE CASCADE
                    );
                    """
                )
        except sqlite3.Error as error:
            raise AgentRunRepositoryError(str(error)) from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AgentRunNotFoundError(f"Agent run '{run_id}' was not found")
