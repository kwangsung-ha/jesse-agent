"""Agent loop tests with deterministic model and tool substitutes."""

from pathlib import Path
from typing import Any

import pytest

from jesseagent.agent.contracts import AgentDecision
from jesseagent.agent.orchestrator import AgentSession
from jesseagent.agent.reducer import AgentRunReductionError, model_messages, reduce_run
from jesseagent.agent.runs import (
    AgentEventType,
    AgentRun,
    AgentRunEvent,
    NewAgentRunEvent,
)
from jesseagent.infrastructure.sqlite.agent_runs import (
    SQLiteAgentRunRepository,
)
from jesseagent.tools.contracts import ToolCall, ToolResult


class StubTools:
    declarations: tuple[dict[str, Any], ...] = ()
    current_video_id: str | None = None

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(name=call.name, ok=True, content={"video_id": "video1"})


class SequenceModel:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = decisions
        self.calls = 0

    def decide(self, *_: Any) -> AgentDecision:
        decision = self._decisions[self.calls]
        self.calls += 1
        return decision


def test_agent_executes_tools_then_returns_final_response() -> None:
    tools = StubTools()
    model = SequenceModel(
        [
            AgentDecision(tool_calls=(ToolCall(name="list_videos"),)),
            AgentDecision(text="캐시된 영상은 하나입니다."),
        ]
    )

    answer = AgentSession(model, tools, max_steps=3).ask("영상 목록 보여줘")

    assert answer == "캐시된 영상은 하나입니다."
    assert [call.name for call in tools.calls] == ["list_videos"]


def test_agent_stops_after_configured_tool_step_limit() -> None:
    tools = StubTools()
    model = SequenceModel([AgentDecision(tool_calls=(ToolCall(name="list_videos"),))])

    answer = AgentSession(model, tools, max_steps=1).ask("영상 목록 보여줘")

    assert "단계 제한" in answer


def test_agent_reconstructs_context_and_state_from_durable_events(
    tmp_path: Path,
) -> None:
    """A fresh session can rebuild the same model input from the persisted run."""
    repository = SQLiteAgentRunRepository(tmp_path / "runs.sqlite3")
    tools = StubTools()
    model = SequenceModel(
        [
            AgentDecision(tool_calls=(ToolCall(name="list_videos"),)),
            AgentDecision(text="완료"),
        ]
    )
    session = AgentSession(
        model, tools, max_steps=3, repository=repository, run_id="durable-run"
    )

    assert session.ask("영상 목록 보여줘") == "완료"

    run = repository.get_run("durable-run")
    events = repository.list_events("durable-run")
    state = reduce_run(run, events)

    assert state.status == "completed"
    assert [message.role for message in model_messages(events)] == ["user", "tool"]
    assert model_messages(events)[0].content == "영상 목록 보여줘"


def test_reducer_rejects_gapped_or_wrong_run_events() -> None:
    """A durable state cannot be reconstructed from an invalid event log."""
    run = AgentRun(run_id="run")
    invalid = NewAgentRunEvent(run_id="other", event_type=AgentEventType.USER_REQUEST)

    with pytest.raises(AgentRunReductionError):
        reduce_run(run, (AgentRunEvent(sequence=2, **invalid.model_dump()),))


def test_resumed_session_does_not_repeat_a_completed_tool_call(tmp_path: Path) -> None:
    """Resume asks for the next decision after persisted tool output."""
    repository = SQLiteAgentRunRepository(tmp_path / "runs.sqlite3")
    run = AgentRun(run_id="resume")
    repository.create_run(run)
    repository.append_event(
        NewAgentRunEvent(
            run_id=run.run_id,
            event_type=AgentEventType.USER_REQUEST,
            payload={"content": "목록 보여줘"},
        )
    )
    repository.append_event(
        NewAgentRunEvent(
            run_id=run.run_id,
            event_type=AgentEventType.TOOL_CALL,
            payload={"name": "list_videos", "arguments": {}},
        )
    )
    repository.append_event(
        NewAgentRunEvent(
            run_id=run.run_id,
            event_type=AgentEventType.TOOL_RESULT,
            payload={"name": "list_videos", "ok": True, "content": {}},
        )
    )
    repository.append_event(
        NewAgentRunEvent(run_id=run.run_id, event_type=AgentEventType.PAUSED)
    )
    tools = StubTools()
    session = AgentSession(
        SequenceModel([AgentDecision(text="재개 완료")]),
        tools,
        max_steps=2,
        repository=repository,
        existing_run=run,
    )

    assert session.resume() == "재개 완료"
    assert tools.calls == []
