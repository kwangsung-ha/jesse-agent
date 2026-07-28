"""Agent loop tests with deterministic model and tool substitutes."""

from typing import Any

from tubetalk.agent.contracts import AgentDecision, ToolCall, ToolResult
from tubetalk.agent.orchestrator import AgentSession


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
