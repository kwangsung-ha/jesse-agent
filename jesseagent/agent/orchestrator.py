"""Deterministic, event-backed control flow around Agent tool decisions."""

from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from jesseagent.agent.context import AgentContextBudget, compact_messages
from jesseagent.agent.contracts import AgentDecision, AgentMessage, ToolResult
from jesseagent.agent.reducer import model_messages, reduce_run
from jesseagent.agent.runs import (
    AgentEventType,
    AgentRun,
    AgentRunEvent,
    NewAgentRunEvent,
)
from jesseagent.agent.tools import VideoToolExecutor
from jesseagent.core.logging import logger
from jesseagent.ports.agent_run_repository import AgentRunRepository


class AgentModelError(Exception):
    """Raised when the model cannot produce a usable Agent decision."""


class AgentModel(Protocol):
    """Choose the next native tool call or return a final natural-language reply."""

    def decide(
        self,
        messages: tuple[AgentMessage, ...],
        declarations: tuple[dict[str, object], ...],
        current_video_id: str | None,
    ) -> AgentDecision:
        """Return one model decision for the supplied compact conversation."""


class AgentSession:
    """A bounded Agent execution reconstructed from durable events when configured."""

    def __init__(
        self,
        model: AgentModel,
        tools: VideoToolExecutor,
        max_steps: int,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        repository: AgentRunRepository | None = None,
        run_id: str | None = None,
        existing_run: AgentRun | None = None,
        context_budget: AgentContextBudget = AgentContextBudget(),
    ) -> None:
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._on_tool_result = on_tool_result
        self._repository = repository
        self._context_budget = context_budget
        self._events: list[NewAgentRunEvent] = []
        if repository is None:
            if existing_run is not None:
                raise ValueError("An existing run requires durable persistence")
            self._run: AgentRun | None = None
        else:
            self._run = existing_run or AgentRun(run_id=run_id or uuid4().hex)
            if existing_run is None:
                repository.create_run(self._run)

    @property
    def run_id(self) -> str | None:
        """Return the durable run identifier when persistence is configured."""
        return self._run.run_id if self._run is not None else None

    def ask(self, request: str) -> str:
        """Execute tool calls until the model returns a final response."""
        if not request.strip():
            return "무엇을 도와드릴까요?"
        self._append(AgentEventType.USER_REQUEST, {"content": request})
        return self.resume()

    def resume(self) -> str:
        """Continue a persisted run without repeating completed tool calls."""
        for _ in range(self._max_steps):
            if self._run is None:
                messages = compact_messages(
                    self._legacy_messages(tuple(self._events)), self._context_budget
                )
                current_video_id = self._tools.current_video_id
            else:
                events = self._persistent_events()
                state = reduce_run(self._run, events)
                messages = compact_messages(
                    model_messages(events), self._context_budget
                )
                current_video_id = state.current_video_id
            try:
                decision = self._model.decide(
                    messages,
                    self._tools.declarations,
                    current_video_id,
                )
            except AgentModelError as error:
                response = f"Agent model error: {error}"
                self._append(AgentEventType.FAILURE, {"message": response})
                return response
            self._append(
                AgentEventType.MODEL_DECISION,
                decision.model_dump(mode="json"),
            )
            logger.bind(event="agent.decision", run_id=self.run_id).debug(
                "tool_calls={}", len(decision.tool_calls)
            )
            if not decision.tool_calls:
                response = (
                    decision.text or "요청을 완료했지만 응답을 생성하지 못했습니다."
                )
                self._append(AgentEventType.FINAL_RESPONSE, {"text": response})
                return response
            for call in decision.tool_calls:
                self._append(AgentEventType.TOOL_CALL, call.model_dump(mode="json"))
                result = self._tools.execute(call)
                self._append(
                    AgentEventType.TOOL_RESULT,
                    result.model_dump(mode="json"),
                )
                if result.error_code == "approval_required":
                    self._append(
                        AgentEventType.APPROVAL_REQUESTED,
                        {"call_id": call.call_id, "tool_name": call.name},
                    )
                    return (
                        "이 작업은 비용 또는 데이터 변경을 수반합니다. 승인해 주세요."
                    )
                if result.name == "request_approval" and result.ok:
                    self._append(
                        AgentEventType.APPROVAL_REQUESTED,
                        result.content,
                    )
                    return "작업 실행 전에 승인이 필요합니다. 승인해 주세요."
                logger.bind(event="agent.tool_result", run_id=self.run_id).debug(
                    "tool={} ok={}", result.name, result.ok
                )
                if self._on_tool_result is not None:
                    self._on_tool_result(result)
        response = "도구 호출 단계 제한에 도달했습니다. 요청을 더 작게 나누어 주세요."
        self._append(AgentEventType.FAILURE, {"message": response})
        return response

    def _append(
        self, event_type: AgentEventType, payload: dict[str, JsonValue]
    ) -> None:
        if self._run is None:
            self._events.append(
                NewAgentRunEvent(
                    run_id="process-local", event_type=event_type, payload=payload
                )
            )
            return
        repository = self._repository
        if repository is None:
            raise RuntimeError("A durable Agent run requires a repository")
        repository.append_event(
            NewAgentRunEvent(
                run_id=self._run.run_id, event_type=event_type, payload=payload
            )
        )

    def _persistent_events(self) -> tuple[AgentRunEvent, ...]:
        if self._run is None or self._repository is None:
            raise RuntimeError("No durable Agent run is configured")
        return self._repository.list_events(self._run.run_id)

    @staticmethod
    def _legacy_messages(
        events: tuple[NewAgentRunEvent, ...],
    ) -> tuple[AgentMessage, ...]:
        return tuple(
            AgentMessage(role="user", content=str(event.payload["content"]))
            if event.event_type == AgentEventType.USER_REQUEST
            else AgentMessage(role="tool", content=str(event.payload))
            for event in events
            if event.event_type
            in {AgentEventType.USER_REQUEST, AgentEventType.TOOL_RESULT}
        )
