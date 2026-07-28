"""Deterministic control flow around an LLM's native tool-call decisions."""

import json
from collections.abc import Callable
from typing import Protocol

from tubetalk.agent.contracts import AgentDecision, AgentMessage, ToolResult
from tubetalk.agent.tools import VideoToolExecutor


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
    """A process-local multi-turn session with bounded tool execution."""

    def __init__(
        self,
        model: AgentModel,
        tools: VideoToolExecutor,
        max_steps: int,
        on_tool_result: Callable[[ToolResult], None] | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._on_tool_result = on_tool_result
        self._messages: list[AgentMessage] = []

    def ask(self, request: str) -> str:
        """Execute tool calls until the model returns a final response."""
        if not request.strip():
            return "무엇을 도와드릴까요?"
        self._messages.append(AgentMessage(role="user", content=request))
        for _ in range(self._max_steps):
            try:
                decision = self._model.decide(
                    tuple(self._messages),
                    self._tools.declarations,
                    self._tools.current_video_id,
                )
            except AgentModelError as error:
                return f"Agent model error: {error}"
            if not decision.tool_calls:
                return decision.text or "요청을 완료했지만 응답을 생성하지 못했습니다."
            for call in decision.tool_calls:
                result = self._tools.execute(call)
                self._messages.append(
                    AgentMessage(
                        role="tool",
                        content=json.dumps(
                            result.model_dump(mode="json"), ensure_ascii=False
                        ),
                    )
                )
                if self._on_tool_result is not None:
                    self._on_tool_result(result)
        return "도구 호출 단계 제한에 도달했습니다. 요청을 더 작게 나누어 주세요."
