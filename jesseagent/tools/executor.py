"""Central validation, approval, and dispatch for registered Agent tools."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from jesseagent.application.knowledge.search import KnowledgeSearchService
from jesseagent.application.video.service import VideoService, VideoServiceError
from jesseagent.tools.contracts import ToolCall, ToolResult
from jesseagent.tools.knowledge import (
    KNOWLEDGE_TOOL_DESCRIPTIONS,
    KnowledgeSearchInput,
    KnowledgeTools,
)
from jesseagent.tools.video import (
    VIDEO_TOOL_DESCRIPTIONS,
    EmptyInput,
    ProcessVideoInput,
    SummaryInput,
    VideoIdInput,
    VideoQuestionInput,
    VideoTools,
)

ToolHandler = Callable[[BaseModel], dict[str, Any]]


class ApprovalRequestInput(BaseModel):
    """A model request to pause before a side-effecting operation."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutor:
    """Validate and dispatch the Agent's bounded application operations."""

    def __init__(
        self,
        video_service: VideoService,
        knowledge_search: KnowledgeSearchService | None = None,
    ) -> None:
        self._video = VideoTools(video_service)
        self._knowledge = KnowledgeTools(knowledge_search)
        self._tools: dict[str, tuple[type[BaseModel], ToolHandler]] = {
            "process_video": (ProcessVideoInput, self._video.process_video),
            "list_videos": (EmptyInput, self._video.list_videos),
            "get_video_status": (VideoIdInput, self._video.get_video_status),
            "get_summary": (SummaryInput, self._video.get_summary),
            "answer_video_question": (
                VideoQuestionInput,
                self._video.answer_video_question,
            ),
            "request_approval": (ApprovalRequestInput, self._request_approval),
            "search_knowledge": (
                KnowledgeSearchInput,
                self._knowledge.search_knowledge,
            ),
        }

    @property
    def current_video_id(self) -> str | None:
        """Return the video context most recently selected by a video tool."""
        return self._video.current_video_id

    @property
    def declarations(self) -> tuple[dict[str, Any], ...]:
        """Return provider-neutral function declarations derived from Pydantic."""
        return tuple(
            {
                "name": name,
                "description": _TOOL_DESCRIPTIONS[name],
                "parameters_json_schema": input_type.model_json_schema(),
            }
            for name, (input_type, _) in self._tools.items()
        )

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate a model request and return expected errors as context."""
        entry = self._tools.get(call.name)
        if entry is None:
            return _failure(
                call,
                code="unknown_tool",
                message=f"Unknown tool '{call.name}'.",
                next_action="Choose one of the declared tools.",
            )
        input_type, handler = entry
        try:
            if self._requires_approval(call):
                return _failure(
                    call,
                    code="approval_required",
                    message="This operation requires explicit approval.",
                    next_action="Request approval before executing this operation.",
                )
            result = handler(input_type.model_validate(call.arguments))
        except ValidationError as error:
            return _failure(
                call,
                code="invalid_arguments",
                message=f"Invalid tool arguments: {error}",
                next_action="Correct the tool arguments and try again.",
            )
        except VideoServiceError as error:
            return _failure(
                call,
                code="video_service_error",
                message=str(error),
                next_action="Explain the issue and suggest a valid video request.",
            )
        return ToolResult(
            name=call.name,
            call_id=call.call_id,
            ok=True,
            content=result,
            user_summary=f"{call.name} completed.",
        )

    def _request_approval(self, payload: BaseModel) -> dict[str, Any]:
        request = ApprovalRequestInput.model_validate(payload)
        call = ToolCall(name=request.tool_name, arguments=request.arguments)
        if not self._requires_approval(call):
            raise VideoServiceError("This operation does not require approval")
        return {"tool_name": request.tool_name, "arguments": request.arguments}

    @staticmethod
    def _requires_approval(call: ToolCall) -> bool:
        if call.name == "process_video":
            return True
        return call.name == "get_summary" and call.arguments.get("generate") is True


_TOOL_DESCRIPTIONS = {
    **VIDEO_TOOL_DESCRIPTIONS,
    "request_approval": (
        "Request explicit approval before a costly or mutating operation."
    ),
    **KNOWLEDGE_TOOL_DESCRIPTIONS,
}


def _failure(
    call: ToolCall, *, code: str, message: str, next_action: str
) -> ToolResult:
    """Return one compact, typed failure contract for Agent recovery."""
    return ToolResult(
        name=call.name,
        call_id=call.call_id,
        ok=False,
        content={"error": message},
        error_code=code,
        user_summary=message,
        next_action=next_action,
    )
