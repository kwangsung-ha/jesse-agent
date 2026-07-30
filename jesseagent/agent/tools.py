"""Video-service tools exposed to the natural-language Agent."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from jesseagent.agent.contracts import ToolCall, ToolResult
from jesseagent.domain.video_status import VideoStatus
from jesseagent.services.video_service import (
    ChatSession,
    VideoService,
    VideoServiceError,
)


class ProcessVideoInput(BaseModel):
    url: str = Field(min_length=1)


class VideoIdInput(BaseModel):
    video_id: str = Field(min_length=1)


class SummaryInput(VideoIdInput):
    generate: bool = False


class VideoQuestionInput(VideoIdInput):
    question: str = Field(min_length=1)


class EmptyInput(BaseModel):
    """Arguments for a parameter-free tool."""


class ApprovalRequestInput(BaseModel):
    """A model request to pause before a side-effecting video operation."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


ToolHandler = Callable[[BaseModel], dict[str, Any]]


class VideoToolExecutor:
    """Validate and execute the Agent's bounded service operations."""

    def __init__(self, service: VideoService) -> None:
        self._service = service
        self._sessions: dict[str, ChatSession] = {}
        self.current_video_id: str | None = None
        self._tools: dict[str, tuple[type[BaseModel], ToolHandler]] = {
            "process_video": (ProcessVideoInput, self._process_video),
            "list_videos": (EmptyInput, self._list_videos),
            "get_video_status": (VideoIdInput, self._get_video_status),
            "get_summary": (SummaryInput, self._get_summary),
            "answer_video_question": (VideoQuestionInput, self._answer_question),
            "request_approval": (ApprovalRequestInput, self._request_approval),
        }

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

    def _process_video(self, payload: BaseModel) -> dict[str, Any]:
        result = self._service.process(ProcessVideoInput.model_validate(payload).url)
        self.current_video_id = result.video_id
        return {
            "video_id": result.video_id,
            "cache_hit": result.cache_hit,
            "transcript_segments": result.transcript_segments,
            "indexing_state": result.indexing.state,
            "summary_state": result.summary.state,
            "vision_state": result.vision.state,
        }

    def _list_videos(self, _: BaseModel) -> dict[str, Any]:
        return {
            "videos": [_status_data(item) for item in self._service.list_statuses()]
        }

    def _get_video_status(self, payload: BaseModel) -> dict[str, Any]:
        video_id = VideoIdInput.model_validate(payload).video_id
        self.current_video_id = video_id
        return {"video": _status_data(self._service.get_status(video_id))}

    def _get_summary(self, payload: BaseModel) -> dict[str, Any]:
        request = SummaryInput.model_validate(payload)
        self.current_video_id = request.video_id
        result = self._service.get_summary(request.video_id, generate=request.generate)
        if result.summary is None:
            return {"state": result.state, "summary": None}
        return {
            "state": result.state,
            "summary": result.summary.text,
            "chapters": [chapter.model_dump() for chapter in result.summary.chapters],
        }

    def _answer_question(self, payload: BaseModel) -> dict[str, Any]:
        request = VideoQuestionInput.model_validate(payload)
        self.current_video_id = request.video_id
        session = self._sessions.get(request.video_id)
        if session is None:
            session = self._service.create_chat_session(request.video_id)
            self._sessions[request.video_id] = session
        answer = session.ask(request.question)
        evidence = {item.source_id: item for item in session.last_evidence}
        return {
            "answer": answer.answer,
            "citations": [
                {
                    **citation.model_dump(),
                    "source": evidence[citation.source_id].source,
                    "evidence": evidence[citation.source_id].text,
                }
                for citation in answer.citations
            ],
        }

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
    "process_video": (
        "Fetch and index a YouTube URL, including summary and vision data."
    ),
    "list_videos": "List videos currently cached by JesseAgent.",
    "get_video_status": "Show cache and index status for one cached video.",
    "get_summary": (
        "Read a cached summary, or generate it only when explicitly requested."
    ),
    "answer_video_question": (
        "Answer one grounded question about a cached, indexed video."
    ),
    "request_approval": (
        "Request explicit approval before a costly or mutating operation."
    ),
}


def _status_data(status: VideoStatus) -> dict[str, Any]:
    """Return concise JSON-safe status data suitable for the Agent context."""
    return {
        "video_id": status.video_id,
        "title": status.title,
        "channel": status.channel,
        "duration": status.duration,
        "transcript_segments": status.transcript_segments,
        "transcript_index_state": status.transcript_index_state,
        "summary_state": status.summary_state,
        "vision_index_state": status.vision_index_state,
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
