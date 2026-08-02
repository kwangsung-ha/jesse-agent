"""Video tool inputs and handlers over the video application service."""

from typing import Any

from pydantic import BaseModel, Field

from jesseagent.application.video.service import ChatSession, VideoService
from jesseagent.domain.video_status import VideoStatus


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


class VideoTools:
    """Translate validated video tool inputs into application service calls."""

    def __init__(self, service: VideoService) -> None:
        self._service = service
        self._sessions: dict[str, ChatSession] = {}
        self.current_video_id: str | None = None

    def process_video(self, payload: BaseModel) -> dict[str, Any]:
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

    def list_videos(self, _: BaseModel) -> dict[str, Any]:
        return {
            "videos": [_status_data(item) for item in self._service.list_statuses()]
        }

    def get_video_status(self, payload: BaseModel) -> dict[str, Any]:
        video_id = VideoIdInput.model_validate(payload).video_id
        self.current_video_id = video_id
        return {"video": _status_data(self._service.get_status(video_id))}

    def get_summary(self, payload: BaseModel) -> dict[str, Any]:
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

    def answer_video_question(self, payload: BaseModel) -> dict[str, Any]:
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


VIDEO_TOOL_DESCRIPTIONS = {
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
