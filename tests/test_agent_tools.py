"""Deterministic Agent-tool tests using a mocked application service."""

from types import SimpleNamespace
from typing import Any

from tubetalk.agent.contracts import ToolCall
from tubetalk.agent.tools import VideoToolExecutor
from tubetalk.services.video_service import VideoNotFoundError


def test_process_tool_calls_service_and_sets_current_video(mocker: Any) -> None:
    service = mocker.Mock()
    service.process.return_value = SimpleNamespace(
        video_id="video1",
        cache_hit=False,
        transcript_segments=3,
        indexing=SimpleNamespace(state="indexed"),
        summary=SimpleNamespace(state="generated"),
        vision=SimpleNamespace(state="generated"),
    )
    tools = VideoToolExecutor(service)

    result = tools.execute(
        ToolCall(name="process_video", arguments={"url": "https://youtu.be/video1"})
    )

    assert result.ok is True
    assert result.call_id
    assert result.user_summary == "process_video completed."
    assert result.content["video_id"] == "video1"
    assert tools.current_video_id == "video1"
    service.process.assert_called_once_with("https://youtu.be/video1")


def test_tool_validation_and_service_errors_are_returned_as_context(
    mocker: Any,
) -> None:
    service = mocker.Mock()
    service.get_status.side_effect = VideoNotFoundError("missing")
    tools = VideoToolExecutor(service)

    invalid = tools.execute(ToolCall(name="get_video_status", arguments={}))
    missing = tools.execute(
        ToolCall(name="get_video_status", arguments={"video_id": "missing"})
    )
    unknown = tools.execute(ToolCall(name="made_up_tool"))

    assert invalid.ok is False
    assert "Invalid tool arguments" in invalid.content["error"]
    assert missing.content["error"] == "missing"
    assert unknown.content["error"] == "Unknown tool 'made_up_tool'."
    assert invalid.error_code == "invalid_arguments"
    assert missing.error_code == "video_service_error"
    assert unknown.next_action == "Choose one of the declared tools."


def test_tool_declarations_expose_only_bounded_video_operations(mocker: Any) -> None:
    tools = VideoToolExecutor(mocker.Mock())

    names = {item["name"] for item in tools.declarations}

    assert names == {
        "process_video",
        "list_videos",
        "get_video_status",
        "get_summary",
        "answer_video_question",
    }


def test_list_status_summary_and_question_tools_return_service_data(
    mocker: Any,
) -> None:
    service = mocker.Mock()
    status = SimpleNamespace(
        video_id="video1",
        title="Title",
        channel="Channel",
        duration=120.0,
        transcript_segments=3,
        transcript_index_state="current",
        summary_state="current",
        vision_index_state="current",
    )
    service.list_statuses.return_value = [status]
    service.get_status.return_value = status
    service.get_summary.return_value = SimpleNamespace(
        state="current",
        summary=SimpleNamespace(
            text="요약",
            chapters=(SimpleNamespace(model_dump=lambda: {"start_sec": 0}),),
        ),
    )
    chat_session = mocker.Mock()
    chat_session.ask.return_value = SimpleNamespace(
        answer="답변",
        citations=(
            SimpleNamespace(
                source_id="source",
                model_dump=lambda: {"source_id": "source", "timestamp_sec": 1},
            ),
        ),
    )
    chat_session.last_evidence = (
        SimpleNamespace(source_id="source", source="transcript", text="근거"),
    )
    service.create_chat_session.return_value = chat_session
    tools = VideoToolExecutor(service)

    listed = tools.execute(ToolCall(name="list_videos"))
    detail = tools.execute(
        ToolCall(name="get_video_status", arguments={"video_id": "video1"})
    )
    summary = tools.execute(
        ToolCall(name="get_summary", arguments={"video_id": "video1", "generate": True})
    )
    answer = tools.execute(
        ToolCall(
            name="answer_video_question",
            arguments={"video_id": "video1", "question": "무슨 내용이야?"},
        )
    )

    assert listed.content["videos"][0]["title"] == "Title"
    assert detail.content["video"]["video_id"] == "video1"
    assert summary.content["chapters"] == [{"start_sec": 0}]
    assert answer.content["citations"][0]["evidence"] == "근거"
    service.get_summary.assert_called_once_with("video1", generate=True)
