"""Gemini native-function-call adapter tests with mocked transport."""

from types import SimpleNamespace
from typing import Any

from jesseagent.agent.contracts import AgentMessage
from jesseagent.infrastructure.gemini.agent_model import GeminiAgentModel


def test_gemini_agent_translates_native_function_call(mocker: Any) -> None:
    client = mocker.Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="",
        function_calls=(SimpleNamespace(name="list_videos", args={}),),
    )
    model = GeminiAgentModel("key", "model", "tool-agent-v1", client=client)

    decision = model.decide(
        (AgentMessage(role="user", content="목록 보여줘"),),
        (
            {
                "name": "list_videos",
                "description": "List cached videos.",
                "parameters_json_schema": {"type": "object"},
            },
        ),
        None,
    )

    assert decision.tool_calls[0].name == "list_videos"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.tools[0].function_declarations[0].name == "list_videos"


def test_gemini_agent_returns_text_when_no_tool_is_called(mocker: Any) -> None:
    client = mocker.Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="도와드릴게요.", function_calls=()
    )
    model = GeminiAgentModel("key", "model", "tool-agent-v1", client=client)

    decision = model.decide((), (), "video1")

    assert decision.text == "도와드릴게요."
