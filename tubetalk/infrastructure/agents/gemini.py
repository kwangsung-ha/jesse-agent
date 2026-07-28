"""Gemini native-function-calling adapter for the TubeTalk Agent."""

from typing import Any, Optional, cast

from google import genai
from google.genai import types
from google.genai.errors import APIError
from httpx import HTTPError

from tubetalk.agent.contracts import AgentDecision, AgentMessage, ToolCall
from tubetalk.agent.orchestrator import AgentModelError
from tubetalk.core.prompts import PromptCatalog, PromptTemplateError


class GeminiAgentModel:
    """Translate Gemini function calls into the application's tool-call contract."""

    def __init__(
        self,
        api_key: str,
        model: str,
        prompt_version: str,
        client: Optional[Any] = None,
        prompts: PromptCatalog | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to run the Agent")
        self._model = model
        self._client = client or genai.Client(api_key=api_key)
        self._prompt_version = prompt_version
        self._prompts = prompts or PromptCatalog()

    def decide(
        self,
        messages: tuple[AgentMessage, ...],
        declarations: tuple[dict[str, object], ...],
        current_video_id: str | None,
    ) -> AgentDecision:
        """Ask Gemini for a native function call or final response."""
        try:
            instruction = self._prompts.render(
                "agent",
                self._prompt_version,
                {"current_video_id": current_video_id or "(none)"},
            )
        except PromptTemplateError as error:
            raise AgentModelError(str(error)) from error
        transcript = "\n\n".join(
            f"{message.role.upper()}: {message.content}" for message in messages
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    tools=[
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(
                                    name=cast(str, declaration["name"]),
                                    description=cast(str, declaration["description"]),
                                    parameters_json_schema=cast(
                                        dict[str, Any],
                                        declaration["parameters_json_schema"],
                                    ),
                                )
                                for declaration in declarations
                            ]
                        )
                    ],
                ),
            )
        except (APIError, HTTPError) as error:
            raise AgentModelError(str(error)) from error
        calls = tuple(
            ToolCall(name=call.name, arguments=dict(call.args or {}))
            for call in (getattr(response, "function_calls", None) or ())
        )
        return AgentDecision(text=getattr(response, "text", "") or "", tool_calls=calls)
