"""Loading and rendering of the versioned prompt templates."""

from importlib import resources
from string import Template
from typing import Mapping


class PromptTemplateError(ValueError):
    """A configured prompt template cannot be loaded or rendered."""


class PromptCatalog:
    """Render package templates selected by a function-specific version."""

    _KIND_DIRECTORIES = {
        "summary": "summary",
        "summary_correction": "summary",
        "chapter_candidates": "summary",
        "chapter_candidates_correction": "summary",
        "vision": "vision",
        "chat": "chat",
        "chat_correction": "chat",
    }

    def render(self, kind: str, version: str, values: Mapping[str, object]) -> str:
        """Return one fully substituted template or a clear configuration error."""
        directory = self._KIND_DIRECTORIES.get(kind)
        if directory is None:
            raise PromptTemplateError(f"Unknown prompt kind '{kind}'")
        resource_name = f"{directory}/{version}.txt"
        try:
            source = (
                resources.files("tubetalk.prompts")
                .joinpath(resource_name)
                .read_text(encoding="utf-8")
            )
        except FileNotFoundError as error:
            raise PromptTemplateError(
                f"Prompt template '{kind}:{version}' was not found"
            ) from error
        try:
            return Template(source).substitute(
                {key: str(value) for key, value in values.items()}
            )
        except (KeyError, ValueError) as error:
            raise PromptTemplateError(
                f"Prompt template '{kind}:{version}' could not be rendered: {error}"
            ) from error
