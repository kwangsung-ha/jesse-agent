"""Pure Markdown parsing and heading-first knowledge chunking."""

from re import DOTALL, MULTILINE, compile
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument

_FRONTMATTER = compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", MULTILINE | DOTALL)
_HEADING = compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", MULTILINE)


class ObsidianChunkPolicy(BaseModel):
    """Bound only sections that are too large for focused retrieval."""

    model_config = ConfigDict(frozen=True)

    max_characters: int = Field(default=1200, gt=0)
    overlap_characters: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ObsidianChunkPolicy":
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")
        return self


def chunk_markdown(
    document: KnowledgeDocument,
    policy: ObsidianChunkPolicy = ObsidianChunkPolicy(),
) -> tuple[KnowledgeChunk, ...]:
    """Create heading-first chunks, splitting only oversized section bodies."""
    _, body = split_frontmatter(document.content)
    sections = _heading_sections(body)
    chunks: list[KnowledgeChunk] = []
    for headings, text in sections:
        for part in _split_section(text, headings, policy):
            metadata = cast(
                dict[str, JsonValue], {**document.metadata, "heading_path": headings}
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.document_id}:chunk:{len(chunks)}",
                    document_id=document.document_id,
                    ordinal=len(chunks),
                    text=part,
                    metadata=metadata,
                )
            )
    return tuple(chunks)


def split_frontmatter(content: str) -> tuple[dict[str, str | list[str]], str]:
    """Return parsed minimal YAML-like frontmatter and the remaining Markdown."""
    match = _FRONTMATTER.match(content)
    if match is None:
        return {}, content
    values: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    for line in match.group(1).splitlines():
        if line.startswith("  - ") and current_list_key:
            existing = values.setdefault(current_list_key, [])
            if isinstance(existing, list):
                existing.append(_unquote(line[4:].strip()))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", 1)
        current_list_key = key.strip()
        value = value.strip()
        values[current_list_key] = _parse_value(value)
    return values, content[match.end() :]


def first_heading(content: str) -> str | None:
    """Return the first Markdown heading text when present."""
    match = _HEADING.search(content)
    return match.group(2).strip() if match else None


def _parse_value(value: str) -> str | list[str]:
    if value.startswith("[") and value.endswith("]"):
        return [
            _unquote(item.strip()) for item in value[1:-1].split(",") if item.strip()
        ]
    return _unquote(value)


def _unquote(value: str) -> str:
    return value.strip().strip("\"'")


def _heading_sections(content: str) -> list[tuple[list[str], str]]:
    matches = list(_HEADING.finditer(content))
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    if matches and content[: matches[0].start()].strip():
        sections.append(([], content[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(heading)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[match.end() : end].strip()
        text = "\n\n".join([" > ".join(heading_stack), body]).strip()
        if text:
            sections.append((heading_stack.copy(), text))
    if not matches and content.strip():
        sections.append(([], content.strip()))
    return sections


def _split_section(
    text: str, headings: list[str], policy: ObsidianChunkPolicy
) -> list[str]:
    if len(text) <= policy.max_characters:
        return [text]
    prefix = " > ".join(headings)
    body = text[len(prefix) :].strip() if prefix and text.startswith(prefix) else text
    available = (
        policy.max_characters - len(prefix) - 2 if prefix else policy.max_characters
    )
    available = max(1, available)
    parts: list[str] = []
    remaining = body
    while remaining:
        cut = min(len(remaining), available)
        if cut < len(remaining):
            boundary = remaining.rfind(" ", 0, cut + 1)
            cut = boundary if boundary > 0 else cut
        piece = remaining[:cut].strip()
        parts.append(f"{prefix}\n\n{piece}".strip() if prefix else piece)
        if cut == len(remaining):
            break
        overlap = remaining[max(0, cut - policy.overlap_characters) : cut].strip()
        remaining = f"{overlap} {remaining[cut:].lstrip()}".strip()
    return parts
