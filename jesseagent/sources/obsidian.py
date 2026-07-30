"""Read Obsidian Markdown notes and split them into retrieval-ready sections."""

from datetime import datetime, timezone
from pathlib import Path
from re import DOTALL, MULTILINE, compile
from typing import cast
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from jesseagent.domain.knowledge import KnowledgeChunk, KnowledgeDocument
from jesseagent.ports.source import SourceConnectorError

_FRONTMATTER = compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", MULTILINE | DOTALL)
_HEADING = compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", MULTILINE)
_WIKI_LINK = compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_TAG = compile(r"(?<!\w)#([\w/-]+)")


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


class ObsidianSourceConnector:
    """Expose an Obsidian vault's Markdown notes in stable relative-path order."""

    source_id = "obsidian"

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        """Read Markdown files without traversing outside the configured vault."""
        if not self._vault_path.is_dir():
            raise SourceConnectorError(
                f"Obsidian vault is not a directory: {self._vault_path}"
            )
        paths = sorted(
            (path for path in self._vault_path.rglob("*.md") if path.is_file()),
            key=lambda path: path.relative_to(self._vault_path).as_posix(),
        )
        return tuple(self._read_document(path) for path in paths)

    def _read_document(self, path: Path) -> KnowledgeDocument:
        try:
            content = path.read_text(encoding="utf-8")
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError as error:
            raise SourceConnectorError(
                f"Could not read Obsidian note: {path}"
            ) from error
        if not content.strip():
            raise SourceConnectorError(f"Obsidian note is empty: {path}")
        relative_path = path.relative_to(self._vault_path).as_posix()
        frontmatter, body = _split_frontmatter(content)
        title = _frontmatter_title(frontmatter) or _first_heading(body) or path.stem
        metadata = {
            "path": relative_path,
            "tags": _unique([*_frontmatter_tags(frontmatter), *_TAG.findall(body)]),
            "wiki_links": _unique(_WIKI_LINK.findall(body)),
        }
        return KnowledgeDocument(
            source_id=self.source_id,
            document_id=f"obsidian:{relative_path}",
            uri=_obsidian_uri(self._vault_path.name, relative_path),
            title=title,
            content=content,
            content_hash=KnowledgeDocument.content_digest(content.strip()),
            updated_at=modified_at,
            metadata=cast(dict[str, JsonValue], metadata),
        )


def chunk_markdown(
    document: KnowledgeDocument,
    policy: ObsidianChunkPolicy = ObsidianChunkPolicy(),
) -> tuple[KnowledgeChunk, ...]:
    """Create heading-first chunks, splitting only oversized section bodies."""
    _, body = _split_frontmatter(document.content)
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


def _split_frontmatter(content: str) -> tuple[dict[str, str | list[str]], str]:
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


def _parse_value(value: str) -> str | list[str]:
    if value.startswith("[") and value.endswith("]"):
        return [
            _unquote(item.strip()) for item in value[1:-1].split(",") if item.strip()
        ]
    return _unquote(value)


def _unquote(value: str) -> str:
    return value.strip().strip("\"'")


def _frontmatter_title(frontmatter: dict[str, str | list[str]]) -> str | None:
    title = frontmatter.get("title")
    return title if isinstance(title, str) and title else None


def _frontmatter_tags(frontmatter: dict[str, str | list[str]]) -> list[str]:
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        return [tags] if tags else []
    return tags


def _first_heading(content: str) -> str | None:
    match = _HEADING.search(content)
    return match.group(2).strip() if match else None


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


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _obsidian_uri(vault_name: str, relative_path: str) -> str:
    return f"obsidian://open?{urlencode({'vault': vault_name, 'file': relative_path})}"
