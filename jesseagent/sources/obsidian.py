"""Read Obsidian Markdown notes as source-neutral knowledge documents."""

from datetime import datetime, timezone
from pathlib import Path
from re import compile
from typing import cast
from urllib.parse import urlencode

from pydantic import JsonValue

from jesseagent.application.knowledge.chunking import first_heading, split_frontmatter
from jesseagent.domain.knowledge import KnowledgeDocument
from jesseagent.sources.contracts import SourceConnectorError

_WIKI_LINK = compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_TAG = compile(r"(?<!\w)#([\w/-]+)")


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
        frontmatter, body = split_frontmatter(content)
        title = _frontmatter_title(frontmatter) or first_heading(body) or path.stem
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


def _frontmatter_title(frontmatter: dict[str, str | list[str]]) -> str | None:
    title = frontmatter.get("title")
    return title if isinstance(title, str) and title else None


def _frontmatter_tags(frontmatter: dict[str, str | list[str]]) -> list[str]:
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        return [tags] if tags else []
    return tags


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _obsidian_uri(vault_name: str, relative_path: str) -> str:
    return f"obsidian://open?{urlencode({'vault': vault_name, 'file': relative_path})}"
