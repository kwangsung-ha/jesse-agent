"""Tests for Obsidian Markdown ingestion and heading-first chunking."""

import os
from datetime import datetime, timezone
from pathlib import Path

from jesseagent.application.knowledge.chunking import (
    ObsidianChunkPolicy,
    chunk_markdown,
)
from jesseagent.infrastructure.obsidian.source import ObsidianSourceConnector


def _write(vault: Path, relative_path: str, content: str) -> Path:
    path = vault / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_obsidian_connector_parses_frontmatter_tags_and_wiki_links(
    tmp_path: Path,
) -> None:
    """Markdown metadata becomes source-neutral, JSON-safe document metadata."""
    note = _write(
        tmp_path,
        "Recipes/kimchi-stew.md",
        """---
title: 김치찌개
tags: [recipe, Korean food]
---
# 김치찌개
[[두부]]와 [[돼지고기|삼겹살]]을 넣는다. #저녁 #recipe
""",
    )
    os.utime(note, (1_700_000_000, 1_700_000_000))

    documents = ObsidianSourceConnector(tmp_path).list_documents()

    assert len(documents) == 1
    document = documents[0]
    assert document.source_id == "obsidian"
    assert document.document_id == "obsidian:Recipes/kimchi-stew.md"
    assert (
        document.uri
        == "obsidian://open?vault=pytest-of-root&file=Recipes%2Fkimchi-stew.md"
        or document.uri.endswith("&file=Recipes%2Fkimchi-stew.md")
    )
    assert document.title == "김치찌개"
    assert document.metadata == {
        "path": "Recipes/kimchi-stew.md",
        "tags": ["recipe", "Korean food", "저녁"],
        "wiki_links": ["두부", "돼지고기"],
    }
    assert document.updated_at == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
    assert document.content_hash == document.content_digest(document.content)


def test_obsidian_connector_detects_content_and_file_timestamp_changes(
    tmp_path: Path,
) -> None:
    """A changed note has a new digest and modification timestamp for later sync."""
    note = _write(tmp_path, "note.md", "# One\n첫 내용\n")
    os.utime(note, (1_700_000_000, 1_700_000_000))
    connector = ObsidianSourceConnector(tmp_path)
    before = connector.list_documents()[0]

    note.write_text("# One\n변경된 내용\n", encoding="utf-8")
    os.utime(note, (1_700_000_100, 1_700_000_100))
    after = connector.list_documents()[0]

    assert after.content_hash != before.content_hash
    assert after.updated_at > before.updated_at


def test_heading_first_chunker_preserves_heading_path_and_splits_long_sections(
    tmp_path: Path,
) -> None:
    """Headings define chunk boundaries before a long section is size-split."""
    _write(
        tmp_path,
        "menu.md",
        """# 이번 주 식단
준비 사항
## 월요일
된장국과 밥
## 화요일
아주 긴 재료 목록 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열
열하나 열둘 열셋 열넷 열다섯 열여섯
""",
    )
    document = ObsidianSourceConnector(tmp_path).list_documents()[0]

    chunks = chunk_markdown(
        document, policy=ObsidianChunkPolicy(max_characters=55, overlap_characters=8)
    )

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["heading_path"] == ["이번 주 식단"]
    assert any(
        chunk.metadata["heading_path"] == ["이번 주 식단", "월요일"] for chunk in chunks
    )
    tuesday = [
        chunk
        for chunk in chunks
        if chunk.metadata["heading_path"] == ["이번 주 식단", "화요일"]
    ]
    assert len(tuesday) == 2
    assert all(len(chunk.text) <= 55 for chunk in chunks)
