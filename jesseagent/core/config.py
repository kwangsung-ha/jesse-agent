"""JesseAgent configuration settings management module."""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jesseagent.domain.chaptering import ChapterBlockPolicy, ChapterWindowPolicy


class Settings(BaseSettings):
    """JesseAgent application configuration settings."""

    gemini_api_key: str = ""
    data_dir: Path = Path("./data")
    vision_model: str = "gemini-3.5-flash"
    vision_prompt_version: str = "vision-scenes-v2-30s"
    llm_model: str = "gemini-3.5-flash-lite"
    summary_model: str = "gemini-3.5-flash-lite"
    summary_language: str = "ko"
    summary_prompt_version: str = "summary-chapters-v2"
    chapter_window_max_seconds: float = 480.0
    chapter_window_max_characters: int = 12000
    chapter_window_overlap_seconds: float = 30.0
    chapter_block_max_seconds: float = 20.0
    chapter_block_max_characters: int = 400
    chapter_block_max_gap_seconds: float = 1.5
    chat_prompt_version: str = "grounded-chat-v1"
    agent_prompt_version: str = "tool-agent-v1"
    agent_max_steps: int = 8
    agent_context_max_messages: int = 24
    agent_context_max_characters: int = 12000
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    embedding_provider: Literal["gemini"] = "gemini"
    vector_repository: Literal["chroma"] = "chroma"
    transcript_chunk_max_seconds: float = 45.0
    transcript_chunk_max_characters: int = 1200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_data_dir(self) -> Path:
        """Ensure that data_dir exists on disk and return its resolved path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    @property
    def chapter_window_policy(self) -> ChapterWindowPolicy:
        """Build the configured coverage-first chapter extraction policy."""
        return ChapterWindowPolicy(
            max_seconds=self.chapter_window_max_seconds,
            max_characters=self.chapter_window_max_characters,
            overlap_seconds=self.chapter_window_overlap_seconds,
        )

    @property
    def chapter_block_policy(self) -> ChapterBlockPolicy:
        """Build the configured caption-fragment merge policy."""
        return ChapterBlockPolicy(
            max_seconds=self.chapter_block_max_seconds,
            max_characters=self.chapter_block_max_characters,
            max_gap_seconds=self.chapter_block_max_gap_seconds,
        )

    @model_validator(mode="after")
    def validate_chapter_window_policy(self) -> "Settings":
        self.chapter_window_policy
        self.chapter_block_policy
        return self


settings = Settings()
