"""TubeTalk configuration settings management module."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """TubeTalk application configuration settings."""

    gemini_api_key: str = ""
    data_dir: Path = Path("./data")
    vision_model: str = "gemini-3.5-flash"
    vision_prompt_version: str = "vision-scenes-v2-30s"
    llm_model: str = "gemini-3.5-flash-lite"
    summary_model: str = "gemini-3.5-flash-lite"
    summary_language: str = "ko"
    summary_prompt_version: str = "summary-chapters-v1"
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


settings = Settings()
