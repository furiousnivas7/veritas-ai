"""Settings loaded from environment variables (prefix ``VERITAS_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERITAS_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    data_dir: Path = Path("./data")
    database_url: str | None = Field(
        default=None, description="Defaults to sqlite:///<data_dir>/veritas.sqlite3"
    )
    max_upload_mb: int = Field(default=50, gt=0)

    # Parsing
    pdf_ocr: bool = Field(default=False, description="OCR is slow; enable for scanned PDFs")

    # Local inference (used from Milestone 2)
    llm_provider: Literal["ollama", "mock"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_num_ctx: int = 8192
    llm_temperature: float = 0.0

    # Optional cloud inference: disabled unless explicitly configured
    cloud_enabled: bool = False

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'veritas.sqlite3'}"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
