"""Settings loaded from environment variables (prefix ``VERITAS_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    # Local inference (Milestone 2). Switch models with VERITAS_OLLAMA_MODEL; no
    # extraction code refers to a specific model.
    llm_provider: Literal["ollama", "mock"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_num_ctx: int = 8192
    # Qwen3 builds advertise "thinking"; reasoning text corrupts JSON, so it is
    # disabled by default. "auto" omits the parameter and lets Ollama decide.
    ollama_think: Literal["off", "on", "auto"] = "off"
    # "schema" = grammar-constrained JSON; "json" = JSON mode + schema in the prompt.
    # Use "json" for qwen3:4b-instruct-2507-q4_K_M (hangs with a schema on Ollama 0.40.18).
    ollama_format: Literal["schema", "json"] = "schema"
    ollama_timeout_s: float = Field(default=180.0, gt=0)
    ollama_keep_alive: str = "10m"
    llm_temperature: float = 0.0
    llm_seed: int = 42

    # Extraction
    extraction_max_retries: int = Field(default=2, ge=0, le=5, description="Retries per batch")
    extraction_batch_chars: int = Field(default=1800, ge=200)
    # document_scoped: names merge only within a document (cross-document matches go
    # to review). global_name_match: name-only merging across documents (the E5
    # "exact matching" baseline; not for production use).
    entity_resolution_policy: Literal["document_scoped", "global_name_match"] = "document_scoped"
    # detect: content-based header detection (docs/data_model.md, "Table headers").
    # never: every table row is data, so no first row is ever withheld from
    # extraction, at the cost of all header labels and header evidence.
    table_header_policy: Literal["detect", "never"] = "detect"
    # Prompt version (1.0.0 kept for reproducing earlier runs).
    extraction_prompt_version: str = "1.1.0"
    # on: deterministic currency/period enrichment from the cited evidence (with a
    # recorded decision per field). off: previous behaviour (no enrichment, no
    # decisions, previous issue names and confidence penalties).
    metadata_enrichment: Literal["on", "off"] = "on"
    # Explicit resolutions for ambiguous currency forms, e.g. {"Rs": "LKR", "$": "USD"}
    # (JSON in VERITAS_CURRENCY_SYMBOL_DEFAULTS). Empty: "$", "Rs.", "rupees", "¥"…
    # resolve to no currency. Recorded in each run's options.
    currency_symbol_defaults: dict[str, str] = Field(default_factory=dict)

    @field_validator("currency_symbol_defaults")
    @classmethod
    def _check_currency_defaults(cls, v: dict[str, str]) -> dict[str, str]:
        from veritas.extraction.grounding import CURRENCY_CODES, normalize_currency_default_key

        out: dict[str, str] = {}
        for key, code in v.items():
            norm = normalize_currency_default_key(key)  # raises for unknown forms
            if code.upper() not in CURRENCY_CODES:
                raise ValueError(f"{code!r} is not a supported ISO 4217 code")
            out[norm] = code.upper()
        return out

    @field_validator("extraction_prompt_version")
    @classmethod
    def _check_prompt_version(cls, v: str) -> str:
        from veritas.extraction.prompts import get_prompt

        get_prompt(v)
        return v

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
