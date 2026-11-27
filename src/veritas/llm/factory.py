"""Build the configured provider. The only place that maps settings to a backend."""

from __future__ import annotations

from veritas.core.config import Settings
from veritas.llm.base import LLMProvider
from veritas.llm.ollama_provider import OllamaProvider


def make_provider(settings: Settings, model: str | None = None) -> LLMProvider:
    """``model`` overrides ``VERITAS_OLLAMA_MODEL`` (e.g. from a CLI flag)."""
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model or settings.ollama_model,
            settings.ollama_host,
            timeout_s=settings.ollama_timeout_s,
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
            format_mode=settings.ollama_format,
            keep_alive=settings.ollama_keep_alive,
        )
    raise ValueError(
        f"provider {settings.llm_provider!r} cannot be built from settings; "
        "construct MockProvider directly in tests"
    )
