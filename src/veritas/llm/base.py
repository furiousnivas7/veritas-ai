"""Provider-neutral LLM interface.

Extraction code depends only on ``LLMProvider``. Concrete providers translate
their client's failures into the three error types below, so callers can decide
what is retryable without knowing which backend is in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMUnavailableError(LLMError):
    """Service unreachable or model not installed. Not retryable."""


class LLMTimeoutError(LLMError):
    """The request exceeded the configured timeout. Retryable."""


class LLMResponseError(LLMError):
    """The service answered with an error or an unusable payload. Retryable."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[ChatMessage, ...]
    json_schema: dict[str, Any] | None = None  # constrained decoding when supported


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    duration_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def options(self) -> dict[str, Any]:
        """Generation settings that affect output; recorded per extraction run."""
        ...

    def check(self) -> None:
        """Raise LLMUnavailableError if the provider cannot serve requests."""
        ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...
