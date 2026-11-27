"""Deterministic provider for tests and offline development.

Responses come from a fixed script (consumed in order) or a responder function
that receives the request. Items in a script may be exceptions, which are raised
to simulate timeouts or outages.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from veritas.llm.base import LLMRequest, LLMResponse, LLMUnavailableError

Responder = Callable[[LLMRequest], str]


class MockProvider:
    def __init__(
        self,
        script: Iterable[str | BaseException] | None = None,
        responder: Responder | None = None,
        *,
        model: str = "mock-model",
        available: bool = True,
    ) -> None:
        if (script is None) == (responder is None):
            raise ValueError("give exactly one of script or responder")
        self._script = list(script) if script is not None else None
        self._responder = responder
        self._model = model
        self._available = available
        self.requests: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return self._model

    def options(self) -> dict[str, Any]:
        return {"deterministic": True}

    def check(self) -> None:
        if not self._available:
            raise LLMUnavailableError("mock provider configured as unavailable")

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.check()
        self.requests.append(request)
        if self._responder is not None:
            text = self._responder(request)
        else:
            assert self._script is not None
            if not self._script:
                raise AssertionError("mock script exhausted")
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            text = item
        return LLMResponse(text=text, model=self._model, provider=self.name)
