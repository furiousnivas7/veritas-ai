"""Ollama adapter.

Error mapping was checked against ollama-python 0.6.3 / Ollama 0.40.18:
* server unreachable   -> builtin ``ConnectionError``     -> LLMUnavailableError
* model not installed  -> ``ResponseError`` status 404    -> LLMUnavailableError
* request too slow     -> ``httpx.TimeoutException``      -> LLMTimeoutError
* other server errors  -> ``ResponseError`` / ``RequestError`` -> LLMResponseError

Constrained output (``format_mode``):
* ``schema`` - pass the JSON Schema as ``format`` (grammar-constrained decoding).
* ``json``   - pass ``format="json"`` and append the schema to the system message.
  Needed for ``qwen3:4b-instruct-2507-q4_K_M`` on Ollama 0.40.18, which stops
  responding when given a schema (observed: timeouts at 60 s, 300 s and 1555 s),
  while ``format="json"`` answers in ~1.5 s. Output is validated by Pydantic in
  both modes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx
import ollama

from veritas.llm.base import (
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)


class OllamaProvider:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        timeout_s: float = 180.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        think: Literal["off", "on", "auto"] = "off",
        format_mode: Literal["schema", "json"] = "schema",
        keep_alive: str = "10m",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout_s
        self._think = think
        self._format_mode = format_mode
        self._keep_alive = keep_alive
        self._gen_options: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
        }
        self._client = client or ollama.Client(host=host, timeout=timeout_s)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def options(self) -> dict[str, Any]:
        return {
            **self._gen_options,
            "think": self._think,
            "format_mode": self._format_mode,
            "timeout_s": self._timeout,
        }

    def check(self) -> None:
        try:
            self._client.show(self._model)
        except ConnectionError as e:
            raise LLMUnavailableError(f"Ollama not reachable at {self._host}: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(f"Ollama at {self._host} did not respond: {e}") from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMUnavailableError(
                    f"model {self._model!r} is not installed (ollama pull {self._model})"
                ) from e
            raise LLMResponseError(str(e)) from e

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "options": self._gen_options,
            "keep_alive": self._keep_alive,
        }
        if request.json_schema is not None:
            if self._format_mode == "schema":
                kwargs["format"] = request.json_schema
            else:
                kwargs["format"] = "json"
                schema_text = json.dumps(request.json_schema, separators=(",", ":"))
                note = f"\n\nRespond with one JSON object matching this JSON Schema:\n{schema_text}"
                if messages and messages[0]["role"] == "system":
                    messages[0] = {**messages[0], "content": messages[0]["content"] + note}
                else:
                    messages.insert(0, {"role": "system", "content": note.strip()})
        if self._think != "auto":
            kwargs["think"] = self._think == "on"

        started = time.perf_counter()
        try:
            resp = self._client.chat(**kwargs)
        except ConnectionError as e:
            raise LLMUnavailableError(f"Ollama not reachable at {self._host}: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"no response within {self._timeout}s from {self._model}") from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMUnavailableError(f"model {self._model!r} is not installed") from e
            raise LLMResponseError(f"ollama error {e.status_code}: {e.error}") from e
        except (ollama.RequestError, httpx.HTTPError) as e:
            raise LLMResponseError(str(e)) from e

        content = getattr(getattr(resp, "message", None), "content", None)
        if not isinstance(content, str):
            raise LLMResponseError("response has no message content")
        return LLMResponse(
            text=content,
            model=self._model,
            provider=self.name,
            duration_s=round(time.perf_counter() - started, 3),
            prompt_tokens=getattr(resp, "prompt_eval_count", None),
            completion_tokens=getattr(resp, "eval_count", None),
        )
