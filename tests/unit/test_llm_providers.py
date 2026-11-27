from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest

from veritas.core.config import Settings
from veritas.llm.base import (
    ChatMessage,
    LLMRequest,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from veritas.llm.factory import make_provider
from veritas.llm.mock_provider import MockProvider
from veritas.llm.ollama_provider import OllamaProvider

REQ = LLMRequest((ChatMessage("user", "hi"),), json_schema={"type": "object"})


class FakeClient:
    """Stands in for ollama.Client; raises or returns what it is told to."""

    def __init__(self, chat: Any = None, show: Any = None) -> None:
        self._chat, self._show = chat, show
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._chat, BaseException):
            raise self._chat
        return self._chat

    def show(self, model: str) -> Any:
        if isinstance(self._show, BaseException):
            raise self._show
        return SimpleNamespace(capabilities=["completion"])


def ok_response(content: str = '{"claims": [], "entities": []}') -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(content=content), prompt_eval_count=10, eval_count=5
    )


def provider(client: FakeClient, **kw: Any) -> OllamaProvider:
    return OllamaProvider("qwen3:4b", client=client, **kw)


def test_success_and_request_shape() -> None:
    client = FakeClient(chat=ok_response())
    resp = provider(client, seed=7, num_ctx=4096).generate(REQ)
    assert resp.text == '{"claims": [], "entities": []}'
    assert (resp.model, resp.provider, resp.completion_tokens) == ("qwen3:4b", "ollama", 5)
    call = client.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["format"] == {"type": "object"}
    assert call["think"] is False  # thinking disabled by default
    assert call["options"] == {"temperature": 0.0, "seed": 7, "num_ctx": 4096}


def test_think_auto_omits_parameter() -> None:
    client = FakeClient(chat=ok_response())
    provider(client, think="auto").generate(REQ)
    assert "think" not in client.calls[0]


def test_options_are_recorded() -> None:
    opts = provider(FakeClient(), think="off", timeout_s=30).options()
    assert opts["think"] == "off" and opts["timeout_s"] == 30 and opts["temperature"] == 0.0


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionError("Failed to connect to Ollama"), LLMUnavailableError),
        (ollama.ResponseError("model 'x' not found", 404), LLMUnavailableError),
        (httpx.ReadTimeout("timed out"), LLMTimeoutError),
        (ollama.ResponseError("internal", 500), LLMResponseError),
        (httpx.RemoteProtocolError("dropped"), LLMResponseError),
    ],
)
def test_error_mapping(exc: BaseException, expected: type[Exception]) -> None:
    with pytest.raises(expected):
        provider(FakeClient(chat=exc)).generate(REQ)


def test_missing_content_is_response_error() -> None:
    with pytest.raises(LLMResponseError):
        provider(FakeClient(chat=SimpleNamespace(message=None))).generate(REQ)


@pytest.mark.parametrize(
    "exc", [ConnectionError("refused"), ollama.ResponseError("not found", 404),
            httpx.ConnectTimeout("slow")]
)  # fmt: skip
def test_check_reports_unavailable(exc: BaseException) -> None:
    with pytest.raises(LLMUnavailableError):
        provider(FakeClient(show=exc)).check()


def test_unreachable_real_client() -> None:
    """A real ollama.Client pointed at a closed port: no mocking of the error path."""
    p = OllamaProvider("qwen3:4b", host="http://127.0.0.1:9", timeout_s=2)
    with pytest.raises(LLMUnavailableError):
        p.check()
    with pytest.raises(LLMUnavailableError):
        p.generate(REQ)


def test_factory_switches_model_without_code_changes() -> None:
    s = Settings(ollama_model="qwen3:4b")
    assert make_provider(s).model == "qwen3:4b"
    assert (
        make_provider(s, "qwen3:4b-instruct-2507-q4_K_M").model == "qwen3:4b-instruct-2507-q4_K_M"
    )
    s2 = Settings(ollama_model="qwen3:4b-instruct-2507-q4_K_M")
    assert make_provider(s2).model == "qwen3:4b-instruct-2507-q4_K_M"
    with pytest.raises(ValueError):
        make_provider(Settings(llm_provider="mock"))


def test_mock_provider_script_and_errors() -> None:
    m = MockProvider(["a", LLMTimeoutError("t"), "b"])
    assert m.generate(REQ).text == "a"
    with pytest.raises(LLMTimeoutError):
        m.generate(REQ)
    assert m.generate(REQ).text == "b"
    assert len(m.requests) == 3
    with pytest.raises(LLMUnavailableError):
        MockProvider(["x"], available=False).check()


def test_json_format_mode_puts_schema_in_system_prompt() -> None:
    client = FakeClient(chat=ok_response())
    req = LLMRequest(
        (ChatMessage("system", "Extract."), ChatMessage("user", "hi")),
        json_schema={"type": "object", "properties": {"claims": {}}},
    )
    p = provider(client, format_mode="json")
    p.generate(req)
    call = client.calls[0]
    assert call["format"] == "json"
    assert call["messages"][0]["content"].startswith("Extract.")
    assert '"claims"' in call["messages"][0]["content"]
    assert req.messages[0].content == "Extract."  # the request itself is not mutated
    assert p.options()["format_mode"] == "json"


def test_schema_format_mode_is_default() -> None:
    client = FakeClient(chat=ok_response())
    provider(client).generate(REQ)
    assert client.calls[0]["format"] == {"type": "object"}
