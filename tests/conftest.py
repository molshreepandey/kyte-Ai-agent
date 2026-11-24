"""Shared test helpers: a scripted fake LLM (real OpenAI SDK objects) and isolated app fixtures.

No test in this suite talks to a real LLM, embedding or speech provider.
"""

from __future__ import annotations

import copy
import io
import json
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2  # the HTTP client the OpenAI SDK is built on
import openai
import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import SecretStr

from app.agent import Agent
from app.config import Settings
from app.rag import VectorStore


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=json.dumps(args))
    )


def answer(text: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)


def calls(*tool_calls: ChatCompletionMessageFunctionToolCall) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=None, tool_calls=list(tool_calls))


class ScriptedLLM:
    """Stands in for `AsyncOpenAI`. Replays a script of messages (or raises scripted exceptions)
    and records every request so tests can assert on what the agent sent."""

    def __init__(self, *script: ChatCompletionMessage | BaseException) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        # Deep-copy: the agent mutates its message list between calls.
        self.requests.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=item)])


def make_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A valid 16-bit mono PCM WAV of silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


def connection_error(message: str = "connection failed") -> BaseException:
    """An OpenAI SDK network error, as raised when the provider is unreachable."""
    return openai.APIConnectionError(message=message, request=httpx2.Request("POST", "https://llm.invalid"))


# ------------------------------------------------------------------ app fixtures
# Isolated apps built from explicit Settings: no test reads the developer's shell
# environment or .env file, and no two apps share state.
_SECRET_FIELDS = {"openrouter_api_key", "admin_api_key"}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings for tests: offline, generous rate limits, runtime data under `tmp_path`."""
    values: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "openrouter_api_key": "",
        "rate_limit_per_minute": 100_000,
        "rate_limit_burst": 100_000,
    }
    values.update(overrides)
    with pytest.MonkeyPatch.context() as mp:
        for name in Settings.model_fields:
            mp.delenv(name.upper(), raising=False)
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]  # pydantic-settings init kwarg


def offline_agent(settings: Settings) -> Agent:
    """An Agent whose knowledge index always uses the local embedder (no network)."""
    agent = Agent(settings)
    agent.store = VectorStore(settings.kb_dir, settings.data_dir)
    return agent


def app_agent(client: TestClient) -> Agent:
    agent: Agent = client.app.state.agent  # type: ignore[attr-defined]
    return agent


@contextmanager
def override_settings(client: TestClient, **updates: Any) -> Iterator[Settings]:
    """Temporarily change settings on a running app (and its agent)."""
    app = client.app
    agent = app_agent(client)
    old: Settings = app.state.settings  # type: ignore[attr-defined]
    fixed = {k: SecretStr(v) if k in _SECRET_FIELDS and isinstance(v, str) else v for k, v in updates.items()}
    new = old.model_copy(update=fixed)
    app.state.settings = agent.settings = new  # type: ignore[attr-defined]
    try:
        yield new
    finally:
        app.state.settings = agent.settings = old  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shield every test from the developer's settings environment variables."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
