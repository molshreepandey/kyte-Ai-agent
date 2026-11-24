"""Shared test helpers: a scripted fake LLM that returns real OpenAI SDK objects.

No test in this suite talks to a real LLM, embedding or speech provider.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function


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
