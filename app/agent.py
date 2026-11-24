"""Orchestrator: STT -> LLM (tool routing) -> RAG | MCP tools -> answer.

The LLM is given one local function tool (`search_knowledge_base`, backed by the
vector store) plus the MCP server tools named in the allowlist (MCP_TOOL_ALLOWLIST).
It decides per query which to call - possibly both - and we loop until it produces
a final answer. Tools the server advertises but the allowlist omits are never shown
to, or executable by, the LLM.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import anyio
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from .config import ROOT, Settings
from .errors import LLMUnavailableError, SpeechToTextError
from .rag import VectorStore

log = logging.getLogger("agent")

RAG_NAME = "search_knowledge_base"


def build_system_prompt(
    company_name: str, company_description: str, kb_topics: str, agent_name: str = ""
) -> str:
    """The voice persona. Business details come from configuration, not code."""
    who = f"You are {agent_name}, a friendly" if agent_name else "You are a friendly"
    return (
        f"{who} voice assistant for {company_name}, {company_description}.\n"
        "Your replies are spoken aloud, so:\n"
        "- Answer in one to three short, natural sentences. No markdown, bullets, emojis or URLs.\n"
        "- Say numbers the way a person would speak them.\n"
        "\n"
        "Tool use:\n"
        f"- For ANY question about {company_name} ({kb_topics}), call {RAG_NAME} and answer ONLY "
        "from what it returns. If it does not contain the answer, say you don't have that information.\n"
        "- For the current time/date, the weather or any arithmetic, call the matching tool. "
        "Never guess these or do maths in your head.\n"
        "- If a question needs both (e.g. a fee from the knowledge base applied to an amount), call the "
        "knowledge base first, then the calculator.\n"
        "- For greetings or small talk, answer directly without tools."
    )


def build_rag_tool(company_name: str, kb_topics: str) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": RAG_NAME,
            "description": f"Semantic search over the {company_name} knowledge base (FAQs and product "
            f"specifications). Use for any question about {company_name}: {kb_topics}.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Self-contained search query."}},
                "required": ["query"],
            },
        },
    }


MAX_TOOL_NAME_CHARS = 64
MCP_PING_TIMEOUT_SECONDS = 2.0


class ToolNotAllowedError(Exception):
    """The LLM asked for a tool that is not exposed to it."""


class InvalidToolArgumentsError(ValueError):
    """The LLM produced tool arguments that are not a JSON object."""


class ToolUnavailableError(Exception):
    """The tool backend (e.g. the MCP server) is not connected."""


# Errors a tool call can fail with. Anything else is a bug and propagates.
_TOOL_ERRORS = (
    ToolNotAllowedError,
    InvalidToolArgumentsError,
    ToolUnavailableError,
    McpError,
    TimeoutError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    openai.OpenAIError,
)


def _tool_error_message(exc: BaseException) -> str:
    """Client- and LLM-safe description of a tool failure (no internal details)."""
    if isinstance(exc, ToolNotAllowedError):
        return "Error: that tool is not available."
    if isinstance(exc, InvalidToolArgumentsError):
        return "Error: the tool arguments were invalid."
    if isinstance(exc, TimeoutError) or (isinstance(exc, McpError) and "timed out" in str(exc).lower()):
        return "Error: the tool took too long to respond."
    return "Error: the tool is unavailable right now."


class Agent:
    """One instance per application. Construction is side-effect free; `start()`
    builds the knowledge index and launches the MCP server subprocess."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key = settings.openrouter_api_key.get_secret_value()
        self.llm = AsyncOpenAI(
            api_key=api_key or "missing",
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        self.store = VectorStore(
            settings.kb_dir,
            settings.data_dir,
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            embedding_model=settings.embedding_model,
        )
        self.system_prompt = build_system_prompt(
            settings.company_name, settings.company_description, settings.kb_topics, settings.agent_name
        )
        self.rag_tool = build_rag_tool(settings.company_name, settings.kb_topics)
        self.mcp_tools: list[ChatCompletionToolParam] = []
        self.sessions: dict[str, list[ChatCompletionMessageParam]] = {}
        self._stack = AsyncExitStack()
        self._mcp: ClientSession | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.store.build()
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT))
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._mcp = await self._stack.enter_async_context(ClientSession(read, write))
        await self._mcp.initialize()
        listed = await self._mcp.list_tools()
        allowed = set(self.settings.tool_allowlist)
        self.mcp_tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
            }
            for t in listed.tools
            if t.name in allowed
        ]
        hidden = sorted(t.name for t in listed.tools if t.name not in allowed)
        missing = sorted(allowed - {t.name for t in listed.tools})
        log.info("MCP connected; exposed tools: %s", [t["function"]["name"] for t in self.mcp_tools])
        if hidden:
            log.info("MCP tools not in MCP_TOOL_ALLOWLIST (hidden from the LLM): %s", hidden)
        if missing:
            log.warning("MCP_TOOL_ALLOWLIST names tools the server does not provide: %s", missing)

    async def stop(self) -> None:
        await self._stack.aclose()
        self._mcp = None

    async def ping_mcp(self) -> bool:
        """True if the MCP server answers a ping in time (used by the readiness check)."""
        if self._mcp is None:
            return False
        try:
            await asyncio.wait_for(self._mcp.send_ping(), MCP_PING_TIMEOUT_SECONDS)
        except (McpError, TimeoutError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            return False
        return True

    @property
    def tool_names(self) -> list[dict[str, str]]:
        return [
            {"name": RAG_NAME, "kind": "rag"},
            *({"name": t["function"]["name"], "kind": "mcp"} for t in self.mcp_tools),
        ]

    # ------------------------------------------------------------------ STT
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Speech-to-text via an audio-capable OpenRouter model. Raises SpeechToTextError."""
        b64 = base64.b64encode(wav_bytes).decode()
        try:
            resp = await self._transcribe_request(b64)
        except openai.OpenAIError as exc:
            raise SpeechToTextError() from exc
        if not resp.choices:
            raise SpeechToTextError()
        text = (resp.choices[0].message.content or "").strip().strip('"')
        if self.settings.log_transcripts:
            log.info("STT: %r", text)
        else:
            log.info("STT: %d characters", len(text))
        return text

    async def _transcribe_request(self, b64: str) -> ChatCompletion:
        return await self.llm.chat.completions.create(
            model=self.settings.stt_model,
            temperature=0,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio verbatim. Output only the spoken words, "
                            "nothing else. If there is no speech, output an empty string.",
                        },
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ],
                }
            ],
        )

    # ------------------------------------------------------------ tool exec
    def is_tool_allowed(self, name: str) -> bool:
        return name == RAG_NAME or any(t["function"]["name"] == name for t in self.mcp_tools)

    async def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        if not self.is_tool_allowed(name):
            raise ToolNotAllowedError(name)
        if name == RAG_NAME:
            hits = self.store.search(args.get("query", ""), k=3)
            return "\n\n".join(f"[{h['source']} / {h['title']}] {h['text']}" for h in hits)
        if self._mcp is None:
            raise ToolUnavailableError("MCP session is not connected")
        result = await self._mcp.call_tool(
            name, args, read_timeout_seconds=timedelta(seconds=self.settings.tool_timeout_seconds)
        )
        return "\n".join(c.text for c in result.content if isinstance(c, TextContent) and c.text) or (
            "(no output)"
        )

    async def _complete(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam]
    ) -> ChatCompletionMessage:
        """Call the LLM, falling through the configured model list on provider errors.

        Raises LLMUnavailableError when every model fails.
        """
        last: Exception | None = None
        for model in self.settings.llm_models:
            try:
                resp = await self.llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=500,
                )
                if resp.choices:
                    return resp.choices[0].message
                last = RuntimeError(f"{model} returned no choices")
            except openai.OpenAIError as exc:  # retry/fail-fast classification arrives in Phase 6
                last = exc
            log.warning("Model %s failed (%s); trying next", model, type(last).__name__)
        raise LLMUnavailableError() from last

    # ----------------------------------------------------------------- chat
    async def chat(self, session_id: str, user_text: str) -> dict[str, Any]:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": user_text})
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            *history[-20:],
        ]
        tools = [self.rag_tool, *self.mcp_tools]
        steps: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - t0) * 1000)

        for _ in range(self.settings.max_tool_rounds):
            msg = await self._complete(messages, tools)
            # Only function tools are offered, so other call types (custom tools) are ignored.
            calls = [c for c in msg.tool_calls or [] if isinstance(c, ChatCompletionMessageFunctionToolCall)]
            if not calls:
                answer = (msg.content or "").strip() or "Sorry, I couldn't come up with an answer."
                history.append({"role": "assistant", "content": answer})
                return {"answer": answer, "steps": steps, "latency_ms": elapsed()}

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in calls
                    ],
                }
            )
            for call in calls:
                name = call.function.name[:MAX_TOOL_NAME_CHARS]
                args: dict[str, Any] = {}
                try:
                    args = _parse_tool_args(call.function.arguments)
                    output = await self._run_tool(name, args)
                except _TOOL_ERRORS as exc:
                    output = _tool_error_message(exc)
                    log.warning("Tool %s failed: %s", name, type(exc).__name__)
                kind = "rag" if name == RAG_NAME else "mcp"
                if self.settings.log_transcripts:
                    log.info("Route -> %s tool=%s args=%s", kind.upper(), name, args)
                else:
                    log.info("Route -> %s tool=%s", kind.upper(), name)
                steps.append({"kind": kind, "tool": name, "args": args, "output": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        answer = "Sorry, that took too many steps. Could you rephrase?"
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "steps": steps, "latency_ms": elapsed()}

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise InvalidToolArgumentsError("arguments are not valid JSON") from exc
    if not isinstance(args, dict):
        raise InvalidToolArgumentsError("arguments must be a JSON object")
    return args
