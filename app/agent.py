"""Orchestrator: STT -> LLM (tool routing) -> RAG | MCP tools -> answer.

The LLM is given one local function tool (`search_knowledge_base`, backed by the
vector store) plus every tool advertised by the MCP server. It decides per query
which to call - possibly both - and we loop until it produces a final answer.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from . import config
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
        "- For current time/date, weather, arithmetic, system info or app logs, call the matching "
        "system tool. Never guess these or do maths in your head.\n"
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


class Agent:
    def __init__(self) -> None:
        self.llm = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY or "missing", base_url=config.OPENROUTER_BASE_URL
        )
        self.store = VectorStore()
        self.system_prompt = build_system_prompt(
            config.COMPANY_NAME, config.COMPANY_DESCRIPTION, config.KB_TOPICS, config.AGENT_NAME
        )
        self.rag_tool = build_rag_tool(config.COMPANY_NAME, config.KB_TOPICS)
        self.mcp_tools: list[ChatCompletionToolParam] = []
        self.sessions: dict[str, list[ChatCompletionMessageParam]] = {}
        self._stack = AsyncExitStack()
        self._mcp: ClientSession | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.store.build()
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(config.ROOT)
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._mcp = await self._stack.enter_async_context(ClientSession(read, write))
        await self._mcp.initialize()
        listed = await self._mcp.list_tools()
        self.mcp_tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
            }
            for t in listed.tools
        ]
        log.info("MCP connected; tools: %s", [t.name for t in listed.tools])

    async def stop(self) -> None:
        await self._stack.aclose()

    @property
    def tool_names(self) -> list[dict[str, str]]:
        return [
            {"name": RAG_NAME, "kind": "rag"},
            *({"name": t["function"]["name"], "kind": "mcp"} for t in self.mcp_tools),
        ]

    # ------------------------------------------------------------------ STT
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Speech-to-text via an audio-capable OpenRouter model."""
        b64 = base64.b64encode(wav_bytes).decode()
        resp = await self.llm.chat.completions.create(
            model=config.STT_MODEL,
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
        text = (resp.choices[0].message.content or "").strip().strip('"')
        log.info("STT: %r", text)
        return text

    # ------------------------------------------------------------ tool exec
    async def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == RAG_NAME:
            hits = self.store.search(args.get("query", ""), k=3)
            return "\n\n".join(f"[{h['source']} / {h['title']}] {h['text']}" for h in hits)
        if self._mcp is None:
            raise RuntimeError("MCP session is not connected; call start() first")
        result = await self._mcp.call_tool(name, args)
        return "\n".join(c.text for c in result.content if isinstance(c, TextContent) and c.text) or (
            "(no output)"
        )

    async def _complete(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam]
    ) -> ChatCompletionMessage:
        """Call the LLM, falling through the configured model list on provider errors."""
        last: Exception | None = None
        for model in config.LLM_MODELS:
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
            except Exception as exc:  # noqa: BLE001 - narrowed to retryable errors in Phase 6
                last = exc
            log.warning("Model %s failed (%s); trying next", model, last)
        raise last or RuntimeError("no LLM configured")

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

        for _ in range(config.MAX_TOOL_ROUNDS):
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
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output = await self._run_tool(name, args)
                except Exception as exc:  # noqa: BLE001 - structured tool errors arrive in Phase 2/6
                    args, output = {}, f"Error running {name}: {exc}"
                    log.error("Tool %s failed: %s", name, exc)
                kind = "rag" if name == RAG_NAME else "mcp"
                log.info("Route -> %s tool=%s args=%s", kind.upper(), name, args)
                steps.append({"kind": kind, "tool": name, "args": args, "output": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        answer = "Sorry, that took too many steps. Could you rephrase?"
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "steps": steps, "latency_ms": elapsed()}

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
