"""Offline tests: calculator safety, RAG retrieval, and the agent's routing loop
(with a scripted fake LLM talking to the real MCP server over stdio)."""

import asyncio

import pytest

from app import config
from app.agent import Agent
from app.mcp_server import calculate, get_current_time
from app.rag import VectorStore
from tests.conftest import ScriptedLLM, answer, calls, tool_call


def test_calculator():
    assert calculate("2 + 3 * 4").endswith("= 14")
    assert "11642.52608" in calculate("10000 * (1 + 0.052) ** 3")
    assert calculate("sqrt(144)").endswith("= 12.0")
    assert calculate("1/0").startswith("Error")
    assert calculate("__import__('os').system('echo hi')").startswith("Error")
    assert calculate("9**9**9").startswith("Error")


def test_time_tool():
    assert "UTC" in get_current_time("UTC")
    assert get_current_time("Nope/Zone").startswith("Error")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    s = VectorStore(data_dir=tmp_path)
    s.build()
    return s


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what is the daily transfer limit on the plus plan", "What are the transfer limits?"),
        ("my card was stolen", "What should I do if my card is lost or stolen?"),
        ("how much cashback does the card give", "Demo Wallet Card"),
        ("how do I reset my pin", "How do I reset my Demo Wallet PIN?"),
    ],
)
def test_rag_retrieval(store, query, expected):
    assert store.search(query, k=2)[0]["title"] == expected


def test_agent_routes_rag_then_mcp(tmp_path, monkeypatch):
    """The 'both' scenario: KB lookup -> calculator -> final answer."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm = ScriptedLLM(
        calls(tool_call("1", "search_knowledge_base", {"query": "international fee premium"})),
        calls(tool_call("2", "calculate", {"expression": "2500 * 0.004"})),
        answer("The fee is ten dollars."),
    )

    async def run():
        agent = Agent()
        agent.store = VectorStore(data_dir=tmp_path)
        await agent.start()
        try:
            assert "calculate" in [t["name"] for t in agent.tool_names]
            agent.llm = llm  # type: ignore[assignment]
            out = await agent.chat("t", "fee for 2500 on premium?")
        finally:
            await agent.stop()
        return out

    out = asyncio.run(run())
    assert [s["kind"] for s in out["steps"]] == ["rag", "mcp"]
    assert out["answer"] == "The fee is ten dollars."
    assert "0.4%" in llm.requests[1]["messages"][-1]["content"]  # RAG result reached the LLM
    assert "= 10.0" in llm.requests[2]["messages"][-1]["content"]  # MCP result reached the LLM
