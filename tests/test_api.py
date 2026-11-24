"""HTTP API tests against the real app factory, real MCP subprocess and real vector store.

The LLM is replaced by `ScriptedLLM`, so no network access is needed.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import (
    ScriptedLLM,
    answer,
    app_agent,
    calls,
    connection_error,
    make_settings,
    make_wav,
    offline_agent,
    override_settings,
    tool_call,
)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = make_settings(tmp_path_factory.mktemp("api"), openrouter_api_key="test-key")
    with TestClient(create_app(settings, agent_factory=offline_agent)) as c:
        yield c


def _sid() -> str:
    return "s-" + uuid.uuid4().hex[:12]


# ------------------------------------------------------------------ status / static
def test_status_reports_pipeline(client: TestClient) -> None:
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["chunks"] > 0
    assert body["embedder"] == "local"
    names = {t["name"]: t["kind"] for t in body["tools"]}
    assert names["search_knowledge_base"] == "rag"
    assert names["calculate"] == "mcp"


def test_index_and_static_assets_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert '<canvas id="wave"' in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


# ------------------------------------------------------------------------- chat
def test_chat_requires_api_key(client: TestClient) -> None:
    with override_settings(client, openrouter_api_key=""):
        r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_configured"
    assert "OPENROUTER" not in r.text  # configuration details are for operators, not customers


def test_chat_rejects_blank_text(client: TestClient) -> None:
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "body.text"


def test_chat_direct_answer(client: TestClient) -> None:
    app_agent(client).llm = ScriptedLLM(answer("Hello there."))  # type: ignore[assignment]
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["answer"] == "Hello there."
    assert body["steps"] == []
    assert isinstance(body["latency_ms"], int)


def test_chat_mcp_tool_route(client: TestClient) -> None:
    app_agent(client).llm = ScriptedLLM(  # type: ignore[assignment]
        calls(tool_call("c1", "calculate", {"expression": "6 * 7"})),
        answer("Forty two."),
    )
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "what is 6 times 7"})
    assert r.status_code == 200
    steps = r.json()["steps"]
    assert [(s["kind"], s["tool"]) for s in steps] == [("mcp", "calculate")]
    assert steps[0]["output"].endswith("= 42")


def test_chat_history_is_per_session(client: TestClient) -> None:
    a, b = _sid(), _sid()
    llm = ScriptedLLM(answer("one"), answer("two"), answer("three"))
    app_agent(client).llm = llm  # type: ignore[assignment]
    client.post("/api/chat", json={"session_id": a, "text": "first from A"})
    client.post("/api/chat", json={"session_id": a, "text": "second from A"})
    client.post("/api/chat", json={"session_id": b, "text": "first from B"})
    contents = [[m.get("content") for m in req["messages"]] for req in llm.requests]
    assert "first from A" in contents[1]  # A's second turn sees A's history
    assert "first from A" not in contents[2]  # B never sees A's history


def test_reset_clears_history(client: TestClient) -> None:
    sid = _sid()
    llm = ScriptedLLM(answer("one"), answer("two"))
    app_agent(client).llm = llm  # type: ignore[assignment]
    client.post("/api/chat", json={"session_id": sid, "text": "remember me"})
    r = client.post("/api/reset", data={"session_id": sid})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    client.post("/api/chat", json={"session_id": sid, "text": "again"})
    assert "remember me" not in [m.get("content") for m in llm.requests[1]["messages"]]


def test_chat_llm_failure_returns_502(client: TestClient) -> None:
    agent = app_agent(client)
    agent.llm = ScriptedLLM(*[connection_error()] * len(agent.settings.llm_models))  # type: ignore[assignment]
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "llm_unavailable"


def test_chat_falls_back_to_next_model(client: TestClient) -> None:
    with override_settings(client, llm_model="primary/model,backup/model"):
        llm = ScriptedLLM(connection_error(), answer("from backup"))
        app_agent(client).llm = llm  # type: ignore[assignment]
        r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.json()["answer"] == "from backup"
    assert [req["model"] for req in llm.requests] == ["primary/model", "backup/model"]


def test_chat_tool_failure_is_reported_not_raised(client: TestClient) -> None:
    app_agent(client).llm = ScriptedLLM(  # type: ignore[assignment]
        calls(tool_call("c1", "no_such_tool", {})),
        answer("Sorry, that did not work."),
    )
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "use a missing tool"})
    assert r.status_code == 200
    # Unknown tools are rejected by the agent's allowlist; they never reach the MCP server.
    assert r.json()["steps"][0]["output"] == "Error: that tool is not available."


# -------------------------------------------------------------------------- stt
def test_stt_requires_api_key(client: TestClient) -> None:
    with override_settings(client, openrouter_api_key=""):
        r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 503


def test_stt_ignores_tiny_audio(client: TestClient) -> None:
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(seconds=0.01), "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": ""}


def test_stt_transcribes(client: TestClient) -> None:
    llm = ScriptedLLM(answer('"what time is it"'))
    app_agent(client).llm = llm  # type: ignore[assignment]
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": "what time is it"}
    content = llm.requests[0]["messages"][0]["content"]
    assert content[1]["type"] == "input_audio"
