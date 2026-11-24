"""HTTP API tests against the real FastAPI app, real MCP subprocess and real vector store.

The LLM is replaced by `ScriptedLLM`, so no network access or API key is needed.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import config
from app.rag import VectorStore
from app.ratelimit import InMemoryRateLimiter
from tests.conftest import ScriptedLLM, answer, calls, connection_error, make_wav, tool_call


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, Any]]:
    from app import main

    # Build the index into a temp dir with no key, so no embedding request leaves the machine.
    saved_key = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = ""
    main.agent.store = VectorStore(data_dir=tmp_path_factory.mktemp("data"))
    # Generous limits so this module's many requests are not throttled (limits are tested separately).
    main.limiter = InMemoryRateLimiter(per_minute=100_000, burst=100_000)
    with TestClient(main.app) as client:
        yield client, main
    config.OPENROUTER_API_KEY = saved_key


@pytest.fixture()
def with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")


def _sid() -> str:
    return "s-" + uuid.uuid4().hex[:12]


# ------------------------------------------------------------------ status / static
def test_status_reports_pipeline(api: tuple[TestClient, Any]) -> None:
    client, _ = api
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] > 0
    assert body["embedder"] == "local"
    names = {t["name"]: t["kind"] for t in body["tools"]}
    assert names["search_knowledge_base"] == "rag"
    assert names["calculate"] == "mcp"


def test_index_and_static_assets_served(api: tuple[TestClient, Any]) -> None:
    client, _ = api
    r = client.get("/")
    assert r.status_code == 200
    assert '<canvas id="wave"' in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


# ------------------------------------------------------------------------- chat
def test_chat_requires_api_key(api: tuple[TestClient, Any]) -> None:
    client, _ = api
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_configured"
    assert "OPENROUTER" not in r.text  # configuration details are for operators, not customers


def test_chat_rejects_blank_text(api: tuple[TestClient, Any], with_key: None) -> None:
    client, _ = api
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "body.text"


def test_chat_direct_answer(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    main.agent.llm = ScriptedLLM(answer("Hello there."))
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["answer"] == "Hello there."
    assert body["steps"] == []
    assert isinstance(body["latency_ms"], int)


def test_chat_mcp_tool_route(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    main.agent.llm = ScriptedLLM(
        calls(tool_call("c1", "calculate", {"expression": "6 * 7"})),
        answer("Forty two."),
    )
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "what is 6 times 7"})
    assert r.status_code == 200
    steps = r.json()["steps"]
    assert [(s["kind"], s["tool"]) for s in steps] == [("mcp", "calculate")]
    assert steps[0]["output"].endswith("= 42")


def test_chat_history_is_per_session(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    a, b = _sid(), _sid()
    llm = ScriptedLLM(answer("one"), answer("two"), answer("three"))
    main.agent.llm = llm
    client.post("/api/chat", json={"session_id": a, "text": "first from A"})
    client.post("/api/chat", json={"session_id": a, "text": "second from A"})
    client.post("/api/chat", json={"session_id": b, "text": "first from B"})
    contents = [[m.get("content") for m in req["messages"]] for req in llm.requests]
    assert "first from A" in contents[1]  # A's second turn sees A's history
    assert "first from A" not in contents[2]  # B never sees A's history


def test_reset_clears_history(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    sid = _sid()
    llm = ScriptedLLM(answer("one"), answer("two"))
    main.agent.llm = llm
    client.post("/api/chat", json={"session_id": sid, "text": "remember me"})
    r = client.post("/api/reset", data={"session_id": sid})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    client.post("/api/chat", json={"session_id": sid, "text": "again"})
    assert "remember me" not in [m.get("content") for m in llm.requests[1]["messages"]]


def test_chat_llm_failure_returns_502(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    main.agent.llm = ScriptedLLM(*[connection_error()] * len(config.LLM_MODELS))
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "llm_unavailable"


def test_chat_tool_failure_is_reported_not_raised(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    main.agent.llm = ScriptedLLM(
        calls(tool_call("c1", "no_such_tool", {})),
        answer("Sorry, that did not work."),
    )
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "use a missing tool"})
    assert r.status_code == 200
    # Unknown tools are rejected by the agent's allowlist; they never reach the MCP server.
    assert r.json()["steps"][0]["output"] == "Error: that tool is not available."


# -------------------------------------------------------------------------- stt
def test_stt_requires_api_key(api: tuple[TestClient, Any]) -> None:
    client, _ = api
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 503


def test_stt_ignores_tiny_audio(api: tuple[TestClient, Any], with_key: None) -> None:
    client, _ = api
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(seconds=0.01), "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": ""}


def test_stt_transcribes(api: tuple[TestClient, Any], with_key: None) -> None:
    client, main = api
    llm = ScriptedLLM(answer('"what time is it"'))
    main.agent.llm = llm
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": "what time is it"}
    content = llm.requests[0]["messages"][0]["content"]
    assert content[1]["type"] == "input_audio"
