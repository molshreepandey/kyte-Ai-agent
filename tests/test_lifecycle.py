"""Application lifecycle (Phase 3): no import side effects, explicit startup, independent
app instances, health probes, structured logging and concurrent-request isolation."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage

from app.logging_config import configure_logging
from app.main import create_app
from tests.conftest import ScriptedLLM, answer, app_agent, make_settings, offline_agent

ROOT = Path(__file__).resolve().parent.parent


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    settings = make_settings(tmp_path, **({"openrouter_api_key": "test-key"} | overrides))
    return TestClient(create_app(settings, agent_factory=offline_agent))


# ======================================================== import side effects
def test_importing_modules_has_no_side_effects() -> None:
    """Importing must not create folders or files, configure logging, read settings or
    start anything. Any mkdir or file handler during import fails this test."""
    code = """
import logging, pathlib, sys
def forbidden(*a, **k):
    raise SystemExit(f"side effect during import: {a!r}")
pathlib.Path.mkdir = forbidden
logging.FileHandler.__init__ = forbidden
import app.main, app.agent, app.rag, app.config, app.api, app.health, app.admin
import app.security, app.errors, app.ratelimit, app.logging_config, app.dependencies
from app.config import get_settings
assert get_settings.cache_info().currsize == 0, "settings were read at import"
assert logging.getLogger().handlers == [], "logging was configured at import"
assert not hasattr(app.main, "app"), "an app instance was created at import"
print("clean")
"""
    env = {"PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == "clean"


def test_invalid_configuration_stops_startup_with_clear_message() -> None:
    env = {"PATH": "/usr/bin:/bin", "ENVIRONMENT": "production", "OPENROUTER_API_KEY": "", "PORT": "0"}
    out = subprocess.run(
        [sys.executable, "-m", "app.main"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 2
    assert "Invalid configuration" in out.stderr
    assert "PORT" in out.stderr
    assert "Traceback" not in out.stderr


# ====================================================== explicit startup
def test_lifespan_creates_runtime_folders_and_index(tmp_path: Path) -> None:
    data_dir = tmp_path / "fresh" / "data"
    assert not data_dir.exists()
    with _client(tmp_path, data_dir=data_dir) as client:
        assert data_dir.is_dir()
        assert (data_dir / "vector_store.npz").exists()
        assert client.get("/api/status").json()["chunks"] > 0


def test_app_instances_do_not_share_state(tmp_path: Path) -> None:
    with _client(tmp_path / "a") as a, _client(tmp_path / "b") as b:
        assert app_agent(a) is not app_agent(b)
        assert a.app.state.limiter is not b.app.state.limiter  # type: ignore[attr-defined]
        app_agent(a).llm = ScriptedLLM(answer("hi"))  # type: ignore[assignment]
        a.post("/api/chat", json={"session_id": "shared-id-123", "text": "only in A"})
        assert "shared-id-123" in app_agent(a).sessions
        assert "shared-id-123" not in app_agent(b).sessions


# ================================================================= health
def test_liveness(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_when_all_dependencies_are_up(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ready",
        "checks": {"knowledge_index": True, "mcp_tools": True, "llm_configured": True},
    }


def test_readiness_fails_without_llm_key(tmp_path: Path) -> None:
    with _client(tmp_path, openrouter_api_key="") as client:
        r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["llm_configured"] is False


def test_readiness_fails_when_mcp_is_down(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        agent = app_agent(client)
        session, agent._mcp = agent._mcp, None
        try:
            r = client.get("/health/ready")
        finally:
            agent._mcp = session
    assert r.status_code == 503
    assert r.json()["checks"]["mcp_tools"] is False


def test_health_is_not_rate_limited(tmp_path: Path) -> None:
    with _client(tmp_path, rate_limit_per_minute=1, rate_limit_burst=1) as client:
        assert all(client.get("/health/live").status_code == 200 for _ in range(5))


# ================================================================ logging
def _json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def test_structured_json_logs_carry_request_and_session_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with _client(tmp_path, log_format="json") as client:
        app_agent(client).llm = ScriptedLLM(answer("hello"))  # type: ignore[assignment]
        r = client.post(
            "/api/chat",
            json={"session_id": "log-session-1", "text": "hi"},
            headers={"X-Request-ID": "req-abcdef12"},
        )
    assert r.headers["x-request-id"] == "req-abcdef12"
    records = _json_lines(capsys.readouterr().out)
    access = [r for r in records if r["logger"] == "access" and r.get("path") == "/api/chat"]
    assert len(access) == 1
    entry = access[0]
    assert entry["request_id"] == "req-abcdef12"
    assert entry["session_id"] == "log-session-1"
    assert entry["status"] == 200 and entry["method"] == "POST"
    assert isinstance(entry["duration_ms"], float)
    started = [r for r in records if r["msg"] == "AI Voice Agent started"]
    assert started and started[0]["chunks"] > 0
    assert "hi" not in [r["msg"] for r in records]  # user text is not logged


def test_configure_logging_is_idempotent_and_keeps_foreign_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        settings = make_settings(tmp_path)
        configure_logging(settings)
        configure_logging(settings)
        owned = [h for h in root.handlers if getattr(h, "_ai_voice_agent_handler", False)]
        assert len(owned) == 1
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_optional_rotating_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "app.log"
    with _client(tmp_path, log_file=log_file, log_format="json") as client:
        client.get("/api/status")
    lines = _json_lines(log_file.read_text(encoding="utf-8"))
    assert any(r.get("path") == "/api/status" for r in lines)


# ============================================================ concurrency
class EchoLLM:
    """Answers each request with the latest user message, after yielding to the event
    loop, so concurrent requests genuinely interleave."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        import asyncio

        await asyncio.sleep(0.01)
        last_user = [m for m in kwargs["messages"] if m["role"] == "user"][-1]["content"]
        msg = ChatCompletionMessage(role="assistant", content=f"echo:{last_user}")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_concurrent_requests_stay_isolated(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        agent = app_agent(client)
        agent.llm = EchoLLM()  # type: ignore[assignment]
        jobs = [(f"conc-{uuid.uuid4().hex[:10]}", f"message-{i}") for i in range(24)]

        def send(job: tuple[str, str]) -> dict[str, Any]:
            sid, text = job
            r = client.post("/api/chat", json={"session_id": sid, "text": text})
            assert r.status_code == 200, r.text
            body: dict[str, Any] = r.json()
            return body

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(send, jobs))

        for (sid, text), body in zip(jobs, results, strict=True):
            assert body["session_id"] == sid
            assert body["answer"] == f"echo:{text}"
            history = [m["content"] for m in agent.sessions[sid]]
            assert history == [text, f"echo:{text}"]  # nothing from any other session
