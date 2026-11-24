"""Security boundary tests (Phase 2): tool exposure, input validation, error sanitisation,
rate limiting, request size limits, security headers and admin authentication."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from app import mcp_server as tools
from app.agent import Agent
from app.main import create_app
from app.ratelimit import InMemoryRateLimiter
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

DEFAULT_MAX_TEXT_CHARS = 2000
DEFAULT_MAX_JSON_BODY_BYTES = 64 * 1024
ADMIN_KEY = "a" * 40


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = make_settings(tmp_path_factory.mktemp("security"), openrouter_api_key="test-key")
    with TestClient(create_app(settings, agent_factory=offline_agent)) as c:
        yield c


def _sid() -> str:
    return "s-" + uuid.uuid4().hex[:12]


def _assert_envelope(r: httpx2.Response, status: int, code: str) -> dict[str, Any]:
    assert r.status_code == status, r.text
    err: dict[str, Any] = r.json()["error"]
    assert err["code"] == code
    assert err["request_id"] == r.headers["x-request-id"]
    assert isinstance(err["message"], str) and err["message"]
    return err


# ===================================================================== tools
def test_mcp_server_no_longer_offers_host_tools() -> None:
    names = {t.name for t in asyncio.run(tools.mcp.list_tools())}
    assert names == {"get_current_time", "calculate", "get_weather"}
    assert not hasattr(tools, "read_system_logs")
    assert not hasattr(tools, "get_system_info")


def test_system_prompt_does_not_advertise_logs_or_system_info(tmp_path: Path) -> None:
    prompt = Agent(make_settings(tmp_path)).system_prompt.lower()
    assert "log" not in prompt
    assert "system info" not in prompt


def _with_agent[T](tmp_path: Path, body: Callable[[Agent], Awaitable[T]], **overrides: Any) -> T:
    """Run `body` against a real agent + MCP subprocess inside one event-loop task
    (anyio requires the MCP session to be opened and closed in the same task)."""

    async def run() -> T:
        agent = offline_agent(make_settings(tmp_path, **overrides))
        await agent.start()
        try:
            return await body(agent)
        finally:
            await agent.stop()

    return asyncio.run(run())


async def _chat(agent: Agent, *script: Any) -> dict[str, Any]:
    agent.llm = ScriptedLLM(*script)  # type: ignore[assignment]
    return await agent.chat(_sid(), "question")


def test_allowlist_hides_and_blocks_other_tools(tmp_path: Path) -> None:
    async def body(agent: Agent) -> tuple[list[str], dict[str, Any], list[str]]:
        llm = ScriptedLLM(calls(tool_call("1", "get_weather", {"city": "Paris"})), answer("Sorry."))
        agent.llm = llm  # type: ignore[assignment]
        out = await agent.chat(_sid(), "weather?")
        offered = [t["function"]["name"] for t in llm.requests[0]["tools"]]
        return [t["name"] for t in agent.tool_names], out, offered

    names, out, offered = _with_agent(tmp_path, body, mcp_tool_allowlist="calculate")
    assert names == ["search_knowledge_base", "calculate"]
    assert offered == ["search_knowledge_base", "calculate"]
    assert out["steps"][0]["output"] == "Error: that tool is not available."


def test_invalid_tool_arguments_are_reported_safely(tmp_path: Path) -> None:
    bad = tool_call("1", "calculate", {})
    bad.function.arguments = "{not json"
    out = _with_agent(tmp_path, lambda agent: _chat(agent, calls(bad), answer("Sorry.")))
    assert out["steps"][0]["output"] == "Error: the tool arguments were invalid."


def test_tool_timeout_is_reported_and_session_recovers(tmp_path: Path) -> None:
    async def body(agent: Agent) -> tuple[str, str]:
        normal = agent.settings
        agent.settings = normal.model_copy(update={"tool_timeout_seconds": 0.000001})
        slow = await _chat(agent, calls(tool_call("1", "get_current_time", {"timezone": "UTC"})), answer("x"))
        agent.settings = normal
        ok = await _chat(agent, calls(tool_call("2", "calculate", {"expression": "1+1"})), answer("two"))
        return slow["steps"][0]["output"], ok["steps"][0]["output"]

    slow, ok = _with_agent(tmp_path, body)
    assert slow == "Error: the tool took too long to respond."
    assert ok.endswith("= 2")  # the shared MCP session still works after a timeout


def test_lost_mcp_session_is_a_tool_error_not_a_crash(tmp_path: Path) -> None:
    async def body(agent: Agent) -> dict[str, Any]:
        session, agent._mcp = agent._mcp, None  # simulate a lost connection
        try:
            return await _chat(agent, calls(tool_call("1", "calculate", {"expression": "1+1"})), answer("x"))
        finally:
            agent._mcp = session

    out = _with_agent(tmp_path, body)
    assert out["steps"][0]["output"] == "Error: the tool is unavailable right now."
    assert out["answer"] == "x"


# ---------------------------------------------------------------- calculator
@pytest.mark.parametrize(
    "expression",
    ["((10**999)**999)**99", "2**5000", "10.0**400", "exp(1000)", "9**9**9", "1" * 300, "sqrt(-1)", "abs()"],
)
def test_calculator_rejects_resource_exhaustion(expression: str) -> None:
    import time

    t0 = time.perf_counter()
    result = tools.calculate(expression)
    assert result.startswith("Error")
    assert time.perf_counter() - t0 < 0.5


def test_calculator_rejects_overly_complex_expressions() -> None:
    assert tools.calculate("+".join(["1"] * 80)).startswith("Error")


# ------------------------------------------------------------------- weather
def _fake_weather_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)  # the weather tool uses httpx.Client


def test_weather_rejects_bad_city_without_network() -> None:
    assert tools.get_weather("").startswith("Error")
    assert tools.get_weather("x" * 101).startswith("Error")


def test_weather_service_down_is_reported_generically(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_weather_client(monkeypatch, lambda req: httpx.Response(503, text="upstream exploded at /srv/x"))
    out = tools.get_weather("Paris")
    assert out == "Error: the weather service is unavailable right now."


def test_weather_malformed_payload_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_weather_client(monkeypatch, lambda req: httpx.Response(200, json={"results": [{"name": "X"}]}))
    assert tools.get_weather("Paris") == "Error: the weather service returned an unexpected response."


def test_weather_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "geocoding" in req.url.host:
            return httpx.Response(
                200, json={"results": [{"name": "Paris", "country": "France", "latitude": 1, "longitude": 2}]}
            )
        current = {
            "temperature_2m": 20,
            "apparent_temperature": 19,
            "relative_humidity_2m": 50,
            "wind_speed_10m": 5,
            "weather_code": 0,
        }
        return httpx.Response(200, json={"current": current})

    _fake_weather_client(monkeypatch, handler)
    assert tools.get_weather("Paris").startswith("Weather in Paris, France: clear sky, 20°C")


# ========================================================= input validation
@pytest.mark.parametrize(
    "payload",
    [
        {"text": "x" * (DEFAULT_MAX_TEXT_CHARS + 1)},
        {"text": "hello\x00world"},
        {"text": "hi", "extra": "zz-unexpected-value"},
        {"text": "hi", "session_id": "../../etc/passwd"},
        {"text": "hi", "session_id": "short"},
        {"text": 123},
        {},
    ],
)
def test_chat_rejects_invalid_input(client: TestClient, payload: Any) -> None:
    r = client.post("/api/chat", json=payload)
    err = _assert_envelope(r, 422, "invalid_request")
    assert err["details"]
    # Submitted values are never echoed back.
    for value in payload.values():
        if isinstance(value, str) and len(value) > 3:
            assert value not in r.text


def test_missing_session_id_gets_a_fresh_private_session(client: TestClient) -> None:
    llm = ScriptedLLM(answer("one"), answer("two"))
    app_agent(client).llm = llm  # type: ignore[assignment]
    first = client.post("/api/chat", json={"text": "my secret is 1234"}).json()
    second = client.post("/api/chat", json={"text": "what is my secret?"}).json()
    assert first["session_id"] != second["session_id"]
    assert len(first["session_id"]) >= 24
    assert "my secret is 1234" not in [m.get("content") for m in llm.requests[1]["messages"]]


def test_reset_requires_valid_session_id(client: TestClient) -> None:
    _assert_envelope(client.post("/api/reset", data={}), 422, "invalid_request")
    _assert_envelope(client.post("/api/reset", data={"session_id": "a b"}), 422, "invalid_request")


# =================================================================== bodies
def test_oversized_json_body_rejected(client: TestClient) -> None:
    big = b'{"text": "' + b"a" * (DEFAULT_MAX_JSON_BODY_BYTES + 10) + b'"}'
    r = client.post("/api/chat", content=big, headers={"content-type": "application/json"})
    _assert_envelope(r, 413, "payload_too_large")


def test_oversized_streamed_body_rejected(client: TestClient) -> None:

    def chunks() -> Iterator[bytes]:  # no Content-Length: the limit must hold while streaming
        yield b'{"text": "'
        for _ in range(200):
            yield b"a" * 1024
        yield b'"}'

    r = client.post("/api/chat", content=chunks(), headers={"content-type": "application/json"})
    _assert_envelope(r, 413, "payload_too_large")


def test_oversized_audio_upload_rejected_before_reading(client: TestClient) -> None:
    big = b"\0" * (client.app.state.settings.stt_max_body_bytes + 1)  # type: ignore[attr-defined]
    r = client.post("/api/stt", files={"audio": ("a.wav", big, "audio/wav")})
    _assert_envelope(r, 413, "payload_too_large")


def test_invalid_content_length_rejected(client: TestClient) -> None:
    r = client.post("/api/reset", content=b"session_id=abcdefgh", headers={"content-length": "abc"})
    assert r.status_code == 400


def test_stt_rejects_wrong_content_type(client: TestClient) -> None:
    r = client.post("/api/stt", files={"audio": ("a.mp3", make_wav(), "audio/mpeg")})
    _assert_envelope(r, 415, "unsupported_media_type")


def test_stt_rejects_non_wav_bytes(client: TestClient) -> None:
    r = client.post("/api/stt", files={"audio": ("a.wav", b"<html>" + b"x" * 5000, "audio/wav")})
    _assert_envelope(r, 415, "unsupported_media_type")


def test_stt_rejects_too_long_audio(client: TestClient) -> None:
    with override_settings(client, max_audio_bytes=10_000):
        r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(seconds=1), "audio/wav")})
    _assert_envelope(r, 413, "payload_too_large")


# ===================================================== error sanitisation
def test_llm_error_details_are_not_leaked(client: TestClient) -> None:
    sensitive = "sk-or-v1-LEAKED-KEY at /Users/ops/app/agent.py"
    app_agent(client).llm = ScriptedLLM(  # type: ignore[assignment]
        *[connection_error(sensitive)] * len(app_agent(client).settings.llm_models)
    )
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    _assert_envelope(r, 502, "llm_unavailable")
    assert "LEAKED" not in r.text and "/Users" not in r.text


def test_unexpected_exception_returns_generic_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def boom(*_: Any) -> Any:
        raise RuntimeError("database password=hunter2 at /srv/internal.py")

    monkeypatch.setattr(app_agent(client), "chat", boom)
    r = client.post("/api/chat", json={"session_id": _sid(), "text": "hi"})
    _assert_envelope(r, 500, "internal_error")
    assert "hunter2" not in r.text and "RuntimeError" not in r.text
    assert r.headers["x-content-type-options"] == "nosniff"  # headers survive the error path


def test_stt_provider_error_is_sanitised(client: TestClient) -> None:
    app_agent(client).llm = ScriptedLLM(connection_error("upstream says: key sk-or-v1-SECRET invalid"))  # type: ignore[assignment]
    r = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    _assert_envelope(r, 502, "stt_failed")
    assert "SECRET" not in r.text


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/api/nope"), 404, "not_found")


# ============================================================ request IDs
def test_request_id_is_generated_or_propagated(client: TestClient) -> None:
    generated = client.get("/api/status").headers["x-request-id"]
    assert len(generated) == 32
    assert client.get("/api/status", headers={"X-Request-ID": "trace-12345678"}).headers["x-request-id"] == (
        "trace-12345678"
    )
    injected = client.get("/api/status", headers={"X-Request-ID": "bad id\r\nSet-Cookie: x"})
    assert injected.headers["x-request-id"] != "bad id\r\nSet-Cookie: x"


# ================================================================ headers
def test_security_headers_on_ui_and_api(client: TestClient) -> None:
    for path in ("/", "/api/status", "/static/app.js"):
        h = client.get(path).headers
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"
        assert "script-src 'self'" in h["content-security-policy"]
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert "strict-transport-security" not in h  # development mode


def test_docs_available_in_development(client: TestClient) -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    assert "content-security-policy" not in r.headers  # Swagger UI needs its CDN assets


def test_production_disables_docs_and_enables_hsts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, environment="production", openrouter_api_key="k")
    app = create_app(settings, agent_factory=offline_agent)
    assert (app.docs_url, app.redoc_url, app.openapi_url) == (None, None, None)
    with TestClient(app) as prod:
        assert prod.get("/docs").status_code == 404
        assert prod.get("/openapi.json").status_code == 404
        assert "max-age=" in prod.get("/health/live").headers["strict-transport-security"]


# ============================================================ rate limits
def test_rate_limit_returns_429_with_retry_after(client: TestClient) -> None:
    state = client.app.state  # type: ignore[attr-defined]
    # Swap explicitly: pytest's monkeypatch cannot restore attributes on Starlette's State.
    original, state.limiter = state.limiter, InMemoryRateLimiter(per_minute=60, burst=2)
    try:
        app_agent(client).llm = ScriptedLLM(answer("a"), answer("b"))  # type: ignore[assignment]
        sid = _sid()
        assert client.post("/api/chat", json={"session_id": sid, "text": "1"}).status_code == 200
        assert client.post("/api/chat", json={"session_id": sid, "text": "2"}).status_code == 200
        r = client.post("/api/chat", json={"session_id": sid, "text": "3"})
    finally:
        state.limiter = original
    _assert_envelope(r, 429, "rate_limited")
    assert int(r.headers["retry-after"]) >= 1


def test_rate_limiter_refills_and_bounds_memory() -> None:
    now = [0.0]
    limiter = InMemoryRateLimiter(per_minute=60, burst=1, max_keys=3, clock=lambda: now[0])

    async def run() -> None:
        assert (await limiter.hit("a")).allowed
        denied = await limiter.hit("a")
        assert not denied.allowed and denied.retry_after_seconds == 1
        now[0] += 1.0
        assert (await limiter.hit("a")).allowed
        for k in "bcdef":
            await limiter.hit(k)
        assert len(limiter) == 3

    asyncio.run(run())


# ================================================================== admin
def test_admin_disabled_without_key(client: TestClient) -> None:
    # The module app has no ADMIN_API_KEY configured.
    _assert_envelope(client.get("/admin/diagnostics", headers={"X-Admin-Key": ""}), 404, "not_found")


def test_admin_requires_correct_key(client: TestClient) -> None:
    with override_settings(client, admin_api_key=ADMIN_KEY):
        _assert_envelope(client.get("/admin/diagnostics"), 401, "unauthorized")
        wrong = client.get("/admin/diagnostics", headers={"X-Admin-Key": "b" * 40})
        _assert_envelope(wrong, 401, "unauthorized")
        r = client.get("/admin/diagnostics", headers={"X-Admin-Key": ADMIN_KEY})
    assert r.status_code == 200
    body = r.json()
    assert {"os", "python", "cpu_count", "disk_free_gb"} <= body["host"].keys()
    assert "calculate" in body["agent"]["exposed_tools"]


# ================================================================ logging
def test_user_content_not_logged_by_default(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    sensitive = "my-account-number-99887766"
    app_agent(client).llm = ScriptedLLM(  # type: ignore[assignment]
        calls(tool_call("1", "search_knowledge_base", {"query": sensitive})), answer("ok"), answer(sensitive)
    )
    with caplog.at_level(logging.DEBUG):
        client.post("/api/chat", json={"session_id": _sid(), "text": sensitive})
        client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert sensitive not in caplog.text
    assert "Route -> RAG tool=search_knowledge_base" in caplog.text
