"""FastAPI server: serves the web UI and the voice/chat API."""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import admin, config
from .agent import Agent
from .errors import NotConfiguredError, install_error_handlers
from .ratelimit import InMemoryRateLimiter
from .request_context import session_id_var
from .security import BodySizeLimitMiddleware, RequestContextMiddleware

config.LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(config.LOG_DIR / "agent.log", encoding="utf-8"), logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("api")

agent = Agent()
limiter = InMemoryRateLimiter(config.RATE_LIMIT_PER_MINUTE, config.RATE_LIMIT_BURST)

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
MIN_AUDIO_BYTES = 2000  # shorter clips cannot contain speech; skip the paid STT call
MULTIPART_OVERHEAD_BYTES = 64 * 1024
IS_PRODUCTION = config.ENVIRONMENT == "production"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.agent = agent
    app.state.limiter = limiter
    await agent.start()
    yield
    await agent.stop()


app = FastAPI(
    title="AI Voice Agent",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)
STT_MAX_BODY_BYTES = config.MAX_AUDIO_BYTES + MULTIPART_OVERHEAD_BYTES
# Middleware added last runs first: request context wraps everything, including size checks.
app.add_middleware(
    BodySizeLimitMiddleware,
    default_max=config.MAX_JSON_BODY_BYTES,
    path_limits={"/api/stt": STT_MAX_BODY_BYTES},
)
app.add_middleware(RequestContextMiddleware, hsts=IS_PRODUCTION)
install_error_handlers(app)


# ------------------------------------------------------------------ models
class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Conversation ID. Omit to start a new conversation; the response returns it.",
    )
    text: str = Field(min_length=1, max_length=config.MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        if _CONTROL_CHARS.search(value):
            raise ValueError("must not contain control characters")
        return value


class Step(BaseModel):
    kind: str
    tool: str
    args: dict[str, Any]
    output: str


class ChatOut(BaseModel):
    session_id: str
    answer: str
    steps: list[Step]
    latency_ms: int


class SttOut(BaseModel):
    text: str


# ------------------------------------------------------------ dependencies
async def rate_limited(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    await _enforce(f"ip:{client}")


async def _enforce(key: str) -> None:
    decision = await limiter.hit(key)
    if not decision.allowed:
        raise HTTPException(
            429,
            "Too many requests. Please wait a moment and try again.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )


def _require_key() -> None:
    if not config.OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY is not set; refusing LLM-backed request")
        raise NotConfiguredError()


# ------------------------------------------------------------------ routes
@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "ready": bool(config.OPENROUTER_API_KEY),
        "llm_model": config.LLM_MODEL,
        "stt_model": config.STT_MODEL if config.STT_MODE == "openrouter" else "browser",
        "stt_mode": config.STT_MODE,
        "embedder": agent.store.embedder,
        "chunks": len(agent.store.chunks),
        "tools": agent.tool_names,
    }


@app.post(
    "/api/stt",
    response_model=SttOut,
    dependencies=[Depends(rate_limited)],
)
async def stt(audio: Annotated[UploadFile, File(description="16-bit PCM WAV, mono")]) -> dict[str, str]:
    _require_key()
    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    if content_type not in WAV_CONTENT_TYPES:
        raise HTTPException(415, "Audio must be a WAV file.")
    data = await audio.read(config.MAX_AUDIO_BYTES + 1)
    if len(data) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, "The audio clip is too long.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(415, "Audio must be a WAV file.")
    if len(data) < MIN_AUDIO_BYTES:
        return {"text": ""}
    return {"text": await agent.transcribe(data)}


@app.post("/api/chat", response_model=ChatOut, dependencies=[Depends(rate_limited)])
async def chat(body: ChatIn) -> dict[str, Any]:
    _require_key()
    session_id = body.session_id or secrets.token_urlsafe(18)
    session_id_var.set(session_id)
    await _enforce(f"session:{session_id}")
    result = await agent.chat(session_id, body.text)
    return {"session_id": session_id, **result}


@app.post("/api/reset", dependencies=[Depends(rate_limited)])
async def reset(session_id: Annotated[str, Form(pattern=SESSION_ID_PATTERN)]) -> dict[str, bool]:
    agent.reset(session_id)
    return {"ok": True}


app.include_router(admin.router, dependencies=[Depends(rate_limited)])


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


def run() -> None:
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    print(f"AI Voice Agent -> http://{config.HOST}:{config.PORT}")
    run()
