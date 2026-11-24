"""Customer-facing API routes: status, speech-to-text, chat and reset."""

from __future__ import annotations

import re
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dependencies import AgentDep, LimiterDep, SettingsDep, enforce_rate_limit, rate_limited
from .errors import NotConfiguredError
from .request_context import session_id_var

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
# Hard ceiling for the schema; the configured MAX_TEXT_CHARS (lower) is checked per request.
ABSOLUTE_MAX_TEXT_CHARS = 20_000
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
MIN_AUDIO_BYTES = 2000  # shorter clips cannot contain speech; skip the paid STT call

router = APIRouter(prefix="/api", tags=["voice agent"])


# ------------------------------------------------------------------ models
class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Conversation ID. Omit to start a new conversation; the response returns it.",
    )
    text: str = Field(min_length=1, max_length=ABSOLUTE_MAX_TEXT_CHARS, description="The user's message.")

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
    kind: str = Field(description='"rag" or "mcp"')
    tool: str
    args: dict[str, Any]
    output: str


class ChatOut(BaseModel):
    session_id: str
    answer: str
    steps: list[Step] = Field(description="Tools the agent used, in order (the routing trace).")
    latency_ms: int


class SttOut(BaseModel):
    text: str


class ToolInfo(BaseModel):
    name: str
    kind: str


class StatusOut(BaseModel):
    ready: bool
    llm_model: str
    stt_model: str
    stt_mode: str
    embedder: str
    chunks: int
    tools: list[ToolInfo]


class ResetOut(BaseModel):
    ok: bool


def _require_llm(settings: SettingsDep) -> None:
    if not settings.llm_configured:
        raise NotConfiguredError()


# ------------------------------------------------------------------ routes
@router.get("/status", response_model=StatusOut, summary="Pipeline status for the UI")
async def status(settings: SettingsDep, agent: AgentDep) -> dict[str, Any]:
    return {
        "ready": settings.llm_configured,
        "llm_model": settings.llm_models[0],
        "stt_model": settings.stt_model if settings.stt_mode == "openrouter" else "browser",
        "stt_mode": settings.stt_mode,
        "embedder": agent.store.embedder,
        "chunks": len(agent.store.chunks),
        "tools": agent.tool_names,
    }


@router.post(
    "/stt",
    response_model=SttOut,
    dependencies=[Depends(rate_limited)],
    summary="Transcribe a short WAV clip",
)
async def stt(
    audio: Annotated[UploadFile, File(description="16-bit PCM WAV, mono")],
    settings: SettingsDep,
    agent: AgentDep,
) -> dict[str, str]:
    _require_llm(settings)
    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    if content_type not in WAV_CONTENT_TYPES:
        raise HTTPException(415, "Audio must be a WAV file.")
    data = await audio.read(settings.max_audio_bytes + 1)
    if len(data) > settings.max_audio_bytes:
        raise HTTPException(413, "The audio clip is too long.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(415, "Audio must be a WAV file.")
    if len(data) < MIN_AUDIO_BYTES:
        return {"text": ""}
    return {"text": await agent.transcribe(data)}


@router.post(
    "/chat",
    response_model=ChatOut,
    dependencies=[Depends(rate_limited)],
    summary="Send a message; the agent routes it to RAG and/or tools and answers",
)
async def chat(body: ChatIn, settings: SettingsDep, agent: AgentDep, limiter: LimiterDep) -> dict[str, Any]:
    if len(body.text) > settings.max_text_chars:
        raise RequestValidationError(
            [
                {
                    "loc": ("body", "text"),
                    "msg": f"String should have at most {settings.max_text_chars} characters",
                    "type": "string_too_long",
                }
            ]
        )
    _require_llm(settings)
    session_id = body.session_id or secrets.token_urlsafe(18)
    session_id_var.set(session_id)
    await enforce_rate_limit(limiter, f"session:{session_id}")
    result = await agent.chat(session_id, body.text)
    return {"session_id": session_id, **result}


@router.post(
    "/reset",
    response_model=ResetOut,
    dependencies=[Depends(rate_limited)],
    summary="Clear a conversation",
)
async def reset(
    session_id: Annotated[str, Form(pattern=SESSION_ID_PATTERN)], agent: AgentDep
) -> dict[str, bool]:
    agent.reset(session_id)
    return {"ok": True}
