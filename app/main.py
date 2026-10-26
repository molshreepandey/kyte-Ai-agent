"""FastAPI server: serves the web UI and the voice/chat API."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .agent import Agent

config.LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(config.LOG_DIR / "agent.log", encoding="utf-8"),
              logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("api")

agent = Agent()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await agent.start()
    yield
    await agent.stop()


app = FastAPI(title="Kyte Voice Agent", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str = "default"
    text: str


def _require_key() -> None:
    if not config.OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not set. Add it to .env and restart.")


@app.get("/api/status")
async def status():
    return {
        "ready": bool(config.OPENROUTER_API_KEY),
        "llm_model": config.LLM_MODEL,
        "stt_model": config.STT_MODEL if config.STT_MODE == "openrouter" else "browser",
        "stt_mode": config.STT_MODE,
        "embedder": agent.store.embedder,
        "chunks": len(agent.store.chunks),
        "tools": agent.tool_names,
    }


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    _require_key()
    data = await audio.read()
    if len(data) < 2000:
        return {"text": ""}
    try:
        return {"text": await agent.transcribe(data)}
    except Exception as exc:
        log.error("STT failed: %s", exc)
        raise HTTPException(502, f"Speech-to-text failed: {exc}")


@app.post("/api/chat")
async def chat(body: ChatIn):
    _require_key()
    if not body.text.strip():
        raise HTTPException(400, "Empty message")
    try:
        return await agent.chat(body.session_id, body.text.strip())
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(502, f"LLM request failed: {exc}")


@app.post("/api/reset")
async def reset(session_id: str = Form("default")):
    agent.reset(session_id)
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


def run() -> None:
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    print(f"Kyte Voice Agent -> http://{config.HOST}:{config.PORT}")
    run()
