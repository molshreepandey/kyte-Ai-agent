"""Central configuration, loaded from environment / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Comma-separated list: tried in order, so a rate-limited free model falls through to the next.
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODEL", "openai/gpt-4o-mini").split(",") if m.strip()]
LLM_MODEL = LLM_MODELS[0]
STT_MODEL = os.getenv("STT_MODEL", "google/gemini-2.5-flash")
# "browser" = free Web Speech API in Chrome/Edge; "openrouter" = audio model (needs >= $0.50 credit)
STT_MODE = os.getenv("STT_MODE", "browser").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Persona / business the assistant represents. Defaults describe the fictional sample
# knowledge base shipped in knowledge_base/ ("Demo Wallet").
AGENT_NAME = os.getenv("AGENT_NAME", "").strip()
COMPANY_NAME = os.getenv("COMPANY_NAME", "Demo Wallet").strip()
COMPANY_DESCRIPTION = os.getenv("COMPANY_DESCRIPTION", "a digital money-management app").strip()
KB_TOPICS = os.getenv(
    "KB_TOPICS", "plans, fees, limits, cards, savings, bills, support, security and how-tos"
).strip()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

KB_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "app" / "static"
LOG_DIR = ROOT / "logs"

MAX_TOOL_ROUNDS = 5


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------- security
# "development" or "production". Production disables the interactive API docs and
# enables HSTS.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
# Protects /admin/* endpoints. Leave empty to disable them entirely.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
# MCP tools the LLM may see and call. Anything else the MCP server advertises is ignored,
# so adding a tool to the server never exposes it to customers by accident.
MCP_TOOL_ALLOWLIST = [
    t.strip()
    for t in os.getenv("MCP_TOOL_ALLOWLIST", "get_current_time,calculate,get_weather").split(",")
    if t.strip()
]
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(2_000_000)))
MAX_JSON_BODY_BYTES = 64 * 1024
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))
# Transcripts and tool arguments are user content. Keep them out of logs unless debugging.
LOG_TRANSCRIPTS = _bool("LOG_TRANSCRIPTS", False)
