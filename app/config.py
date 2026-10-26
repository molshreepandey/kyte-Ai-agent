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

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

KB_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "app" / "static"
LOG_DIR = ROOT / "logs"

MAX_TOOL_ROUNDS = 5
