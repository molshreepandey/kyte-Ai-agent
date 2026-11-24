# AI Voice Agent

A voice-activated AI agent that answers by combining **RAG** (knowledge retrieval) and **MCP** (real-time tool execution), with a "phone call" web UI. Built in Python; the LLM and embeddings go through [OpenRouter](https://openrouter.ai). Speech-to-text uses the browser's Web Speech API by default, or an audio-capable OpenRouter model when `STT_MODE=openrouter`.

**Pipeline:** mic audio → STT → LLM decides (RAG and/or MCP tool) → tool result → LLM answer → TTS

- **RAG:** FAQ and product spec documents (`knowledge_base/`) are chunked, embedded and searched by cosine similarity.
- **MCP:** a standalone MCP server (`app/mcp_server.py`) exposes customer-safe tools: current time, a bounded safe calculator and live weather. Only tools listed in `MCP_TOOL_ALLOWLIST` are shown to the LLM.
- **Voice UI:** hands-free call mode with silence detection, live waveform, barge-in, and a routing trace that shows which RAG / MCP step each answer used. A text box is available as fallback.

See [docs/HLD.md](docs/HLD.md) for the architecture diagram and design trade-offs.

```mermaid
flowchart LR
    A([Voice]) --> B[STT] --> C{LLM router}
    C -->|domain Q| D[(RAG: vector store)]
    C -->|system Q| E[MCP server tools]
    D --> C
    E --> C
    C --> F[TTS] --> G([Spoken answer])
```

## Prerequisites

- Python 3.12+ (on Apple Silicon, use a native arm64 Python; an x86_64/Rosetta build cannot compile the `cryptography` dependency)
- An OpenRouter API key: https://openrouter.ai/keys (some credit is needed for the paid models)
- **Chrome or Edge** for the UI (microphone + speech synthesis). Allow microphone access when asked.

## Setup (Windows PowerShell)

```powershell
git clone <your-repo-url> ai-voice-agent
cd ai-voice-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
notepad .env        # paste your OPENROUTER_API_KEY
```

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
```

## Run

```bash
python -m app.main
```

Open **http://127.0.0.1:8000** and click **Start call**. Speak, pause, and the agent answers aloud. Click the orb while it speaks to interrupt it. Click **End call** to hang up.

On startup the app builds the vector index (cached in `data/`), launches the MCP server as a subprocess, and lists the discovered tools in the UI.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | yes | none | OpenRouter key for LLM, STT and embeddings |
| `LLM_MODEL` | no | `openai/gpt-4o-mini` | Chat model, or a comma-separated fallback list tried in order; must support tool calling |
| `STT_MODE` | no | `browser` | `browser` = free Web Speech API (Chrome/Edge); `openrouter` = audio model via OpenRouter (needs >= $0.50 credit) |
| `STT_MODEL` | no | `google/gemini-2.5-flash` | Audio-input model used for transcription |
| `EMBEDDING_MODEL` | no | `openai/text-embedding-3-small` | RAG embeddings (falls back to local TF-IDF if unavailable) |
| `COMPANY_NAME` | no | `Demo Wallet` | Business the assistant represents (used in the prompt and RAG tool description) |
| `COMPANY_DESCRIPTION` | no | `a digital money-management app` | One-line description of that business |
| `KB_TOPICS` | no | `plans, fees, limits, ...` | Topics the knowledge base covers; tells the LLM when to search it |
| `AGENT_NAME` | no | (none) | Optional name the assistant introduces itself with |
| `ENVIRONMENT` | no | `development` | `production` disables `/docs`, `/redoc`, `/openapi.json` and enables HSTS |
| `ADMIN_API_KEY` | no | (none) | Enables `/admin/*` (sent as `X-Admin-Key`). Unset = admin endpoints do not exist. Use 32+ random characters |
| `MCP_TOOL_ALLOWLIST` | no | `get_current_time,calculate,get_weather` | MCP tools the LLM may see and call |
| `TOOL_TIMEOUT_SECONDS` | no | `10` | Per-call timeout for MCP tools |
| `LLM_TIMEOUT_SECONDS` | no | `30` | Per-request timeout for LLM / STT calls |
| `MAX_TEXT_CHARS` | no | `2000` | Maximum chat message length |
| `MAX_AUDIO_BYTES` | no | `2000000` | Maximum STT upload size (about 60 s of 16 kHz audio) |
| `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_BURST` | no | `30` / `10` | Per-client-IP (and per-session) token bucket for `/api/*` and `/admin/*` |
| `LOG_TRANSCRIPTS` | no | `false` | Log user transcripts and tool arguments (debugging only; this is user data) |
| `HOST` / `PORT` | no | `127.0.0.1` / `8000` | Server bind address |

## Try these queries

| Query | Route |
|---|---|
| "What are the transfer limits on the Plus plan?" | RAG |
| "What's the weather in Mumbai right now?" | MCP `get_weather` |
| "If I send 2,500 dollars abroad on the Premium plan, how much is the fee?" | RAG (0.4% fee) then MCP `calculate` |
| "What time is it in New York, and when does your phone support close?" | MCP `get_current_time` + RAG |

## Project layout

```
app/
  main.py          FastAPI server and API routes
  agent.py         Orchestrator: MCP client, LLM tool-calling loop, STT
  rag.py           Chunking, embeddings, vector store, search
  mcp_server.py    MCP server with the system tools
  config.py        Environment config
  static/          Web UI (index.html, style.css, app.js)
knowledge_base/    Fictional "Demo Wallet" sample FAQ + product specs (replace with your own .md/.txt)
docs/HLD.md        High-level design diagram + trade-offs
tests/             Offline tests (no API key needed)
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | UI status: models, index size, exposed tools |
| `POST` | `/api/chat` | `{"text", "session_id"?}` -> `{"session_id", "answer", "steps", "latency_ms"}`. Omit `session_id` to start a new private conversation |
| `POST` | `/api/stt` | multipart `audio` (16-bit PCM WAV) -> `{"text"}` |
| `POST` | `/api/reset` | form `session_id` -> clears that conversation |
| `GET` | `/admin/diagnostics` | Host and agent diagnostics. Requires `X-Admin-Key` |

Interactive docs are at `/docs` in development. Every error uses one format, and internal exception details are never returned:

```json
{"error": {"code": "invalid_request", "message": "The request is invalid.", "request_id": "3f2c...", "details": [{"field": "body.text", "message": "..."}]}}
```

The `request_id` matches the `X-Request-ID` response header and the server logs.

## Security

- **Tools:** only allowlisted, reviewed MCP tools reach the LLM. Host diagnostics are at the admin endpoint, not exposed as LLM tools; there is no log-reading tool or endpoint.
- **Input limits:** message length, session ID format, audio type/size and request body size (64 KB JSON, ~2 MB audio) are validated server-side.
- **Rate limiting:** token bucket per client IP and per session. The limiter is in-process; behind a reverse proxy run uvicorn with `--proxy-headers` so the client IP is correct.
- **Headers:** strict Content-Security-Policy (no inline or third-party scripts), `nosniff`, `DENY` framing, `no-referrer`; HSTS in production.
- **Secrets:** only in environment variables / `.env` (git-ignored). Never commit `.env`.

## Customising

- **Use your own business:** replace the sample files in `knowledge_base/` and set `COMPANY_NAME`, `COMPANY_DESCRIPTION` and `KB_TOPICS` in `.env`.
- **Add knowledge:** drop `.md` / `.txt` files in `knowledge_base/` (use `## Heading` per topic) and restart. The index rebuilds automatically.
- **Add a tool:** add a function decorated with `@mcp.tool()` in `app/mcp_server.py`, review it (every tool is reachable by anonymous customers through the LLM, so it must be side-effect free, bounded in time and memory, and must not reveal host or other users' data), add its name to `MCP_TOOL_ALLOWLIST`, and restart.
- **Debug the MCP server alone:** `npx @modelcontextprotocol/inspector python -m app.mcp_server`

## Development

```bash
pip install -r requirements-dev.txt   # production deps + pytest, ruff, mypy

ruff check .            # lint
ruff format .           # format
mypy                    # type-check (strict)
pytest                  # backend tests
```

Browser tests (Playwright, needs Node 20+):

```bash
npm ci
npx playwright install chromium
npx playwright test     # starts the server itself; LLM calls are stubbed
```

No test needs an API key or network access. The backend suite covers calculator safety, RAG retrieval ranking, the agent routing loop (scripted LLM, real MCP server over stdio) and the HTTP API. CI (`.github/workflows/ci.yml`) runs all of the above on every push and pull request.

`requirements.txt` pins exact production versions; `requirements-dev.txt` adds the tooling. Upgrade by editing a pin and re-running the full suite.

## Troubleshooting

- **402 "requires more credits"**: your OpenRouter balance is too low. Add credit at https://openrouter.ai/settings/credits ($5 lasts a long time with `gpt-4o-mini`), or set a free model in `.env`, e.g. `LLM_MODEL=google/gemma-4-31b-it:free` (rate-limited; pick another `:free` model with tool support if it errors).

- **"OPENROUTER_API_KEY is not set"**: create `.env` from `.env.example` and restart.
- **Speech-to-text failed (400/404)**: your `STT_MODEL` doesn't accept audio. Use `google/gemini-2.5-flash` or another model with audio input.
- **Mic button does nothing**: use Chrome/Edge on `localhost` and allow microphone permission. You can still type.
- **No spoken reply**: check that the "Voice on" toggle is enabled and the system volume is up.
- **"The assistant is not configured yet"** in the UI: the server has no `OPENROUTER_API_KEY`. The server log states the exact cause; customers only see the generic message.
- Logs are written to stdout and `logs/agent.log`. User transcripts and tool arguments are not logged unless `LOG_TRANSCRIPTS=true`.

## Demo video outline (3-5 min)

1. (0:00-2:00) Walk through `docs/HLD.md`: voice in, STT, LLM router, RAG vs MCP branches, TTS out. Explain the trade-offs.
2. (2:00-4:30) Live demo: ask the "2,500 dollars on Premium" question (RAG then MCP), then the weather and time queries, and show the routing trace cards.
