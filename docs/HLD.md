# High-Level Design

```mermaid
flowchart TD
    U([User speaks]) --> MIC["Browser mic capture<br/>MediaRecorder + VAD (silence detection)"]
    MIC -->|"16 kHz mono WAV"| STT["POST /api/stt<br/>Speech-to-Text<br/>(OpenRouter audio model)"]
    STT -->|transcript| CHAT["POST /api/chat<br/>Agent orchestrator (FastAPI)"]

    subgraph BRAIN["LLM decision routing  -  OpenRouter tool-calling loop"]
        LLM{{"LLM<br/>reads query + tool schemas<br/>decides which tool(s) to call"}}
    end
    CHAT --> LLM

    LLM -->|"domain question<br/>search_knowledge_base(query)"| RAG
    LLM -->|"time / weather / math<br/>MCP tools/call (allowlisted)"| MCP

    subgraph RAG["RAG pipeline"]
        direction TB
        KB[("knowledge_base/*.md<br/>FAQ + product specs<br/>(sample: Demo Wallet)")] -->|"chunk by section"| EMB["Embeddings<br/>OpenRouter, local TF-IDF fallback"]
        EMB --> VS[("Vector store<br/>numpy + cosine similarity")]
        VS --> TOPK["Top-k chunks"]
    end

    subgraph MCP["MCP server (stdio subprocess)"]
        direction TB
        T1["get_current_time"]
        T2["calculate (safe AST)"]
        T3["get_weather (Open-Meteo)"]
    end

    TOPK -->|"context as tool result"| LLM
    MCP -->|"tool result"| LLM
    LLM -->|"final short spoken answer"| TTS["Text-to-Speech<br/>browser speechSynthesis"]
    TTS --> OUT([User hears answer])
    LLM -.->|"answer + routing trace"| UI["Web UI: transcript,<br/>RAG / MCP step cards"]
    OUT -. "call mode: listen again" .-> MIC
```

## Components

| Layer | File | Responsibility |
|---|---|---|
| Voice UI | `app/static/*` | Call UI, mic capture, VAD, WAV encoding, TTS playback, routing trace |
| App factory | `app/main.py` | `create_app()`: lifespan (logging, folders, index, MCP), middleware, routers |
| API | `app/api.py`, `app/health.py` | `/api/stt`, `/api/chat`, `/api/status`, `/api/reset`; `/health/live`, `/health/ready` |
| Config | `app/config.py` | Validated settings from env / `.env`; fails fast on invalid values |
| Logging | `app/logging_config.py` | JSON or text logs with request and session IDs |
| Orchestrator | `app/agent.py` | MCP client, tool schema merge, LLM tool-calling loop, STT call |
| RAG | `app/rag.py` | Chunking, embeddings, persistent vector store, cosine search |
| MCP server | `app/mcp_server.py` | Customer-safe tools (time, bounded calculator, weather) over the Model Context Protocol |
| Security | `app/security.py`, `app/errors.py`, `app/ratelimit.py` | Request IDs, security headers, body limits, error format, rate limiting, admin auth |
| Admin | `app/admin.py` | Operator diagnostics behind `ADMIN_API_KEY` |

## Design decisions and trade-offs

1. **The LLM is the router.** RAG is exposed as just another tool next to the MCP tools. No hand-written intent classifier, so a query like *"fee for 2,500 USD abroad on Premium"* can chain RAG (fee rate) and then MCP `calculate` (exact maths). Trade-off: routing quality depends on the model and on clear tool descriptions.
2. **MCP over stdio subprocess.** The server is a real, standalone MCP server (`python -m app.mcp_server`) that any MCP client can use. The agent discovers tools at startup via `tools/list`, so adding a tool needs no agent change. Trade-off: one extra process.
3. **Own vector store (numpy) instead of Chroma/FAISS.** The corpus is tiny. A normalised matrix and one matmul gives exact cosine search, zero native dependencies, and it works on any Python version. Swap in a real vector DB past ~100k chunks.
4. **Embedding fallback.** If OpenRouter embeddings are unavailable, a local TF-IDF hashing embedder keeps retrieval working offline, at lower quality for paraphrases.
5. **STT via an audio-capable LLM on OpenRouter**, since OpenRouter has no dedicated Whisper endpoint. This keeps one API key and one vendor. Trade-off: higher latency than a streaming STT engine. The browser downsamples to 16 kHz WAV so any model accepting `input_audio` works.
6. **TTS in the browser (`speechSynthesis`).** Zero cost and zero latency to first audio. Trade-off: voice quality depends on the OS. Replaceable by a server TTS behind the same `speak()` function.
7. **Voice-first prompt.** The system prompt forces short, markdown-free replies, which sound natural when spoken, and keeps latency down.
8. **Safety.** The calculator parses with `ast` and a whitelist (no `eval`), and bounds expression length, AST size and integer size so no input can exhaust CPU or memory. The RAG prompt restricts answers to retrieved context to limit hallucination.
9. **Least privilege for tools.** Every MCP tool is reachable by anonymous customers through the LLM, so only allowlisted, side-effect-free tools are exposed, each call has a timeout, and tool failures are returned as generic messages. Host diagnostics moved to an authenticated admin endpoint; the former log-reading tool was removed because logs contain other users' data.
10. **Explicit startup, no import side effects.** `create_app(settings)` builds an isolated app; the lifespan configures logging, creates folders, builds the index and starts MCP. Tests build many independent apps from explicit settings, and the same factory runs under `uvicorn --factory`.
