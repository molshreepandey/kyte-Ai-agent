"""Application factory: serves the web UI and the voice/chat API.

Importing this module has no side effects. `create_app()` builds an app; its lifespan
configures logging, creates runtime folders, builds the knowledge index and starts
the MCP tool server. Run with either::

    python -m app.main
    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from . import admin, api, health
from .agent import Agent
from .config import Settings, get_settings
from .errors import install_error_handlers
from .logging_config import configure_logging
from .ratelimit import InMemoryRateLimiter
from .security import BodySizeLimitMiddleware, RequestContextMiddleware

log = logging.getLogger("app")

APP_TITLE = "AI Voice Agent"
APP_DESCRIPTION = (
    "Voice-first assistant API. The LLM routes each message to the knowledge base (RAG) "
    "and/or allowlisted MCP tools. All errors use the format "
    '`{"error": {"code", "message", "request_id", "details?"}}`.'
)


def create_app(
    settings: Settings | None = None,
    *,
    agent_factory: Callable[[Settings], Agent] = Agent,
) -> FastAPI:
    """Build the application. `agent_factory` is a seam for tests (e.g. an offline index)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        agent = agent_factory(settings)
        app.state.agent = agent
        app.state.limiter = InMemoryRateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        app.state.started_at = time.monotonic()
        await agent.start()
        log.info(
            "%s started",
            APP_TITLE,
            extra={
                "environment": settings.environment,
                "llm_configured": settings.llm_configured,
                "chunks": len(agent.store.chunks),
                "embedder": agent.store.embedder,
                "tools": [t["name"] for t in agent.tool_names],
            },
        )
        if not settings.llm_configured:
            log.warning("OPENROUTER_API_KEY is not set: chat and speech-to-text will return 503")
        try:
            yield
        finally:
            await agent.stop()
            log.info("%s stopped", APP_TITLE)

    docs = not settings.is_production
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings

    # Middleware added last runs first: request context wraps everything, including size checks.
    app.add_middleware(
        BodySizeLimitMiddleware,
        default_max=settings.max_json_body_bytes,
        path_limits={"/api/stt": settings.stt_max_body_bytes},
    )
    app.add_middleware(RequestContextMiddleware, hsts=settings.is_production)
    if settings.trust_proxy_headers:
        # Outermost: set the real client address before rate limiting sees the request.
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    install_error_handlers(app)

    app.include_router(api.router)
    app.include_router(health.router)
    app.include_router(admin.router)

    index_html = settings.static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_html)

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    return app


class _LazyApp:
    """ASGI entry point for ``uvicorn app.main:app`` (without ``--factory``).

    The real application is built on the first ASGI call (uvicorn's lifespan startup),
    so importing this module still reads no settings and has no side effects.
    """

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._app is None:
            self._app = create_app()
        await self._app(scope, receive, send)


app = _LazyApp()


def describe_config_errors(exc: ValidationError) -> str:
    """One line per invalid setting, naming the environment variable. Input values are
    deliberately omitted: they may be secrets."""
    lines = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]).upper() or "SETTINGS"
        lines.append(f"  {field}: {err['msg']}")
    return "Invalid configuration:\n" + "\n".join(lines)


def run() -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(describe_config_errors(exc), file=sys.stderr)
        raise SystemExit(2) from None
    print(f"{APP_TITLE} -> http://{settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,  # our logging configuration (set in the lifespan) handles uvicorn's loggers
        access_log=False,  # RequestContextMiddleware writes structured access logs
    )


if __name__ == "__main__":
    run()
