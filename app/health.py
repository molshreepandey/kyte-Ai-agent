"""Health probes for load balancers and orchestrators. Not rate limited, no auth,
and they reveal only pass/fail per dependency."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .dependencies import AgentDep, SettingsDep

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="The process is up and serving requests")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="The service can answer customers (index loaded, tools connected, LLM configured)",
    responses={503: {"description": "One or more dependencies are not ready"}},
)
async def ready(settings: SettingsDep, agent: AgentDep) -> JSONResponse:
    checks = {
        "knowledge_index": bool(agent.store.chunks),
        "mcp_tools": await agent.ping_mcp(),
        "llm_configured": settings.llm_configured,
    }
    ok = all(checks.values())
    return JSONResponse(
        {"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503
    )
