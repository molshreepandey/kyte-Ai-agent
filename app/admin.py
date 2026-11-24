"""Operator-only endpoints, protected by ADMIN_API_KEY (header ``X-Admin-Key``).

Host details that used to be exposed to every customer through the LLM
(`get_system_info`) live here instead. There is deliberately no log-reading endpoint:
logs contain user content and belong in the log pipeline (stdout / container logs),
not behind an HTTP API.
"""

from __future__ import annotations

import os
import platform
import shutil
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from .config import ROOT
from .dependencies import AgentDep, SettingsDep, rate_limited
from .security import require_admin

# Rate limiting runs first, so key guessing is throttled too.
router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(rate_limited), Depends(require_admin)]
)


@router.get("/diagnostics", summary="Host and agent diagnostics (requires X-Admin-Key)")
async def diagnostics(request: Request, settings: SettingsDep, agent: AgentDep) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    return {
        "environment": settings.environment,
        "uptime_seconds": int(time.monotonic() - request.app.state.started_at),
        "host": {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "disk_free_gb": round(disk.free / 1e9, 1),
            "disk_total_gb": round(disk.total / 1e9, 1),
        },
        "agent": {
            "active_sessions": len(agent.sessions),
            "index_chunks": len(agent.store.chunks),
            "embedder": agent.store.embedder,
            "mcp_connected": await agent.ping_mcp(),
            "exposed_tools": [t["name"] for t in agent.tool_names],
            "mcp_tool_allowlist": settings.tool_allowlist,
            "llm_models": settings.llm_models,
        },
        "rate_limiter_keys": len(request.app.state.limiter),
    }
