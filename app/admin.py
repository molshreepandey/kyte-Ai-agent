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

from . import config
from .security import make_admin_guard

_STARTED = time.monotonic()

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(make_admin_guard(lambda: config.ADMIN_API_KEY))],
)


@router.get("/diagnostics")
async def diagnostics(request: Request) -> dict[str, Any]:
    agent = request.app.state.agent
    disk = shutil.disk_usage(config.ROOT)
    return {
        "environment": config.ENVIRONMENT,
        "uptime_seconds": int(time.monotonic() - _STARTED),
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
            "exposed_tools": [t["name"] for t in agent.tool_names],
            "mcp_tool_allowlist": config.MCP_TOOL_ALLOWLIST,
            "llm_models": config.LLM_MODELS,
        },
        "rate_limiter_keys": len(request.app.state.limiter),
    }
