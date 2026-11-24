"""FastAPI dependencies: per-application objects live on `app.state`, created in the
lifespan, so several app instances (e.g. in tests) never share state."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from .agent import Agent
from .config import Settings
from .ratelimit import RateLimiter


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_agent(request: Request) -> Agent:
    return cast(Agent, request.app.state.agent)


def get_limiter(request: Request) -> RateLimiter:
    return cast(RateLimiter, request.app.state.limiter)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
AgentDep = Annotated[Agent, Depends(get_agent)]
LimiterDep = Annotated[RateLimiter, Depends(get_limiter)]


async def enforce_rate_limit(limiter: RateLimiter, key: str) -> None:
    decision = await limiter.hit(key)
    if not decision.allowed:
        raise HTTPException(
            429,
            "Too many requests. Please wait a moment and try again.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )


async def rate_limited(request: Request, limiter: LimiterDep) -> None:
    """Per-client-IP limit. Behind a reverse proxy, run uvicorn with --proxy-headers so
    `request.client` is the real client address."""
    client = request.client.host if request.client else "unknown"
    await enforce_rate_limit(limiter, f"ip:{client}")
