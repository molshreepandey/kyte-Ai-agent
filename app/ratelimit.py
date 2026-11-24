"""Request rate limiting (token bucket).

`InMemoryRateLimiter` is per-process: correct for a single instance and for development.
Multi-instance deployments need a shared backend (Redis), which plugs in behind the same
`RateLimiter` protocol (planned with the Redis session store).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class RateLimiter(Protocol):
    async def hit(self, key: str) -> RateLimitDecision: ...


class InMemoryRateLimiter:
    """Token bucket per key: `burst` requests at once, refilled at `per_minute` per minute.

    Memory is bounded: at most `max_keys` buckets are kept, least-recently-used evicted.
    """

    def __init__(
        self,
        per_minute: int,
        burst: int,
        *,
        max_keys: int = 50_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute <= 0 or burst <= 0:
            raise ValueError("per_minute and burst must be positive")
        self.rate = per_minute / 60.0
        self.capacity = float(burst)
        self.max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    async def hit(self, key: str) -> RateLimitDecision:
        now = self._clock()
        tokens, last = self._buckets.pop(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            decision = RateLimitDecision(True)
            tokens -= 1.0
        else:
            decision = RateLimitDecision(False, max(1, math.ceil((1.0 - tokens) / self.rate)))
        self._buckets[key] = (tokens, now)
        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)
        return decision

    def __len__(self) -> int:
        return len(self._buckets)
