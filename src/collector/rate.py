from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, Optional

from aiolimiter import AsyncLimiter


class RateController:
    """Helper that combines a global limiter with optional per-endpoint caps."""

    def __init__(
        self, global_rps: int, per_endpoint: Optional[Dict[str, int]] = None
    ) -> None:
        self.global_limiter = AsyncLimiter(max(global_rps, 1), time_period=1.0)
        self.endpoint_limiters = {
            name: AsyncLimiter(max(limit, 1), time_period=1.0)
            for name, limit in (per_endpoint or {}).items()
            if limit
        }

    @asynccontextmanager
    async def throttle(self, endpoint: Optional[str] = None):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self.global_limiter)
            if endpoint and endpoint in self.endpoint_limiters:
                await stack.enter_async_context(self.endpoint_limiters[endpoint])
            yield
