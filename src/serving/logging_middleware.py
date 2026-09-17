"""Request logging middleware for the inference API."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("serving.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"
        return response
