"""Request logging middleware for the inference API."""

import logging

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("serving.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response
