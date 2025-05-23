"""Prometheus metrics for the API."""

import time
from collections.abc import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

# Module-level so they register once, regardless of how many apps are created.
REQUESTS = Counter("rag_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "rag_request_duration_seconds", "Request latency in seconds", ["method", "path"]
)


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record request count and latency for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
    return response


def render_metrics() -> Response:
    """Render the current metrics in Prometheus exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
