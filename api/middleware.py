"""
api/middleware.py — Module 21: Production REST API
Custom ASGI middleware for security headers, rate limiting, request tracing, and sanitized logging.
"""

from __future__ import annotations

import time
import uuid
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.settings import settings

logger = logging.getLogger("MedicalAI.API.Middleware")


# ── 1. Security Headers Middleware ───────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Applies production security headers to all outgoing HTTP responses:
      - HSTS (Strict-Transport-Security)
      - X-Content-Type-Options: nosniff
      - X-Frame-Options: DENY
      - Referrer-Policy: strict-origin-when-cross-origin
      - Permissions-Policy: camera=(), microphone=(), geolocation=()
      - Content-Security-Policy
      - Custom Server header masking
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security Headers
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # CSP adjustment for Swagger UI & ReDoc documentation static assets
        path = request.url.path
        is_docs_path = (
            path in ("/docs", "/redoc", "/openapi.json")
            or path.startswith(("/docs/", "/redoc/"))
        )

        if is_docs_path:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.jsdelivr.net; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self' https://cdn.jsdelivr.net; "
                "frame-ancestors 'none';"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"

        # Mask generic server signature
        response.headers["Server"] = "MedicalAI-Engine/1.0.0"
        return response


# ── 2. Request Tracing Middleware ─────────────────────────────────────────────

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Generates and injects a unique X-Request-ID header into request state and response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_id = request.headers.get("X-Request-ID")
        req_id = incoming_id if incoming_id else f"req_{uuid.uuid4().hex[:12]}"

        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


# ── 3. In-Memory Token Bucket Rate Limiter ─────────────────────────────────────

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    In-memory rate limiting middleware per client IP address.
    Configurable limits per endpoint path prefix.
    """

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        # Store timestamp history per IP: Dict[IP, List[float]]
        self._ip_history: Dict[str, List[float]] = defaultdict(list)

    def _get_limit_for_path(self, path: str) -> int:
        if path.startswith("/predict") or path.startswith("/batch-predict"):
            return settings.RATE_LIMIT_PREDICT_PER_MIN
        if path.startswith("/history"):
            return settings.RATE_LIMIT_HISTORY_PER_MIN
        if path.startswith("/health"):
            return settings.RATE_LIMIT_HEALTH_PER_MIN
        return settings.RATE_LIMIT_DEFAULT_PER_MIN

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0  # 1-minute sliding window

        path = request.url.path
        limit = self._get_limit_for_path(path)

        # Clean timestamps older than 60s
        history = self._ip_history[client_ip]
        history = [t for t in history if t > window_start]
        self._ip_history[client_ip] = history

        if len(history) >= limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip} on path '{path}' ({len(history)}/{limit})")
            from fastapi.responses import JSONResponse
            from api.schemas import ErrorResponse
            from datetime import datetime, timezone

            req_id = getattr(request.state, "request_id", "N/A")
            payload = ErrorResponse(
                error_code="RATE_LIMIT_EXCEEDED",
                message=f"Rate limit exceeded ({limit} requests/minute). Please retry in 60 seconds.",
                request_id=req_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                path=path,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=payload.model_dump(),
                headers={"Retry-After": "60", "X-Request-ID": req_id},
            )

        # Record current request timestamp
        self._ip_history[client_ip].append(now)

        return await call_next(request)


# ── 4. Process Timing & Sanitized Logging Middleware ──────────────────────────

class ProcessTimingAndLoggingMiddleware(BaseHTTPMiddleware):
    """
    Measures total wall-clock latency, sets X-Process-Time-Ms, and emits sanitized log lines.
    Masks sensitive query parameters like api_key.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        req_id = getattr(request.state, "request_id", "N/A")

        # Mask sensitive query params in logs
        url_str = str(request.url)
        if "api_key=" in url_str:
            import re
            url_str = re.sub(r"api_key=[^&]+", "api_key=***", url_str)

        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"--> [{req_id}] {request.method} {request.url.path} (Client: {client_ip})")

        response = await call_next(request)

        process_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_ms:.2f}"

        logger.info(f"<-- [{req_id}] {request.method} {request.url.path} [{response.status_code}] ({process_ms:.2f} ms)")
        return response


# ── Setup Function ─────────────────────────────────────────────────────────────

def setup_middleware(app: FastAPI) -> None:
    """Configure all security and operational middleware on FastAPI application."""
    # CORS Middleware (Environment-based origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # Operational Middleware Stack (order matters: executed bottom to top for request)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ProcessTimingAndLoggingMiddleware)
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(RequestTracingMiddleware)
