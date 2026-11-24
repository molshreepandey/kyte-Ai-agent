"""HTTP hardening: request IDs, access logs, security headers, body limits, last-resort
error handling and admin authentication."""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
import uuid
from typing import Annotated

from fastapi import Header, HTTPException, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import GENERIC_500, error_body
from .request_context import request_id_var, session_id_var

log = logging.getLogger("security")
access_log = logging.getLogger("access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_DOCS_PATHS = ("/docs", "/redoc")
# High-volume, low-value paths are logged at DEBUG.
_QUIET_PATHS = ("/health/", "/static/")

# The UI is served from this origin only: no inline scripts, no third-party assets.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "media-src 'self' blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


class RequestContextMiddleware:
    """Outermost application middleware.

    * Assigns each request an ID (reusing a well-formed inbound ``X-Request-ID``) that is
      returned in the ``X-Request-ID`` header, in error bodies and in logs.
    * Writes one structured access-log record per request with its latency.
    * Adds security headers to every response.
    * Converts any exception that escaped the app into a generic 500 in the standard
      error format, after logging it with its traceback. Clients never see internals.
    """

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = inbound if _REQUEST_ID_RE.match(inbound) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        session_token = session_id_var.set("-")
        path: str = scope.get("path", "")
        started = False
        status = 500
        t0 = time.perf_counter()

        async def send_with_headers(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = list(message.get("headers", []))
                headers += self._security_headers(path)
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:  # last-resort boundary: log with traceback, return a generic 500
            if started:
                raise  # a response is already on the wire; let the server handle the failure
            log.exception("unhandled error on %s %s", scope.get("method"), path)
            await _send_json(send_with_headers, 500, error_body(500, GENERIC_500))
        finally:
            access_log.log(
                logging.DEBUG if path.startswith(_QUIET_PATHS) else logging.INFO,
                "%s %s %s",
                scope.get("method"),
                path,
                status,
                extra={
                    "method": scope.get("method"),
                    "path": path,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
            session_id_var.reset(session_token)
            request_id_var.reset(token)

    def _security_headers(self, path: str) -> list[tuple[bytes, bytes]]:
        headers = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"permissions-policy", b"microphone=(self), camera=(), geolocation=()"),
            (b"cross-origin-opener-policy", b"same-origin"),
        ]
        # Swagger/ReDoc pages load their own scripts from a CDN; they are disabled in production.
        if not path.startswith(_DOCS_PATHS):
            headers.append((b"content-security-policy", CONTENT_SECURITY_POLICY.encode()))
        if self.hsts:
            headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
        return headers


class BodySizeLimitMiddleware:
    """Enforce a per-path maximum request body size, with the standard error format.

    * A declared ``Content-Length`` above the limit is rejected before any body is read.
    * Streamed (chunked) bodies are counted as they arrive; crossing the limit raises a
      413 ``HTTPException`` from ``receive``, which FastAPI turns into the error format.
    """

    def __init__(self, app: ASGIApp, *, default_max: int, path_limits: dict[str, int]) -> None:
        self.app = app
        self.default_max = default_max
        self.path_limits = path_limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.path_limits.get(scope.get("path", ""), self.default_max)
        raw = dict(scope["headers"]).get(b"content-length")
        if raw is not None:
            try:
                declared = int(raw)
            except ValueError:
                await _send_json(send, 400, error_body(400, "Invalid Content-Length header."))
                return
            if declared > limit:
                await _send_json(send, 413, error_body(413, _TOO_LARGE))
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, _TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)


_TOO_LARGE = "The request body is too large."


async def _send_json(send: Send, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def require_admin(request: Request, x_admin_key: Annotated[str | None, Header()] = None) -> None:
    """Dependency for /admin routes.

    No key configured -> 404 (the admin surface does not exist). Wrong or missing key -> 401.
    The comparison is constant-time.
    """
    admin_api_key: str = request.app.state.settings.admin_api_key.get_secret_value()
    if not admin_api_key:
        raise HTTPException(404, "Not Found")
    if not x_admin_key or not secrets.compare_digest(x_admin_key.encode(), admin_api_key.encode()):
        raise HTTPException(401, "Invalid or missing admin key.", headers={"WWW-Authenticate": "X-Admin-Key"})
