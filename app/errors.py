"""Application exceptions and the single JSON error format returned to clients.

Every error response has the shape::

    {"error": {"code": "<machine code>", "message": "<safe text>", "request_id": "<id>"}}

Validation errors add ``"details"``: a list of ``{"field", "message"}``. Internal
exception text is logged server-side and never returned.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .request_context import request_id_var

log = logging.getLogger("errors")

GENERIC_500 = "Something went wrong on our side. Please try again."

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}


class AppError(Exception):
    """Base for errors with a client-safe status, code and message."""

    status_code = 500
    code = "internal_error"
    message = GENERIC_500

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message


class LLMUnavailableError(AppError):
    status_code = 502
    code = "llm_unavailable"
    message = "The assistant is temporarily unavailable. Please try again in a moment."


class SpeechToTextError(AppError):
    status_code = 502
    code = "stt_failed"
    message = "Speech recognition failed. Please try again or type your message."


class NotConfiguredError(AppError):
    status_code = 503
    code = "not_configured"
    message = "The assistant is not configured yet. Please contact the site operator."


def error_body(
    status: int, message: str, code: str | None = None, details: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    err: dict[str, Any] = {
        "code": code or _CODES.get(status, "error"),
        "message": message,
        "request_id": request_id_var.get(),
    }
    if details:
        err["details"] = details
    return {"error": err}


async def _app_error(_: Request, exc: AppError) -> JSONResponse:
    log.warning("%s: %s", exc.code, exc.__cause__ or exc)
    return JSONResponse(error_body(exc.status_code, exc.message, exc.code), status_code=exc.status_code)


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Messages we raise ourselves are safe; anything 5xx gets the generic text.
    message = exc.detail if isinstance(exc.detail, str) and exc.status_code < 500 else GENERIC_500
    return JSONResponse(
        error_body(exc.status_code, message), status_code=exc.status_code, headers=exc.headers
    )


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Never echo the submitted values back (pydantic includes them in `input`).
    details = [
        {"field": ".".join(str(p) for p in err.get("loc", ())), "message": str(err.get("msg", "invalid"))}
        for err in exc.errors()
    ]
    return JSONResponse(error_body(422, "The request is invalid.", details=details), status_code=422)


_Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def install_error_handlers(app: FastAPI) -> None:
    # Starlette dispatches by exception class, so each handler only ever receives its own type.
    app.add_exception_handler(AppError, cast(_Handler, _app_error))
    app.add_exception_handler(StarletteHTTPException, cast(_Handler, _http_error))
    app.add_exception_handler(RequestValidationError, cast(_Handler, _validation_error))
